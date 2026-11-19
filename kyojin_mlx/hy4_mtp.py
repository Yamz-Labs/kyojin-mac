# SPDX-License-Identifier: Apache-2.0
"""Flag hy4_mtp (KYOJIN_HY4_MTP, default off): MTP speculative decode for the skeleton Model of a GGUF that ships a next-n (MTP) block (Hy4-flash).

The head is read by role from the file (no tensor of it is named here beyond the `nextn.*` roles): the block after the trunk (`nextn_predict_layers` > 0)
is one ordinary decoder layer of the trunk's own class without hyper-connections, plus four tensors:
    x = eh_proj(cat[enorm(embed(token j+1)), hnorm(h_j)])     h_j = the trunk state the output head reads (after the hc head and output_norm)
    y = layer(x)            attention (MLA + kpool selection, its own KV cache) and the MoE of the trunk's class, sandwich norms, plain residuals
    y = shared_head_norm(y);  logits = trunk output head(y);  the next chain step feeds y as `h`
Ported from llama.cpp (MIT): `llama_model_hy_v4_flash::graph_mtp` (src/models/hy-v4-flash.cpp) for the head graph, and
`common_speculative_impl_draft_mtp::process / draft` (common/speculative.cpp) for the pairing (token j with the trunk state of j-1, pending state of
the previous round, greedy chain at positions pos0 + i + 1).

Verify = the trunk runs [pending, d1..dk] as ONE forward (the prefill path, k + 1 rows); acceptance = longest prefix whose greedy token equals the draft, so
the output is the plain greedy output whatever the head says. A partly accepted pass rolls back every per-position state: the plain KV caches (the kpool rows
ride in them) are trimmed, the Engram n-gram history is cut to the kept positions (foreign_skeleton.engram_source_of(...).truncate); the hyper-connection
streams live inside one forward and carry nothing across positions. The head's own KV keeps only pairs of committed tokens (the chain rows are scratch)."""
from __future__ import annotations

import dataclasses
import os
import re
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from . import foreign_gguf as GF
from . import foreign_moe_generic as G
from . import foreign_skeleton as SK
from . import foreign_skeleton_moe as SM
from .foreign_skeleton_kpool import fires as kp_fires
from .foreign_skeleton_rows import ROWS

ENV, DEPTH_ENV = "KYOJIN_HY4_MTP", "KYOJIN_HY4_MTP_DEPTH"
STATS = dict(rounds=0, drafted=0, accepted=0)
PROF = {"on": False, "n": 0, "build": 0.0, "eval": 0.0}
NEXTN_ROLES = ("nextn.eh_proj", "nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm")


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("0", "")


def default_depth() -> int:
    """Most drafts per round. 5 since 19/11 (c124 depth4: 4 depths x 5 prompts x 3 reps, warm-up at every depth). On the prompts whose text is equal at every depth
    (p1 / p2 / p3, same md5) D5 wins: 67.8 / 58.2 / 54.9 tok/s vs D4 60.1 / 57.9 / 54.0, D6 65.5 / 57.4 / 52.5, D7 63.7 / 57.2 / 49.8. On p4 / p5 the text differs per depth
    (the verify is not row-invariant), so those rows measure text luck (p4: D4 75.3, D5 56.9, D6 65.8, D7 77.3). llama.cpp n_max = 3 is the waypoint, not the optimum here."""
    return int(os.environ.get(DEPTH_ENV, "5"))


def fixed_depth() -> bool:
    """KYOJIN_HY4_MTP_FIXED (default on, 0 = the engine's adaptive depth policy): every round drafts exactly the default depth. The adaptive policy under-used
    depth on this engine (c117 default policy: p4 56-66 / p1 47-52 tok/s against 67-77 / 60 pinned)."""
    return os.environ.get("KYOJIN_HY4_MTP_FIXED", "1") not in ("0", "")


class HeadRefused(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------- the head
class DenseSwitch:
    """Dense routed-expert stack kept in its stored dtype (bf16 on the device: no per-call cast of 3 x 2.4 GiB), __call__(x [.., H], inds [.., K]) -> [.., K, H]
    un-weighted, like foreign_moe_generic._Stack (which casts the stacks to f32 at every call: right for the CPU tests, not for 7 GiB of bf16)."""

    def __init__(self, gate_w, up_w, down_w, spec):
        self.g, self.u, self.d, self.spec = gate_w, up_w, down_w, spec

    def __call__(self, x, inds):
        s = self.spec
        f = mx.float32 if mx.default_device() == mx.cpu else self.u.dtype            # gather_mm on the CPU device only takes f32
        xe = mx.expand_dims(x.astype(f), (-2, -3))
        u = mx.gather_mm(xe, self.u.astype(f).swapaxes(-1, -2), rhs_indices=inds)
        g = mx.gather_mm(xe, self.g.astype(f).swapaxes(-1, -2), rhs_indices=inds) if self.g is not None else None
        h = G.apply_act(s.act, g, u, s, s.clamp, s.clamp_mode)
        return mx.gather_mm(h, self.d.astype(f).swapaxes(-1, -2), rhs_indices=inds).squeeze(-2).astype(x.dtype)


class MtpHead(nn.Module):
    def __init__(self, sp, il, ffn_factory):
        super().__init__()
        d = sp.hidden_size
        self.enorm, self.hnorm = SK.Norm(d, sp.norm_kind, sp.norm_eps), SK.Norm(d, sp.norm_kind, sp.norm_eps)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.block = SK.Block(dataclasses.replace(sp, hc_count=0), il, ffn_factory)       # the trunk's layer class, plain residual (no hc streams)
        self.shared_head_norm = SK.Norm(d, sp.norm_kind, sp.norm_eps)
        self.il = il

    def __call__(self, model, toks, h, cache):
        """toks int32 [1, n], h [1, n, D] (trunk state of the previous position of each token), cache: the head's KVCache -> y [1, n, D] (after shared_head_norm)."""
        from mlx_lm.models.base import create_attention_mask
        m = model.model
        x = self.eh_proj(mx.concatenate([self.enorm(m.embed_tokens(toks)), self.hnorm(h)], axis=-1))
        sw = self.block.self_attn.swa
        mask = model._window_mask(x.shape[1], cache) if sw else create_attention_mask(x, cache)
        return self.shared_head_norm(self.block(x, mask, cache, model._freqs(sw)))


def head_logits(model, y):
    """The trunk's own output head on the head's state (Model.__call__ tail)."""
    sp = model.args
    out = model.model.embed_tokens.as_linear(y) if sp.tie_word_embeddings else model.lm_head(y)
    out = out * sp.logit_scale if sp.logit_scale != 1.0 else out
    if sp.logit_softcap:
        out = mx.tanh(out / sp.logit_softcap) * sp.logit_softcap
    return out


def _head_view(g, row):
    """GGUF -> (kv, tensor shapes, nextn count) of the file seen as `trunk + head` regular layers: the MTP block becomes layer n of an ordinary spec
    (build_spec / SM.detect read it with the trunk's own rules). Per-layer metadata lists must carry an entry for it (the indexer `types` list may stop at
    the trunk: the draft block always owns an indexer)."""
    arch = g.kv["general.architecture"]
    p = arch + "."
    nextn = int(g.kv.get(p + "nextn_predict_layers", 0) or 0)
    if nextn < 1:
        raise HeadRefused(f"{arch}: no nextn block (nextn_predict_layers 0)")
    if nextn > 1:
        raise HeadRefused(f"{arch}: {nextn} chained next-n blocks (one head step per block is not wired; only 1)")
    n = int(g.kv[p + "block_count"]) - nextn
    for r in NEXTN_ROLES:
        if f"blk.{n}.{r}.weight" not in g.tensors:
            raise HeadRefused(f"{arch}: blk.{n}.{r}.weight missing")
    kv = dict(g.kv)
    kv[p + "block_count"], kv[p + "nextn_predict_layers"] = n + 1, 0
    for k in ("attention.indexer.types",):
        v = kv.get(p + k)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
            kv[p + k] = [int(x) for x in v] + [1]
    ne = {k: ti.ne for k, ti in g.tensors.items() if not re.match(r"blk\.\d+\.nextn\.", k)}
    for k in [k for k in ne if k.startswith("blk.0.hc_")]:          # the block has no hyper-connection streams: the spec only needs the trunk's hc shapes to stay uniform
        ne.setdefault(f"blk.{n}." + k[len("blk.0."):], ne[k])
    return arch, kv, ne, n


def build_head(g, model, row=None, dtype=mx.float32, expert_dtype=None, log=print) -> MtpHead:
    """The MTP head of GGUF `g` (a GgufFile) next to the trunk `model` (foreign_skeleton.Model): dense decode of the blk.<n> tensors (`dtype`, routed experts
    in `expert_dtype`), strict coverage like load_cpu. Raises HeadRefused (with the reason) when the file has no usable head."""
    from mlx.utils import tree_flatten
    arch, kv, ne, n = _head_view(g, row)
    r = ROWS.get(arch) if row is None else row
    moe = SM.detect(arch, kv, ne, r) if r is not None else None
    try:
        sp = SK.build_spec(arch, kv, ne, row=r, allow_moe=moe is not None)
    except SK.SkeletonError as e:
        raise HeadRefused(f"{arch}: head spec refused: {e}") from e
    if sp.hidden_size != model.args.hidden_size or sp.vocab_size != model.args.vocab_size:
        raise HeadRefused("head and trunk disagree on hidden size / vocab")
    head = MtpHead(sp, n, moe.factory if moe is not None else None)
    expert_dtype = dtype if expert_dtype is None else expert_dtype
    kpl = {il for il, t in enumerate(sp.layer_tensors) if kp_fires(t)}
    pre, w, experts = f"blk.{n}.", [], {}
    for name in g.tensors:
        if not name.startswith(pre):
            continue
        kind = name[len(pre):].rsplit(".", 1)[0]
        if kind in NEXTN_ROLES:
            tgt = kind.split(".", 1)[1]
            w.append((tgt + "." + name.rsplit(".", 1)[1], mx.array(g.tensor_f32(name).astype(np.float32)).astype(dtype)))
            continue
        if kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            experts[kind[4]] = mx.array(g.tensor_f32(name).astype(np.float32)).astype(expert_dtype)      # g / u / d
            continue
        pn = SK.param_name(name, moe.names if moe is not None else None, kpl)
        if pn is None:
            raise HeadRefused(f"{name}: no skeleton parameter")
        w.append(("block." + pn.split(f"model.layers.{n}.", 1)[1], mx.array(g.tensor_f32(name).astype(np.float32)).astype(dtype)))
    ffn = head.block.mlp
    if experts:
        if set(experts) != {"g", "u", "d"}:
            raise HeadRefused("head experts: gate / up / down stacks not all present")
        ffn.set_stored("switch_mlp", DenseSwitch(experts["g"], experts["u"], experts["d"], ffn.moe_spec))
    have = {k for k, _ in w}
    missing = [k for k, _ in tree_flatten(head.parameters()) if k not in have]
    if missing:
        raise HeadRefused(f"{len(missing)} head parameters without a tensor, e.g. {missing[:4]}")
    head.load_weights(w, strict=True)
    mx.eval(head.parameters(), *[a for a in experts.values()])
    log(f"[hy4-mtp] head: layer {n}, {len(w)} tensors" + (f" + 3 expert stacks ({expert_dtype})" if experts else ""))
    return head


# ---------------------------------------------------------------------------------------------------------------- rollback of the trunk state
def rollback(model, cache, n: int) -> None:
    """Drop the last `n` positions of every per-position state of a trunk session: the KV caches (plain mlx-lm caches: `trim`; the kpool rows live in them) and
    the Engram n-gram history. The hyper-connection streams are per forward and hold nothing."""
    if n <= 0:
        return
    for c in cache:
        c.trim(n)
    src = getattr(model, "engram_source", None)
    if src is not None:
        src.truncate(int(cache[0].offset))


def kept_state(cache) -> int:
    return int(cache[0].offset)


# ---------------------------------------------------------------------------------------------------------------- draft state and rounds
class DraftState:
    """Per session. `toks`: the tokens the trunk consumed (position = index). `cache`: the head's KV (offset mv = positions with a pair). `pend`: trunk states
    h_{mv-1} .. h_{P-1} (P = len(toks)) not yet paired; h_{-1} is zeros (llama.cpp pending_h starts empty)."""
    __slots__ = ("cache", "toks", "pend", "dim")

    def __init__(self, dim: int):
        from mlx_lm.models.cache import KVCache
        self.cache, self.toks, self.dim = KVCache(), [], dim
        self.pend = [mx.zeros((1, 1, dim), dtype=mx.float32)]

    def rows(self):
        return self.pend[0] if len(self.pend) == 1 else mx.concatenate(self.pend, axis=1)


class Mtp:
    """The head with its trunk: feed / draft / verify / commit on one session's caches. Backend independent (the serve backend and the tests drive it)."""

    def __init__(self, model, head: MtpHead):
        self.model, self.head = model, head
        self.dim = model.args.hidden_size
        self._dq = None
        bits = int(os.environ.get("KYOJIN_HY4_HEAD_Q", "8") or 0)   # default 8: draft-only 8-bit head (c119: ids equal, draft eval 9.6 -> 4.7 ms); 0 keeps the dense f32 head
        if bits:
            self._quantize_head(bits)
        if os.environ.get("KYOJIN_HY4_DRAFT_PROF", "0") not in ("0", ""):
            self._profile_head()

    def _profile_head(self):
        """KYOJIN_HY4_DRAFT_PROF=1: bytes of the head by dtype and the output head's type, printed once; draft build / eval ms printed at exit."""
        import atexit
        from mlx.utils import tree_flatten
        by = {}
        for k, a in tree_flatten(self.head.parameters()):
            by[str(a.dtype)] = by.get(str(a.dtype), 0) + a.nbytes
        lm = getattr(self.model, "lm_head", None)
        print(f"[hy4-mtp] draft prof: head dense bytes by dtype {by}; lm_head {type(lm).__name__} {[ (k, str(a.dtype), a.shape) for k, a in tree_flatten(lm.parameters())][:4] if lm is not None else None}; embed {type(self.model.model.embed_tokens).__name__}", flush=True)
        PROF["on"] = True
        atexit.register(lambda: print(f"[hy4-mtp] draft prof: n {PROF['n']} build {1e3 * PROF['build'] / max(1, PROF['n']):.2f} ms eval {1e3 * PROF['eval'] / max(1, PROF['n']):.2f} ms per draft call", flush=True))

    def _quantize_head(self, bits: int):
        self.set_head_q(bits)

    def set_head_q(self, bits: int):
        """Switch the head to `bits` (0 = the dense head, 4 | 8 = draft-only low-bit copy of the head's Linears and of the output projection; the trunk keeps its own).
        Reversible: the dense Linears stay in `_lin[0]`, the quantized ones are built once per width. The drafts may change (acceptance), the verified output cannot."""
        from mlx.utils import tree_unflatten
        bits = int(bits)
        if not hasattr(self, "_lin"):
            self._lin = {0: {k: m for k, m in self.head.named_modules() if isinstance(m, nn.Linear)}}
        if bits not in self._lin:
            self._lin[bits] = {k: nn.QuantizedLinear.from_linear(m, group_size=64, bits=bits) for k, m in self._lin[0].items() if m.weight.shape[-1] % 64 == 0}
            mx.eval([q.parameters() for q in self._lin[bits].values()])
        self.head.update_modules(tree_unflatten([(k, (self._lin[bits].get(k) or m)) for k, m in self._lin[0].items()]))
        self._dq = None
        lm = getattr(self.model, "lm_head", None)
        w = getattr(lm, "weight", None)
        if bits and w is not None and isinstance(w, mx.array) and w.ndim == 2 and w.dtype in (mx.float32, mx.bfloat16, mx.float16) and w.shape[-1] % 64 == 0:
            ck = ("lm", bits)
            if ck not in self._lin:
                wq, sc, bi = mx.quantize(w, group_size=64, bits=bits); mx.eval(wq, sc, bi); self._lin[ck] = (wq, sc, bi, bits)
            self._dq = self._lin[ck]
        print(f"[hy4-mtp] head bits {bits}; draft-only output projection: {'yes' if self._dq else 'no'}", flush=True)

    def head_logits(self, y):
        if self._dq is None:
            return head_logits(self.model, y)
        wq, sc, bi, bits = self._dq
        out = mx.quantized_matmul(y, wq, sc, bi, transpose=True, group_size=64, bits=bits)
        sp = self.model.args
        out = out * sp.logit_scale if sp.logit_scale != 1.0 else out
        if sp.logit_softcap:
            out = mx.tanh(out / sp.logit_softcap) * sp.logit_softcap
        return out

    def new_state(self) -> DraftState:
        return DraftState(self.dim)

    # -- trunk forward with the hidden rows
    def forward(self, cache, ids):
        """ids: list / int32 array of n tokens -> (logits [n, V] f32, h [1, n, D]) ; the trunk's cache advances by n."""
        x = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
        sink = []
        lg = self.model(x[None], cache=cache, hidden_sink=sink)
        return lg[0].astype(mx.float32), sink[0]

    def feed(self, st: DraftState, ids, h) -> None:
        """The trunk consumed `ids` (positions P..) and produced `h` [1, n, D]: remember them for the next pairing."""
        st.toks.extend(int(t) for t in ids)
        st.pend.append(h)

    def absorb(self, st: DraftState) -> None:
        """Pair every consumed token that has its previous state (prefill chunks): head forward on those rows, output discarded; the last state stays pending."""
        rows = st.rows()
        n = rows.shape[1] - 1
        if n <= 0:
            return
        mv = st.cache.offset
        self.head(self.model, mx.array(st.toks[mv:mv + n], dtype=mx.int32)[None], rows[:, :n], st.cache)
        st.pend = [rows[:, n:]]

    def draft(self, st: DraftState, pending: int, depth: int):
        """Greedy chain after `pending` (the token at position P, not yet in the trunk). Returns up to `depth` int tokens, or None when the head state does not line
        up with the trunk (nothing consumed / mismatch). Keeps the pair of `pending` (true inputs); the chain rows are scratch."""
        rows = st.rows()
        mv = st.cache.offset
        tk = st.toks[mv:] + [int(pending)]
        if depth <= 0 or rows.shape[1] != len(tk):
            return None
        tb0 = time.perf_counter()
        y = self.head(self.model, mx.array(tk, dtype=mx.int32)[None], rows, st.cache)[:, -1:]
        keep = st.cache.offset
        st.pend = []
        ds = []
        for j in range(depth):
            t = mx.argmax(self.head_logits(y)[0, 0])
            ds.append(t)
            if j + 1 < depth:
                y = self.head(self.model, t.astype(mx.int32).reshape(1, 1), y, st.cache)
        tb1 = time.perf_counter()
        mx.eval(mx.stack(ds))
        if PROF["on"]:
            PROF["n"] += 1; PROF["build"] += tb1 - tb0; PROF["eval"] += time.perf_counter() - tb1
        if st.cache.offset > keep:
            st.cache.trim(st.cache.offset - keep)
        out = [int(t.item()) for t in ds]
        STATS["drafted"] += len(out)
        return out

    # -- verify + commit
    def verify(self, cache, pending: int, drafts):
        """One trunk forward over [pending, *drafts]: (logits [k+1, V], h [1, k+1, D])."""
        return self.forward(cache, [int(pending)] + [int(t) for t in drafts])

    @staticmethod
    def accept(logits, drafts):
        """Greedy acceptance: (m accepted drafts, next token). Row j of `logits` predicts the token after input j."""
        g = np.array(mx.argmax(logits, axis=-1))
        m = 0
        while m < len(drafts) and int(g[m]) == int(drafts[m]):
            m += 1
        return m, int(g[m])

    def commit(self, cache, st, ids, h, m: int, exact: bool = False) -> None:
        """After a verify pass over `ids` (= [pending, *drafts]): keep pending + m drafts, roll the rest back, hand the kept rows to the head state.
        exact (KYOJIN_HY4_MTP_EXACT): after a rejection the kept rows are not trusted from the k-row pass (a matmul over k rows can differ from the one-row
        step in the last bits): the cache goes back to the pre-pass state and the kept tokens are replayed one row at a time, so the state is bit-equal to the
        plain decode's. Costs `keep` extra one-row steps per rejecting round."""
        keep = m + 1
        if exact and keep < len(ids):
            rollback(self.model, cache, len(ids))
            hs = []
            for t in ids[:keep]:
                _, hr = self.forward(cache, [t])
                hs.append(hr)
            h = hs[0] if keep == 1 else mx.concatenate(hs, axis=1)
        else:
            rollback(self.model, cache, len(ids) - keep)
        if st is not None:
            self.feed(st, ids[:keep], h[:, :keep])
        STATS["rounds"] += 1
        STATS["accepted"] += m


def exact_commit() -> bool:
    return os.environ.get(ENV + "_EXACT", "0") not in ("0", "")


def generate(mtp: Mtp, cache, st, ids, n_new: int, depth: int, chunk: int = 4096, spec: bool = True, stats=None, exact=None):
    """Reference greedy decode driver (the engine's rounds, one session): prefill `ids`, then `n_new` tokens; spec=False is the plain one-row decode.
    Returns the list of generated ids (exactly n_new)."""
    ids = [int(t) for t in ids]
    lg = None
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        lg_all, h = mtp.forward(cache, part)
        lg = lg_all[-1]
        if st is not None:
            mtp.feed(st, part, h)
            mtp.absorb(st)
    out = [int(mx.argmax(lg).item())]
    while len(out) < n_new:
        pending = out[-1]
        ds = mtp.draft(st, pending, depth) if (spec and st is not None) else None
        if not ds:
            lg_all, h = mtp.forward(cache, [pending])
            if st is not None:
                mtp.feed(st, [pending], h)
            out.append(int(mx.argmax(lg_all[0]).item()))
            continue
        lg_all, h = mtp.verify(cache, pending, ds)
        m, nxt = mtp.accept(lg_all, ds)
        mtp.commit(cache, st, [pending] + ds, h, m, exact=exact_commit() if exact is None else exact)
        out.extend(ds[:m] + [nxt])
        if stats is not None:
            stats.append((len(ds), m))
    return out[:n_new]
