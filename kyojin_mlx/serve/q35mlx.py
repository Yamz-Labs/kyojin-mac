# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 on a standard mlx-lm quantized pack (MLX affine 4-bit, kyojin_mlx.mlxpack) behind the serving engine, with MTP speculative decode.

The MTP head is not in the mlx-lm pack: it comes from a sidecar directory (`<pack>/mtp/*.safetensors` with the `mtp.*` tensors of the bf16
checkpoint, written by tools/mtp_sidecar.py, or `--mtp-head DIR`). With it the backend speaks the engine's speculative protocol
(backends.py module doc): per session a draft state (the MTP layer's KV and the trunk hiddens it still lacks), a verify pass of
[pending, d1 .. dD] rows through the trunk, and a rollback to the accepted rows (full attention: KV trim; GDN: conv state slice plus a
recurrent re-run for the kept rows). Drafts only propose; the verify rows decide every emitted token (greedy sessions only).

Row linears of 5..8 rows go through q4mma (kyojin_mlx.q4mma, simdgroup_matrix, flat cost 1..8 rows). KYOJIN_Q4MMA=0 turns it off."""
from __future__ import annotations

import glob
import hashlib
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import gated_delta_update

from .. import tq as TQ
from .. import tq_step as TQS
from ..kvwin import WinKVCache
from .backends import Q35Backend
from ..depchain import chain as _dep_chain      # test-only slope probe (KYOJIN_TEST_DEP_CHAIN), identity when 0

MTP_DIR_NAME = "mtp"


def find_mtp_dir(pack: str, override: str | None = None) -> str | None:
    """The directory holding the `mtp.*` tensors: --mtp-head, else <pack>/mtp (sidecar)."""
    for d in (override, os.path.join(pack, MTP_DIR_NAME)):
        if d and glob.glob(os.path.join(d, "*.safetensors")):
            return d
    top = os.path.join(pack, "model_mtp.safetensors")             # the file next to the weights (NVFP4 + draft head repos)
    if os.path.isfile(top):
        src = os.path.realpath(top)
        key = hashlib.sha1(f"{src}|{os.path.getsize(src)}".encode()).hexdigest()[:16]
        d = os.path.join(os.path.expanduser("~/.cache/kyojin_mlx/mtp"), key)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, "model_mtp.safetensors")
        if not os.path.islink(link):
            os.symlink(src, link)
        return d
    return None


DRAFT_WIN = int(os.environ.get("KYOJIN_DRAFT_WIN", "1024"))
DRAFT_SINK = int(os.environ.get("KYOJIN_DRAFT_SINK", "16"))
DRAFT_WIN_FROM = int(os.environ.get("KYOJIN_DRAFT_WIN_FROM", "8192"))


class DraftKV(KVCache):
    """The MTP layer's cache: every pair is kept (snapshots, trims), but from DRAFT_WIN_FROM pairs on the draft attends over the first DRAFT_SINK
    pairs (the attention sink) + the last DRAFT_WIN ones (a small copy). The draft row then stays flat with the context; the verify pass stays
    exact, only the proposals can change. 4B 32K (tools/flat92_dwin.py, 256 positions): agreement with the full draft 0.984 at sink 16 + 1024
    (0.91 without the sink, at any W), depth-1 acceptance 0.785 full vs 0.781. KYOJIN_DRAFT_WIN=0 = full context."""

    tail = None                     # scratch rows of a draft chain (begin_chain .. end_chain): not written into the buffer

    def windowed(self, extra: int = 1) -> bool:
        return bool(DRAFT_WIN) and self.offset + extra > max(DRAFT_WIN_FROM, DRAFT_WIN + extra + DRAFT_SINK)

    def begin_chain(self):
        """The chain steps of a draft append rows that the next draft trims anyway. Written into the buffer, each step's slice update copies the
        whole buffer (the previous step's lazy attention still holds it): 4B 32K +1.7 ms per drafted token. While windowed they go to a small
        tail instead; the attention sees the same keys in the same order, so the proposals are bit-identical."""
        if self.windowed() and self.keys is not None:
            self.tail = (self.offset, None, None)

    def end_chain(self):
        if self.tail is not None:
            self.offset, self.tail = self.tail[0], None

    def update_and_fetch(self, keys, values):
        S = DRAFT_SINK
        if self.tail is not None:
            t0, tk, tv = self.tail
            tk = keys if tk is None else mx.concatenate([tk, keys], axis=2)
            tv = values if tv is None else mx.concatenate([tv, values], axis=2)
            self.tail = (t0, tk, tv)
            self.offset += keys.shape[2]
            n, t = DRAFT_WIN + keys.shape[2], tk.shape[2]
            mk, mv = self.keys[..., :t0, :], self.values[..., :t0, :]
            k = [mk[..., :S, :]] if S else []
            v = [mv[..., :S, :]] if S else []
            if n > t:
                k.append(mk[..., t0 - (n - t):, :]); v.append(mv[..., t0 - (n - t):, :])
            k.append(tk[..., max(0, t - n):, :]); v.append(tv[..., max(0, t - n):, :])
            return mx.concatenate(k, axis=2), mx.concatenate(v, axis=2)
        k, v = super().update_and_fetch(keys, values)
        n = DRAFT_WIN + keys.shape[2]
        if DRAFT_WIN and k.shape[2] > max(DRAFT_WIN_FROM, n + S):
            k = mx.concatenate([k[..., :S, :], k[..., -n:, :]], axis=2) if S else k[..., -n:, :]
            v = mx.concatenate([v[..., :S, :], v[..., -n:, :]], axis=2) if S else v[..., -n:, :]
        return k, v


class DraftState:
    """Per session: the MTP layer's KV cache (`mv` valid pairs: pair j = token j + 1 with the trunk hidden of position j) and the trunk hiddens of
    the positions from `mv` on, which the next draft turns into pairs (pend: list of [1, k, H])."""
    __slots__ = ("mc", "mv", "pend")

    def __init__(self):
        self.mc, self.mv, self.pend = DraftKV(), 0, []


_ATT_GROUP = os.environ.get("KYOJIN_SPEC_ATT_GROUP", "1") != "0"      # A/B switch: attention ops around the cache (norms, rope, gate) once per group of sessions
_KV_MANY = os.environ.get("KYOJIN_SPEC_KV_MANY", "1") != "0"           # A/B switch: one rotation + one TQ write launch per layer for all sessions
_ATT_SHARED = os.environ.get("KYOJIN_SPEC_ATT_SHARED", "1") != "0"      # A/B switch: the shared prefix pool is read once for all sessions
_BATCH_DRAFT = os.environ.get("KYOJIN_SPEC_DRAFT_BATCH", "1") != "0"      # A/B switch: one drafter pass over the rows of all sessions (linears + head once), attention per session
_BATCH_VERIFY = os.environ.get("KYOJIN_SPEC_BATCH", "1") != "0"      # one verify pass over all sessions (A/B switch: 0 = one pass per session, solo-only spec)
_MAX_SPEC_SESSIONS = int(os.environ.get("KYOJIN_SPEC_SESSIONS", "16"))      # beyond this the plain batched step is as fast (4B, 4K prefix: N=8 spec 157 vs plain 153 tok/s)
_GDN_IN = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
_ATT_IN = ("q_proj", "k_proj", "v_proj")
_OUT = {True: "out_proj", False: "o_proj"}


class _Stub:
    """Stands in for a projection inside a stock attention / GDN module: returns the rows computed earlier for all sessions (None = identity)."""
    def __init__(self, y):
        self.y = y

    def __call__(self, x):
        return x if self.y is None else self.y


class _Bc:
    """The 'batched cache' of the spec path: the sessions' own caches (each verified on its own)."""

    def __init__(self, caches):
        self.caches = list(caches)


class Q35MlxBackend(Q35Backend):
    kind = "q35"
    row_cost = 0.08                 # depth policy numbers measured on the Air (Qwen3.5-4B mlx4 + q4mma): one more verify row / the draft pass, in solo steps
    draft_cost = 0.25
    default_mtp_depth = 4
    multi_depth = int(os.environ["KYOJIN_SPEC_MULTI_DEPTH"]) if "KYOJIN_SPEC_MULTI_DEPTH" in os.environ else None      # fixed drafts per session while several sessions share a verify pass (unset: the engine tunes it from measured rates, mu_tuner.py)
    multi_auto = multi_depth is None

    def __init__(self, model):
        super().__init__(model)
        self.lm = model.language_model
        self.mtp = self.dhead = None
        self.gdn_index = {id(l.linear_attn): i for i, l in enumerate(self.lm.model.layers) if l.is_linear}
        self._rounds = 0
        if os.environ.get("KYOJIN_SPEC_LAZY", "1") == "0":     # A/B switch: drafts evaluated before the verify graph is built
            self.draft_lazy = None

    # -- speculative decode
    def enable_mtp(self, mtp_dir: str | None, dhead_k: int = 32768, q4mma: bool | None = None, max_rows: int = int(os.environ.get("KYOJIN_SPEC_ROWS", "64")),
                   weights: dict | None = None, quant_bits: int = 4) -> dict:
        """weights / quant_bits=0: a draft head read from a GGUF (fp16 as decoded from the file's types, not re-quantised)."""
        from .. import qspec as QS
        self.QS = QS
        QS.install_capture()
        self.mtp = QS.load_mtp(self.model, mtp_dir, quant_bits=quant_bits, weights=weights)
        self.dhead = self._pack_draft_head(int(os.environ.get("KYOJIN_DHEAD_K", dhead_k)))        # draft-head rows (most frequent ids); a value >= the vocab keeps the full head
        use = os.environ.get("KYOJIN_Q4MMA", "1") != "0" if q4mma is None else q4mma
        rep = {"mtp_dir": mtp_dir or "gguf nextn block", "dhead": None if self.dhead is None else dhead_k}
        if use:
            from .. import q4mma as QM
            rep["q4mma_linears"] = QM.install(self.model)
            rep["q4mma_mtp"] = QM.install_mtp(self.mtp)
        self.max_spec_rows = max_rows
        self.supports_spec = True
        return rep

    @staticmethod
    def _pick(head, logits):
        """Draft token ids: argmax of the logits, mapped back to ids when the head is a frequency-ranked sub-head."""
        return head.pick(logits) if hasattr(head, "pick") else mx.argmax(logits, -1)

    def _pack_draft_head(self, k: int):
        """Draft-only output head: the first k rows (most frequent BPE ids) of the pack's own 4-bit tied embedding. Drafts need not be exact."""
        e = self.lm.model.embed_tokens
        from .. import draftvocab as DV
        K = DV.size(int(e.weight.shape[0]))
        if K:                                                     # flag draftvocab: any head type (foreign pack, MLX quantized / dense, tied), first K rows
            h = self.lm.__dict__.get("lm_head")
            if h is None:
                try:
                    h = self.lm.lm_head
                except AttributeError:
                    h = e
            sub = DV.sub_head(self, h, K, int(e.weight.shape[0]), getattr(self, "pack", None))
            if sub is not None:
                return sub
        from ..exl3_linear import Exl3Linear
        head = self.lm["lm_head"] if "lm_head" in self.lm else None
        if isinstance(head, Exl3Linear):                          # EXL3 pack: fold the first k output rows of the head to fp16 once, quantize them to 4 bit
            from .. import dense_pf
            k = min(k, head.out_features) // 128 * 128
            w = dense_pf.fold(head, max_rows=k)
            q = nn.QuantizedLinear(head.in_features, k, bias=False, group_size=64, bits=4)
            q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=4)
            mx.eval(q.parameters())
            return q
        if "lm_head" in self.lm:                                  # untied pack (9B): its own output matrix, same row order
            e = self.lm["lm_head"]
            if not (isinstance(e, nn.QuantizedLinear) and e.bits == 4 and e.group_size == 64) or e.weight.shape[0] <= k:
                return None
            q = nn.QuantizedLinear(e.weight.shape[1] * 8, k, bias=False, group_size=64, bits=4)
            q.weight, q.scales, q.biases = e.weight[:k], e.scales[:k], e.biases[:k]
            return q
        if not (isinstance(e, nn.QuantizedEmbedding) and e.bits == 4 and e.group_size == 64) or e.weight.shape[0] <= k:
            return None
        q = nn.QuantizedLinear(e.weight.shape[1] * 8, k, bias=False, group_size=64, bits=4)
        q.weight, q.scales, q.biases = e.weight[:k], e.scales[:k], e.biases[:k]
        return q

    def _head(self, h):
        lm = self.lm
        head = lm.__dict__.get("lm_head")
        if head is None:
            try:
                head = lm.lm_head
            except AttributeError:
                head = None
        return head(h) if head is not None else lm.model.embed_tokens.as_linear(h)

    def disk_namespace(self):
        ns = super().disk_namespace()
        if ns is not None and self.supports_spec:
            ns["mtp"] = "q35mlx"                        # the draft ring is part of the checkpoint
        return ns

    # -- draft state
    def new_draft(self):
        return DraftState() if self.supports_spec else None

    def draft_snapshot(self, d):
        if d is None or d.mc.keys is None or not d.pend:
            return None
        n = d.mc.offset
        return [d.mc.keys[..., :n, :], d.mc.values[..., :n, :], d.pend[-1]]

    def draft_restore(self, snap):
        d = DraftState()
        d.mc.keys, d.mc.values = snap[0], snap[1]
        d.mc.offset = d.mv = snap[0].shape[2]
        d.pend = [snap[2]]
        return d

    def draft_arrays(self, d) -> list:
        if d is None:
            return []
        return [a for a in (d.mc.keys, d.mc.values, *d.pend) if a is not None]

    def draft_nbytes(self, d) -> int:
        return sum(a.nbytes for a in self.draft_arrays(d))

    def _mtp_rows(self, d, toks, hid):
        """MTP layer on tokens [R] (int) and trunk hiddens [1, R, H] (hidden j goes with token j + 1): post-norm output [1, R, H]."""
        m = self.mtp
        e = m.pre_fc_norm_embedding(self.lm.model.embed_tokens(toks)[None])
        x = m.fc(mx.concatenate([e, m.pre_fc_norm_hidden(hid)], axis=-1)).astype(m.norm.weight.dtype)      # bf16 embeddings x fp16 norms promoted the whole layer (and its KV) to fp32
        y = m.layer(x, mask=create_attention_mask(x, d.mc), cache=d.mc)
        return m.norm(y)

    def _absorb(self, d, ids, h, p):
        """A trunk chunk (tokens `ids` at positions p ..., hiddens h [1, n, H]) goes into the draft state: pairs (token j, hidden j - 1)."""
        n = len(ids)
        if p == 0:
            toks, hid = ids[1:], h[:, :-1]
        else:
            toks, hid = ids, mx.concatenate([d.pend[-1], h[:, :-1]], axis=1)
        if len(toks):
            self._mtp_rows(d, mx.array(toks, dtype=mx.int32), hid)
        d.mv, d.pend = p + n - 1, [h[:, -1:]]

    @staticmethod
    def _pos(cache) -> int:
        for c in cache:
            if isinstance(c, (KVCache, WinKVCache)):                # WinKVCache: kvwin / TurboQuant pool + overlay
                return c.offset
        raise ValueError("no full attention cache")

    def prefill_chunk(self, cache, ids, draft=None):
        h = self.lm.model(mx.array(ids, dtype=mx.int32)[None], cache)
        lg = self._head(h[:, -1:])[:, 0].astype(mx.float32)
        if draft is not None:
            self._absorb(draft, ids, h, self._pos(cache) - len(ids))
        mx.eval(lg, *self.state(cache), *self.draft_arrays(draft))
        return lg

    def replay(self, cache, tokens, draft=None):
        if draft is None or not self.supports_spec:
            return super().replay(cache, tokens, draft)
        for i in range(0, len(tokens), 256):
            self.prefill_chunk(cache, [int(t) for t in tokens[i:i + 256]], draft=draft)
        return cache

    # -- batching. Speculative rounds verify each session on its own cache (`_Bc`, solo only: spec_ok); several sessions use the batched plain step
    # (MultiUser stacked caches, the weights are read once for all rows), which also feeds the draft states with the trunk hiddens.
    def spec_ok(self, n_sessions: int) -> bool:
        return self.supports_spec and (n_sessions == 1 or (_BATCH_VERIFY and n_sessions <= _MAX_SPEC_SESSIONS and n_sessions <= self.max_spec_rows))

    def batch_spec(self, caches):
        return _Bc(caches)

    def bind_drafts(self, bc, drafts):
        self._bound = list(drafts)

    def unbatch(self, bc, n):
        return bc.caches if isinstance(bc, _Bc) else super().unbatch(bc, n)

    def step(self, bc, tokens):
        if isinstance(bc, _Bc):
            outs = [self.MU.logits_last(self.model, tokens[i:i + 1, None], c).astype(mx.float32) for i, c in enumerate(bc.caches)]
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        bound = getattr(self, "_bound", None) if self.supports_spec else None
        if not bound:
            return super().step(bc, tokens)
        h = self.lm.model(tokens[:, None], bc)[:, -1:]
        for i, d in enumerate(bound):
            if d is not None:
                d.pend.append(h[i:i + 1])                          # the hidden of the position this token sits at: a later draft turns it into a pair
        return self._head(h)[:, -1].astype(mx.float32)

    def step_feed(self, bc, tokens, drafts, p0s):
        """A plain decode step of the speculative cache (no session has a draft this round): one row per session, and the trunk hidden of its
        position goes to the draft state, as the commit of a one-row verify pass would leave it."""
        outs = []
        for i, c in enumerate(bc.caches):
            h = self.lm.model(tokens[i:i + 1, None], c)[:, -1:]
            if drafts[i] is not None:
                drafts[i].pend.append(h)
            outs.append(self._head(h)[:, -1].astype(mx.float32))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    def _mtp_rows_multi(self, ds, toks, hids):
        """_mtp_rows for several sessions at once: toks[i] (int array [R_i]) and hids[i] ([1, R_i, H]) feed the MTP layer of draft state ds[i]. The
        dense linears (fc, q/k/v, o, MLP) and the norms run ONCE over the rows of all sessions; the attention core runs per session on its own
        draft cache (stubs hand it the projected rows). Returns the post-norm outputs per session [1, R_i, H]."""
        m = self.mtp
        sizes = [int(t.shape[0]) for t in toks]
        e = m.pre_fc_norm_embedding(self.lm.model.embed_tokens(mx.concatenate(toks))[None])
        x = m.fc(mx.concatenate([e, m.pre_fc_norm_hidden(mx.concatenate(hids, axis=1))], axis=-1)).astype(m.norm.weight.dtype)
        layer = m.layer
        mod = layer.self_attn
        xn = layer.input_layernorm(x)
        ys = {n: getattr(mod, n)(xn) for n in _ATT_IN}
        outs, r = [], 0
        for d, S in zip(ds, sizes):
            sl = slice(r, r + S)
            r += S
            xi = x[:, sl]
            mask = create_attention_mask(xi, d.mc)
            for n in _ATT_IN:
                mod.__dict__[n] = _Stub(ys[n][:, sl])
            mod.__dict__["o_proj"] = _Stub(None)
            try:
                outs.append(mod(xi, mask, d.mc))
            finally:
                for n in _ATT_IN + ("o_proj",):
                    mod.__dict__.pop(n, None)
        h = x + mod.o_proj(mx.concatenate(outs, axis=1))
        y = m.norm(h + layer.mlp(layer.post_attention_layernorm(h)))
        res, r = [], 0
        for S in sizes:
            res.append(y[:, r:r + S])
            r += S
        return res

    def _draft_arrays_batched(self, jobs):
        """_draft_arrays with the drafter run once over all sessions: one pass over the pairs they lack, then one step per chain position over the
        sessions still drafting (rows stacked), the draft head once per step."""
        act = []
        for i, (d, pending, p0, depth, hist) in enumerate(jobs):
            if d is None or depth <= 0 or not d.pend:
                continue
            hid = d.pend[0] if len(d.pend) == 1 else mx.concatenate(d.pend, axis=1)
            k = hid.shape[1]
            if p0 - d.mv != k:
                continue
            d.mc.trim(d.mc.offset - d.mv)
            act.append((i, d, depth, mx.array(hist[1][-k:], dtype=mx.int32), hid, p0))
        if not act:
            return {}
        os_ = self._mtp_rows_multi([a[1] for a in act], [a[3] for a in act], [a[4] for a in act])
        head = self.dhead if self.dhead is not None else self._head
        cur = [o[:, -1:] for o in os_]
        for a in act:
            a[1].mv, a[1].pend = a[5], []
            a[1].mc.begin_chain()
        ds = [[] for _ in act]
        live = list(range(len(act)))
        j = 0
        while live:
            o = cur[live[0]] if len(live) == 1 else mx.concatenate([cur[q] for q in live], axis=1)
            t = self._pick(head, head(o)[0])                           # [n live]
            for n, q in enumerate(live):
                ds[q].append(t[n])
            nxt = [(n, q) for n, q in enumerate(live) if j + 1 < act[q][2]]
            if nxt:
                outs = self._mtp_rows_multi([act[q][1] for n, q in nxt], [t[n:n + 1] for n, q in nxt], [cur[q] if len(live) == 1 else o[:, n:n + 1] for n, q in nxt])
                for (n, q), oo in zip(nxt, outs):
                    cur[q] = oo
            live = [q for n, q in nxt]
            j += 1
        lazy = {}
        for q, a in enumerate(act):
            a[1].mc.end_chain()
            lazy[a[0]] = mx.stack(ds[q])
        return lazy

    def _draft_arrays(self, jobs):
        """Lazy proposals: {job index: mx array [depth]}. Every job is a chain of single-row MTP steps after one pass over the pairs it still lacks."""
        if _BATCH_DRAFT and sum(1 for j in jobs if j[0] is not None and j[3] > 0 and j[0].pend) > 1:
            return self._draft_arrays_batched(jobs)
        lazy = {}
        for i, (d, pending, p0, depth, hist) in enumerate(jobs):
            if d is None or depth <= 0 or not d.pend:
                continue
            hid = d.pend[0] if len(d.pend) == 1 else mx.concatenate(d.pend, axis=1)
            k = hid.shape[1]
            if p0 - d.mv != k:
                continue                                              # the state does not match this position: no drafts
            d.mc.trim(d.mc.offset - d.mv)
            o = self._mtp_rows(d, mx.array(hist[1][-k:], dtype=mx.int32), hid)[:, -1:]
            d.mv, d.pend = p0, []
            head = self.dhead if self.dhead is not None else self._head
            ds = []
            d.mc.begin_chain()
            for j in range(depth):
                t = self._pick(head, head(o)[0, 0])
                ds.append(t)
                if j + 1 < depth:
                    o = self._mtp_rows(d, t[None], o)
            d.mc.end_chain()                                          # the chain rows are scratch: the next draft starts from mv
            lazy[i] = mx.stack(ds)
        return lazy

    def draft_many(self, jobs, want_logits: bool = False):
        """jobs: [(draft state, pending token, its position p0, depth, (prompt ids, emitted tokens))] -> the proposed tokens per job (evaluated).
        want_logits: this head exposes no block logits (KYOJIN_SPEC_RS only runs on a DSpark drafter), the proposals are greedy as before."""
        lazy = self._draft_arrays(jobs)
        if lazy:
            mx.eval(*lazy.values())
        return [lazy[i].tolist() if i in lazy else [] for i in range(len(jobs))]

    def draft_lazy(self, jobs, want_logits: bool = False):
        """Like draft_many, but the proposals stay on the GPU (async_eval started): spec_verify_lazy builds the verify graph from them while the
        GPU drafts, and one evaluation of the verify logits also delivers the drafts. No block logits: None when they are asked for."""
        if want_logits:
            return None
        lazy = self._draft_arrays(jobs)
        if lazy:
            mx.async_eval(*lazy.values())
        return _Lazy(lazy, len(jobs))

    def spec_verify_lazy(self, bc, lh, pendings):
        rows = {i: lh.toks[i][:d] for i, d in enumerate(lh.depths) if d > 0} if getattr(lh, "is_lookup", False) else lh.arrays
        return self._verify(bc, [mx.concatenate([mx.array([p], dtype=mx.int32), rows[i].astype(mx.int32)]) if i in rows else mx.array([p], dtype=mx.int32)
                                 for i, p in enumerate(pendings)])

    def spec_verify_bc(self, bc, token_lists):
        return self._verify(bc, [mx.array(t, dtype=mx.int32) for t in token_lists])

    def _verify(self, bc, xs):
        if _BATCH_VERIFY and len(xs) > 1:
            return self._verify_batched(bc, xs)
        QS = self.QS
        parts, starts, r = [], [], 0
        for c, x in zip(bc.caches, xs):
            QS.CAP = []
            h = self.lm.model(x[None], c)
            lg = self._head(h)[0].astype(mx.float32)
            cap, QS.CAP = QS.CAP, None
            parts.append((lg, h, cap, x.shape[0]))
            starts.append(r)
            r += x.shape[0]
        return _Pass(self, bc, parts, starts)

    def _verify_batched(self, bc, xs):
        """One verify pass over several sessions: every dense linear (qkv/o projections, GDN in/out projections, MLP, head) runs ONCE over the rows of
        all sessions, so the weights are read once. The GDN cores run once per group of sessions with the same row count (stacked states, one
        recurrence launch for the group); the attention cores run per session on its own cache (precomputed projections reach the stock module
        through stubs). Same ops and capture as `_verify`, row by row."""
        lm = self.lm.model
        order = sorted(range(len(xs)), key=lambda i: int(xs[i].shape[0]))          # sessions with the same row count sit side by side
        sizes = [int(xs[i].shape[0]) for i in order]
        starts = [sum(sizes[:k]) for k in range(len(order))]
        groups, k = [], 0
        while k < len(order):
            e = k
            while e < len(order) and sizes[e] == sizes[k]:
                e += 1
            groups.append((k, e))
            k = e
        caches = [bc.caches[i] for i in order]
        caps = [[] for _ in order]
        h = lm.embed_tokens(mx.concatenate([xs[i] for i in order])[None])
        fa = lm.fa_idx
        for li, layer in enumerate(lm.layers):
            x = layer.input_layernorm(h)
            mod = layer.linear_attn if layer.is_linear else layer.self_attn
            names = _GDN_IN if layer.is_linear else _ATT_IN
            ys = {n: getattr(mod, n)(x) for n in names}
            outm = getattr(mod, _OUT[layer.is_linear])
            if layer.is_linear:
                outs = [self._gdn_group(mod, li, ys, starts[g0], g1 - g0, sizes[g0], caches[g0:g1], caps[g0:g1]) for g0, g1 in groups]
            elif _ATT_GROUP:
                outs = [self._attn_group(mod, li, ys, starts[g0], g1 - g0, sizes[g0], x.dtype, caches[g0:g1], create_attention_mask, fa) for g0, g1 in groups]
            else:
                outs = []
                for i in range(len(order)):
                    sl = slice(starts[i], starts[i] + sizes[i])
                    xi = x[:, sl]
                    for n in names:
                        mod.__dict__[n] = _Stub(ys[n][:, sl])
                    mod.__dict__["o_proj"] = _Stub(None)
                    try:
                        outs.append(mod(xi, create_attention_mask(xi, caches[i][fa]), caches[i][li]))
                    finally:
                        for n in names + ("o_proj",):
                            mod.__dict__.pop(n, None)
            h = h + outm(mx.concatenate(outs, axis=1))
            h = _dep_chain(h + layer.mlp(layer.post_attention_layernorm(h)))
        h = lm.norm(h)
        lg_all = self._head(h)[0].astype(mx.float32)
        parts = [None] * len(order)
        for k, i in enumerate(order):
            sl = slice(starts[k], starts[k] + sizes[k])
            parts[i] = (lg_all[sl], h[:, sl], caps[k], sizes[k])
        out_starts, r = [], 0
        for part in parts:                                       # _Pass.logits concatenates the parts in session order
            out_starts.append(r)
            r += part[3]
        return _Pass(self, bc, parts, out_starts)

    def _attn_group(self, mod, li, ys, r0, g, S, dt, caches, mask_fn, fa):
        """One full-attention layer for g sessions with S rows each (rows r0 .. r0 + g S of the projections ys): the q/k norms, the rope (one offset per
        session), the gate and the q/k/v layout run once for the group; the cache write and the attention core stay per session (each its own cache).
        Same ops per row as Qwen3NextAttention.__call__. Returns the pre-o_proj rows [1, g S, Hq D]."""
        import mlx_lm.models.qwen3_next as QN
        sl = slice(r0, r0 + g * S)
        Hq, Hk = mod.num_attention_heads, mod.num_key_value_heads
        if g == 1 and S == 1:                                        # one row: norms, rope, rotation, quantize, pool + ring writes in one launch, gated merge (tq_step)
            fused = TQS.step(caches[0][li], mod, ys["q_proj"][:, sl], ys["k_proj"][:, sl], ys["v_proj"][:, sl], mod.scale)
            if fused is not None:
                return fused
        queries, gate = mx.split(ys["q_proj"][:, sl].reshape(g, S, Hq, -1), 2, axis=-1)
        gate = gate.reshape(g, S, -1)
        keys = ys["k_proj"][:, sl].reshape(g, S, Hk, -1)
        values = ys["v_proj"][:, sl].reshape(g, S, Hk, -1).transpose(0, 2, 1, 3)
        queries = mod.q_norm(queries).transpose(0, 2, 1, 3)
        keys = mod.k_norm(keys).transpose(0, 2, 1, 3)
        cs = [c[li] for c in caches]
        off = mx.array([c.offset for c in cs]) if g > 1 else cs[0].offset
        queries = mod.rope(queries, offset=off)
        keys = mod.rope(keys, offset=off)
        if _KV_MANY and g > 1 and all(type(c) is TQ.TQKVCache and c.kbits == cs[0].kbits and c.vbits == cs[0].vbits for c in cs):
            kvs = TQ.TQKVCache.update_many(cs, keys, values)         # one rotation + one quantize launch for the whole group
        else:
            kvs = [c.update_and_fetch(keys[j:j + 1], values[j:j + 1]) for j, c in enumerate(cs)]
        out = None
        if _ATT_SHARED and g > 1 and S > 1:
            out = TQ.TQKVCache.attend_shared(cs, queries, mod.scale, self.MU.shared_prefix(cs))
        if out is None:
            outs = []
            for j, c in enumerate(cs):
                xm = mx.zeros((1, S, 1), dtype=dt)
                outs.append(QN.scaled_dot_product_attention(queries[j:j + 1], kvs[j][0], kvs[j][1], cache=c, scale=mod.scale, mask=mask_fn(xm, caches[j][fa])))
            out = outs[0] if g == 1 else mx.concatenate(outs, axis=0)
        out = out.transpose(0, 2, 1, 3).reshape(g, S, -1) * mx.sigmoid(gate)
        return out.reshape(1, g * S, -1)

    def _gdn_group(self, mod, li, ys, r0, g, S, caches, caps):
        """The core of one GatedDeltaNet layer for g sessions with S rows each (rows r0 .. r0 + g S of the projections ys), batch dim g: the same ops as
        qspec._gdn_call. Returns the pre-out_proj rows [1, g S, Hv Dv]."""
        from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
        sl = slice(r0, r0 + g * S)
        qkv = ys["in_proj_qkv"][:, sl].reshape(g, S, -1)
        z = ys["in_proj_z"][:, sl].reshape(g, S, mod.num_v_heads, mod.head_v_dim)
        b = ys["in_proj_b"][:, sl].reshape(g, S, -1)
        a = ys["in_proj_a"][:, sl].reshape(g, S, -1)
        cs = [c[li] for c in caches]
        if any(c[0] is None or c[1] is None or c.lengths is not None for c in cs):
            return self._gdn_solo(mod, li, ys, r0, g, S, caches, caps)
        conv_state = cs[0][0] if g == 1 else mx.concatenate([c[0] for c in cs], axis=0)
        state0 = cs[0][1] if g == 1 else mx.concatenate([c[1] for c in cs], axis=0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        n_keep = mod.conv_kernel_size - 1
        conv_out = nn.silu(mod.conv1d(conv_input))
        q, k, v = [t.reshape(g, S, hh, d) for t, hh, d in zip(mx.split(conv_out, [mod.key_dim, 2 * mod.key_dim], -1),
                                                              [mod.num_k_heads, mod.num_k_heads, mod.num_v_heads],
                                                              [mod.head_k_dim, mod.head_k_dim, mod.head_v_dim])]
        q, k = normalize_qk(q, k, inv_scale=mod.head_k_dim ** -0.5, eps=1e-6)
        out, state = gated_delta_update(q, k, v, a, b, mod.A_log, mod.dt_bias, state0, None, use_kernel=True)
        for j, c in enumerate(cs):
            js = slice(j, j + 1)
            c[0] = mx.contiguous(conv_input[js, -n_keep:, :])
            c[1] = state[js]
            c.advance(S)
            caps[j].append((mod, conv_input[js], q[js], k[js], v[js], a[js], b[js], state0[js]))
        out = mod.norm(out, z)
        return out.reshape(1, g * S, -1)

    def _gdn_solo(self, mod, li, ys, r0, g, S, caches, caps):
        """Fallback (a cache without state yet, or with lengths): the stock captured call per session."""
        outs = []
        for j in range(g):
            sl = slice(r0 + j * S, r0 + (j + 1) * S)
            for n in _GDN_IN:
                mod.__dict__[n] = _Stub(ys[n][:, sl])
            mod.__dict__["out_proj"] = _Stub(None)
            self.QS.CAP = caps[j]
            try:
                outs.append(mod(mx.zeros((1, S, 1)), None, caches[j][li]))
            finally:
                for n in _GDN_IN + ("out_proj",):
                    mod.__dict__.pop(n, None)
                self.QS.CAP = None
        return mx.concatenate(outs, axis=1)

    def settle(self, bc, drafts=()):
        """Bound the lazy graph: the rollback and the draft state are evaluated every 16 rounds (the verify evaluation of the next round carries them)."""
        self._rounds += 1
        if self._rounds % 16 == 0:
            mx.eval(*[a for c in bc.caches for a in self.state(c)], *[a for d in drafts for a in self.draft_arrays(d)])


class _Lazy:
    """Proposals of a draft_lazy call: arrays {job: [depth]}, counts per job (0 = no drafts)."""

    def __init__(self, arrays, n):
        self.arrays = arrays
        self.counts = [arrays[i].shape[0] if i in arrays else 0 for i in range(n)]
        self.toks, self.slot, self.keep = arrays, {i: i for i in arrays}, None     # the LazyDrafts interface LookupDrafts reads for its other rows

    def tolist(self):
        return [self.arrays[i].tolist() if i in self.arrays else [] for i in range(len(self.counts))]


class _Pass:
    def __init__(self, be, bc, parts, starts):
        self.be, self.bc, self.parts, self.starts = be, bc, parts, starts
        lgs = [p[0] for p in parts]
        self.logits = lgs[0] if len(lgs) == 1 else mx.concatenate(lgs, axis=0)

    def commit(self, accepted, drafts=None, p0s=None):
        be = self.be
        for k, (c, (lg, h, cap, S)) in enumerate(zip(self.bc.caches, self.parts)):
            keep = accepted[k] + 1
            if keep < S:
                for l, cc in zip(be.lm.model.layers, c):
                    if not l.is_linear:
                        cc.trim(S - keep)
                for rec in cap:
                    be.QS.rollback_gdn(rec[0], c[be.gdn_index[id(rec[0])]], keep, rec)
            if drafts is not None and drafts[k] is not None:
                drafts[k].pend.append(h[:, :keep])
