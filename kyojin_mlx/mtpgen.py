# SPDX-License-Identifier: Apache-2.0
"""Flag mtpgen (KYOJIN_MTPGEN): one loader for a shipped next-token draft head ("MTP" / "next-n" layer) on any trunk that mlx-lm / mlx-vlm already runs.

What it reads: the config keys (num_nextn_predict_layers | mtp_num_hidden_layers | num_mtp_layers, top level or text_config), the weight file
(model_mtp.safetensors, else the shards of the pack) and the trunk's own classes. Nothing is sized by a constant: hidden size, vocab, norm eps, layer
class and layer index come from the loaded trunk; the number of head layers from the config.

The head every layout below ships is the same computation (checked in vLLM / sglang loader code, see docs/reports/cmtpgen.md):
    x = proj(cat[norm_e(embed(token j+1)), norm_h(trunk hidden j)])      (MiMo-7B: cat[h, e])
    y = block(x)            a decoder layer of the trunk's own class (dense, MoE, MLA, full attention ... whatever its weights fit)
    logits = head(final_norm(y))
and only the tensor names differ. `split_head` maps the names; `build_block` picks the trunk layer index whose parameter names and shapes equal the head
layer's, runs the family's own `sanitize` (expert stacking, MLA splits) and loads it. A head that fits nothing is refused with the reason (HeadRefused).

Drives the Backend contract (backends.py): new_draft / draft_many / prefill_chunk(draft=) / spec_verify_bc -> pass.commit(accepted, drafts, p0s) / settle.
Verify = the trunk runs [pending, d1..dD] as ONE forward per session; commit keeps pending + accepted drafts: a cache whose layers all trim is trimmed, any
other (GDN / SSM state) is restored from a fork taken before the pass and the kept rows are run again. Drafts need not be exact: only the verify rows decide
the output. The verify-4 contract (4 rows in one forward == 4 steps, tools/dayone p_draft) is checked at load; a trunk that fails it is refused.
Not covered: see REFUSED."""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

CFG_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")
HEAD_FILE = "model_mtp.safetensors"
EPS_ATTRS = ("rms_norm_eps", "layer_norm_epsilon", "layernorm_epsilon", "norm_eps", "layer_norm_eps")
PARAMS = ("weight", "bias", "scales", "biases")


class HeadRefused(Exception):
    """The head cannot be driven; the message says why. The model is then served without speculative decode."""


# tensor-name roles. Every pattern is a full match of the module path (the key without its last component weight / bias / scales / biases).
ROLES = {
    "emb_norm": (r"enorm", r"pre_fc_norm_embedding", r"token_layernorm", r"mtp_emb_norm"),
    "hid_norm": (r"hnorm", r"pre_fc_norm_hidden", r"hidden_layernorm", r"mtp_hidden_norm"),
    "proj": (r"eh_proj", r"fc", r"input_proj", r"mtp_linear_proj"),
    "final_norm": (r"shared_head\.norm", r"final_layernorm", r"norm"),
    "head": (r"shared_head\.head",),
    "embed": (r"embed_tokens",),
}
CAT_HIDDEN_FIRST = ("token_layernorm",)          # MiMo-7B: cat[hidden, embedding]; every other layout: cat[embedding, hidden]
# (layer scope, shared scope): where layer i's tensors live. L = num_hidden_layers of the trunk.
SCOPES = (("mtp.layers.{i}.", "mtp."),                 # Qwen3-Next, Qwen3.5, EXAONE-MoE
          ("model.mtp.layers.{i}.", "model.mtp."),     # MiMo-V2-Flash, LongCat-Flash (single shared scope)
          ("model.mtp_layers.{i}.", ""),               # MiMo-7B
          ("model.layers.{L+i}.", ""))                 # DeepSeek-V3 / GLM-4.5 / Step-3.5 / Hunyuan / Bailing / ERNIE-4.5: layers appended after the trunk's
ERNIE = re.compile(r"^model\.(mtp_emb_norm|mtp_hidden_norm|mtp_linear_proj|mtp_block)\.(\d+)\.")
REFUSED = {
    "nemotron_h MTP (num_nextn_predict_layers + hybrid pattern of several sublayers per step, projections on the first sublayer only)": "multi-sublayer step",
    "Gemma-4 assistant head (pre/post projection, shares the trunk KV, q-only attention)": "shares trunk KV, no own cache",
    "EAGLE / EAGLE-3 draft models (llama_eagle, qwen*_eagle3, mistral_eagle ...: a separate checkpoint with its own config and aux-layer taps)": "separate model, not a layer of the trunk",
    "a head whose block class differs from every trunk layer (no trunk index with the same parameter names and shapes)": "no matching layer class",
    "a head with no final norm tensor, or whose own embedding differs from the trunk's": "unknown final norm / second embedding",
    "a trunk that fails the verify-4 contract (e.g. hy_v4 mlx-vlm sparse path, cdayone report)": "contract",
}


def trunk_view(pack: str):
    """Context manager: a directory that holds the pack without HEAD_FILE (the stock strict loaders glob model*.safetensors and would reject the head tensors)."""
    import contextlib
    import tempfile

    @contextlib.contextmanager
    def cm():
        if not os.path.isfile(os.path.join(pack, HEAD_FILE)):
            yield pack
            return
        with tempfile.TemporaryDirectory() as d:
            for f in os.listdir(pack):
                if f != HEAD_FILE:
                    os.symlink(os.path.abspath(os.path.join(pack, f)), os.path.join(d, f))
            yield d
    return cm()


def head_count(cfg: dict) -> tuple[str, int]:
    """(config key, number of head layers) from the config, top level or text_config; ("", 0) when the config declares none."""
    for src in (cfg, cfg.get("text_config") or {}):
        for k in CFG_KEYS:
            if int(src.get(k) or 0) > 0:
                return k, int(src[k])
    return "", 0


def read_head_tensors(pack: str) -> dict:
    """The tensors that may belong to a head: model_mtp.safetensors whole, else every shard tensor whose name says mtp / nextn or sits past the trunk
    (decided later by split_head). Lazy: mx.load maps the file, only the picked tensors are evaluated by the loader."""
    f = os.path.join(pack, HEAD_FILE)
    if os.path.isfile(f):
        return dict(mx.load(f))
    out = {}
    for f in sorted(glob.glob(os.path.join(pack, "*.safetensors"))):
        for k, v in mx.load(f).items():
            if "mtp" in k or "nextn" in k or "shared_head" in k or re.match(r"model\.layers\.\d+\.(enorm|hnorm|eh_proj)", k):
                out[k] = v
            elif k.startswith("model.layers."):
                out[k] = v                                   # candidate appended layers; split_head keeps those >= L
    return out


@dataclass
class Split:
    layout: str
    layer: dict                                  # block tensors, names relative to the decoder layer
    roles: dict = field(default_factory=dict)    # role -> {param: array}
    order: str = "eh"


def split_head(raw: dict, L: int, i: int, layout_hint: str | None = None) -> Split:
    """Head layer i: role tensors and block tensors out of the raw names. Raises HeadRefused when no layout holds layer i."""
    raw = dict(raw)
    for k in [k for k in raw if ERNIE.match(k)]:                       # ERNIE-4.5: model.mtp_X.<i>.* -> model.layers.<L+i>.mtp_X.*
        m = ERNIE.match(k)
        raw[f"model.layers.{L + int(m.group(2))}.{m.group(1)}.{k[m.end():]}"] = raw.pop(k)
    for scope, shared in SCOPES:
        lp = scope.replace("{L+i}", str(L + i)).replace("{i}", str(i))
        mine = {k[len(lp):]: v for k, v in raw.items() if k.startswith(lp)}
        if not mine:
            continue
        if shared:
            mine.update({k[len(shared):]: v for k, v in raw.items() if k.startswith(shared) and not k.startswith(shared + "layers.")})
        roles, block = {}, {}
        for k, v in mine.items():
            mod, _, p = k.rpartition(".")
            mod = re.sub(r"\.\d+$", "", mod) if re.match(r"mtp_(emb_norm|hidden_norm|linear_proj)\.\d+$", mod) else mod
            role = next((r for r, pats in ROLES.items() if p in PARAMS and any(re.fullmatch(x, mod) for x in pats)), None)
            if role is not None:
                roles.setdefault(role, {})[p] = v
            else:
                block[re.sub(r"^mtp_block(\.\d+)?\.", "", k)] = v
        order = "he" if any(re.fullmatch(x, mod) for mod in [k.rpartition(".")[0] for k in mine] for x in CAT_HIDDEN_FIRST) else "eh"
        missing = [r for r in ("emb_norm", "hid_norm", "proj", "final_norm") if r not in roles]
        if missing or not block:
            raise HeadRefused(f"head layer {i} under '{lp}': no tensor for {missing or 'the decoder block'} (names seen: {sorted(mine)[:6]})")
        return Split(lp, block, roles, order)
    raise HeadRefused(f"head layer {i}: no tensor under any known prefix {[s[0] for s in SCOPES]} (num_hidden_layers {L})")


def _trunk(model):
    from . import multiuser as MU
    lm = model.language_model if "language_model" in model else model
    return lm, MU._trunk(lm)


def _layer_prefix(model, j: int) -> str:
    for k, _ in tree_flatten(model.parameters()):
        m = re.search(rf"^(.*?layers)\.{j}\.", k)
        if m:
            return m.group(1)
    return "model.layers"


def _strip_q(keys):
    return {k for k in keys if not k.endswith((".scales", ".biases"))}


def build_block(model, block: dict, norm_shift: float | None = None):
    """(layer module, trunk index j, norm shift) for the head's decoder block: the first trunk layer j whose parameter names (and, for dense weights,
    shapes) equal the block's once the family's sanitize ran on it. The layer is built with the trunk's own class and loaded from the head tensors."""
    lm, t = _trunk(model)
    san = lm.sanitize if hasattr(lm, "sanitize") else (model.sanitize if hasattr(model, "sanitize") else (lambda w: w))
    why = []
    for j, ref in enumerate(t.layers):
        want = dict(tree_flatten(ref.parameters()))
        try:
            pre = f"model.layers.{j}."
            w = san({pre + k: v for k, v in block.items()})
            w = {k[len(pre):]: v for k, v in w.items() if k.startswith(pre)}
        except Exception as e:                                           # a sanitizer that needs the whole checkpoint
            why.append(f"layer {j}: sanitize {type(e).__name__}")
            continue
        if _strip_q(w) != _strip_q(want):
            why.append(f"layer {j}: names differ ({sorted(_strip_q(w) ^ _strip_q(want))[:3]})")
            continue
        if any(not k.endswith(("scales", "biases")) and k.replace("weight", "scales") not in want and want[k].shape != v.shape for k, v in w.items() if k in want):
            why.append(f"layer {j}: shapes differ")
            continue
        cls = type(ref)
        try:
            layer = cls(lm.args, j)
        except TypeError:
            try:
                layer = cls(lm.args)
            except Exception as e:
                raise HeadRefused(f"cannot construct {cls.__name__} for the head: {type(e).__name__}: {e}")
        shift = _norm_shift(w, want) if norm_shift is None else norm_shift
        out = []
        for k, v in w.items():
            if shift and v.ndim == 1 and k.endswith("norm.weight"):
                v = (v.astype(mx.float32) + shift).astype(v.dtype)
            out.append((k, v))
        qk = [k for k, _ in out if k.endswith(".scales")]
        if qk:
            q = getattr(lm.args, "quantization", None) or _CFG_Q.get("q") or {}
            if not q:
                raise HeadRefused("quantized head tensors (scales) but the config has no quantization section")
            have = {k[:-len(".scales")] for k in qk}
            nn.quantize(layer, group_size=q.get("group_size", 64), bits=q.get("bits", 4), mode=q.get("mode", "affine"),
                        class_predicate=lambda path, m: path in have and hasattr(m, "to_quantized"))
        layer.load_weights(out, strict=True)
        layer.eval()
        mx.eval(layer.parameters())
        return layer, j, shift
    raise HeadRefused("no trunk layer has the head block's parameters: " + "; ".join(why[:4]))


_CFG_Q: dict = {}


def _norm_shift(head_w: dict, trunk_w: dict) -> float:
    """Zero-centred norms (Qwen3.5 / Qwen3-Next checkpoints store w - 1, and the family's sanitize only shifts them when the whole checkpoint is raw, which
    a head subset is not): pick 0 or 1 by which one puts the head's 1-D norm tensors closer to the trunk's same-named (already loaded) ones.
    Drafts need not be exact, so a wrong pick costs acceptance, never output."""
    d0 = d1 = 0.0
    for k, v in head_w.items():
        if v.ndim == 1 and k.endswith("norm.weight") and k in trunk_w:
            m, ref = mx.mean(v.astype(mx.float32)), mx.mean(trunk_w[k].astype(mx.float32))
            d0 += abs(float((m - ref).item()))
            d1 += abs(float((m + 1.0 - ref).item()))
    return 1.0 if d1 < d0 else 0.0


def _linear(parts: dict, n_in: int, n_out: int):
    """A dense or quantized linear from role tensors; quantized when the tensors carry scales (the pack's quantization section gives the grid)."""
    bias = "bias" in parts
    if "scales" in parts:
        q = _CFG_Q.get("q") or {}
        if not q:
            raise HeadRefused("quantized head tensors (scales) but the config has no quantization section")
        lin = nn.QuantizedLinear(n_in, n_out, bias=bias, group_size=q.get("group_size", 64), bits=q.get("bits", 4), mode=q.get("mode", "affine"))
        lin.load_weights(list(parts.items()), strict=False)
    else:
        lin = nn.Linear(n_in, n_out, bias=bias)
        lin.weight = parts["weight"]
        if bias:
            lin.bias = parts["bias"]
    return lin


def _norm(parts: dict, H: int, eps: float, shift: float):
    n = nn.RMSNorm(H, eps=eps)
    w = parts["weight"]
    n.weight = (w.astype(mx.float32) + shift).astype(w.dtype) if shift else w
    return n


class MtpHead(nn.Module):
    """One head layer: proj(cat[norm_e(embed), norm_h(hidden)]) -> block -> final norm -> logits."""

    def __init__(self, emb_norm, hid_norm, proj, layer, final_norm, head, order):
        super().__init__()
        self.emb_norm, self.hid_norm, self.proj, self.layer, self.final_norm = emb_norm, hid_norm, proj, layer, final_norm
        self.head = head                         # own output head or None (the trunk's)
        self.order = order

    def hidden(self, embed, hid, cache):
        """embed [1, R, H] token embeddings, hid [1, R, H] trunk hiddens -> block output [1, R, H] (cache grows by R)."""
        e, h = self.emb_norm(embed), self.hid_norm(hid)
        x = self.proj(mx.concatenate([e, h] if self.order == "eh" else [h, e], axis=-1))
        return self.layer(x, create_attention_mask(x, cache), cache)


def load_head(model, pack: str, cfg: dict, layer_i: int = 0) -> tuple[MtpHead, dict]:
    """The head of `pack` for the loaded trunk `model` (layer_i of the config's head layers). Raises HeadRefused."""
    key, n = head_count(cfg)
    if not n:
        raise HeadRefused(f"config declares no draft head (none of {CFG_KEYS})")
    if layer_i >= n:
        raise HeadRefused(f"head layer {layer_i} asked, the config declares {n}")
    lm, t = _trunk(model)
    L = len(t.layers)
    _CFG_Q["q"] = cfg.get("quantization") or (cfg.get("text_config") or {}).get("quantization")
    sp = split_head(read_head_tensors(pack), L, layer_i)
    if "embed" in sp.roles:                                              # a second embedding must be the trunk's, else drafts would see other tokens
        mine, theirs = sp.roles["embed"]["weight"], getattr(t, "embed_tokens", None)
        if theirs is None or "weight" not in theirs or mine.shape != theirs.weight.shape or not bool(mx.array_equal(mine, theirs.weight).item()):
            raise HeadRefused("the head ships its own embedding that differs from the trunk's")
    H = lm.args.hidden_size
    eps = next((float(getattr(lm.args, a)) for a in EPS_ATTRS if getattr(lm.args, a, None) is not None), None)
    if eps is None:
        raise HeadRefused(f"no norm epsilon in the trunk args (looked for {EPS_ATTRS})")
    layer, j, shift = build_block(model, sp.layer)
    r = sp.roles
    head = None
    if "head" in r:
        head = _linear(r["head"], H, r["head"]["weight"].shape[0])
    mod = MtpHead(_norm(r["emb_norm"], H, eps, shift), _norm(r["hid_norm"], H, eps, shift), _linear(r["proj"], 2 * H, H), layer,
                  _norm(r["final_norm"], H, eps, shift), head, sp.order)
    mod.eval()
    mx.eval(mod.parameters())
    return mod, {"layout": sp.layout, "order": sp.order, "layers": n, "layers_used": 1, "config_key": key, "block_class": type(layer).__name__,
                 "trunk_layer": j, "norm_shift": shift, "own_head": head is not None}


# ------------------------------------------------------------------------------------------------ draft state + backend mixin
class DraftCtx:
    """Per-session draft state: the head's KV (pairs of positions < pos) and the trunk hiddens that still wait for their next token."""

    def __init__(self):
        self.cache = KVCache()
        self.hid = None                           # [1, R, H] hiddens whose pair is not in the cache yet
        self.tok: list = []                       # the tokens that follow hid[0 .. R-2]; the one after hid[R-1] arrives with the next draft call
        self.rows = []                            # (draft_arrays compat)


class MtpGenMixin:
    """Speculative decode for Q35Backend / GenericBackend (enable(): the instance's class becomes this mixin + its own)."""
    supports_spec = True
    couples = False

    def _lm(self):
        return _trunk(self.model)

    def _logits(self, h):
        from . import multiuser as MU
        lm, _ = self._lm()
        return MU.head(lm, h).astype(mx.float32)

    # -- draft state
    def new_draft(self):
        return DraftCtx()

    def draft_snapshot(self, d):
        if d is None:
            return None
        return (d.cache.keys, d.cache.values, d.cache.offset, d.hid, list(d.tok))

    def draft_restore(self, snap):
        d = DraftCtx()
        d.cache.keys, d.cache.values, d.cache.offset, d.hid, d.tok = snap[0], snap[1], snap[2], snap[3], list(snap[4])
        return d

    def draft_arrays(self, d) -> list:
        return [] if d is None else [a for a in (d.cache.keys, d.cache.values, d.hid) if a is not None]

    def draft_nbytes(self, d) -> int:
        return sum(a.nbytes for a in self.draft_arrays(d))

    def _embed(self, toks):
        _, t = self._lm()
        return t.embed_tokens(mx.array(toks, dtype=mx.int32)[None])

    def _absorb(self, d, hid, toks_after):
        """Feed the pairs (token after hidden j, hidden j) whose token is known; keep the last hidden (its token is still unknown)."""
        h = hid if d.hid is None else mx.concatenate([d.hid, hid], axis=1)
        toks = d.tok + list(toks_after)
        n = min(int(h.shape[1]), len(toks))
        if n > 0:
            self.mtp_head.hidden(self._embed(toks[:n]), h[:, :n], d.cache)
        d.hid, d.tok = h[:, n:], toks[n:]

    def prefill_chunk(self, cache, ids, draft=None):
        if draft is None:
            return super().prefill_chunk(cache, ids)
        from . import multiuser as MU
        lm, t = self._lm()
        x = mx.array(ids)[None]
        h = (t(x, cache=cache) if MU._kw_cache(type(t)) else t(x, cache))
        lg = MU.head(lm, h[:, -1:])[:, -1].astype(mx.float32)
        self._absorb(draft, h, ids if draft.hid is not None else ids[1:])
        mx.eval(lg, *self.state(cache), *self.draft_arrays(draft))
        return lg

    # -- drafts: greedy chain of the head layer
    def draft_many(self, jobs):
        out = []
        for d, pending, p0, depth, hist in jobs:
            if depth <= 0 or d is None or d.hid is None or len(d.tok) + 1 != int(d.hid.shape[1]):
                out.append([])                       # no hidden for the position before `pending` (a plain step ran without the head): no draft this round
                continue
            toks = d.tok + [int(pending)]
            o = self.mtp_head.hidden(self._embed(toks), d.hid, d.cache)
            d.hid, d.tok = None, []                  # consumed: the next verify commit hands over the hiddens of its kept rows
            base = d.cache.offset                    # the R pairs are final; the chain below is speculative
            ds = []
            for i in range(depth):
                last = o[:, -1:]
                lg, sub = self._draft_logits(self.mtp_head.final_norm(last))
                tk = sub.pick(lg[0, -1]) if sub is not None else mx.argmax(lg[0, -1], -1)
                ds.append(tk)
                if i + 1 < depth:
                    o = self.mtp_head.hidden(self._embed_arr(tk), last, d.cache)
            d.cache.trim(d.cache.offset - base)
            out.append([int(x) for x in mx.stack(ds).tolist()])
        return out

    def _draft_logits(self, x):
        """(logits, sub): with KYOJIN_DRAFT_VOCAB=K only the K most frequent rows of the head (sub.pick maps the argmax back to ids; the verify keeps the full head)."""
        from . import draftvocab as DV
        lm, _ = self._lm()
        hd = self.mtp_head.head
        sub = None
        V = (hd.out_features if hasattr(hd, "out_features") else 0) or self.model_vocab()
        K = DV.size(V)
        if K:
            sub = DV.sub_head(self, DV.head_of(lm, hd), K, V, getattr(self, "pack", None))
        if sub is not None:
            return sub(x).astype(mx.float32), sub
        return (hd(x).astype(mx.float32) if hd is not None else self._logits(x)), None

    def model_vocab(self):
        lm, t = self._lm()
        return int(t.embed_tokens.weight.shape[0] if "embed_tokens" in t else t.embeddings.weight.shape[0])

    def _embed_arr(self, tk):
        _, t = self._lm()
        return t.embed_tokens(tk.reshape(1, 1))

    # -- verify + rollback
    def spec_verify_bc(self, bc, token_lists):
        from . import multiuser as MU
        lm, t = self._lm()
        users = MU.unbatch_caches(bc, len(token_lists))
        logits, hids, pre, starts = [], [], [], []
        r = 0
        for cache, toks in zip(users, token_lists):
            starts.append(r)
            r += len(toks)
            trim = len(toks) > 1 and all(hasattr(c, "is_trimmable") and c.is_trimmable() for c in cache)
            pre.append(None if (len(toks) == 1 or trim) else MU.fork_cache(cache))
            x = mx.array([int(v) for v in toks], dtype=mx.int32)[None]
            h = (t(x, cache=cache) if MU._kw_cache(type(t)) else t(x, cache))
            logits.append(MU.head(lm, h)[0].astype(mx.float32))
            hids.append(h)
        return _Pass(self, bc, users, token_lists, pre, mx.concatenate(logits, axis=0), hids, starts)

    def settle(self, bc, drafts=()):
        mx.eval(*self.state(bc), *[a for d in drafts for a in self.draft_arrays(d)])


class _Pass:
    can_defer = False

    def __init__(self, be, bc, users, token_lists, pre, logits, hids, starts):
        self.be, self.bc, self.users, self.tokens, self.pre, self.logits, self.hids, self.starts = be, bc, users, token_lists, pre, logits, hids, starts
        self.taps = {}

    def commit(self, accepted, drafts=None, p0s=None, defer=None):
        from . import multiuser as MU
        be = self.be
        lm, t = be._lm()
        users = []
        for k, (cache, toks) in enumerate(zip(self.users, self.tokens)):
            keep, T = accepted[k] + 1, len(toks)
            if keep < T:
                if self.pre[k] is not None:                              # state that cannot trim: the kept rows run again from the fork
                    cache = self.pre[k]
                    x = mx.array([int(v) for v in toks[:keep]], dtype=mx.int32)[None]
                    t(x, cache=cache) if MU._kw_cache(type(t)) else t(x, cache)
                else:
                    for c in cache:
                        c.trim(T - keep)
            users.append(cache)
            d = drafts[k] if drafts is not None else None
            if d is not None:
                new = self.hids[k][:, :keep]
                if d.hid is None:                                        # the pending token's pair partner is already in the cache
                    d.hid, d.tok = new, [int(v) for v in toks[1:keep]]
                else:                                                    # no draft ran for this session this round: its waiting hiddens + the kept rows
                    d.hid, d.tok = mx.concatenate([d.hid, new], axis=1), d.tok + [int(v) for v in toks[:keep]]
        self.bc[:] = MU.batch_caches(users)
        mx.eval(*be.state(users))


def contract_ok(be, vocab: int | None = None, n: int = 20, k: int = 4, tol: float | None = None) -> float:
    """The verify-k contract on the loaded trunk: k rows in one forward on a forked cache == k single steps (max abs fp32 logit diff); raises HeadRefused above tol."""
    from . import conformance as C, multiuser as MU
    lm, t = _trunk(be.model)
    tol = C.TOL if tol is None else tol
    V = vocab or int(lm.args.vocab_size)
    ids, new = C.seq(n, V, seed=21), C.seq(k, V, seed=22)
    cache = be.make_cache()
    be.prefill_chunk(cache, ids)
    a, b = MU.fork_cache(cache), MU.fork_cache(cache)
    rows = []
    for tk in new:
        x = mx.array([[tk]])
        h = (t(x, cache=a) if MU._kw_cache(type(t)) else t(x, a))
        rows.append(MU.head(lm, h)[0, -1].astype(mx.float32))
    x = mx.array([new])
    h = (t(x, cache=b) if MU._kw_cache(type(t)) else t(x, b))
    out = MU.head(lm, h)[0].astype(mx.float32)
    mx.eval(out, *rows)
    d = max(float(mx.max(mx.abs(out[i] - rows[i])).item()) for i in range(k))
    if d > tol:
        raise HeadRefused(f"verify-{k} contract: {d:.2e} > {tol:.0e} (rows of one forward differ from single steps; the verify could not decide the output)")
    return d


def enable(be, pack: str, cfg: dict, log=print) -> dict:
    """Arm the loader on a Q35Backend / GenericBackend instance. Raises HeadRefused (the caller serves without a draft head)."""
    d = contract_ok(be)
    head, rep = load_head(be.model, pack, cfg)
    be.mtp_head = head
    be.__dict__.setdefault("pack", pack)          # tokenizer dir of the draft vocabulary ranking
    be.__class__ = type("MtpGen" + type(be).__name__, (MtpGenMixin, type(be)), {})
    be.max_spec_rows = 16
    rep["verify_contract_diff"] = d
    log(f"[kyojin] mtpgen: draft head {rep['layout']!r} ({rep['block_class']} like trunk layer {rep['trunk_layer']}), verify-4 diff {d:.1e}")
    return rep
