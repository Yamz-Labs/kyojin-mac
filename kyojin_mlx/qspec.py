# SPDX-License-Identifier: Apache-2.0
"""Speculative decode with the Qwen3.5 MTP head on the Kyojin EXL3 model (greedy, token identical to plain decode).

    model, _ = load_exl3(pack); mtp = load_mtp(model, bf16_dir)
    sp = QSpec(model, mtp, depth=4); sp.prefill(ids); toks = sp.generate(n)

Round (state: trunk cache holds tokens < n, `t0` = token n, accepted but not fed; mtp cache holds the pairs j < n):
  1. verify: the trunk runs [t0, d1 .. dD] as ONE forward of D + 1 rows (EXL3 multi-row GEMV: a row keeps its solo bits). Row i gives
     the greedy token that follows position n + i. a = number of leading drafts that equal the trunk's own tokens; the new
     tokens are d1 .. da and the trunk's token of row a (bonus).
  2. rollback by position, no copies: full attention KV = offset trim; GDN layers = conv state is a slice of the saved conv
     input, the recurrent state is re-run for a + 1 steps from the saved pre-state (skipped when everything was accepted).
  3. draft: the MTP layer sees the pairs (token j + 1, trunk hidden j) of the accepted rows (one multi-row pass that rewrites
     the cache entries the previous draft wrote with its own hidden), then D - 1 single-row steps that feed back their output.
Drafts need not be exact; only step 1 decides the output."""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5 as Q
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from .rowinv import rowdot

# ------------------------------------------------------------------------------------------------ GDN call with capture
CHAIN_PRENORM = os.environ.get("QSPEC_CHAIN_PRENORM", "0") == "1"
CAP: list | None = None            # when a list: every GDN layer appends (module, conv_input, q, k, v, a, b, pre_state)


class FusedGdnCapture:
    """The conv_input slot of a capture taken by the fused gated-delta rows path: conv_input = concat(cs, qkv) is built only when a rollback needs it, and the
    a / b slots then hold the fused front end's g / beta (the recurrence replays on them)."""
    __slots__ = ("cs", "qkv")

    def __init__(self, cs, qkv):
        self.cs, self.qkv = cs, qkv


def rollback_gdn(mod, c, keep, rec):
    """Set the cache entry `c` of gated-delta layer `mod` to the state after the first `keep` rows of a captured pass (rec = one CAP entry)."""
    _, conv_in, q, k, v, aa, bb, pre = rec
    n = mod.conv_kernel_size - 1
    if isinstance(conv_in, FusedGdnCapture):
        from mlx_lm.models.gated_delta import gated_delta_kernel
        c[0] = mx.contiguous(mx.concatenate([conv_in.cs, conv_in.qkv], axis=1)[:, keep:keep + n, :])
        _, c[1] = gated_delta_kernel(q[:, :keep], k[:, :keep], v[:, :keep], aa[:, :keep], bb[:, :keep], pre, None)
        return
    c[0] = mx.contiguous(conv_in[:, keep:keep + n, :])
    _, c[1] = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep], aa[:, :keep], bb[:, :keep], mod.A_log, mod.dt_bias, pre, None, use_kernel=True)


def _gdn_call(self, inputs, mask=None, cache=None):
    """mlx_lm GatedDeltaNet.__call__ (same ops), plus the capture of what a rollback needs. Decode / verify rows take the fused front end (gdn_pf.forward_rows)."""
    from . import gdn_pf
    if gdn_pf.rows_ok(self, inputs, mask, cache):
        out, (cs, qkv, q, k, v, g, beta, state0) = gdn_pf.forward_rows(self, inputs, cache)
        if CAP is not None:
            CAP.append((self, FusedGdnCapture(cs, qkv), q, k, v, g, beta, state0))
        return out
    B, S, _ = inputs.shape
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)
    conv_state = cache[0] if cache is not None and cache[0] is not None else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(self.conv1d(conv_input))
    q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                                                         [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                                                         [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
    state = cache[1] if cache else None
    q, k = normalize_qk(q, k, inv_scale=self.head_k_dim ** -0.5, eps=1e-6)
    if CAP is not None:
        CAP.append((self, conv_input, q, k, v, a, b, state))
    out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask, use_kernel=not self.training)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = self.norm(out, z)
    return self.out_proj(out.reshape(B, S, -1))


_ORIG_GDN = Q.GatedDeltaNet.__call__


def install_capture() -> None:
    Q.GatedDeltaNet.__call__ = _gdn_call


def uninstall_capture() -> None:
    Q.GatedDeltaNet.__call__ = _ORIG_GDN


# ------------------------------------------------------------------------------------------------ row-invariant dense linears
class RowLinearS(nn.Module):
    """Dense linear (GDN in_proj_a / in_proj_b) whose output for a row does not depend on the number of rows: decode and verify
    activations ([1, S <= 16, K]) take `rowdot`, longer prompts the stock matmul. T = 1 uses it too, so plain decode and a verify row agree."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = lin.weight
        if "bias" in lin:
            self.bias = lin.bias

    def __call__(self, x):
        if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] <= 16 and x.dtype in (mx.float16, mx.float32) and self.weight.dtype == x.dtype:
            y = rowdot(x[0], self.weight)[None]
        else:
            y = x @ self.weight.T
        return y + self.bias if "bias" in self else y


def row_linears(model) -> int:
    from mlx.utils import tree_unflatten
    from .exl3_linear import Exl3Linear
    mods = [(n, RowLinearS(m)) for n, m in model.named_modules() if type(m) is nn.Linear and not isinstance(m, Exl3Linear)]
    model.update_modules(tree_unflatten(mods))
    return len(mods)


# ------------------------------------------------------------------------------------------------ MTP head
class Mtp(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layer = Q.DecoderLayer(args, args.full_attention_interval - 1)         # a full attention layer, dense MLP
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)


def load_draft_head(bf16_dir: str, k: int, bits: int = 4) -> nn.Module:
    """Draft-only output head: the first `k` rows (the most frequent BPE ids) of the checkpoint's lm_head / tied embedding, 4-bit.
    Drafts need not be exact, so the 248K-row EXL3 head (the largest read of a draft step) is replaced by a k-row one."""
    pick = None
    for f in sorted(glob.glob(os.path.join(bf16_dir, "*.safetensors"))):
        d = mx.load(f)
        for name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
            if name in d and (pick is None or name == "lm_head.weight"):
                pick = d[name][:k]
        if pick is not None and "lm_head.weight" in d:
            break
    lin = nn.Linear(pick.shape[1], k, bias=False)
    lin.weight = pick.astype(mx.float16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=bits)
    mx.eval(q.parameters())
    return q


def load_mtp(model, bf16_dir: str, dtype=mx.float16, quant_bits: int = 0, weights: dict | None = None) -> Mtp:
    """The MTP head from the bf16 checkpoint shards (the EXL3 packs do not carry it). Norm weights carry the +1 of the checkpoint.
    weights: the `mtp.*` tensors already in memory (a GGUF's nextn blocks, foreign_linear.take_mtp) instead of the shards; same convention."""
    lm = model.language_model
    mtp = Mtp(lm.args)
    w = {}
    if weights is not None:
        w = {k[4:]: v for k, v in weights.items() if k.startswith("mtp.")}
    for f in sorted(glob.glob(os.path.join(bf16_dir, "*.safetensors"))) if weights is None else ():
        for k, v in mx.load(f).items():
            if k.startswith("mtp."):
                w[k[4:]] = v
    out = {}
    for k, v in w.items():
        k = k.replace("layers.0.", "layer.")
        if v.ndim == 1:
            v = v.astype(mx.float32) + 1.0                       # zero-centred RMSNorm weights
        out[k] = v.astype(dtype)
    mtp.load_weights(list(out.items()), strict=True)
    mtp.eval()
    if quant_bits:
        nn.quantize(mtp, group_size=64, bits=quant_bits)
    mx.eval(mtp.parameters())
    return mtp


# ------------------------------------------------------------------------------------------------ engine
@dataclass
class Stats:
    rounds: int = 0
    emitted: int = 0
    accept_pos: list = field(default_factory=lambda: [0] * 32)      # accept_pos[i] = rounds in which draft i + 1 was accepted
    reach_pos: list = field(default_factory=lambda: [0] * 32)       # rounds in which draft i + 1 existed and all earlier were accepted
    t_verify: float = 0.0
    t_rollback_draft: float = 0.0


class QSpec:
    def __init__(self, model, mtp: Mtp, depth: int = 4, dhead=None, auto: bool = False, overlap: bool = True):
        self.model, self.mtp, self.depth, self.dhead = model, mtp, depth, dhead
        self.auto, self.overlap = auto, overlap                  # auto: depth 1..`depth` per round from measured acceptance x round time
        self.p_acc = [0.75] * 32
        self.v_ms: dict = {}
        self.lm = model.language_model
        self.gdn_index = {id(l.linear_attn): i for i, l in enumerate(self.lm.model.layers) if l.is_linear}
        self.stats = Stats()
        install_capture()

    # -- trunk
    def trunk(self, x):
        h = self.lm.model(x, self.cache)
        return self.lm.lm_head(h), h

    # -- MTP rows: tokens [R] (int), hidden [1, R, H] -> post-norm output [1, R, H]
    def mtp_rows(self, toks, hid):
        m = self.mtp
        e = m.pre_fc_norm_embedding(self.lm.model.embed_tokens(toks)[None])
        x = m.fc(mx.concatenate([e, m.pre_fc_norm_hidden(hid)], axis=-1))
        mask = create_attention_mask(x, self.mcache)
        y = m.layer(x, mask=mask, cache=self.mcache)
        self.last_pre = y
        return m.norm(y)

    def prefill(self, ids: list[int], chunk: int = 512):
        self.cache = make_prompt_cache(self.model)
        self.mcache = KVCache()
        x = mx.array(ids)[None]
        hs = []
        for i in range(0, len(ids), chunk):
            h = self.lm.model(x[:, i:i + chunk], self.cache)
            hs.append(h); mx.eval(h)
        H = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
        t0 = mx.argmax(self.lm.lm_head(H[:, -1:])[0, 0], -1)
        if len(ids) > 1:                                          # pairs (token j + 1, hidden j), j < P - 1
            self.mtp_rows(mx.array(ids[1:]), H[:, :-1])
            mx.eval(t0, self.mcache.keys, self.mcache.values)
        else:
            mx.eval(t0)
        self.n = len(ids)                                         # trunk cache length = position of t0
        self.t0 = int(t0.item())
        self.pend = ([self.t0], H[:, -1:])                        # pairs the MTP cache still lacks: tokens, trunk hiddens
        self.out = [self.t0]

    # -- depth policy: expected tokens per ms from the measured acceptance per position and the measured round pieces
    def choose_depth(self) -> int:
        if not self.auto:
            return self.depth
        p, vt = self.p_acc, self.v_ms
        best, bd = -1.0, 1
        for d in range(1, self.depth + 1):
            tok, run = 1.0, 1.0
            for i in range(d):
                run *= p[i]; tok += run
            S = d + 1
            known = [s for s in vt if s <= S]
            if S in vt:
                v = vt[S]
            elif known:                                           # unseen row count: nearest smaller one, plus a small slope
                s0 = max(known); v = vt[s0] * (1 + 0.05 * (S - s0))
            else:
                v = 60.0
            rate = tok / v
            if rate > best:
                best, bd = rate, d
        return bd

    def _observe(self, S, ms, a, D):
        for i in range(D):
            if i <= a:
                x = 1.0 if i < a else 0.0
                self.p_acc[i] += 0.1 * (x - self.p_acc[i])
        self.v_ms[S] = ms if S not in self.v_ms else self.v_ms[S] + 0.3 * (ms - self.v_ms[S])

    def _draft_lazy(self, D: int):
        """Chain of D draft tokens (lazy mx array [D]) from the pending pairs; the MTP cache is first trimmed to the pairs that stay valid."""
        toks, hid = self.pend
        self.mcache.trim(self.mcache.offset - (self.n - len(toks)))
        o = self.mtp_rows(mx.array(toks), hid)[:, -1:]
        ds = []
        for i in range(D):
            d = mx.argmax((self.dhead or self.lm.lm_head)(o)[0, 0], -1)
            ds.append(d)
            if i + 1 < D:
                o = self.mtp_rows(d[None], self.last_pre[:, -1:] if CHAIN_PRENORM else o)
        return mx.stack(ds)

    def step(self) -> list[int]:
        """One round: draft (lazy) -> verify D + 1 rows (one eval, one host sync) -> accept -> rollback (lazy). Returns the new tokens."""
        global CAP
        D = self.choose_depth()
        t0 = time.perf_counter()
        darr = self._draft_lazy(D)
        if self.overlap:
            mx.async_eval(darr)                                   # the GPU drafts while the host builds the verify graph
        CAP = []
        x = mx.concatenate([mx.array([self.t0]), darr])[None]
        lg, h = self.trunk(x)
        am = mx.argmax(lg[0], -1)
        t1 = time.perf_counter()
        mx.eval(am, darr)
        t2 = time.perf_counter()
        cap, CAP = CAP, None
        am, drafts = am.tolist(), darr.tolist()
        a = 0
        while a < D and am[a] == drafts[a]:
            a += 1
        S, keep = D + 1, a + 1
        st = self.stats
        st.rounds += 1
        for i in range(D):
            st.reach_pos[i] += 1 if i <= a else 0
            st.accept_pos[i] += 1 if i < a else 0
        if keep < S:                                              # rollback
            for l, c in zip(self.lm.model.layers, self.cache):
                if not l.is_linear:
                    c.trim(S - keep)
            for rec in cap:
                rollback_gdn(rec[0], self.cache[self.gdn_index[id(rec[0])]], keep, rec)
        new = drafts[:a] + [am[a]]
        self.pend = (new, h[:, :keep])
        self.n += keep
        self.t0 = am[a]
        t3 = time.perf_counter()
        self._observe(S, (t2 - t0) * 1e3, a, D)                   # draft + verify of this round as one number per S
        st.t_verify += t2 - t1
        st.t_rollback_draft += (t1 - t0) + (t3 - t2)
        st.emitted += keep
        return new

    def generate(self, n: int) -> list[int]:
        while len(self.out) < n:
            self.out += self.step()
        return self.out[:n]


def plain_generate(model, ids: list[int], n: int, sync: bool = False) -> list[int]:
    """Reference: plain greedy decode, T = 1, mlx-lm style async (the next graph is queued while the GPU runs the current)."""
    lm = model.language_model
    cache = make_prompt_cache(model)
    h = lm.model(mx.array(ids)[None], cache)
    t = mx.argmax(lm.lm_head(h[:, -1:])[:, 0], -1)
    mx.eval(t)
    out = [int(t.item())]
    step = lambda y: mx.argmax(lm.lm_head(lm.model(y[:, None], cache))[:, 0], -1)
    nxt = step(t)
    mx.async_eval(nxt)
    for i in range(n - 1):
        cur = nxt
        if i + 2 < n:
            nxt = step(cur)
            mx.async_eval(nxt)
        out.append(int(cur.item()))
    return out
