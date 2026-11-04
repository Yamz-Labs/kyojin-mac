# SPDX-License-Identifier: Apache-2.0
"""Synthetic DS4.1 model (2 layers: a local layer and a ratio-1 kv / index source) with the served module types -- EXL3 trellis
linears, Exl3MultiLinear wo_a, MoeGather routed experts -- for the dec_stage tests and the dispatch census. No pack is read;
every weight is random. `build(**over)` takes ModelConfig fields; FX.REAL is the served attention shape."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.deepseek_v41.config import ModelConfig
from mlx_vlm.models.deepseek_v41.language import LanguageModel

from kyojin_mlx import exl3_linear as EL
from kyojin_mlx import moe as M
from kyojin_mlx import patch as P
from kyojin_mlx import fused_group as FG
from kyojin_mlx import ref_numpy as R

DIM = dict(
    vocab_size=128, hidden_size=256, moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=1, head_dim=128, qk_rope_head_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=2,
    swiglu_limit=10.0, rms_norm_eps=1e-6, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    routed_scaling_factor=1.5, sliding_window=64, compress_ratios=[0, 1], kv_source_layer_ids=[1],
    index_source_layer_ids=[1], candidate_source_layer_id=1, index_n_heads=4, index_head_dim=128, index_topk=8,
    candidate_topk_blocks=8, candidate_block_size=8, hc_mult=4, hc_sinkhorn_iters=4, engram_layer_ids=[],
    engram_num_embeddings=[], max_position_embeddings=4096, rope_theta=10000.0, compress_rope_theta=160000.0)


# the served attention shape (head_dim 512, 32 index heads): exercises the kv chain, the q / inverse rope fuse and the fused index select
REAL = dict(hidden_size=512, num_attention_heads=4, head_dim=512, q_lora_rank=512, o_lora_rank=128, o_groups=4,
            moe_intermediate_size=128, index_n_heads=32, index_head_dim=128, sliding_window=64, compress_ratios=[0, 1])


def _trellis(kin, nout, seed, K=2):
    rng = np.random.default_rng(seed)
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    return tr, suh, svh


_n = [0]


def exl3(lin, out_dtype=mx.float16, K=2):
    _n[0] += 1
    tr, su, sv = _trellis(lin.weight.shape[1], lin.weight.shape[0], _n[0], K)
    return EL.Exl3Linear(tr, su, sv, bias=None, out_dtype=out_dtype)


def multilin(ml, G, kin, nout, K=2):
    tr, su, sv = [], [], []
    for g in range(G):
        _n[0] += 1
        a, b, c = _trellis(kin, nout, _n[0], K)
        tr.append(a); su.append(b); sv.append(c)
    return P.Exl3MultiLinear(mx.array(np.stack(tr)), mx.array(np.stack(su)), mx.array(np.stack(sv)), "mul1", fq_in=True)


def stack(experts, kin, nout, seed, K=2):
    ws, off, ks, suh, svh = [], [0], [], [], []
    for e in range(experts):
        tr, su, sv = _trellis(kin, nout, seed * 100 + e, K)
        w = np.ascontiguousarray(tr.astype("<i2")).view("<u4").reshape(-1)
        ws.append(mx.array(w))
        off.append(off[-1] + w.size)
        ks.append(K)
        suh.append(su); svh.append(sv)
    return M.Stack(mx.concatenate(ws), mx.array(off, dtype=mx.uint32), mx.array(ks, dtype=mx.uint8),
                   mx.array(np.stack(suh)), mx.array(np.stack(svh)), kin // 16, nout // 16, "mul1")


def exl3ify(model):
    """The served module types for one/two layers (the roles patch_model sets from a pack)."""
    cfg = model.config
    for layer in model.language_model.layers:
        a, f = layer.attn, layer.ffn
        a.wq_a, a.wkv, a.wq_b = exl3(a.wq_a), exl3(a.wkv), exl3(a.wq_b)
        a.wo_a = multilin(a.wo_a, cfg.o_groups, (a.n_heads // cfg.o_groups) * a.head_dim, cfg.o_lora_rank)
        a.wo_b = exl3(a.wo_b)
        if a.compressor is not None and a.compress_ratio > 1:
            a.compressor.wkv, a.compressor.wgate = exl3(a.compressor.wkv), exl3(a.compressor.wgate)
        elif a.compressor is not None:
            a.compressor.wkv = exl3(a.compressor.wkv)
        if a.indexer is not None:
            a.indexer.wq_b = exl3(a.indexer.wq_b)
            a.indexer.wk = exl3(a.indexer.wk)
        g = stack(cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size, 7)
        u = stack(cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size, 8)
        d = stack(cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size, 9)
        mg = M.MoeGather(g, u, d, cfg.num_experts_per_tok, limit=cfg.swiglu_limit, wpre=True, fq=True,
                         pin_rows=1, union=False, grouped=False, hotk=0)
        f.switch_mlp = P._moe_module(f.switch_mlp, mg, "slot")
        f.switch_mlp.msh = None
        f.shared_experts.gate_proj = exl3(f.shared_experts.gate_proj)
        f.shared_experts.up_proj = exl3(f.shared_experts.up_proj)
        f.shared_experts.down_proj = exl3(f.shared_experts.down_proj)
    return model


def build(seed=0, **over):
    cfg = ModelConfig.from_dict({**DIM, **over})
    mx.random.seed(seed)
    from mlx_vlm.models.deepseek_v41.deepseek_v41 import Model
    model = Model(cfg)
    lm = model.language_model
    flat = []
    from mlx.utils import tree_flatten, tree_unflatten
    for k, v in tree_flatten(lm.parameters()):
        if k.endswith("norm.weight") or k.endswith("_norm.weight"):
            w = 1.0 + 0.1 * mx.random.normal(v.shape)
        elif k.endswith("attn_sink"):
            w = 0.5 * mx.random.normal(v.shape)
        elif "hc_" in k and k.endswith("_scale"):
            w = 0.5 + 0.1 * mx.random.normal(v.shape)
        elif "hc_" in k:
            w = 0.05 * mx.random.normal(v.shape)
        elif k.endswith("gate.weight") or k.endswith("gate.bias"):
            w = 0.3 * mx.random.normal(v.shape)
        elif k.endswith("embed_tokens.weight") or k.endswith("head.weight"):
            w = 0.5 * mx.random.normal(v.shape)
        else:
            w = mx.random.normal(v.shape) * (v.shape[-1] ** -0.5)
        if v.dtype == mx.float32 and not ("hc_" in k or k.endswith("gate.bias") or k.endswith("gate.bias_vl")
                                         or k.endswith("head.weight") or "image_" in k):
            w = w.astype(mx.float16)
        flat.append((k, w.astype(v.dtype) if w.dtype == mx.float32 and v.dtype == mx.float32 else w))
    lm.update(tree_unflatten(flat))
    mx.eval(lm.parameters())
    return exl3ify(model), cfg


def toks(n: int, start: int = 0) -> list:
    """`n` token ids of the fixture's own vocabulary (vocab_size = 128, the table the random `embed_tokens` weight has).

    An id >= vocab_size is not an error in MLX: `take` reads the row anyway, i.e. whatever device memory follows the table. That memory is another
    buffer of the process (a weight, a cache layer, a kernel output), so the embedding output of those positions changes between two processes and
    every digest downstream moves with it while nothing in the engine is wrong. tests/test_dec_attn.py prefilled list(range(5, 305)) here and read
    the tail as "the tiny fixture's logits move between runs" (tests/test_nondet.py::test_forward_is_identical_in_fresh_processes)."""
    return [5 + ((start + i) % 120) for i in range(n)]


VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice",
         "AsStrided", "StopGradient", "Depends"}


def census(out, label=""):
    """Dispatches of the lazy graph ending in `out` (view-only primitives are not dispatches)."""
    import collections
    import os
    import re
    import tempfile
    path = os.path.join(tempfile.gettempdir(), f"decstage_{os.getpid()}.dot")
    mx.export_to_dot(path, out)
    txt = open(path).read()
    os.unlink(path)
    labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt)
    c = collections.Counter(labels)
    views = sum(v for k, v in c.items() if k in VIEWS)
    return dict(label=label, nodes=len(labels), views=views, dispatches=len(labels) - views, by=dict(c.most_common()))

# ----------------------------------------------------------------------------------------------------- f32 reference of the same weights
def fold_linear(lin) -> mx.array:
    """W [in, out] float32 of an Exl3Linear, decoded from its trellis (ref_numpy; no kernel, no quantisation)."""
    tr = np.array(lin._words.view(mx.int16).reshape(lin._kt, lin._nt, 16 * lin.K))
    return mx.array(R.reconstruct(tr, np.array(lin._suh), np.array(lin._svh), lin.cb).astype(np.float32))


def fold_stack(st, e: int) -> mx.array:
    """W [in, out] float32 of routed expert `e` of a Stack."""
    tr = np.array(st.expert_trellis(e))
    return mx.array(R.reconstruct(tr, np.array(st.suh[e]), np.array(st.svh[e]), st.cb).astype(np.float32))


def _mm(x, w):
    """x [..., in] @ w [in, out] in exact fp32: the stock `@` of a GPU with tensor units (M5) rounds the operands to ~10 mantissa bits for 2+ rows
    (1e-3), which would make the "float32 reference" the least accurate side of the comparison (stagek2.exact_f32_matmul_t)."""
    from kyojin_mlx import stagek2 as SK2
    lead = x.shape[:-1]
    return SK2.exact_f32_matmul_t(x.reshape(-1, x.shape[-1]), w.T).reshape(*lead, w.shape[1])


class _F32Linear(nn.Module):
    def __init__(self, w: mx.array):
        super().__init__()
        self.weight = w

    def __call__(self, x, *a, **k):
        return _mm(x.astype(mx.float32), self.weight).astype(x.dtype)


class _F32Multi(nn.Module):
    def __init__(self, ws, fq_in: bool):
        super().__init__()
        self.weights, self.fq_in = ws, fq_in

    def __call__(self, x, transpose: bool = True):
        if self.fq_in:
            from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
            x = fake_quant_fp8_ue8m0(x)
        return mx.stack([_mm(x[:, g].astype(mx.float32), self.weights[g]) for g in range(len(self.weights))], axis=1).astype(x.dtype)


class _F32Moe(nn.Module):
    """Per-expert float32 matmuls of the same trellis, the routing weights applied before the down projection (DS4.1)."""

    def __init__(self, moe: M.MoeGather):
        super().__init__()
        self.topk, self.limit, self.fq = moe.topk, moe.limit, getattr(moe, "fq", False)
        self.g = moe.g
        self.W = [[fold_stack(st, e) for e in range(st.experts)] for st in (moe.g, moe.u, moe.d)]

    def __call__(self, x, ids, wts, fq8: bool = False):
        if self.fq or fq8:
            from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0
            x = fake_quant_fp8_ue8m0(x)
        outs = []
        for r in range(x.shape[0]):
            acc = None
            for j in range(self.topk):
                e = int(ids[r, j])
                g = x[r:r + 1] @ self.W[0][e]
                u = x[r:r + 1] @ self.W[1][e]
                a = nn.silu(mx.minimum(g, self.limit)) * mx.clip(u, -self.limit, self.limit)
                y = (a * wts[r, j]) @ self.W[2][e]
                acc = y if acc is None else acc + y
            outs.append(acc)
        return mx.concatenate(outs, axis=0).astype(x.dtype)


class _F32Switch(nn.Module):
    def __init__(self, moe: M.MoeGather):
        super().__init__()
        self.moe, self.topk = _F32Moe(moe), moe.topk

    def __call__(self, x, inds, weights):
        y = self.moe(x.reshape(-1, x.shape[-1]), inds.reshape(-1, self.topk), weights.reshape(-1, self.topk))
        return y.reshape(*x.shape[:-1], 1, -1)


def use_f32_reference(model):
    """Swap every EXL3 / grouped-wo_a / routed-MoE module for a float32 matmul of the same decoded weights. Returns `restore()`.
    Used by the f32 reference test: the chain and the dec_stage path both have to sit within kernel numerics of this model."""
    todo = []
    for path, mod in model.language_model.named_modules():
        moe = getattr(mod, "moe", None)
        if isinstance(mod, (EL.Exl3Linear, P.Exl3MultiLinear, FG.FusedMember)) or isinstance(moe, M.MoeGather):
            todo.append((path, mod, moe))
    todo.sort(key=lambda kv: kv[0].count("."))
    saved, done = [], []
    for path, mod, moe in todo:
        if any(path == d or path.startswith(d + ".") for d in done):
            continue
        if isinstance(mod, FG.FusedMember):
            new = _F32Linear(fold_linear(mod._orig))
        elif isinstance(mod, EL.Exl3Linear):
            new = _F32Linear(fold_linear(mod))
        elif isinstance(mod, P.Exl3MultiLinear):
            new = _F32Multi([fold_linear(g) for g in mod.groups], getattr(mod, "fq_in", False))
        else:
            new = _F32Switch(moe)
        saved.append((path, mod))
        P._set(model.language_model, path, new)
        done.append(path)

    def restore():
        for path, mod in reversed(saved):
            P._set(model.language_model, path, mod)
    return restore
