#!/usr/bin/env python3
"""Launch census per decode step by op class, outside routed MoE, on random-weight DS4.1 slices from the real config.

Two 1-layer slices (real dims: hidden 5120, q_lora 1280, head 512x64, o_groups 8, hc_mult 4, sinkhorn 20, window 128):
  full  = kv source + index source, ratio 2 (layers 2,8,14,20)
  reuse = ratio 1, no compressor/indexer (layers 21-23, 25-27, ...)
The routed SwitchGLU gather is stubbed to zeros (moefuse owns it); the shared expert stays.
usage: ds41_slice_census.py [--no-fuse] [--role full|reuse|both] [--json OUT]
"""
import collections, json, os, re, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx

VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims",
         "Split", "Slice", "AsStrided", "StopGradient", "Depends"}


def build(role: str, exl3: bool = False, dtype=None):
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    import numpy as np
    dt = dtype or mx.bfloat16
    spec = {"full": [2], "reuse": [1], "pair": [2, 1], "empty": [], "local": [0], "reindex": [2, 1]}[role]
    n = len(spec)
    kv = [0] if 2 in spec else []
    idx = [0] if 2 in spec else []
    if role == "reindex":
        idx = [0, 1]
    cfg = ModelConfig.from_dict(dict(
        vocab_size=1024, hidden_size=5120, moe_intermediate_size=512,
        num_hidden_layers=n, num_attention_heads=64, num_key_value_heads=1,
        head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
        rms_norm_eps=1e-6, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=1.5, sliding_window=128,
        compress_ratios=spec,
        kv_source_layer_ids=kv,
        index_source_layer_ids=idx,
        candidate_source_layer_id=-1, index_n_heads=32, index_head_dim=128, index_topk=512,
        candidate_topk_blocks=2048, candidate_block_size=8, hc_mult=4, hc_sinkhorn_iters=20,
        engram_layer_ids=[], engram_num_embeddings=[]))
    lm = LanguageModel(cfg)
    rng = np.random.default_rng(0)
    from mlx.utils import tree_flatten, tree_unflatten
    flat = []
    for k, v in tree_flatten(lm.parameters()):
        if v.ndim >= 2 and not any(s in k for s in ("head", "gate", "hc_")):
            w = (rng.standard_normal(v.shape) * 0.02).astype(np.float16)
            flat.append((k, mx.array(w).astype(dt)))
        elif "norm" in k:
            w = (rng.standard_normal(v.shape) * 0.02).astype(np.float16)
            flat.append((k, mx.array(w).astype(dt)))
        else:
            w = (rng.standard_normal(v.shape) * 0.02).astype(np.float32)
            flat.append((k, mx.array(w)))
    lm.update(tree_unflatten(flat))
    mx.eval(lm.parameters())
    if exl3:
        to_exl3(lm)
    return lm


def to_exl3(lm, K: int = 2, seed: int = 0):
    """Replace dense linears (kin, nout % 128) with synthetic-treillis Exl3Linears (the real pack is all-EXL3 dense).

    Skips the router gate + head (fp32), hc mixers, weights_proj (32-wide) and wo_a (stock MultiLinear: one batched matmul, like the grouped path)."""
    import numpy as np
    import mlx.nn as nn
    from kyojin_mlx.exl3_linear import Exl3Linear
    rng = np.random.default_rng(seed)
    skip = ("head", "hc_", "sink", "weights_proj", "wo_a")
    targets = []
    for layer in getattr(lm, "layers", []):
        a = layer.attn
        targets += [(a, n) for n in ("wq_a", "wq_b", "wkv", "wo_b")]
        se = getattr(layer.ffn, "shared_experts", None)
        if se is not None:
            targets += [(se, n) for n in ("gate_proj", "up_proj", "down_proj")]
        comp = getattr(a, "compressor", None)
        if comp is not None:
            targets += [(comp, n) for n in ("wkv", "wgate")]
        idx = getattr(a, "indexer", None)
        if idx is not None:
            targets += [(idx, n) for n in ("wq_b", "wk")]
    n = 0
    for owner, name in targets:
        if any(s in name for s in skip):
            continue
        mod = getattr(owner, name, None)
        if not isinstance(mod, nn.Linear):
            continue
        kin, nout = mod.weight.shape[1], mod.weight.shape[0]
        if kin % 128 or nout % 128:
            continue
        tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K)).astype(np.int16)
        su = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
        sv = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
        setattr(owner, name, Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16))
        n += 1
    mx.eval(lm.parameters())
    return n


def stub_routed(lm):
    for layer in lm.layers:
        stock = layer.ffn.switch_mlp

        def call(self, x, inds, weights, _s=stock):
            return mx.zeros((x.shape[0], x.shape[1], 1, x.shape[-1]), dtype=x.dtype)
        layer.ffn.switch_mlp.__class__ = type("StubGLU", (type(stock),), {"__call__": call})


def census(out):
    path = os.path.join(tempfile.gettempdir(), f"slice_census_{os.getpid()}.dot")
    mx.export_to_dot(path, out)
    txt = open(path).read()
    os.unlink(path)
    labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt)
    c = collections.Counter(labels)
    views = sum(v for k, v in c.items() if k in VIEWS)
    return c, len(labels) - views


def one(role):
    from kyojin_mlx import archs
    from kyojin_mlx import fuse as FU
    args = sys.argv
    lm = build(role, exl3="--exl3" in args, dtype=mx.float16 if "--exl3" in args else mx.bfloat16)
    FU.install(archs.DS41)
    stub_routed(lm)
    if "--s5" in args:
        from kyojin_mlx import fused_group as FG
        def fwd():
            cache = lm.make_cache()
            lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache)
            lm(mx.array([[9]]), cache=cache)
        info = FG.install(lm, fwd)
        print("S5:", {k: v for k, v in info.items() if k != "names"}, flush=True)
    cache = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache))
    cache2 = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache2))
    lg = lm(mx.array([[9]]), cache=cache2).logits
    outs = [lg]
    if "--keepalive" in sys.argv:
        e = cache2[0]
        for t in (e.topk_idxs, e.compress_kv, e.index_k, e.candidates):
            if t is not None:
                outs.append(t.sum() * 0)
    return census(outs)


def main():
    nofuse = "--no-fuse" in sys.argv
    if nofuse:
        os.environ["KYOJIN_FUSE"] = "0"
        os.environ["KYOJIN_FUSE_ATTN"] = "0"
        os.environ["KYOJIN_FUSE_INDEX"] = "0"
        os.environ["KYOJIN_FUSE_COMPILE"] = "0"
    role = "both"
    if "--role" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1]
    roles = ["full", "reuse", "pair", "empty"] if role == "both" else [role]
    res = {}
    for r in roles:
        c, disp = one(r)
        res[r] = dict(dispatches=disp, by_type=dict(c))
        print(f"role={r} routed-stubbed fuse={'off' if nofuse else 'on'}: {disp} dispatches")
        for k, v in c.most_common():
            print(f"  {k:34s} {v:5d}{'' if k not in VIEWS else '  (view)'}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
