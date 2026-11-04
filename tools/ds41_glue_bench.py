#!/usr/bin/env python3
"""Interleaved A/B of the decode-step glue fuse (KYOJIN_GLUE_FUSE) on a synthetic DS4.1 stack at the real dims and layer roles.

Stack: the real text config (hidden 5120, 64 heads x 512, q_lora 1280, o_lora 1024 x 8 groups, shared expert 2304, 384 experts / top 6,
window 128, hc_mult 4 / sinkhorn 20, compress_ratios and the kv / index source ids of the pack), the first `--layers` trunk layers.
Dense linears are synthetic EXL3 trellis (K=2, mul1) like the 2.6 bpw pack; the routed SwitchGLU is replaced by a stub that returns zeros
(the MoE is not this bench's turf: moefuse owns it, its 5 launches/layer are added to the projection). The lm head is not run (vocab 1024
would misstate its 0.5 ms; it is one launch and unchanged by the glue fuse).

Arms (same install path as the served decode: fuse.install + rows.enable + S5):
  stock       GLUE_FUSE=0, fuse.MIXPRE on (defaults)
  glue1       GLUE_FUSE=1: collapse + RMSNorm in one launch (bit-identical)
  nomixpre    fuse.MIXPRE=0: the mix+collapse fusion off, the arrangement of the batched served path
  sg          KYOJIN_PORT_SUGLUE=1: fused hyper-connection stages A/B/E (suglue)
  lc          sg + KYOJIN_LAUNCHCUT=1 (launchcut groups; the bench MoE is a zero stub, so the logits gate checks chain/router/comb plumbing only)
  glue1x      GLUE_FUSE=1 and MIXPRE=0, i.e. the glue fuse on top of that arrangement
  stock2      a second stock model (the harness' own noise floor: two identical arms must be bitwise equal)
Steps interleave round-robin so machine drift hits every arm alike; dispatch counts come from the exported graph of each arm.

usage: ds41_glue_bench.py [--layers 40] [--steps 9] [--arms stock,glue1,glue2] [--no-rows]
"""
import collections
import json
import os
import re
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided",
         "StopGradient", "Depends"}

# ---- the real pack config (metadata only read: ~/ds41/model/config.json), engram off (its layers are not the glue's turf)
RATIOS = [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
KV_SOURCE = [2, 8, 14, 20]
INDEX_SOURCE = [2, 8, 14, 20, 24, 28, 32, 36]


def config(n_layers: int):
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    return ModelConfig.from_dict(dict(
        vocab_size=1024, hidden_size=5120, moe_intermediate_size=2304, num_hidden_layers=n_layers, num_attention_heads=64,
        num_key_value_heads=1, head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
        rms_norm_eps=1e-20, n_routed_experts=384, n_shared_experts=1, num_experts_per_tok=6, routed_scaling_factor=1.5,
        swiglu_limit=10.0, sliding_window=128, compress_ratios=RATIOS[:n_layers],
        kv_source_layer_ids=[i for i in KV_SOURCE if i < n_layers], index_source_layer_ids=[i for i in INDEX_SOURCE if i < n_layers],
        candidate_source_layer_id=20 if n_layers > 20 else -1, index_n_heads=32, index_head_dim=128, index_topk=512,
        candidate_topk_blocks=2048, candidate_block_size=8, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        compress_rope_theta=160000, rope_theta=10000,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536},
        engram_layer_ids=[], engram_num_embeddings=[]))


class _Wrap(nn.Module):
    """rows.enable / S5 take the model wrapper whose .language_model is the LanguageModel."""

    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


def stub_routed(language):
    """Replace DeepseekV41SwitchGLU by a tiny module: the routed MoE is stubbed (its launches/bytes are added analytically)."""
    import mlx.nn as nn

    class StubGLU(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.pad = mx.zeros((1,), dtype=mx.float16)

        def __call__(self, x, inds, weights):
            n = x.shape
            if getattr(self, "_z", None) is None or self._z.shape != (n[0], n[1], 1, n[-1]):
                self._z = mx.zeros((n[0], n[1], 1, n[-1]), dtype=x.dtype)
            return self._z

    language.DeepseekV41SwitchGLU = StubGLU
    language.DeepseekV41MoE.__init__.__globals__["DeepseekV41SwitchGLU"] = StubGLU


def to_exl3(lm, K: int = 2, seed: int = 0):
    """Synthetic-trellis EXL3 linears (the real pack is all-EXL3 dense): every linear the pack quantizes, at the pack's bitrate class."""
    import mlx.nn as nn
    from kyojin_mlx.exl3_linear import Exl3Linear
    rng = np.random.default_rng(seed)
    target = []
    for layer in lm.layers:
        a = layer.attn
        target += [(a, n) for n in ("wq_a", "wq_b", "wkv", "wo_b")]
        se = getattr(layer.ffn, "shared_experts", None)
        if se is not None:
            target += [(se, n) for n in ("gate_proj", "up_proj", "down_proj")]
        comp = getattr(a, "compressor", None)
        if comp is not None:
            target += [(comp, n) for n in ("wkv", "wgate")]
        idx = getattr(a, "indexer", None)
        if idx is not None:
            target += [(idx, n) for n in ("wq_b", "wk")]
    n = 0
    for owner, name in target:
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
    return n


def build(n_layers: int, rows: bool = True, seed: int = 0):
    import mlx_vlm.models.deepseek_v41.language as language
    from kyojin_mlx import archs
    from kyojin_mlx import fuse as FU
    from kyojin_mlx import fused_group as FG
    from kyojin_mlx import rows as RW
    stub_routed(language)
    lm = language.LanguageModel(config(n_layers))
    rng = np.random.default_rng(seed)
    from mlx.utils import tree_flatten, tree_unflatten
    flat = []
    for k, v in tree_flatten(lm.parameters()):
        if v.ndim >= 2 and not any(s in k for s in ("head", "gate", "hc_")):
            flat.append((k, mx.array((rng.standard_normal(v.shape) * 0.02).astype(np.float16))))
        elif "norm" in k:
            flat.append((k, mx.array((rng.standard_normal(v.shape) * 0.02).astype(np.float16))))
        else:
            flat.append((k, mx.array(rng.standard_normal(v.shape).astype(np.float32))))
    lm.update(tree_unflatten(flat))
    n = to_exl3(lm, seed=seed)
    mx.eval(lm.parameters())
    FU.install(archs.DS41)

    def fwd():
        c = lm.make_cache()
        mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=c))
        mx.eval(lm(mx.array([[9]]), cache=c))

    if rows:
        RW.enable(_Wrap(lm), archs.DS41)                      # the served install: row-invariant sites + S5 groups + the attention core adapter
    return lm, n


def dispatches(outs) -> tuple:
    path = os.path.join(tempfile.gettempdir(), f"glue_dot_{os.getpid()}.dot")
    mx.export_to_dot(path, outs)
    txt = open(path).read()
    os.unlink(path)
    labels = re.findall(r'label ?= ?"([^"]*)", shape=rectangle', txt)
    c = collections.Counter(labels)
    return c, len(labels) - sum(v for k, v in c.items() if k in VIEWS)


def glue_only(arms, layers: int, reps: int):
    """A/B of the block glue alone: the same blocks with the sublayers replaced by a deterministic stub, 2 * layers chained calls per
    measurement (dependent, like a decode step), arms interleaved. Isolates the launch change from the dense / MoE / attention work."""
    import mlx_vlm.models.deepseek_v41.language as L
    from kyojin_mlx import fuse as F
    from kyojin_mlx import glue_fuse as GF

    class SA:
        def __call__(self, x, cache):
            return (x * 0.5 + 0.25).astype(x.dtype)

    class SF:
        def __call__(self, x, image_mask=None):
            return (x * -0.75 + 0.125).astype(x.dtype)

    def stock_body(blk, h, pre, mask, cache):
        residual = h
        ap, apo, ac = blk.hc_mixes(h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base)
        x = blk.attn_norm(blk.hc_pre(h, pre))
        x = blk.attn(x, cache)
        x = L.hc_expand(x, residual, apo, ac)
        residual = x
        fp, fpo, fc = blk.hc_mixes(x, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base)
        x = blk.ffn_norm(blk.hc_pre(x, ap))
        x = blk.ffn(x, None)
        return L.hc_expand(x, residual, fpo, fc), fp

    models, blks = {}, {}
    for a in arms:
        GF.ENABLED = a == "glue1"
        models[a], _ = build(4, rows=True)                              # the block's own weights (norm weight, mix matrices)
        b = models[a].layers[2]
        b.attn, b.ffn = SA(), SF()
        blks[a] = b
    rng = np.random.default_rng(1)
    h0 = mx.array((rng.standard_normal((1, 1, 4, 5120)) * 0.02).astype(np.float16))
    pre0 = mx.array(rng.standard_normal((1, 1, 4)).astype(np.float32))

    def run(a, n):
        GF.ENABLED = a == "glue1"
        h, pre = h0, pre0
        for _ in range(n):
            out, pre = (GF.block(blks[a], h, pre, None, None) if a == "glue1" else stock_body(blks[a], h, pre, None, None))
            h = out
        return out

    for _ in range(3):                                                  # warm-up
        for a in arms:
            mx.eval(run(a, 2))
    n = 2 * layers
    res = {a: [] for a in arms}
    for _ in range(reps):
        for a in arms:                                                  # interleaved
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(run(a, n))
            res[a].append((time.perf_counter() - t0) / n)
    print(f"glue-only chain ({n} blocks = {layers} layers of 2 sublayers, stub sublayers):")
    for a in arms:
        print(f"  {a:6s} median {1e6 * statistics.median(res[a]):8.2f} us/block  min {1e6 * min(res[a]):8.2f}")
    if len(arms) == 2:
        d = [y - x for x, y in zip(res[arms[0]], res[arms[1]])]
        se = statistics.stdev(d) / len(d) ** 0.5
        print(f"  paired {arms[1]} - {arms[0]}: median {1e6 * statistics.median(d):+7.2f} us/block  mean {1e6 * statistics.mean(d):+7.2f} "
              f"+- {1e6 * se:.2f} (SE)  {sum(1 for x in d if x < 0)}/{len(d)} reps faster")
        print(f"  projected on 43 layers (86 blocks): {1e3 * statistics.median(d) * 86:+7.3f} ms/step")


def main():
    args = sys.argv
    layers = int(args[args.index("--layers") + 1]) if "--layers" in args else 40
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 9
    arms = (args[args.index("--arms") + 1] if "--arms" in args else "stock,glue1").split(",")
    use_rows = "--no-rows" not in args
    from kyojin_mlx import glue_fuse as GF
    if "--glue-only" in args:
        reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 11
        glue_only(arms, layers, reps)
        return
    models, counts = {}, {}
    for a in arms:
        GF.ENABLED = a in ("glue1", "glue1x")
        from kyojin_mlx import fuse as FU0
        FU0.MIXPRE = a not in ("nomixpre", "glue1x")
        models[a], counts[a] = build(layers, rows=use_rows, seed=0)
        print(f"[{a}] built {layers} layers, {counts[a]} EXL3 linears, peak {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    from kyojin_mlx import fuse as FU_

    def flags(a):
        GF.ENABLED = a in ("glue1", "glue1x")                          # glue1x: the glue fuse with the mix+collapse fusion off
        FU_.MIXPRE = a not in ("nomixpre", "glue1x")
        from kyojin_mlx import suglue as SG_
        from kyojin_mlx import launchcut as LC_
        LC_.ENABLED = a == "lc"
        SG_.ENABLED = a in ("sg", "sg2", "lc")                               # sg / sg2: KYOJIN_PORT_SUGLUE stages (sg2 = a second, identical arm)

    rng = np.random.default_rng(3)
    ids = [int(x) for x in rng.integers(3, 1024, 8)]
    caches = {a: models[a].make_cache() for a in arms}          # every arm built first, then the caches advance in lockstep: building a
    for t in range(-1, 6):                                      # model later must not land inside another arm's stream
        for a in arms:
            flags(a)
            tok = int(ids[0]) if t < 0 else (t * 37) % 997 + 3
            mx.eval(models[a](mx.array([[tok]]), cache=caches[a]).logits)
    res = {a: [] for a in arms}
    lg = {}
    for s in range(steps):
        for a in arms:
            flags(a)
            t0 = time.perf_counter()
            out = models[a](mx.array([[(s * 61) % 997 + 3]]), cache=caches[a]).logits
            mx.eval(out)
            res[a].append(time.perf_counter() - t0)
            if s == steps - 1:
                lg[a] = np.array(out[0, -1]).astype(np.float32)
    print()
    for a in arms:
        v = res[a]
        print(f"{a:6s} median {1e3 * statistics.median(v):7.3f}  min {1e3 * min(v):7.3f}  max {1e3 * max(v):7.3f} ms over {steps} steps")
    if len(arms) == 2:                                          # paired per-step difference: robust to the machine drifting under the arms
        d = [y - x for x, y in zip(res[arms[0]], res[arms[1]])]
        se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        print(f"paired {arms[1]} - {arms[0]}: median {1e3 * statistics.median(d):+7.3f} ms  mean {1e3 * statistics.mean(d):+7.3f} "
              f"+- {1e3 * se:.3f} (SE)  {sum(1 for x in d if x < 0)}/{len(d)} steps faster")
    a0 = lg[arms[0]]
    for a in arms[1:]:
        if np.array_equal(a0, lg[a]):
            print(f"gate {a}: logits BIT-IDENTICAL to {arms[0]}")
        else:
            d = float(np.max(np.abs(a0 - lg[a]))) / (float(np.max(np.abs(a0))) + 1e-30)
            l2 = float(np.linalg.norm(a0 - lg[a])) / (float(np.linalg.norm(a0)) + 1e-30)
            print(f"gate {a}: logits maxrel {d:.3e} relL2 {l2:.3e} vs {arms[0]}")
    # dispatch counts: one more step per arm, counted on its graph
    print()
    for a in arms:
        flags(a)
        out = models[a](mx.array([[7]]), cache=caches[a]).logits
        c, disp = dispatches([out])
        print(f"{a:6s} {disp} dispatches/step ({disp / layers:.2f}/layer)")
        if "--json" in args:
            json.dump(dict(arm=a, layers=layers, dispatch=disp, by_type=dict(c),
                           ms={k: statistics.median(v) for k, v in res.items()}),
                      open(args[args.index("--json") + 1] + f".{a}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
