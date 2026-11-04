#!/usr/bin/env python3
"""Interleaved A/B of the decode glue on a multi-layer DS4.1 stack (synthetic EXL3 trellis, real dims, routed MoE stubbed).

Two stacks of the same seed are built in one process, one per arm: the compiled regions of a block cache their trace, so an arm cannot be
toggled inside one model (a mid-run flag flip would keep replaying the first trace). Each arm owns its cache and warms up on its own model;
the steps then interleave round-robin, so machine drift hits every arm equally. Reports median / min ms per step, dispatches per step (dot
census, views excluded) and the gate against the first arm (bit-identical logits or rel L2).

usage: glue_ab.py [--layers 4] [--role reuse] [--steps 15] [--arms off,on] [--json OUT]
"""
import json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))

import mlx.core as mx
import numpy as np

def cbcount():
    """tools/cbcount: sums the GPU time of the command buffers that complete (the step's GPU busy time, robust to host noise)."""
    import ctypes, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    src, so = os.path.join(here, "cbcount", "cbcount.mm"), os.path.join(here, "cbcount", "libcbcount.dylib")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["clang++", "-O2", "-std=c++17", "-fobjc-arc", "-dynamiclib", "-framework", "Metal", "-framework", "Foundation", src, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.cb_gpu_us.restype = ctypes.c_double
    lib.cb_commits.restype = ctypes.c_long
    lib.cb_completed.restype = ctypes.c_long
    lib.cb_reset.argtypes = []
    lib.cb_install()
    return lib


VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided", "StopGradient", "Depends"}


def build_stack(layers: int, role: str = "reuse", seed: int = 0):
    """n-layer slice of the real DS4.1 dims, every dense linear a synthetic-treillis EXL3 (K=2); role reuse = ratio 1 (no compressor / indexer)."""
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx_vlm.models.deepseek_v41.language import LanguageModel
    from mlx.utils import tree_flatten, tree_unflatten
    from kyojin_mlx.exl3_linear import Exl3Linear
    import mlx.nn as nn
    spec = [1] * layers if role == "reuse" else [2] * layers
    cfg = ModelConfig.from_dict(dict(
        vocab_size=1024, hidden_size=5120, moe_intermediate_size=512,
        num_hidden_layers=layers, num_attention_heads=64, num_key_value_heads=1,
        head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
        rms_norm_eps=1e-6, n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=1.5, sliding_window=128, compress_ratios=spec,
        kv_source_layer_ids=list(range(layers)) if role != "reuse" else [],
        index_source_layer_ids=list(range(layers)) if role != "reuse" else [],
        candidate_source_layer_id=-1, index_n_heads=32, index_head_dim=128, index_topk=512,
        candidate_topk_blocks=2048, candidate_block_size=8, hc_mult=4, hc_sinkhorn_iters=20,
        engram_layer_ids=[], engram_num_embeddings=[]))
    lm = LanguageModel(cfg)
    rng = np.random.default_rng(seed)
    flat = []
    for k, v in tree_flatten(lm.parameters()):
        if v.ndim >= 2 and not any(s in k for s in ("head", "gate", "hc_")):
            flat.append((k, mx.array((rng.standard_normal(v.shape) * 0.02).astype(np.float16))))
        elif "norm" in k:
            flat.append((k, mx.array((rng.standard_normal(v.shape) * 0.02).astype(np.float16))))
        else:
            flat.append((k, mx.array(rng.standard_normal(v.shape).astype(np.float32))))
    lm.update(tree_unflatten(flat))
    mx.eval(lm.parameters())
    skip = ("head", "hc_", "sink", "weights_proj", "wo_a")
    n = 0
    for layer in lm.layers:
        a = layer.attn
        targets = [(a, nm) for nm in ("wq_a", "wq_b", "wkv", "wo_b")]
        se = getattr(layer.ffn, "shared_experts", None)
        if se is not None:
            targets += [(se, nm) for nm in ("gate_proj", "up_proj", "down_proj")]
        for owner, name in targets:
            if any(s in name for s in skip):
                continue
            mod = getattr(owner, name, None)
            if not isinstance(mod, nn.Linear):
                continue
            kin, nout = mod.weight.shape[1], mod.weight.shape[0]
            if kin % 128 or nout % 128:
                continue
            tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 32)).astype(np.int16)
            su = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
            sv = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
            setattr(owner, name, Exl3Linear(mx.array(tr), mx.array(su), mx.array(sv), cb="mul1", out_dtype=mx.float16))
            n += 1
    mx.eval(lm.parameters())

    def stub(self, x, inds, weights):
        return mx.zeros((x.shape[0], x.shape[1], 1, x.shape[-1]), dtype=x.dtype)
    for layer in lm.layers:
        layer.ffn.switch_mlp.__class__ = type("StubGLU", (type(layer.ffn.switch_mlp),), {"__call__": stub})
    return lm, n


def census(out):
    import re, tempfile
    p = os.path.join(tempfile.gettempdir(), f"glueab_{os.getpid()}.dot")
    mx.export_to_dot(p, out)
    txt = open(p).read()
    os.unlink(p)
    labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt)
    return sum(1 for x in labels if x not in VIEWS)


def identical_state_gate(names, setarm, layers, role):
    """Same model, same cache state, one arm at a time: the glue cuts must be bit-identical. Runs the eager graph (no compiled regions), so
    a flag flip takes effect immediately (a traced region would keep the first trace) and every glue launch is visible."""
    os.environ["KYOJIN_FUSE_COMPILE"] = "0"
    import importlib
    from kyojin_mlx import archs, fuse as FU, pf_glue as PG, rows as RW
    lm, _ = build_stack(layers, role)
    FU.install(archs.DS41)
    PG.memoize_fq()
    for layer in lm.layers:
        RW._override(layer, "hc_mixes", RW.SITES["hc_block_mixes"])
        RW._override(layer.ffn.gate, "__call__", RW.SITES["router_sqrtsoftplus"])
    cache = lm.make_cache()[0]                            # the DS4.1 cache holds every layer's state in one object
    mx.eval(lm(mx.array([[3, 7, 11, 13, 17, 19, 23, 29]]), cache=[cache]))
    for t in range(6):
        mx.eval(lm(mx.array([[t + 3]]), cache=[cache]))
    snaps = [(list(cache.state), cache.offset)]
    for s in range(4):                                     # a few states along the run, all arms replay each of them
        mx.eval(lm(mx.array([[s + 41]]), cache=[cache]))
        snaps.append((list(cache.state), cache.offset))
    out = {}
    ref = None
    for nm in names:
        setarm(nm)
        got = []
        for state, off in snaps:
            cache.state = list(state)
            cache.offset = off
            mx.eval(*cache.state)
            y = lm(mx.array([[7]]), cache=[cache]).logits
            mx.eval(y)
            got.append(np.array(y))
        out[nm] = got
        if ref is None:
            ref = got
    res = {}
    for nm, got in out.items():
        same = all(np.array_equal(a, b) for a, b in zip(ref, got))
        worst = max(float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30)) for a, b in zip(ref, got))
        res[nm] = "ref" if nm == names[0] else ("BIT-IDENTICAL" if same else f"maxrel {worst:.2e}")
    return res


def main():
    argv = sys.argv
    layers = int(argv[argv.index("--layers") + 1]) if "--layers" in argv else 4
    role = argv[argv.index("--role") + 1] if "--role" in argv else "reuse"
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else 15
    from kyojin_mlx import archs, fuse as FU, pf_glue as PG, rows as RW, gluefuse as GF

    def patch(lm):
        FU.install(archs.DS41)
        PG.memoize_fq()
        for layer in lm.layers:
            RW._override(layer, "hc_mixes", RW.SITES["hc_block_mixes"])
            RW._override(layer.ffn.gate, "__call__", RW.SITES["router_sqrtsoftplus"])

    from kyojin_mlx import fuse as F

    def setarm(name):
        """Arms: off, mixvec, mixargs, mixnorm, mixpre_off (the fold off), on (every cut). The fold is read when a region is traced, so it
        follows the module flag at build / warm-up time (one model per arm)."""
        GF.ENABLED = name not in ("off", "mixpre_off")
        for flag in GF.PARTS:
            setattr(GF, flag, (name == "on") or (name.lower() == flag.lower()))
        F.MIXPRE = name not in ("mixpre_off", "on")

    names = (argv[argv.index("--arms") + 1].split(",") if "--arms" in argv else ["off", "on", "all"])
    models = {}
    for nm in names:
        setarm(nm)
        lm, nlin = build_stack(layers, role)                  # same seed: identical weights in every arm
        patch(lm)
        if nm == names[0]:
            saved, GF.ENABLED = GF.ENABLED, True              # install once (the arms toggle ENABLED / the part flags at call time)
            n_inst = GF.install()
            GF.ENABLED = saved
            print(f"stack: {layers} layers ({role}), {nlin} EXL3 linears, gluefuse install {n_inst}", flush=True)
        models[nm] = lm
    setarm("off")
    rng = np.random.default_rng(7)
    prompt = rng.integers(3, 1024, 8).tolist()
    caches = {}
    for nm in names:
        setarm(nm)
        c = models[nm].make_cache()
        mx.eval(models[nm](mx.array([prompt]), cache=c))
        for t in range(12):
            mx.eval(models[nm](mx.array([[t % 997 + 3]]), cache=c))
        caches[nm] = c
    lib = cbcount()
    res = {nm: [] for nm in names}
    gpu = {nm: [] for nm in names}
    for s in range(steps):
        for nm in names:
            setarm(nm)
            mx.synchronize()
            lib.cb_reset()
            t0 = time.perf_counter()
            y = models[nm](mx.array([[(s % 997) + 3]]), cache=caches[nm]).logits
            mx.eval(y)
            mx.synchronize()
            res[nm].append(time.perf_counter() - t0)
            gpu[nm].append(lib.cb_gpu_us() / 1e3)
    disp = {}
    for nm in names:
        setarm(nm)
        disp[nm] = census(models[nm](mx.array([[17]]), cache=caches[nm]).logits)
        mx.eval(models[nm](mx.array([[17]]), cache=caches[nm]).logits)
    setarm("off")
    base = res[names[0]]
    gate = identical_state_gate(names, setarm, layers, role) if "--no-gate" not in argv else {}
    print(f"{'arm':8s} {'med ms':>8s} {'min ms':>8s} {'gpu med':>8s} {'gpu min':>8s} {'disp':>6s} {'gate':>15s}", flush=True)
    for nm in names:
        v = res[nm]
        print(f"{nm:8s} {1e3 * statistics.median(v):8.3f} {1e3 * min(v):8.3f} {statistics.median(gpu[nm]):8.3f} {min(gpu[nm]):8.3f} {disp[nm]:6d} {gate.get(nm, '-'):>15s}", flush=True)
    for nm in names[1:]:
        d = statistics.median(res[nm]) - statistics.median(base)
        n = max(1, len(gpu[nm]))
        dg = statistics.median(gpu[nm]) - statistics.median(gpu[names[0]])
        dm = min(gpu[nm]) - min(gpu[names[0]])
        se = statistics.stdev(gpu[nm]) / n ** 0.5 if n > 1 else 0.0
        print(f"delta {nm:6s} gpu med {dg:+7.3f} +- {se:5.3f} (SE), gpu min {dm:+7.3f} ms/step on {layers} layers -> "
              f"{dg * 40 / layers:+7.3f} (med) / {dm * 40 / layers:+7.3f} (min) ms on the 40-layer step, wall {1e3 * d:+7.3f}, "
              f"dispatches {disp[nm] - disp[names[0]]:+d}/step -> {(disp[nm] - disp[names[0]]) * 40 // layers:+d} on 40", flush=True)
    if "--json" in argv:
        json.dump(dict(layers=layers, role=role, steps=steps, arms={nm: dict(median=statistics.median(res[nm]), min=min(res[nm]), disp=disp[nm],
                  gpu=gpu[nm], samples=res[nm]) for nm in names}), open(argv[argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
