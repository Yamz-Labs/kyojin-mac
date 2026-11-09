"""Micro gate of the DS4.1 decode attention core (dsv4_core.core, plain decode, n=1): real attention shapes, synthetic caches at CTX positions,
the six layer kinds of the served model, chained in one eval. usage: bench_attncore.py CTX [reps] [fuse]  (fuse: attn_core_fuse 0/1)
Prints per kind: launches (graph census), us per call; and the step-equivalent (kind counts of the 40 layers)."""
import os, sys, time, collections
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import decstage_fixture as FX
from kyojin_mlx import archs, dsv4_core as DC, rows as RW

KINDS = ["local", "full_r2", "reuse_r2", "full_r1", "reindex_r1", "reuse_r1"]
COUNT = dict(local=2, full_r2=3, reuse_r2=15, full_r1=1, reindex_r1=4, reuse_r1=15)       # of the 40 served layers
REAL = dict(hidden_size=5120, num_attention_heads=64, head_dim=512, q_lora_rank=1280, o_lora_rank=1024, o_groups=8, moe_intermediate_size=128,
            index_n_heads=32, index_head_dim=128, index_topk=512, sliding_window=128, candidate_topk_blocks=2048, candidate_block_size=8,
            num_hidden_layers=6, compress_ratios=[0, 2, 2, 1, 1, 1], kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 4],
            candidate_source_layer_id=3, max_position_embeddings=1 << 17)


def build():
    m, cfg = FX.build(**REAL)
    for lay in m.language_model.layers:                       # the pack keeps the sink in fp32 (the fixture rounds every fp32 parameter to fp16)
        lay.attn.attn_sink = lay.attn.attn_sink.astype(mx.float32)
    RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
    return m


def census(out):
    c = FX.census(out)
    return c["dispatches"]


def prefill(m, ctx):
    ids = FX.toks(ctx)
    cache = m.make_cache()
    t = time.time()
    for s in range(0, ctx, 4096):
        with RW.HL.last_row():
            lg = m.language_model(mx.array(ids[s:s + 4096])[None], cache=cache).logits
        mx.eval(lg)
    print(f"prefill {ctx}: {time.time() - t:.1f}s", flush=True)
    return RW.batch([cache])


def inputs(m, seed=1):
    rng = np.random.default_rng(seed)
    cfg = m.language_model.layers[0].attn.config
    H, D = cfg.num_attention_heads, cfg.head_dim
    f = lambda *s, sc=1.0: mx.array((rng.standard_normal(s) * sc).astype(np.float16))
    out = []
    for l, lay in enumerate(m.language_model.layers):
        x = f(1, 1, cfg.hidden_size)
        out.append((x, (f(1, 1, cfg.q_lora_rank), f(1, 1, H, D, sc=0.5), f(1, 1, D, sc=2.0))))
    return out


def run_layers(m, cache, ins, ls):
    outs = []
    for l in ls:
        attn = m.language_model.layers[l].attn
        x, pre = ins[l]
        outs.append(DC.core(attn, x, cache[0], pre=pre, stage=True))
    return outs


def main():
    ctx = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    m = build()
    cache = prefill(m, ctx)
    ins = inputs(m)
    ARMS = [int(a) for a in (sys.argv[3] if len(sys.argv) > 3 else '0,1').split(',')]
    res = {k: {a: [0, [], []] for a in ARMS} for k in KINDS}
    for fz in ARMS + ARMS:                                  # warm both arms (plan constants, kernels)
        DC.ATTN_CORE_FUSE = bool(fz)
        mx.eval(run_layers(m, cache, ins, range(6)))
    for l, kind in enumerate(KINDS):
        for fz in ARMS:
            DC.ATTN_CORE_FUSE = bool(fz)
            o = run_layers(m, cache, ins, [l]); res[kind][fz][0] = census(o[0]); mx.eval(o)
    NC = 16
    for l, kind in enumerate(KINDS):
        for r in range(reps):
            for fz in (ARMS if r % 2 == 0 else ARMS[::-1]):
                DC.ATTN_CORE_FUSE = bool(fz)
                time.sleep(0.02)
                t0 = time.perf_counter()
                o = run_layers(m, cache, ins, [l] * NC)         # NC calls chained per eval (same layer, same state)
                t1 = time.perf_counter()
                mx.eval(o)
                t2 = time.perf_counter()
                res[kind][fz][1].append((t2 - t0) / NC * 1e6); res[kind][fz][2].append((t1 - t0) / NC * 1e6)
    for k in KINDS:
        for fz in ARMS:
            r = res[k][fz]
            print(f"ctx {ctx} fuse {fz} {k:11s} launches {r[0]:3d}  us/call min {min(r[1]):7.1f} med {np.median(r[1]):7.1f}  host(build) min {min(r[2]):6.1f}")
    for fz in ARMS:
        print(f"ctx {ctx} fuse {fz} STEP-EQUIV launches {sum(COUNT[k] * res[k][fz][0] for k in KINDS)}  ms min {sum(COUNT[k] * min(res[k][fz][1]) for k in KINDS) / 1e3:.2f}"
              f"  med {sum(COUNT[k] * np.median(res[k][fz][1]) for k in KINDS) / 1e3:.2f}")


if __name__ == "__main__":
    main()
