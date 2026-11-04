#!/usr/bin/env python3
"""Per-dispatch census of ONE DS4.1 decode layer: every Metal dispatch with its kernel name, grid, GPU time and the wall time of its
command buffer (gap to the previous one), grouped by op class.

Method: the Metal hooks of replay/rp.mm (kyojin_mlx/replay.py) tape every dispatch of one eager decode step (kernel name + grid, rp_dump), and
with MLX_MAX_OPS_PER_BUFFER=1 every dispatch sits in its own command buffer so the same hooks give its GPUStart/GPUEnd time (rp_time). The
graph is the eager one (KYOJIN_FUSE_COMPILE=0), i.e. exactly the kernels the compiled regions replay. Views are not dispatches.

usage: glue_tape.py [--role reuse|full|pair] [--exl3] [--no-fuse] [--json OUT]
"""
import ctypes, json, os, re, sys, collections

os.environ["MLX_MAX_OPS_PER_BUFFER"] = "1"                      # one dispatch per command buffer: per-dispatch GPU times
os.environ.pop("KYOJIN_REPLAY", None)                            # no replay: the tape must see the region bodies
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))


def groups():
    """op class of a Metal kernel name / mlx primitive name."""
    G = [
        ("gemv_exl3", r"^kyojin_.*(gemv|rot|final|partial)|^exl3|mul1|trellis"),
        ("copy/quant", r"quant|fp8|as_type|astype|copy|contiguous|concatenate|Concatenate"),
        ("norm", r"rms|norm"),
        ("router", r"router|sinkhorn|mix|hc_"),
        ("mhc", r"mhc"),
        ("attn", r"attend|sdpa|viterbi|window|kv_ring|compressor|indexer|dsa|select|scores"),
        ("matmul", r"matmul|^Matmul|gemv|qmm|Steel"),
        ("rope", r"rope|RoPE"),
        ("elementwise", r"mul|add|div|sub|exp|log|sqrt|sigmoid|maximum|minimum|sum|reduce|Mean|Softmax|slice|gather"),
    ]
    return [(n, re.compile(p, re.I)) for n, p in G]


def klass(name):
    for n, p in groups():
        if p.search(name):
            return n
    return "other"


def main():
    from ds41_slice_census import build, stub_routed
    from kyojin_mlx import archs, fuse as FU
    role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "reuse"
    if "--no-fuse" in sys.argv:
        os.environ["KYOJIN_FUSE"] = "0"; os.environ["KYOJIN_FUSE_ATTN"] = "0"
        os.environ["KYOJIN_FUSE_INDEX"] = "0"; os.environ["KYOJIN_FUSE_COMPILE"] = "0"
    import mlx.core as mx
    from kyojin_mlx import replay as RP
    RP.install()                                                # tape hooks before any pipeline is built
    lm = build(role, exl3="--exl3" in sys.argv, dtype=mx.float16 if "--exl3" in sys.argv else mx.bfloat16)
    FU.install(archs.DS41)
    from kyojin_mlx import gluefuse as GF
    if GF.ENABLED:
        GF.install()
    if "--served" in sys.argv:                                  # the decode stack of a patched pack: rows sites (mHC mix vector, router) + fq memo
        from kyojin_mlx import rows as RW, pf_glue as PG
        PG.memoize_fq()
        for layer in lm.layers:
            RW._override(layer, "hc_mixes", RW.SITES["hc_block_mixes"])
            RW._override(layer.ffn.gate, "__call__", RW.SITES["router_sqrtsoftplus"])
    stub_routed(lm)
    if "--no-h32" in sys.argv:                                  # probe: what does the fp32 mHC side channel cost in launches?
        FU.take_f32 = lambda x: None
    if "--core-zero" in sys.argv:                               # census of the block glue alone: the attention core (fuse_attn) is not my subject
        import kyojin_mlx.fuse_attn as FA
        from kyojin_mlx.rows import Sink
        FA_dims = (lm.layers[0].attn.o_groups, lm.layers[0].attn.head_dim, lm.layers[0].attn.n_heads // lm.layers[0].attn.o_groups)
        FA.core = lambda a, L, cache, x, qr, q, kv: (mx.zeros((1, FA_dims[0], 1, FA_dims[1] * FA_dims[2]), dtype=x.dtype), None)
    lib = RP._load()
    lib.rp_dump.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.rp_time.argtypes = [ctypes.c_int]
    lib.rp_gpu_times.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    cache = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=cache))
    for t in range(4):
        mx.eval(lm(mx.array([[9 + t]]), cache=cache))
    mx.synchronize()
    SLOT = 12
    lib.rp_rec_begin(SLOT)
    lib.rp_time(1)
    mx.eval(lm(mx.array([[21]]), cache=cache).logits)
    mx.synchronize()
    n = lib.rp_rec_end()
    path = os.path.join(HERE, "..", "scratch", f"tape_{role}.txt")
    lib.rp_dump(SLOT, path.encode())
    N = 8192
    g = (ctypes.c_double * N)(); w = (ctypes.c_double * N)()
    nt = lib.rp_gpu_times(g, w, N)
    lines = open(path).read().splitlines()
    rows = []
    for i, ln in enumerate(lines):
        p = ln.split()
        idx, name = int(p[0]), p[1]
        grid = " ".join(p[2:4])
        rows.append(dict(i=idx, name=name, grid=grid, gpu=g[i] if i < nt else float("nan"), wall=w[i] if i < nt else float("nan")))
    tot_g = sum(r["gpu"] for r in rows if r["gpu"] == r["gpu"])
    tot_w = sum(r["wall"] for r in rows if r["wall"] == r["wall"])
    print(f"role={role} exl3={'--exl3' in sys.argv} fuse={'off' if '--no-fuse' in sys.argv else 'on'}: {n} dispatches, "
          f"gpu {tot_g/1e3:.3f} ms, wall {tot_w/1e3:.3f} ms, {nt} timed command buffers", flush=True)
    by = collections.defaultdict(lambda: dict(n=0, gpu=0.0, wall=0.0))
    for r in rows:
        k = klass(r["name"])
        b = by[k]
        b["n"] += 1
        b["gpu"] += r["gpu"] if r["gpu"] == r["gpu"] else 0.0
        b["wall"] += r["wall"] if r["wall"] == r["wall"] else 0.0
    print(f"{'class':14s} {'disp':>5s} {'gpu us':>9s} {'gpu/disp':>9s} {'wall us':>9s}", flush=True)
    for k, b in sorted(by.items(), key=lambda kv: -kv[1]["gpu"]):
        print(f"{k:14s} {b['n']:5d} {b['gpu']:9.1f} {b['gpu']/b['n']:9.2f} {b['wall']:9.1f}", flush=True)
    print(f"{'SUM':14s} {n:5d} {tot_g:9.1f} {tot_g/max(n,1):9.2f} {tot_w:9.1f}", flush=True)
    print("--- per dispatch (i, kernel, grid, gpu us, wall us)", flush=True)
    for r in rows:
        print(f"{r['i']:4d} {r['name'][:44]:44s} {r['grid'][:34]:34s} {r['gpu']:8.2f} {r['wall']:9.2f}", flush=True)
    if "--json" in sys.argv:
        json.dump(dict(role=role, dispatches=n, rows=rows, by_class={k: v for k, v in by.items()}),
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
