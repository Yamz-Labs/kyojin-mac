#!/usr/bin/env python3
"""CPU census of one DS4.1 decode step: Python->mlx calls, graph nodes (views included) and Metal dispatches, per layer role.

No Metal GPU is touched: every `mx.fast.metal_kernel` is replaced by a stub that returns one `Arange` node (one node = one launch, unique
start so mx.compile cannot merge two launches), and the whole step runs on the CPU stream. The stack is the synthetic DS4.1 stack of
tools/ds41_glue_bench.py (real dims and layer roles, EXL3 linears, routed MoE stubbed to zeros, so the 5 MoE launches/layer are NOT in the
numbers unless --moe is given: then the stub is replaced by a 5-launch stand-in with the real MoeGather launch count).

  arms: base (suglue off), sg (KYOJIN_PORT_SUGLUE), lc (KYOJIN_LAUNCHCUT, when the module exists)
usage: launchcut_census.py [--layers 40] [--arms base,sg,lc] [--json out.json]
"""
import collections
import os
import time
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("KYOJIN_VLM_PKG", os.path.expanduser("~/kyojin-mlx/vlm_pkg")))

import mlx.core as mx

VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided",
         "StopGradient", "Depends"}
_SEQ = [0]
TAPE: list = []


_REAL = [None]
STUB_ON = [True]


class _Kernel:
    def __init__(self, name, **kw):
        self.name = name
        self.real = _REAL[0](name, **kw)

    def __call__(self, inputs, output_shapes, output_dtypes, **kw):
        TAPE.append(self.name)
        if not STUB_ON[0]:                                  # census step: the real CustomKernel node, graph only (never evaluated)
            return self.real(inputs=inputs, output_shapes=output_shapes, output_dtypes=output_dtypes, **kw)
        outs = [mx.zeros(sh, dt) for sh, dt in zip(output_shapes, output_dtypes)]
        return mx.depends(outs, list(inputs))


def install_stub():
    mx.set_default_device(mx.cpu)
    _REAL[0] = mx.fast.metal_kernel
    mx.fast.metal_kernel = lambda name, **kw: _Kernel(name, **kw)
    mx.metal.is_available = lambda: True


def arrays_of(obj, seen=None, out=None):
    seen = set() if seen is None else seen
    out = [] if out is None else out
    if isinstance(obj, mx.array):
        out.append(obj)
    elif id(obj) in seen:
        return out
    elif isinstance(obj, (list, tuple)):
        seen.add(id(obj))
        [arrays_of(v, seen, out) for v in obj]
    elif isinstance(obj, dict):
        seen.add(id(obj))
        [arrays_of(v, seen, out) for v in obj.values()]
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        seen.add(id(obj))
        [arrays_of(v, seen, out) for v in vars(obj).values()]
    return out


def dot_counts(outs):
    path = os.path.join(tempfile.gettempdir(), f"lc_dot_{os.getpid()}.dot")
    mx.export_to_dot(path, outs)
    txt = open(path).read()
    os.unlink(path)
    labels = re.findall(r'label ?= ?"([^"]*)", shape=rectangle', txt)
    c = collections.Counter(labels)
    return c, len(labels), len(labels) - sum(v for k, v in c.items() if k in VIEWS)


class BuildTime:
    """Wall time of the python graph build of the step (never evaluated): what the host pays before it encodes. (A count of python->mlx calls is not
    available: mlx's nanobind functions raise no profile events.)"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1e3


def install_moe(lm, experts=384, K=2):
    """Replace the zero stub of every layer by the served routed MoE (patch.build_moe: MoeGather, or sumoe / moefuse when their flag is on).
    The stacks hold dummy words: the stub kernels never read them and the measured graph is never evaluated."""
    import numpy as np
    from types import SimpleNamespace
    from kyojin_mlx import archs, moe as M, patch as P

    def stack(kin, nout):
        kt, nt = kin // 16, nout // 16
        return M.Stack(mx.zeros((4,), mx.uint32), mx.zeros((experts + 1,), mx.uint32), mx.array(np.full(experts, K, "u1")),
                       mx.ones((experts, kin), mx.float16), mx.ones((experts, nout), mx.float16), kt, nt, "mul1")

    n = 0
    for layer in lm.layers:
        st = {"gate": stack(5120, 2304), "up": stack(5120, 2304), "down": stack(2304, 5120)}
        old = SimpleNamespace(activation=SimpleNamespace(limit=10.0))
        layer.ffn.switch_mlp = P.build_moe(old, layer.ffn, st, archs.DS41.moe, materialize=False)
        n += 1
    return n


def advance(lm, c, tok, rows=1):
    """One real (stub-kernel, CPU) step: evaluates the logits and every cache array so the graph of the next step starts clean."""
    from kyojin_mlx import flags as FL
    if not STUB_ON[0]:
        STUB_ON[0] = True
        FL.GEN += 1                                          # compiled regions retrace with the stub kernels
    mx.set_default_device(mx.cpu)
    mx.eval(lm(mx.array([[tok]]), cache=c).logits)
    mx.eval(arrays_of(c))
    if rows > 1:                                             # warm the chained path (its first call per row count runs the stock code) on the stub kernels
        for _ in range(2):
            mx.eval(make_run(lm, tok, rows)(c))


def make_run(lm, tok, rows=1):
    """The decode step the census builds: rows == 1 the solo step on a fork; rows > 1 a chained verify of `rows` positions (spec.make_chain, the MTP path)."""
    from kyojin_mlx import rows as RW, spec as SP
    if rows == 1:
        return lambda c: lm(mx.array([[tok]]), cache=RW.fork(c)).logits
    toks = mx.array([[tok + i] for i in range(rows)], dtype=mx.int32)
    return lambda c: lm(toks, cache=[SP.make_chain(c, rows)]).logits


def measure(lm, c, tok, rows=1):
    """Build (never evaluate) the lazy graph of one decode step with the real CustomKernel nodes; count python calls, nodes, dispatches."""
    import mlx_vlm.models.deepseek_v41.language as L
    from kyojin_mlx import flags as FL
    outs, orig = [], L.DeepseekV41Block.__call__

    marks = []

    def hook(self, *a, **k):
        r = orig(self, *a, **k)
        outs.append(r)
        marks.append(len(TAPE))
        return r

    L.DeepseekV41Block.__call__ = hook
    TAPE.clear()
    FL.GEN += 1                                              # compiled regions retrace: their tapes now hold the real CustomKernel nodes
    STUB_ON[0] = False
    mx.set_default_device(mx.gpu)                            # graph building only: nothing below is evaluated, no Metal work is submitted
    try:
        from kyojin_mlx import rows as RW
        run = make_run(lm, tok, rows)
        run(c)                                               # trace pass on a fork (the compiled regions trace here), graph dropped
        names = [TAPE[a:b] for a, b in zip([0] + marks[:-1], marks)]                   # custom kernels named per layer (region bodies appear at their trace)
        outs.clear()
        TAPE.clear()
        with BuildTime() as pc:
            outs.clear()
            logits = run(c)
        tot_c, tot_nodes, tot_disp = dot_counts([logits])
        per, per_types, prev_n, prev_d, prev_c = [], [], 0, 0, collections.Counter()
        for h, pm in outs:
            cc, n, d = dot_counts([h, pm])
            per.append((n - prev_n, d - prev_d))
            per_types.append(cc - prev_c)
            prev_n, prev_d, prev_c = n, d, cc
    finally:
        L.DeepseekV41Block.__call__ = orig
        mx.set_default_device(mx.cpu)
    return dict(build_ms=pc.ms, nodes=tot_nodes, dispatches=tot_disp, by_type=dict(tot_c), per_layer=per, per_types=per_types, names=names)


def census(lm, steps=2, rows=1):
    """Average over `steps` consecutive positions (the compressors pool every ratio-th token: one position is not representative)."""
    c = lm.make_cache()
    mx.eval(lm(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]), cache=c).logits)
    for s in range(3):
        advance(lm, c, 9 + s)
    res = []
    for k in range(steps):
        advance(lm, c, 30 + k, rows)
        res.append(measure(lm, c, 40 + k, rows))
        STUB_ON[0] = True
    n = len(res)
    avg = lambda f: sum(f(r) for r in res) / n
    per = [(sum(r["per_layer"][i][0] for r in res) / n, sum(r["per_layer"][i][1] for r in res) / n) for i in range(len(res[0]["per_layer"]))]
    return dict(build_ms=avg(lambda r: r["build_ms"]), nodes=avg(lambda r: r["nodes"]), dispatches=avg(lambda r: r["dispatches"]),
                per_layer=per, by_type={k: sum(r["by_type"].get(k, 0) for r in res) / n for k in set().union(*[r["by_type"] for r in res])},
                steps=res)


def main():
    a = sys.argv
    layers = int(a[a.index("--layers") + 1]) if "--layers" in a else 40
    arms = (a[a.index("--arms") + 1] if "--arms" in a else "base,sg").split(",")
    steps = int(a[a.index("--steps") + 1]) if "--steps" in a else 2
    rows = int(a[a.index("--rows") + 1]) if "--rows" in a else 1          # > 1: chained verify of that many positions
    install_stub()
    import ds41_glue_bench as G
    from kyojin_mlx import suglue as SG
    lm, n = G.build(layers, rows=True)
    if "--flags" in a:                                                  # e.g. --flags port_sumoe=1,moefuse=1 (kyojin_mlx.flags names)
        from kyojin_mlx import flags as FL
        FL.set_many(dict(kv.split("=") for kv in a[a.index("--flags") + 1].split(",")))
    if rows > 1:
        from kyojin_mlx import archs, spec as SP
        SP.install(G._Wrap(lm), archs.DS41)                                 # the chained-verify attention adapter (spec.dsv4_attention_chain)
        for layer in lm.layers:                                             # the bench keeps wo_a a MultiLinear: give rows.Sink the width it reads off an Exl3Linear
            sink = layer.attn.wo_a
            if hasattr(sink, "inner") and not hasattr(sink.inner, "out_features"):
                sink.inner.out_features = sink.inner.weight.shape[-2]
    if "--moe" in a:
        print("real MoE installed on", install_moe(lm), "layers")
    res = {}
    for arm in arms:
        SG.ENABLED = arm in ("sg", "lc")
        if arm == "lc":
            from kyojin_mlx import launchcut as LC
            LC.ENABLED = True
        res[arm] = r = census(lm, steps, rows)
        print(f"[{arm}{'' if rows == 1 else f' verify R={rows}'}] {layers} layers (mean of {steps} positions): dispatches {r['dispatches']:.0f} ({r['dispatches'] / layers:.2f}/layer)  "
              f"graph nodes {r['nodes']:.0f} ({r['nodes'] / layers:.1f}/layer)  graph build {r['build_ms']:.1f} ms ({r['build_ms'] / layers:.2f} ms/layer)", flush=True)
        by = collections.defaultdict(list)
        for i, (nn_, dd) in enumerate(r["per_layer"][:layers]):
            by[G.RATIOS[i]].append((nn_, dd))
        for ratio, v in sorted(by.items()):
            print(f"   ratio {ratio}: {len(v)} layers  disp/layer {sum(d for _, d in v) / len(v):.1f}  nodes/layer {sum(x for x, _ in v) / len(v):.1f}")
        for k, st in enumerate(r["steps"]):
            print(f"   position {k}: dispatches {st['dispatches']} ({st['dispatches'] / layers:.2f}/layer)  per layer:", " ".join(str(d) for _, d in st["per_layer"]))
        if "--detail" in a:                                             # e.g. --detail 3,20: node types and custom-kernel names of those layers (position 0)
            st0 = r["steps"][0]
            for li in (int(v) for v in a[a.index("--detail") + 1].split(",")):
                pt = {k: v for k, v in st0["per_types"][li].items() if v}
                print(f"   layer {li} (ratio {G.RATIOS[li]}): " + ", ".join(f"{k[:44]} {v}" for k, v in sorted(pt.items(), key=lambda kv: -kv[1])))
                print(f"      kernels: " + " ".join(st0["names"][li]))
        top = sorted(r["by_type"].items(), key=lambda kv: -kv[1])[:14]
        print("   top nodes:", ", ".join(f"{k[:40]} {v:.0f}" for k, v in top))
    if "--json" in a:
        import json
        json.dump({k: {kk: vv for kk, vv in v.items() if kk not in ("steps",)} for k, v in res.items()}, open(a[a.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
