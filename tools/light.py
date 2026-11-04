"""GPU-light mode for CPU-bound measurements: every custom Metal kernel launches 1/F of its threadgroups along x, so the GPU work shrinks about F times
while launches, arguments and the python / encode path stay identical. Outputs are garbage: use only for timing the CPU critical path (what bounds a
GPU that is F times wider). usage: import light before kyojin_mlx (tools do it when M10_LIGHT=F is set)."""
import os
import mlx.core as mx

F = int(os.environ.get("M10_LIGHT", "1"))
if F > 1 and not getattr(mx.fast, "_m10_light", False):
    _orig = mx.fast.metal_kernel

    def metal_kernel(*a, **k):
        kern = _orig(*a, **k)

        def call(*ca, **ck):
            g, tg = ck.get("grid"), ck.get("threadgroup")
            if g is not None and tg is not None and tg[0] > 0 and g[0] // tg[0] >= F:
                nx = max(tg[0], (g[0] // F) // tg[0] * tg[0])
                ck["grid"] = (nx, g[1], g[2])
            return kern(*ca, **ck)
        return call

    mx.fast.metal_kernel = metal_kernel
    mx.fast._m10_light = True
