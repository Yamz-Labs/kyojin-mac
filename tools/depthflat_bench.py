"""Synthetic bench of the dense reads of the fp4-packed cache (KYOJIN_DEPTHFLAT), no model. For every context L: a key layer (D=128, block 32; the owner of the
index keys, one append per step) and a pool layer (D=512, block 16; read through PackedView.dense() as a chained verify of R rows does once per reuse layer)
are filled with L rows, then `--steps` steps run: append R rows, read the dense layer, evaluate a consumer. ms/step with the stock unpack (flag OFF) and with the
dense mirror (flag ON) in the same process. Runs on any device (CPU or Metal); hold /tmp/ultra_heavy.lock when it runs on the Ultra GPU.
usage: PYTHONPATH=. python tools/depthflat_bench.py [--lens 1024,4096,32768,131072,1048576] [--steps 20] [--rows 1,4] [--cow]"""
import argparse, time

import mlx.core as mx
import numpy as np

from kyojin_mlx import depthflat as DF, fp4pack, fp4pool

ap = argparse.ArgumentParser()
ap.add_argument("--lens", default="1024,4096,32768,131072,1048576"); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--rows", default="1,4"); ap.add_argument("--cow", action="store_true", help="worst case: alias (cow) the layer before every append")
a = ap.parse_args()


def grid(n, D, block, e4, seed):
    x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D)).astype(np.float32)).astype(mx.float16)
    c, s = fp4pack.pack(x, block, e4)
    return fp4pack.unpack(c, s, block, e4, mx.float16)


def fill(slots, L, D, block, e4):
    for a0 in range(0, L, 65536):
        slots.append_rows(0, grid(min(65536, L - a0), D, block, e4, a0))
    mx.eval(slots.raw(0).codes)


def step_ms(slots, extra, view, steps, cow=False):
    e = slots.raw(0)
    t = time.perf_counter()
    for i in range(steps):
        held = slots.cow(slots.raw(0)) if cow else None                           # a chained verify / snapshot aliases the layer before the append: the write may copy the shared buffers
        slots.append_rows(0, extra[i])
        e = slots.raw(0)
        d = fp4pool.PackedView(e.codes, e.scales, e.n, e.dtype, (slots, e)).dense() if view else slots[0]
        mx.eval(d[:, -1].astype(mx.float32).sum())
    return (time.perf_counter() - t) / steps * 1e3


print(f"device {mx.default_device()}  steps {a.steps}")
for R in [int(x) for x in a.rows.split(",")]:
    print(f"-- R = {R} rows appended per step")
    print(f"{'L':>9} {'layer':>5} {'stock ms':>10} {'depthflat ms':>13} {'mirror MiB':>11}")
    for L in [int(x) for x in a.lens.split(",")]:
        for name, cls, D, block, e4, view in (("keys", fp4pool.KeySlots, 128, 32, False, False), ("pool", fp4pool.PoolSlots, 512, 16, True, True)):
            extra = [grid(R, D, block, e4, 9000 + i) for i in range(a.steps)]
            out = {}
            for flag in (False, True):
                DF.ENABLED = flag
                s = cls([None]); fill(s, L, D, block, e4)
                mx.eval(s[0] if not flag else s[0])                                  # warm: the first dense read (the mirror build) is not a step
                out[flag] = step_ms(s, extra, view, a.steps, a.cow)
            mib = L * D * 2 / 2 ** 20
            print(f"{L:>9} {name:>5} {out[False]:>10.2f} {out[True]:>13.2f} {mib:>11.0f}", flush=True)
DF.ENABLED = False
