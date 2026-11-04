"""roundfuse_gate.py: micro-gate of the roundfuse kernels against the stock ops (warm-up, interleaved arms, min of runs, buffers < 1 GiB).
A: residual add + RMSNorm (H = 5120) at 1 / 6 / 16 / 48 rows, 64 chained pairs per timing. B: gated-delta mixer of the 27B shape (16 / 48 heads of 128) without
its projections (front end + recurrence + gated norm), fused rows path vs the stock ops of the capture call, 16 layers chained per timing.
Only the evaluation is timed (the graph is built first), so the numbers are GPU time plus encode, not Python overhead.
Run from the repo root with PYTHONPATH=$PWD."""
import time

import mlx.core as mx
import mlx.nn as nn

from kyojin_mlx import decfuse as DF

REPS = 9


def best(fn):
    """fn builds a lazy graph and returns its outputs; the evaluation alone is timed."""
    mx.eval(fn())
    t = []
    for _ in range(REPS):
        out = fn()
        t0 = time.perf_counter()
        mx.eval(out)
        t.append((time.perf_counter() - t0) * 1e3)
    return min(t)


def interleaved(stock, fused):
    res = {}
    for arm in ("stock", "fused", "stock", "fused", "stock", "fused"):
        res[arm] = min(res.get(arm, 1e9), best(stock if arm == "stock" else fused))
    return res["stock"], res["fused"]


def gate_addnorm():
    H, N = 5120, 64
    n = nn.RMSNorm(H)
    n.weight = (1 + 0.1 * mx.random.normal((H,))).astype(mx.float16)
    for R in (1, 6, 16, 48):
        x, r = mx.random.normal((1, R, H)).astype(mx.float16), mx.random.normal((1, R, H)).astype(mx.float16)

        def stock():
            h = x
            for _ in range(N):
                h = h + r
                h = h + n(h) * 1e-3
            return h

        def fused():
            h = x
            for _ in range(N):
                h, y = DF.add_rms(h, r, n)
                h = h + y * 1e-3
            return h

        a, b = interleaved(stock, fused)
        print(f"add+rms H={H} R={R}: stock {a:.3f} ms, fused {b:.3f} ms per {N} pairs (4 ops vs 3 per pair incl. the shared mul + add)")


def gate_gdn():
    from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_update, normalize_qk

    from kyojin_mlx import gdn_pf
    hk, hv = 16, 48
    C = 2 * hk * 128 + hv * 128
    conv_w = (mx.random.normal((C, 4, 1)) * 0.3).astype(mx.float16)
    A = mx.random.normal((hv,)).astype(mx.float32) * 0.5
    dtb = mx.random.normal((hv,)).astype(mx.float16)
    nw = (1 + 0.1 * mx.random.normal((128,))).astype(mx.float16)
    for R in (1, 6, 16, 48):
        qkv = (mx.random.normal((1, R, C)) * 0.5).astype(mx.float16)
        z = mx.random.normal((1, R, hv, 128)).astype(mx.float16)
        a, b = mx.random.normal((1, R, hv)).astype(mx.float16), mx.random.normal((1, R, hv)).astype(mx.float16)
        cs0 = mx.random.normal((1, 3, C)).astype(mx.float16)
        st0 = (mx.random.normal((1, hv, 128, 128)) * 0.1).astype(mx.float32)

        def stock():
            st, cs, ys = st0, cs0, []
            for _ in range(16):
                ci = mx.concatenate([cs, qkv], axis=1)
                cs = mx.contiguous(ci[:, -3:, :])
                co = nn.silu(mx.conv1d(ci, conv_w, groups=C))
                q, k, v = [t.reshape(1, R, h, 128) for t, h in zip(mx.split(co, [hk * 128, 2 * hk * 128], -1), [hk, hk, hv])]
                q, k = normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)
                o, st = gated_delta_update(q, k, v, a, b, A, dtb, st, None, use_kernel=True)
                ys.append(nn.silu(z.astype(mx.float32)).astype(mx.float16) * mx.fast.rms_norm(o, nw, 1e-6))
            return st, cs, ys

        def fused():
            st, cs, ys = st0, cs0, []
            for _ in range(16):
                q, k, v, cs, g, beta = gdn_pf.pre(qkv, cs, conv_w, a, b, A, dtb, hk, hv)
                o, st = gated_delta_kernel(q, k, v, g, beta, st, None)
                ys.append(gdn_pf.gated_norm(o.astype(mx.float16), z, nw, 1e-6))
            return st, cs, ys

        a_, b_ = interleaved(stock, fused)
        print(f"gdn front end + core + norm R={R}: stock {a_:.3f} ms, fused {b_:.3f} ms per 16 layers")


if __name__ == "__main__":
    gate_addnorm()
    gate_gdn()
