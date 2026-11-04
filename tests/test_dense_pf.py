"""dense_pf: W_eff fold equals the rotated EXL3 linear (fp16-matmul level) on a random trellis; install() respects the cap."""
import mlx.core as mx
import numpy as np
from kyojin_mlx import exl3_linear as EL, dense_pf as DP


def _lin(kt=16, nt=32, K=3, cb="mcg"):
    rng = np.random.default_rng(0)
    tr = mx.array(rng.integers(0, 65536, (kt, nt, 16 * K)).astype(np.uint16))
    suh = mx.array(rng.choice([-1.0, 1.0], kt * 16).astype(np.float16))
    svh = mx.array(rng.choice([-1.0, 1.0], nt * 16).astype(np.float16) * 0.05)
    return EL.Exl3Linear(tr, suh, svh, cb=cb)


def test_fold_matches_rotated_path():
    lin = _lin()
    x = (mx.random.normal((200, lin.in_features)) * 0.7).astype(mx.float16)
    ref = lin(x).astype(mx.float32)
    we = DP.fold(lin)
    y = (x @ we.T).astype(mx.float32)
    mx.eval(ref, y)
    assert we.shape == (lin.out_features, lin.in_features) and we.dtype == mx.float16
    assert float(mx.linalg.norm(y - ref) / mx.linalg.norm(ref)) < 3e-3


def test_call_uses_fold_only_for_prefill_rows():
    lin = _lin()
    x = (mx.random.normal((256, lin.in_features)) * 0.7).astype(mx.float16)
    ref = lin(x); xs = x[:2]; ref_s = lin(xs)
    lin._weff = DP.fold(lin)
    y, y_s = lin(x), lin(xs)
    mx.eval(ref, y, ref_s, y_s)
    assert bool(mx.all(y_s == ref_s))                      # decode rows keep their bits
    assert float(mx.linalg.norm((y - ref).astype(mx.float32)) / mx.linalg.norm(ref.astype(mx.float32))) < 3e-3
