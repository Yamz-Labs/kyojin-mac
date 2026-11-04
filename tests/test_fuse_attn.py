"""DS4.1 attention decode cut (kyojin_mlx.fuse_attn): kv_ring is value-identical (signed zeros aside) to the stock rope + fp8 round trip + ring write; attend matches the stock
sparse attention + inverse rope to the error of the stock composite (against an fp64 reference, not bit equality: the stock fp32 matmuls are ~1e-3 accurate
on the M5 tensor units, the kernel sums in fp32 FMA). Needs a Mac (Metal) and mlx-vlm 0.7.10 on the path."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
from kyojin_mlx import fuse_attn as FA                                     # noqa: E402
from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE, _sparse_pooled_attention  # noqa: E402
from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0 as fq   # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

H, D, RD, W, G = 64, 512, 64, 128, 8
YARN = {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536}


def ropes():
    return {"compressed": DeepseekV4RoPE(RD, 160000, YARN, 1048576), "plain": DeepseekV4RoPE(RD, 10000, None, 1048576)}


def bits(a):
    """Values as float32 for ==: the only bit-level difference to the stock chain is the sign of a zero (the stock rope turns -0 into +0 in a nope pair
    when its neighbour is negative, the kernel leaves nope dims alone); no later op sees the sign of a zero."""
    return np.array(a.astype(mx.float32))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("kind", ["compressed", "plain"])
def test_kv_ring_bit_identical(dtype, kind):
    rope = ropes()[kind]
    rng = np.random.default_rng(3)
    for trial in range(60):
        pos = int(rng.choice([1, 2, 5, 127, 128, 129, 255, 300, 4095, 65535, 100000, int(rng.integers(1, 200000))]))
        scale = float(rng.choice([1e-3, 0.1, 1.0, 30.0, 400.0]))
        kvn = mx.array((rng.standard_normal((1, 1, D)) * scale).astype(np.float32)).astype(dtype)
        if trial % 7 == 0:
            kvn = kvn * mx.array(rng.choice([0.0, 1.0, 1.0, 1.0], size=(1, 1, D)).astype(np.float32)).astype(dtype)
        ring = mx.array((rng.standard_normal((1, W, D)) * 2).astype(np.float32)).astype(dtype)
        ref_row = fq(rope(kvn[:, None], pos).reshape(1, 1, D))
        stock = ring[...]
        stock[:1, mx.array([pos % W])] = ref_row
        rt, _ = FA.rope_tables(rope, pos)
        got = FA.kv_ring(kvn, ring, rt, FA.pos_array(pos), RD)
        assert np.array_equal(bits(stock), bits(got)), (trial, pos, scale)


def test_rope_tables_match_stock_rope():
    rope = ropes()["compressed"]
    rng = np.random.default_rng(1)
    for pos in (1, 7, 127, 5000, 70000, 123456):
        x = mx.array(rng.standard_normal((1, 1, 1, RD)).astype(np.float32))
        rt, it = FA.rope_tables(rope, pos)
        c, s = np.array(rt).reshape(-1, 2).T
        xr = np.array(x).reshape(-1, 2)
        fwd = np.stack([xr[:, 0] * c - xr[:, 1] * s, xr[:, 0] * s + xr[:, 1] * c], 1).reshape(-1)
        assert np.allclose(fwd, np.array(rope(x, pos)).reshape(-1), atol=2e-6)


def stock_attention(q, ring_new, pool, idxs, sinks, rope, pos, scale):
    """q [1, 1, H * D], stock chain: rope(q), window = the keys of positions base .. pos, `_sparse_pooled_attention`, inverse rope -> [1, H, 1, D]."""
    qh = rope(q.reshape(1, 1, H, D).transpose(0, 2, 1, 3), pos)
    base = max(0, pos - W + 1)
    window = mx.take(ring_new, mx.arange(base, pos + 1) % W, axis=1)
    mask = mx.ones((1, 1, 1, window.shape[1]), dtype=mx.bool_)
    out = _sparse_pooled_attention(qh.astype(mx.float32), window[:, None], pool, idxs, mask, (idxs != -1)[:, None], scale, sinks).astype(q.dtype)
    return rope(out, pos, inverse=True)


def fp64_attention(q, ring_new, pool, idxs, sinks, rope, pos, scale):
    qh = np.array(rope(q.reshape(1, 1, H, D).transpose(0, 2, 1, 3), pos).astype(mx.float32)).astype(np.float64)[0, :, 0]           # [H, D]
    base = max(0, pos - W + 1)
    win = np.array(mx.take(ring_new, mx.arange(base, pos + 1) % W, axis=1).astype(mx.float32)).astype(np.float64)[0]               # [nw, D]
    ids = np.array(idxs).reshape(-1)
    pl = np.array(pool.astype(mx.float32)).astype(np.float64)[0]
    keys = np.concatenate([win] + ([pl[ids[ids >= 0]]] if (ids >= 0).any() else []), 0)
    s = qh @ keys.T * scale                                                                                                          # [H, nk]
    sk = np.array(sinks).astype(np.float64)[:, None]
    m = np.maximum(s.max(-1, keepdims=True), sk)
    e = np.exp(s - m)
    den = e.sum(-1, keepdims=True) + np.exp(sk - m)
    return (e / den) @ keys                                                                                                          # [H, D] before the inverse rope


def inv_rope64(o, rope, pos):
    rt, it = FA.rope_tables(rope, pos)
    c, s = np.array(it).astype(np.float64).reshape(-1, 2).T
    o = o.copy()
    tail = o[:, D - RD:].reshape(H, -1, 2)
    x1, x2 = tail[..., 0].copy(), tail[..., 1].copy()
    tail[..., 0] = x1 * c - x2 * s
    tail[..., 1] = x1 * s + x2 * c
    o[:, D - RD:] = tail.reshape(H, RD)
    return o


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("scenario", ["young", "full_window", "few_pooled", "many_pooled", "ragged_idx"])
def test_attend_matches_stock_within_stock_error(dtype, scenario):
    rope = ropes()["compressed"]
    rng = np.random.default_rng(sum(map(ord, scenario)))
    scale = D ** -0.5
    pos = {"young": 20, "full_window": 5000, "few_pooled": 700, "many_pooled": 9000, "ragged_idx": 3000}[scenario]
    n_pool = {"young": 10, "full_window": 2500, "few_pooled": 3, "many_pooled": 4500, "ragged_idx": 1500}[scenario]
    K = {"young": 10, "full_window": 512, "few_pooled": 3, "many_pooled": 512, "ragged_idx": 512}[scenario]
    q = mx.array((rng.standard_normal((1, 1, H * D)) * 1.5).astype(np.float32)).astype(dtype)
    ring = mx.array((rng.standard_normal((1, W, D)) * 1.5).astype(np.float32)).astype(dtype)
    kvn = mx.array((rng.standard_normal((1, 1, D)) * 1.5).astype(np.float32)).astype(dtype)
    pool = mx.array((rng.standard_normal((1, n_pool, D)) * 1.5).astype(np.float32)).astype(dtype)
    ids = np.sort(rng.choice(n_pool, size=K, replace=False)).astype(np.int32)
    if scenario == "ragged_idx":
        ids[rng.random(K) < 0.4] = -1
    idxs = mx.array(ids).reshape(1, 1, K)
    sinks = mx.array((rng.standard_normal(H) * 2).astype(np.float32))
    rt, it = FA.rope_tables(rope, pos)
    new_ring = FA.kv_ring(kvn, ring, rt, FA.pos_array(pos), RD)
    got = FA.attend(q, new_ring, pool, idxs, sinks, rt, it, FA.pos_array(pos), H, RD, scale)
    mx.eval(got)
    ref64 = inv_rope64(fp64_attention(q, new_ring, pool, idxs, sinks, rope, pos, scale), rope, pos)
    stock = stock_attention(q, new_ring, pool, idxs, sinks, rope, pos, scale)
    g, s = np.array(got.astype(mx.float32))[0, :, 0], np.array(stock.astype(mx.float32))[0, :, 0]
    eg, es = rel(g, ref64), rel(s, ref64)
    floor = 2.0 ** -8 if dtype == mx.bfloat16 else 2.0 ** -11
    assert eg <= max(1.5 * es, 0.4 * floor), (eg, es)                         # same level as the stock composite (both are rounded to T)
    # the stock fp32 matmuls run on the M5 tensor units at ~1e-3 relative error (measured: tools/scratch12_mm.py), so stock and kernel differ by about the stock error
    assert rel(g, s) <= 2.0 * (eg + es) + 1e-6, (rel(g, s), eg, es)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_attend_split_pv_matches_sequential(dtype):
    """The key-split P @ V (kernel default) agrees with the sequential loop to the rounding of the output dtype (fp32 sum order only)."""
    rng = np.random.default_rng(5)
    H, D, W, RD, n, K = 64, 512, 128, 64, 4000, 300
    pool = mx.array(rng.standard_normal((1, n, D)).astype(np.float32)).astype(dtype)
    ring = mx.array(rng.standard_normal((1, W, D)).astype(np.float32)).astype(dtype)
    q = mx.array(rng.standard_normal((1, 1, H * D)).astype(np.float32)).astype(dtype)
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    ang = rng.uniform(0, 6, RD // 2)
    rt = mx.array(np.stack([np.cos(ang), np.sin(ang)], -1).reshape(-1).astype(np.float32))
    it = mx.array(np.stack([np.cos(ang), -np.sin(ang)], -1).reshape(-1).astype(np.float32))
    idx = mx.array(np.sort(rng.choice(n, K, replace=False)).astype(np.int32))
    outs = []
    old = FA.SPLIT_PV
    try:
        for flag in (False, True):
            FA.SPLIT_PV = flag
            outs.append(np.array(FA.attend(q, ring, pool, idx, sink, rt, it, FA.pos_array(9000), H, RD, 0.0442).astype(mx.float32)))
    finally:
        FA.SPLIT_PV = old
    a, b = outs
    tol = 2.0 ** -7 if dtype == mx.bfloat16 else 2.0 ** -10
    assert np.abs(a - b).max() <= tol * np.abs(a).max() * 2 and (a == b).mean() > 0.99


# ------------------------------------------------------------------------------------- rope_pos_hint: the hint on the verify rows (no `.item()` mid-build)
YARN_T = (16.0, 32.0, 1.0, 65536.0)


def _stock_rope_at_positions(L, x, positions):
    return L._apply_rope_at_positions(x, positions, RD, 160000, YARN_T)


@pytest.mark.parametrize("start,s,ratio", [(0, 2, 128), (0, 5, 128), (120, 6, 128), (4096, 5, 128),
                                           (32768, 7, 128), (300, 24, 128), (1, 2, 32), (65000, 64, 128)])
def test_rope_pos_hint_verify_rows_exact(start, s, ratio):
    """The hinted table is the stock table: the stock `_apply_rope_at_positions` sizes its cos/sin from `int(mx.max(positions).item())`, the hint from the python
    ints of `(start_pos // ratio + seqlen - 1) * ratio`; only the rows `cos[positions]` selects are used and a row's value depends on its index alone, so the
    latents are bit-identical (bf16 and fp32 output)."""
    import importlib
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    rng = np.random.default_rng(11)
    for dtype in (mx.bfloat16, mx.float32):
        for n_lat in (1, max(1, s // 2), s):
            positions = (start // ratio + mx.arange(n_lat)) * ratio
            x = mx.array(rng.standard_normal((1, n_lat, RD)).astype(np.float32)).astype(dtype)
            ref = _stock_rope_at_positions(L, x, positions)
            wrapped = FA._impl_rope_positions(L._apply_rope_at_positions)
            FA._hint[0] = (start // ratio + s - 1) * ratio
            try:
                got = wrapped(x, positions, RD, 160000, YARN_T)
            finally:
                FA._hint[0] = None
            assert np.array_equal(bits(ref), bits(got)), (start, s, n_lat, str(dtype))


def test_rope_pos_hint_only_on_verify_rows():
    """The hint is set for 2..HINT_ROWS_MAX rows with the lever on (never for a prompt chunk), and for 1 row it is the solo-decode hint either way."""
    seen = []

    def orig(self, x, qr, start_pos, cache):
        seen.append(FA._hint[0])
        return None

    class Stub:
        compress_ratio = 128

    fn = FA._impl_compress_part(orig)
    old = FA.ROPE_POS_HINT
    try:
        for lever, s, start, want in ((False, 1, 300, (300 // 128) * 128), (True, 1, 300, (300 // 128) * 128),
                                      (False, 5, 300, None), (True, 5, 300, (300 // 128 + 4) * 128),
                                      (True, 64, 1000, (1000 // 128 + 63) * 128), (True, FA.HINT_ROWS_MAX + 1, 1000, None),
                                      (True, 8192, 0, None)):
            FA.ROPE_POS_HINT = lever
            seen.clear()
            fn(Stub(), mx.zeros((1, s, D)), None, start, None)
            assert seen == [want], (lever, s, start, seen, want)
            assert FA._hint[0] is None
    finally:
        FA.ROPE_POS_HINT = old
