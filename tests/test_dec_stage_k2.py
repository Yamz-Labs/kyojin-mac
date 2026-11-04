"""dec_stage k1-k2 levers (stagek2.qkv_stage, decstage QKV / IDXQ): CPU tests (run anywhere mlx imports) of the kernel source, the references against a
float64 oracle and the flag wiring; Metal tests (skipped without a GPU, for the lane) of the kernel against the kv chain (bitwise) and the q_norm reference."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import decstage as DS, dsv4_core as DC, flags as FL, stagek2 as SK2  # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def kv_inputs(R=3, D=512, W=8, RD=64, seed=0):
    rng = np.random.default_rng(seed)
    kvr = (rng.standard_normal((R, D)) * 0.8).astype(np.float16)
    w = rng.uniform(0.5, 1.5, D).astype(np.float16)
    ring = (rng.standard_normal((R, W, D)) * 0.5).astype(np.float16)
    sess = np.arange(R)[::-1].astype(np.uint32).copy()
    wm = -np.ones(R * W, np.int32)
    for r in range(R):
        wm[r * W + (r * 3 + 1) % W] = r                      # row r writes its own slot
    if R > 1:
        wm[1 * W + 0] = 0                                    # and a predecessor's entry shows in the next row's snapshot (verify layout)
    rt = rng.uniform(-1, 1, (R, RD)).astype(np.float32)
    return kvr, w, ring, sess, wm, rt


def test_source_is_the_q_role_plus_the_kv_chain_text_shifted():
    src = SK2.SOURCE
    assert src.count("threadgroup_position_in_grid.x - uint(QT)") == 1
    assert DC._KV_CHAIN.replace(SK2._SHIFT, "const uint tg = threadgroup_position_in_grid.x - uint(QT);") in src
    assert src.count("{") == src.count("}") and src.count("(") == src.count(")")
    # the q role returns before the kv text and keeps its own names (no clash with the kv text's lid / sg / sl / inv / ls)
    assert src.index("return;") < src.index("const uint b = tg / uint(W);")
    head = src[: src.index("const uint b = tg / uint(W);")]
    for name in ("lid", "inv", "ls"):
        assert f"const uint {name}" not in head.replace("qlid", "") or name == "inv"


@pytest.mark.parametrize("nq", [1280, 1536, 512, 4096])
def test_q_reference_is_no_worse_than_one_fp16_rounding_against_float64(nq):
    rng = np.random.default_rng(nq)
    qa = (rng.standard_normal((4, nq)) * 1.7).astype(np.float16)
    w = rng.uniform(0.5, 1.5, nq).astype(np.float16)
    got = np.array(SK2.qnorm_ref(mx.array(qa), mx.array(w), 1e-6)).astype(np.float64)
    x = qa.astype(np.float64)
    want = w.astype(np.float64) * x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    # w * T(x * inv): one rounding of x * inv (rel 2^-11) then the product rounded to fp16 (rel 2^-11)
    assert np.all(np.abs(got - want) <= 2.0 ** -10 * np.abs(want) + 1e-6)


def test_kv_reference_keeps_the_base_ring_and_writes_the_quantized_row():
    kvr, w, ring, sess, wm, rt = kv_inputs()
    R, D = kvr.shape
    W = ring.shape[1]
    out = np.array(SK2.kv_ref(mx.array(kvr), mx.array(w), mx.array(ring), mx.array(sess), mx.array(wm), mx.array(rt), 1e-6)).astype(np.float32)
    assert out.shape == (R, W, D)
    wm2 = wm.reshape(R, W)
    for r in range(R):
        for s in range(W):
            if wm2[r, s] < 0:
                np.testing.assert_array_equal(out[r, s], ring[sess[r], s].astype(np.float32))
    x = kvr.astype(np.float64)
    nb = w.astype(np.float64) * x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    for r in range(R):
        for s in range(W):
            e = wm2[r, s]
            if e >= 0:
                row = nb[e].copy()
                tail = row[D - 64:].reshape(32, 2)
                c, sn = rt[e].astype(np.float64).reshape(32, 2).T
                row[D - 64:] = np.stack([tail[:, 0] * c - tail[:, 1] * sn, tail[:, 0] * sn + tail[:, 1] * c], -1).reshape(64)
                # e4m3 keeps 3 mantissa bits: a block-scaled round trip is within 2^-4 of the block max
                blk = np.abs(row).reshape(-1, 32).max(-1).repeat(32)
                assert np.all(np.abs(out[r, s] - row) <= blk * 2.0 ** -4 + 2e-3)


def test_supported_is_the_kv_chain_geometry():
    qa = mx.zeros((2, 1280), dtype=mx.float16)
    kv = mx.zeros((2, 512), dtype=mx.float16)
    ring = mx.zeros((2, 8, 512), dtype=mx.float16)
    assert SK2.supported(qa, kv, ring)
    assert not SK2.supported(qa.astype(mx.bfloat16), kv, ring)
    assert not SK2.supported(mx.zeros((2, 1282), dtype=mx.float16), kv, ring)
    assert not SK2.supported(qa, mx.zeros((2, 256), dtype=mx.float16), mx.zeros((2, 8, 256), dtype=mx.float16))
    assert not SK2.supported(qa, mx.zeros((3, 512), dtype=mx.float16), ring)


def test_levers_are_off_by_default_and_the_module_literal_matches_the_table():
    for name, attr in (("dec_qkv", "QKV"), ("dec_idxq", "IDXQ")):
        assert FL.FLAGS[name].default == "0"
        assert getattr(DS, attr) is (FL.FLAGS[name].default != "0")
    import inspect
    src = inspect.getsource(DS)
    assert 'os.environ.get("KYOJIN_DEC_QKV", "0")' in src and 'os.environ.get("KYOJIN_DEC_IDXQ", "0")' in src


def test_flag_switch_reaches_the_module(monkeypatch):
    old = (DS.QKV, DS.IDXQ)
    try:
        FL.set_many({"dec_qkv": 1, "dec_idxq": 1})
        assert DS.QKV and DS.IDXQ
        FL.set_many({"dec_qkv": 0, "dec_idxq": 0})
        assert not DS.QKV and not DS.IDXQ
    finally:
        DS.QKV, DS.IDXQ = old


def test_without_metal_the_lever_reports_unusable(monkeypatch):
    monkeypatch.setitem(SK2.STATE, "checked", None)
    if not mx.metal.is_available():
        assert SK2.usable() is False
    monkeypatch.setitem(SK2.STATE, "checked", None)


def test_qkv_args_is_none_while_the_lever_is_off(monkeypatch):
    monkeypatch.setattr(DS, "QKV", False)
    qa = mx.zeros((1, 1280), dtype=mx.float16)
    assert DS._qkv_args(None, qa, mx.zeros((1, 512), dtype=mx.float16), None) is None


@needs_metal
@pytest.mark.parametrize("R,NQ", [(1, 1280), (3, 1536), (4, 1280)])
def test_kernel_matches_the_kv_chain_bitwise_and_q_norm_reference(R, NQ):
    kvr, w, ring, sess, wm, rt = kv_inputs(R)
    rng = np.random.default_rng(R)
    qa = mx.array((rng.standard_normal((R, NQ)) * 1.2).astype(np.float16))
    wq = mx.array(rng.uniform(0.5, 1.5, NQ).astype(np.float16))
    args = [mx.array(a) for a in (kvr, w, ring, sess, wm, rt)]
    qr, win = SK2.qkv_stage(qa, wq, 1e-6, *args, 1e-6)
    want = DC.kv_chain(*args[:5], args[5], 1e-6)
    ref = SK2.qnorm_ref(qa, wq, 1e-6)
    mx.eval(qr, win, want, ref)
    assert bool(mx.array_equal(win, want))
    assert float(mx.max(mx.abs(qr.astype(mx.float32) - ref.astype(mx.float32)))) < 2e-2


@needs_metal
def test_first_use_self_check_passes():
    SK2.STATE["checked"] = None
    assert SK2.usable() is True


# ------------------------------------------------------------------------------------------------ dec_cproj (xgemv)
def test_xgemv_source_is_balanced_and_reads_the_stored_dtype():
    src = SK2._XGEMV
    assert src.count("{") == src.count("}") and "device WT*" in src and "simd_sum" in src


def test_xgemv_reference_is_the_f32_matmul():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((3, 256))).astype(np.float16)
    w0 = (rng.standard_normal((20, 256)) * 0.05).astype(np.float32)
    w1 = (rng.standard_normal((13, 256)) * 0.05).astype(np.float16)
    r0, r1 = SK2.xgemv_ref(mx.array(x), mx.array(w0), mx.array(w1.astype(np.float32)))
    np.testing.assert_allclose(np.array(r0), x.astype(np.float64) @ w0.astype(np.float64).T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(r1), x.astype(np.float64) @ w1.astype(np.float64).T, rtol=1e-5, atol=1e-5)


def test_xgemv_supported_gates():
    x = mx.zeros((2, 256), dtype=mx.float16)
    w = mx.zeros((8, 256), dtype=mx.float32)
    assert SK2.xgemv_supported(x, w, w)
    assert not SK2.xgemv_supported(x, w, w.astype(mx.float16))                    # one stored dtype
    assert not SK2.xgemv_supported(x, mx.zeros((8, 192), dtype=mx.float32), mx.zeros((8, 192), dtype=mx.float32))
    assert not SK2.xgemv_supported(x, mx.zeros((8, 256), dtype=mx.uint32), mx.zeros((8, 256), dtype=mx.uint32))


def test_cproj_is_off_by_default_and_follows_the_table():
    assert FL.FLAGS["dec_cproj"].default == "0" and DC.CPROJ is False
    import inspect
    assert 'os.environ.get("KYOJIN_DEC_CPROJ", "0")' in inspect.getsource(DC)
    old = DC.CPROJ
    try:
        FL.set_many({"dec_cproj": 1})
        assert DC.CPROJ
    finally:
        FL.set_many({"dec_cproj": 0})
        DC.CPROJ = old


@needs_metal
@pytest.mark.parametrize("R", [1, 2, 5, 6, 24])
@pytest.mark.parametrize("wdt", ["float16", "float32", "bfloat16"])
def test_xgemv_matches_the_f32_matmul(R, wdt):
    rng = np.random.default_rng(R)
    dt = getattr(mx, wdt)
    x = mx.array((rng.standard_normal((R, 5120)) * 0.7).astype(np.float16))
    w0 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
    w1 = mx.array((rng.standard_normal((512, 5120)) * 0.02).astype(np.float32)).astype(dt)
    y0, y1 = SK2.xgemv(x, w0, w1)
    r0, r1 = SK2.xgemv_ref(x, w0, w1)
    mx.eval(y0, y1, r0, r1)
    # judged against float64, not against `xgemv_ref`: that fp32 matmul keeps 10 bits of each operand on the tensor units, so a float32 weight is up to 1.7e-3 off
    # exact (tests/test_cmd5_kernel_refs.py); the kernel's order is within 1.1e-6 (bound 1e-5). The reference's error is printed, not asserted.
    xe = np.array(x.astype(mx.float32)).astype(np.float64)
    e0 = np.abs(np.array(y0).astype(np.float64) - xe @ np.array(w0.astype(mx.float32)).astype(np.float64).T).max()
    e1 = np.abs(np.array(y1).astype(np.float64) - xe @ np.array(w1.astype(mx.float32)).astype(np.float64).T).max()
    print(f"\n[R={R} {wdt}] |kernel - exact| {max(e0, e1):.2e}   |matmul ref - kernel| {float(mx.max(mx.abs(y0 - r0))):.2e}")
    assert e0 < 1e-5 and e1 < 1e-5
