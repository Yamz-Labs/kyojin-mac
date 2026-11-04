"""KYOJIN_MOEROOF: cheaper integer decodes (dec_mul1r1, dec_mul1r2, dec_mul1fdr) and next-tile word prefetch of the per-pair gather GEMV.
Everything above the `metal` tests runs on any machine (numpy only, through kyojin_mlx.emulate): the new decodes equal the stock ones bit for bit over
all 65536 trellis states, the emulated gather GEMV (R1 and R4 rows, K = 2 and 3) is bit-identical, and the generated prefetch loop loads every tile
once, in order, in bounds. The `metal` tests need a Metal GPU (skipped cleanly otherwise, or with KYOJIN_TEST_NO_METAL=1)."""
import os
import re

import numpy as np
import pytest

from kyojin_mlx import emulate as E, metal_src as S, ref_numpy as R

ALL = np.arange(65536, dtype=np.uint32)


def bits(a):
    return np.asarray(a).view(np.uint16)


# ----------------------------------------------------------------------------------------------- decodes (exhaustive)

def test_emulated_stock_decode_matches_reference():
    assert np.array_equal(bits(E.mul1_values(ALL)), bits(R.codebook_values(ALL.astype(np.uint16), "mul1")))


@pytest.mark.parametrize("level", [1, 2])
def test_mul1r_equals_stock_all_states(level):
    assert np.array_equal(bits(E.mul1r_values(ALL, level)), bits(E.mul1_values(ALL)))


def test_fdr_byte_pairs_equal_stock_all_states():
    """dec_mul1fdr returns the same two lane sums (the bits of the half2) as dec_mul1fd."""
    assert np.array_equal(E.byte_pairs(ALL, roof=True), E.byte_pairs(ALL, roof=False))


def test_lane_sum_never_carries():
    t = E.byte_pairs(ALL)
    assert int((t & np.uint32(0xFFFF)).max()) <= 510 and int((t >> np.uint32(16)).max()) <= 510
    u = t * np.uint32(0x00010001) + np.uint32(0x64000000)
    assert int((u & np.uint32(0xFFFF)).max()) <= 510          # the lower half keeps lo: nothing leaks upwards
    assert int((u >> np.uint32(16)).max()) == 0x6400 + 1020 or int((u >> np.uint32(16)).max()) < 0x6800


def test_metal_text_matches_the_port():
    h = S.HEADER_MOE_ROOF
    for line in ("uint t = h - 255u * ((h >> 8) & 0x00ff00ffu);",
                 "as_type<half2>(t * 0x00010001u + 0x64000000u).y;",
                 "return as_type<half2>(h - 255u * ((h >> 8) & 0x00ff00ffu));",
                 "uint t = (h & 0x00ff00ffu) + ((h >> 8) & 0x00ff00ffu);"):
        assert line in h, line
    assert h.count("uint h = s * 0x83DCD12Du;") == 3
    assert "dec_mul1r" not in S.HEADER_MOE and "dec_mul1fdr" not in S.HEADER_MOE      # lever off: the header is the stock one


def test_affine_helpers_treat_fdr_like_fd():
    assert S.af_fold("mul1fdr") and S.af_consts("mul1fdr") == S.af_consts("mul1fd")
    assert not S.af_fold("mul1r1") and not S.af_fold("mul1r2")


# ----------------------------------------------------------------------------------------------- emulated GEMV (R1, R4)

@pytest.mark.parametrize("K", [2, 3])
def test_lane_values_bit_identical(K):
    rng = np.random.default_rng(K)
    words = rng.integers(0, 2 ** 32, 8 * K, dtype=np.uint32)
    for g in range(8):
        ref = E.lane_values(K, words, g, "mul1")
        for cb in ("mul1r1", "mul1r2"):
            got = E.lane_values(K, words, g, cb)
            assert np.array_equal(bits(np.array(got, np.float16)), bits(np.array(ref, np.float16))), (cb, g)


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("K", [2, 3])
def test_emulated_gemv_bit_identical(K, rows):
    kt, nt, tgk = 8, 8, 2
    trellis, suh, _ = R.random_linear(kt * 16, nt * 16, K, seed=7 + K)
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, kt * 16)).astype(np.float32)
    xh = (R.fwht128(x * suh.astype(np.float32), axis=1) * R.INV_SQRT_HAD).astype(np.float32)
    words = R.trellis_words(trellis)
    ref = E.gemv_partial(words, K, kt, nt, xh, tgk, "mul1")
    for cb in ("mul1r1", "mul1r2"):
        assert np.array_equal(E.gemv_partial(words, K, kt, nt, xh, tgk, cb).view(np.uint32), ref.view(np.uint32)), cb


# ----------------------------------------------------------------------------------------------- generated source

def _run_loads(stmts, flat, tp, g, wprev, log):
    """Execute the generated `n_x = <expr>;` load statements against a flat word array at tile base `tp`; returns {name: value}."""
    def vec(n, idx):
        log.extend(range(tp + n * idx, tp + n * idx + n))
        return tuple(int(flat[tp + n * idx + c]) for c in range(n))

    def sc(idx):
        log.append(tp + idx)
        return int(flat[tp + idx])
    out = {}
    for name, expr in re.findall(r"(n_\w+) = ([^;]+);", stmts):
        e = re.sub(r"\(\(const device uint(\d)\*\) tp\)\[(.*)\]", r"vec(\1, \2)", expr)
        e = re.sub(r"\btp\[(.*)\]", r"sc(\1)", e)
        out[name] = eval(e, {"vec": vec, "sc": sc, "g": g, "wprev": wprev})
    return out


@pytest.mark.parametrize("K", [2, 3])
@pytest.mark.parametrize("myn", [0, 1, 2, 5])
def test_prefetch_loop_matches_plain_loop(K, myn):
    """Python run of the generated prefetch statements (pre-loop load, in-loop load of the next tile): each iteration sees the words of its own tile,
    every tile is read once, nothing is read past the last tile, and nothing at all when myn == 0."""
    src = S.gather_gemv_source((K,), 1, "mul1r2", True)
    pre = re.search(r"if \(myn > 0u\) \{ (.*?) \}\n", src).group(1)
    nxt = re.search(r"if \(i \+ 1u < myn\) \{ tp \+= tstride; (.*?) \}", src).group(1)
    TW, NT, g = 8 * K, 6, 5
    wprev = (K * g + TW - 1) % TW
    tstride = NT * TW
    flat = (np.arange(8 * NT * TW, dtype=np.uint64) * 2654435761 % 2 ** 32).astype(np.uint32)
    tp0 = 1 * TW                                                 # ks0 = 0, nt = 1
    log, seen = [], []
    tp = tp0
    cur = _run_loads(pre, flat, tp, g, wprev, log) if myn > 0 else {}
    for i in range(myn):
        seen.append(cur)
        if i + 1 < myn:
            tp += tstride
            cur = _run_loads(nxt, flat, tp, g, wprev, log)
    for i, got in enumerate(seen):                               # the plain loop's loads of tile i
        want = _run_loads(re.sub(r"\bn_", "n_", pre), flat, tp0 + i * tstride, g, wprev, [])
        assert got == want
    per_tile = len(log) // myn if myn else 0
    assert len(log) == myn * per_tile and (myn == 0 or len(set(log)) == len(log))
    assert max(log, default=0) < tp0 + myn * tstride and (myn == 0 or max(log) >= tp0 + (myn - 1) * tstride)


def test_prefetch_source_keeps_the_decode_lines():
    plain = S.gather_gemv_source((2, 3), 2, "mul1r2", False)
    pf = S.gather_gemv_source((2, 3), 2, "mul1r2", True)
    dec = lambda s: [l.strip() for l in s.splitlines() if "dec_mul1r2(" in l]
    assert dec(plain) == dec(pf) and len(dec(pf)) == 2 * 32
    assert "n_w" not in plain and "n_q" not in plain and "n_wp" in pf
    assert pf.count("i + 1u < myn") == 2


def test_lever_off_source_is_stock():
    stock = S.gather_gemv_source((2, 3), 2, "mul1")
    assert stock == S.gather_gemv_source((2, 3), 2, "mul1", False)
    assert "dec_mul1(" in stock and "n_wp" not in stock and "mul1r" not in stock


# ----------------------------------------------------------------------------------------------- python side (mlx import, no GPU)

def _import_moe():
    pytest.importorskip("mlx.core")
    from kyojin_mlx import moe as M
    return M


def test_codebook_selection(monkeypatch):
    M = _import_moe()
    for on, dec, cb, want in ((False, 2, "mul1", "mul1"), (False, 2, "mul1fd", "mul1fd"), (True, 2, "mul1", "mul1r2"), (True, 1, "mul1", "mul1r1"),
                              (True, 2, "mul1fd", "mul1fdr"), (True, 2, "mul1af", "mul1af"), (True, 2, "mcg", "mcg")):
        monkeypatch.setattr(M, "ROOF", on)
        monkeypatch.setattr(M, "ROOF_DEC", dec)
        assert M._roof(cb) == want, (on, dec, cb)


def test_flags_registered():
    pytest.importorskip("mlx.core")
    from kyojin_mlx import flags
    assert flags.FLAGS["moeroof"].default == "1" and flags.FLAGS["moeroof"].runtime
    assert flags.FLAGS["moeroof_dec"].kind == "int" and flags.FLAGS["moeroof_pf"].env == "KYOJIN_MOEROOF_PF"


# ----------------------------------------------------------------------------------------------- Metal (Ultra)

def _metal():
    if os.environ.get("KYOJIN_TEST_NO_METAL") == "1":
        return False
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


needs_metal = pytest.mark.skipif(not _metal(), reason="Metal GPU not available (or KYOJIN_TEST_NO_METAL=1)")


@needs_metal
@pytest.mark.parametrize("fd", [False, True])
@pytest.mark.parametrize("dec", [1, 2])
@pytest.mark.parametrize("pf", [False, True])
@pytest.mark.parametrize("rows,union", [(1, False), (4, False), (4, True)])
def test_metal_roof_bit_identical(monkeypatch, rows, union, pf, dec, fd):
    """Layer output with KYOJIN_MOEROOF on == off, bit for bit, for the stock decode (fd=False) and the FASTDEC decode (fd=True; rows >= 2 grouped path ignores fd)."""
    import mlx.core as mx
    from kyojin_mlx import moe as M
    from test_moe import KS, TOPK, make_stack, _route, _x
    kin, ninter = 256, 384
    st = (make_stack(kin, ninter, KS, "mul1", 1)[0], make_stack(kin, ninter, KS, "mul1", 2)[0], make_stack(ninter, kin, KS, "mul1", 3)[0])
    ids, w = _route(rows, 40 + rows)
    x = _x(rows, kin, 9)

    def run(on):
        monkeypatch.setattr(M, "ROOF", on)
        monkeypatch.setattr(M, "ROOF_DEC", dec)
        monkeypatch.setattr(M, "ROOF_PF", pf)
        m = M.MoeGather(*st, TOPK, limit=10.0, union=union)
        m.fastdec = fd
        y = m(x, ids, w)
        mx.eval(y)
        return np.array(y)

    ref, got = run(False), run(True)
    assert np.array_equal(ref.view(np.uint32), got.view(np.uint32))
