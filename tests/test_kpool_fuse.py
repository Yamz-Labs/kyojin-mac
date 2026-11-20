"""Flag kpool_fuse (kyojin_mlx/kpool_fuse.py): the one-launch decode pool selection (scores + top-k + slot mask) against the stock path
(foreign_skeleton_kpool.select_pools + keep_mask). CPU parts always: a numpy model of the kernel's algorithm (order-preserving key, 32-pass radix select,
tie scan, mask formulas) must equal the stock path, at Hy4-flash's shapes (16 heads, d 128, kp 4, top_k 2048, tail) and at small ones, with forced ties.
Metal parts need KYOJIN_METAL_TESTS=1 and a GPU (cell c90; run behind can_build.sh): selected pools EXACTLY equal, keep mask equal, scores within 1e-5 relative."""
import os

import numpy as np
import mlx.core as mx
import pytest

from kyojin_mlx import foreign_skeleton_kpool as KP
from kyojin_mlx import kpool_fuse as KF

METAL = os.environ.get("KYOJIN_METAL_TESTS", "0") == "1" and mx.metal.is_available()
needs_metal = pytest.mark.skipif(not METAL, reason="Metal kernel test: KYOJIN_METAL_TESTS=1 on a GPU (cell c90)")

@pytest.fixture(autouse=True)
def _device():
    """CPU parts run on the CPU stream (no GPU); the Metal parts on the GPU."""
    old = mx.default_device()
    mx.set_default_device(mx.gpu if METAL else mx.cpu)
    yield
    mx.set_default_device(old)


# (H, d, kp, top_k, select_tail, S, L, offsets)
CASES = [
    (3, 16, 4, 12, True, 31, 1, [30]),
    (3, 16, 4, 12, False, 40, 3, [37, 20]),
    (2, 8, 2, 6, True, 25, 8, [17, 5, 0]),
    (16, 128, 4, 2048, True, 3000, 1, [2999]),            # the real Hy4-flash shape: 749 pools, 511 chosen
    (16, 128, 4, 2048, True, 2300, 4, [2290, 1500]),
    (16, 128, 4, 2048, False, 9000, 1, [8999]),
]


def _inputs(H, d, kp, top_k, S, L, offs, seed=0, dup=False):
    rng = np.random.default_rng(seed)
    B, J = len(offs), max(S // kp, 1)
    iq = rng.standard_normal((B, H, L, d)).astype(np.float32)
    w = (rng.standard_normal((B, L, H)) / np.sqrt(d * H)).astype(np.float32)
    pk = rng.standard_normal((B, J, d)).astype(np.float32)
    if dup:                                                 # exact ties: pools 2i and 2i + 1 are the same key
        pk[:, 1::2] = pk[:, 0::2][:, :pk[:, 1::2].shape[1]]
    valid = np.arange(J)[None, :] < (np.array(offs)[:, None] + L) // kp
    return iq, w, pk, valid


def _scores_np(iq, w, pk):
    kq = np.maximum(np.einsum("bhld,bjd->bhlj", iq.astype(np.float64), pk.astype(np.float64)), 0.0)
    return np.einsum("bhlj,blh->blj", kq, w.astype(np.float64))


# ---- the kernel's algorithm in numpy
def _key(s):
    s = np.where(s == 0, np.float32(0), s).astype(np.float32)
    u = s.view(np.uint32)
    return np.where(u & 0x80000000, ~u, u | 0x80000000).astype(np.uint32)


def _radix(keys, nt):
    """The NT-th largest key and how many of the keys equal to it are taken (the kernel's 32 passes)."""
    prefix, k = np.uint32(0), nt
    for bit in range(31, -1, -1):
        cand = np.uint32(prefix | np.uint32(1 << bit))
        him = np.uint32((0xFFFFFFFF << bit) & 0xFFFFFFFF)
        tot = int(((keys & him) == cand).sum())
        if tot >= k:
            prefix = cand
        else:
            k -= tot
    return prefix, k


def model_select(score, off, S, L, kp, top_k, tail, valid):
    """score [B, L, J] float32 (-inf = invisible, the kernel's phase 1 output) -> sel [B, L, J], keep [B, 1, L, S]."""
    B, _, J = score.shape
    nt = min(J, top_k // kp - int(tail))
    sel = np.zeros(score.shape, bool)
    keep = np.zeros((B, 1, L, S), bool)
    for b in range(B):
        for l in range(L):
            q = off[b] + l
            keys = _key(score[b, l])
            if nt > 0:
                T, krem = _radix(keys, nt)
                ties = np.nonzero(keys == T)[0][:krem]            # ascending pool index
                pick = keys > T
                pick[ties] = True
                nv = min(q // kp if tail else (q + 1) // kp, J)
                sel[b, l] = pick & (np.arange(J) < nv) & valid[b]
            pad = S - L - off[b]
            gs = (q // kp) * kp if tail else q - (q + 1) % kp + 1
            for s in range(S):
                kpos = s - pad
                if 0 <= kpos <= q:
                    kj = kpos // kp
                    keep[b, 0, l, s] = (kj < J and sel[b, l, kj]) or kpos >= gs
    return sel, keep


def _stock(iq, w, pk, valid, offs, S, kp, top_k, tail):
    st = KP.select_pools(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
    return np.array(st["sel"]), np.array(KP.keep_mask(st))


def _phase1(iq, w, pk, valid, offs, L, kp, tail):
    sc = _scores_np(iq, w, pk).astype(np.float32)
    J = pk.shape[1]
    q = np.array(offs)[:, None] + np.arange(L)[None, :]
    nv = (q // kp if tail else (q + 1) // kp)
    vis = (np.arange(J)[None, None, :] < nv[..., None]) & valid[:, None, :]
    return np.where(vis, sc, -np.inf).astype(np.float32)


def test_key_is_order_preserving():
    x = np.array([-np.inf, -3.5, -1e-30, -0.0, 0.0, 1e-30, 2.0, 7.5, np.inf], np.float32)
    k = _key(x)
    assert (np.diff(k.astype(np.int64)) >= 0).all() and k[3] == k[4] and len(set(k.tolist())) == len(x) - 1


@pytest.mark.parametrize("n", [1, 5, 50])
def test_radix_ties_equal_stable_argsort(n):
    rng = np.random.default_rng(n)
    for _ in range(20):
        s = rng.integers(-3, 4, 60).astype(np.float32) * 0.5          # lots of exact ties
        T, krem = _radix(_key(s), n)
        keys = _key(s)
        pick = keys > T
        pick[np.nonzero(keys == T)[0][:krem]] = True
        want = np.zeros(60, bool)
        want[np.argsort(-s, kind="stable")[:n]] = True
        assert (pick == want).all()


@pytest.mark.parametrize("cs", CASES)
def test_model_equals_stock(cs):
    H, d, kp, top_k, tail, S, L, offs = cs
    iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs)
    s1, k1 = _stock(iq, w, pk, valid, offs, S, kp, top_k, tail)
    s2, k2 = model_select(_phase1(iq, w, pk, valid, offs, L, kp, tail), offs, S, L, kp, top_k, tail, valid)
    assert (s1 == s2).all() and (k1 == k2).all()
    assert s1.sum() > 0


def test_model_with_ties_equals_stock_rule():
    """Duplicated pooled keys give exactly equal scores: the lowest pool index wins, as the stable argsort of the stock path."""
    H, d, kp, top_k, tail, S, L, offs = 3, 16, 4, 12, True, 60, 2, [50, 33]
    iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, dup=True)
    sc = _phase1(iq, w, pk, valid, offs, L, kp, tail)
    s2, _ = model_select(sc, offs, S, L, kp, top_k, tail, valid)
    n = top_k // kp - 1
    for b in range(2):
        for l in range(L):
            want = np.zeros(sc.shape[-1], bool)
            want[np.argsort(-sc[b, l], kind="stable")[:n]] = True
            nv = min((offs[b] + l) // kp, sc.shape[-1])
            want &= (np.arange(sc.shape[-1]) < nv) & valid[b]
            assert (s2[b, l] == want).all()


def test_flag_registered_and_off():
    from kyojin_mlx import flags
    assert flags.FLAGS["kpool_fuse"].env == "KYOJIN_KPOOL_FUSE" and flags.FLAGS["kpool_fuse"].default == "0"
    assert KF.ON is False or os.environ.get("KYOJIN_KPOOL_FUSE", "0") != "0"
    assert not KF.use(9)


# ---- Metal (cell c90)
@needs_metal
@pytest.mark.parametrize("pre_j", [10 ** 9, 0])
@pytest.mark.parametrize("cs", CASES)
def test_metal_equals_stock(cs, pre_j, monkeypatch):
    """pre_j = 0: the scores come from the multi-threadgroup scoring launch (long contexts); 10 ** 9: scored inside the selection launch."""
    monkeypatch.setattr(KF, "PRE_J", pre_j)
    H, d, kp, top_k, tail, S, L, offs = cs
    for seed in range(3):
        iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, seed=seed)
        s1, k1 = _stock(iq, w, pk, valid, offs, S, kp, top_k, tail)
        keep, score, sel = KF.select(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
        keep, score, sel = np.array(keep), np.array(score), np.array(sel)
        assert (sel == s1).all(), "selected pools differ from the stock path"
        assert (keep == k1).all(), "keep mask differs from the stock path"
        ref = _phase1(iq, w, pk, valid, offs, L, kp, tail)
        fin = np.isfinite(ref)
        assert (np.isfinite(score) == fin).all()
        np.testing.assert_allclose(score[fin], ref[fin], rtol=1e-5, atol=1e-6 * float(np.abs(ref[fin]).max()))


@needs_metal
def test_metal_ties_lowest_index():
    H, d, kp, top_k, tail, S, L, offs = 3, 16, 4, 12, True, 60, 2, [50, 33]
    iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, dup=True)
    keep, score, sel = KF.select(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail)
    score = np.array(score)
    s2, k2 = model_select(score, offs, S, L, kp, top_k, tail, valid)
    assert (np.array(sel) == s2).all() and (np.array(keep) == k2).all()
    assert (score[:, :, 0::2][np.isfinite(score[:, :, 0::2])].size > 0)


@needs_metal
def test_metal_one_launch_graph():
    iq, w, pk, valid = _inputs(3, 16, 4, 12, 31, 1, [30])
    out = KF.select(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array([30], dtype=mx.int32), 31, 4, 12, True)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import count_launches as CL
    c = CL.graph_census(list(out))
    assert sum(v for k, v in c["by_type"].items() if k == "CustomKernel") == 1


@needs_metal
@pytest.mark.parametrize("cs", CASES)
def test_metal_left_aligned_equals_stock(cs):
    """The fixed-state KV buffer is left-aligned (slot == position, S = buffer length >= off + L): the fused launch with left=True equals select_pools(left=True) + keep_mask."""
    H, d, kp, top_k, tail, S0, L, offs = cs
    S = S0 + 40                                                      # buffer capacity past the valid rows
    for seed in range(2):
        iq, w, pk, valid = _inputs(H, d, kp, top_k, S, L, offs, seed=seed)
        st = KP.select_pools(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail, left=True)
        s1, k1 = np.array(st["sel"]), np.array(KP.keep_mask(st))
        keep, score, sel = KF.select(mx.array(iq), mx.array(w), mx.array(pk), mx.array(valid), mx.array(offs, dtype=mx.int32), S, kp, top_k, tail, left=True)
        assert (np.array(sel) == s1).all(), "selected pools differ from the stock left-aligned path"
        assert (np.array(keep) == k1).all(), "keep mask differs from the stock left-aligned path"
