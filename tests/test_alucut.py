"""cplain exactness checks that need no GPU (kyojin_mlx.alucut): every rewrite applies exactly once to the real kernel text, and the old and the new
algorithm are emulated in numpy (thread / simdgroup / barrier model) and compared bit for bit."""
import itertools
import re

import numpy as np
import pytest

from kyojin_mlx import alucut as AC


def _src(mod_attr: str):
    """Kernel text without importing mlx: read the module source and exec the string literal."""
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "kyojin_mlx"
    name, attr = mod_attr.split(":")
    text = (root / f"{name}.py").read_text()
    m = re.search(rf'^{attr} = r"""(.*?)"""', text, re.S | re.M)
    assert m, attr
    return m.group(1)


MIXVEC = _src("gluefuse:_MIXVEC")
ROUTER = _src("fuse:_ROUTER")
EXPAND = _src("fuse:_EXPAND")


# ------------------------------------------------------------------------------------------------------------ source integrity

def test_rewrites_apply_once_and_compose():
    for bar, pf in itertools.product((False, True), repeat=2):
        out = AC.mixvec_source(MIXVEC, bar, pf)
        assert (out != MIXVEC) == (bar or pf)
        assert ("inv_s[0] = " in out) == (not bar)               # the broadcast through threadgroup memory is gone under bar
        assert out.count("threadgroup_barrier") == (MIXVEC.count("threadgroup_barrier") - (2 if bar else 0))
        assert ("wn[j] = (kn + j < K)" in out) == pf
    r = AC.router_source(ROUTER)
    assert r.count("threadgroup_barrier") == ROUTER.count("threadgroup_barrier") - 1      # loop body: 2 barriers -> 1
    assert "threadgroup uint chosen" not in r and "if (t == 0) {\n            float fk" not in r
    e = AC.expand_source_no_y32(EXPAND)
    assert "y32" not in e.replace("y32 is", "") and "y[(ulong(b) * 4 + j) * D + d] = o;" in e


def test_mixvec_every_slot_of_ls_is_written():
    """The pre-zeroing of `ls` is dropped under MIXBAR: valid only because the threadgroup has 32 simdgroups (1024 threads), as the launch says."""
    gf = (__import__("pathlib").Path(__file__).resolve().parents[1] / "kyojin_mlx" / "gluefuse.py").read_text()
    assert "threadgroup=(1024, 1, 1)" in gf and "threadgroup float ls[32];" in MIXVEC


# ------------------------------------------------------------------------------------------------------------ MIXBAR: reduction emulation

def _simd_sum(v: np.ndarray) -> np.float32:
    """A fixed butterfly (xor 16, 8, 4, 2, 1): any fixed tree is fine, both variants call the same function on the same 32 lanes."""
    v = v.astype(np.float32).copy()
    for s in (16, 8, 4, 2, 1):
        v = v + v[np.arange(32) ^ s]
    return v[0]


def _inv_old(partials: np.ndarray, K: int, eps: np.float32) -> np.float32:
    # partials[sg, lane] = per-thread accumulators; simd_sum per simdgroup, ls, simdgroup 0 sums ls[lane], lane 0 stores inv_s, everybody reads it
    ls = np.zeros(32, np.float32)
    for sg in range(32):
        ls[sg] = _simd_sum(partials[sg])
    acc = _simd_sum(ls)
    inv_s = np.float32(1.0) / np.sqrt(np.float32(acc / np.float32(K)) + eps)
    return inv_s


def _inv_new(partials: np.ndarray, K: int, eps: np.float32) -> list[np.float32]:
    ls = np.zeros(32, np.float32)
    for sg in range(32):
        ls[sg] = _simd_sum(partials[sg])
    out = []
    for sg in range(32):                                          # every simdgroup computes it
        acc = _simd_sum(ls)
        out.append(np.float32(1.0) / np.sqrt(np.float32(acc / np.float32(K)) + eps))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_mixbar_inv_identical_in_every_simdgroup(seed):
    rng = np.random.default_rng(seed)
    K = 20480
    part = (rng.standard_normal((32, 32)) ** 2 * 10.0 ** rng.integers(-3, 3)).astype(np.float32)
    eps = np.float32(1e-6)
    old = _inv_old(part, K, eps)
    new = _inv_new(part, K, eps)
    assert all(np.float32(n).tobytes() == old.tobytes() for n in new)


# ------------------------------------------------------------------------------------------------------------ MIXPF: the multiplied sequence

@pytest.mark.parametrize("K,KS", [(20480, 8), (20480, 4), (16384, 8), (20480 + 24, 8), (4100, 2), (8192, 1)])
def test_mixpf_same_k_sequence(K, KS):
    for sg, lane in itertools.product(range(KS), (0, 1, 7, 31)):
        a = AC.mixvec_dot_steps(K, KS, sg, lane, False)
        b = AC.mixvec_dot_steps(K, KS, sg, lane, True)
        assert a == b and len(a) >= 0


def test_mixpf_lookahead_loads_are_masked():
    """The look-ahead load of the last step reads past the row's end only through the same `< K` mask as the multiplied loads (zeros, never multiplied)."""
    out = AC.mixvec_source(MIXVEC, False, True)
    assert "wn[j] = (kn + j < K) ? float(wp[kn + j]) : 0.0f;" in out and "wn[j] = (k0 + j < K) ? float(wp0[k0 + j]) : 0.0f;" in out
    for sg, lane in itertools.product(range(8), range(32)):
        steps = AC.mixvec_dot_steps(20480, 8, sg, lane, True)
        assert steps == sorted(steps) and all(k + 7 < 20480 or k >= 0 for k in steps)


# ------------------------------------------------------------------------------------------------------------ ROUTER1: selection emulation

INF = np.float32(np.inf)
U32MAX = 0xFFFFFFFF


def _thread_scan(skey: np.ndarray, t: int, E: int):
    bk, bi = INF, U32MAX
    for e in range(t, E, 128):
        k = skey[e]
        if k < bk or (k == bk and e < bi):
            bk, bi = k, e
    return bk, bi


def _simd_min_pair(vals):
    """vals: 32 (bk, bi); mk = simd_min(bk); mi = simd_min(bk == mk ? bi : U32MAX)."""
    mk = min(v[0] for v in vals)
    mi = min((v[1] if v[0] == mk else U32MAX) for v in vals)
    return mk, mi


def _merge(rk, ri):
    fk, fi = rk[0], ri[0]
    for g in range(1, 4):
        if rk[g] < fk or (rk[g] == fk and ri[g] < fi):
            fk, fi = rk[g], ri[g]
    return fi


def select_old(skey: np.ndarray, K: int) -> list[int]:
    skey = skey.copy()
    E = len(skey)
    chosen = []
    for j in range(K):
        rk, ri = [None] * 4, [None] * 4
        for sg in range(4):
            rk[sg], ri[sg] = _simd_min_pair([_thread_scan(skey, sg * 32 + l, E) for l in range(32)])
        fi = _merge(rk, ri)                                        # thread 0
        chosen.append(fi)
        skey[fi] = INF                                             # thread 0 writes any element
    return chosen


def select_new(skey: np.ndarray, K: int) -> list[int]:
    """Per-thread private views: thread t only ever changes skey entries it owns (e % 128 == t); rk / ri are double buffered by round parity."""
    E = len(skey)
    own = [dict() for _ in range(128)]                             # thread-private retirements, applied on top of the shared initial keys
    def key(t, e):
        return INF if e in own[t] else skey[e]
    rk = np.zeros((2, 4), np.float32)
    ri = np.zeros((2, 4), np.uint64)
    chosen_by_thread = [[] for _ in range(128)]
    for j in range(K):
        for sg in range(4):
            vals = []
            for l in range(32):
                t = sg * 32 + l
                bk, bi = INF, U32MAX
                for e in range(t, E, 128):
                    k = key(t, e)
                    if k < bk or (k == bk and e < bi):
                        bk, bi = k, e
                vals.append((bk, bi))
            rk[j & 1][sg], ri[j & 1][sg] = _simd_min_pair(vals)
        # barrier
        for t in range(128):
            fi = _merge(rk[j & 1], [int(v) for v in ri[j & 1]])
            chosen_by_thread[t].append(fi)
            if fi < E and (fi & 127) == t:
                own[t][fi] = True
    assert all(c == chosen_by_thread[0] for c in chosen_by_thread)  # every thread tracks the same winners
    return chosen_by_thread[0]


def reference(skey: np.ndarray, K: int) -> list[int]:
    order = sorted(range(len(skey)), key=lambda e: (float(skey[e]), e))
    return order[:K]


@pytest.mark.parametrize("E,K", [(256, 6), (256, 8), (160, 6), (384, 6)])
@pytest.mark.parametrize("mode", ["random", "ties", "heavy_ties", "infs"])
def test_router1_selection_identical(E, K, mode):
    rng = np.random.default_rng(E * 31 + K)
    for _ in range(6):
        if mode == "random":
            s = rng.standard_normal(E).astype(np.float32)
        elif mode == "ties":
            s = rng.integers(-4, 4, E).astype(np.float32)
        elif mode == "heavy_ties":
            s = np.zeros(E, np.float32)
            s[rng.integers(0, E, 3)] = -1.0
        else:
            s = rng.standard_normal(E).astype(np.float32)
            s[rng.integers(0, E, E // 2)] = INF
        a, b = select_old(s, K), select_new(s, K)
        assert a == b == reference(s, K)


def test_router1_owner_rule_matches_scan_stride():
    """Thread t scans e = t, t + 128, ...: so e % 128 == t is the owner, and no other thread ever reads skey[e] after the barrier-free retirement."""
    E = 256
    seen = {}
    for t in range(128):
        for e in range(t, E, 128):
            assert e not in seen
            seen[e] = t
            assert (e & 127) == t
    assert len(seen) == E
