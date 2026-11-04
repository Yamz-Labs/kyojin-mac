"""The fused row core of a served pass (kyojin_mlx.dsv4_core ROWS_CORE_FUSE / STATE_NOCOPY): the kernels against the ones they replace.

No pack needed: the row attention and the ring of a pass are checked against `attend_rows` (the chained verify of the draft head, one session) and
against host-built rings; the pool through the base state plus the pass's own compressor rows against pool_groups of the snapshot it replaces; the
commit's state write against the snapshot + pick. The whole model on a shrunk pack is tools/rows_fused_check.py (KYOJIN_SHRUNK=<dir with ds2/>).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

from kyojin_mlx import dsv4_core as DC   # noqa: E402
from kyojin_mlx import fuse_attn as FA  # noqa: E402

SHRUNK = os.environ.get("KYOJIN_SHRUNK")
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def c(x):
    """A contiguous copy: MLX hands an array with a storage offset to a Metal kernel as a constant, which these kernels cannot bind."""
    return mx.array(np.array(x))


def bits(a):
    a = np.array(a)
    return a.view(np.uint16 if a.dtype == np.float16 else np.uint32)


def rnd(rng, shape, scale=1.0):
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float16))


def one_session(n=3, H=8, D=512, W=16, Cp=6, K=5, seed=0):
    """q / base ring / pass kv rows / pool / top-k of ONE session at n consecutive positions."""
    rng = np.random.default_rng(seed)
    pos = np.arange(9, 9 + n, dtype=np.int32)
    q = rnd(rng, (n, H, D), 0.5)
    base = rnd(rng, (1, W, D))
    kvp = rnd(rng, (n, D))
    pool = rnd(rng, (1, Cp, D))
    idx = np.full((n, K), -1, np.int32)
    for u in range(n):
        m = min(K, 2 + u)
        idx[u, :m] = np.sort(rng.choice(Cp, m, replace=False))
    rt = mx.array(rng.standard_normal((n, 64)).astype(np.float32))
    it = mx.array(rng.standard_normal((n, 64)).astype(np.float32))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    return dict(q=q, base=base, kvp=kvp, pool=pool, idx=mx.array(idx), pos=mx.array(pos), rt=rt, it=it, sink=sink, scale=0.044, W=W, H=H, D=D)


def test_attend_rows_m_equals_attend_rows_one_session():
    """One session: the session-aware kernel is the chained-verify kernel bit for bit (sess = 0, first = 0)."""
    a = one_session()
    n = a["q"].shape[0]
    o1 = FA.attend_rows(a["q"].reshape(n, -1), a["base"], a["kvp"], a["pool"], a["idx"], a["sink"], a["rt"], a["it"], a["pos"], a["H"], 64, a["scale"])
    o2 = FA.attend_rows_m(a["q"].reshape(n, -1), a["base"], a["kvp"], a["pool"], a["idx"], a["sink"], a["rt"], a["it"], a["pos"],
                          mx.array(np.zeros(n, np.uint32)), mx.array(np.zeros(n, np.int32)), a["H"], 64, a["scale"])
    assert np.array_equal(bits(o1), bits(o2))


def test_attend_rows_m_row_is_independent_of_the_other_sessions():
    """Two sessions of two rows: a row of the pass keeps the bits it has alone in its own session (own ring, own pool, own top-k)."""
    a, b = one_session(2, seed=1), one_session(2, seed=2)
    b["pos"] = mx.array(np.array(b["pos"]) + 40)                        # another session at another position
    n = 4
    q = mx.concatenate([a["q"], b["q"]], axis=0)
    kvp = mx.concatenate([a["kvp"], b["kvp"]], axis=0)
    idx = mx.concatenate([a["idx"], b["idx"]], axis=0)
    rt = mx.concatenate([a["rt"], b["rt"]], axis=0)
    it = mx.concatenate([a["it"], b["it"]], axis=0)
    pos = mx.concatenate([a["pos"], b["pos"]])
    ring = mx.concatenate([a["base"], b["base"]], axis=0)
    pool = mx.concatenate([a["pool"], b["pool"]], axis=0)
    sess = mx.array(np.array([0, 0, 1, 1], np.uint32))
    fst = mx.array(np.array([0, 0, 2, 2], np.int32))
    whole = np.array(FA.attend_rows_m(q.reshape(n, -1), ring, kvp, pool, idx, a["sink"], rt, it, pos, sess, fst, a["H"], 64, a["scale"]))
    for s in (0, 1):                                                     # the rows of one session, as a pass of that session alone
        rows = [r for r in range(n) if int(np.array(sess)[r]) == s]
        k = len(rows)
        one = np.array(FA.attend_rows_m(c(mx.concatenate([q[r:r + 1] for r in rows])).reshape(k, -1), ring, c(mx.concatenate([kvp[r:r + 1] for r in rows])),
                                        pool, c(mx.concatenate([idx[r:r + 1] for r in rows])), a["sink"], c(mx.concatenate([rt[r:r + 1] for r in rows])),
                                        c(mx.concatenate([it[r:r + 1] for r in rows])), c(mx.concatenate([pos[r:r + 1] for r in rows])),
                                        mx.array(np.full(k, s, np.uint32)), mx.array(np.zeros(k, np.int32)), a["H"], 64, a["scale"]))
        for j, r in enumerate(rows):
            assert np.array_equal(whole[r:r + 1].view(np.uint16), one[j:j + 1].view(np.uint16)), (s, r)


def test_kv_apply_m_writes_the_ring_each_row_sees():
    """The rings of the rows of a pass over two sessions against the host's ring arithmetic (slot = position % W, later rows win)."""
    W, D = 16, 512
    rng = np.random.default_rng(3)
    pos = np.array([14, 15, 16, 40, 41], np.int32)
    sess = np.array([0, 0, 0, 1, 1], np.uint32)
    fst = np.array([0, 0, 0, 3, 3], np.int32)
    ring = rnd(rng, (2, W, D))
    kvp = rnd(rng, (5, D))
    got = np.array(FA.kv_apply_m(ring, kvp, mx.array(sess), mx.array(fst), mx.array(pos)))
    for j in range(5):
        w = np.array(ring[int(sess[j])])
        for i in range(int(fst[j]), j + 1):
            w[pos[i] % W] = np.array(kvp[i])
        assert np.array_equal(got[j].view(np.uint16), np.array(w).view(np.uint16)), j


def test_pool_overlay_equals_pool_groups_of_the_snapshot():
    """The pool through base + the pass's own compressor rows against _snapshot + overlay + pool_groups, bit for bit (r = 4, 6 rows)."""
    rng = np.random.default_rng(4)
    S, R, D = 1, 4, 64
    n = 6
    base_k = mx.array(rng.standard_normal((S, R, D)).astype(np.float32))
    base_s = mx.array(rng.standard_normal((S, R, D)).astype(np.float32) * 0.5)
    kvc = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    scc = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    plan = DC.Plan([11, 12, 13, 14, 15, 16], [n])
    em = plan.emit(R)
    ref_k = DC._snapshot(base_k, plan, (R, D), mx.float32, 0.0)
    ref_s = DC._snapshot(base_s, plan, (R, D), mx.float32, -np.inf)
    ov = plan.overlay(R)
    ref_k[ov.rows, ov.slots] = kvc if ov.src is None else kvc[ov.src]
    ref_s[ov.rows, ov.slots] = scc if ov.src is None else scc[ov.src]
    ref = DC.pool_groups(ref_k, ref_s, em.rows_arr)
    got = DC.pool_overlay(base_k, base_s, kvc, scc, em.rows_arr, plan.sess_idx, plan.first_arr(), plan.pos_arr)
    assert em.n > 0
    assert np.array_equal(bits(ref), bits(got)), np.abs(np.array(ref) - np.array(got)).max()


def test_commit_state_write_equals_the_snapshot_pick():
    """STATE_NOCOPY's commit write (one entry per (session, slot), the later row wins) against the snapshot + pick of the batched core."""
    rng = np.random.default_rng(5)
    S, R, D = 2, 4, 8
    pos = [11, 12, 13, 14, 15, 16, 40, 41, 42]
    counts = [7, 2]
    plan = DC.Plan(pos, counts)
    n = len(pos)
    base = mx.array(rng.standard_normal((S, R, D)).astype(np.float32))
    kvc = mx.array(rng.standard_normal((n, D)).astype(np.float32))
    snap = DC._snapshot(base, plan, (R, D), mx.float32, 0.0)
    ov = plan.overlay(R)
    snap[ov.rows, ov.slots] = kvc if ov.src is None else kvc[ov.src]
    for accepted in ([0, 0], [3, 1], [6, 1], [4, 1]):
        last = [plan.first[s] + a for s, a in enumerate(accepted)]
        p_last = [plan.p0[s] + a for s, a in enumerate(accepted)]
        ref = np.array(snap[mx.array(last, dtype=mx.int32)])
        e = plan.accepted(p_last, R)
        got = np.array(DC._write_rows(mx.array(np.array(base)), e, kvc))
        assert np.array_equal(bits(ref), bits(got)), accepted


@pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set (dir with ds2/)")
def test_rows_fused_whole_model(tmp_path):
    """tools/rows_fused_check.py on the shrunk pack: ratio 1 (ds2) and ratio 4 (a wgate added by tools/make_ds2r.py, the compressor state path):
    the fused row core's bound, the copy removal bit for bit, and the rollback with each lever on and off."""
    src = os.path.join(SHRUNK, "ds2")
    if not os.path.isdir(src):
        pytest.skip(f"{src} missing")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    r4 = str(tmp_path / "ds2r")
    subprocess.run([sys.executable, os.path.join(TOOLS, "make_ds2r.py"), src, r4], check=True, capture_output=True, env=env)
    for p in (src, r4):
        r = subprocess.run([sys.executable, os.path.join(TOOLS, "rows_fused_check.py"), p, "--seeds", "2", "--rows", "2,4,6"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, (p, r.stdout[-4000:] + r.stderr[-2000:])
