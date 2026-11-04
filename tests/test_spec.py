"""Chained speculative verify (spec.py): D + 1 consecutive positions of ONE sequence in one batched pass equal D + 1 solo steps bit for bit,
and the rollback cache equals the solo cache. Needs a Mac (Metal) and the 2-layer DS4.1 slices (KYOJIN_SPEC_PACKS=<dir>:<dir>... ; ds2 plus the
variants tools/slice_variant.py derives from it, the Engram slice results_mac/engram_pack). Each case runs tools/spec_check.py, whose negative
controls (no layer-state link, no rollback, stock matmul, unpinned MoE) must fail."""
import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import rowinv as RI                                       # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = [p for p in os.environ.get("KYOJIN_SPEC_PACKS", "").split(":") if p]


def bits(a):
    a = np.array(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint16)


@pytest.mark.parametrize("xd,wd", [(mx.bfloat16, mx.bfloat16), (mx.float16, mx.bfloat16), (mx.float32, mx.bfloat16)])
def test_rowdot_bf16_row_invariant(xd, wd):
    rng = np.random.default_rng(7)
    K, N = 6144, 96
    w = mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(wd)
    x = mx.array(rng.standard_normal((11, K)).astype(np.float32)).astype(xd)
    one = [RI.rowdot(x[i:i + 1], w) for i in range(11)]
    out = mx.float32 if xd != wd else xd
    assert one[0].dtype == out
    for B in (1, 2, 3, 4, 8, 11):
        y = RI.rowdot(x[:B], w)
        for i in range(B):
            assert np.array_equal(bits(y.astype(mx.float32))[i].view(np.uint32), bits(one[i].astype(mx.float32))[0].view(np.uint32)), (B, i)
    ref = np.array(x.astype(mx.float32), dtype=np.float64) @ np.array(w.astype(mx.float32), dtype=np.float64).T
    got = np.array(RI.rowdot(x[:4], w).astype(mx.float32), dtype=np.float64)
    assert np.abs(got - ref[:4]).max() <= 1e-2 * np.abs(ref[:4]).max()


def run_check(pack, *args, nax=None):
    """tools/spec_check.py; nax="0" forces the bitwise mode (KYOJIN_ROWS_NAX=0, row-count dependent attention kernels off), default: the environment (NAX rows ON = tolerance mode above 4 rows)"""
    cmd = [sys.executable, os.path.join(ROOT, "tools", "spec_check.py"), pack, "--quiet", *args]
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if nax is not None:
        env["KYOJIN_ROWS_NAX"] = nax
    if nax == "0":      # bitwise mode: the multi-query MMA attention and the split P@V (ON by default) sum in a row-count dependent order, so they also go off
        env["KYOJIN_ATTEND_MMA"] = env["KYOJIN_ATTEND_SPLIT"] = "0"
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


# (pack suffix, --set values): layouts of the compressor / indexer that a 2-layer slice can hold
CASES = [
    ("ds2", []),
    ("ds2", ["compress_ratios=[4,4]"]),
    ("ds2", ["index_topk=8"]),
    ("ds2f", ["compress_ratios=[2,2]", "kv_source_layer_ids=[0,1]", "index_source_layer_ids=[0,1]", "index_topk=8"]),
    ("ds2r", ["compress_ratios=[2,2]", "index_source_layer_ids=[0,1]", "index_topk=8"]),
    ("ds2l", ["compress_ratios=[0,2]", "kv_source_layer_ids=[1]", "index_source_layer_ids=[1]", "candidate_source_layer_id=1", "index_topk=8"]),
    ("ds2f11", ["compress_ratios=[1,1]", "kv_source_layer_ids=[0,1]", "index_source_layer_ids=[0,1]", "index_topk=8"]),
    ("ds2f1", ["compress_ratios=[2,1]", "kv_source_layer_ids=[0,1]", "index_source_layer_ids=[0,1]", "index_topk=8", "candidate_source_layer_id=1"]),
    ("engram_pack", []),
]


def pack_dir(name):
    for p in PACKS:
        if os.path.basename(p.rstrip("/")) == name:
            return p
    pytest.skip(f"pack {name} not in KYOJIN_SPEC_PACKS")


@pytest.mark.parametrize("name,over", CASES, ids=[f"{n}-{'-'.join(o)[:40]}" for n, o in CASES])
def test_chain_verify_exact(name, over):
    r = run_check(pack_dir(name), "--steps", "40", *(["--set", *over] if over else []))
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "EXACT" in r.stdout


@pytest.mark.parametrize("neg", ["nolink", "norollback", "stock_matmul", "unpinned_moe"])
def test_negative_controls_fail(neg):
    # stock_matmul / unpinned_moe move the logits at fp16 rounding only: they are caught by the bitwise mode, inside the tolerance of the NAX-on mode
    r = run_check(pack_dir("ds2"), "--steps", "24", "--neg", neg, nax="0" if neg in ("stock_matmul", "unpinned_moe") else None)
    assert r.returncode == 1 and "MISMATCH" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


@pytest.mark.parametrize("nax", ["0", None], ids=["bitwise-nax0", "tolerance-nax-on"])
def test_chain_verify_depth5(nax):
    """depth 5 = up to 6 rows: above the GEMV chunk, the NAX kernel when it is on. Bitwise with KYOJIN_ROWS_NAX=0, within the rounding class otherwise."""
    r = run_check(pack_dir("ds2m"), "--steps", "30", "--depth", "5", nax=nax)
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    assert ("bitwise" in r.stdout) == (nax == "0") or nax is None


def test_chain_verify_exact_with_parallel_engram_gather():
    r = run_check(pack_dir("engram_pack"), "--steps", "40", "--parallel-gather")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert "EXACT" in r.stdout
