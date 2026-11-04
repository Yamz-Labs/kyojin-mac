"""CPU conformance suite on tiny synthetic models of four shapes (MoE+CED, MoE, dense, dense hybrid): load through the real loaders, prefill, decode,
KV snapshot / restore (in memory and on disk), each checked against a slow reference (kyojin_mlx.conformance). Also the mutation tests: a check that
cannot fail proves nothing, so each one is shown failing when the thing it guards is broken.
Set KYOJIN_CONF_JSON=<file> to write the measured numbers of every shape (docs/reports/cgeneric_REPORT.md quotes them)."""
import json
import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.deepseek_v41")
pytest.importorskip("mlx_lm")
from kyojin_mlx import conformance as C, multiuser, rows, synth as S  # noqa: E402

RESULTS = {}


@pytest.fixture(autouse=True)
def _cpu_device():
    """These are CPU conformance checks (kyojin_mlx/conformance.py): on a Metal device the same model already gives 5e-3 .. 3.5 max abs logit difference
    between a chunked prefill and the one-shot forward (measured on q35_dense, stock mlx-lm, no Kyojin patch), which is the GPU backend's kernel choice, not
    a property of a loader or a cache. Pin the CPU device so TOL 2e-3 means what it was calibrated on; the Metal paths have tests/cgeneric_mac.py."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.mark.parametrize("name", list(S.SHAPES))
def test_shape_conforms(name, tmp_path):
    r = C.run_shape(S.SHAPES[name], str(tmp_path / name))
    RESULTS[name] = r.numbers
    assert not r.failures, r.failures
    n = r.numbers
    assert n["prefill_vs_ref"] <= C.TOL and n["decode_vs_ref"] <= C.TOL
    assert n["snapshot_fork1_bitwise"] and n["snapshot_fork2_bitwise"] and n["disk_bitwise"] and n["disk_hit_at"] == 16


def test_shapes_cover_moe_and_ced_both_ways():
    cells = {(s.moe, s.ced) for s in S.SHAPES.values()}
    assert (True, True) in cells and (True, False) in cells and (False, False) in cells
    assert (False, True) not in cells          # no stock CED-capable block is dense: said, not faked (kyojin_mlx/synth.py)


def test_ced_shape_exercises_the_cut(tmp_path):
    r = C.run_shape(S.SHAPES["ds_moe_ced"], str(tmp_path / "c"))
    assert r.numbers["ced_exact_tail_rows_cut"] > 0 and r.numbers["ced_receptive_rows"] == 76
    assert r.numbers["ced_exact_tail_diff"] <= C.CED_TOL
    assert r.numbers["ced_default_tail_diff"] > r.numbers["ced_exact_tail_diff"]       # the default tail IS an approximation: the check can tell them apart


# ------------------------------------------------------------------------------------------------------------------ mutation tests
@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    out = {}
    for name in ("ds_moe_ced", "q35_dense"):
        sh = S.SHAPES[name]
        p = str(tmp_path_factory.mktemp(name))
        S.write_pack(sh, p)
        out[name] = (sh, *S.load_pack(sh, p))
    return out


@pytest.mark.parametrize("name", ["ds_moe_ced", "q35_dense"])
def test_reference_check_fails_against_another_model(loaded, name):
    sh, model, lm, mk = loaded[name]
    _, lm2, mk2 = S.build_reference(sh, seed=7)                                  # different random weights = a wrong reference
    vocab = int(sh.text["vocab_size"])
    ids = C.seq(37, vocab)
    ref_other = C.reference_last(lm2, mk2, ids)
    _, lg = C.prefill(lm, mk, ids, 8)
    assert C.maxdiff(lg, ref_other) > 10 * C.TOL


@pytest.mark.parametrize("name,fork", [("ds_moe_ced", rows.fork), ("q35_dense", multiuser.fork_cache)])
def test_snapshot_check_fails_when_the_fork_aliases(loaded, name, fork):
    sh, model, lm, mk = loaded[name]
    vocab = int(sh.text["vocab_size"])
    assert all(C.check_snapshot(lm, mk, fork, vocab).values())                   # the real fork passes
    s = C.check_snapshot(lm, mk, lambda cache: cache, vocab)                     # a "fork" that is the same cache: the second decode continues the first
    assert not (s["snapshot_fork1_bitwise"] and s["snapshot_fork2_bitwise"])


def test_disk_check_fails_when_the_restored_cache_is_wrong(loaded, monkeypatch, tmp_path):
    sh, model, lm, mk = loaded["q35_dense"]
    from kyojin_mlx import kvdisk as K
    vocab = int(sh.text["vocab_size"])
    ok = C.check_disk(lm, mk, vocab, {"arch": "t"}, root=str(tmp_path / "a"))
    assert ok["disk_bitwise"] and ok["disk_hit_at"] == 16
    orig = K.GenericAdapter.restore

    def broken(self, make_cache, parts):
        cache = orig(self, make_cache, parts)
        for c in cache:
            if getattr(c, "keys", None) is not None:
                c.keys = c.keys + 1e-3
        return cache
    monkeypatch.setattr(K.GenericAdapter, "restore", broken)
    assert not C.check_disk(lm, mk, vocab, {"arch": "t2"}, root=str(tmp_path / "b"))["disk_bitwise"]


def test_decode_check_fails_when_the_cache_loses_a_row(loaded):
    sh, model, lm, mk = loaded["q35_dense"]
    vocab = int(sh.text["vocab_size"])
    ids = C.seq(20, vocab)
    cache, lg = C.prefill(lm, mk, ids, 5)
    for c in cache:                                                              # forget the last prompt row of every full-attention layer
        if getattr(c, "keys", None) is not None:
            c.keys, c.values, c.offset = c.keys[:, :, :-1], c.values[:, :, :-1], c.offset - 1
    step = C.decode(lm, cache, 5, 1)[0][1]
    assert C.maxdiff(step, C.reference_last(lm, mk, ids + [5])) > C.TOL


def teardown_module(_):
    p = os.environ.get("KYOJIN_CONF_JSON")
    if p and RESULTS:
        json.dump(RESULTS, open(p, "w"), indent=1, sort_keys=True, default=str)
