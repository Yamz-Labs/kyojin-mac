"""KYOJIN_MOE_VIEWS: the routed MoE kernels bind a prefix view of each expert bank (same buffer, tiny data_size) so MLX bills the command buffer
for a few KiB instead of the whole bank. The kernels index the whole bank, so the results must be bit-identical to the plain bind."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import moe as M
from test_moe import KS, TOPK, make_stack, _route, _x


@pytest.fixture(scope="module")
def stacks():
    kin, ninter = 256, 384
    return (make_stack(kin, ninter, KS, "mul1", 1)[0], make_stack(kin, ninter, KS, "mul1", 2)[0], make_stack(ninter, kin, KS, "mul1", 3)[0])


@pytest.fixture
def views(monkeypatch):
    monkeypatch.setattr(M, "VIEWS", True)
    monkeypatch.setattr(M, "VIEW_MIN_BYTES", 0)          # the synthetic banks are small
    monkeypatch.setattr(M, "VIEW_WORDS", 64)


def test_bound_is_a_small_view(stacks, views):
    for st in stacks:
        st._view = None
        b = st.bound
        assert b.size == 64 and b.size < st.words.size
        assert st.bound is b                                   # built once
        assert np.array_equal(np.array(b), np.array(st.words)[:64])


def test_flag_off_binds_the_bank(stacks, monkeypatch):
    monkeypatch.setattr(M, "VIEWS", False)
    assert stacks[0].bound is stacks[0].words


def test_small_bank_stays(stacks, monkeypatch):
    monkeypatch.setattr(M, "VIEWS", True)
    for st in stacks:
        st._view = None
        assert st.bound.size == st.words.size                  # below VIEW_MIN_BYTES


@pytest.mark.parametrize("rows", [1, 2, 4, 7])
@pytest.mark.parametrize("wpre", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_identical(stacks, monkeypatch, rows, wpre, grouped):
    ids, w = _route(rows, 20 + rows)
    x = _x(rows, stacks[0].kin, 4)
    monkeypatch.setattr(M, "VIEWS", False)
    ref = np.array(M.MoeGather(*stacks, TOPK, limit=10.0, wpre=wpre, grouped=grouped)(x, ids, w))
    monkeypatch.setattr(M, "VIEWS", True)
    monkeypatch.setattr(M, "VIEW_MIN_BYTES", 0)
    monkeypatch.setattr(M, "VIEW_WORDS", 64)
    for st in stacks:
        st._view = None
    got = np.array(M.MoeGather(*stacks, TOPK, limit=10.0, wpre=wpre, grouped=grouped)(x, ids, w))
    assert np.array_equal(got.view(np.uint16), ref.view(np.uint16)) if got.dtype == np.float16 else np.array_equal(got.view(np.uint32), ref.view(np.uint32))
