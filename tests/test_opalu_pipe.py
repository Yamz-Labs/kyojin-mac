"""pf_pipe (KYOJIN_NAX_PIPE): the prefill expert GEMM decodes k step j+1 into a second right-input cooperative tensor and issues the matmul of
step j behind it, so the decode ALU and the tensor unit overlap instead of alternating. Same tiles, same decode, same k order into the same
accumulators: the routed output must be the same bits as the unpipelined kernel, over every bit width of the pack (2..6).

Needs Metal (the kernels run on the GPU). The generator checks run without it.
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from kyojin_mlx import flags as FL, metal_nax as NX  # noqa: E402

CFG = (32, 2, 32, 64)


def test_flag_is_off_by_default():
    assert FL.FLAGS["pf_pipe"].default == "0"
    assert FL.FLAGS["pf_pipe"].env == "KYOJIN_NAX_PIPE"
    assert NX.PIPE is False


def test_generated_source_has_two_tensors_and_a_decode_ahead():
    src = NX.coop_source((3,), "mul1", *CFG[:2], CFG[2], CFG[3], pipe=True)
    assert src.count("auto rB2 = op.get_right_input_cooperative_tensor") == 1
    assert src.count("auto rB = op.get_right_input_cooperative_tensor") == 1
    assert "op.run(sA, rB2," in src and "op.run(sA, rB," in src
    # the decode of the second step is written before the first step's matmul is issued
    assert src.index("rB2[0] = dec_mul1") < src.index("op.run(sA, rB,") or src.index("rB[0] = dec_mul1") < src.index("op.run(sA, rB2,")
    # the pass tail (k tiles not a multiple of 2 * nk) is still decoded and run
    assert "for (uint kt = KT0 + ((KT1 - KT0) /" in src
    assert src.count("{") == src.count("}")


def test_off_source_is_the_stock_source():
    on = NX.coop_source((2, 3), "mul1", *CFG[:2], CFG[2], CFG[3], pipe=True)
    off = NX.coop_source((2, 3), "mul1", *CFG[:2], CFG[2], CFG[3], pipe=False)
    assert "rB2" not in off and "PIPERB2" not in on and "PIPERB2" not in off
    assert on != off and off.count("op.run(") == 4                   # nmb 2 x two K bodies


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
@pytest.mark.parametrize("K,rows", [(2, 40), (3, 40), (4, 40), (5, 40), (6, 40), (3, 128), (3, 9), (5, 71)])
def test_pipe_is_bit_identical_over_every_bit_width(monkeypatch, K, rows):
    """Same fp16 output bits as the unpipelined kernel, on real stacked experts of bit width K and several row counts."""
    from bench_moe import synth_stack

    E, H, I = 3, 512, 768
    g = synth_stack(H, I, [K] * E, "mul1", 1)
    u = synth_stack(H, I, [K] * E, "mul1", 2)
    d = synth_stack(I, H, [K] * E, "mul1", 3)
    rng = np.random.default_rng(0)
    eids = mx.array(np.arange(E, dtype=np.int32))
    A = mx.array(NX.permute_k(rng.standard_normal((2 * E * rows, H)).astype(np.float16)))
    Ad = mx.array(NX.permute_k(rng.standard_normal((E * rows, I)).astype(np.float16)))
    cl = (np.arange(E + 1) * rows).astype(np.uint32)
    ro2 = mx.array(np.concatenate([cl[:-1], E * rows + cl]))
    ro1 = mx.array(cl)
    mx.eval(A, Ad)

    def run(pipe):
        monkeypatch.setattr(NX, "PIPE", pipe)
        NX._kc.clear()
        return (np.array(NX.expert_gemm((g, u), eids, ro2, A, 2 * E * rows, rows, CFG, True, "full")),
                np.array(NX.expert_gemm((d,), eids, ro1, Ad, E * rows, rows, CFG, True, "full")))

    a_g, a_d = run(False)
    b_g, b_d = run(True)
    assert np.array_equal(a_g.view(np.uint32), b_g.view(np.uint32))
    assert np.array_equal(a_d.view(np.uint32), b_d.view(np.uint32))
    monkeypatch.setattr(NX, "PIPE", False)
    NX._kc.clear()
