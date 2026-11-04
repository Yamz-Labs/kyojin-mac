# SPDX-License-Identifier: Apache-2.0
"""dec_stage: the DS4.1 served decode block as six stages (KYOJIN_DEC_STAGE, sub-flags _IN / _ATTN / _FFN / _MOE).

The stage path must be the chain bit for bit (every piece is the arithmetic of the op it replaces), for a plain step and for verify
blocks of 2, 4 and 6 rows, with each sub-flag alone. The synthetic 2-layer model carries the served module types (EXL3 linears,
Exl3MultiLinear wo_a, MoeGather experts, the real attention shape via FX.REAL), so the same-input group, the fp8 folds, the kv chain,
the rope fuses and the fused index select all run. The dispatch census prints the chain and the stage path per pass (the served lane's
glue_fuse off and the repo default on): the lazy graph's nodes minus the view-only primitives, the number tools/count_launches.py uses.
"""
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decstage_fixture as FX  # noqa: E402

from kyojin_mlx import ablit as AB, archs, decstage as DS, dsv4_core as DC, glue_fuse as GN, rows as RW  # noqa: E402

SPEC = archs.arch_of({"model_type": "deepseek_v41"})
ROWS = [1, 2, 4, 6]


@pytest.fixture(scope="module")
def model():
    m, _ = FX.build(**FX.REAL)
    RW.enable(m, SPEC)
    return m


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """The levers this test flips live on module attributes; a test always leaves the chain's defaults (all off)."""
    yield
    monkeypatch.setattr(DS, "ENABLED", False)
    monkeypatch.setattr(DS, "GLUE", True)
    monkeypatch.setattr(GN, "ENABLED", True)
    for name in ("IN", "ATTN", "FFN", "MOE"):
        monkeypatch.setattr(DS, name, False)


def run(model, n, stage, glue=True, subs=True, glue_k=False):
    """One pass of `n` consecutive positions of one session; (logits as np, census of the graph or None).
    `subs=False` keeps the sub-flags as the test set them (the bisect). `glue_k`: the one-launch mix / expand stages (stagek), not the chain's
    arithmetic: the bit-equality tests run with them off (the other pieces are the chain's bits), test_glue_stage_* with them on."""
    GN.ENABLED = glue
    DS.ENABLED = stage
    DS.GLUE = glue_k
    if stage and subs:
        for name in ("IN", "ATTN", "FFN", "MOE"):
            setattr(DS, name, True)
    cache = RW.batch([model.make_cache()])
    p = RW.verify(model, cache, [list(range(5, 5 + n))])
    lg = p.logits
    c = FX.census(lg) if stage is not None else None
    out = np.array(lg)
    mx.eval(lg)
    p.commit([n - 1])
    return out, c


@pytest.mark.parametrize("n", ROWS)
@pytest.mark.parametrize("glue", [False, True], ids=["lane", "repos"])
def test_block_path_bit_equal(model, n, glue):
    """rows 1/2/4/6: the stage path equals the chain (logits array_equal) and runs fewer dispatches."""
    ref, cr = run(model, n, False, glue)
    got, cg = run(model, n, True, glue)
    assert np.array_equal(ref, got), int((ref != got).sum())
    print(f"\n[n={n} glue={int(glue)}] chain {cr['dispatches']} -> stage {cg['dispatches']} dispatches "
          f"({cr['dispatches'] - cg['dispatches']} fewer per 2 layers)")
    assert cg["dispatches"] < cr["dispatches"]


def test_plain_steps_bit_equal(model):
    """A plain decode step (one row per session, no verify pass) keeps the chain's logits over 6 positions, and the census of one step."""
    def steps(stage, glue):
        GN.ENABLED = glue
        DS.GLUE = False
        DS.ENABLED = stage
        if stage:
            for name in ("IN", "ATTN", "FFN", "MOE"):
                setattr(DS, name, True)
        c = RW.batch([model.make_cache()])
        out, census = [], None
        for i in range(6):
            lg = RW.step(model, c, mx.array([5 + i]))
            census = FX.census(lg) if i == 0 else census
            out.append(np.array(lg))
            mx.eval(lg)
        return out, census
    for glue in (False, True):
        ref, cr = steps(False, glue)
        got, cg = steps(True, glue)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got)), int(glue)
        print(f"\n[plain step glue={int(glue)}] chain {cr['dispatches']} -> stage {cg['dispatches']} dispatches per 2 layers")


@pytest.mark.parametrize("flag", ["IN", "ATTN", "FFN", "MOE"])
def test_sub_flags_bisect(model, flag, monkeypatch):
    """Each sub-flag alone (the other three on) still equals the chain: a stage can be bisected on the pack."""
    ref, _ = run(model, 4, False)
    monkeypatch.setattr(DS, flag, False)
    got, c = run(model, 4, True, subs=False)
    assert np.array_equal(ref, got), flag
    assert c["dispatches"] > 0


def test_attn_stage_core_pre_bit_equal(model, monkeypatch):
    """The hoisted `pre` (qr, q, kv) + stage mode of the core equal the core's own projections, after the wo_a / wo_b tail."""
    import mlx_vlm.models.deepseek_v41.language as L
    n = 4
    blk = model.language_model.layers[1]
    h = mx.broadcast_to(mx.array((np.random.default_rng(3).standard_normal((n, 1, 1, FX.REAL["hidden_size"])) * 0.2).astype(np.float16)),
                        (n, 1, 4, FX.REAL["hidden_size"]))
    pre_mix = L.make_identity_pre_mix(n, 1, 4)
    monkeypatch.setattr(DS, "IN", True)
    x, qr, q, kv, mix = DS.stage_in(blk, h, pre_mix)
    c1 = RW.batch([model.make_cache()])
    s1 = DC.stack_rows(c1[0])
    s1.begin([n])
    o1 = DC.core(blk.attn, x, s1)
    c2 = RW.batch([model.make_cache()])
    s2 = DC.stack_rows(c2[0])
    s2.begin([n])
    o2 = DC.core(blk.attn, x, s2, pre=(qr, q, kv), stage=True)
    tail = blk.attn.wo_a(o2).transpose(0, 2, 1, 3).flatten(-2)
    tail = blk.attn.wo_b(L.fake_quant_fp8_ue8m0(tail))
    mx.eval(o1, tail)
    a, b = np.array(o1), np.array(tail)
    assert np.array_equal(a, b), float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    assert mix[0].shape == (n, 1, 4)


def test_fallback_outside_the_covered_shapes(model):
    """hc_mult != 4 and more rows than the stage serves (MAX_ROWS + 1; the stage serves every verify width up to MAX_ROWS since ec9785a, a 7-row pass
    no longer falls back) return None: the caller runs the chain."""
    import mlx_vlm.models.deepseek_v41.language as L
    DS.ENABLED = True
    m2, _ = FX.build(**{**FX.REAL, "hc_mult": 2})
    RW.enable(m2, SPEC)
    cache = RW.batch([m2.make_cache()])
    st = DC.stack_rows(cache[0])
    st.begin([1])
    h = mx.zeros((1, 1, 2, FX.REAL["hidden_size"]), dtype=mx.float16)
    assert DS.block(m2.language_model.layers[0], h, L.make_identity_pre_mix(1, 1, 2), None, st) is None
    cache = RW.batch([model.make_cache()])
    st = DC.stack_rows(cache[0])
    wide = DS.MAX_ROWS + 1
    st.begin([wide])
    h7 = mx.zeros((wide, 1, 4, FX.REAL["hidden_size"]), dtype=mx.float16)
    pre7 = L.make_identity_pre_mix(wide, 1, 4)
    assert DS.block(model.language_model.layers[1], h7, pre7, None, st) is None


def test_ablit_hook_applies_in_the_stage_path(model, tmp_path, monkeypatch):
    """The refusal-direction hook sits before both expands: armed, the stage path equals the armed chain bit for bit and moves the
    output away from the unarmed model (the hook is applied, at the attention and the MoE output)."""
    n = 4
    layers = model.language_model.layers
    ref_off, _ = run(model, n, False)
    got_off, _ = run(model, n, True)
    assert np.array_equal(ref_off, got_off)
    rng = np.random.default_rng(9)
    R = np.linalg.qr(rng.standard_normal((2, FX.REAL["hidden_size"])).T)[0].T.astype(np.float32)
    n_layers = len(layers)
    wa, wm = [0.0] * n_layers, [0.0] * n_layers
    wa[1], wm[1] = 0.8, 0.7
    p = tmp_path / "edit_spec.json"
    import json
    json.dump({"hidden": FX.REAL["hidden_size"], "n_layers": n_layers, "attn_w": wa, "mlp_w": wm}, open(p, "w"))
    from safetensors.numpy import save_file
    save_file({"R": R}, str(tmp_path / "edit_spec.safetensors"))
    assert AB.attach(model, str(p)) == 1
    try:
        ref_on, _ = run(model, n, False)
        got_on, _ = run(model, n, True)
    finally:
        AB.detach(model)
    assert np.array_equal(ref_on, got_on)
    assert not np.array_equal(ref_on, ref_off), "the hook must change the logits"


def test_f32_reference_is_no_worse_than_the_chain(model):
    """T6 rule: the stage path must not move the logits away from a float32 reference of the same weights more than the chain.
    The reference swaps every EXL3 / grouped-wo_a / routed-MoE module for a float32 matmul of its decoded trellis (no packing kernel)."""
    def rel(a, b):
        return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / np.linalg.norm(b.astype(np.float64)))

    for n in (1, 4):
        chain, _ = run(model, n, False)
        stage, _ = run(model, n, True)
        restore = FX.use_f32_reference(model)
        try:
            ref, _ = run(model, n, False)
        finally:
            restore()
        ec, es = rel(chain, ref), rel(stage, ref)
        top_c = int((chain.argmax(-1) == ref.argmax(-1)).sum())
        top_s = int((stage.argmax(-1) == ref.argmax(-1)).sum())
        print(f"\n[f32 n={n}] chain rel {ec:.4g} top1 {top_c}/{n} | stage rel {es:.4g} top1 {top_s}/{n}")
        assert np.array_equal(chain, stage)
        assert es <= ec + 1e-6, (n, ec, es)
        assert top_s == top_c, (n, top_c, top_s)


def test_flag_defaults_match_the_module_literals(monkeypatch):
    """docs/cloud/LEVERS.md + flags.py convention: every dec_stage lever's module literal is its flags.py default."""
    from kyojin_mlx import flags as FL
    for name, attr in (("dec_stage", "ENABLED"), ("dec_stage_in", "IN"), ("dec_stage_attn", "ATTN"),
                       ("dec_stage_ffn", "FFN"), ("dec_stage_moe", "MOE")):
        f = FL.FLAGS[name]
        module, attribute = f.attrs[0][:2]
        assert module == "decstage" and attribute == attr, (name, f.attrs)
        assert f.default in ("0", "1") and f.runtime
    assert FL.FLAGS["dec_stage"].default == "0"
    for name in ("dec_stage_in", "dec_stage_attn", "dec_stage_ffn", "dec_stage_moe"):
        assert FL.FLAGS[name].default == "0"


def test_glue_stage_is_row_independent_and_no_worse_than_the_chain_per_stage(model):
    """The one-launch mix / expand stages (stagek) on the whole block: a row has the bits it has alone whatever the width of the pass (the plain
    step, the verify block and the replay agree), and the logits stay within the chain's distance of the float32 reference (per-stage accuracy
    against float64 is in tests/test_dec_stage_k.py: the end logits of random weights go through a non-contractive sinkhorn, so this only bounds them)."""
    def rel(a, b):
        return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)) / np.linalg.norm(b.astype(np.float64)))
    solo, _ = run(model, 1, True, glue_k=True)
    for n in (2, 3):                                                   # the GEMV band, where the linears keep a row's bits (conftest: HALF_ROWS = 0)
        wide, _ = run(model, n, True, glue_k=True)
        assert np.array_equal(wide[:1], solo[:1]), n
    chain, _ = run(model, 4, False)
    stage, _ = run(model, 4, True, glue_k=True)
    restore = FX.use_f32_reference(model)
    try:
        ref, _ = run(model, 4, False)
    finally:
        restore()
    ec, es = rel(chain, ref), rel(stage, ref)
    print(f"\n[glue stage vs f32] chain rel {ec:.4g} | stage rel {es:.4g}")
    assert es <= 1.5 * ec + 1e-6
