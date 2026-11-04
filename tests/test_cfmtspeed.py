"""Flag cfmtspeed (KYOJIN_CFMTSPEED): same-input groups of foreign linears (one concatenated matrix, one launch, per-row global scale) and the E2M1 table decode.

CPU half (always runs, mlx on CPU is enough): the concatenated pack is the members' bytes; the group's dequant is, row for row, the member's (bit for bit);
the table equals the bit trick on all 16 codes; with the flag off the kernel sources are the ones of 3d5fd49; find_groups / group_same_input / GroupMember
plumbing on a tiny model with the Metal kernels replaced by their numpy reference (`_ref_kern`).
Mac half (needs_gpu): the real kernels, flag on vs off, array_equal at R = 1, 4, 9, 16, 24, 70, for every kernel format."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_metal as FM, foreign_pack as FP, foreign_synth as SY

RNG = lambda s=0: np.random.default_rng(s)
FMTS = ("q4_0", "iq4_nl", "q4_k", "nvfp4", "mxfp4")
TYPE = {"q4_0": "Q4_0", "iq4_nl": "IQ4_NL", "q4_k": "Q4_K", "nvfp4": "NVFP4", "mxfp4": "MXFP4"}


def _packs(fmt, ns, k=512, seed=0):
    ps = []
    for i, n in enumerate(ns):
        p = GF.to_pack(TYPE[fmt], SY.random_blocks(TYPE[fmt], n, k, RNG(seed + i)), k)
        if fmt == "nvfp4":
            p.gscale = np.float32(0.25 * (i + 1) + 0.0625)               # members carry different per-tensor scales
        ps.append(p)
    return ps


@pytest.mark.parametrize("fmt", FMTS)
def test_concat_rows_is_the_members_bytes_and_dequant(fmt):
    ps = _packs(fmt, (16, 32, 48))
    pk, rg = FP.concat_rows(ps)
    assert pk.n == 96 and rg.dtype == np.float32 and rg.shape == (96,)
    assert np.array_equal(pk.qs, np.concatenate([p.qs for p in ps])) and np.array_equal(pk.meta, np.concatenate([p.meta for p in ps]))
    want = np.concatenate([FP.dequant(p, g=False) * np.float32(p.gscale if fmt == "nvfp4" else 1.0) for p in ps])
    got = FP.dequant(pk, g=False) * rg[:, None]
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    a = 0
    for p in ps:                                                          # the reference reader on the group = the reference reader on each member
        assert np.array_equal(FP.dequant(pk, g=False)[a:a + p.n], FP.dequant(p, g=False))
        a += p.n


def test_concat_rows_refuses_mixed_members():
    a, b = _packs("nvfp4", (16,))[0], _packs("mxfp4", (16,))[0]
    with pytest.raises(ValueError):
        FP.concat_rows([a, b])
    with pytest.raises(ValueError):
        FP.concat_rows([_packs("nvfp4", (16,), k=512)[0], _packs("nvfp4", (16,), k=1024)[0]])
    with pytest.raises(ValueError):
        FP.concat_rows([FP.Pack("dense", 4, 4, np.zeros((4, 4), np.float16))])


def test_e2m1_table_equals_bit_trick_for_all_codes():
    import re
    bits = FP.fp4v_bits(np.arange(16)).view(np.uint32)
    assert np.array_equal(bits, FP.E2M1.view(np.uint32))                  # incl. -0.0 at code 8
    tab = re.search(r"E2M1T\[16\] = \{(.*?)\}", FM.FP4V_LUT)[1]
    vals = np.array([float(x.strip().rstrip("f")) for x in tab.split(",")], np.float32)
    assert np.array_equal(vals.view(np.uint32), bits)
    assert FM.FP4V_BITS in FM.HDR and FM.FP4V_LUT in FM.HDR_LUT and FM.FP4V_BITS not in FM.HDR_LUT


@pytest.mark.parametrize("fmt", FMTS)
def test_flag_off_sources_are_untouched_and_variants_differ_only_where_stated(fmt):
    for kind in ("gemv", "mma"):
        assert FM.header(fmt, kind) == (FM.HDR if kind == "gemv" else FM.MMA_HDR)
    base_g, base_m, base_d = FM.gemv_source(fmt, 2), FM.mma_source(fmt, 1), FM.dequant_source(fmt)
    assert FM.gemv_source(fmt, 2, False, False) == base_g and FM.mma_source(fmt, 1, False) == base_m and FM.dequant_source(fmt, False) == base_d
    for src, row in ((FM.gemv_source(fmt, 2, False, True), "gs[rbase + j]"), (FM.mma_source(fmt, 1, True), "gs[nn]"), (FM.dequant_source(fmt, True), "gs[rn]")):
        assert row in src and src.count("{") == src.count("}") and src.count("(") == src.count(")")
    assert FM.gemv_source(fmt, 2, False, True).replace("gs[rbase + j]", "GS") == base_g
    assert FM.mma_source(fmt, 1, True).replace("gs[nn]", "GS") == base_m
    assert FM.dequant_source(fmt, True).replace("gs[rn]", "GS") == base_d
    with pytest.raises(AssertionError):
        FM.gemv_source(fmt, 2, True, True)                                # a routed-expert gather keeps one scale per expert


def test_flag_is_registered_default_off_not_runtime():
    from kyojin_mlx import flags
    f = flags.FLAGS["cfmtspeed"]
    assert f.env == "KYOJIN_CFMTSPEED" and f.default == "0" and not f.runtime and f.reason


def test_find_groups():
    calls = [(1, 10), (1, 11), (1, 12), (2, 13), (3, 14), (3, 15), (4, 16), (5, 16)]    # module 16 is called twice: left alone
    names = {i: f"m{i}" for i in range(10, 17)}
    assert find_groups_(calls, names) == [["m10", "m11", "m12"], ["m14", "m15"]]
    assert find_groups_([(1, 10), (1, 10)], names) == []


def find_groups_(calls, names):
    from kyojin_mlx import foreign_linear as FL
    return FL.find_groups(calls, names)


# ------------------------------------------------------------------------------------------------------------------ plumbing on CPU: kernels replaced by numpy
def _ref_kern(monkeypatch):
    """Replace the three Metal kernels by their float32 reference with the same call shape: y = half(x @ dequant(rows, g=False)^T * gs[row or 0])."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    calls = []

    def mk(kind, fmt, rowgs):
        def run(inputs, template, **kw):
            t = dict(template)
            n, k = t["N"], t["KIN"]
            if kind == "deq":
                qs, mt, gs = (np.array(a) for a in inputs)
                x = None
            else:
                x, qs, mt, gs = (np.array(a) for a in inputs)
            pk = FP.Pack(fmt, n, k, np.ascontiguousarray(qs).view(np.uint8).reshape(n, k // 2), mt, 1.0)
            w = FP.dequant(pk, g=False)
            g = gs.astype(np.float32) if rowgs else np.full(n, gs[0], np.float32)
            calls.append((kind, fmt, rowgs, n))
            if kind == "deq":
                return [mx.array((w * g[:, None]).astype(np.float16))]
            if kind == "mma":
                idx = FP.xperm_index(fmt, k)
                xo = np.zeros_like(x)
                xo[:, idx] = x
                x = xo
            return [mx.array(((x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32) * g[None, :]).astype(np.float16))]
        return run

    monkeypatch.setattr(FL, "_gemv_kern", lambda fmt, nbr, gather, rowgs=False, lut=False: mk("gemv", fmt, rowgs))
    monkeypatch.setattr(FL, "_mma_kern", lambda fmt, ntn, rowgs=False, lut=False: mk("mma", fmt, rowgs))
    monkeypatch.setattr(FL, "_deq_kern", lambda fmt, rowgs=False, lut=False: mk("deq", fmt, rowgs))
    return calls


@pytest.mark.parametrize("fmt", ("nvfp4", "q4_k", "mxfp4"))
@pytest.mark.parametrize("R", (1, 4, 9, 16, 24, 70))
def test_group_member_outputs_equal_the_separate_linears(monkeypatch, fmt, R):
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    monkeypatch.setenv("KYOJIN_F4COPY", "0")                                    # the assert below counts the stored path's decode; test_f4copy covers the copy
    calls = _ref_kern(monkeypatch)
    ps = _packs(fmt, (32, 64, 16))
    k = 512
    lins = [FL.ForeignLinear(p) for p in ps]
    x = mx.array(RNG(3).standard_normal((R, k)).astype(np.float16))
    want = [np.array(l(x)) for l in lins]
    dev = FL.DevPack.grouped([l.dev for l in lins])
    pk, rg = FP.concat_rows(ps)
    assert np.array_equal(np.array(dev.gs), rg) and dev.n == 112 and dev.rowgs
    cuts = [0, 32, 96, 112]
    grp = FL._Group(dev, cuts)
    mem = [FL.GroupMember(grp, i, l) for i, l in enumerate(lins)]
    n0 = len(calls)
    got = [np.array(m(x)) for m in mem]
    for w, g in zip(want, got):
        assert w.dtype == g.dtype and np.array_equal(w, g)
    assert len([c for c in calls[n0:] if c[0] != "deq"]) <= max(1, -(-R // 16)) and (R > 32) == any(c[0] == "deq" for c in calls[n0:])      # the group is ONE product (chunks of 16 rows from 17), not one per member


def test_group_same_input_on_a_tiny_model(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from kyojin_mlx import foreign_linear as FL
    calls = _ref_kern(monkeypatch)
    pq, pk_, pv, po = _packs("nvfp4", (32, 16, 16, 64))
    bias = mx.array(RNG(5).standard_normal(16).astype(np.float16))

    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj, s.k_proj, s.v_proj, s.o_proj = FL.ForeignLinear(pq), FL.ForeignLinear(pk_), FL.ForeignLinear(pv, bias), FL.ForeignLinear(po)

        def __call__(s, x):
            return s.o_proj(mx.tile(mx.concatenate([s.q_proj(x), s.k_proj(x), s.v_proj(x)], axis=-1), (1, 1, 8)))

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.layers = [Attn(), Attn()]

        def __call__(s, tok):
            x = mx.ones((*tok.shape, 512), mx.float16) * 0.01
            for l in s.layers:
                x = mx.broadcast_to(l(x)[..., :1], (*tok.shape, 512))
            return x

    m = M()
    x = mx.array(RNG(6).standard_normal((1, 3, 512)).astype(np.float16))
    ref = [np.array(l(x)) for l in m.layers]
    logs = []
    st = FL.group_same_input(m, logs.append)
    assert st["groups"] == 2 and st["members"] == 6 and st["launches_saved"] == 4, logs      # q, k, v of each layer; o_proj reads another input
    for l, r in zip(m.layers, ref):
        assert type(l.q_proj) is FL.GroupMember and type(l.o_proj) is FL.ForeignLinear and l.q_proj._g is l.k_proj._g is l.v_proj._g
        assert np.array_equal(np.array(l(x)), r)
    assert FL.cfmt() is False


def test_group_ok_refuses_what_would_change_a_row():
    from kyojin_mlx import foreign_linear as FL
    a, b = (FL.ForeignLinear(p) for p in _packs("nvfp4", (32, 16)))
    assert FL.group_ok([a, b])
    c = FL.ForeignLinear(_packs("mxfp4", (16,))[0])
    assert not FL.group_ok([a, c])                                        # two formats
    d = FL.ForeignLinear(FP.Pack("dense", 16, 512, np.zeros((16, 512), np.float16)))
    assert not FL.group_ok([a, d])                                        # dense (FP8 / Q8_0 / Q6_K decoded to fp16): left alone
    e = FL.ForeignLinear(_packs("nvfp4", (24,))[0])                       # 24 % 16 != 0: it takes the MMA kernel alone, the group would take GEMV
    assert not FL.group_ok([a, e])


# ------------------------------------------------------------------------------------------------------------------ Mac only
def _gpu():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu(), reason="no Metal GPU")


@needs_gpu
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("R", (1, 4, 9, 16, 24, 70))
def test_metal_group_and_table_decode_are_bit_identical(monkeypatch, fmt, R):
    """flag off: separate ForeignLinears. flag on: one GroupMember set + table decode. Every output array_equal."""
    import mlx.core as mx
    from kyojin_mlx import foreign_linear as FL
    ps = _packs(fmt, (64, 128, 32), k=1024)
    x = mx.array(RNG(9).standard_normal((R, 1024)).astype(np.float16))
    monkeypatch.setenv("KYOJIN_CFMTSPEED", "0")
    want = [np.array(FL.ForeignLinear(p)(x)) for p in ps]
    wdeq = np.array(FL.DevPack(ps[0]).dequant())
    monkeypatch.setenv("KYOJIN_CFMTSPEED", "1")
    lins = [FL.ForeignLinear(p) for p in ps]
    grp = FL._Group(FL.DevPack.grouped([l.dev for l in lins]), [0, 64, 192, 224])
    got = [np.array(FL.GroupMember(grp, i, l)(x)) for i, l in enumerate(lins)]
    for w, g in zip(want, got):
        assert np.array_equal(w, g)
    for fn in ("_gemv", "_mma"):                                          # table decode alone, ungrouped
        if fn == "_gemv" and (R > 16 or not lins[0].dev.gemv_ok()):
            continue
        monkeypatch.setenv("KYOJIN_CFMTSPEED", "0")
        a = np.array(getattr(FL.DevPack(ps[0]), fn)(x))
        monkeypatch.setenv("KYOJIN_CFMTSPEED", "1")
        b = np.array(getattr(FL.DevPack(ps[0]), fn)(x))
        assert np.array_equal(a, b), fn
    assert np.array_equal(wdeq, np.array(FL.DevPack(ps[0]).dequant()))
