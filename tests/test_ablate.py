"""KYOJIN_ABLATE (kyojin_mlx.ablate, flag `ablate`): per-op-class stand-ins that price the exposed cost of a decode-step class.

Pins (1) the flag parses / validates / round-trips through flags.set and is off by default, (2) with the flag off the batched step and the verify rows are bit
identical to the base (before any use and after a set / reset), (3) each class runs end to end on a tiny DS4.1 model (batched core, `rows.enable`: plain step and verify
rows) and its stand-in site is reached (HITS) where the mini model carries that site, (4) the sites the mini model does not carry (MoeGather stages, the sumoe
routed call, the shared-expert impl, the fused mHC functions, the decensor projection, the fused-attention solo core) return the shapes / dtypes their consumers read."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, os.path.dirname(__file__))

from kyojin_mlx import ablate as AZ                                                     # noqa: E402
from kyojin_mlx import flags as FL                                                      # noqa: E402

needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def _off():
    env = os.environ.get(AZ.ENV)
    was = AZ.ACTIVE
    AZ.HITS.clear()
    yield
    AZ.ACTIVE = was
    AZ.HITS.clear()
    if env is None:
        os.environ.pop(AZ.ENV, None)
    else:
        os.environ[AZ.ENV] = env


@pytest.fixture(autouse=True)
def _pristine_fuse():
    """rows.enable installs the process-wide fused replacements (fuse.install): put every replaced function back so later test files see a fresh process."""
    try:
        import importlib
        from kyojin_mlx import fuse
        from kyojin_mlx.archs import DS41
    except Exception:
        yield
        return
    saved, before = [], set(fuse._installed)
    try:
        for path, attr, impl in DS41.fuse.funcs:
            modpath, _, cls = path.partition(":")
            owner = importlib.import_module(modpath)
            owner = getattr(owner, cls) if cls else owner
            saved.append((owner, attr, owner.__dict__[attr] if cls else getattr(owner, attr)))
    except Exception:
        saved = []
    yield
    for owner, attr, fn in saved:
        setattr(owner, attr, fn)
    fuse._installed.intersection_update(before)


# --------------------------------------------------------------------------------------------------------------- flag
def test_flag_is_off_by_default_and_validated():
    f = FL.FLAGS["ablate"]
    assert f.env == "KYOJIN_ABLATE" and f.default == "" and f.kind == "str" and f.runtime
    assert AZ.parse("") == frozenset() and AZ.parse(None) == frozenset()
    assert AZ.parse(" Norm , attn,,") == {"norm", "attn"}
    assert AZ.normalize("attn,moe_routed") == "moe_routed,attn"                       # table order
    with pytest.raises(ValueError):
        AZ.parse("norm,nope")
    with pytest.raises(ValueError):
        FL.set("ablate", "nope")                                                      # refused before anything is applied
    assert set(AZ.CLASSES) == {"moe_routed", "moe_shared", "router", "moe_glue", "mhc", "norm", "attn", "idx", "ablit"}


def test_flag_set_writes_env_active_and_bumps_gen():
    g = FL.GEN
    FL.set("ablate", "router, mhc")
    assert os.environ[AZ.ENV] == "router,mhc" and AZ.ACTIVE == {"router", "mhc"} and FL.GEN == g + 1
    assert FL.get_all()["ablate"]["value"] == "router,mhc"
    FL.set("ablate", "")
    assert os.environ[AZ.ENV] == "" and AZ.ACTIVE == frozenset() and not AZ.chain_only()


def test_chain_only_names_the_chain_classes():
    AZ.ACTIVE = frozenset({"attn", "router"})
    assert not AZ.chain_only()
    AZ.ACTIVE = frozenset({"attn", "norm"})
    assert AZ.chain_only()
    AZ.ACTIVE = frozenset({"mhc"})
    assert AZ.chain_only()


# --------------------------------------------------------------------------------------------------------------- the tiny model, end to end
def mini_model(seed=3):
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    import mini_ds41 as M
    from mlx.utils import tree_map
    from kyojin_mlx.archs import DS41
    from kyojin_mlx import rows as R

    lm, cfg = M.make(head_dim=128, dtype=mx.float16, seed=seed)
    lm.update(tree_map(lambda a: a.astype(mx.float16), lm.parameters()))
    mx.eval(lm.parameters())

    class Wrap:
        def __init__(self, lm):
            self.language_model = lm

        def __getattr__(self, n):
            return getattr(self.language_model, n)

    m = Wrap(lm)
    R.enable(m, DS41, grouped=False)
    return m, R, M


def decode(m, R, M, flag, steps=3):
    """Prefill with the flag off, then `steps` plain steps and one 3-row verify pass (all drafts rejected) under `flag`. -> (step logits [steps, V], verify logits)."""
    FL.set("ablate", "")
    cache, lg = R.prefill(m, M.tokens(200))
    bc = R.batch([cache])
    tok = int(np.array(lg).argmax(-1)[0])
    FL.set("ablate", flag)
    out = []
    for _ in range(steps):
        l = R.step(m, bc, mx.array([tok]))
        mx.eval(l)
        out.append(np.array(l))
        tok = 11
    p = R.verify(m, bc, [[tok, 12, 13]])
    mx.eval(p.logits)
    p.commit([0])
    FL.set("ablate", "")
    return np.concatenate(out, 0), np.array(p.logits)


@needs_metal
def test_off_is_bit_identical_to_the_base():
    m, R, M = mini_model()
    a, va = decode(m, R, M, "")
    FL.set("ablate", "norm,attn")                                                      # used, then back to off
    FL.set("ablate", "")
    b, vb = decode(m, R, M, "")
    assert np.array_equal(a, b) and np.array_equal(va, vb)
    assert not AZ.HITS


@needs_metal
@pytest.mark.parametrize("cls", ["router", "mhc", "norm", "attn", "idx"])
def test_class_runs_end_to_end_and_its_site_is_reached(cls):
    m, R, M = mini_model()
    base, vbase = decode(m, R, M, "")
    AZ.HITS.clear()
    got, vgot = decode(m, R, M, cls)
    assert got.shape == base.shape and vgot.shape == vbase.shape
    assert np.isfinite(got).all() and np.isfinite(vgot).all()
    assert AZ.HITS[cls] > 0, dict(AZ.HITS)
    assert not np.array_equal(got, base)                                               # a stand-in really replaced work


@needs_metal
def test_all_classes_at_once_run():
    m, R, M = mini_model()
    got, vgot = decode(m, R, M, ",".join(AZ.CLASSES))
    assert np.isfinite(got).all() and np.isfinite(vgot).all()


@needs_metal
def test_attn_leaves_the_cache_untouched():
    m, R, M = mini_model()
    FL.set("ablate", "")
    cache, lg = R.prefill(m, M.tokens(200))
    bc = R.batch([cache])
    from kyojin_mlx import dsv4_core as DC
    st = DC.stack_rows(bc[0])
    snap = lambda: [(None if a is None else np.array(a)) for lst in (st.win, st.cmp, st.keys, st.kvs, st.scs) for a in lst]
    before = snap()
    assert any(b is not None for b in before)
    FL.set("ablate", "attn")
    for _ in range(2):
        mx.eval(R.step(m, bc, mx.array([5])))
    FL.set("ablate", "")
    after = snap()
    assert all((a is None and b is None) or np.array_equal(a, b) for a, b in zip(before, after))


# --------------------------------------------------------------------------------------------------------------- router stand-in
def test_fixed_route_shapes_and_dtypes():
    ids, w = AZ.fixed_route(3, 6, 16, 2.5)
    assert ids.shape == (3, 1, 6) and ids.dtype == mx.int32 and w.shape == (3, 1, 6) and w.dtype == mx.float32
    i = np.array(ids)
    assert (0 <= i).all() and (i < 16).all() and len(set(i[0, 0])) == 6 and not set(i[0, 0]) & set(i[1, 0])
    assert np.allclose(np.array(w), 2.5 / 6)


# --------------------------------------------------------------------------------------------------------------- mHC stand-ins
@needs_metal
def test_fused_mhc_functions_use_mean_and_add():
    from kyojin_mlx import fuse as F
    pre_f = F._impl_mhc_pre(lambda x, p: pytest.fail("stock pre must not run"))
    exp_f = F._impl_mhc_expand(lambda *a: pytest.fail("stock expand must not run"))
    x = (mx.random.normal((2, 1, 4, 64)) * 3).astype(mx.float16)
    y = (mx.random.normal((2, 1, 64)) * 3).astype(mx.float16)
    z = mx.zeros((2, 1, 4), dtype=mx.float32)
    AZ.ACTIVE = frozenset({"mhc"})
    p = pre_f(x, z)
    assert p.shape == (2, 1, 64) and p.dtype == mx.float16
    assert np.allclose(np.array(p, dtype=np.float32), np.array(x.astype(mx.float32).mean(axis=2)), atol=2e-2)
    F._last = (None, None)
    e = exp_f(y, x, z, mx.zeros((2, 1, 4, 4), dtype=mx.float32))
    assert e.shape == (2, 1, 4, 64) and e.dtype == mx.float16 and F._last == (None, None)
    assert np.allclose(np.array(e, dtype=np.float32), np.array(x, dtype=np.float32) + np.array(y, dtype=np.float32)[:, :, None], atol=2e-2)
    big = mx.full((1, 1, 64), 60000, dtype=mx.float16)
    assert np.isfinite(np.array(exp_f(big, mx.full((1, 1, 4, 64), 60000, dtype=mx.float16), z[:1], mx.zeros((1, 1, 4, 4))), dtype=np.float32)).all()   # the fp16 saturation stays
    pr, po, cb = AZ.mhc_coeffs(2, 4)
    assert pr.shape == (2, 1, 4) and po.shape == (2, 1, 4) and cb.shape == (2, 1, 4, 4)


# --------------------------------------------------------------------------------------------------------------- norm
def test_norm_is_identity_on_decode_rows_only():
    import mlx.nn as nn
    n = nn.RMSNorm(64)
    AZ.ACTIVE = AZ.configure("norm")
    x = mx.random.normal((2, 1, 64))
    assert np.array_equal(np.array(n(x)), np.array(x))
    pf = mx.random.normal((1, 5, 64))
    assert not np.array_equal(np.array(n(pf)), np.array(pf))                           # prompt rows keep the real norm
    AZ.ACTIVE = frozenset()
    assert not np.array_equal(np.array(n(x)), np.array(x))


# --------------------------------------------------------------------------------------------------------------- routed / shared (patch._routed, pf_glue shared impl)
@needs_metal
def test_moe_routed_and_shared_in_the_routed_call():
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    from kyojin_mlx import patch as P
    calls = []

    class Moe:
        topk = 2

        def __call__(self, x, ids, w, fq8=False, max_pairs=None):
            calls.append("routed")
            return mx.ones((x.shape[0], x.shape[1]), dtype=mx.float32)

    class Msh:
        def __call__(self, x, ids, w, fq8=False):
            calls.append("sumoe")
            return mx.ones((x.shape[0], x.shape[1]), dtype=mx.float32) * 2

    dense = SimpleNamespace(dense=lambda x: mx.ones(x.shape, dtype=mx.float16) * 3)
    me = SimpleNamespace(msh=Msh(), moe=Moe(), sumoe_takes=lambda rows: True, __dict__={"_shared_stub": dense})
    me.__dict__["_shared_stub"] = dense
    x = mx.zeros((2, 1, 128), dtype=mx.float16)
    ids = mx.zeros((2, 1, 2), dtype=mx.int32)
    w = mx.ones((2, 1, 2), dtype=mx.float32)

    def run():
        calls.clear()
        return np.array(P._routed(me, x, ids, w, None))

    assert (run() == 2).all() and calls == ["sumoe"]                                   # off: the fused routed + shared launches
    AZ.ACTIVE = frozenset({"moe_routed"})
    y = run()
    assert y.shape == (2, 128) and (y == 3).all() and calls == []                      # routed zeros + the dense shared chain
    AZ.ACTIVE = frozenset({"moe_routed", "moe_shared"})
    assert (run() == 0).all() and calls == []
    AZ.ACTIVE = frozenset({"moe_shared"})
    assert (run() == 1).all() and calls == ["routed"]                                  # routed-only launches (no shared slot)
    me.msh = None
    AZ.ACTIVE = frozenset({"moe_routed"})
    assert (run() == 0).all() and calls == []


@needs_metal
def test_shared_expert_impl_returns_zeros(monkeypatch):
    pytest.importorskip("mlx_vlm.models.deepseek_v41")
    from kyojin_mlx import pf_glue as PG
    monkeypatch.setattr(PG, "SHARED_MLP", False)                                       # off: the stock call (the stub `orig`)
    f = PG._impl_shared_mlp(lambda self, x: mx.ones(x.shape, dtype=x.dtype))
    for dt in (mx.float16, mx.bfloat16):
        x = mx.ones((2, 1, 64), dtype=dt)
        assert (np.array(f(None, x).astype(mx.float32)) == 1).all()
        AZ.ACTIVE = frozenset({"moe_shared"})
        y = f(None, x)
        assert y.shape == x.shape and y.dtype == dt and (np.array(y.astype(mx.float32)) == 0).all()
        AZ.ACTIVE = frozenset()


# --------------------------------------------------------------------------------------------------------------- moe_glue (MoeGather stage methods)
@needs_metal
def test_moe_glue_stages_return_the_shapes_the_gemvs_read():
    from kyojin_mlx import moe as M
    g = SimpleNamespace(nout=256, kin=512)
    d = SimpleNamespace(nout=512, kin=256)
    me = SimpleNamespace(topk=3, g=g, d=d)
    AZ.ACTIVE = frozenset({"moe_glue"})
    x = mx.zeros((2, 512), dtype=mx.float16)
    ids = mx.zeros((6,), dtype=mx.int32)
    assert M.MoeGather.rot_gu(me, x, ids).shape == (12, 512) and M.MoeGather.rot_gu(me, x, ids).dtype == mx.float32
    assert M.MoeGather.rot_gu(me, x, ids, half=True).dtype == mx.float16
    assert M.MoeGather.mid(me, None, ids, None, 2).shape == (6, 256)
    assert M.MoeGather.mid(me, None, ids, None, 2, half=True).dtype == mx.float16
    f = M.MoeGather.final(me, None, ids, None, 2)
    assert f.shape == (2, 512) and f.dtype == mx.float32
    assert M.MoeGather.final_par(me, None, ids, None, 2).shape == (2, 512)
    assert AZ.HITS["moe_glue"] == 7


# --------------------------------------------------------------------------------------------------------------- ablit / fused-attention solo core
@needs_metal
def test_ablit_projection_is_identity():
    from kyojin_mlx import ablit as AB
    y = mx.random.normal((2, 1, 128)).astype(mx.float16)
    R = mx.random.normal((1, 128))
    R = R / mx.sqrt((R * R).sum())
    ref = np.array(AB.project(y, 0.8, R), dtype=np.float32)
    assert not np.array_equal(ref, np.array(y, dtype=np.float32))
    AZ.ACTIVE = frozenset({"ablit"})
    assert np.array_equal(np.array(AB.project(y, 0.8, R), dtype=np.float32), np.array(y, dtype=np.float32))
    assert AZ.HITS["ablit"] == 1


def test_solo_attention_core_returns_the_wo_a_input_shape():
    from kyojin_mlx import fuse_attn as FA
    attn = SimpleNamespace(o_groups=2, n_heads=8, head_dim=64)
    AZ.ACTIVE = frozenset({"attn"})
    o, out = FA.core(attn, None, None, mx.zeros((1, 1, 32), dtype=mx.float16), None, None, None)
    assert out is None and o.shape == (1, 2, 1, 4 * 64) and o.dtype == mx.float16
