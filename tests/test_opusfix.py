"""OPUS-FIXES (review of glm-speed 046f852c): kda_nax NSG validation, hc_pf_fuse norm type + per-instance seams, spec_depth_auto ordering / ownership / crowd routing /
context-bucketed costs, the SELECT_PAR flag registration. CPU only except the one GPU test (run it as a RULE 7 micro-job)."""
import types

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import test_glm_adaptive as TGA                       # noqa: E402
import test_spec_depth_auto as TSD                    # noqa: E402
import test_spec_lookup as TSL                        # noqa: E402
from kyojin_mlx import flags as F                     # noqa: E402
from kyojin_mlx import fired as FD                    # noqa: E402
from kyojin_mlx import hc_pf_fuse as HF               # noqa: E402
from kyojin_mlx import kda_nax as K                   # noqa: E402
from kyojin_mlx import spec_depth as SD               # noqa: E402
from kyojin_mlx.serve import engine as E              # noqa: E402


@pytest.fixture(autouse=True)
def clean_audit():
    FD.reset()
    yield
    FD.reset()


# ------------------------------------------------------------------------------------------------ (1) kda_nax NSG
@pytest.mark.parametrize("n,want", [(1, (1, None)), (2, (2, None)), (4, (4, None)), (8, (4, "nsg=8")), (0, (4, "nsg=0")), (3, (4, "nsg=3")), (-2, (4, "nsg=-2")), ("x", (4, "nsg=x"))])
def test_nsg_eff(n, want):
    assert K.nsg_eff(n) == want


def test_nsg_env_garbage_is_the_shipped_4(monkeypatch):
    monkeypatch.setenv("KYOJIN_KDA_SCAN_NSG", "eight")
    assert K._env_nsg() == 0 and K.nsg_eff(K._env_nsg())[0] == 4
    monkeypatch.setattr(K, "NSG", 8)
    assert K.nsg_eff() == (4, "nsg=8")
    assert K._kernels() is K._kernels_for(K.SPLIT, 4)           # the kernels built are the NSG = 4 ones, not an 8 variant


def test_nsg_flag_still_reaches_the_module(monkeypatch):
    f = F.FLAGS["kda_scan_nsg"]
    assert f.attrs == (("kda_nax", "NSG", int),)


def test_nsg_8_is_bit_identical_to_4_and_audited():
    """GPU (micro-job): NSG = 8 used to build a kernel with Dk / NT = 0 gate channels per thread; it now runs the shipped 4 and says why."""
    if not mx.metal.is_available():
        pytest.skip("needs the GPU")
    T, H, D = 64, 2, 128
    mx.random.seed(0)
    bf = mx.bfloat16
    q = (mx.random.normal((1, T, H, D)) * 0.01).astype(bf)
    k = (mx.random.normal((1, T, H, D)) * 0.09).astype(bf)
    v = mx.random.normal((1, T, H, D)).astype(bf)
    a = (mx.random.normal((1, T, H, D)) * 2).astype(bf)
    beta = mx.sigmoid(mx.random.normal((1, T, H)).astype(bf))
    A_log = mx.log(mx.random.uniform(1, 16, (H,)))
    dtb = mx.random.normal((H, D)) * 0.5
    S0 = mx.random.normal((1, H, D, D)) * 0.01
    old = K.NSG
    try:
        outs = {}
        for n in (4, 8):
            K.set_nsg(n)
            outs[n] = K.kda_nax(q, k, v, a, beta, A_log, dtb, -5.0, S0, prep=True)
            mx.eval(outs[n])
    finally:
        K.set_nsg(old)
    assert mx.array_equal(outs[8][0], outs[4][0]).item() and mx.array_equal(outs[8][1], outs[4][1]).item()
    assert FD.STATS.get("kda_scan_nsg.fb.nsg=8", 0) >= 1


# ------------------------------------------------------------------------------------------------ (2) hc_pf_fuse norm type + per-instance seams
class GemmaRMSNorm(nn.Module):                       # a model-local RMSNorm that scales by (1 + w): same name, weight and eps as MLX's
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = mx.zeros((d,), dtype=mx.float16)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class SubRMS(nn.RMSNorm):
    def __call__(self, x):
        return super().__call__(x) * 2


def _norm_why(norm, monkeypatch):
    monkeypatch.setattr(HF, "NORM", True)
    return HF.norm_why(mx.zeros((1, 64, 4, 4096), dtype=mx.float16), norm, 4, 4096)


def test_norm_why_accepts_mlx_rmsnorm(monkeypatch):
    n = nn.RMSNorm(4096, eps=1e-5)
    n.weight = n.weight.astype(mx.float16)
    assert _norm_why(n, monkeypatch) in (None, "no_gpu")


def test_norm_why_refuses_layernorm_and_one_plus_w_norms(monkeypatch):
    ln = nn.LayerNorm(4096, eps=1e-5)
    assert _norm_why(ln, monkeypatch) == "norm_type=LayerNorm"
    assert _norm_why(GemmaRMSNorm(4096), monkeypatch) == "norm_type=GemmaRMSNorm"
    assert _norm_why(SubRMS(4096), monkeypatch) == "norm_type=SubRMS"                  # an RMSNorm whose __call__ is not MLX's
    ns = types.SimpleNamespace(weight=mx.ones((4096,), dtype=mx.float16), eps=1e-5)    # duck typing is not enough
    assert _norm_why(ns, monkeypatch) == "norm_type=SimpleNamespace"
    n = nn.RMSNorm(4096)
    n.eps = None
    assert _norm_why(n, monkeypatch) == "norm_eps"


def _hcs(n=1):
    from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection
    cfg = types.SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-5, hidden_size=64)
    return [HyperConnection(cfg) for _ in range(n)]


def test_install_acts_only_on_the_instances_of_the_glm_model(monkeypatch):
    from mlx_vlm.models.deepseek_v4 import hyper_connection as hc
    from kyojin_mlx import glm_pf_glue as G
    calls = []
    monkeypatch.setattr(hc.HyperConnection, "__call__", lambda self, x: calls.append("orig") or "stock")
    monkeypatch.setattr(hc.HyperConnection, "apply_branch", lambda self, x, norm, branch, *a, **k: calls.append("orig_ab") or "stock_ab")
    monkeypatch.setattr(hc, "hc_expand", hc.hc_expand)
    monkeypatch.setattr(hc, "_kyojin_glm_pf", False, raising=False)
    monkeypatch.setattr(G, "ENABLED", True)
    monkeypatch.setattr(G, "MIX", True)
    monkeypatch.setattr(HF, "NORM", True)
    mine, other = _hcs(2)

    class Lang(nn.Module):
        pass
    Lang.__module__ = "mlx_vlm.models.glm5_next.language"
    lm = Lang()
    lm.hc = mine
    model = types.SimpleNamespace(language_model=lm)
    assert G.install(model) >= 2
    assert G._mine(mine) and not G._mine(other)
    x = mx.zeros((1, 64, 4, 64), dtype=mx.float16)
    assert other(x) == "stock" and other.apply_branch(x, nn.RMSNorm(64), lambda z: z) == "stock_ab"      # the shared class, another model's instance: stock, no audit line
    assert not [k for k in FD.STATS if k.startswith("pf_hc")]
    calls.clear()
    mine(x)                                                                                              # the GLM instance takes the seam (fn is not a served dtype here: reported, then stock)
    assert [k for k in FD.STATS if k.startswith("pf_hc_mix")]
    again = _hcs(1)[0]
    lm.hc2 = again
    assert G.install(model) == 2 and G._mine(again)                                                      # a second GLM model on the patched class is tagged too


# ------------------------------------------------------------------------------------------------ (3) / (4) / (5) spec_depth_auto in the engine
def test_auto_runs_after_the_nondet_fix(monkeypatch):
    monkeypatch.setattr(E, "NONDET_FIX", True)
    eng = TSD._engine(monkeypatch, "1", 1)
    assert eng._plan_depth(eng.members[0]) == 4                                   # D = cap, no measured ms
    assert not any(k.startswith("spec_auto_d") and v for k, v in eng.stats.items())
    assert not eng._auto_owns_depth
    monkeypatch.setattr(E, "NONDET_FIX", False)
    assert eng._plan_depth(eng.members[0]) == 1 and eng._auto_owns_depth          # the policy's first round is d1


def test_a_crowd_of_two_keeps_its_row_cap_under_the_nondet_fix(monkeypatch):
    monkeypatch.setattr(E, "NONDET_FIX", True)
    eng = TSD._engine(monkeypatch, "1", 2)
    assert [eng._plan_depth(s) for s in eng.members] == [1, 1]


def _run_env(monkeypatch, env, **kw):
    monkeypatch.setenv(TSD.ENV, "1")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    monkeypatch.setenv("KYOJIN_GLM_MULTI_PLAIN", "0")
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(SD.COST_ENV, TSD.CHEAP)
    reqs = {"r0": ([80, 5, 9, 11, 13], {"max_new": 120})}
    return TSL.run(reqs, backend=TGA.Scripted(10 ** 9), sessions=1, mtp_depth=4, **kw)


@pytest.mark.parametrize("env", [{"KYOJIN_GLM_PLAN": "1"}, {"KYOJIN_GLM_ADAPTIVE": "1"}])
def test_glm_plan_and_adaptive_do_not_overwrite_the_auto_depth(monkeypatch, env):
    base, st0, _ = _run_env(monkeypatch, {})
    out, st, _ = _run_env(monkeypatch, env)
    assert {k: v[:2] for k, v in out.items()} == {k: v[:2] for k, v in base.items()}
    assert st["spec_auto_d4"] > 5, {k: v for k, v in st.items() if k.startswith("spec_auto")}
    assert st["spec_glm_plan_rounds"] == 0 and st.get("spec_ad_rounds", 0) == 0               # neither controller planned: auto owned the depth
    assert {k: v for k, v in st.items() if k.startswith("spec_auto")} == {k: v for k, v in st0.items() if k.startswith("spec_auto")}


def test_glm_plan_still_plans_without_auto(monkeypatch):
    monkeypatch.setenv(TSD.ENV, "0")
    monkeypatch.setenv("KYOJIN_GLM_PLAN", "1")
    monkeypatch.setenv("KYOJIN_GLM_ADAPTIVE", "0")
    monkeypatch.setenv("KYOJIN_SPEC_LOOKUP", "0")
    monkeypatch.setenv("KYOJIN_GLM_MULTI_PLAIN", "0")
    _, st, _ = TSL.run({"r0": ([80, 5, 9, 11, 13], {"max_new": 60})}, backend=TGA.Scripted(3), sessions=1, mtp_depth=4)
    assert st["spec_glm_plan_rounds"] > 0


def test_auto_routes_a_crowd_of_three_to_the_plain_tick(monkeypatch):
    for n in (1, 2, 3, 4):
        eng = TSD._engine(monkeypatch, "1", n)
        assert eng._plain_min_sessions == 3
        assert eng._plain_crowd(n) == (n >= 3)
        assert eng._glm_multi_plain
    eng = TSD._engine(monkeypatch, "0", 3)
    assert eng._plain_min_sessions == 0 and not eng._plain_crowd(3)                  # the flag off: legacy
    eng = TSD._engine(monkeypatch, "1", 3, depth_override=2)
    assert eng._plain_min_sessions == 0                                              # a fixed depth wins
    monkeypatch.setenv("KYOJIN_SPEC_PLAIN_MIN_SESSIONS", "2")
    assert TSD._engine(monkeypatch, "1", 2)._plain_min_sessions == 2                 # a lower explicit threshold is kept
    monkeypatch.setenv("KYOJIN_SPEC_PLAIN_MIN_SESSIONS", "4")
    assert TSD._engine(monkeypatch, "1", 3)._plain_min_sessions == 3                 # a higher one does not leave 3 on depth 0 through the plan
    monkeypatch.delenv("KYOJIN_SPEC_PLAIN_MIN_SESSIONS")
    assert TSD._engine(monkeypatch, "0", 3)._plain_min_sessions == 0


def test_three_sessions_run_the_plain_tick_and_keep_the_text(monkeypatch):
    base, _, _ = TSD.run(monkeypatch, "0", sessions=3, override=0)
    out, st, _ = TSD.run(monkeypatch, "1", sessions=3)
    assert TSD.toks(out) == TSD.toks(base)
    assert st["spec_multi_plain_rounds"] > 0


def test_ctx_bucket():
    assert [SD.ctx_bucket(n) for n in (1, 8192, 16384, 16385, 32768, 65536, 65537, 131072, 10 ** 6)] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_round_costs_are_tagged_by_context_bucket(monkeypatch):
    eng = TSD._engine(monkeypatch, "1", 1)
    s = eng.members[0]
    eng.plain_ms = 10.0
    for ms in (50.0, 50.0, 50.0):                                                    # a d3 round at ~8K: slow (50 ms)
        eng._k_warm.add(3)
        eng._spec_obs, eng._spec_obs_bucket = (ms, 3, 4), SD.ctx_bucket(8000)
        eng._learn_k()
    assert eng._d_ms_ctx == {0: {3: [50.0, 50.0, 50.0]}} and eng._d_ms[3] == [50.0] * 3
    seen = []
    real = SD.choose
    monkeypatch.setattr(SD, "choose", lambda st, cap, plain, d_ms, prior=None: seen.append(d_ms) or real(st, cap, plain, d_ms, prior))
    s.ids = list(range(8000))
    eng._plan_auto(s, 4)
    assert seen[-1] == {3: [50.0] * 3}                                               # at 8K its own samples
    s.ids = list(range(40000))
    eng._plan_auto(s, 4)
    assert seen[-1] == {}                                                            # at 32K: nothing measured there, the prior shape prices it


def test_the_depth_clock_restarts_when_a_crowd_runs(monkeypatch):
    eng = TSD._engine(monkeypatch, "1", 2)
    for s in eng.members:
        s.sd = SD.State()
        s.sd.last = 4
    eng._spec_tick = eng._decode_tick = lambda: None
    eng._decode_phase()
    assert [s.sd.last for s in eng.members] == [0, 0]
    solo = TSD._engine(monkeypatch, "1", 1)
    solo.members[0].sd = SD.State()
    solo.members[0].sd.last = 4
    solo._spec_tick = solo._decode_tick = lambda: None
    solo._decode_phase()
    assert solo.members[0].sd.last == 4                                              # alone: untouched
    st = SD.State()
    st.p = [0.95, 0.95, 0.95, 0.95]
    assert SD.choose(st, 4, 10.0, {}, tuple(float(x) for x in TSD.CHEAP.split(","))) == 1    # a restarted clock climbs from d1, one row per round


# ------------------------------------------------------------------------------------------------ (6) flag registration
def test_select_par_is_registered_default_off():
    f = F.FLAGS["moe_router_select_par"]
    assert f.env == "KYOJIN_MOE_ROUTER_SELECT_PAR" and f.default == "0" and f.runtime
    assert f.attrs == (("glm_moe", "PF_SELPAR", F._on),)
