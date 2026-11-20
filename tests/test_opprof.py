"""tools/opprof.py on tiny synthetic models (CPU, random weights): op registry and byte accounting, class hooks (pass-through, restore, ablation stubs),
the exact launch census, expert union, the GPU gate, and the whole decode + prefill pipeline into the JSON / Markdown report. No timing number is asserted
(CPU, shared machine): only structure, invariants and the identities between columns."""
import collections
import fcntl
import json
import os

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_lm")
from tools import opprof as OP  # noqa: E402

V, E, K = 200, 8, 2


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("KYOJIN_WIRED_MB", "0")
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _moe_backend():
    from mlx_lm.models import qwen3_moe
    from kyojin_mlx.serve import backends as B
    args = qwen3_moe.ModelArgs(model_type="qwen3_moe", hidden_size=64, num_hidden_layers=4, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=16, rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, num_experts=E, num_experts_per_tok=K,
                               decoder_sparse_step=1, mlp_only_layers=[0], moe_intermediate_size=32, norm_topk_prob=True, tie_word_embeddings=False)
    m = qwen3_moe.Model(args)
    mx.eval(m.parameters())
    cfg = dict(model_type="qwen3_moe", num_experts=E, num_experts_per_tok=K, vocab_size=V)
    return B.GenericBackend(m), cfg


def _dense_tied_backend():
    from mlx_lm.models import qwen3
    from kyojin_mlx.serve import backends as B
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=True)
    m = qwen3.Model(args)
    mx.eval(m.parameters())
    return B.GenericBackend(m), dict(model_type="qwen3", vocab_size=V)


@pytest.fixture(scope="module")
def moe():
    return _moe_backend()


def _fast_opts(**kw):
    d = dict(rows=(1, 2), ctx=48, prefill=64, chunk=32, arms=("decode", "prefill"), blocks=2, steps=3, bw_gibs=(0.01,))
    d.update(kw)
    return OP.Opts(**d)


def test_registry_groups_layers_by_kind_and_counts_bytes(moe):
    be, cfg = moe
    reg = OP.Registry(be.model, cfg)
    assert len(reg.layers) == 4
    assert collections.Counter(reg.layers) == {"attn+mlp": 1, "attn+moe": 3}
    assert reg.n_experts == E and reg.top_k == K
    gp = reg.ops["attn+moe:mlp.switch_mlp.gate_proj"]
    assert gp.layers == 3 and gp.expert and gp.E == E
    one = be.model.model.layers[1].mlp.switch_mlp.gate_proj
    assert gp.stacked == 3 * one.weight.nbytes
    # R=1 reads K experts of E; R large touches (almost) all
    assert OP.op_bytes(gp, 1, reg, distinct=K) == pytest.approx(gp.stacked * K / E)
    assert OP.op_bytes(gp, 1, reg) == pytest.approx(gp.stacked * K / E)
    assert OP.op_bytes(gp, 64, reg) > 0.99 * gp.stacked
    router = reg.ops["attn+moe:mlp.gate"]
    assert router.router and not router.expert
    emb = next(op for k, op in reg.ops.items() if k.endswith("embed_tokens") and not k.endswith("as_linear"))
    assert emb.gather and OP.op_bytes(emb, 3, reg) == pytest.approx(emb.fixed * 3 / V)
    assert reg.ops["layer.attn+moe"].parent is None and reg.ops["attn+moe:mlp"].parent == "layer.attn+moe"
    assert reg.ops["attn+moe:mlp.switch_mlp"].parent == "attn+moe:mlp"
    assert any(op.trunk for op in reg.ops.values()) and not any(op.trunk for op in reg.ops.values() if op.kind != "top")
    mix = [k for k, op in reg.ops.items() if op.mixer]
    assert sorted(mix) == ["attn+mlp:self_attn", "attn+moe:self_attn"]
    # flops: a dense linear is 2 * n * k per row; experts multiply by top_k
    q = reg.ops["attn+moe:self_attn.q_proj"]
    assert q.flops_row == pytest.approx(3 * 2 * 64 * 64)
    assert gp.flops_row == pytest.approx(3 * 2 * 32 * 64 * K)


def test_tied_head_gets_an_as_linear_op():
    be, cfg = _dense_tied_backend()
    reg = OP.Registry(be.model, cfg)
    heads = [k for k in reg.ops if k.endswith(".as_linear")]
    assert len(heads) == 1
    op = reg.ops[heads[0]]
    assert op.meth == "as_linear" and op.fixed == be.model.model.embed_tokens.weight.nbytes and not op.gather
    hk = OP.Hooks(reg)
    with hk:
        be.model.model.embed_tokens.as_linear     # still reachable while wrapped
        c = be.make_cache()
        be.prefill_chunk(c, list(range(5, 20)))
        bc = be.batch([be.fork(c)])
        hk.mode = "record"
        mx.eval(be.step(bc, mx.array([3])))
    assert hk.calls[heads[0]] == 1


def test_hooks_are_passthrough_and_restored(moe):
    be, cfg = moe
    reg = OP.Registry(be.model, cfg)
    before = {(c, m): c.__dict__.get(m) for c, m in reg.classes()}
    c = be.make_cache()
    lg0 = be.prefill_chunk(c, list(range(5, 30)))
    base = be.step(be.batch([be.fork(c)]), mx.array([7]))
    mx.eval(base)
    hk = OP.Hooks(reg)
    with hk:
        for mode in (None, "host", "sync", "route", "launch", "record"):
            hk.mode = mode
            y = be.step(be.batch([be.fork(c)]), mx.array([7]))
            if mode == "launch":
                hk.launch_flush([y])
            mx.eval(y)
            assert mx.array_equal(y, base).item(), mode
        hk.mode = None
    assert {(c_, m): c_.__dict__.get(m) for c_, m in reg.classes()} == before
    assert lg0.shape == (1, V)


def test_route_and_union_measurement(moe):
    be, cfg = moe
    reg = OP.Registry(be.model, cfg)
    hk = OP.Hooks(reg)
    c = be.make_cache()
    be.prefill_chunk(c, list(range(5, 30)))
    with hk:
        hk.mode = "route"
        bc = be.batch([be.fork(c) for _ in range(4)])
        for t in range(3):
            mx.eval(be.step(bc, mx.array([3 + t, 40 + t, 90 + t, 120 + t])))
        hk.mode = None
    st = hk.route["attn+moe:mlp.switch_mlp.gate_proj"]
    calls, distinct, slots = st
    assert calls == 3 * 3                       # 3 MoE layers x 3 steps
    assert slots / calls == 4 * K               # 4 rows x top_k slots
    assert K <= distinct / calls <= min(E, 4 * K)


def test_ablation_stub_keeps_shapes_and_changes_values(moe):
    be, cfg = moe
    reg = OP.Registry(be.model, cfg)
    hk = OP.Hooks(reg, ctx=64)
    c = be.make_cache()
    be.prefill_chunk(c, list(range(5, 30)))
    tok = mx.array([7, 9])
    with hk:
        hk.mode = "record"
        mx.eval(be.step(be.batch([be.fork(c), be.fork(c)]), tok))
        called = {k for k in reg.ops if hk.calls.get(k)}
        assert "attn+moe:mlp.gate" in called and "attn+moe:self_attn" in called
        base = be.step(be.batch([be.fork(c), be.fork(c)]), tok)
        mx.eval(base)
        hk.mode, hk.stub = "ablate", set()
        same = be.step(be.batch([be.fork(c), be.fork(c)]), tok)
        mx.eval(same)
        assert mx.array_equal(base, same).item()           # ablate mode with nothing stubbed = identity
        for key in ("attn+moe:self_attn", "attn+moe:mlp", "attn+moe:mlp.gate", "layer.attn+moe"):
            hk.stub = {key}
            y = be.step(be.batch([be.fork(c), be.fork(c)]), tok)
            mx.eval(y)
            assert y.shape == base.shape and not mx.array_equal(y, base).item(), key
            assert not np.isnan(np.array(y)).any(), key
        hk.mode = None


def test_router_stub_draws_fresh_experts(moe):
    be, cfg = moe
    reg = OP.Registry(be.model, cfg)
    hk = OP.Hooks(reg)
    spec = ("a", (1, 1, K), mx.uint32)
    picks = {tuple(sorted(np.array(hk._build(spec, True)).ravel().tolist())) for _ in range(30)}
    assert len(picks) > 3
    assert all(len(set(p)) == K and max(p) < E for p in picks)
    f = hk._build(("a", (2, 1, E), mx.float16), True)
    assert f.dtype == mx.float16 and float(f.min()) > 0


def test_graph_census_counts_dispatches_not_views():
    a = mx.random.normal((8, 8))
    b = mx.random.normal((8, 8))
    mx.eval(a, b)
    y = (a @ b).T + a
    c = OP.graph_census([y])
    assert c["Matmul"] == 1 and c["Add"] == 1 and "Transpose" not in c and sum(c.values()) == 2
    mx.eval(y)
    assert sum(OP.graph_census([y]).values()) == 0           # an evaluated array has no pending node
    z = mx.sum(mx.exp(a))
    assert sum(OP.graph_census([z]).values()) == 2


def test_launch_attribution_is_exact_on_a_known_module():
    import mlx.nn as nn

    class Blk(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16, bias=False)
            self.norm = nn.RMSNorm(16)

        def __call__(self, x):
            return self.norm(self.fc(x)) + x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Blk(), Blk()]

        def __call__(self, x):
            for l in self.layers:
                x = l(x)
            return x * 2

    net = Net()
    mx.eval(net.parameters())
    reg = OP.Registry(net)
    x = mx.random.normal((1, 16))
    mx.eval(x)
    total = OP.graph_census([net(x)])
    hk = OP.Hooks(reg)
    with hk:
        hk.mode = "launch"
        y = net(x)
        hk.launch_flush([y])
        hk.mode = None
    attributed = sum(v[1] for v in hk.launch.values())
    assert attributed == sum(total.values())
    fc = hk.launch["blk:fc"] if "blk:fc" in hk.launch else next(v for k, v in hk.launch.items() if k.endswith(":fc"))
    assert fc == [2, 2]                         # 2 layers x 1 matmul each (calls, dispatches)
    assert hk.launch["<top>"][1] >= 1           # the final `* 2`


def test_gate_flock_and_checkpoint(tmp_path):
    p = str(tmp_path / "gate.lock")
    g = OP.Gate(p, hold_max=0.0, say=lambda *a: None)
    with g.hold():
        f2 = open(p, "a")
        with pytest.raises(BlockingIOError):
            fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert g.checkpoint() is True                # held longer than hold_max: handed on and re-taken
        with pytest.raises(BlockingIOError):
            fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)   # free after the context
    f2.close()
    assert OP.NullGate().checkpoint() is False


def test_full_pipeline_decode_and_prefill_report(moe, tmp_path):
    be, cfg = moe
    res = OP.profile(be, cfg, None, _fast_opts(), say=lambda *a: None)
    assert not res.get("errors"), res.get("errors")
    for R in ("1", "2"):
        d = res["decode"][R]
        rows = {r["key"]: r for r in d["rows_table"]}
        s = d["summary"]
        assert s["step_ms"] > 0 and s["launches_step"] > 0 and s["graph_build_ms"] > 0 and s["bytes"] > 0
        # the hooked per-op count reproduces the unsynced graph census (same nodes, attributed to modules / <top>)
        # (not bit exact: mx.compile fuses elementwise nodes across module boundaries in the unsynced graph, a few nodes)
        assert abs(s["launches_attributed"] - s["launches_step"]) <= max(3, 0.02 * s["launches_step"])
        assert d["launch"]["attributed"] == sum(v["self"] for v in d["launch"]["per_op"].values())
        gp = rows["attn+moe:mlp.switch_mlp.gate_proj"]
        assert gp["experts_distinct"] is not None and 1 <= gp["experts_distinct"] <= E
        assert gp["bytes"] == pytest.approx(OP.op_bytes(Reg(be, cfg).ops[gp["key"]], int(R), Reg(be, cfg), gp["experts_distinct"]))
        for r in rows.values():
            if r.get("self_ms") is not None:
                assert r["gap_ms"] == pytest.approx(r["self_ms"] - r["ideal_ms"])
            assert r["ideal_ms"] == pytest.approx(r["bytes"] / 1e9 / res["meta"]["B_used_gbs"] * 1e3)
        kvrows = [r for r in rows.values() if r["kv_bytes"] > 0]
        assert {r["key"] for r in kvrows} == {"attn+mlp:self_attn", "attn+moe:self_attn"}
        # rows scale the KV read: R sessions read R caches
    kv1 = {r["key"]: r["kv_bytes"] for r in res["decode"]["1"]["rows_table"]}
    kv2 = {r["key"]: r["kv_bytes"] for r in res["decode"]["2"]["rows_table"]}
    assert kv2["attn+moe:self_attn"] == pytest.approx(2 * kv1["attn+moe:self_attn"])
    ranked = OP.rank(res["decode"]["1"]["rows_table"])
    assert [r["gap_ms"] for r in ranked] == sorted((r["gap_ms"] for r in ranked), reverse=True)
    pf = res["prefill"]
    assert pf["tokens"] == 64 and pf["summary"]["launches_tok"] > 0
    prow = {r["key"]: r for r in pf["rows_table"]}
    gpp = prow["attn+moe:mlp.switch_mlp.gate_proj"]
    assert gpp["calls"] == 3 * 2                        # 3 MoE layers x 2 chunks
    assert gpp["experts_distinct"] > K and gpp["flops"] == pytest.approx(3 * 2 * 32 * 64 * K * 64)
    jp, mp = OP.write_report(res, str(tmp_path))
    back = json.load(open(jp))
    assert back["decode"]["1"]["summary"]["launches_step"] == res["decode"]["1"]["summary"]["launches_step"]
    md = open(mp).read()
    assert "## Decode R=1" in md and "## Decode R=2" in md and "## Prefill 64 tokens" in md and "| rank | op |" in md
    assert OP.main(["--report-only", jp]) == 0


def Reg(be, cfg):
    return OP.Registry(be.model, cfg)


def test_bandwidth_arm_and_cli_args():
    r = OP.arm_bw((0.01,), reps=5, say=lambda *a: None)
    assert r["B_best"] > 0 and "sum_axis1_0.01GiB_u32" in r
    assert OP.arm_peak((128,), reps=2, say=lambda *a: None)["fp16_gemm_tflops"] > 0
    a = OP.parse_args(["--model", "x", "--decode-rows", "1,3", "--no-gate", "--bw-gib", "0.5,1"])
    o = OP.opts_from(a)
    assert o.rows == (1, 3) and isinstance(o.gate, OP.NullGate) and o.bw_gibs == (0.5, 1.0) and o.ctx == 8192 and o.prefill == 8192
    assert isinstance(OP.opts_from(OP.parse_args(["--model", "x"])).gate, OP.Gate)


def test_bandwidth_and_peak_arm_set_B_and_roof(moe):
    be, cfg = moe
    res = OP.profile(be, cfg, None, _fast_opts(rows=(1,), arms=("bw",)), say=lambda *a: None)
    assert res["meta"]["B_used_gbs"] == res["bandwidth"]["B_best"]
    assert res["meta"]["peak_tflops_used"] == res["bandwidth"]["fp16_gemm_tflops"]
    res2 = OP.profile(be, cfg, None, _fast_opts(rows=(1,), arms=("bw",), B=1000.0, peak_tflops=50.0), say=lambda *a: None)
    assert res2["meta"]["B_used_gbs"] == 1000.0 and res2["meta"]["peak_tflops_used"] == 50.0


def test_render_md_handles_missing_columns():
    res = {"meta": {"model_type": "x", "backend": "B", "layers": 1, "layer_kinds": {"k": 1}, "B_used_gbs": 100.0},
           "decode": {"1": {"ctx": 8, "summary": {"step_ms": 1.0, "tok_s": 1000.0, "bytes": 1e6, "ideal_ms": 0.1, "pct_B": 10.0, "unattributed_ms": None,
                                                  "launches_step": 5, "launches_tok": 5.0, "launches_attributed": 5, "graph_build_ms": 0.2,
                                                  "host_roundtrip_ms": 1.1, "host_bound": False},
                            "launch": {"by_type": {"Matmul": 2}},
                            "rows_table": [{"key": "a", "layers": 1, "calls_step": 1.0, "self_ms": None, "noise_ms": None, "bytes": 1e6, "ideal_ms": 0.01, "launch_self": None,
                                            "host_us_self": None},
                                           {"key": "b", "layers": 1, "calls_step": 1.0, "self_ms": 0.5, "noise_ms": 0.1, "bytes": 1e6, "ideal_ms": 0.01, "gap_ms": 0.49,
                                            "launch_self": 2, "host_us_self": 3.0, "launch_tok": 2.0}]}}}
    md = OP.render_md(res)
    assert "| 1 | b |" in md and "| a |" not in md


def test_cli_end_to_end_on_a_written_qwen3_pack(tmp_path):
    """--model DIR through load_backend (the served load path), a dense Qwen3 pack: JSON + Markdown land in --out."""
    from mlx.utils import tree_flatten
    from tools.dayone import engine
    cfg = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=64, num_hidden_layers=3, intermediate_size=128, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=False)
    pack = tmp_path / "pack"
    pack.mkdir()
    model, _, _ = engine.write_pack(cfg, str(pack))
    mx.save_safetensors(str(pack / "model.safetensors"), {k: v.astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}, {"format": "pt"})
    out = tmp_path / "out"
    import subprocess
    import sys
    env = dict(os.environ, KYOJIN_WIRED_MB="0")          # a fresh process: the CLI loads, wires and warms like a user's run
    r = subprocess.run([sys.executable, OP.__file__, "--device", "cpu", "--model", str(pack), "--no-gate", "--arms", "bw,decode,prefill", "--bw-gib", "0.01", "--decode-rows", "1,2", "--ctx", "40", "--prefill", "48",
                  "--chunk", "24", "--blocks", "2", "--steps", "3", "--out", str(out)], env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.load(open(out / "opprof.json"))
    assert res["meta"]["backend"] in ("Q35Backend", "GenericBackend") and res["meta"]["layer_kinds"] == {"attn+mlp": 3}
    assert res["bandwidth"]["B_best"] > 0 and set(res["decode"]) == {"1", "2"}
    assert "## Prefill 48 tokens" in (out / "opprof.md").read_text()
