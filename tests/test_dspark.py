"""DSpark draft head: ArchSpec tables (no Metal needed) and the end to end speculative generate on the DS4.1 slice with REAL mtp.0-2 weights
(KYOJIN_SPEC_PACKS must list the ds2m pack: trunk layers 2-3 of the real pack, 32 experts, vocab 4096, mtp stages cut the same way,
built by tools/shrink_pack.py --mtp 0,1,2 --taps 0,1,1). The drafter only proposes; the trunk's verify decides, so the output must equal plain
greedy whatever the proposals are: the oracle mode feeds the true continuation with injected errors (every accepted count 0..D happens)."""
import os
import re
import subprocess
import sys

import pytest

from kyojin_mlx import archs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = [p for p in os.environ.get("KYOJIN_SPEC_PACKS", "").split(":") if p]


def test_ds41_declares_a_draft_head_glm_does_not():
    assert archs.DS41.mtp is not None and archs.DS41.mtp.kind == "dspark"
    assert archs.GLM5.mtp is None
    with pytest.raises(AssertionError):
        archs.mtp_arch(archs.GLM5)


@pytest.mark.parametrize("pack_name,module", [
    ("mtp.0.attn.wq_a", "language_model.layers.0.attn.wq_a"),
    ("mtp.2.attn.wo_a.slices", "language_model.layers.2.attn.wo_a"),
    ("mtp.1.ffn.experts.w1", "language_model.layers.1.ffn.switch_mlp.gate_proj"),
    ("mtp.1.ffn.shared_experts.w2", "language_model.layers.1.ffn.shared_experts.down_proj"),
    ("mtp.0.main_proj", "language_model.main_proj"),
])
def test_stage_names_map_to_drafter_modules(pack_name, module):
    assert archs.target(archs.mtp_arch(archs.DS41), pack_name) == module


def test_trunk_spec_still_drops_mtp():
    assert archs.target(archs.DS41, "mtp.0.attn.wq_a") is None


def run_gen(*args):
    pack = next((p for p in PACKS if os.path.basename(p.rstrip("/")) == "ds2m"), None)
    if pack is None:
        pytest.skip("pack ds2m not in KYOJIN_SPEC_PACKS")
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "spec_gen.py"), pack, *args], cwd=ROOT, env=env, capture_output=True, text=True)


def test_draft_tail_rows_is_the_deepest_planned_depth():
    """Flag draft_tail_cut: the tail runs on the rows the round verifies (its deepest planned depth), clamped to the block; the flag off keeps the whole block."""
    pytest.importorskip("mlx.core")
    from kyojin_mlx import dspark as D

    class Stub:
        block = 5

    saved = os.environ.pop("KYOJIN_DRAFT_TAIL_CUT", None)
    try:
        assert D.Drafter._tail_rows(Stub(), [1, 3, 2]) == 3                  # on by default
        os.environ["KYOJIN_DRAFT_TAIL_CUT"] = "0"
        assert D.Drafter._tail_rows(Stub(), [1, 3, 2]) is None
        os.environ["KYOJIN_DRAFT_TAIL_CUT"] = "1"
        assert D.Drafter._tail_rows(Stub(), [1, 3, 2]) == 3
        assert D.Drafter._tail_rows(Stub(), [0, 5]) == 5
        assert D.Drafter._tail_rows(Stub(), [0, 9]) == 5                     # clamped to the block
        assert D.Drafter._tail_rows(Stub(), []) == 1
    finally:
        os.environ.pop("KYOJIN_DRAFT_TAIL_CUT", None)
        if saved is not None:
            os.environ["KYOJIN_DRAFT_TAIL_CUT"] = saved


def test_generate_equals_plain_greedy_with_tail_cut():
    """Flag draft_tail_cut: the tail cut never changes an emitted token (the verify decides them; the kept rows keep their proposals and confidences)."""
    os.environ["KYOJIN_DRAFT_TAIL_CUT"] = "1"
    try:
        r = run_gen("--n", "24", "--depths", "1,2,3,5")
    finally:
        os.environ.pop("KYOJIN_DRAFT_TAIL_CUT", None)
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-900:] + r.stderr[-600:]


def test_generate_equals_plain_greedy_with_real_drafter():
    r = run_gen("--n", "24", "--depths", "0,1,3")
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-900:] + r.stderr[-600:]


def test_generate_exact_for_every_accepted_count():
    r = run_gen("--n", "40", "--depths", "2,3", "--oracle")
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-900:] + r.stderr[-600:]
    assert all(float(x) < 1e-6 for x in re.findall(r"max relative diff ([0-9.e+-]+)", r.stdout))        # ring == ring from solo-step taps


def test_blind_accept_is_caught():
    r = run_gen("--n", "40", "--depths", "3", "--oracle", "--neg", "blind")
    assert r.returncode == 1 and "MISMATCH" in r.stdout, r.stdout[-900:] + r.stderr[-600:]


def test_two_interleaved_sessions_share_one_drafter():
    r = run_gen("--n", "30", "--depths", "3", "--oracle", "--interleave")
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-900:] + r.stderr[-600:]


def test_chunked_prefill_keeps_the_tail_of_all_chunks():
    r = run_gen("--n", "24", "--len", "150", "--chunk", "64", "--depths", "3", "--oracle")
    assert r.returncode == 0 and "EXACT" in r.stdout, r.stdout[-900:] + r.stderr[-600:]
    assert "shape (1, 128, 512)" in r.stdout                      # 150 tokens in chunks of 64 + 64 + 22: the ring still holds 128 rows


def test_draft_attention_equals_the_trunk_local_attention_for_one_token():
    """The draft attention is the trunk's local (window) attention with the cache replaced by explicit main-kv rows and the position of the block:
    for a block of one token whose rows are the trunk's own window, both must give the same output. Checks rope offsets, sinks, inverse rope,
    wo_a / wo_b flow of DSparkAttention against the validated trunk code."""
    import copy
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    pack = next((p for p in PACKS if os.path.basename(p.rstrip("/")) == "ds2l"), None)
    if pack is None:
        pytest.skip("pack ds2l not in KYOJIN_SPEC_PACKS")
    import numpy as np
    from kyojin_mlx import dspark as D, patch as P, rows as R
    over = {"compress_ratios": [0, 2], "kv_source_layer_ids": [1], "index_source_layer_ids": [1], "candidate_source_layer_id": 1, "index_topk": 8}
    model, spec, cfg = P.build_model(pack, over)
    P.patch_model(model, pack, spec, cfg)
    lm = model.language_model
    attn = lm.layers[0].attn
    assert attn.compress_ratio == 0
    rng = np.random.default_rng(3)
    n = 60
    cache = lm.make_cache()
    lm(mx.array(rng.integers(16, cfg["text_config"]["vocab_size"] - 16, n).tolist())[None], cache=cache)
    mx.eval(cache[0].window[0])
    x = mx.array(rng.standard_normal((1, 1, cfg["text_config"]["hidden_size"])).astype(np.float32)).astype(mx.float16)
    want = attn(x, R.fork(cache)[0])
    draft = copy.copy(attn)
    draft.__class__ = D._make_attention_class(D._classes())
    draft.fq_out = True                                              # the patched trunk rounds the attention output to fp8 before wo_a as well (ArchSpec.fq_grouped_in)
    ctx = D._Ctx(1)
    ctx.p0, ctx.rows[0] = n, cache[0].window[0][:, :n]               # positions 0..n-1 sit in ring slots 0..n-1 (no wrap yet)
    got = draft(x, ctx)
    w, g = np.array(want.astype(mx.float32)), np.array(got.astype(mx.float32))
    assert np.abs(w - g).max() <= 2e-3 * (np.abs(w).max() + 1e-6), (np.abs(w - g).max(), np.abs(w).max())


def test_compiled_draft_and_append_equal_eager_at_advancing_positions():
    """The compiled draft forward (block start position as an array input, one trace) gives the eager one's tokens and confidences to the bit, and the compiled `append`
    the eager rows to the bit, over 24 consecutive rounds at advancing positions with 1..4 appended rows each, on the ds2m slice with the REAL mtp weights; no retrace after the
    warm-up (dspark.STATS)."""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    pack = next((p for p in PACKS if os.path.basename(p.rstrip("/")) == "ds2m"), None)
    if pack is None:
        pytest.skip("pack ds2m not in KYOJIN_SPEC_PACKS")
    import numpy as np
    from kyojin_mlx import dspark as D, patch as P, rows as R, spec as S
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    R.enable(model, spec)
    tc = cfg["text_config"]
    drafter = D.build_drafter(model, pack, spec, cfg, conf_threshold=0.0)
    S.install(model, spec, tap_layers=set(drafter.tap_ids))
    rng = np.random.default_rng(2)
    prompt = rng.integers(16, tc["vocab_size"] - 16, 135).tolist()               # > sliding_window: the ring is full, the compiled path is eligible
    cache, lg, taps, first = S.prefill(model, prompt, drafter.tap_ids)
    ctx_a, ctx_b = drafter.new_ctx(), drafter.new_ctx()
    drafter.ctx = ctx_a
    drafter.append([taps[l] for l in drafter.tap_ids], first)
    ctx_b.rows = list(ctx_a.rows)
    mx.eval(ctx_a.rows)
    H = tc["hidden_size"]
    p, snap = 135, None
    for r in range(24):
        ids = mx.array([[int(rng.integers(16, tc["vocab_size"] - 16))] + [drafter.noise] * (drafter.block - 1)])
        drafter.ctx = ctx_a
        want_t, want_c = D.Drafter._eager(drafter, ids, p)                          # eager, python int position
        got_t, got_c = drafter._cfwd(ids, mx.array(p, dtype=mx.int32), *ctx_a.rows) if r else (want_t, want_c)
        mx.eval(want_t, want_c, got_t, got_c)
        assert np.array_equal(np.array(want_t), np.array(got_t)) and np.array_equal(np.array(want_c), np.array(got_c)), r
        n = 1 + r % 4
        tp = [mx.random.normal((n, H)).astype(mx.float16) for _ in drafter.tap_ids]
        mx.eval(tp)
        D.COMPILE = False
        drafter.ctx = ctx_a
        drafter.append(tp, p)                                                        # eager append
        D.COMPILE = True
        drafter.ctx = ctx_b
        drafter.append(tp, p)                                                        # compiled once warm
        mx.eval(ctx_a.rows, ctx_b.rows)
        for x, y in zip(ctx_a.rows, ctx_b.rows):
            assert np.array_equal(np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))), (r, n)
        ctx_b.rows = list(ctx_a.rows)                                                # same state for the next round whatever the path
        p += n
        if r == 8:
            snap = dict(D.STATS)
    assert snap is not None and D.STATS["trace_draft"] == snap["trace_draft"] and D.STATS["trace_append"] == snap["trace_append"], (snap, D.STATS)
    assert D.STATS["trace_draft"] == 1 and D.STATS["trace_append"] <= 4
