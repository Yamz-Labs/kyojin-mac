"""Speculative decode inside the batch and the KV memory budget, on a fake backend (needs mlx, no model): rows = sum over sessions of the pending
token + drafts, accept per session, rollback per session, adaptive depth; dropped caches rebuilt from the prompt + a replay, parked sessions
counted in the budget. The fake's next token is a hash of the whole history, so any lost, repeated or misplaced token changes everything after."""
import threading
import time

import mlx.core as mx
import pytest

from kyojin_mlx.serve.engine import Engine, free_unified_bytes
from kyojin_mlx.serve.sampling import Sampling
from test_serve_engine import Fake, V, collect, nxt, onehot


class Slow(Fake):
    """Fake whose decode step takes a moment, so that requests pile up behind the busy slots (yields, drops) whatever the machine does."""

    def step(self, bc, tokens):
        time.sleep(0.0004)
        return super().step(bc, tokens)


class FakeSpec(Slow):
    """Fake with a draft head: the drafter proposes the true continuation, wrong at positions p with p % err == 0 (err = 0: always right,
    err = 1: always wrong). Verify rows are the logits after each fed token on the session's own history."""
    kind = "fakespec"
    supports_spec = True
    max_spec_rows = 8

    def __init__(self, err: int = 3, rows: int = 8):
        self.err, self.max_spec_rows = err, rows
        self.passes, self.depth_asked, self.replays = [], [], []

    def new_draft(self):
        return {"fed": 0}

    def draft_snapshot(self, d):
        return dict(d)

    def draft_restore(self, snap):
        return dict(snap)

    def draft_nbytes(self, d):
        return 8

    def draft_arrays(self, d):
        return []

    def prefill_chunk(self, cache, ids, draft=None):
        if draft is not None:
            draft["fed"] += len(ids)
        return super().prefill_chunk(cache, ids)

    def replay(self, cache, tokens, draft=None):
        self.replays.append(len(tokens))
        cache = {"h": cache["h"] + list(tokens)}
        if draft is not None:
            draft["fed"] += len(tokens)
        return cache

    def draft(self, d, pending, p0, depth, history):
        ids, out = history
        assert out[-1] == pending and p0 == len(ids) + len(out) - 1
        self.depth_asked.append(depth)
        h, res = ids + out, []
        for i in range(depth):
            t = nxt(h)
            if self.err and (p0 + i) % self.err == 0:
                t = (t + 1) % V
            res.append(t)
            h = h + [t]
        return res

    def spec_verify_bc(self, bc, token_lists):
        time.sleep(0.0004)
        self.passes.append(sum(len(t) for t in token_lists))
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [nxt(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return SpecPass(onehot(rows), starts, bc, token_lists)


class SpecPass:
    def __init__(self, logits, starts, bc, token_lists):
        self.logits, self.starts, self.bc, self.token_lists = logits, starts, bc, token_lists

    def commit(self, accepted, drafts=None, p0s=None):
        for k, (cache, toks) in enumerate(zip(self.bc, self.token_lists)):
            if drafts is not None and drafts[k] is not None:
                drafts[k]["fed"] += accepted[k] + 1
            cache["h"].extend(toks[:accepted[k] + 1])             # the batched cache of the fake is the sessions' own dicts


def run(reqs, backend=None, sessions=3, delay=0.01, **kw):
    """reqs: {name: (ids, kwargs for submit)}. Returns ({name: (tokens, reason, stats)}, engine status, backend)."""
    be = backend or Fake()
    kw.setdefault("ctx", 4096)
    eng = Engine(be, sessions=sessions, chunk=kw.pop("chunk", 4), block=kw.pop("block", 8), **kw)
    eng.start()
    out = {}

    def worker(i, name):
        time.sleep(delay * i)
        ids, sub = reqs[name]
        out[name] = collect(eng.submit(ids, **sub))
    ths = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(reqs)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    st = eng.status()
    eng.stop()
    return out, st, be


def prompts(n=6, base=10):
    return {f"r{i}": ([(i * 7 + j * 3) % V for j in range(base + 4 * i)], dict(max_new=20 + 9 * i)) for i in range(n)}


def solo_all(reqs, **kw):
    return {n: run({n: r}, sessions=1, prefix_bytes=0, **kw)[0][n] for n, r in reqs.items()}


@pytest.mark.parametrize("err", [0, 3, 1])
@pytest.mark.parametrize("depth", [1, 3])
def test_spec_equals_plain(err, depth):
    reqs = prompts()
    want = solo_all(reqs)
    got, st, be = run(reqs, FakeSpec(err=err), mtp_depth=depth, prefix_bytes=1 << 20)
    for n in reqs:
        assert got[n][0] == want[n][0] and got[n][1] == want[n][1] == "length", n
    assert max(be.passes) <= be.max_spec_rows
    if err == 0:                                        # a perfect drafter: far fewer passes than tokens
        assert st["spec_rounds"] < sum(len(w[0]) for w in want.values()) // 2 and st["spec_accepted"] > 0
    assert st["spec_rows"] == sum(be.passes)


def test_rows_are_summed_over_sessions_and_capped():
    reqs = {f"r{i}": ([i + 1, i + 5, i + 9], dict(max_new=40)) for i in range(4)}
    got, st, be = run(reqs, FakeSpec(err=0, rows=6), sessions=4, mtp_depth=4, delay=0)
    want = solo_all(reqs)
    assert all(got[n][0] == want[n][0] for n in reqs)
    assert max(be.passes) == 6 and any(p > 4 for p in be.passes)        # sum(D + 1) over the sessions, never above the cap


def test_stop_token_and_length_inside_a_round():
    ids = [3, 1, 4, 1, 5]
    full, _, _ = run({"a": (ids, dict(max_new=30))}, sessions=1)
    seq = full["a"][0]
    stop = seq[11]
    for depth in (2, 4):
        got, _, _ = run({"a": (ids, dict(max_new=30, stop_ids=[stop])), "b": (ids[::-1], dict(max_new=7))}, FakeSpec(err=0), mtp_depth=depth)
        assert got["a"][1] == "stop" and got["a"][0] == seq[:seq.index(stop)]
        assert got["b"][1] == "length" and len(got["b"][0]) == 7


def test_sampled_session_drafts_and_greedy_neighbour_equals_solo():
    s = Sampling(0.9, 0.95, 20, 0.02, 11)
    reqs = {"g": ([1, 2, 3, 4], dict(max_new=30)), "s": ([5, 6, 7], dict(max_new=30, sampling=s))}
    want = solo_all(reqs)
    got, st, be = run(reqs, FakeSpec(err=0), mtp_depth=3)
    assert got["g"][0] == want["g"][0]                  # greedy: independent of D and of the sampled neighbour
    assert len(got["s"][0]) == 30                      # sampled: the distribution test is tests/test_serve_spec_sampling.py
    assert st["spec_accepted"] > 0


def test_depth_falls_to_zero_when_drafts_are_wrong_and_recovers_by_probing(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                             # the predictor x cost policy (D = 1 probes)
    reqs = {"a": ([1, 2, 3], dict(max_new=120))}
    got, st, be = run(reqs, FakeSpec(err=1), sessions=1, mtp_depth=3, probe_every=8)
    assert got["a"][0] == run(reqs, sessions=1)[0]["a"][0]
    asked = be.depth_asked
    assert asked[0] >= 2 and 0 not in asked                                 # draft() is only called for D > 0
    assert st["spec_rounds"] > len(asked) * 2                               # most rounds were plain steps (D = 0)
    assert st["spec_rounds"] - len(asked) >= 40
    assert 1 in asked[2:] and asked.count(1) >= 2                          # probes with D = 1 (the gap doubles while they fail)


def test_rate_rule_keeps_tokens_on_wrong_drafts():
    reqs = {"a": ([1, 2, 3], dict(max_new=120))}
    got, st, be = run(reqs, FakeSpec(err=1), sessions=1, mtp_depth=3, probe_every=8)
    assert got["a"][0] == run(reqs, sessions=1)[0]["a"][0]
    assert set(be.depth_asked) <= {1, 2, 3} and be.depth_asked[0] == 3     # the cap (less only where the reply's room ends)


def _policy_depth(n_sessions, acc, trials=100.0, **kw):
    """The depth the policy asks for a greedy session whose drafts were accepted with rate `acc`, when `n_sessions` share the verify pass."""
    from types import SimpleNamespace
    eng = Engine(FakeSpec(err=0), sessions=4, mtp_depth=3, ctx=4096, chunk=4, block=8, **kw)
    eng.members = [object()] * n_sessions
    s = SimpleNamespace(sampling=Sampling(), max_new=500, n_gen=0, n_prompt=10, acc_succ=acc * trials, acc_trial=trials, idle_rounds=0)
    return eng._plan_depth(s)


def test_depth_policy_shares_the_fixed_cost_of_the_step(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                             # the solo case below is the formula's (the default solo rule is the cap's rate)
    # alone, a plain step is the whole solo step: drafts pay at 0.735 (D = 2), as with the costs measured for one user
    assert _policy_depth(1, 0.735) == 2
    # four sessions share the weights read of the pass: a plain row costs 0.55 solo steps, so the same acceptance no longer pays
    assert _policy_depth(4, 0.735) == 0
    assert _policy_depth(4, 0.85) == 0
    # a very accurate drafter still pays with four sessions
    assert _policy_depth(4, 0.95) == 3
    # nothing to share: any number of sessions with row_cost 1 (a row costs a whole step) means drafts never pay at these acceptances
    assert _policy_depth(4, 0.85, row_cost=1.0) == 0


def test_prefix_hit_equals_miss_with_a_draft_state():
    shared = [(i * 5) % V for i in range(40)]
    a, b = shared + [1, 2, 3], shared + [9, 8]
    be = FakeSpec(err=0)
    got, st, _ = run({"a": (a, dict(max_new=12)), "b": (b, dict(max_new=12))}, be, sessions=1, mtp_depth=2, prefix_bytes=1 << 20)
    want = solo_all({"a": (a, dict(max_new=12)), "b": (b, dict(max_new=12))})
    assert got["a"][0] == want["a"][0] and got["b"][0] == want["b"][0]
    assert got["b"][2]["cached_tokens"] == 40 and st["prefix_hits"] == 1


# ------------------------------------------------------------------------------------------------ memory budget
def churn_reqs(n=8):
    return {f"c{i}": ([(i * 11 + j) % V for j in range(12 + 3 * i)], dict(max_new=70 + 5 * i)) for i in range(n)}


@pytest.mark.parametrize("mtp", [0, 3])
def test_kv_budget_churn_equals_solo(mtp):
    reqs = churn_reqs()
    want = solo_all(reqs)
    ctx = 300                                                             # below one page: every share is the floor, so replies yield after rq_floor tokens
    budget = 8 * 330                                                      # two active sessions (<= 145 tokens each) and about one parked (8 bytes a token)
    be = FakeSpec(err=3) if mtp else Slow()
    got, st, _ = run(reqs, be, sessions=2, ctx=ctx, rq_floor=24, kv_budget=budget, prefix_bytes=0, mtp_depth=mtp, replay_chunk=7, delay=0, kv_refresh=1)
    for n in reqs:
        assert got[n][0] == want[n][0] and got[n][1] == "length", n
    assert st["yields"] > 0 and st["kv_drops"] > 0 and st["rebuilds"] > 0 and st["replay_tokens"] > 0
    assert st["kv_peak_bytes"] <= budget, (st["kv_peak_bytes"], budget)


def test_active_sessions_are_never_dropped_and_admission_waits():
    reqs = {f"a{i}": ([i + 1] * 20, dict(max_new=60)) for i in range(3)}
    want = solo_all(reqs)
    got, st, _ = run(reqs, Fake(), sessions=3, ctx=1024, kv_budget=8 * 40, prefix_bytes=0, delay=0)
    assert all(got[n][0] == want[n][0] for n in reqs)
    assert st.get("kv_drops", 0) == 0                                      # nothing parked: nothing to drop; the others waited for a slot


@pytest.mark.parametrize("mtp", [0, 2])
def test_dropped_session_is_rebuilt_with_the_prefix_cache(mtp):
    shared = [(i * 3) % V for i in range(48)]
    reqs = {"long": (shared + [1, 2], dict(max_new=90)), "x": (shared + [7], dict(max_new=50)), "y": ([9, 9, 9], dict(max_new=30)),
            "z": ([4, 4, 4, 4], dict(max_new=30))}
    want = solo_all(reqs)
    be = FakeSpec(err=2) if mtp else Slow()
    got, st, _ = run(reqs, be, sessions=2, ctx=300, rq_floor=16, kv_budget=8 * 250, prefix_bytes=1 << 20, block=16, chunk=8, delay=0.003, kv_refresh=1,
                     mtp_depth=mtp)
    assert all(got[n][0] == want[n][0] for n in reqs)
    assert st["rebuilds"] >= 1 and st["prefix_hit_tokens"] >= 32 and st["kv_peak_bytes"] <= 8 * 250


def test_free_unified_bytes_is_sane_or_none():
    f = free_unified_bytes()
    assert f is None or f > 0


# ------------------------------------------------------------------------------------------------ negative controls
class NoReplay(Slow):
    def replay(self, cache, tokens, draft=None):
        return cache                                      # a rebuilt cache without the tokens it had produced


class NoRollback(FakeSpec):
    def spec_verify_bc(self, bc, token_lists):
        p = super().spec_verify_bc(bc, token_lists)
        c = p.commit
        p.commit = lambda accepted, drafts=None, p0s=None: c([len(t) - 1 for t in token_lists], drafts, p0s)      # keeps every drafted row, accepted or not
        return p


def test_negative_controls_fail():
    reqs = churn_reqs(6)
    want = solo_all(reqs)
    got, st, _ = run(reqs, NoReplay(), sessions=2, ctx=300, rq_floor=24, kv_budget=8 * 330, prefix_bytes=0, delay=0, kv_refresh=1)
    assert st["rebuilds"] > 0 and any(got[n][0] != want[n][0] for n in reqs)
    got, st, _ = run(reqs, NoRollback(err=3), sessions=2, mtp_depth=3, prefix_bytes=0, delay=0)
    assert any(got[n][0] != want[n][0] for n in reqs)


def _measured_depth(acc, ctx, costs, trials=100.0, pos=None):
    from types import SimpleNamespace
    eng = Engine(FakeSpec(err=0), sessions=1, mtp_depth=4, ctx=65536, chunk=4, block=8, spec_costs=costs)
    eng.members = [object()]
    s = SimpleNamespace(sampling=Sampling(), max_new=500, n_gen=0, n_prompt=ctx, acc_succ=acc * trials, acc_trial=trials, idle_rounds=0, cap_rounds=0,
                        pos_succ=[] if pos is None else [a * trials for a in pos], pos_trial=[] if pos is None else [trials] * len(pos))
    return eng._plan_depth(s), eng


# 4B fp16 on the Air (tools/cost103.py, ms of a solo round at depth 0..5)
AIR_4B = {4096: [21.6, 23.4, 25.8, 29.3, 31.3, 33.0], 32768: [29.3, 31.6, 35.8, 43.3, 45.7, 47.8]}


def test_measured_depth_policy_follows_acceptance_and_context():
    assert _measured_depth(0.85, 4096, AIR_4B)[0] == 4                     # accurate drafts: deepest
    assert _measured_depth(0.5, 32768, AIR_4B)[0] == 2                     # low acceptance at 32K: verify rows cost ~3 ms each, D = 2
    assert _measured_depth(0.05, 32768, AIR_4B)[0] == 0                    # drafts that never land: plain steps
    d, eng = _measured_depth(0.5, 18432, AIR_4B)
    assert abs(eng.round_ms(18432, 2) - (25.8 + 35.8) / 2) < 1e-9           # linear in the context between the measured lengths
    assert abs(eng.round_ms(4096, 6) - 34.7) < 1e-9                         # linear in the depth past the deepest measured one
    # per-position acceptance: a strong first position and a weak second one stop at D = 1
    assert _measured_depth(0.6, 32768, AIR_4B, pos=[0.9, 0.1])[0] == 1
    # streaky text: the conditional acceptance rises with depth and beats the pooled geometric estimate -> the cap, also when only the first
    # positions were measured (the unmeasured ones inherit their predecessor, not the pooled p)
    assert _measured_depth(0.55, 32768, AIR_4B, pos=[0.75, 0.8, 0.85, 0.9])[0] == 4
    assert _measured_depth(0.55, 32768, AIR_4B, pos=[0.75, 0.8])[0] == 4


def test_measured_depth_policy_probes_the_cap():
    from types import SimpleNamespace
    eng = Engine(FakeSpec(err=0), sessions=1, mtp_depth=4, ctx=65536, chunk=4, block=8, spec_costs=AIR_4B, cap_probe=4)
    eng.members = [object()]
    s = SimpleNamespace(sampling=Sampling(), max_new=500, n_gen=0, n_prompt=32768, acc_succ=50.0, acc_trial=100.0, idle_rounds=0, cap_rounds=0,
                        pos_succ=[], pos_trial=[], lk_adapt=None, lk_streak=True, lk_suffix=0)
    got = [eng._plan_depth(s) for _ in range(8)]
    assert got == [2, 2, 2, 4, 2, 2, 2, 4]                                  # one round at the cap every cap_probe rounds keeps the deep positions measured
    eng._spec_update(s, 2, 1)                                                # a shallow round leaves the positions it did not reach as they were
    s.pos_succ, s.pos_trial = [5.0, 4.0, 3.0], [6.0, 5.0, 4.0]
    eng._spec_update(s, 1, 1)
    assert s.pos_trial[1:] == [5.0, 4.0] and abs(s.pos_trial[0] - 6.4) < 1e-9


def test_measured_depth_policy_greedy_output_unchanged():
    reqs = {"a": ([1, 2, 3], dict(max_new=80))}
    got, st, _ = run(reqs, FakeSpec(err=3), sessions=1, mtp_depth=4, spec_costs=AIR_4B)
    assert got["a"][0] == run(reqs, sessions=1)[0]["a"][0]
    assert sum(v for k, v in st.items() if k.startswith("spec_d") and k[6:].isdigit()) == st["spec_rounds"]


class FakeGated(FakeSpec):
    """FakeSpec with the Engram gated decode: plain steps of the batched cache go through launch_gated (as VlmBackend with the gate on)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.gated, self.gated_launches = object(), 0

    def step_feed(self, bc, tokens, drafts, p0s):
        assert self.gated is None, "an ungated plain step on a backend with the gate"
        return self.step(bc, tokens)

    def launch_gated(self, bc, toks, sample, drafts=None, p0s=None):
        self.gated_launches += 1
        y = sample(Fake.step(self, bc, toks))
        mx.async_eval(y)
        return y


def test_depth_zero_sessions_of_a_speculative_server_take_the_gated_path():
    """Task golive-1: a session that runs at D = 0 inside an MTP server gets the commit-gated plain step, as a --mtp-depth 0 server does."""
    s = Sampling(0.9, 0.95, 20, 0.0, 3)
    reqs = {"g": ([1, 2, 3, 4], dict(max_new=30)), "s": ([5, 6, 7], dict(max_new=30, sampling=s))}
    want = solo_all(reqs)
    got, st, be = run(reqs, FakeGated(err=0), mtp_depth=3, depth_override=0)
    assert be.gated_launches > 0 and st["spec_plain"] > 0 and st["spec_d0"] > 0
    assert got["g"][0] == want["g"][0] and len(got["s"][0]) == 30


def test_engram_gate_is_opt_in():
    from kyojin_mlx import gate_wanted
    assert gate_wanted("1") and not gate_wanted("auto") and not gate_wanted("0")


def test_ds41_one_buffer_env_only_with_the_gate(tmp_path, monkeypatch):
    """The one-buffer MLX settings apply to a DS4.1 pack only when the Engram gate is asked for (KYOJIN_ENGRAM_GATE=1), never to another model
    or after mlx is imported; an explicit value wins. Default buffers with the weights wired decode faster (40.0/70.9 vs 32.0/62.5 tok/s, Ultra)."""
    import json, sys, types
    import kyojin_mlx
    ds, other = tmp_path / "ds41", tmp_path / "other"
    for d, mt in ((ds, "deepseek_v41"), (other, "llama")):
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": mt}))
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(modules={}))      # as if mlx were not imported yet
    monkeypatch.setattr(sys, "argv", ["server", str(ds)])
    for mode in (None, "auto", "0"):
        env = {} if mode is None else {"KYOJIN_ENGRAM_GATE": mode}
        assert not kyojin_mlx.apply_ds41_buffer_env(env) and "MLX_MAX_OPS_PER_BUFFER" not in env
    env = {"KYOJIN_ENGRAM_GATE": "1"}
    assert kyojin_mlx.apply_ds41_buffer_env(env)
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "100000" and env["MLX_MAX_MB_PER_BUFFER"] == "1000000"
    env = {"KYOJIN_ENGRAM_GATE": "1", "MLX_MAX_OPS_PER_BUFFER": "50"}
    kyojin_mlx.apply_ds41_buffer_env(env)
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "50"                                     # an explicit value wins
    monkeypatch.setattr(sys, "argv", ["server", str(other)])
    env = {"KYOJIN_ENGRAM_GATE": "1"}
    assert not kyojin_mlx.apply_ds41_buffer_env(env) and "MLX_MAX_OPS_PER_BUFFER" not in env
    monkeypatch.setattr(sys, "argv", ["server", str(ds)])
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(modules={"mlx.core": 1}))
    assert not kyojin_mlx.apply_ds41_buffer_env({"KYOJIN_ENGRAM_GATE": "1"})


def test_lazy_verify_resets_the_gated_engram_history():
    """The lazy verify (served default) must put the gated decode's look-ahead Engram history back before it builds the verify pass, as the
    eager verify does: else the next gated plain launch restores the pre-verify history and the hashes of the committed drafts are lost."""
    from kyojin_mlx.serve.backends import LazyDrafts, VlmBackend

    calls = []

    class Gate:
        stage = type("Stage", (), {"cg_k": None})()                   # not calibrated: the verify stays ungated under KYOJIN_GATED_VERIFY=1 too

        def reset(self, bc):
            calls.append("reset")

    class R:
        @staticmethod
        def verify(model, bc, xs):
            calls.append(("verify", [x.tolist() for x in xs]))
            return type("Pass", (), {"logits": None, "taps": {}, "starts": [0]})()

    be = object.__new__(VlmBackend)
    be.gated, be.R, be.model = Gate(), R, None
    lh = object.__new__(LazyDrafts)
    lh.toks, lh.slot, lh.depths = mx.array([[7, 8, 9]]), {0: 0}, [2]
    be.spec_verify_lazy([None], lh, [5])
    assert calls == ["reset", ("verify", [[5, 7, 8]])]


def test_round_cost_table_and_path_counters():
    """The served round cost table (spec_rt_r<rows> / spec_rn_r<rows>) and the path counters: every round of the request is one bucket of the
    table (the rows it verified), the draft path that ran is counted, and no lever counter is invented for a path that did not fire."""
    reqs = {"a": ([1, 2, 3, 4], dict(max_new=48))}
    got, st, be = run(reqs, FakeSpec(err=0), sessions=1, mtp_depth=3, prefix_bytes=0)
    buckets = {k: v for k, v in st.items() if k.startswith("spec_rn_r")}
    assert sum(buckets.values()) == st["spec_rounds"]                  # one bucket entry per round, plain ticks included
    assert buckets["spec_rn_r4"] > len(buckets) and buckets["spec_rn_r4"] > st["spec_rounds"] // 2   # D = 3 rounds dominate: 1 + 3 rows
    assert st["spec_rt_r4"] > 0 and st["spec_rwait_r4"] > 0 and st["spec_rpost_r4"] > 0
    assert st["spec_eager_rounds"] + st["spec_lazy_rounds"] == st["spec_rounds"] - st.get("spec_plain", 0)
    assert st["spec_lazy_rounds"] == 0                                 # the fake has no draft_lazy
    assert st["spec_gate_cut"] == 0 and st["spec_block_cut"] == 0      # nothing cut rows the verify had paid for


def test_plain_ticks_land_in_the_one_row_bucket(monkeypatch):
    """A round without drafts is a one-row round: it goes to the same table as a verify pass of one row, so the table prices the plain step and
    the verify rows in one unit (the depth policy's S + slope * rows). On this fake a D = 0 round is the one-row verify (no step_feed)."""
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                        # the predictor x cost policy does fall back to plain rounds
    reqs = {"a": ([1, 2, 3], dict(max_new=120))}
    got, st, be = run(reqs, FakeSpec(err=1), sessions=1, mtp_depth=3, probe_every=8)
    assert st["spec_d0"] > 0 and st["spec_rn_r1"] == st["spec_d0"]
    assert sum(v for k, v in st.items() if k.startswith("spec_rn_r")) == st["spec_rounds"]


def test_gated_verify_flag_dispatches_the_pipelined_build():
    """flags.gated_verify: the lazy verify takes backends._verify_gated (the pass is built and encoded while the drafter runs, the Engram rows are read
    once) only when the Engram commit gate is calibrated; without a gate the pass stays ungated and the counter says so, so the lever cannot look
    like a silent no-op."""
    from kyojin_mlx import flags as FL
    from kyojin_mlx.serve import backends as B
    from kyojin_mlx.serve.backends import LazyDrafts, VlmBackend

    class Stage:
        cg_k = 3

    class Gate:
        stage = Stage()

        def reset(self, bc):
            pass

    class R:
        @staticmethod
        def verify(model, bc, xs, stage=None):
            return type("Pass", (), {"logits": None, "taps": {}, "starts": [0], "eh": None})()

    be = object.__new__(VlmBackend)
    be.gated, be.R, be.model = Gate(), R, None
    lh = object.__new__(LazyDrafts)
    lh.toks, lh.slot, lh.depths = mx.array([[7, 8, 9]]), {0: 0}, [2]
    seen = []
    be._verify_gated = lambda bc, xs: seen.append(xs) or "gated"
    base = dict(B.STATS)
    try:
        assert B.GATED_VERIFY is False                                  # default 0
        assert be.spec_verify_lazy([None], lh, [5]) != "gated" and not seen
        FL.set("gated_verify", 1)
        assert B.GATED_VERIFY is True
        assert be.spec_verify_lazy([None], lh, [5]) == "gated" and len(seen) == 1
        assert [int(t) for t in seen[0][0].tolist()] == [5, 7, 8]        # pending token + the rows the pass paid for
        Stage.cg_k = None                                               # the plain steps have not calibrated the gate yet
        assert be.spec_verify_lazy([None], lh, [5]) != "gated"
        assert B.STATS["gated_verify_nogate"] == base["gated_verify_nogate"] + 1
    finally:
        Stage.cg_k = 3
        FL.set("gated_verify", 0)
        B.GATED_VERIFY = False
        B.STATS.update(base)
    assert B.GATED_VERIFY is False


def test_kv_budget_comes_from_the_fixed_terms():
    from kyojin_mlx.serve.engine import GIB, kv_budget_for
    b = kv_budget_for(256 * GIB, 243 * GIB, int(173.2 * GIB), 16 * GIB)
    assert 20 * GIB < b < 24 * GIB                                      # 31/10 lane: 1M of KV (3.9 GiB active) fits, the 68.8 GiB of the old default does not
    assert kv_budget_for(64 * GIB, 48 * GIB, 60 * GIB, 6 * GIB) == GIB // 4


def test_decode_buffer_env(monkeypatch, tmp_path):
    """Larger command buffers by default, off on a DS4.1 pack, off by flag, an explicit value wins, nothing after mlx is imported."""
    import json, sys, types
    import kyojin_mlx
    ds = tmp_path / "ds41"
    ds.mkdir()
    (ds / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(modules={}))
    monkeypatch.setattr(sys, "argv", ["server", "model.gguf"])
    env = {}
    assert kyojin_mlx.apply_decode_buffer_env(env) and env == {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "400"}
    env = {"MLX_MAX_OPS_PER_BUFFER": "50"}
    kyojin_mlx.apply_decode_buffer_env(env)
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "50" and env["MLX_MAX_MB_PER_BUFFER"] == "400"
    assert not kyojin_mlx.apply_decode_buffer_env({"KYOJIN_DECODE_BUFFERS": "0"})
    monkeypatch.setattr(kyojin_mlx, "physical_ram", lambda: 16 * 2 ** 30)           # a 16 GB Mac keeps MLX's command buffers (warm-up out of memory with 400 MB ones)
    assert not kyojin_mlx.apply_decode_buffer_env({})
    assert kyojin_mlx.apply_decode_buffer_env({"KYOJIN_DECODE_BUFFERS": "2"})
    monkeypatch.undo()
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(modules={}))
    monkeypatch.setattr(sys, "argv", ["server", "model.gguf"])
    monkeypatch.setattr(sys, "argv", ["server", str(ds)])
    assert not kyojin_mlx.apply_decode_buffer_env({})
    monkeypatch.setattr(sys, "argv", ["server", "model.gguf"])
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(modules={"mlx.core": 1}))
    assert not kyojin_mlx.apply_decode_buffer_env({})


def test_lean_malloc_env_only_for_a_serve_process(monkeypatch):
    import types
    import kyojin_mlx
    fake = types.SimpleNamespace(platform="darwin", executable="/py", orig_argv=["python", "-m", "kyojin_mlx.serve", "--model", "m"], modules={})
    monkeypatch.setattr(kyojin_mlx, "_sys2", fake)
    monkeypatch.setattr(kyojin_mlx, "physical_ram", lambda: 16 * 2 ** 30)
    calls = []
    env = {}
    assert kyojin_mlx.apply_lean_malloc_env(env, execv=lambda *a: calls.append(a)) and env["MallocSpaceEfficient"] == "1"
    assert calls == [("/py", ["/py", "-m", "kyojin_mlx.serve", "--model", "m"])]
    env = {"MallocSpaceEfficient": "1"}
    assert not kyojin_mlx.apply_lean_malloc_env(env, execv=lambda *a: calls.append(a)) and len(calls) == 1      # already set: no second exec
    assert not kyojin_mlx.apply_lean_malloc_env({"KYOJIN_LEAN_MALLOC": "0"})
    assert not kyojin_mlx.apply_lean_malloc_env({}, argv=["python", "-m", "pytest"])
    monkeypatch.setattr(kyojin_mlx, "physical_ram", lambda: 256 * 2 ** 30)                                  # large RAM: speed first, unless forced
    assert not kyojin_mlx.apply_lean_malloc_env({}, execv=lambda *a: calls.append(a))
    assert kyojin_mlx.apply_lean_malloc_env({"KYOJIN_LEAN_MALLOC": "2"}, execv=lambda *a: calls.append(a)) and len(calls) == 2
    monkeypatch.setattr(kyojin_mlx, "_sys2", types.SimpleNamespace(platform="linux", orig_argv=fake.orig_argv, modules={}))
    assert not kyojin_mlx.apply_lean_malloc_env({})

