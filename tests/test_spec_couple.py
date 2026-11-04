"""Flag spec_couple (key-coupled drafting for sampled sessions), CPU only.

The draft chain mixes the target's own Gumbel noise of verify row i into proposal i. Every emitted token stays the verify's sample at its row, so the
output distribution must be the plain-sampling one whatever the proposals are (chi-square against the exact distribution, on the fake backend of
test_serve_spec_sampling), while a draft that is close to the target is accepted far more often than with the argmax proposal."""
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kyojin_mlx import dspark
from kyojin_mlx.serve.sampling import Sampling, sample_rows
from test_serve_engine import V
from test_serve_spec import run
from test_serve_spec_sampling import CTX, FakeSoft, check, soft, transitions


def qlogits(c: int, skew: float) -> np.ndarray:
    """The fake draft head's logits for context c: the target's, plus a fixed perturbation of size `skew` (0 = the draft knows the target exactly)."""
    return soft(c) + skew * np.random.default_rng(5000 + c).normal(0.0, 1.0, V).astype(np.float32)


class FakeCouple(FakeSoft):
    """FakeSoft whose draft head has its own logits q and takes `couple` like VlmBackend.draft_many: proposal i = argmax(q_i + scale x Gumbel(key_i))."""
    couples = True

    def __init__(self, skew: float = 0.0, **kw):
        super().__init__(err=0, **kw)
        self.skew, self.couple_calls, self.uncoupled_calls = skew, 0, 0

    def draft_many(self, jobs, couple=None):
        if couple is not None and any(couple):
            self.couple_calls += 1
        else:
            self.uncoupled_calls += 1
        out = []
        for n, (d, pending, p0, depth, hist) in enumerate(jobs):
            cp = couple[n] if couple else None
            res, last = [], pending
            for i in range(depth):
                lg = mx.array(qlogits(last % CTX, self.skew))
                if cp is not None:
                    scale, keys = cp
                    lg = lg + scale * mx.random.gumbel((V,), key=keys[i])
                t = int(mx.argmax(lg).item())
                res.append(t)
                last = t
            out.append(res)
        return out


def go(monkeypatch, pct, s, backend, n_req=4, per=9000, depth=3, sessions=4):
    monkeypatch.setenv("KYOJIN_SPEC_COUPLE", str(pct))
    reqs = {f"r{i}": ([1 + i, 2, 3 + i], dict(max_new=per, sampling=Sampling(s.temperature, s.top_p, s.top_k, s.min_p, 100 + i))) for i in range(n_req)}
    got, st, be = run(reqs, backend, sessions=sessions, ctx=20000, mtp_depth=depth, delay=0, prefix_bytes=0, chunk=8)
    counts = np.zeros((CTX, V), np.int64)
    for i in range(n_req):
        toks, why, _ = got[f"r{i}"]
        assert why == "length" and len(toks) == per
        for c, t in transitions(toks, reqs[f"r{i}"][0][-1]):
            counts[c][t] += 1
    return counts, st, be


def rate(st):
    return st["spec_accepted"] / max(st["spec_drafted"], 1)


def test_noise_is_the_sampler_noise():
    """categorical(logits, key) = argmax(logits + gumbel(key)): the array the drafter adds is the one the verify's sampler adds."""
    T = 0.8
    for seed in range(40):
        lg = mx.random.normal((2000,), key=mx.random.key(seed)) * 2.0
        key = mx.random.key(1000 + seed)
        want = int(sample_rows(lg[None], [Sampling(T)], [key]).item())
        got = int(mx.argmax(lg + T * mx.random.gumbel((2000,), key=key)).item())
        assert got == want, seed


def test_identical_draft_is_accepted_every_time(monkeypatch):
    """q = p exactly (no filters): the coupled proposal IS the target's sample, so every draft the verify reaches is accepted."""
    s = Sampling(1.0, 1.0, 0, 0.0)
    counts, st, be = go(monkeypatch, 100, s, FakeCouple(skew=0.0), n_req=2, per=1500)
    assert be.couple_calls > 0
    assert st["spec_drafted"] > 0 and st["spec_accepted"] == st["spec_drafted"]
    monkeypatch.setenv("KYOJIN_SPEC_COUPLE", "0")
    _, st0, _ = go(monkeypatch, 0, s, FakeCouple(skew=0.0), n_req=2, per=1500)
    assert rate(st0) < 0.7                                               # the argmax proposal: p(argmax q) on soft logits


@pytest.mark.parametrize("pct", [100, 50])
@pytest.mark.parametrize("s,depth", [(Sampling(1.0, 1.0, 0, 0.0), 3), (Sampling(0.7, 0.95, 20, 0.0), 3), (Sampling(0.9, 1.0, 8, 0.02), 4)])
def test_coupled_draft_follows_the_exact_distribution(monkeypatch, s, depth, pct):
    counts, st, be = go(monkeypatch, pct, s, FakeCouple(skew=1.0), depth=depth)
    assert be.couple_calls > 0 and st["spec_couple_rounds"] > 0
    assert st["spec_accepted"] > 0 and st["spec_drafted"] > 0
    check(counts, s, 0.06, f"couple={pct} T={s.temperature} p={s.top_p} k={s.top_k}")


def test_coupling_raises_acceptance_on_an_imperfect_draft(monkeypatch):
    s = Sampling(1.0, 1.0, 0, 0.0)
    _, st1, be1 = go(monkeypatch, 100, s, FakeCouple(skew=0.8), n_req=3, per=3000)
    _, st0, be0 = go(monkeypatch, 0, s, FakeCouple(skew=0.8), n_req=3, per=3000)
    assert be0.couple_calls == 0 and be1.couple_calls > 0
    assert rate(st1) > 1.25 * rate(st0), (rate(st1), rate(st0))


def test_flag_off_and_greedy_are_untouched(monkeypatch):
    g = Sampling(0.0)
    reqs = {"g1": ([1, 2, 3, 4], dict(max_new=80, sampling=g)), "g2": ([9, 8, 7], dict(max_new=60, sampling=g))}
    outs = []
    for pct in (0, 100):
        monkeypatch.setenv("KYOJIN_SPEC_COUPLE", str(pct))
        be = FakeCouple(skew=0.5)
        got, st, be = run(reqs, be, sessions=2, ctx=4000, mtp_depth=3, delay=0, prefix_bytes=0)
        outs.append({k: v[0] for k, v in got.items()})
        assert be.couple_calls == 0 and st.get("spec_couple_rounds", 0) == 0     # a greedy session never takes the coupled draft
    assert outs[0] == outs[1]


def test_seeded_stream_is_reproducible_with_coupling(monkeypatch):
    s = Sampling(0.9, 0.95, 20, 0.0, 77)
    monkeypatch.setenv("KYOJIN_SPEC_COUPLE", "100")
    a = run({"a": ([1, 2, 3], dict(max_new=200, sampling=s))}, FakeCouple(skew=0.5), sessions=1, mtp_depth=3, prefix_bytes=0)[0]["a"][0]
    b = run({"a": ([1, 2, 3], dict(max_new=200, sampling=s))}, FakeCouple(skew=0.5), sessions=1, mtp_depth=3, prefix_bytes=0)[0]["a"][0]
    assert a == b and len(a) == 200


# ---- the drafter's side (no model): noise array and the chain


def bare_drafter(Vd=50, rank=6, H=8, block=4):
    d = dspark.Drafter.__new__(dspark.Drafter)
    d.block = block
    d.lm = types.SimpleNamespace(markov_embed=nn.Embedding(Vd, rank), markov_head=nn.Linear(rank, Vd, bias=False), confidence=nn.Linear(H + rank, 1, bias=False))
    return d


def test_couple_noise_layout(monkeypatch):
    monkeypatch.setattr(dspark, "CAND", 0)
    d = bare_drafter()
    keys = [mx.random.key(i) for i in range(3)]
    n = d._couple_noise([None, (0.5, keys)], 2, 4)
    assert n.shape == (2, 4, 50) and n.dtype == mx.float32
    assert float(mx.abs(n[0]).max().item()) == 0.0                                  # an uncoupled session adds nothing
    assert float(mx.abs(n[1, 3]).max().item()) == 0.0                               # a row past the keys adds nothing
    assert mx.allclose(n[1, 1], 0.5 * mx.random.gumbel((50,), key=keys[1])).item()
    assert d._couple_noise([None, None], 2, 4) is None
    monkeypatch.setattr(dspark, "CAND", 64)
    assert d._couple_noise([None, (0.5, keys)], 2, 4) is None                       # draft_cand keeps its own tail: no noise


def test_chain_with_noise(monkeypatch):
    monkeypatch.setattr(dspark, "CAND", 0)
    from kyojin_mlx import rowinv
    monkeypatch.setattr(rowinv, "rowmm", lambda x, w: x @ w.T)                      # the Metal row-invariant product; same values on CPU
    d = bare_drafter()
    S, L, Vd, H = 2, 4, 50, 8
    x = mx.random.normal((S * L, 1, H))
    logits = mx.random.normal((S, L, Vd)) * 3.0
    pend = mx.array([3, 7])
    t0, c0, _ = d._rows_tail(x, logits, pend, S)
    t1, c1, _ = d._rows_tail(x, logits, pend, S, noise=mx.zeros((S, L, Vd)))
    assert t0.tolist() == t1.tolist() and mx.allclose(c0, c1).item()               # zero noise = the argmax chain
    big = mx.zeros((S, L, Vd))
    big[:, 0, 11] = 1e6
    t2, _, _ = d._rows_tail(x, logits, pend, S, noise=big)
    assert t2[:, 0].tolist() == [11, 11]                                           # the noise decides the pick, and the chain continues from it
    t3, _, _ = d._rows_tail(x, logits[:, :2], pend, S, nrows=2, noise=mx.zeros((S, 2, Vd)))     # draft_tail_cut: the head made 2 rows
    assert t3[:, :2].tolist() == t0[:, :2].tolist() and t3[:, 2:].tolist() == [[0, 0], [0, 0]]


# ---- the real draft path (Metal + the ds2m slice pack; skipped elsewhere, like the tail-cut test of test_dspark.py)


def test_real_drafter_coupling_scale_zero_and_noise_effect():
    """Drafter.draft_many(coupling=...) on the real ds2m draft head: scale 0 gives the uncoupled proposals and confidences of the same call; a
    positive scale runs without error, keeps shapes, and the confidences of the first row (computed before any noise acts) are unchanged."""
    import os
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    packs = [p for p in os.environ.get("KYOJIN_SPEC_PACKS", "").split(":") if p]
    pack = next((p for p in packs if os.path.basename(p.rstrip("/")) == "ds2m"), None)
    if pack is None:
        pytest.skip("pack ds2m not in KYOJIN_SPEC_PACKS")
    from kyojin_mlx import patch as P, rows as R, spec as S
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    R.enable(model, spec)
    tc = cfg["text_config"]
    d = dspark.build_drafter(model, pack, spec, cfg, conf_threshold=0.0)
    S.install(model, spec, tap_layers=set(d.tap_ids))
    rng = np.random.default_rng(4)
    prompt = rng.integers(16, tc["vocab_size"] - 16, 135).tolist()
    cache, lg, taps, first = S.prefill(model, prompt, d.tap_ids)
    ctx = d.new_ctx()
    d.ctx = ctx
    d.append([taps[l] for l in d.tap_ids], first)
    mx.eval(ctx.rows)
    pend, p0, depth = int(rng.integers(16, tc["vocab_size"] - 16)), 135, 3
    keys = [mx.random.key(i) for i in range(depth + 1)]
    base = d.draft_many([ctx], [pend], [p0], [depth])
    base_raw = d.last_raw
    zero = d.draft_many([ctx], [pend], [p0], [depth], coupling=[(0.0, keys)])
    assert zero == base and np.allclose(np.array(d.last_raw[1]), np.array(base_raw[1]))
    hot = d.draft_many([ctx], [pend], [p0], [depth], coupling=[(1.0, keys)])
    assert len(hot) == 1 and len(hot[0]) <= depth
    assert abs(d.last_raw[1][0][0] - base_raw[1][0][0]) < 1e-6                      # confidence of row 0 depends on the pending token's embedding only
