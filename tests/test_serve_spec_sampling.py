"""Speculative sampling (temperature > 0 with drafts) on a fake backend with SOFT logits: the tokens a sampled session emits must follow the exact
distribution of plain sampling (temperature, top-k, top-p, min-p applied to the target logits), whatever the drafter proposes. The fake's next-token
logits depend on (last token mod CTX), so the empirical conditional distribution per context can be compared with the exact one."""
import math
import time

import numpy as np
import pytest

from kyojin_mlx.serve.sampling import Sampling
from test_serve_engine import V, collect
from test_serve_spec import FakeSpec, SpecPass, run, solo_all

import mlx.core as mx

CTX = 3


def soft(c: int) -> np.ndarray:
    return np.random.default_rng(1000 + c).normal(0.0, 1.6, V).astype(np.float32)


def exact_p(c: int, s: Sampling) -> np.ndarray:
    """Distribution of plain sampling, written from the definition (top-k, then top-p on the kept mass, min-p, temperature)."""
    lg = soft(c).astype(np.float64)
    lp = lg - (lg.max() + math.log(np.exp(lg - lg.max()).sum()))
    keep = np.ones(V, bool)
    if s.top_k and s.top_k > 0:
        keep &= lp >= np.sort(lp)[-s.top_k]
    if s.top_p and 0 < s.top_p < 1:
        pr = np.where(keep, np.exp(lp), 0.0)
        order = np.argsort(-pr)                                  # descending: a token stays when the mass above it is below top_p x the kept mass
        above = np.cumsum(pr[order]) - pr[order]
        drop = np.zeros(V, bool)
        drop[order] = above >= s.top_p * pr.sum()
        keep &= ~drop
    if s.min_p and s.min_p > 0:
        pr = np.exp(lp)
        keep &= pr >= s.min_p * pr[keep].max()
    z = np.where(keep, lp / s.temperature, -np.inf)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


class FakeSoft(FakeSpec):
    """FakeSpec with soft logits and a drafter that proposes the most likely token (wrong at positions p with p % err == 0)."""

    def _rows(self, hist_list):
        return mx.array(np.stack([soft(h[-1] % CTX) for h in hist_list]), dtype=mx.float32)

    def prefill_chunk(self, cache, ids, draft=None):
        if draft is not None:
            draft["fed"] += len(ids)
        cache["h"].extend(ids)
        return self._rows([cache["h"]])

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return self._rows([c["h"] for c in bc])

    def draft(self, d, pending, p0, depth, history):
        ids, out = history
        self.depth_asked.append(depth)
        res, last = [], pending
        for i in range(depth):
            t = int(np.argmax(soft(last % CTX)))
            if self.err and (p0 + i) % self.err == 0:
                t = (t + 5) % V
            res.append(t)
            last = t
        return res

    def spec_verify_bc(self, bc, token_lists):
        self.passes.append(sum(len(t) for t in token_lists))
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [soft((cache["h"] + toks[:j + 1])[-1] % CTX) for j in range(len(toks))]
        return SpecPass(mx.array(np.stack(rows), dtype=mx.float32), starts, bc, token_lists)


def transitions(seq, first_prev):
    prev = first_prev
    for t in seq:
        yield prev % CTX, t
        prev = t


def check(counts, s, tv_max, label):
    chi_tot, dof = 0.0, 0
    for c in range(CTX):
        p = exact_p(c, s)
        n = counts[c].sum()
        assert n > 5000, (label, c, n)
        assert counts[c][p == 0].sum() == 0, f"{label}: token outside the support of context {c}"
        tv = 0.5 * np.abs(counts[c] / n - p).sum()
        exp = n * p
        ok = exp >= 5
        chi = (((counts[c][ok] - exp[ok]) ** 2) / exp[ok]).sum() + (counts[c][~ok].sum() - exp[~ok].sum()) ** 2 / max(exp[~ok].sum(), 1e-9)
        chi_tot += chi
        dof += int(ok.sum())
        assert tv < tv_max, f"{label}: TV {tv:.4f} in context {c}"
    # chi-square of the pooled statistic: mean dof, sd sqrt(2 dof); 6 sd is a very safe bound for a deterministic seed
    assert chi_tot < dof + 6 * math.sqrt(2 * dof) + 10, f"{label}: chi2 {chi_tot:.0f} for {dof} dof"


CASES = [
    (Sampling(1.0, 1.0, 0, 0.0), 3),
    (Sampling(0.7, 0.95, 20, 0.0), 3),
    (Sampling(1.3, 0.8, 0, 0.0), 2),
    (Sampling(0.9, 1.0, 8, 0.02), 4),
]


@pytest.mark.parametrize("err", [0, 3])
@pytest.mark.parametrize("s,depth", CASES)
def test_speculative_sampling_follows_the_exact_distribution(s, depth, err):
    n_req, per = 10, 10000                                       # 1e5 sampled tokens
    reqs = {f"r{i}": ([1 + i, 2, 3 + i], dict(max_new=per, sampling=Sampling(s.temperature, s.top_p, s.top_k, s.min_p, 100 + i))) for i in range(n_req)}
    got, st, be = run(reqs, FakeSoft(err=err), sessions=4, ctx=20000, mtp_depth=depth, delay=0, prefix_bytes=0, chunk=8)
    counts = np.zeros((CTX, V), np.int64)
    for i in range(n_req):
        ids = reqs[f"r{i}"][0]
        toks, why, _ = got[f"r{i}"]
        assert why == "length" and len(toks) == per
        for c, t in transitions(toks, ids[-1]):
            counts[c][t] += 1
    assert st["spec_accepted"] > 0 and st["spec_drafted"] > 0         # sampled sessions really drafted
    check(counts, s, 0.06, f"T={s.temperature} p={s.top_p} k={s.top_k} err={err}")


def test_negative_control_wrong_distribution_is_caught():
    s = Sampling(1.0, 1.0, 0, 0.0)
    counts = np.zeros((CTX, V), np.int64)
    rng = np.random.default_rng(0)
    for c in range(CTX):
        counts[c] = rng.multinomial(20000, exact_p(c, Sampling(0.8, 1.0, 0, 0.0)))     # drawn at the wrong temperature
    with pytest.raises(AssertionError):
        check(counts, s, 0.06, "control")


def test_greedy_unchanged_and_mixed_batch():
    g = Sampling(0.0)
    reqs = {"g1": ([1, 2, 3, 4], dict(max_new=60)), "g2": ([9, 8, 7], dict(max_new=45, sampling=g)),
            "s1": ([5, 6, 7], dict(max_new=3000, sampling=Sampling(0.8, 0.9, 30, 0.0, 5))),
            "s2": ([4, 4, 2], dict(max_new=3000, sampling=Sampling(1.1, 1.0, 0, 0.0, 6)))}
    want = solo_all({k: reqs[k] for k in ("g1", "g2")}, backend=FakeSoft(err=0))
    got, st, be = run(reqs, FakeSoft(err=3), sessions=4, ctx=8000, mtp_depth=3, delay=0, prefix_bytes=0)
    for k in ("g1", "g2"):
        assert got[k][0] == want[k][0] and got[k][1] == "length", k          # greedy tokens do not depend on D nor on sampled neighbours
    counts = np.zeros((CTX, V), np.int64)
    for k in ("s1", "s2"):
        for c, t in transitions(got[k][0], reqs[k][0][-1]):
            counts[c][t] += 1
    assert st["spec_accepted"] > 0
    for c in range(CTX):
        n = counts[c].sum()
        assert n > 1000
    # the two sampled sessions use different transforms: compare each against its own exact distribution
    for k in ("s1", "s2"):
        cnt = np.zeros((CTX, V), np.int64)
        for c, t in transitions(got[k][0], reqs[k][0][-1]):
            cnt[c][t] += 1
        sm = reqs[k][1]["sampling"]
        for c in range(CTX):
            n = cnt[c].sum()
            assert cnt[c][exact_p(c, sm) == 0].sum() == 0
            assert 0.5 * np.abs(cnt[c] / n - exact_p(c, sm)).sum() < 0.12, (k, c)


def test_seeded_stream_is_reproducible():
    s = Sampling(0.9, 0.95, 20, 0.0, 77)
    a = run({"a": ([1, 2, 3], dict(max_new=200, sampling=s))}, FakeSoft(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)[0]["a"][0]
    b = run({"a": ([1, 2, 3], dict(max_new=200, sampling=s))}, FakeSoft(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)[0]["a"][0]
    assert a == b and len(a) == 200
