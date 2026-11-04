"""Speculative rejection sampling (kyojin_mlx.specsam, KYOJIN_SPEC_RS) and the verify-depth levers on synthetic logits: the tokens a proposal q
hands back must follow the target p exactly, acceptance must be sum_x min(p, q) (not p(top1)), and the served path (engine) must keep the greedy
tokens of a plain session identical with spec_rs / spec_full / spec_gate on or off."""
import types

import mlx.core as mx
import numpy as np
import pytest

from kyojin_mlx import specsam
from kyojin_mlx.serve.backends import LazyDrafts
from kyojin_mlx.serve.sampling import Sampling
from test_serve_engine import V, nxt, onehot
from test_serve_spec import FakeSpec, run, solo_all

LOGITS = np.array([[2.0, 1.5, 1.0, 0.6, 0.2, -0.4, -1.0, -2.0]], dtype=np.float32)     # target: peaked, top1 mass ~0.33
DRAFT = LOGITS + np.array([[0.0, -0.2, 0.9, 0.1, 0.0, 0.3, -0.5, 0.2]], dtype=np.float32)   # a drafter that disagrees with it


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def keys(n, seed=0):
    return [mx.random.split(mx.random.key(seed + i))[1] for i in range(n)]


def hist(tokens, v):
    h = np.bincount(np.asarray(tokens), minlength=v).astype(np.float64)
    return h / h.sum()


# ------------------------------------------------------------------------------------------------ the maths
def sample_many(plogits, qlogits, sampling, m, seed=0):
    """m independent rounds of one session: arrays -> (tokens [m, n] as ints, acc [m, n-1] bools)."""
    lq = specsam.logq(mx.array(qlogits), sampling.temperature, sampling.top_k)
    pl = np.asarray(plogits, dtype=np.float32)
    if pl.ndim == 1:                                                      # one target row: the draft rows and the bonus row share it
        pl = np.stack([pl] * (lq.shape[0] + 1))
    pl = mx.array(pl)
    outs, accs = [], []
    for i in range(m):
        ks = keys(pl.shape[0] + 2, seed=seed + 7919 * i)
        x = specsam.draw(lq, ks[:lq.shape[0]])
        out, acc = specsam.decide(pl, lq, x, sampling, ks)
        outs.append(out.tolist())
        accs.append(acc.tolist())
    return np.array(outs), np.array(accs)


def test_logq_is_tempered_and_top_k_truncated():
    lg = mx.array(LOGITS)
    q = np.array(mx.exp(specsam.logq(lg, 1.0)))
    assert np.allclose(q, softmax(LOGITS[0]), atol=1e-6)
    q3 = np.array(mx.exp(specsam.logq(lg, 1.0, 3)))[0]
    keep = np.argsort(-LOGITS[0])[:3]
    assert np.allclose(q3[keep], softmax(LOGITS[0][keep]), atol=1e-6) and q3.sum() == pytest.approx(1.0, abs=1e-5)
    assert q3[np.setdiff1d(np.arange(8), keep)].sum() == 0.0


def test_draw_follows_q():
    lq = specsam.logq(mx.array(DRAFT), 1.0)
    ks = keys(20000)
    toks = np.array(specsam.draw(mx.repeat(lq, len(ks), axis=0), ks))
    assert np.abs(hist(toks, 8) - softmax(DRAFT[0])).max() < 0.02


def test_decide_emits_the_target_distribution_and_accepts_sum_min():
    s = Sampling(1.0)
    p, q = softmax(LOGITS[0]), softmax(DRAFT[0])
    out, acc = sample_many(LOGITS[0], DRAFT, s, 6000)                     # n = 2 rows: the draft row and the bonus row
    assert np.abs(hist(out[:, 0], 8) - p).max() < 0.04                     # the emitted token is p, not q and not argmax(p)
    assert abs(acc[:, 0].mean() - float(np.minimum(p, q).sum())) < 0.03    # P(accept) = 1 - TV(p, q)
    assert acc[:, 0].mean() > p.max() + 0.05                              # the point of the lever: more than p(top1)


@pytest.mark.parametrize("temp", [0.7, 1.3])
def test_decide_is_exact_at_the_sampler_temperature(temp):
    """q must be the very distribution the drafts were drawn from: tempering it a second time inside decide would break exactness."""
    s = Sampling(temp)
    p = softmax(LOGITS[0] / temp)
    out, acc = sample_many(LOGITS[0], DRAFT, s, 6000, seed=31)
    q = np.array(mx.exp(specsam.logq(mx.array(DRAFT), temp)))
    assert np.abs(hist(out[:, 0], 8) - p).max() < 0.04
    assert abs(acc[:, 0].mean() - float(np.minimum(p, q[0]).sum())) < 0.03


def test_second_row_is_the_target_conditional_on_the_emitted_prefix():
    s = Sampling(1.0)
    p0, p1, q = softmax(LOGITS[0]), softmax(DRAFT[0][::-1]), softmax(DRAFT[0])
    pl = np.stack([LOGITS[0], DRAFT[0][::-1], LOGITS[0]])                  # row 1's target is a different p1
    out, acc = sample_many(pl, DRAFT, s, 6000, seed=500)
    assert np.abs(hist(out[:, 0], 8) - p0).max() < 0.04
    got = out[acc[:, 0], 1]                                               # row 1 as emitted, over the accepted row 0
    assert len(got) > 2000 and np.abs(hist(got, 8) - p1).max() < 0.05       # exactly p1 whatever q was
    assert abs(acc[:, 0].mean() - float(np.minimum(p0, q).sum())) < 0.03


def test_residual_carries_the_mass_q_puts_elsewhere():
    """q (top-2 truncated) excludes p's argmax and keeps only two wrong tokens: proposals are mostly rejected and the residual still gives p."""
    s = Sampling(1.0)                                                     # p keeps its whole support (no filter on the target)
    p = softmax(LOGITS[0])
    dq = np.array([[-6.0, -6.0, 0.0, 0.5, -6.0, -6.0, -6.0, -6.0]], dtype=np.float32)   # q keeps tokens 3 and 2 only
    lq = specsam.logq(mx.array(dq), 1.0, 2)
    pl = mx.array(np.stack([LOGITS[0]] * 2))
    outs, accs = [], []
    for i in range(6000):
        ks = keys(4, seed=9 + 7919 * i)
        x = specsam.draw(lq, ks[:1])
        out, acc = specsam.decide(pl, lq, x, s, ks)
        outs.append(out.tolist())
        accs.append(acc.tolist())
    out, acc = np.array(outs), np.array(accs)
    q = np.array(mx.exp(lq))[0]
    assert abs(acc[:, 0].mean() - float(np.minimum(p, q).sum())) < 0.03
    assert acc[:, 0].mean() < 0.5 and np.abs(hist(out[:, 0], 8) - p).max() < 0.04   # p's mass outside q comes back via the residual


def test_gate_threshold_follows_the_calibration():
    assert specsam.break_even(5.5, 22.0) == pytest.approx(0.25)
    assert specsam.gate_confidence(None, 0.25) is None

    class Cal:
        ready = True
        def prob(self, c):                                               # an under-confident head: 0.2 already means 0.5 acceptance
            return min(1.0, 2.5 * c)
    assert specsam.gate_confidence(Cal(), 0.25) == pytest.approx(0.10, abs=0.011)


# ------------------------------------------------------------------------------------------------ the served path
class _Be:
    """The bits of a backend that LazyDrafts reads: the drafter's gate and the cut counter."""
    def __init__(self, keep):
        self.drafter = types.SimpleNamespace(conf_keep=self._keep)
        self.keep = keep

    def _keep(self, toks, conf, depths):
        return [min(d, self.keep) for d in depths]

    def _note_cut(self, cut):
        pass


class FakeLazy(FakeSpec):
    """FakeSpec whose draft pass is the lazy one: a full block of true-continuation proposals, a confidence gate that keeps `gate` positions,
    and (want_logits) block logits q peaking on the proposal."""
    kind = "fakelazy"

    def __init__(self, err=0, block=4, gate=1):
        super().__init__(err=err)
        self.block, self.gate = block, gate
        self.want_logits = None

    def draft_lazy(self, jobs, want_logits=False):
        self.want_logits = want_logits
        live = [i for i, j in enumerate(jobs) if j[3] > 0]
        if not live:
            return None
        toks, confs = [], []
        for i in live:
            d, pending, p0, depth, (ids, out) = jobs[i]
            h, dr = list(ids) + list(out), []
            for _ in range(self.block):
                dr.append(nxt(h))
                h = h + [dr[-1]]
            toks.append(dr)
            confs.append([0.9 if j < self.gate else 0.1 for j in range(self.block)])
        lg = None
        if want_logits:
            z = np.zeros((len(live), self.block, V), dtype=np.float32)
            for i, ts in enumerate(toks):
                z[i, np.arange(self.block), ts] = 4.0
            lg = mx.array(z)
        return LazyDrafts(_Be(self.gate), jobs, live, mx.array(toks, dtype=mx.int32), mx.array(confs, dtype=mx.float32), lg)

    def spec_verify_lazy(self, bc, lh, pendings):
        lists = [[p] + lh.tolist()[k][:lh.depths[k]] for k, p in enumerate(pendings)]
        return FakeSpec.spec_verify_bc(self, bc, lists)

    def settle(self, bc, drafts=()):
        pass


def reqs(sampling=None):
    return {"a": ([1, 2, 3, 4, 5], dict(max_new=60, **({"sampling": sampling} if sampling else {})))}


def test_spec_full_accepts_every_verified_draft(monkeypatch):
    """The gate keeps one proposal but the lazy pass paid for four rows: spec_full accepts them (more tokens per round), same plain tokens."""
    want = solo_all(reqs())["a"][0]
    st = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("KYOJIN_SPEC_FULL", flag)
        got, s, be = run(reqs(), FakeLazy(err=0), sessions=1, mtp_depth=3, depth_override=3, prefix_bytes=0, ctx=4096)
        assert got["a"][0] == want and got["a"][1] == "length"
        st[flag] = s
    assert st["1"]["spec_accepted"] > st["0"]["spec_accepted"]
    assert st["1"]["spec_drafted"] / st["1"]["spec_rounds"] > st["0"]["spec_drafted"] / st["0"]["spec_rounds"]


def test_spec_rs_drafts_from_q_and_emits_the_target(monkeypatch):
    monkeypatch.setenv("KYOJIN_SPEC_RS", "1")
    s = Sampling(1e-3, 1.0, 0, 0.0, 3)                                    # a near one-hot p: the emitted tokens must be its argmax
    rr = reqs(s)
    want = solo_all(rr)["a"][0]
    got, st, be = run(rr, FakeLazy(err=0), sessions=1, mtp_depth=3, depth_override=3, prefix_bytes=0, ctx=4096)
    assert be.want_logits is True and st["spec_rows"] > st["spec_rounds"]  # the q-sampled path really drafted
    assert got["a"][0] == want and st["spec_accepted"] > 0


@pytest.mark.parametrize("flag", ["KYOJIN_SPEC_RS", "KYOJIN_SPEC_FULL", "KYOJIN_SPEC_GATE"])
def test_greedy_session_is_unchanged_by_the_levers(flag, monkeypatch):
    monkeypatch.setenv(flag, "1")
    rr = reqs()
    want = solo_all(rr)["a"][0]
    got, st, _ = run(rr, FakeSpec(err=3), sessions=1, mtp_depth=3, prefix_bytes=0)
    assert got["a"][0] == want and got["a"][1] == "length"


def test_refine_tail_teacher_forces_the_bigram_bias_and_keeps_logits():
    """KYOJIN_DSPARK_REFINE: the second pass re-picks tokens from the drafted prefix as block ids (each position's bigram bias comes from the
    token it sees, not from the previous proposal), and the same tail hands out the block logits spec_rs draws from."""
    import mlx.nn as nn
    from kyojin_mlx import dspark as D

    class LM(nn.Module):
        def __init__(self, v=8, h=4, rank=2):
            super().__init__()
            self.markov_embed = nn.Embedding(v, rank)
            self.markov_head = nn.Linear(rank, v, bias=False)
            self.confidence = nn.Linear(h + rank, 1, bias=False)
    dr = object.__new__(D.Drafter)
    dr.lm, dr.block, dr.n_stages, dr._conf_w = LM(), 3, 0, None
    ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    x = mx.zeros((3, 1, 4))
    logits = mx.zeros((1, 3, 8))
    toks, conf, lgs = dr._rows_tail(x, logits, ids[:, 0], 1, keep=True, ids=ids)
    want = [int(mx.argmax(dr.lm.markov_head(dr.lm.markov_embed(ids[:, i]))[0]).item()) for i in range(3)]
    assert toks.tolist()[0] == want                                    # the bias follows the input ids (teacher forced)
    chain = dr._rows_tail(x, logits, ids[:, 0], 1, ids=None)[0].tolist()[0]
    assert chain != want                                               # the greedy chain (no refine) drifts off the drafted prefix
    assert lgs.shape == (1, 3, 8) and conf.shape == (1, 3)
