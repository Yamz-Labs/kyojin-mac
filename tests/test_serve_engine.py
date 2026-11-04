"""Continuous-batching engine on a fake backend: bookkeeping, fairness, prefix hits, cancel (needs mlx, no model)."""
import threading
import time

import mlx.core as mx
import pytest

from kyojin_mlx.serve.backends import Backend
from kyojin_mlx.serve.engine import Engine, RQ_MIN_TOKENS, rq_budget
from kyojin_mlx.serve.sampling import Sampling, sample_rows

V = 97


def nxt(hist):
    return (sum((i + 3) * t for i, t in enumerate(hist)) * 31 + len(hist)) % V


def onehot(rows):
    out = [[0.0] * V for _ in rows]
    for r, t in zip(out, rows):
        r[t] = 10.0
    return mx.array(out, dtype=mx.float32)


class Fake(Backend):
    """Next token = a hash of the whole history: any lost, repeated or misplaced token changes everything after it."""
    kind = "fake"
    max_rows = 8

    def make_cache(self):
        return {"h": []}

    def snapshot(self, cache):

        return self.fork(cache)


    def fork(self, cache):
        return {"h": list(cache["h"])}

    def prefill_chunk(self, cache, ids):
        cache["h"].extend(ids)
        return onehot([nxt(cache["h"])])

    def batch(self, caches):
        return list(caches)

    def unbatch(self, bc, n):
        return list(bc)

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt(c["h"]) for c in bc])

    def state(self, cache):
        return []

    def nbytes(self, cache):
        return sum(8 * len(c["h"]) for c in cache) if isinstance(cache, list) else 8 * len(cache["h"])      # a batched cache is a list


def collect(h, timeout=120):
    toks = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        ev = h.events.get(timeout=timeout)
        if ev[0] == "tok":
            toks.append(ev[1])
        else:
            return toks, ev[1], ev[2]
    raise TimeoutError


def solo(ids, **kw):
    eng = Engine(Fake(), sessions=1, ctx=kw.pop("ctx", 4096), chunk=kw.pop("chunk", 4), block=8, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, **kw))
    finally:
        eng.stop()


def test_rq_budget_cases():
    assert rq_budget(32768, 1, 100, 10 ** 6) is None                       # one session: no sharing
    assert rq_budget(32768, 4, 100, 100) is None                           # the reply fits the share
    b = rq_budget(32768, 4, 100, 30000)
    assert b == 32768 // 4 // 256 * 256 - 100 == 8092                      # share of the context, minus the prompt
    assert rq_budget(32768, 4, 7000, 30000) == RQ_MIN_TOKENS               # never below the floor
    assert rq_budget(4096, 4, 100, 10 ** 6) == RQ_MIN_TOKENS
    assert rq_budget(32768, 4, 32700, 10 ** 6) == 32768 // 256 * 256 - 32700   # room, not floor, caps it


def test_churn_equals_solo():
    prompts = {n: [(n * 7 + i * 3) % V for i in range(10 + 5 * n)] for n in range(6)}
    limits = {0: 40, 1: 5, 2: 60, 3: 1, 4: 25, 5: 33}
    want = {n: solo(prompts[n], max_new=limits[n]) for n in prompts}
    eng = Engine(Fake(), sessions=3, ctx=4096, chunk=4, block=8, prefix_bytes=1 << 20)
    eng.start()
    out = {}

    def worker(n):
        time.sleep(0.01 * n)
        out[n] = collect(eng.submit(prompts[n], max_new=limits[n]))
    ths = [threading.Thread(target=worker, args=(n,)) for n in prompts]
    [t.start() for t in ths]; [t.join() for t in ths]
    st = eng.status()
    eng.stop()
    for n in prompts:
        assert out[n][0] == want[n][0], n
        assert out[n][1] == want[n][1] == "length"
        assert len(out[n][0]) == limits[n]
    assert st["epochs"] >= 2 and st["waiting"] == 0 and st["decoding"] == 0


def test_stop_id_and_usage():
    ids = list(range(12))
    full, _, _ = solo(ids, max_new=30)
    stop = full[7]
    toks, reason, stats = solo(ids, max_new=30, stop_ids={stop})
    assert reason == "stop" and toks == full[:full.index(stop)]
    assert stats["prompt_tokens"] == 12 and stats["completion_tokens"] == len(toks)


def test_lone_request_reaches_the_end_of_the_context():
    eng = Engine(Fake(), sessions=4, ctx=3000, chunk=8, block=16, prefix_bytes=0)
    eng.start()
    toks, reason, stats = collect(eng.submit(list(range(10)), max_new=None))
    st = eng.status()
    eng.stop()
    assert reason == "length" and len(toks) == 3000 - 10 - 1 and "yields" not in st


def test_share_is_kept_when_others_wait():
    ctx, prompt = 6144, list(range(10))
    long_want = [solo(prompt[:], max_new=None, ctx=ctx), solo(prompt[::-1], max_new=None, ctx=ctx)]
    eng = Engine(Fake(), sessions=2, ctx=ctx, chunk=8, block=16, prefix_bytes=0)
    eng.start()
    a = eng.submit(prompt[:], max_new=None)
    b = eng.submit(prompt[::-1], max_new=None)
    time.sleep(0.2)
    c = eng.submit([5, 6, 7, 8], max_new=6)                      # arrives when both slots are busy
    tc = collect(c)
    ta, tb = collect(a), collect(b)
    st = eng.status()
    eng.stop()
    assert len(tc[0]) == 6 and tc[1] == "length"                  # served long before the long replies end
    assert st.get("yields", 0) >= 1
    assert ta[0] == long_want[0][0] and tb[0] == long_want[1][0]  # yielding and resuming change nothing
    assert len(ta[0]) >= rq_budget(ctx, 2, 10, ctx) and len(tb[0]) >= rq_budget(ctx, 2, 10, ctx)


def test_prefix_hit_counts_and_equals_miss():
    shared = [(i * 5) % V for i in range(40)]
    a, b = shared + [1, 2, 3], shared + [9, 8]
    want_a, want_b = solo(a, max_new=12), solo(b, max_new=12)
    eng = Engine(Fake(), sessions=2, ctx=4096, chunk=4, block=8, prefix_bytes=1 << 20)
    eng.start()
    ra = collect(eng.submit(a, max_new=12))
    rb = collect(eng.submit(b, max_new=12))
    st = eng.status()
    eng.stop()
    assert ra[0] == want_a[0] and rb[0] == want_b[0]
    assert ra[2]["cached_tokens"] == 0 and rb[2]["cached_tokens"] == 40       # the longest snapshot below len(b)
    assert st["prefix_hits"] == 1


def test_cancel_frees_the_slot():
    eng = Engine(Fake(), sessions=1, ctx=100000, chunk=8, block=16, prefix_bytes=0)
    eng.start()
    h = eng.submit(list(range(10)), max_new=None)
    for _ in range(5):
        assert h.events.get(timeout=30)[0] == "tok"
    h.cancel()
    toks, reason, _ = collect(h)
    assert reason == "cancelled"
    toks2, reason2, _ = collect(eng.submit([1, 2, 3], max_new=4))        # the single slot is free again
    eng.stop()
    assert len(toks2) == 4 and reason2 == "length"


def test_submit_validation():
    eng = Engine(Fake(), sessions=1, ctx=64, chunk=4, block=8, prefix_bytes=0)
    with pytest.raises(ValueError):
        eng.submit([])
    with pytest.raises(ValueError):
        eng.submit(list(range(64)))


def test_sampling_rows_do_not_depend_on_the_batch():
    mx.random.seed(0)
    logits = mx.random.normal((4, 500)) * 3
    params = [Sampling(0.8, 0.9, 40, 0.02, 1), Sampling(1.2, 1.0, 0, 0.0, 2), Sampling(0.0), Sampling(0.7, 0.95, 20, 0.05, 3)]
    keys = [mx.random.key(10 + i) for i in range(4)]
    both = sample_rows(logits, params, keys).tolist()
    for i in range(4):
        alone = sample_rows(logits[i:i + 1], [params[i]], [keys[i]]).tolist()
        assert alone == [both[i]]
    assert both[2] == int(mx.argmax(logits[2]).item())


class CountUnbatch(Fake):
    n_unbatch = 0

    def unbatch(self, bc, n):
        self.n_unbatch += 1
        return super().unbatch(bc, n)


def test_finish_without_write_back(monkeypatch):
    """Flag fin_nosync: a batch that ends whole never writes its cache back; a member that stays still gets it, and the tokens are those of a solo run."""
    prompts = {0: [3, 5, 7, 9] * 3, 1: [2, 4, 6] * 4}
    want = {n: solo(prompts[n], max_new=m) for n, m in ((0, 12), (1, 30))}
    be = CountUnbatch()
    eng = Engine(be, sessions=2, ctx=4096, chunk=4, block=8, prefix_bytes=0)
    eng.start()
    assert collect(eng.submit(prompts[0], max_new=12))[0] == want[0][0]
    assert be.n_unbatch == 0
    out = {}
    ths = [threading.Thread(target=lambda n=n, m=m: out.__setitem__(n, collect(eng.submit(prompts[n], max_new=m)))) for n, m in ((0, 12), (1, 30))]
    [t.start() for t in ths]; [t.join() for t in ths]
    eng.stop()
    assert out[0][0] == want[0][0] and out[1][0] == want[1][0]
