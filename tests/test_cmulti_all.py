"""cloud/cmulti-2, the four levers together: a decoder, a parent with children on a shared prefix, an unrelated long prompt, a tight KV budget. Every text equals its solo run."""
import random
import threading
import time

import pytest

from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect

CH = 16


def solo(ids, max_new, cuts=()):
    eng = Engine(GridFake(), sessions=1, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, max_new=max_new, cuts=cuts))[0]
    finally:
        eng.stop()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_all_levers_on_every_text_is_solo(monkeypatch, seed):
    for k, v in (("KYOJIN_MIX_DECODE", "40"), ("KYOJIN_MIX_SLICE_MS", "1"), ("KYOJIN_PF_SHARE", "1"), ("KYOJIN_KV_ADMIT", "1")):
        monkeypatch.setenv(k, v)
    r = random.Random(seed)
    pre = [r.randrange(1, 90) for _ in range(300 + 7 * seed)]
    reqs = [(pre[:], 20, ()), *[(pre + [r.randrange(1, 90) for _ in range(5 + i)], 20, ()) for i in range(4)],
            ([r.randrange(1, 90) for _ in range(700)], 15, ()), ([r.randrange(1, 90) for _ in range(30)], 250, ()),
            (pre + [3, 4, 5], 10, (100,))]
    be = GridFake(pf_ms=0.01, step_ms=0.3, layers=6)
    eng = Engine(be, sessions=5, ctx=1 << 15, chunk=CH, block=1 << 14, prefix_bytes=0, kv_budget=8 * 2600, kv_refresh=1)
    eng.start()
    out = {}

    def worker(i):
        time.sleep(0.004 * i * (seed % 2))
        ids, n, cuts = reqs[i]
        out[i] = collect(eng.submit(ids, max_new=n, cuts=cuts), timeout=120)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(len(reqs))]
    [t.start() for t in ths]; [t.join() for t in ths]
    st = eng.status()
    eng.stop()
    for i, (ids, n, cuts) in enumerate(reqs):
        assert out[i][1] == "length" and out[i][0] == solo(ids, n, cuts), i
    print({k: st.get(k, 0) for k in ("share_hits", "share_tokens", "mix_slice_rounds", "kv_parks", "kv_drops", "rebuilds", "epochs")}, "peak", st["kv_peak_bytes"])
    assert st.get("share_hits", 0) + st.get("mix_slice_rounds", 0) > 0                         # the levers did run in this test
    assert st["kv_peak_bytes"] <= 8 * 2600 + 8 * 5 * 16          # the budget, plus growth between two measurements
