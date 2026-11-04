"""cloud/cmulti-2, sub-agents on a shared prefix: flag pf_share. Grid-sensitive fake: any chunk partition that differs from the session's solo run changes the text."""
import threading
import time

import pytest

from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake
from test_serve_engine import collect

CH, BLK = 8, 1024                                   # chunk 8, memory block far above the prompts: only the share path can reuse a prefix


def solo(ids, max_new=6, cuts=(), chunk=CH):
    eng = Engine(GridFake(), sessions=1, ctx=1 << 14, chunk=chunk, block=BLK, prefix_bytes=0)
    eng.start()
    try:
        return collect(eng.submit(ids, max_new=max_new, cuts=cuts))[0]
    finally:
        eng.stop()


def prompts(P=77, n=4, suffix=5):
    pre = [(5 * i + 2) % 90 for i in range(P)]
    kids = [pre + [(31 * k + 7 * j + 1) % 90 for j in range(suffix + k)] for k in range(n)]
    return pre, kids


def burst(monkeypatch, share, prompt_list, sessions=8, prefix_bytes=0, order=None, max_new=6, **ekw):
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1" if share else "0")
    be = GridFake()
    eng = Engine(be, sessions=sessions, ctx=1 << 14, chunk=CH, block=BLK, prefix_bytes=prefix_bytes, **ekw)
    hs = [eng.submit(p, max_new=max_new) for p in prompt_list]          # all queued before the loop runs: they arrive at once
    eng.start()
    outs = [collect(h)[0] for h in hs]
    st = eng.status()
    eng.stop()
    return outs, st, be


def test_children_start_from_the_parent_cache_without_a_new_prefill(monkeypatch):
    pre, kids = prompts()
    allp = [pre] + kids
    off, st_off, _ = burst(monkeypatch, False, allp)
    on, st_on, be = burst(monkeypatch, True, allp)
    want = [solo(p) for p in allp]
    assert on == want and off == want                                    # texts equal the solo runs, flag on and off
    ideal = len(pre) + sum(len(k) - 72 for k in kids)                    # 72 = last chunk end of the parent that is on every child's grid (77 is not)
    assert st_on["prefill_tokens"] == ideal, st_on["prefill_tokens"]
    assert st_off["prefill_tokens"] == len(pre) + sum(len(k) for k in kids)
    assert st_on["share_hits"] == len(kids) and st_on["share_seeds"] == 1
    assert be.forks == 1 + len(kids)                                      # one hand-off view + one per child: a count independent of the prefix length
    print("prefill tokens off/on:", st_off["prefill_tokens"], st_on["prefill_tokens"])


def test_prefix_length_does_not_change_the_fork_count(monkeypatch):
    forks = []
    for P in (40, 400, 4000):
        pre, kids = prompts(P=P, n=3)
        _, st, be = burst(monkeypatch, True, [pre] + kids)
        forks.append(be.forks)
    assert forks == [4, 4, 4]


def test_children_first_one_becomes_the_leader(monkeypatch):
    pre, kids = prompts()
    on, st, _ = burst(monkeypatch, True, kids)                            # no parent at all: the first child prefills the shared part once
    assert on == [solo(k) for k in kids]
    assert st["share_hits"] == len(kids) - 1
    assert st["prefill_tokens"] == len(kids[0]) + sum(len(k) - 72 for k in kids[1:])


def test_more_children_than_slots(monkeypatch):
    pre, kids = prompts(n=6)
    on, st, _ = burst(monkeypatch, True, [pre] + kids, sessions=2)
    assert on == [solo(p) for p in [pre] + kids]
    assert st["share_hits"] >= 1                                          # the ones waiting for a slot are handed the cache too


def test_late_child_falls_back_and_stays_exact(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
    pre, kids = prompts(P=200)
    be = GridFake(pf_ms=0.05)
    eng = Engine(be, sessions=8, ctx=1 << 14, chunk=CH, block=BLK, prefix_bytes=0)
    eng.start()
    h0 = eng.submit(pre, max_new=3)
    collect(h0)                                                           # the parent is done before the child arrives
    out = collect(eng.submit(kids[0], max_new=6))[0]
    st = eng.status()
    eng.stop()
    assert out == solo(kids[0]) and st.get("share_hits", 0) == 0


def test_off_grid_cut_snapshot_is_not_reused_with_pf_path(monkeypatch):
    """The pre-existing divergence (tests/test_cmulti_base.py) is gone with pf_path=1: the other prompt does not hit the cut it has no cut for."""
    monkeypatch.setenv("KYOJIN_PF_PATH", "1")
    base = list(range(1, 38))
    a, b = base + [50, 51, 52], base + [60, 61, 62]
    eng = Engine(GridFake(), sessions=1, ctx=4096, chunk=8, block=16, prefix_bytes=1 << 20)
    eng.start()
    try:
        collect(eng.submit(a, max_new=2, cuts=(11,)))
        tb = collect(eng.submit(b, max_new=8))[0]
        skipped = eng.prefix.off_grid_skipped
    finally:
        eng.stop()
    assert tb == solo(b, max_new=8) and skipped >= 1


def test_the_cut_a_prompt_owns_is_still_reused_with_pf_path(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_PATH", "1")
    base = list(range(1, 38))
    a, b = base + [50, 51, 52], base + [60, 61, 62]
    eng = Engine(GridFake(), sessions=1, ctx=4096, chunk=8, block=16, prefix_bytes=1 << 20)
    eng.start()
    try:
        collect(eng.submit(a, max_new=2, cuts=(11,)))
        tb = collect(eng.submit(b, max_new=8, cuts=(11,)))[0]
        hits = eng.prefix.hits
    finally:
        eng.stop()
    assert tb == solo(b, max_new=8, cuts=(11,)) and hits >= 1


def test_budget_refuses_the_seed_not_the_request(monkeypatch):
    pre, kids = prompts()
    on, st, _ = burst(monkeypatch, True, [pre] + kids, kv_budget=1000)
    assert on == [solo(p) for p in [pre] + kids]
    assert st.get("share_refused_budget", 0) > 0


def test_single_session_trace_is_the_same_with_the_flag(monkeypatch):
    pre, kids = prompts()
    traces = []
    for share in (False, True):
        _, _, be = burst(monkeypatch, share, [kids[0]], sessions=1)
        traces.append(be.calls)
    assert traces[0] == traces[1]


def test_with_the_prefix_cache_on_and_the_flag_on(monkeypatch):
    pre, kids = prompts(P=300, n=3)
    on, st, _ = burst(monkeypatch, True, [pre] + kids, prefix_bytes=1 << 20)
    assert on == [solo(p) for p in [pre] + kids]


def test_leader_cut_the_child_lacks_limits_the_handoff_to_the_common_path(monkeypatch):
    """The leader's chunks end at 8, 11 (its cut), 16...; the child's at 8, 16...: the paths agree only up to 8, so the child may start from 8 and not from 72."""
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
    pre, kids = prompts()
    eng = Engine(GridFake(), sessions=4, ctx=1 << 14, chunk=CH, block=BLK, prefix_bytes=0)
    hp = eng.submit(pre, max_new=3, cuts=(11,))
    hk = eng.submit(kids[0], max_new=6)
    eng.start()
    tp, tk = collect(hp)[0], collect(hk)[0]
    st = eng.status()
    eng.stop()
    assert tp == solo(pre, max_new=3, cuts=(11,)) and tk == solo(kids[0])
    assert st["share_hits"] == 1 and st["share_tokens"] == 8


def test_same_cut_on_both_is_handed_over(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
    pre, kids = prompts()
    eng = Engine(GridFake(), sessions=4, ctx=1 << 14, chunk=CH, block=BLK, prefix_bytes=0)
    hp = eng.submit(pre, max_new=3, cuts=(11,))
    hk = [eng.submit(k, max_new=6, cuts=(11,)) for k in kids]
    eng.start()
    tp, tk = collect(hp)[0], [collect(h)[0] for h in hk]
    st = eng.status()
    eng.stop()
    assert tp == solo(pre, max_new=3, cuts=(11,)) and tk == [solo(k, cuts=(11,)) for k in kids]
    assert st["share_hits"] == len(kids)


def turn_prefill_tokens(path_flag):
    """The chat layer's cuts: end of the system block and the whole conversation without the generation prompt. Turn 2 extends turn 1, so turn 1's last cut (90)
    is NOT a cut of turn 2 (its cuts are 20 and 140). Returns (prefill tokens of turn 2, its text, the solo text of turn 2)."""
    import os
    os.environ["KYOJIN_PF_PATH"] = path_flag
    try:
        sysp = [(3 * i + 1) % 90 for i in range(20)]
        t1 = sysp + [(7 * i + 2) % 90 for i in range(70)] + [11, 12, 13]
        t2 = t1[:90] + [(5 * i + 9) % 90 for i in range(50)] + [14, 15]
        be = GridFake()
        eng = Engine(be, sessions=1, ctx=1 << 14, chunk=CH, block=BLK, prefix_bytes=1 << 20)
        eng.start()
        try:
            collect(eng.submit(t1, max_new=3, cuts=(20, 90)))
            n0 = eng.stats["prefill_tokens"]
            txt = collect(eng.submit(t2, max_new=6, cuts=(20, 140)))[0]
            n = eng.stats["prefill_tokens"] - n0
        finally:
            eng.stop()
        return n, txt, solo(t2, cuts=(20, 140))
    finally:
        os.environ.pop("KYOJIN_PF_PATH", None)


def test_pf_path_cost_on_a_second_chat_turn():
    off_n, off_txt, want = turn_prefill_tokens("0")
    on_n, on_txt, _ = turn_prefill_tokens("1")
    print("turn 2 prefill tokens, pf_path off/on:", off_n, on_n, "| text equals solo off/on:", off_txt == want, on_txt == want)
    assert on_txt == want                                   # exact with the path filter
    assert off_n == 142 - 90 and on_n == 142 - 20          # off: restarts at turn 1 cut (90); on: only the system cut (20) is on turn 2 path
    assert off_txt != want                                  # on the fake, the off-path restart differs from the solo partition (pre-existing)


def test_pf_share_alone_does_not_touch_a_second_turn(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
    n, txt, want = turn_prefill_tokens("0")
    assert n == 142 - 90
