"""Memory guard (CPU, no Metal): budget, projection and admission with injected readers."""
import pytest

from kyojin_mlx.serve import memguard as M

G = M.GIB


def guard(active=100, cache=0, foot=None, budget=225, slope=0.36, clear=None):
    st = {"a": active * G, "c": cache * G}
    g = M.MemGuard(budget * G, slope * G / 1000, lambda: st["a"], lambda: st["c"], (lambda: foot * G) if foot else None, clear)
    return g, st


def test_budget_default_is_88_percent_of_ram():
    assert M.budget_bytes({}, ram=256 * G) == int(0.88 * 256 * G)
    assert M.budget_bytes({"KYOJIN_MEM_BUDGET_GIB": "200"}, ram=256 * G) == 200 * G
    assert M.budget_bytes({"KYOJIN_MEM_BUDGET_GIB": "0"}, ram=256 * G) is None
    assert M.budget_bytes({}, ram=0) is None


def test_slope_env():
    assert M.slope_bytes_per_token({}) == pytest.approx(0.36 * G / 1000)
    assert M.slope_bytes_per_token({"KYOJIN_MEM_GIB_PER_1K": "0.5"}) == pytest.approx(0.5 * G / 1000)


def test_projection_and_max_tokens():
    g, _ = guard(active=150)
    assert g.project(100_000) == pytest.approx(150 * G + 36 * G)
    assert g.max_tokens() == pytest.approx(75 / 0.36 * 1000, abs=2)


def test_admit_refuses_with_a_clear_message():
    g, _ = guard(active=190)
    g.admit(90_000)                                       # 190 + 32.4 = 222.4 <= 225
    with pytest.raises(M.PromptTooLong) as e:
        g.admit(100_000)                                  # 190 + 36 = 226 > 225
    assert "prompt too long for the memory budget" in str(e.value) and "max ~" in str(e.value)


def test_admission_counts_the_footprint_less_the_reclaimable_cache():
    g, _ = guard(active=100, cache=10, foot=170)          # the process holds 160 GiB beyond its MLX cache that active does not show
    assert g.base() == 160 * G
    g2, _ = guard(active=100, cache=10, foot=105)
    assert g2.base() == 100 * G


def test_over_uses_active_plus_cache_or_footprint():
    g, _ = guard(active=200, cache=20)
    assert not g.over() and g.over(10 * G)
    g, _ = guard(active=100, cache=0, foot=230)
    assert g.over()


def _engine(g):
    import collections
    import itertools
    import threading

    from kyojin_mlx.serve import engine as E
    e = E.Engine.__new__(E.Engine)
    e.memguard, e.ctx, e.sessions, e.rq_floor, e.kv_budget = g, 1 << 20, 1, None, None
    e.stats, e._ids, e.cv, e._inbox = collections.Counter(), itertools.count(1), threading.Condition(), collections.deque()
    e.backend = type("B", (), {})()
    return e


def test_submit_refuses_a_prompt_over_the_budget_and_keeps_the_slot_free():
    g, _ = guard(active=190)
    e = _engine(g)
    with pytest.raises(M.PromptTooLong):
        e.submit(list(range(1, 100_001)))
    assert e.stats["mem_refused"] == 1 and not e._inbox
    h = e.submit(list(range(1, 1001)))                      # a short prompt still goes in
    assert h.session is e._inbox[0]


class _Handle:
    def __init__(self):
        import threading
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()


def _live(sid, kv, n_prompt=1000):
    from types import SimpleNamespace
    return SimpleNamespace(sid=sid, handle=_Handle(), kv_bytes=kv, n_prompt=n_prompt, n_gen=0, mem_abort=False)


def _relief_engine(g, snaps=(), live=()):
    from kyojin_mlx.serve import engine as E
    e = _engine(g)
    e.log, e.prefilling, e.ready, e.members = (lambda *a: None), [], [], []
    e.prefix = E.PrefixCache(e.backend, 512, 100 * G)
    for i, sz in enumerate(snaps):                          # entries the guard may drop, each frees `sz` bytes of the fake meter
        e.prefix.entries[(512 * (i + 1), b"k%d" % i)] = (None, sz, None)
        e.prefix.bytes += sz
    e.members.extend(live)
    return e


def test_under_budget_does_nothing():
    cleared = []
    g, _ = guard(active=100, clear=lambda: cleared.append(1))
    e = _relief_engine(g)
    assert e._mem_relieve(1 * G) is False and not cleared


def test_clearing_the_cache_is_enough():
    g, st = guard(active=200, cache=30)                     # 230 > 225
    g._clear = lambda: st.update(c=0)
    e = _relief_engine(g, snaps=[5 * G])
    assert e._mem_relieve() is False
    assert e.stats["mem_clear"] == 1 and len(e.prefix.entries) == 1


def test_prefix_snapshots_are_dropped_lru_before_any_request_stops():
    g, st = guard(active=230)
    e = _relief_engine(g, snaps=[10 * G, 10 * G, 10 * G, 10 * G], live=[_live(1, 50 * G)])

    def free():                                             # every dropped snapshot hands its bytes back
        st["a"] = 230 * G - (40 * G - e.prefix.bytes)
    g._clear = free
    assert e._mem_relieve() is False                        # 230 -> under 225 once 10 GiB went (oldest first)
    assert e.prefix.bytes == 30 * G or e.prefix.bytes == 20 * G
    assert list(e.prefix.entries)[0][0] > 512               # the LRU end went first
    assert not e.members[0].mem_abort


def test_still_over_after_everything_stops_the_victim_and_not_the_server():
    g, _ = guard(active=240)
    small, big = _live(1, 10 * G), _live(2, 90 * G)
    e = _relief_engine(g, snaps=[5 * G], live=[small, big])
    assert e._mem_relieve() is True
    assert big.mem_abort and big.handle.cancelled.is_set() and not small.mem_abort
    assert not e.prefix.entries and e.stats["mem_aborts"] == 1


def test_prefill_victim_and_next_chunk_projection():
    g, _ = guard(active=223)                                # 223 + 8192 rows x 0.36 GiB / 1K = 226 -> over
    pf = _live(7, 0)
    e = _relief_engine(g, live=[_live(1, 90 * G)])
    assert e._mem_relieve(8192 * g.slope, pf) is True
    assert pf.mem_abort and not e.members[0].mem_abort


def test_finish_maps_a_memory_abort_to_its_own_reason():
    g, _ = guard()
    e = _engine(g)
    e._live = {}
    s = e.submit(list(range(1, 11))).session
    s.mem_abort = True
    e._finish(s, "cancelled")
    ev = s.handle.events.get_nowait()
    assert ev[:2] == ("done", "memory") and e.stats["finished_memory"] == 1
