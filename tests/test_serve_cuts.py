"""Prefix snapshots at cuts (off-grid reuse points given by the chat layer): memory tier, disk tier, and the chat helpers (fake backends, no model)."""
import pytest

from kyojin_mlx.serve import chat as C
from test_serve_engine import Fake, collect
from kyojin_mlx.serve.engine import Engine, PrefixCache, prefix_digests

cache_mod = pytest.importorskip("mlx_lm.models.cache")
from test_serve_kvdisk import FakeKV, disk  # noqa: E402


def eng(be, **kw):
    e = Engine(be, sessions=2, ctx=4096, chunk=4, block=8, prefix_bytes=1 << 20, **kw)
    e.start()
    return e


def test_shared_system_prefix_and_next_turn_hit_at_the_cuts():
    S = [(i * 5 + 1) % 90 + 1 for i in range(10)]                      # system block: shorter than one block, ends off-grid
    r1 = S + [11, 12, 13, 14, 15, 16, 17] + [91, 92, 93]               # + user + generation prompt
    e = eng(Fake())
    try:
        collect(e.submit(r1, max_new=3, cuts=[10, 17]))
        assert e.stats["prefix_hit_tokens"] == 0 and sorted(e.prefix.off) == [10, 17]
        r2 = S + [21, 22, 23, 24, 25, 26]
        got = collect(e.submit(r2, max_new=3, cuts=[10]))[0]
        assert e.stats["prefix_hit_tokens"] == 10                      # 2nd request: system cut
        r3 = r1[:17] + [31, 32, 33, 34, 35] + [41, 42, 43, 44]
        collect(e.submit(r3, max_new=3, cuts=[10, 17, 22]))
        assert e.stats["prefix_hit_tokens"] == 10 + 17                 # 3rd request: end of request 1 without the generation prompt
    finally:
        e.stop()
    e2 = eng(Fake())
    try:
        assert collect(e2.submit(r2, max_new=3))[0] == got              # a hit gives the tokens of a plain prefill
    finally:
        e2.stop()


def test_cut_snapshots_follow_the_lru_budget():
    e = eng(Fake())
    try:
        e.prefix.budget = 8 * 12                                        # room for one small snapshot
        collect(e.submit(list(range(1, 21)), max_new=1, cuts=[10, 17]))
        assert e.prefix.bytes <= e.prefix.budget
        assert set(e.prefix.off) == {n for n, _ in e.prefix.entries if n % 8}
    finally:
        e.stop()


def test_disk_hit_at_a_cut_after_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")                                 # exact snapshot sets of the cuts alone (the tail anchor: tests/test_serve_kvdisk.py)
    P = [(i * 7 + 3) % 90 + 1 for i in range(40)]
    a = eng(FakeKV(), disk=disk(tmp_path))
    try:
        collect(a.submit(P, max_new=2, cuts=[10, 21]))
    finally:
        a.stop()
    assert sorted(b for b, _ in a.prefix.disk.index) == [10, 16, 21, 32]
    b = eng(FakeKV(), disk=disk(tmp_path))                              # restart: new memory tier, same directory
    try:
        collect(b.submit(P[:21] + [5, 6, 7, 8, 9], max_new=2))
        assert b.stats["prefix_hit_tokens"] == 21 and b.status()["disk_hits"] == 1
        collect(b.submit(P[:12] + [95, 96, 97, 98], max_new=2))         # shares only the first 12 tokens: the cut at 10 (no parent file)
        assert b.stats["prefix_hit_tokens"] == 21 + 10
    finally:
        b.stop()
    c = eng(FakeKV(), disk=disk(tmp_path))
    try:
        ref = collect(c.submit(P[:21] + [5, 6, 7, 8, 9], max_new=6))[0]
    finally:
        c.stop()
    d = eng(FakeKV())
    try:
        assert collect(d.submit(P[:21] + [5, 6, 7, 8, 9], max_new=6))[0] == ref
    finally:
        d.stop()


def test_cuts_that_are_not_a_prefix_of_the_ids_are_dropped(monkeypatch):
    monkeypatch.setenv("KYOJIN_KV_TAIL", "0")                                 # exact snapshot sets of the cuts alone (the tail anchor: tests/test_serve_kvdisk.py)
    ids = list(b"hello world")
    enc = lambda t: list(t.encode())                                    # noqa: E731
    assert C.token_cuts(["hello", "hello wor", "hello world", "", "xyz"], ids, enc) == [5, 9]   # the full prompt, empty and foreign texts go
    merged = lambda t: [1000] if t == "hello " else list(t.encode())    # noqa: E731  (a token that merges across the boundary)
    assert C.token_cuts(["hello ", "hello"], ids, merged) == [5]
    e = eng(Fake())
    try:
        h = e.submit(list(range(1, 13)), max_new=1, cuts=[0, 12, 99, 5, 5])
        collect(h)
        assert e.prefix.off.keys() == {5}                               # out-of-range cuts are ignored by the engine
    finally:
        e.stop()


TPL = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"


def test_jinja_and_ds41_cut_texts_are_prefixes_of_the_prompt():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    prompt = C.render_jinja(TPL, msgs, None)
    cuts = C.jinja_cut_texts(TPL, msgs, None, prompt)
    assert cuts == ["<system>be brief\n", "<system>be brief\n<user>hi\n"] and all(prompt.startswith(c) for c in cuts)
    fam = C.Ds41Family(None, "")
    body = {"messages": msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]}
    r = fam.render(body)
    assert len(r.cut_texts) == 2 and all(r.prompt.startswith(c) and c != r.prompt for c in r.cut_texts)
    assert r.cut_texts[0] < r.cut_texts[1] and r.cut_texts[1].endswith("more")


def test_disk_is_skipped_when_it_gains_little_over_memory(tmp_path):
    class Disk:
        block, calls = 16, []

        def lookup(self, ids, above=0, ok=None):
            self.calls.append(above)
            return 0, None, None

    be = Fake()
    pc = PrefixCache(be, 16, 1 << 20, disk=Disk())
    ids = list(range(40))
    pc.store(prefix_digests(ids, [16])[0], be.make_cache())
    assert pc.lookup(ids)[0] == 16
    assert Disk.calls == [16 + 16 // 4]        # only a checkpoint worth its load is asked for


def test_idle_engine_keeps_the_gpu_awake(monkeypatch):
    import threading
    import time
    from kyojin_mlx.serve import engine as E
    monkeypatch.setattr(E, "GPU_KEEPALIVE_S", 0.01)
    be = Fake()
    calls = []
    be.gpu_keepalive = lambda: calls.append(1)
    e = eng(be)
    t = threading.Thread(target=e.run, daemon=True)
    t.start()
    time.sleep(0.2)
    e.stop()
    t.join(5)
    assert len(calls) >= 3
