"""cloud/cmulti-2 on a real (tiny, random) DS4.1 text model, CPU: the production hook (bound_layers.YIELD), and the engine levers with real arithmetic.
Stock decode path (the row-invariant Metal kernels are not enabled: no Metal here)."""
import os
import sys
import threading
import time
import types

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import bound_layers as BL, rows as R  # noqa: E402
from kyojin_mlx.serve.backends import Backend, VlmBackend  # noqa: E402
from kyojin_mlx.serve.engine import Engine  # noqa: E402
from test_serve_engine import collect  # noqa: E402

CH = 32


class Mini(Backend):
    kind = "mini"
    max_rows = 8
    supports_yield = True

    def __init__(self):
        self.lm, _ = M.make(sliding_window=16)
        self.model = types.SimpleNamespace(language_model=self.lm)
        self.pf = []

    def set_prefill_yield(self, cb):
        BL.YIELD[0] = cb

    def make_cache(self):
        return self.lm.make_cache()

    def fork(self, cache):
        return R.fork(cache)

    def prefill_chunk(self, cache, ids, draft=None, skip_dec=False):
        self.pf.append(len(ids))
        lg = self.lm(mx.array(ids)[None], cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg, *self.state(cache))
        self.last = lg
        return lg

    def batch(self, caches):
        return R.batch(caches)

    def unbatch(self, bc, n):
        users = [[] for _ in range(n)]
        for c in bc:
            for i in range(n):
                users[i].append(c.extract(i))
        return users

    def step(self, bc, tokens):
        return R.step(self.model, bc, tokens)


@pytest.fixture(autouse=True)
def bounded(monkeypatch):
    """Install the layer bound on the mini model's layer class for this test only (the class is process-wide)."""
    be = Mini()
    cls = type(be.lm.layers[0])
    saved = (cls.__call__, getattr(cls, "_kyojin_bound", False))
    if hasattr(cls, "_kyojin_bound"):
        del cls._kyojin_bound
    BL.install(be.model)
    monkeypatch.setattr(BL, "EVERY", 2)
    monkeypatch.setattr(BL, "MIN_ROWS", 16)
    BL.YIELD[0] = None
    yield be
    BL.YIELD[0] = None
    cls.__call__ = saved[0]
    if saved[1]:
        cls._kyojin_bound = True
    elif hasattr(cls, "_kyojin_bound"):
        del cls._kyojin_bound


def test_the_hook_is_called_every_bound_of_a_chunk_and_never_in_a_decode_step(bounded):
    be = bounded
    n_layers = len(be.lm.layers)
    calls = []
    BL.YIELD[0] = lambda: calls.append(1)
    cache = be.make_cache()
    be.prefill_chunk(cache, M.tokens(CH))
    assert len(calls) == n_layers // BL.EVERY, (len(calls), n_layers)
    calls.clear()
    bc = be.batch([cache])
    lg = be.step(bc, mx.array([5], dtype=mx.int32)); mx.eval(lg)
    assert calls == []                                            # a decode step (1 row) never yields
    be.prefill_chunk(be.make_cache(), M.tokens(8, seed=3))        # below MIN_ROWS rows: no bound, no hook
    assert calls == []


def test_vlm_backend_wires_the_hook():
    b = object.__new__(VlmBackend)
    b.lm = None                                                   # the Engram ahead lookup finds nothing on a bare backend
    f = lambda: None
    VlmBackend.set_prefill_yield(b, f)
    assert BL.YIELD[0] is f and b.supports_yield == BL.ENABLED
    VlmBackend.set_prefill_yield(b, None)
    assert BL.YIELD[0] is None


def drive(eng, client):
    """MLX streams are per thread: the engine loop runs on THIS thread (the one that built the model), the clients on another. Returns client()'s value."""
    box = {}

    def run():
        try:
            box["v"] = client()
        except BaseException as e:                                  # noqa: BLE001
            box["e"] = e
        finally:
            eng.stop()
    th = threading.Thread(target=run)
    th.start()
    eng.run()
    th.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def solo(be_factory, ids, n, chunk=CH):
    eng = Engine(be_factory(), sessions=1, ctx=4096, chunk=chunk, block=chunk * 4, prefix_bytes=0)
    return drive(eng, lambda: collect(eng.submit(ids, max_new=n), timeout=300)[0])


def test_decoder_keeps_flowing_through_the_real_hook_and_texts_equal_the_flag_off_run(monkeypatch):
    a_ids, b_ids = M.tokens(24, seed=5), M.tokens(256, seed=6)
    outs = {}
    for flags in ({"KYOJIN_MIX_SLICE_MS": "0", "KYOJIN_MIX_DECODE": "0"}, {"KYOJIN_MIX_SLICE_MS": "1", "KYOJIN_MIX_DECODE": "50"}):
        for k, v in flags.items():
            monkeypatch.setenv(k, v)
        eng = Engine(Mini(), sessions=2, ctx=4096, chunk=CH, block=CH * 4, prefix_bytes=0)

        def client():
            ha = eng.submit(a_ids, max_new=400)
            ta = []
            while not ta:
                ev = ha.events.get(timeout=120)
                ta += [ev[1]] if ev[0] == "tok" else []
            hb = eng.submit(b_ids, max_new=6)
            tb = collect(hb, timeout=300)[0]
            ha.cancel()
            while True:
                ev = ha.events.get(timeout=120)
                if ev[0] != "tok":
                    break
                ta.append(ev[1])
            return ta, tb
        ta, tb = drive(eng, client)
        st = eng.status()
        outs[flags["KYOJIN_MIX_SLICE_MS"]] = (ta, tb, st.get("mix_slice_rounds", 0))
    print("slice rounds with the flag on:", outs["1"][2])
    assert outs["0"][2] == 0 and outs["1"][2] > 0                   # the production hook fired inside real prefill chunks
    for k in "01":
        assert outs[k][0] == solo(Mini, a_ids, len(outs[k][0]))[:len(outs[k][0])]          # the decoder's own text: every token it produced while B prefilled is its solo text
    assert outs["0"][1] == outs["1"][1] == solo(Mini, b_ids, 6)       # the prefilling session's text: flag off, flag on (hook through real layers), solo


def test_real_model_chunk_partition_changes_the_bits():
    """The pre-existing hole in real arithmetic: the same prompt prefilled on two chunk partitions (a cut at 20 vs none) on the tiny DS4.1 model."""
    be = Mini()
    ids = M.tokens(96, seed=9)

    def run(parts):
        c, s, lg = be.make_cache(), 0, None
        for e in parts:
            lg = be.prefill_chunk(c, ids[s:e]); s = e
        return c, lg
    c1, lg1 = run([32, 64, 96])
    c2, lg2 = run([20, 32, 64, 96])
    d = float(mx.max(mx.abs(lg1 - lg2)).item())
    print("logits max abs diff, partition (32,64,96) vs (20,32,64,96):", d, "bit-equal:", bool(mx.array_equal(lg1, lg2).item()))
    assert d < 1e-3                                                  # close, whatever the bits


def test_pf_share_on_the_real_model(monkeypatch):
    monkeypatch.setenv("KYOJIN_PF_SHARE", "1")
    pre = M.tokens(100, seed=11)
    kids = [pre + M.tokens(6 + i, seed=20 + i) for i in range(3)]
    eng = Engine(Mini(), sessions=4, ctx=4096, chunk=CH, block=CH * 64, prefix_bytes=0)
    hs = [eng.submit(p, max_new=8) for p in [pre, *kids]]
    outs = drive(eng, lambda: [collect(h, timeout=300)[0] for h in hs])
    st = eng.status()
    want = [solo(Mini, p, 8) for p in [pre, *kids]]
    print("share hits", st.get("share_hits"), "prefill tokens", st["prefill_tokens"], "no-share would be", len(pre) + sum(len(k) for k in kids))
    assert st["share_hits"] == 3
    assert outs == want                                                # real arithmetic: the child that started from the leader's cache gives its solo text
