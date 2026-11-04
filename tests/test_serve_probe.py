"""hostsync.Timed (the per-blocking-call split of a section) and serve.pysample (the engine thread's wall attribution). No model, no GPU run."""
import collections
import os
import threading
import time

import mlx.core as mx
import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx import hostsync as HS
from kyojin_mlx.serve import pysample as PS


@pytest.fixture(autouse=True)
def fresh():
    saved = (HS.TIMED, os.environ.get("KYOJIN_HOST_TIMED"))
    yield
    HS.TIMED = saved[0]
    if saved[1] is None:
        os.environ.pop("KYOJIN_HOST_TIMED", None)
    else:
        os.environ["KYOJIN_HOST_TIMED"] = saved[1]


def test_timed_counts_each_blocking_call_kind_and_restores_mlx():
    st = collections.Counter()
    a = mx.zeros((4,))
    before = (mx.eval, mx.async_eval, mx.array.tolist, mx.array.item)
    with HS.Timed(st, "t") as tm:
        mx.eval(a + 1)                                           # one blocking eval
        mx.async_eval(a + 2)                                     # no wait
        a.tolist()                                               # one blocking read
    assert (mx.eval, mx.async_eval, mx.array.tolist, mx.array.item) == before
    assert st["n_hs_t"] == 1 and st["t_hs_t_w"] > 0
    assert st["n_hs_t_eval"] == 1 and st["n_hs_t_async_eval"] == 1 and st["n_hs_t_tolist"] == 1
    assert "n_hs_t_item" not in st
    assert st["t_hs_t_w"] >= tm.sync_wall() > 0
    assert 0 <= st["t_hs_t_c"] <= st["t_hs_t_w"] + 1e-3
    assert st["max_hs_t_eval_w"] > 0


def test_timed_is_null_when_off():
    st = collections.Counter()
    HS.TIMED = False
    with HS.timed(st, "x") as cm:
        assert not isinstance(cm, HS.Timed)
        mx.eval(mx.zeros((2,)))
    assert not st


def test_timed_flag_is_runtime_and_reads_its_variable():
    f = FL.FLAGS["host_timed"]
    assert f.default == "0" and f.runtime and f.env == "KYOJIN_HOST_TIMED"
    (mod, attr, fn), = f.attrs
    assert mod == "hostsync" and attr == "TIMED" and fn("1") is True and fn("0") is False
    src = open(HS.__file__).read()
    assert 'os.environ.get("KYOJIN_HOST_TIMED", "0")' in src


def test_sampler_attributes_wall_to_the_site_it_sampled(tmp_path):
    """The sampler prices the engine thread's own frame: a loop that holds the GIL (a C wait does: MLX waits with it) is charged to that line."""
    mine = threading.get_ident()
    s = PS.Sampler(mine, 0.001)
    x = (1 << 20)
    t0 = time.perf_counter()
    for _ in range(40):
        time.sleep(0.005)                                        # releases the GIL: the sampler runs on time, no gil_held
    busy = time.perf_counter() - t0
    for _ in range(5):
        end = time.perf_counter() + 0.02
        while time.perf_counter() < end:                          # holds the GIL: samples arrive late
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
    out = tmp_path / "s.txt"
    s.dump(str(out))
    text = out.read_text().splitlines()
    head = dict(zip(text[0].split()[::2], text[0].split()[1::2]))
    assert float(head["total"]) >= busy
    assert int(head["samples"]) > 5 and int(head["sites"]) >= 1
    assert float(head["gil_held"]) > 0.01                        # the busy loops held it
    sites = [ln.split(None, 3) for ln in text[1:]]
    assert sites and all(len(r) == 4 and ":" in r[3] for r in sites)


def test_sync_arms_on_the_env_and_dumps_when_off(tmp_path, monkeypatch):
    st = collections.Counter()
    out = tmp_path / "p.txt"
    monkeypatch.setenv("KYOJIN_HOST_SAMPLE", "1000")
    monkeypatch.setenv("KYOJIN_HOST_SAMPLE_OUT", str(out))
    try:
        PS.sync(st)
        assert PS._sampler is not None and st["n_hs_sample_arm"] == 1
        time.sleep(0.03)
        monkeypatch.setenv("KYOJIN_HOST_SAMPLE", "0")
        PS.sync(st)
        assert PS._sampler is None and out.exists() and st["n_hs_sample_dump"] == 1
        assert st["n_hs_sample"] > 3 and st["t_hs_sample_total"] > 0
    finally:
        monkeypatch.setenv("KYOJIN_HOST_SAMPLE", "0")
        PS.sync(st)
        PS._sampler = None


def test_sample_flags_are_off_by_default():
    for n in ("host_timed", "host_sample"):
        f = FL.FLAGS[n]
        assert f.default == "0" and f.runtime, n
    assert FL.FLAGS["host_sample"].kind == "int"
