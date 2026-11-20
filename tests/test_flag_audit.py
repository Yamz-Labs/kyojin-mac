"""Flag audit (KYOJIN_FIRED): every GLM speed-flag guard reports a reason, fire / fb counters are off by default and cost one global read when off."""
from types import SimpleNamespace as NS

import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import fired as FD, glm_dsa, glm_hc, glm_pf_glue, glm_pf_idx, glm_dsa_pf, kda_pf        # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    on, log, sink = FD.AUDIT, FD.LOG, FD._sink
    FD.reset()
    yield
    FD.AUDIT, FD.LOG, FD._sink = on, log, sink
    FD.reset()


def test_audit_is_on_by_default():
    import importlib, os
    assert os.environ.get("KYOJIN_FLAG_AUDIT", "1") != "0" and FD.AUDIT


def test_fire_and_fb_are_silent_when_audit_off():
    FD.AUDIT = False
    FD.fire("x")
    FD.fb("x", "why")
    assert FD.STATS == {}


def test_fire_and_fb_count_and_log_each_reason_once(capsys):
    FD.AUDIT, FD.LOG, FD._sink = True, True, None
    FD.fire("x")
    FD.fb("x", "why")
    FD.fb("x", "why")
    FD.fb("x", "other")
    assert FD.STATS == {"x.fire": 1, "x.fb.why": 2, "x.fb.other": 1}
    err = capsys.readouterr().err
    assert err.count("[fired] x: stock path, why") == 1 and err.count("other") == 1


def test_summary_line_and_reporter():
    FD.AUDIT, FD.LOG = True, False
    for _ in range(3):
        FD.fire("a")
    FD.fb("a", "rows>4")
    FD.fb("b", "cache=BatchKVCache")
    FD.fb("b", "cache=BatchKVCache")
    assert FD.summary() == "a 3/1 (rows>4:1); b 0/2 (cache=BatchKVCache:2)"
    snap = dict(FD.STATS)
    FD.fire("a")
    assert FD.summary(snap) == "a 1/0"
    lines = []
    t = FD.start_reporter(lines.append, period=0.05)
    FD.fb("c", "z")
    import time
    time.sleep(0.3)
    assert any(l.startswith("[flags 0s] ") and "c 0/1 (z:1)" in l for l in lines) or any("c 0/1" in l for l in lines)


def _hc(fn_dtype=mx.float32, base=mx.float32, mult=4, H=2048):
    return NS(fn=mx.zeros((24, 4 * H), dtype=fn_dtype), base=mx.zeros((24,), dtype=base), scale=mx.zeros((3,), dtype=base), hc_mult=mult)


def test_hc_fuse_reasons():
    x = mx.zeros((1, 1, 4, 2048), dtype=mx.float16)
    assert glm_hc.why_shape(x, _hc()) is None and glm_hc.ok_shape(x, _hc())
    assert glm_hc.why_shape(x, _hc(mx.float16)) is None                           # the served loader stores fn in fp16 (bug 4091894: this once fell back)
    assert glm_hc.why_shape(x, _hc(mx.bfloat16)) == "fn_dtype=bfloat16"
    assert glm_hc.why_shape(mx.zeros((1, 17, 4, 2048), dtype=mx.float16), _hc()) == "tokens>16"
    assert glm_hc.why_shape(mx.zeros((1, 1, 4, 2048), dtype=mx.float32), _hc()) == "x_dtype=float32"
    assert glm_hc.why_shape(mx.zeros((1, 1, 4, 1536), dtype=mx.float16), _hc(H=1536)) == "hidden%1024"
    assert glm_hc.why_shape(x, _hc(base=mx.float16)) == "base_scale_dtype"


def test_pf_hc_mix_reasons():
    x = mx.zeros((64, 4 * 2048), dtype=mx.float16)
    fn = mx.zeros((24, 4 * 2048), dtype=mx.float32)
    assert glm_pf_glue.mix_why(x, fn) in (None, "no_gpu")
    assert glm_pf_glue.mix_why(x[:8], fn) == "rows<16"
    assert glm_pf_glue.mix_why(x, fn.astype(mx.float16)) in (None, "no_gpu")                     # the served loader's dtype (BP-PFFIRE)
    assert glm_pf_glue.mix_why(x, fn.astype(mx.bfloat16)) in ("fn_dtype=bfloat16", "no_gpu")
    assert glm_pf_glue.mix_why(x.astype(mx.float32), fn) == "x_dtype=float32"


def test_pf_idx_and_dsa_pf_reasons(monkeypatch):
    monkeypatch.setattr(glm_pf_idx, "ENABLED", True)
    q = mx.zeros((1, 128, 32, 128), dtype=mx.float16)
    pk = mx.zeros((1, 256, 128), dtype=mx.float16)
    w = mx.zeros((1, 128, 32), dtype=mx.float32)
    assert glm_pf_idx.why_not(q[:, :8], pk, w, 64) == "rows<64"
    assert glm_pf_idx.why_not(mx.zeros((2, 128, 32, 128), dtype=mx.float16), pk, w, 64) == "batch=2"
    assert glm_pf_idx.why_not(q, pk[:, :100], w, 64) == "few_pools"
    assert glm_pf_idx.why_not(q, pk, w.astype(mx.float16), 64) == "weights_dtype=float16"
    monkeypatch.setattr(glm_dsa_pf, "ENABLED", True)
    ql = mx.zeros((1, 64, 128, 512), dtype=mx.float16)
    lat = mx.zeros((1, 1, 300, 512), dtype=mx.float16)
    tk = mx.zeros((1, 128, 64), dtype=mx.int32)
    assert glm_dsa_pf.why_not(ql[:, :, :1], lat, tk[:, :1]) == "rows<64"
    assert glm_dsa_pf.why_not(mx.zeros((4, 64, 128, 512), dtype=mx.float16), lat, tk) == "batch=4"
    assert glm_dsa_pf.why_not(ql, lat, tk[:, :, :]) in (None, "no_nax")


def test_kda_pf_reasons_decode_rows_first():
    me = NS(head_dim=128, conv_kernel=4, lower_bound=-5.0)
    assert kda_pf._why(me, mx.zeros((4, 1, 8), dtype=mx.float16), None, NS()) == "rows<64"
    assert kda_pf._why(me, mx.zeros((4, 128, 8), dtype=mx.float16), None, NS()) in ("batch=4", "no_gpu")
    assert kda_pf._why(me, mx.zeros((1, 128, 8), dtype=mx.float16), mx.zeros(1), NS()) == "mask"


def test_dsa_fast_reasons():
    from mlx_vlm.models.cache import KVCache, PoolingCache
    me = NS(training=False, n_heads=32, head_dim=128, always_select_tail=True, index_kpool=4)
    x = mx.zeros((1, 1, 8), dtype=mx.float16)
    assert glm_dsa._why(me, x, None, None, None) == "no_cache"
    assert glm_dsa._why(me, mx.zeros((4, 1, 8), dtype=mx.float16), None, KVCache(), PoolingCache(4)) == "shape B4xL1"
    assert glm_dsa._why(me, mx.zeros((1, 2, 8), dtype=mx.float16), None, KVCache(), PoolingCache(4)) == "rows L2"
    assert glm_dsa._why(me, x, None, NS(), PoolingCache(4)) == "cache=SimpleNamespace"
    assert glm_dsa._why(me, x, None, KVCache(), PoolingCache(4)) == "pools_not_1to1"
