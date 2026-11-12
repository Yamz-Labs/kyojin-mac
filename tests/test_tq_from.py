"""KYOJIN_TQ_FROM (KV format by depth): a plain cache converted to TurboQuant at admission attends like one prefilled into TurboQuant, and a plain-KV disk
checkpoint restores as plain KV when the default cache is TurboQuant. Needs a Mac (Metal)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache                                     # noqa: E402
from kyojin_mlx import kvdisk, kvwin as KW, tq as TQ                       # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
CFG: dict = dict(kbits=6, vbits=6, sink=4, window=64)


def _rows(n, hk=4, d=128, seed=0):
    rng = np.random.default_rng(seed)
    return (mx.array(rng.standard_normal((1, hk, n, d)).astype(np.float16)), mx.array(rng.standard_normal((1, hk, n, d)).astype(np.float16)))


def test_from_kv_attends_like_prefilled_tq():
    KW.install()
    k, v = _rows(600)
    plain = KVCache(); plain.update_and_fetch(k, v)
    conv = TQ.from_kv(plain, CFG)
    pre = TQ.TQKVCache(**CFG)
    for a in range(0, 600, 256):                                            # chunked prefill
        pre.update_and_fetch(k[:, :, a:a + 256], v[:, :, a:a + 256])
    assert conv.offset == pre.offset == 600
    q = mx.array(np.random.default_rng(1).standard_normal((1, 16, 1, 128)).astype(np.float16))
    a, b = conv.attend(q, 128 ** -0.5), pre.attend(q, 128 ** -0.5)
    mx.eval(a, b)
    rel = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))) / (mx.max(mx.abs(b.astype(mx.float32))) + 1e-6))
    assert rel < 1e-2, rel


def test_from_kv_empty():
    assert TQ.from_kv(KVCache(), CFG).offset == 0


def test_plain_checkpoint_restores_plain_under_tq_default():
    k, v = _rows(100, seed=2)
    plain = KVCache(); plain.update_and_fetch(k, v)
    ad = kvdisk.GenericAdapter()
    fixed, grow, rows, meta = ad.dump([plain], {})
    part = ({**{"f." + n: a for n, a in fixed.items()}, **{"g." + n: a for n, a in grow.items()}}, meta)
    got = ad.restore(lambda: [TQ.TQKVCache(**CFG)], [part])
    assert type(got[0]) is KVCache and got[0].offset == 100
    np.testing.assert_array_equal(np.array(got[0].keys[:, :, :100]), np.array(k))


def test_mixed_chain_is_refused():
    k, v = _rows(100, seed=3)
    plain = KVCache(); plain.update_and_fetch(k, v)
    ad = kvdisk.GenericAdapter()
    f1, g1, r1, m1 = ad.dump([plain], {})
    p1 = ({**{"f." + n: a for n, a in f1.items()}, **{"g." + n: a for n, a in g1.items()}}, m1)
    p0 = ({}, dict(m1))                                                     # an earlier part without this layer's plain rows (another cache kind)
    with pytest.raises(ValueError):
        ad.restore(lambda: [TQ.TQKVCache(**CFG)], [p0, p1])
