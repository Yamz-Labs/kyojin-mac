"""Multi-user batched decode: row invariance of the dense kernels, cache fork, end-to-end bitwise equality with solo decode.
The model test needs a Qwen3.5 EXL3 pack: KYOJIN_Q35=<dir> (e.g. the Qwen3.5-2B 2.10 bpw pack) and a prompts file
KYOJIN_MU_PROMPTS=<json from tools/make_mu_prompts.py>; it is skipped otherwise. Needs a Mac (Metal)."""
import json
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from kyojin_mlx import multiuser as MU                                   # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


def bits(a):
    return np.array(a).view(np.uint16)


@pytest.mark.parametrize("K,N", [(2048, 16), (2048, 384), (4096, 100), (1000, 24)])
def test_rowdot_row_invariant(K, N):
    rng = np.random.default_rng(K + N)
    w = mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float16))
    x = mx.array(rng.standard_normal((11, K)).astype(np.float16))
    one = [np.array(MU.rowdot(x[i:i + 1], w)) for i in range(11)]
    for B in (1, 2, 3, 4, 8, 11):                                         # 11 > ROWDOT_MAX: split launches
        y = np.array(MU.rowdot(x[:B], w))
        for i in range(B):
            assert np.array_equal(y[i].view(np.uint16), one[i][0].view(np.uint16)), (B, i)
    ref = np.array(x, dtype=np.float64) @ np.array(w, dtype=np.float64).T
    got = np.array(MU.rowdot(x[:4], w), dtype=np.float64)
    assert np.abs(got - ref[:4]).max() <= 2e-3 * np.abs(ref[:4]).max() + 1e-3


def test_fork_cache_is_independent():
    from mlx_lm.models.cache import ArraysCache, KVCache
    rng = np.random.default_rng(0)
    kv = KVCache()
    k = mx.array(rng.standard_normal((1, 2, 5, 16)).astype(np.float16))
    kv.update_and_fetch(k, k)
    ac = ArraysCache(2)
    ac[0] = mx.array(rng.standard_normal((1, 3, 8)).astype(np.float16))
    ac[1] = mx.array(rng.standard_normal((1, 4, 4)).astype(np.float32))
    mx.eval(kv.keys, ac[0], ac[1])
    before = np.array(kv.keys[..., :5, :]).copy()
    a0 = np.array(ac[0]).copy()
    f = MU.fork_cache([kv, ac])
    f[0].update_and_fetch(k * 2, k * 2)
    f[1][0] = f[1][0] + 1
    mx.eval(f[0].keys, f[1][0])
    assert kv.offset == 5 and f[0].offset == 10
    assert np.array_equal(before, np.array(kv.keys[..., :5, :]))
    assert np.array_equal(a0, np.array(ac[0]))


def _ref_attn(q, k, v, scale):
    G = q.shape[1] // k.shape[1]
    k, v = mx.repeat(k, G, axis=1).astype(mx.float32), mx.repeat(v, G, axis=1).astype(mx.float32)
    return mx.softmax((q.astype(mx.float32) @ k.transpose(0, 1, 3, 2)) * scale, axis=-1) @ v


@pytest.mark.parametrize("P,N", [(0, 3), (700, 1), (2048, 4), (3000, 20)])
def test_pooled_attention_matches_reference(P, N):
    from kyojin_mlx import mu_attn as MA, sdpa_dec as SD
    if not SD.nax_ok():
        pytest.skip("needs the NAX tiles")
    Hq, Hk, D = 16, 4, 128
    mx.random.seed(1)
    lens = [5 + 7 * u for u in range(N)]
    pk = mx.random.normal((1, Hk, P, D)).astype(mx.float16) if P else None
    pv = mx.random.normal((1, Hk, P, D)).astype(mx.float16) if P else None
    tk = mx.random.normal((N, Hk, 512, D)).astype(mx.float16)
    tv = mx.random.normal((N, Hk, 512, D)).astype(mx.float16)
    q = mx.random.normal((N, Hq, 1, D)).astype(mx.float16)
    sc = D ** -0.5
    out = MA.pooled_attention(q, pk, pv, tk, tv, mx.array(lens, dtype=mx.int32), sc).astype(mx.float32)
    for u in range(N):
        k, v = tk[u:u + 1, :, :lens[u]], tv[u:u + 1, :, :lens[u]]
        if P:
            k, v = mx.concatenate([pk, k], 2), mx.concatenate([pv, v], 2)
        assert mx.abs(out[u:u + 1] - _ref_attn(q[u:u + 1], k, v, sc)).max().item() < 2e-3


def test_batch_caches_pools_forked_prefix():
    from mlx_lm.models.cache import KVCache
    from kyojin_mlx import mu_attn as MA, sdpa_dec as SD
    if not SD.nax_ok():
        pytest.skip("needs the NAX tiles")
    mx.random.seed(2)
    base = KVCache()
    x = mx.random.normal((1, 4, 300, 64)).astype(mx.float16)
    base.update_and_fetch(x, x * 0.5)
    users = []
    for u in range(3):
        f = MU.fork_cache([base])
        t = mx.random.normal((1, 4, 3 + u, 64)).astype(mx.float16)
        f[0].update_and_fetch(t, t)
        users.append(f)
    solo = [MU.fork_cache(f) for f in users]
    bc = MU.batch_caches([MU.fork_cache(f) for f in users])
    assert isinstance(bc[0], MA.PoolKV) and bc[0].P == 300
    assert isinstance(MU.batch_caches([MU.fork_cache(f) for f in users], prefix_len=0)[0], MU.RowsKV)
    nk = mx.random.normal((3, 4, 1, 64)).astype(mx.float16)
    bc[0].update_and_fetch(nk, nk)
    back = MU.unbatch_caches(bc, 3)
    for u in range(3):
        solo[u][0].update_and_fetch(nk[u:u + 1], nk[u:u + 1])
        n = solo[u][0].offset
        assert back[u][0].offset == n
        assert np.array_equal(bits(back[u][0].keys[..., :n, :]), bits(solo[u][0].keys[..., :n, :]))
        assert np.array_equal(bits(back[u][0].values[..., :n, :]), bits(solo[u][0].values[..., :n, :]))
    assert isinstance(MU.batch_caches(back)[0], MA.PoolKV)                        # the fork records survive unbatch


Q35 = os.environ.get("KYOJIN_Q35")
PROMPTS = os.environ.get("KYOJIN_MU_PROMPTS")


@pytest.mark.skipif(not (Q35 and PROMPTS), reason="set KYOJIN_Q35 and KYOJIN_MU_PROMPTS")
def test_batched_decode_equals_solo_bitwise():
    from kyojin_mlx.exl3_linear import load_exl3
    model, _ = load_exl3(Q35)
    assert MU.install(model) > 0
    prompts = [p[:n] for p, n in zip(json.load(open(PROMPTS))["prompts"], (200, 90, 150))]    # different lengths
    cs, firsts = [], []
    for ids in prompts:
        c, lg = MU.prefill(model, ids)
        cs.append(c)
        firsts.append(mx.argmax(lg, axis=-1))
    first = mx.concatenate(firsts)
    ref = [MU.decode_solo(model, MU.fork_cache(cs[i]), first[i:i + 1], 12) for i in range(3)]
    bc = MU.batch_caches([MU.fork_cache(c) for c in cs])
    lb = np.array(MU.logits_last(model, first[:, None], bc))
    ls = [np.array(MU.logits_last(model, first[i:i + 1, None], MU.fork_cache(cs[i]))) for i in range(3)]
    for i in range(3):
        assert np.array_equal(lb[i].view(np.uint16), ls[i][0].view(np.uint16))
    got = MU.decode_batched(model, MU.batch_caches([MU.fork_cache(c) for c in cs]), first, 12)
    assert got == ref
