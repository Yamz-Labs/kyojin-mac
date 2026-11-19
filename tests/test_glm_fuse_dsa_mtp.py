"""G3e: the MTP head (layer 45) under KYOJIN_GLM_FUSE_DSA. The head runs its OWN indexer + pooling cache, fed in pieces by glm_nextn.feed (CHUNK = 512 pairs, a last
piece of 1..4 pairs after a prefill chunk): on the full model the 2..4-row piece raised 'unexpected pool state after the cache update: pool float32' because the head's
indexer weights are fp32 (pool fp32) and the dtype was only checked AFTER the caches had been updated. Now the dtype is checked before any cache write and the call
takes the stock indexer with a counted reason (glm_fuse_dsa.WHY)."""
import json
import os

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.glm5_next.language")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
os.environ["KYOJIN_GLM_DSA_FAST"] = "1"
os.environ["KYOJIN_GLM_FUSE_DSA"] = "1"
from kyojin_mlx import glm_dsa as G, glm_fuse_dsa as F     # noqa: E402

CFG = os.path.expanduser("/path/to/data")


def _head(dt, seed=0):
    """The real MTP block class (indexer, MLA attention, MoE) at a small hidden size; every weight cast to `dt`, the indexer's norm / compress params to `idx_dt`."""
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
    if not os.path.exists(CFG):
        pytest.skip("no GLM config on this machine")
    c = json.load(open(CFG))["text_config"]
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        c[k] = c[k][:1]
    c.update(num_hidden_layers=1, hidden_size=512, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, intermediate_size=256, first_k_dense_replace=0)
    d = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict({"text_config": c}))
    mx.random.seed(seed)
    new = []
    for k, v in tree_flatten(d.parameters()):
        if v.ndim >= 2:
            new.append((k, (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float32 if "compress" in k else dt)))
        else:
            new.append((k, (mx.zeros(v.shape) if "bias" in k else mx.ones(v.shape)).astype(dt)))
    d.update(tree_unflatten(new))
    mx.eval(d.parameters())
    d.eval()
    return d


def _feed(d, cache, n, pos, dt, seed):
    """glm_nextn.NextN.feed's loop: pieces of CHUNK = 512 pairs through eh_proj + mtp_block (random embeddings / hidden states instead of the target's)."""
    from kyojin_mlx.glm_nextn import CHUNK
    mx.random.seed(seed)
    e = mx.random.normal((1, n, 512)).astype(dt)
    hid = mx.random.normal((1, n, 512)).astype(dt)
    h = None
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        x = d.eh_proj(mx.concatenate([d.enorm(e[:, a:b]), d.hnorm(hid[:, a:b])], axis=-1))
        h = d.mtp_block(x, cache[0], last_only=True)
        mx.eval(h)
    return h


def _install():
    mx.set_default_device(mx.gpu)                      # other glm test files leave the default device on the CPU
    G.ENABLED = F.ENABLED = True                       # a glob run imports the modules before the env above is set
    assert G.install() and F.install()


@pytest.mark.parametrize("dt", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("tail", [1, 2, 3, 4])
def test_mtp_block_feed_chunks_and_remainders(dt, tail):
    """Prefill-like feed: 2100 pairs in a few calls, then pieces of `tail` pairs after remainders 0..3 of the pool window. Flag on = flag off (bit for bit), no exception."""
    _install()
    d = _head(dt)
    runs = []
    for en in (False, True):
        F.ENABLED = en
        F.WHY.clear()
        rows0 = F.STATS["ix_rows"]
        cache = d.make_cache()
        outs = []
        pos = 0
        for call, n in enumerate([1030, 1025, 45 + 0] + [tail] * 4 + [1, 3, tail, 1]):       # pieces end on every pool remainder; the 1030 / 1025 pieces cross 512-pair chunks
            outs.append(_feed(d, cache, n, pos, dt, 100 + call))
            pos += n
        pc = cache[0].caches[2]
        runs.append((outs, pc.pooled, pc.remainder, cache[0].caches[1].offset, dict(F.WHY), F.STATS["ix_rows"] - rows0))
    F.ENABLED = True
    for a, b in zip(runs[0][0], runs[1][0]):
        assert mx.array_equal(a, b)
    assert mx.array_equal(runs[0][1], runs[1][1]) and runs[0][2] == runs[1][2] and runs[0][3] == runs[1][3]
    why = runs[1][4]
    if dt == mx.float32:
        assert tail == 1 or why.get("pool_dtype_float32", 0) > 0, why          # the fp32 head: stock, counted
    else:
        assert not any(k.startswith("pool_dtype") for k in why), why
        assert tail == 1 or runs[1][5] > 0                                   # the rows path really ran on the head's 2..4-row pieces


@pytest.mark.parametrize("n", [8160, 8161, 8162, 8163])
def test_mtp_block_prefill_length_remainders_fp32(n):
    """The failing shape of the full model: the head fed prompt-pair counts 8160..8163 (remainders 0..3), last piece 2 pairs, fp32 pool."""
    _install()
    d = _head(mx.float32)
    F.WHY.clear()
    cache = d.make_cache()
    ref = None
    for en in (True, False):
        F.ENABLED = en
        cache = d.make_cache()
        h = _feed(d, cache, n - 2, 0, mx.float32, 7)
        h = _feed(d, cache, 2, n - 2, mx.float32, 8)
        if ref is None:
            ref = (h, cache[0].caches[2].pooled)
        else:
            assert mx.array_equal(h, ref[0]) and mx.array_equal(cache[0].caches[2].pooled, ref[1])
    F.ENABLED = True


@pytest.mark.parametrize("Lr", [1, 3])
def test_batch_fp32_indexer_takes_stock_before_any_cache_write(Lr, monkeypatch):
    """The batch path had the same late check: B sessions with an fp32 indexer / pool give the stock result and a counted reason."""
    import test_glm_fuse_dsa as T
    L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
    monkeypatch.setattr(T, "DT", mx.float32)
    cfg = T._cfg()
    ix = T._rand(L.Glm5NextIndexer(cfg, 3), 0)
    _install()
    ctxs = [4096, 4000]
    A, Bc = T._batched(ctxs), T._batched(ctxs)
    for step in range(3):
        mx.random.seed(90 + step)
        x = (mx.random.normal((len(ctxs), Lr, cfg.hidden_size)) * 0.5).astype(mx.float32)
        qr = mx.random.normal((len(ctxs), Lr, cfg.q_lora_rank)).astype(mx.float32)
        outs = []
        F.WHY.clear()
        for en, c in ((False, A), (True, Bc)):
            F.ENABLED = en
            t = L.Glm5NextIndexer.__call__(ix, x, qr, None, c.caches[1], c.caches[2], c.caches[0].offset)
            mx.eval(t)
            outs.append(t)
        assert mx.array_equal(outs[0], outs[1])
    F.ENABLED = True
    assert F.WHY.get("pool_dtype_float32", 0) > 0, F.WHY
