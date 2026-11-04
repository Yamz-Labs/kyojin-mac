"""pfuse3 levers of the DS4.1 prefill: E3 (fused indexer score kernel), E4 (exact radix top-k of the indexer block scores), E5 (window + pool as two
attention kernel inputs) and E6 (no window mask when the gathered-key path runs). Every test compares the flagged path with the unflagged one on synthetic
shapes: same scores, same indices, same attention output. Needs a GPU (the score kernel and the attention kernels use matmul2d)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

from kyojin_mlx import flags            # noqa: E402
from kyojin_mlx import attn_prefill as AP  # noqa: E402
from kyojin_mlx import decstage as DS   # noqa: E402
from kyojin_mlx import ds_pf_idx as PI  # noqa: E402
from kyojin_mlx import idx_topk as IT   # noqa: E402
from kyojin_mlx import metal_nax as NX  # noqa: E402

NH, D = 32, 128


def make_idx(T, Tc, seed=0, quant=False):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, T, NH, D)).astype(np.float16)
    k = rng.standard_normal((1, Tc, D)).astype(np.float16)
    w = (rng.standard_normal((1, T, NH)) * 0.05).astype(np.float16)
    if quant:                                                          # few distinct values: exact ties in the scores
        q, k = (np.round(q * 2) / 2).astype(np.float16), (np.round(k * 2) / 2).astype(np.float16)
        w = (np.round(w * 64) / 64).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


# ----------------------------------------------------------------------------------------------------------------- E3: fused score kernel
@pytest.mark.skipif(not NX.nax_available(), reason="no neural accelerators on this GPU")
@pytest.mark.parametrize("T,Tc,start,ratio,n0,tpt,quant", [(64, 4096, 4096, 4, 0, 4, False), (8, 1000, 4000, 4, 0, 8, True),
                                                          (16, 512, 2048, 2, 0, 1, False), (33, 300, 1300, 4, 100, 2, True),
                                                          (7, 4097, 8194, 2, 0, 8, False), (5, 130, 130, 1, 0, 3, True)])
def test_fscore_equals_matmul_post(T, Tc, start, ratio, n0, tpt, quant, monkeypatch):
    """Bit for bit against the matmul + post kernel pair (product rounded to half, relu * w, heads summed in order 0..31, causal mask)."""
    q, k, w = make_idx(T, Tc, seed=Tc, quant=quant)
    monkeypatch.setattr(PI, "FSCORE", False)
    ref = np.array(PI.scores_range(q, k, w, start, ratio, 0, T, n0=n0))
    monkeypatch.setattr(PI, "FSCORE", True)
    monkeypatch.setattr(PI, "FSCORE_TPT", tpt)
    got = np.array(PI.scores_range(q, k, w, start, ratio, 0, T, n0=n0))
    assert np.array_equal(np.isfinite(ref), np.isfinite(got))
    assert np.array_equal(got, ref)


@pytest.mark.skipif(not NX.nax_available(), reason="no neural accelerators on this GPU")
def test_fscore_row_blocks_and_scores_mm(monkeypatch):
    """The rows of a split equal the same rows of the whole chunk (nothing but S depends on the row blocking now)."""
    q, k, w = make_idx(40, 9000, seed=3)
    monkeypatch.setattr(PI, "FSCORE", True)
    whole = np.array(PI.scores_mm(q, k, w, 4 * 9000, 4))
    a = np.array(PI.scores_range(q, k, w, 4 * 9000, 4, 0, 17))
    b = np.array(PI.scores_range(q, k, w, 4 * 9000, 4, 17, 23))
    assert np.array_equal(np.concatenate([a, b], axis=0), whole)


@pytest.mark.skipif(not NX.nax_available(), reason="no neural accelerators on this GPU")
def test_fscore_topk_indices_equal(monkeypatch):
    """The selection of the served path (top-k over a score block) is the same with the fused scores."""
    q, k, w = make_idx(48, 6000, seed=5)
    monkeypatch.setattr(PI, "FSCORE", True)
    got = np.array(IT.topk(PI.scores_range(q, k, w, 4 * 6000, 4, 0, 48), 64))
    q, k, w = make_idx(48, 6000, seed=5)
    monkeypatch.setattr(PI, "FSCORE", False)
    assert np.array_equal(got, np.array(IT.topk(PI.scores_range(q, k, w, 4 * 6000, 4, 0, 48), 64)))


def test_fscore_gate():
    """The fused path only claims the shapes it was built for (anything else keeps the matmul + post pair)."""
    q, k, w = make_idx(64, 1000)
    assert PI.fscore_ok(q, k, w) == (bool(PI.FSCORE) and NX.nax_available())
    assert not PI.fscore_ok(q[:, :, :16], k, w)                        # 16 heads
    assert not PI.fscore_ok(q[..., :64], k[..., :64], w)               # head dim
    assert not PI.fscore_ok(q, k, w.astype(mx.float32))                # the served weights are half
    assert not PI.fscore_ok(q, k[:, :64], w)                           # a key width below one tail tile


# ----------------------------------------------------------------------------------------------------------------- E4: exact radix top-k
@pytest.mark.parametrize("n,Tc,k,quant", [(64, 4096, 512, False), (33, 9001, 512, True), (8, 20000, 128, True), (16, 64, 64, False),
                                          (4, 1000, 8, True)])
def test_topk_mt_is_the_stock_selection(n, Tc, k, quant, monkeypatch):
    """The radix select returns the very indices of sort(argsort(-s))[:k] (value desc, lower index first on ties), -inf filler included."""
    rng = np.random.default_rng(Tc)
    s = rng.standard_normal((n, Tc)).astype(np.float32)
    s[rng.random((n, Tc)) < 0.2] = -np.inf                              # causal / candidate masked columns
    if quant:
        s = (np.round(s * 2) / 2).astype(np.float32)
    s = mx.array(s)
    monkeypatch.setattr(PI, "TOPK_MT", False)
    ref = np.array(PI.topk_block(s, k))
    monkeypatch.setattr(PI, "TOPK_MT", True)
    assert np.array_equal(np.array(PI.topk_block(s, k)), ref)


# ----------------------------------------------------------------------------------------------------------------- E5/E6: attention table + mask
def make_attn(T, hist, Np, topk, H=32, Dh=256, seed=0, ratio=2):
    rng = np.random.default_rng(seed)
    q = mx.array((rng.standard_normal((1, H, T, Dh)) * 0.6).astype(np.float16))
    window = mx.array((rng.standard_normal((1, hist + T, Dh)) * 0.6).astype(np.float16))
    pool = mx.array((rng.standard_normal((1, Np, Dh)) * 0.6).astype(np.float16))
    idx = np.full((1, T, topk), -1, np.int32)
    for t in range(T):
        n = min(Np, (hist + t + 1) // ratio)
        k = min(topk, n)
        if k:
            idx[0, t, :k] = np.sort(rng.choice(n, k, replace=False))
    return q, window, pool, mx.array(idx), mx.array(rng.standard_normal(H).astype(np.float32))


@pytest.mark.skipif(not NX.nax_available(), reason="no neural accelerators on this GPU")
@pytest.mark.parametrize("T,hist,Np,topk,chunk", [(70, 0, 40, 48, 512), (96, 127, 80, 64, 32), (65, 30, 33, 320, 512), (130, 127, 150, 100, 64)])
def test_pool_as_two_inputs_bitwise(T, hist, Np, topk, chunk, monkeypatch):
    """E5: the same [1, H, T, D] output, bit for bit, with the pool as a second kernel input instead of the concatenated table."""
    monkeypatch.setattr(AP, "CHUNK", chunk)
    q, window, pool, idxs, sinks = make_attn(T, hist, Np, topk)
    monkeypatch.setattr(AP, "POOLIN", False)
    ref = AP.sparse_attention(q, window, pool, idxs, 256 ** -0.5, sinks, 128)
    monkeypatch.setattr(AP, "POOLIN", True)
    got = AP.sparse_attention(q, window, pool, idxs, 256 ** -0.5, sinks, 128)
    assert np.array_equal(np.array(got), np.array(ref))


def _local_attention_module():
    import importlib
    from mlx.utils import tree_map
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    L = importlib.import_module("mlx_vlm.models.deepseek_v41.language")
    c = dict(model_type="deepseek_v41_text", vocab_size=64, hidden_size=512, moe_intermediate_size=64, num_hidden_layers=1, num_attention_heads=32,
             num_key_value_heads=1, head_dim=128, qk_rope_head_dim=32, q_lora_rank=128, o_lora_rank=64, o_groups=4, rms_norm_eps=1e-6, rope_theta=10000,
             n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, sliding_window=128, compress_ratios=[0], compress_rope_theta=160000,
             kv_source_layer_ids=[], index_source_layer_ids=[], candidate_source_layer_id=-1, index_n_heads=4, index_head_dim=32, index_topk=16, hc_mult=4,
             max_position_embeddings=4096, rope_scaling=None, scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True, routed_scaling_factor=1.5)
    try:
        cfg = ModelConfig.from_dict(c)
    except Exception as e:
        pytest.skip(f"ModelConfig: {e}")
    A = L.DeepseekV41Attention(cfg, 0)
    rng = np.random.default_rng(0)
    A.update(tree_map(lambda a: mx.array((rng.standard_normal(a.shape) * (0.05 if a.ndim > 1 else 0.5)).astype(np.float16)), A.parameters()))
    A.attn_sink = mx.array(rng.standard_normal(32).astype(np.float32))
    mx.eval(A.parameters())
    return L, A, rng


def test_window_part_without_mask_matches_the_vendor(monkeypatch):
    """E6: `attn_prefill.window_part` gives the vendor window rows bit for bit (same ring buffer update) and skips the mask; on demand it gives the same one."""
    L, A, rng = _local_attention_module()
    monkeypatch.setattr(L, "fake_quant_fp8_ue8m0", lambda x: x)
    xs = [mx.array(rng.standard_normal((1, n, 512)).astype(np.float16)) for n in (150, 60)]
    outs = {}
    for name, need in (("stock", None), ("lazy", False), ("full", True)):
        cache = L.DeepseekV41Cache(1)
        res = []
        for x in xs:
            part, mask = (A._window_part(x, cache.offset, cache) if need is None else AP.window_part(A, x, cache.offset, cache, need_mask=need))
            res.append((np.array(part), None if mask is None else np.array(mask)))
            cache.advance(x.shape[1])
        outs[name] = res
    for (p0, m0), (p1, m1), (p2, m2) in zip(outs["stock"], outs["lazy"], outs["full"]):
        assert np.array_equal(p0, p1) and np.array_equal(p0, p2)        # same window rows, same ring buffer
        assert m1 is None and np.array_equal(m0, m2)


def test_win_mask_matches_the_vendor_mask():
    """E6: the on-demand mask is the vendor `valid` tensor."""
    T, win, start_pos = 33, 128, 60
    positions = np.arange(start_pos, start_pos + T)
    pos = np.concatenate([np.arange(start_pos), positions])             # part = [history, this chunk]
    ref = ((pos[None, :] <= positions[:, None]) & (pos[None, :] > positions[:, None] - win))[None, None]
    assert np.array_equal(np.array(AP.win_mask(win, start_pos, T, pos.size)), ref)


def test_flags_switch_the_module_attributes():
    """Every lever with a module literal is in the flag table, aligned with that literal, and switches the attribute the code reads per call.
    (The four pfuse3 levers are on by default upstream; the default is the literal, not always "0".)"""
    for name, mod, attr in (("pf_idx_fscore", PI, "FSCORE"), ("pf_idx_topk_mt", PI, "TOPK_MT"),
                            ("pf_attn_pool", AP, "POOLIN"), ("pf_attn_lazymask", AP, "LAZY_MASK"),
                            ("dec_stage", DS, "ENABLED"), ("dec_stage_in", DS, "IN"), ("dec_stage_attn", DS, "ATTN"),
                            ("dec_stage_ffn", DS, "FFN"), ("dec_stage_moe", DS, "MOE")):
        f = flags.FLAGS[name]
        module, attribute, conv = f.attrs[0]
        assert module == mod.__name__.rsplit(".", 1)[-1] and attribute == attr, (name, f.attrs)
        assert f.default in ("0", "1") and f.runtime, name
        try:
            flags.set(name, "1")
            assert getattr(mod, attr) is conv("1"), name
            flags.set(name, "0")
            assert getattr(mod, attr) is conv("0"), name
        finally:
            flags.set(name, f.default)
        assert getattr(mod, attr) is conv(f.default), name


@pytest.mark.skipif(not NX.nax_available(), reason="no neural accelerators on this GPU")
def test_prompt_attention_with_both_attention_levers_matches_stock(monkeypatch):
    """E5 + E6 through the replacement of `DeepseekV41Attention._attention`: two prompt chunks of a window-only layer (the second sees the ring of the
    first), lazy mask and two-input table on, against the stock method."""
    L, A, rng = _local_attention_module()
    monkeypatch.setattr(L, "fake_quant_fp8_ue8m0", lambda x: x)
    stock = L.DeepseekV41Attention._attention
    new = AP._impl_attention_local(stock)
    xs = [mx.array(rng.standard_normal((1, n, 512)).astype(np.float16)) for n in (200, 90)]
    outs = {}
    for name, fn, lazy, poolin in (("stock", stock, False, False), ("new", new, True, True)):
        monkeypatch.setattr(AP, "LAZY_MASK", lazy)
        monkeypatch.setattr(AP, "POOLIN", poolin)
        cache = L.DeepseekV41Cache(1)
        res = []
        for x in xs:
            res.append(np.array(fn(A, cache, x).astype(mx.float32)))
            cache.advance(x.shape[1])
        outs[name] = res
    for a, b in zip(outs["new"], outs["stock"]):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 3e-3
