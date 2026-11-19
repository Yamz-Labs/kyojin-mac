"""GLM-5.3 block drafters (kyojin_mlx.glm_block) on tiny random weights: the per-session context ring, the window, the tap hooks. The real checkpoints are checked
against the references by tools/glm_block_check.py."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from kyojin_mlx import glm_block as GB

H, V, HD, NH, KV, NL, T, FF = 64, 300, 16, 4, 2, 2, 3, 96
WIN = 8


def _w(*shape):
    return (mx.random.normal(shape) * 0.1).astype(mx.bfloat16)


def _layer_w(w, kind):
    for i in range(NL):
        p = f"layers.{i}."
        kvh = KV if kind == "dflash2" else NH
        w[p + "input_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        w[p + "post_attention_layernorm.weight"] = mx.ones((H,), dtype=mx.bfloat16)
        w[p + "self_attn.q_norm.weight"] = mx.ones((HD,), dtype=mx.bfloat16)
        w[p + "self_attn.k_norm.weight"] = mx.ones((HD,), dtype=mx.bfloat16)
        w[p + "self_attn.q_proj.weight"] = _w(NH * HD, H)
        w[p + "self_attn.k_proj.weight"] = _w(kvh * HD, H)
        w[p + "self_attn.v_proj.weight"] = _w(kvh * HD, H)
        w[p + "self_attn.o_proj.weight"] = _w(H, NH * HD)
        w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"], w[p + "mlp.down_proj.weight"] = _w(FF, H), _w(FF, H), _w(H, FF)


def make(kind):
    cfg, w, extra = weights(kind)
    if kind == "dflash2":
        emb, head = extra
        return cfg, GB.DFlash2(None, cfg, w, 0, lambda ids: emb[ids], lambda h: h @ head.T)
    return cfg, GB.DSpark(None, cfg, w, 0)


def weights(kind, vocab=None, hidden_cfg_out=None):
    """(Cfg, tensors, extra): tiny random checkpoint; `hidden_cfg_out` (dict) receives the config.json text of the checkpoint."""
    mx.random.seed(3)
    if kind == "dflash2":
        d = dict(num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=KV, head_dim=HD, hidden_size=H, rms_norm_eps=1e-5, sliding_window=WIN,
                 rope_parameters={"rope_theta": 10000.0}, vocab_size=V, is_causal=False,
                 dflash_config=dict(block_size=8, mask_token_id=V - 1, target_layer_ids=list(range(T)), conv_kernel_size=2, conv_group_size=16, selector_rank=8, selector_top_k=4))
        cfg = GB.Cfg(d, "dflash2")
    else:
        d = dict(block_size=8, mask_token_id=V - 1, aux_hidden_state_layer_ids=list(range(1, T + 1)), sliding_window_non_causal=False, markov_rank=8, enable_confidence_head=True,
                 confidence_head_with_markov=True,
                 transformer_layer_config=dict(num_hidden_layers=NL, num_attention_heads=NH, num_key_value_heads=NH, head_dim=HD, hidden_size=H, rms_norm_eps=1e-5,
                                               sliding_window=WIN, rope_parameters={"rope_theta": 10000.0}, vocab_size=V))
        cfg = GB.Cfg(d, "dspark")
    w = {"fc.weight": _w(H, T * H), "hidden_norm.weight": mx.ones((H,), dtype=mx.bfloat16), "norm.weight": mx.ones((H,), dtype=mx.bfloat16)}
    _layer_w(w, kind)
    if kind == "dflash2":
        for i in range(NL):
            for n in ("attention_conv", "mlp_conv"):
                w[f"layers.{i}.{n}.base_kernel"] = _w(2, 2, H)
                w[f"layers.{i}.{n}.kernel_projection.weight"] = _w(2 * 2 * (H // 16), H)
        w["candidate_selector.hidden_projection.weight"] = _w(8, H)
        w["candidate_selector.predecessor_codebook"], w["candidate_selector.successor_codebook"] = _w(V, 8), _w(V, 8)
        if hidden_cfg_out is not None:
            hidden_cfg_out.update(d)
        return cfg, w, (_w(V, H), _w(V, H))
    w["embed_tokens.weight"], w["lm_head.weight"] = _w(V, H), _w(V, H)
    w["markov_head.markov_w1.weight"], w["markov_head.markov_w2.weight"] = _w(V, 8), _w(V, 8)
    w["confidence_head.proj.weight"], w["confidence_head.proj.bias"] = _w(1, H + 8), _w(1)
    if hidden_cfg_out is not None:
        hidden_cfg_out.update(d)
    return cfg, w, None


def taps(n):
    return (mx.random.normal((n, T * H)) * 2).astype(mx.bfloat16)


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_ring_incremental_equals_oneshot(kind):
    cfg, dr = make(kind)
    x = taps(30)
    a = GB.BlockDraft(cfg.window)
    a.append(x, None, 0)
    dr.feed(a)
    b = GB.BlockDraft(cfg.window)
    for s in range(0, 30, 7):
        b.append(x[s:s + 7], None, s)
        if s % 14 == 0:
            dr.feed(b)
    dr.feed(b)
    assert a.rpos == b.rpos and a.rn == b.rn == cfg.window
    for i in range(NL):
        assert mx.allclose(a.ks[i].astype(mx.float32), b.ks[i].astype(mx.float32), atol=1e-2)
        assert mx.allclose(a.vs[i].astype(mx.float32), b.vs[i].astype(mx.float32), atol=1e-2)


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_window_ring_matches_short_context(kind):
    """Rows older than the window are never seen: a state that only ever got the last `window` rows drafts the same tokens."""
    cfg, dr = make(kind)
    x = taps(25)
    full = GB.BlockDraft(cfg.window)
    full.append(x, None, 0)
    tail = GB.BlockDraft(cfg.window)
    tail.append(x[25 - cfg.window:], None, 25 - cfg.window)
    nd = 4 if kind == "dflash2" else 5
    h1 = dr.block_hidden(_fed(dr, full), 5, 25, nd + 1)
    h2 = dr.block_hidden(_fed(dr, tail), 5, 25, nd + 1)
    assert mx.allclose(h1.astype(mx.float32), h2.astype(mx.float32), atol=2e-2)


def _fed(dr, st):
    dr.feed(st)
    return st


@pytest.mark.parametrize("kind", ["dflash2", "dspark"])
def test_draft_shapes_alignment_and_fork(kind):
    cfg, dr = make(kind)
    st = GB.BlockDraft(cfg.window)
    assert dr.draft(st, 3, 0, 4) is None                         # nothing fed
    st.append(taps(12), None, 0)
    assert dr.draft(st.fork(), 3, 99, 4) is None                 # not lined up with p0
    nd = 7 if kind == "dflash2" else 8
    o = dr.draft(st.fork(), 3, 12, nd)
    mx.eval(o)
    assert o.shape == (nd,) and o.dtype == mx.int32
    assert st.hid is not None and st.pos == 12 - cfg.window      # the fork consumed its own copy of the pending rows (a 12-row append keeps the last window rows)
    o2 = dr.draft(st, 3, 12, nd)
    assert o.tolist() == o2.tolist()
    if kind == "dspark":
        assert st.conf is not None and st.conf.shape == (nd,)
        assert float(st.conf.max()) <= 1.0 and bool(mx.all(st.conf[1:] <= st.conf[:-1] + 1e-6))      # cumulative product


def test_dspark_block_is_causal_prefix():
    """A shorter DSpark draft is the prefix of the longer one."""
    cfg, dr = make("dspark")
    st = GB.BlockDraft(cfg.window)
    st.append(taps(12), None, 0)
    a = dr.draft(st.fork(), 3, 12, 8).tolist()
    b = dr.draft(st.fork(), 3, 12, 3).tolist()
    assert a[:3] == b


def test_gap_restarts_ring():
    cfg, dr = make("dspark")
    st = GB.BlockDraft(cfg.window)
    st.append(taps(6), None, 0)
    dr.feed(st)
    st.append(taps(3), None, 40)                                 # prefix-cache style jump
    assert st.ks is None and st.pos == 40 and st.m == 3


class _Layer(nn.Module):
    def __call__(self, x, mask=None, cache=None, prev=None):
        return x * 2, prev


def test_taps_hooks():
    layers = [_Layer() for _ in range(4)]
    tp = GB.Taps(layers, [1, 3])
    x = mx.ones((1, 2, 3, 5))                                    # [B, L, hc, H]
    for l in layers:
        x, _ = l(x)
    assert tp.d == {} and type(layers[0]) is _Layer              # not installed: nothing touched
    tp.install()
    assert type(layers[1]) is not _Layer and type(layers[0]) is _Layer
    x = mx.ones((1, 2, 3, 5))
    for l in layers:
        x, _ = l(x)
    assert tp.d == {}                                            # installed but off: nothing recorded
    tp.begin()
    x = mx.ones((1, 2, 3, 5))
    for l in layers:
        x, _ = l(x)
    out = tp.end()
    assert out.shape == (1, 2, 10) and not tp.on
    assert float(out[0, 0, 0]) == 4.0 and float(out[0, 0, 5]) == 16.0      # output of layer 1 = 2^2, of layer 3 = 2^4
    with pytest.raises(ValueError):
        GB.Taps(layers, [9])
    assert GB.Taps(layers, [9], clamp=True).have == [3]
