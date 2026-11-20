"""Hy4 wiring end to end (CPU only): a synthetic 2-layer Hy4-shaped GGUF through build_spec / load_cpu / Model, against a float64 numpy transcription of
llama.cpp's hy-v4-flash.cpp graph (independent of the kyojin_mlx blocks).
  layer 0: N1 MLA + N3 kpool selection (top_k small enough that pools are dropped), gate, sinks, dense FFN, N2 streams
  layer 1: windowed MLA (no indexer, its own rope base), N4 Engram (Hy4 form), latent MoE (sigmoid + selection bias, normed and scaled weights,
           pre-activation swiglu clamp, shared expert), N2 streams; then the hc head, output norm, lm head.
Prefill of 13 tokens in one call, then the same tokens as 9 + 4 through a KV cache: equal logits."""
import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402
from kyojin_mlx import foreign_skeleton_rows as R  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402
from kyojin_mlx import engram_gguf as EG  # noqa: E402

A = "hy_v4_flash"
D, HC, V = 16, 2, 30
NH = [2, 4]
QL, KVL, NOPE, ROPE, VD = 8, 8, 6, 4, 6
IH, IDX, KP, TOPK = 2, 8, 2, 6
FF, E, K, LAT, EFF, CLAMP = 12, 4, 2, 8, 6, 0.5
NG, NHG, HD = 3, 2, 8
BASE, BASE_SWA, WIN = 10000.0, 500.0, 4
EPS, HC_EPS, MAG = 1e-5, 1e-2, 2.0                           # HC eps large enough that its +eps on pre and post shows
PRIMES = [101, 103, 107, 109]
PAD_TOK = 3
FLT = float(np.finfo(np.float32).eps)


def _model_tensors(rng):
    """{gguf name: numpy array in numpy order (reversed ne)}."""
    f = lambda *s, k=0.3: (rng.standard_normal(s) * k).astype(np.float32)     # noqa: E731
    one = lambda n: (1 + rng.standard_normal(n) * 0.1).astype(np.float32)     # noqa: E731
    t = {"token_embd.weight": f(V, D, k=1.0), "output.weight": f(V, D), "output_norm.weight": one(D),
         "output_hc_fn.weight": f(HC, HC * D), "output_hc_base.weight": f(HC), "output_hc_scale.weight": f(1)}
    for il in range(2):
        nh, p = NH[il], f"blk.{il}."
        t.update({p + "attn_norm.weight": one(D), p + "attn_q_a.weight": f(QL, D), p + "attn_q_a_norm.weight": one(QL),
                  p + "attn_q_b.weight": f(nh * (NOPE + ROPE), QL), p + "attn_kv_a_mqa.weight": f(KVL + ROPE, D), p + "attn_kv_a_norm.weight": one(KVL),
                  p + "attn_k_b.weight": f(nh, KVL, NOPE), p + "attn_v_b.weight": f(nh, VD, KVL), p + "attn_output.weight": f(D, nh * VD),
                  p + "attn_gate.weight": f(nh * VD, D), p + "attn_sinks.weight": f(nh, k=1.0), p + "post_attention_norm.weight": one(D),
                  p + "ffn_norm.weight": one(D), p + "post_ffw_norm.weight": one(D)})
        for s in ("attn", "ffn"):
            t.update({p + f"hc_{s}_fn.weight": f(2 * HC, HC * D), p + f"hc_{s}_base.weight": f(2 * HC), p + f"hc_{s}_scale.weight": f(2, k=1.0)})
    p = "blk.0."
    t.update({p + "indexer.attn_q_b.weight": f(IH * IDX, QL, k=1.0), p + "indexer.attn_k.weight": f(IDX, D, k=1.0), p + "indexer.k_norm.weight": one(IDX),
              p + "indexer.proj.weight": f(IH, D, k=1.0), p + "indexer_compressor_value.weight": f(IDX, IDX, k=1.0),
              p + "indexer_compressor_gate.weight": f(IDX, IDX, k=1.0), p + "indexer_compressor_ape.weight": f(KP, IDX), p + "indexer_compressor_norm.weight": one(IDX),
              p + "ffn_gate.weight": f(FF, D), p + "ffn_up.weight": f(FF, D), p + "ffn_down.weight": f(D, FF)})
    p = "blk.1."
    nc = (NG - 1) * NHG
    t.update({p + "engram_kv.weight": f(D * (HC + 1), nc * HD), p + "engram_kv_b.weight": f(HC + 1, D), p + "engram_gate.weight": f(HC, D, k=1.0),
              p + "engram_hc_scale.weight": f(HC, D), p + "engram_lookup_norm.weight": one(nc * HD).reshape(nc, HD), p + "engram_pre_gamma.weight": one(D),
              p + "ffn_gate_inp.weight": f(E, D, k=1.0), p + "exp_probs_b.bias": f(E), p + "ffn_routed_down.weight": f(LAT, D), p + "ffn_routed_up.weight": f(D, LAT),
              p + "ffn_gate_exps.weight": f(E, EFF, LAT, k=1.0), p + "ffn_up_exps.weight": f(E, EFF, LAT, k=1.0), p + "ffn_down_exps.weight": f(E, LAT, EFF),
              p + "ffn_gate_shexp.weight": f(EFF, D), p + "ffn_up_shexp.weight": f(EFF, D), p + "ffn_down_shexp.weight": f(D, EFF)})
    return t


def _kv():
    offs = np.concatenate([[0], np.cumsum(PRIMES)[:-1]]).tolist()
    tmap = ((np.arange(V) * 7) % 23).tolist()
    return {"general.architecture": A, f"{A}.block_count": 2, f"{A}.context_length": 64, f"{A}.embedding_length": D, f"{A}.feed_forward_length": FF,
            f"{A}.attention.head_count": NH, f"{A}.attention.head_count_kv": 1, f"{A}.attention.key_length": KVL + ROPE, f"{A}.attention.value_length": KVL,
            f"{A}.attention.layer_norm_rms_epsilon": EPS, f"{A}.rope.freq_base": BASE, f"{A}.rope.freq_base_swa": BASE_SWA, f"{A}.rope.dimension_count": ROPE,
            f"{A}.attention.sliding_window": WIN, f"{A}.attention.sliding_window_pattern": [False, True], f"{A}.vocab_size": V,
            f"{A}.attention.q_lora_rank": QL, f"{A}.attention.kv_lora_rank": KVL, f"{A}.attention.key_length_mla": NOPE + ROPE, f"{A}.attention.value_length_mla": VD,
            f"{A}.attention.indexer.head_count": IH, f"{A}.attention.indexer.key_length": IDX, f"{A}.attention.indexer.top_k": TOPK,
            f"{A}.attention.indexer.types": [True, False], f"{A}.attention.indexer.kpool": KP, f"{A}.attention.indexer.kpool_select_tail": True,
            f"{A}.hyper_connection.count": HC, f"{A}.hyper_connection.epsilon": HC_EPS, f"{A}.hyper_connection.magnitude": MAG,
            f"{A}.expert_count": E, f"{A}.expert_used_count": K, f"{A}.expert_shared_count": 1, f"{A}.expert_feed_forward_length": EFF,
            f"{A}.expert_weights_scale": 2.0, f"{A}.expert_weights_norm": True, f"{A}.expert_gating_func": 2, f"{A}.leading_dense_block_count": 1,
            f"{A}.expert_latent_length": LAT, f"{A}.swiglu_clamp_exp": [0.0, CLAMP],
            f"{A}.ple.layers": [1], f"{A}.ple.ngram_size": NG, f"{A}.ple.heads_per_ngram": NHG, f"{A}.embedding_length_per_layer_input": HD,
            f"{A}.ple.layer_multipliers": [2 * 1234567 + 1, 2 * 7654321 + 1, 2 * 1111111 + 1], f"{A}.ple.head_vocab_sizes": PRIMES, f"{A}.ple.head_offsets": offs,
            f"{A}.engram.token_map": tmap, f"{A}.engram.pad_token_id": PAD_TOK}


def _write(path, t, table):
    out = [(EG.TABLE, (HD, table.shape[0]), "F8_E4M3", table)]
    for n, a in t.items():
        out.append((n, a.shape[::-1], "F32", a.astype(np.float32)))
    SY.write_gguf(path, _kv(), out)


# ---------------------------------------------------------------------------------------------------------------- numpy reference (float64)
def _rms(x, w=None, eps=EPS):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _rope_norm(x, pos, base):
    """ggml rope NORM on the whole last axis (consecutive pairs); x [..., L, n] with pos [L]."""
    n = x.shape[-1]
    th = pos[:, None] * base ** (-np.arange(0, n, 2) / n)
    c, s = np.cos(th), np.sin(th)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
    return out


def _hc_pre(x, t, p):
    L = x.shape[0]
    m = _rms(x.reshape(L, -1)) @ t[p + "_fn.weight"].T
    sc, b = t[p + "_scale.weight"], t[p + "_base.weight"]
    pre = _sig(m[:, :HC] * sc[0] + b[:HC]) + HC_EPS
    post = _sig(m[:, HC:2 * HC] * sc[1] + b[HC:2 * HC]) * MAG + HC_EPS
    return (pre[..., None] * x).sum(1), post


def _kpool_keep(t, cur, qr, pos, base):
    p = "blk.0."
    L = cur.shape[0]
    iq = (qr @ t[p + "indexer.attn_q_b.weight"].T).reshape(L, IH, IDX)
    iq = np.concatenate([iq[..., :IDX - ROPE], _rope_norm(iq[..., IDX - ROPE:].transpose(1, 0, 2), pos, base).transpose(1, 0, 2)], -1)
    ik = _rms(cur @ t[p + "indexer.attn_k.weight"].T, t[p + "indexer.k_norm.weight"])
    val, gate = ik @ t[p + "indexer_compressor_value.weight"].T, ik @ t[p + "indexer_compressor_gate.weight"].T
    w = cur @ t[p + "indexer.proj.weight"].T / np.sqrt(IDX * IH)
    npool = L // KP
    pk = []
    for j in range(npool):
        lg = gate[j * KP:(j + 1) * KP] + t[p + "indexer_compressor_ape.weight"]
        pr = np.exp(lg - lg.max(0)) / np.exp(lg - lg.max(0)).sum(0)
        pooled = _rms((pr * val[j * KP:(j + 1) * KP]).sum(0), t[p + "indexer_compressor_norm.weight"])
        back = _rope_norm(pooled[None, None, IDX - ROPE:], np.array([j * KP], np.float64), base)[0, 0]
        pk.append(np.concatenate([pooled[:IDX - ROPE], back]))
    pk = np.array(pk)
    keep = np.zeros((L, L), bool)
    n_top = TOPK // KP - 1
    dropped = 0
    for i in range(L):
        vis = [j for j in range(npool) if (j + 1) * KP <= i]               # pools that end before the token (select_tail)
        sc = {j: (np.maximum(iq[i] @ pk[j], 0) * w[i]).sum() for j in vis}
        top = sorted(vis, key=lambda j: -sc[j])[:n_top]
        dropped += len(vis) - len(top)
        for j in top:
            keep[i, j * KP:(j + 1) * KP] = True
        g0 = (i // KP) * KP
        keep[i, g0:i + 1] = True                                            # the tail: own group up to the token
    return keep, dropped


def _attention(t, il, cur, pos):
    p, nh = f"blk.{il}.", NH[il]
    L = cur.shape[0]
    base = BASE_SWA if il == 1 else BASE
    qr = _rms(cur @ t[p + "attn_q_a.weight"].T, t[p + "attn_q_a_norm.weight"])
    q = (qr @ t[p + "attn_q_b.weight"].T).reshape(L, nh, NOPE + ROPE).transpose(1, 0, 2)
    q_nope, q_pe = q[..., :NOPE], _rope_norm(q[..., NOPE:], pos, base)
    kv = cur @ t[p + "attn_kv_a_mqa.weight"].T
    lat, k_pe = _rms(kv[:, :KVL], t[p + "attn_kv_a_norm.weight"]), _rope_norm(kv[None, :, KVL:], pos, base)[0]
    k_nope = np.einsum("sc,hcn->hsn", lat, t[p + "attn_k_b.weight"])
    v = np.einsum("sc,hvc->hsv", lat, t[p + "attn_v_b.weight"])
    s = (np.einsum("hln,hsn->hls", q_nope, k_nope) + np.einsum("hlr,sr->hls", q_pe, k_pe)) / np.sqrt(NOPE + ROPE)
    qi, ki = np.arange(L)[:, None], np.arange(L)[None]
    vis = ki <= qi
    dropped = 0
    if il == 0:
        keep, dropped = _kpool_keep(t, cur, qr, pos, base)
        vis = vis & keep
    else:
        vis = vis & (qi - ki < WIN)
    s = np.where(vis[None], s, -np.inf)
    s = np.concatenate([s, np.broadcast_to(t[p + "attn_sinks.weight"][:, None, None], (nh, L, 1))], -1)
    pr = np.exp(s - s.max(-1, keepdims=True))
    pr = (pr / pr.sum(-1, keepdims=True))[..., :L]
    o = np.einsum("hls,hsv->lhv", pr, v).reshape(L, -1)
    o = o * _sig(cur @ t[p + "attn_gate.weight"].T)
    return _rms(o @ t[p + "attn_output.weight"].T, t[p + "post_attention_norm.weight"]), dropped


def _silu(x):
    return x * _sig(x)


def _moe(t, x):
    p = "blk.1."
    probs = _sig(x @ t[p + "ffn_gate_inp.weight"].T)
    sel = probs + t[p + "exp_probs_b.bias"]
    xl = x @ t[p + "ffn_routed_down.weight"].T
    out = np.zeros_like(xl)
    for i in range(x.shape[0]):
        top = np.argsort(-sel[i], kind="stable")[:K]
        w = probs[i, top]
        w = w / max(w.sum(), 6.103515625e-5) * 2.0
        for e, we in zip(top, w):
            g = np.minimum(xl[i] @ t[p + "ffn_gate_exps.weight"][e].T, CLAMP)
            u = np.clip(xl[i] @ t[p + "ffn_up_exps.weight"][e].T, -CLAMP, CLAMP)
            out[i] += we * ((g * _sig(g) * u) @ t[p + "ffn_down_exps.weight"][e].T)
    sh = (_silu(x @ t[p + "ffn_gate_shexp.weight"].T) * (x @ t[p + "ffn_up_shexp.weight"].T)) @ t[p + "ffn_down_shexp.weight"].T
    return out @ t[p + "ffn_routed_up.weight"].T + sh


def _engram(t, x, rows, valid):
    p = "blk.1."
    L = x.shape[0]
    r = _rms(rows, t[p + "engram_lookup_norm.weight"], FLT) * valid[:, None, None]
    kv = (r.reshape(L, -1) @ t[p + "engram_kv.weight"].T).reshape(L, HC + 1, D) + t[p + "engram_kv_b.weight"]
    key, value = kv[:, :HC], kv[:, HC]
    dot = (_rms(x, None, FLT) * _rms(key, None, FLT) * t[p + "engram_gate.weight"]).sum(-1) / np.sqrt(D)
    gate = _sig(np.sqrt(np.maximum(np.abs(dot), 1e-6)) * np.where(dot < 0, -1.0, 1.0))
    delta = _rms(value, t[p + "engram_pre_gamma.weight"], FLT)[:, None] * (gate[..., None] + t[p + "engram_hc_scale.weight"])
    return x + delta * valid[:, None, None]


def _reference(t, ids, rows, valid):
    t = {k: v.astype(np.float64) for k, v in t.items()}
    L = len(ids)
    pos = np.arange(L, dtype=np.float64)
    x = np.repeat(t["token_embd.weight"][ids][:, None], HC, 1)
    drops = 0
    for il in range(2):
        p = f"blk.{il}."
        if il == 1:
            x = _engram(t, x, rows, valid)
        cur, post = _hc_pre(x, t, p + "hc_attn")
        a, dr = _attention(t, il, _rms(cur, t[p + "attn_norm.weight"]), pos)
        drops += dr
        x = x + post[..., None] * a[:, None]
        cur, post = _hc_pre(x, t, p + "hc_ffn")
        c = _rms(cur, t[p + "ffn_norm.weight"])
        f = ((_silu(c @ t[p + "ffn_gate.weight"].T) * (c @ t[p + "ffn_up.weight"].T)) @ t[p + "ffn_down.weight"].T) if il == 0 else _moe(t, c)
        x = x + post[..., None] * _rms(f, t[p + "post_ffw_norm.weight"])[:, None]
    m = _rms(x.reshape(L, -1)) @ t["output_hc_fn.weight"].T
    pre = _sig(m * t["output_hc_scale.weight"] + t["output_hc_base.weight"]) + HC_EPS
    y = _rms((pre[..., None] * x).sum(1), t["output_norm.weight"])
    return y @ t["output.weight"].T, drops


# ---------------------------------------------------------------------------------------------------------------------------------- tests
@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module=None):
    rng = np.random.default_rng(7)
    t = _model_tensors(rng)
    table = rng.integers(0, 256, (sum(PRIMES), HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38                                    # no NaN codes
    p = str(tmp_path_factory.mktemp("hy4") / "hy4_2l.gguf")
    _write(p, t, table)
    model, rep = SK.load_cpu(p, row=R.ROWS[A])
    return t, table, p, model, rep


IDS = [5, 17, 3, 29, 8, 8, 12, 1, 44, 21, 7, 0, 13]                     # taken modulo the vocab; the dead-id case is its own test


def _ids():
    return [i % V for i in IDS]


def _rows_valid(table, ids_for_engram):
    spec = EG.EngramGgufSpec((1,), NG, NHG, HD, np.array([_kv()[f"{A}.ple.layer_multipliers"]], np.int64), np.array(PRIMES, np.int64).reshape(1, NG - 1, NHG),
                             np.concatenate([[0], np.cumsum(PRIMES)[:-1]]).reshape(1, -1), np.array(_kv()[f"{A}.engram.token_map"], np.int64),
                             int(_kv()[f"{A}.engram.token_map"][PAD_TOK]), sum(PRIMES), "F8_E4M3")
    r, v = EG.GgufRowMap(spec).rows_valid(ids_for_engram)
    return GF.F8_E4M3_TAB[table[r[:, 0]]].astype(np.float64), v.astype(np.float64)


def test_spec_fields(built):
    _, _, _, model, rep = built
    sp = rep["spec"]
    assert (sp.mla_q_lora, sp.mla_kv_lora, sp.mla_nope, sp.mla_v, sp.kpool, sp.kpool_select_tail) == (QL, KVL, NOPE, VD, KP, True)
    assert (sp.hc_count, sp.hc_post_eps, sp.engram_layers, sp.swa_layers) == (HC, True, [1], [False, True])
    assert sp.engram_eps == float(np.finfo(np.float32).eps) == 1.1920928955078125e-07     # FLT_EPSILON (llama.cpp engram rms), not the 1e-5 norm eps
    a0, a1 = model.layers[0].self_attn, model.layers[1].self_attn
    assert a0.kp is not None and a1.kp is None and (a0.base, a1.base) == (BASE, BASE_SWA) and (a0.swa, a1.swa) == (False, True)
    assert "engram" in model.layers[1] and "engram" not in model.layers[0]


def test_end_to_end_vs_llama_graph(built):
    t, table, _, model, _ = built
    ids = _ids()
    rows, valid = _rows_valid(table, ids)
    want, dropped = _reference(t, np.array(ids), rows, valid)
    assert dropped > 0                                                   # the kpool selection really drops pools
    got = np.array(model(mx.array([ids]))[0]).astype(np.float64)
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-5, rel


def test_dead_engram_id_masks_the_position(built):
    """An id outside token_map: no Engram delta there (llama.cpp valid = 0), and its window slot is the pad for later positions."""
    t, table, p, model, _ = built
    ids = _ids()
    src = SK.engram_source_of(GF.GgufFile(p))
    eng = src(np.array(ids[:8] + [V + 5] + ids[9:]), 0)
    rows, valid = eng[1]
    assert valid.tolist() == [True] * 8 + [False] + [True] * 4
    want, _ = _reference(t, np.array(ids), rows.astype(np.float64), valid.astype(np.float64))
    model.engram_source = lambda a, off: eng
    try:
        got = np.array(model(mx.array([ids]))[0]).astype(np.float64)
    finally:
        model.engram_source = SK.engram_source_of(GF.GgufFile(p))
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5


def test_cached_prefill_equals_one_call(built):
    from mlx_lm.models.cache import KVCache
    _, _, _, model, _ = built
    ids = _ids()
    full = np.array(model(mx.array([ids]))[0])
    cache = [KVCache() for _ in model.layers]
    a = np.array(model(mx.array([ids[:9]]), cache=cache)[0])
    b = np.array(model(mx.array([ids[9:]]), cache=cache)[0])
    got = np.concatenate([a, b])
    assert np.abs(got - full).max() / np.abs(full).max() < 1e-5
