"""Hy4's Engram forward deltas and dead token ids (kyojin_mlx.engram_gguf), CPU only, synthetic tensors.
Reference = a float64 numpy transcription of llama.cpp build_engram / engram_hash (hy-v4-flash.cpp:340-386, 457-518). The DS4.1 form (no Hy4
tensor given) is checked bit-equal to a frozen copy of the forward before the change."""
import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from kyojin_mlx import engram_gguf as EG  # noqa: E402

EPS = EG.FLT_EPSILON


def _frozen_ds41(x, rows, gate_w, kv_w, eps, clamp=1e-6, act_fp8=True, kv_dtype=None):
    """engram_forward as of 85e38d1f (the DS4.1 form), copied verbatim."""
    L, hc, dim = x.shape
    emb = mx.array(rows).astype(x.dtype).reshape(L, -1)
    if act_fp8:
        emb = EG.fp8_roundtrip(emb)
    kd = mx.float32 if kv_dtype is None else kv_dtype
    kv = (emb.astype(kd) @ mx.array(kv_w, mx.float32).T.astype(kd)).astype(mx.float32)
    key, value = kv[:, :hc * dim].reshape(L, hc, dim), kv[:, hc * dim:]
    h = x.astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(h * h, -1) + eps) * mx.rsqrt(mx.mean(key * key, -1) + eps)
    dot = mx.sum(h * mx.array(gate_w, mx.float32) * key, -1) * rstd * (dim ** -0.5)
    mag = mx.sqrt(mx.maximum(mx.abs(dot), clamp))
    score = mx.where(dot >= 0, mag, -mag)
    out = (h + mx.sigmoid(score)[..., None] * value[:, None, :]).astype(x.dtype)
    return score, out


def _weights(L=7, hc=4, dim=48, n_cols=6, hd=16, seed=0):
    r = np.random.default_rng(seed)
    f = lambda *s, k=1.0: (r.standard_normal(s) * k).astype(np.float32)   # noqa: E731
    return dict(x=f(L, hc, dim), rows=f(L, n_cols, hd, k=0.3), gate=f(hc, dim), kv=f(dim * (hc + 1), n_cols * hd, k=0.1), kv_b=f(hc + 1, dim, k=0.2),
                lookup=f(n_cols, hd, k=0.5) + 1, gamma=f(dim, k=0.3) + 1, hc_scale=f(hc, dim, k=0.2))


def _rms(a, eps):
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps)


def _ref_hy4(w, valid):
    """llama.cpp build_engram in float64."""
    x, L = w["x"].astype(np.float64), w["x"].shape[0]
    hc, dim = x.shape[1:]
    r = _rms(w["rows"].astype(np.float64), EPS) * w["lookup"] * np.broadcast_to(np.asarray(valid, np.float64).reshape(L, -1, 1), w["rows"].shape)
    kv = (r.reshape(L, -1) @ w["kv"].T.astype(np.float64)).reshape(L, hc + 1, dim) + w["kv_b"]
    key, value = kv[:, :hc], kv[:, hc]
    dot = (_rms(x, EPS) * _rms(key, EPS) * w["gate"]).sum(-1) / np.sqrt(dim)
    sign = np.where(dot < 0, -1.0, 1.0)
    gate = 1 / (1 + np.exp(-np.sqrt(np.clip(np.abs(dot), 1e-6, None)) * sign))
    signal = _rms(value, EPS) * w["gamma"]
    delta = signal[:, None, :] * (gate[..., None] + w["hc_scale"])
    anyv = (np.asarray(valid, np.float64).reshape(L, -1).sum(-1) > 0).astype(np.float64)
    return x + delta * anyv[:, None, None], gate


@pytest.mark.parametrize("act_fp8,kv_dtype,dtype", [(True, None, mx.float32), (False, None, mx.float32), (True, mx.bfloat16, mx.bfloat16)])
def test_ds41_form_bit_equal_frozen(act_fp8, kv_dtype, dtype):
    w = _weights()
    x = mx.array(w["x"]).astype(dtype)
    a = EG.engram_forward(x, w["rows"], w["gate"], w["kv"], eps=1e-6, act_fp8=act_fp8, kv_dtype=kv_dtype)
    b = _frozen_ds41(x, w["rows"], w["gate"], w["kv"], eps=1e-6, act_fp8=act_fp8, kv_dtype=kv_dtype)
    assert all(mx.array_equal(p, q).item() for p, q in zip(a, b))


def test_hy4_form_vs_llama_reference():
    w = _weights()
    valid = np.ones(w["x"].shape[0], bool)
    valid[3] = False                                                   # a dead token: no delta at all, though kv_b is not zero
    score, out = EG.engram_forward(mx.array(w["x"]), w["rows"], w["gate"], w["kv"], eps=EPS, act_fp8=False, kv_b=w["kv_b"], lookup_norm=w["lookup"],
                                   pre_gamma=w["gamma"], hc_scale=w["hc_scale"], valid=valid)
    ref, gate = _ref_hy4(w, valid)
    out = np.array(out)
    assert np.abs(out - ref).max() / np.abs(ref).max() < 2e-6
    assert np.array_equal(out[3], w["x"][3])
    g = 1 / (1 + np.exp(-np.array(score)))
    assert np.abs(g - gate)[valid].max() < 1e-6


def test_each_delta_matters():
    """Dropping any one Hy4 tensor changes the output: each delta is really applied."""
    w = _weights(seed=1)
    full = dict(kv_b=w["kv_b"], lookup_norm=w["lookup"], pre_gamma=w["gamma"], hc_scale=w["hc_scale"])
    base = np.array(EG.engram_forward(mx.array(w["x"]), w["rows"], w["gate"], w["kv"], eps=EPS, act_fp8=False, **full)[1])
    for k in full:
        part = {a: b for a, b in full.items() if a != k}
        o = np.array(EG.engram_forward(mx.array(w["x"]), w["rows"], w["gate"], w["kv"], eps=EPS, act_fp8=False, **part)[1])
        assert np.abs(o - base).max() > 1e-3, k


def _spec(vocab=40, cvs=17, pad_tok=5):
    nl, ng, nh = 2, 3, 2
    primes = np.array([101, 103, 107, 109, 113, 127, 131, 137], np.int64).reshape(nl, ng - 1, nh)
    offs = np.concatenate([[0], np.cumsum(primes.reshape(-1))[:-1]]).reshape(nl, -1)
    mult = np.array([[3, 5, 7], [11, 13, 17]], np.int64) * 1001 + 1
    tmap = (np.arange(vocab) * 7) % cvs
    return EG.EngramGgufSpec((1, 3), ng, nh, 8, mult, primes, offs, tmap.astype(np.int64), int(tmap[pad_tok]), int(primes.sum()), "F8_E4M3"), pad_tok


def _llama_hash(spec, ids):
    """engram_hash transcribed (one sequence, prev = the earlier ids)."""
    V = len(spec.token_map)
    cid = lambda t: None if t < 0 or t >= V else int(spec.token_map[t])   # noqa: E731
    out, valid = [], []
    for i, t in enumerate(ids):
        c = cid(t)
        dead = c is None
        toks = [spec.pad_id if dead else c]
        for s in range(1, spec.ngram_size):
            c2 = None if i - s < 0 else cid(ids[i - s])
            toks.append(spec.pad_id if c2 is None else c2)
        row = []
        for li in range(len(spec.layer_ids)):
            m = [int(v) for v in spec.multipliers[li]]
            rolling, cols = (toks[0] * m[0]) % 2**64, []
            for o in range(1, spec.ngram_size):
                rolling ^= (toks[o] * m[o]) % 2**64
                for h in range(spec.n_heads):
                    col = (o - 1) * spec.n_heads + h
                    cols.append(rolling % int(spec.primes[li, o - 1, h]) + int(spec.offsets[li, col]))
            row.append(cols)
        out.append(row)
        valid.append(not dead)
    return np.array(out, np.int64), np.array(valid)


def test_dead_ids_to_pad_as_llama():
    spec, pad_tok = _spec()
    ids = [3, 99, 7, -1, 12, 40, 1, 2, 39]                            # 99, -1, 40 outside token_map [0, 40)
    m = EG.GgufRowMap(spec)
    r1, v1 = m.rows_valid(ids[:4])
    r2, v2 = m.rows_valid(ids[4:])                                     # second call: the history carries the dead ones as the pad
    want, wv = _llama_hash(spec, ids)
    assert np.array_equal(np.concatenate([r1, r2]), want) and np.array_equal(np.concatenate([v1, v2]), wv)
    assert wv.tolist() == [True, False, True, False, True, False, True, True, True]
    m.reset()
    rp = m([pad_tok if (t < 0 or t >= 40) else t for t in ids])        # dead = the pad token's row ids (only validity differs)
    assert np.array_equal(rp, want)
    m.reset()
    assert np.array_equal(m(ids), want)                                # __call__ = rows only, unchanged contract
