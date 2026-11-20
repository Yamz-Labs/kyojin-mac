"""Item 3: attention with value head dim != key head dim, a value scale (output multiplier after o_proj) and attention sinks, as plain ops, through the GGUF loader,
against float64 numpy written from mimo2.cpp:149-166 (rope, build_attn with sinks, scale) and llama-graph.cpp:3120-3130 (sink = extra softmax logit, no value).
Token-by-token decode through the mlx-lm cache equals the full forward (keys and values have different widths in the cache). CPU device only."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

from kyojin_mlx import foreign_skeleton as SK, foreign_synth as SY  # noqa: E402

_spec = importlib.util.spec_from_file_location("tf", os.path.join(os.path.dirname(__file__), "test_foreign_skeleton_fields.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
D, NH, NKV, HD, HV, FF, V, NL, ROT = 16, 4, 2, 6, 4, 24, 20, 2, 4
IDS, EPS = F.IDS, F.EPS


def build(tmp_path, seed=1, vscale=0.0, sinks=False, window=0, gate=False, o_norm=False, same_v=False, indexer=False, blk=2, topk=2, local=1):
    rng = np.random.default_rng(seed)
    hv = HD if same_v else HV
    kv = {"general.architecture": "synth", "synth.block_count": NL, "synth.embedding_length": D, "synth.feed_forward_length": FF, "synth.attention.head_count": NH,
          "synth.attention.head_count_kv": NKV, "synth.attention.key_length": HD, "synth.attention.value_length": hv, "synth.context_length": 64,
          "synth.rope.dimension_count": ROT, "synth.rope.freq_base": 100.0, "synth.attention.layer_norm_rms_epsilon": EPS}
    if vscale:
        kv["synth.attention.value_scale"] = vscale
    if window:
        kv["synth.attention.sliding_window"], kv["synth.attention.sliding_window_pattern"] = window, [1, 0]
    if indexer:
        kv.update({"synth.attention.indexer.head_count": NKV, "synth.attention.indexer.key_length": HD, "synth.attention.indexer.top_k": topk,
                   "synth.attention.indexer.block_size": blk, "synth.attention.indexer.local_blocks": local})
    T = {}
    w = lambda n, *sh, s=0.4: T.__setitem__(n, (rng.standard_normal(sh) * s).astype(np.float32))      # noqa: E731
    w("token_embd.weight", V, D, s=1.0)
    T["output_norm.weight"] = (1 + 0.2 * rng.standard_normal(D)).astype(np.float32)
    for i in range(NL):
        p = f"blk.{i}."
        for n in ("attn_norm", "ffn_norm"):
            T[p + n + ".weight"] = (1 + 0.2 * rng.standard_normal(D)).astype(np.float32)
        w(p + "attn_q.weight", NH * HD, D)
        w(p + "attn_k.weight", NKV * HD, D)
        w(p + "attn_v.weight", NKV * hv, D)
        w(p + "attn_output.weight", D, NH * hv)
        w(p + "ffn_gate.weight", FF, D)
        w(p + "ffn_up.weight", FF, D)
        w(p + "ffn_down.weight", D, FF)
        if sinks:
            w(p + "attn_sinks.weight", NH, s=1.5)
        if gate:
            w(p + "attn_gate.weight", NH * hv, D, s=0.8)
        if indexer and i == 1:                                      # layer 0 stays dense (a layer without indexer tensors is dense)
            w(p + "indexer.q_proj.weight", NKV * HD, D, s=0.8)
            w(p + "indexer.k_proj.weight", HD, D, s=0.8)
            for n in ("q", "k"):
                T[p + f"indexer.{n}_norm.weight"] = (1 + 0.2 * rng.standard_normal(HD)).astype(np.float32)
    path = str(tmp_path / "a.gguf")
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    row = dict(norm="rms", act="silu", rope="neox", norm_pos="pre", **({"attn_out_norm": True} if o_norm else {}))
    return path, row, T, dict(vscale=vscale, sinks=sinks, window=window, gate=gate, o_norm=o_norm, hv=hv, indexer=indexer, blk=blk, topk=topk, local=local)


def ref(T, o, ids):
    g = lambda n: T[n].astype(np.float64)      # noqa: E731
    L, hv = len(ids), o["hv"]
    x = g("token_embd.weight")[ids]
    pos = np.arange(L, dtype=np.float64)
    for i in range(NL):
        p = f"blk.{i}."
        n1 = F.rms(x, g(p + "attn_norm.weight"))
        q = (n1 @ g(p + "attn_q.weight").T).reshape(L, NH, HD)
        k = (n1 @ g(p + "attn_k.weight").T).reshape(L, NKV, HD)
        v = (n1 @ g(p + "attn_v.weight").T).reshape(L, NKV, hv)
        q, k = F.rope(q, pos, 100.0, ROT), F.rope(k, pos, 100.0, ROT)
        win = o["window"] and i == 0                                # pattern [1, 0]: layer 0 windowed
        out = np.zeros((L, NH, hv))
        keepg = None
        if o["indexer"] and i == 1:                                 # minimax-m3.cpp:346-356 (projections, norm, rope), 398-410 / 479-494 (scores, block max, local bias, top-k)
            iq = F.rope(F.rms((n1 @ g(p + "indexer.q_proj.weight").T).reshape(L, NKV, HD), g(p + "indexer.q_norm.weight")), pos, 100.0, ROT)
            ik = F.rope(F.rms((n1 @ g(p + "indexer.k_proj.weight").T).reshape(L, 1, HD), g(p + "indexer.k_norm.weight")), pos, 100.0, ROT)[:, 0]
            blk, nblk = o["blk"], -(-L // o["blk"])
            keepg = np.zeros((NKV, L, L), bool)
            for gi in range(NKV):
                for t in range(L):
                    sc = np.full(nblk * blk, -np.inf)
                    sc[:t + 1] = ik[:t + 1] @ iq[t, gi]
                    bs = sc.reshape(nblk, blk).max(-1)
                    for b in range(nblk):
                        if b <= t // blk and b > t // blk - o["local"]:
                            bs[b] += 1e30
                    sel = np.argsort(-bs, kind="stable")[:min(o["topk"], nblk)]
                    keepg[gi, t] = np.isin(np.arange(L) // blk, sel)
        for h in range(NH):
            s = (q[:, h] @ k[:, h // (NH // NKV)].T) / np.sqrt(HD)
            vis = np.tril(np.ones((L, L), bool))
            if win:
                vis &= (np.arange(L)[:, None] - np.arange(L)[None, :]) < o["window"]
            if keepg is not None:
                vis &= keepg[h // (NH // NKV)]
            s = np.where(vis, s, -np.inf)
            m = s.max(-1, keepdims=True)
            e = np.exp(s - m)
            den = e.sum(-1, keepdims=True) + (np.exp(g(p + "attn_sinks.weight")[h] - m) if o["sinks"] else 0.0)
            out[:, h] = (e / den) @ v[:, h // (NH // NKV)]
        if o["o_norm"]:
            out = F.rms(out, None)
        if o["gate"]:
            gt = (n1 @ g(p + "attn_gate.weight").T).reshape(L, NH, hv)
            out = out * (1 / (1 + np.exp(-gt)))
        a = out.reshape(L, -1) @ g(p + "attn_output.weight").T
        if o["vscale"]:
            a = a * o["vscale"]
        x = x + a
        n2 = F.rms(x, g(p + "ffn_norm.weight"))
        x = x + ((F.silu(n2 @ g(p + "ffn_gate.weight").T) * (n2 @ g(p + "ffn_up.weight").T)) @ g(p + "ffn_down.weight").T)
    return F.rms(x, g("output_norm.weight")) @ g("token_embd.weight").T


CASES = {
    "value_dim_only": dict(),
    "value_scale": dict(vscale=0.707),
    "sinks": dict(sinks=True),
    "sinks_window_value_scale": dict(sinks=True, window=3, vscale=0.707),
    "window_value_scale_no_sinks": dict(window=3, vscale=0.5),
    "gate_and_out_norm_use_the_value_dim": dict(gate=True, o_norm=True, sinks=True),
    "equal_dims_with_sinks": dict(sinks=True, same_v=True),
}


@pytest.mark.parametrize("name", list(CASES))
def test_value_dim_scale_sinks(tmp_path, name):
    path, row, T, o = build(tmp_path, **CASES[name])
    model, _ = SK.load_cpu(path, row=row)
    got = np.array(model(mx.array(IDS[None]))[0]).astype(np.float64)
    want = ref(T, o, IDS)
    diff = np.abs(got - want).max()
    print(name, "max abs diff vs numpy %.3g" % diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    cd = F.cache_consistent(model, got)
    print(name, "token-by-token vs full forward %.3g" % cd)
    assert cd < 1e-4


@pytest.mark.parametrize("name,kw", [("indexer", dict(indexer=True)), ("indexer_top1", dict(indexer=True, topk=1, local=1)), ("indexer_block3_top2_local2", dict(indexer=True, blk=3, topk=2, local=2)),
                                     ("indexer_sinks_window_vscale", dict(indexer=True, sinks=True, window=4, vscale=0.7))])
def test_block_sparse_indexer(tmp_path, name, kw):
    """Indexer formula: minimax-m3.cpp (a learned block-level top-k: max of iq . ik per block of positions, the newest blocks forced, only the kept blocks are attended)."""
    path, row, T, o = build(tmp_path, seed=3, **kw)
    model, _ = SK.load_cpu(path, row=row)
    got = np.array(model(mx.array(IDS[None]))[0]).astype(np.float64)
    want = ref(T, o, IDS)
    diff = np.abs(got - want).max()
    print(name, "max abs diff vs numpy %.3g" % diff)
    assert diff < 2e-4 * max(1.0, np.abs(want).max())
    cd = F.cache_consistent(model, got)
    print(name, "token-by-token vs full forward %.3g" % cd)
    assert cd < 1e-4
    assert model.layers[1].self_attn.idx and not model.layers[0].self_attn.idx
    dense = dict(kw, indexer=False)
    path2, row2, T2, o2 = build(tmp_path, seed=3, **dense)
    d = np.array(SK.load_cpu(path2, row=row2)[0](mx.array(IDS[None]))[0])
    # the same weights without the indexer give another output: the selection really drops blocks (the indexer tensors are unused there, so the files differ only by them)
    assert np.abs(d - got).max() > 1e-4


def test_indexer_refusals(tmp_path):
    path, row, T, o = build(tmp_path, indexer=True)
    import kyojin_mlx.foreign_gguf as GF
    kv = {k: v for k, v in GF.GgufFile(path).kv.items() if not k.startswith("tokenizer.")}
    kv["synth.attention.indexer.head_count"] = 3
    SY.write_gguf(path, kv, [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    with pytest.raises(SK.SkeletonError, match="indexer"):
        SK.load_cpu(path, row=row)


def test_fields_change_the_output_and_bad_shapes_are_refused(tmp_path):
    outs = []
    for kw in (dict(), dict(sinks=True), dict(vscale=0.5)):
        path, row, T, o = build(tmp_path, **kw)
        outs.append(np.array(SK.load_cpu(path, row=row)[0](mx.array(IDS[None]))[0]))
    assert np.abs(outs[0] - outs[1]).max() > 1e-3 and np.abs(outs[0] - outs[2]).max() > 1e-3
    path, row, T, o = build(tmp_path, sinks=True)
    T["blk.1.attn_sinks.weight"] = T["blk.1.attn_sinks.weight"][:3]
    SY.write_gguf(path, {k: v for k, v in __import__("kyojin_mlx.foreign_gguf", fromlist=["x"]).GgufFile(path).kv.items() if not k.startswith("tokenizer.")},
                  [(n, tuple(reversed(a.shape)), "F32", np.ascontiguousarray(a)) for n, a in T.items()])
    with pytest.raises(SK.SkeletonError, match="attn_sinks"):
        SK.load_cpu(path, row=row)
