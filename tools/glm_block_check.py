# Numerics of the GLM-5.3 block drafters (kyojin_mlx.glm_block) in isolation: random target taps, random stand-in embedding / lm_head, CPU only (no GPU).
#   DFlash2: block hidden + selected path vs z-lab/dflash dflash/model_mlx.py (MIT; clone it and pass --zlab DIR), real weights.
#   DSpark : block hidden, markov chain and confidence vs a naive full-sequence fp32 reference written from vllm-project/speculators (models/dflash + dspark).
#   Both   : the per-session ring (incremental append, window trimming) vs one-shot context.
# usage: python tools/glm_block_check.py --dflash2 ~/models/glm-5_3-flash-dflash2 --dspark ~/models/glm-5_3-flash-speculator_dspark --zlab /path/to/data [--gpu]
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
import mlx.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--dflash2")
ap.add_argument("--dspark")
ap.add_argument("--zlab")
ap.add_argument("--n", type=int, default=45, help="context rows")
ap.add_argument("--window", type=int, default=32, help="sliding window used by both sides (small: exercises the window)")
ap.add_argument("--gpu", action="store_true", help="run on the GPU (only when the parent says the GPU is free)")
a = ap.parse_args()
if not a.gpu:
    mx.set_default_device(mx.cpu)

from kyojin_mlx import glm_block as GB

mx.random.seed(7)
V, H = 154880, 4096
rep = {}


def cmp(name, x, y, tol):
    x, y = mx.array(x).astype(mx.float32), mx.array(y).astype(mx.float32)
    err = float(mx.abs(x - y).max())
    rel = err / (float(mx.abs(y).max()) + 1e-9)
    ok = rel <= tol
    rep[name] = (ok, rel)
    print(f"  {name:34s} max|d| {err:.3e}  rel {rel:.2e}  {'ok' if ok else 'FAIL'}")


def mk_taps(T, n):
    return (mx.random.normal((n, T * H)) * 2.0).astype(mx.bfloat16)


# ------------------------------------------------------------------------------------------------ DFlash2
if a.dflash2:
    print("DFlash2")
    p = Path(a.dflash2).expanduser()
    d = json.loads((p / "config.json").read_text())
    d["sliding_window"] = a.window
    cfg = GB.Cfg(d, "dflash2")
    w = GB._load(p)
    emb = (mx.random.normal((V, H)) * 0.02).astype(mx.bfloat16)
    head_w = (mx.random.normal((V, H)) * 0.05).astype(mx.bfloat16)
    mine = GB.DFlash2(p, cfg, w, 0, lambda ids: emb[ids], lambda h: h @ head_w.T)
    T, n = len(cfg.tap_ids), a.n
    taps = mk_taps(T, n)
    last, p0 = 12345, n
    st = GB.BlockDraft(cfg.window)
    st.append(taps, None, 0)
    t0 = time.perf_counter()
    toks = mine.draft(st, last, p0, 7)
    mx.eval(toks)
    print(f"  draft {time.perf_counter() - t0:.2f}s tokens {toks.tolist()}")
    hid_mine = mine.block_hidden(st, last, p0)
    # incremental ring == one-shot ring
    st2 = GB.BlockDraft(cfg.window)
    for s in range(0, n, 7):
        st2.append(taps[s:s + 7], None, s)
        if s % 14 == 0:
            mine.feed(st2)
    mine.feed(st2)
    cmp("ring incremental vs one-shot (K l0)", st2.ks[0], st.ks[0], 1e-3)
    cmp("ring incremental vs one-shot (V l4)", st2.vs[4], st.vs[4], 1e-3)
    if a.zlab:
        sys.path.insert(0, a.zlab)
        import importlib
        ZM = importlib.import_module("dflash.model_mlx")
        zc = ZM.DFlashConfig(
            hidden_size=d["hidden_size"], num_hidden_layers=d["num_hidden_layers"], num_attention_heads=d["num_attention_heads"], num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"], intermediate_size=d["intermediate_size"], vocab_size=d["vocab_size"], rms_norm_eps=d["rms_norm_eps"], rope_theta=10000.0,
            max_position_embeddings=d["max_position_embeddings"], block_size=8, target_layer_ids=tuple(cfg.tap_ids), num_target_layers=45, mask_token_id=cfg.mask_id,
            layer_types=tuple(d["layer_types"]), sliding_window=a.window, conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=16, is_causal=d.get("is_causal"))
        z = ZM.DFlash2DraftModel(zc)
        ww = dict(w)
        for nme in ("predecessor_codebook", "successor_codebook"):
            ww[f"candidate_selector.{nme}.weight"] = ww.pop(f"candidate_selector.{nme}")
        z.load_weights(list(ww.items()))
        z.embed_tokens = SimpleNamespace(**{})
        z.embed_tokens = lambda ids: emb[ids]
        z.lm_head = lambda h: h @ head_w.T
        z.embed_scale = 1.0
        ids = mx.array([[last] + [cfg.mask_id] * 7])
        cache = z.make_cache()
        hid_ref = z.hidden_states(ids, taps[None], cache, 0)[0]
        cmp("block hidden vs z-lab", hid_mine, hid_ref, 3e-2)
        cache = z.make_cache()
        path, cand, _ = z.propose(ids, taps[None], cache, 0.0, logits_start=1)
        mx.eval(path)
        print(f"  z-lab tokens {path[0].tolist()}")
        same = toks.tolist() == path[0].tolist()
        print(f"  selected path identical: {same}")
        rep["path identical"] = (same, 0.0)

# ------------------------------------------------------------------------------------------------ DSpark


def naive_dspark(w, cfg, taps, last, p0, window, markov=True):
    """Full-sequence fp32 reference from the speculators training code: kv = [ctx rows ; block rows], one big mask, no ring."""
    f = lambda k: w[k].astype(mx.float32)
    n = taps.shape[0]
    H_ = cfg.hidden

    def rms(x, wt):
        return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + cfg.eps) * wt

    def rope(x, pos):                                       # x [h, L, D]  rotate_half
        D = x.shape[-1]
        inv = 1.0 / (cfg.theta ** (mx.arange(0, D, 2, dtype=mx.float32) / D))
        ang = pos[:, None].astype(mx.float32) * inv[None]
        cos, sin = mx.concatenate([mx.cos(ang)] * 2, -1), mx.concatenate([mx.sin(ang)] * 2, -1)
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return x * cos + mx.concatenate([-x2, x1], -1) * sin

    ctx = rms(taps.astype(mx.float32) @ f("fc.weight").T, f("hidden_norm.weight"))
    ids = [last] + [cfg.mask_id] * (cfg.block - 1)
    x = f("embed_tokens.weight")[mx.array(ids)]
    B = cfg.block
    qpos = p0 + mx.arange(B)
    kpos = mx.concatenate([mx.arange(n), p0 + mx.arange(B)])
    isblk = mx.concatenate([mx.zeros(n), mx.ones(B)]) > 0
    vis = ((~isblk)[None, :] & (kpos[None, :] < p0) & (kpos[None, :] > qpos[:, None] - window))
    blk = isblk[None, :] & ((kpos[None, :] <= qpos[:, None]) if cfg.causal else True)
    mask = vis | blk
    for i in range(cfg.layers):
        pf = f"layers.{i}."
        hn = rms(x, f(pf + "input_layernorm.weight"))
        q = (hn @ f(pf + "self_attn.q_proj.weight").T).reshape(B, cfg.heads, cfg.hd)
        kc, vc = ctx @ f(pf + "self_attn.k_proj.weight").T, ctx @ f(pf + "self_attn.v_proj.weight").T
        kn, vn = hn @ f(pf + "self_attn.k_proj.weight").T, hn @ f(pf + "self_attn.v_proj.weight").T
        k = mx.concatenate([kc, kn]).reshape(n + B, cfg.kv, cfg.hd)
        v = mx.concatenate([vc, vn]).reshape(n + B, cfg.kv, cfg.hd)
        q = rms(q, f(pf + "self_attn.q_norm.weight")).transpose(1, 0, 2)
        k = rms(k, f(pf + "self_attn.k_norm.weight")).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        q, k = rope(q, qpos), rope(k, kpos)
        sc = (q @ k.transpose(0, 2, 1)) * cfg.hd ** -0.5
        sc = mx.where(mask[None], sc, -1e30)
        o = (mx.softmax(sc, -1) @ v).transpose(1, 0, 2).reshape(B, -1)
        x = x + o @ f(pf + "self_attn.o_proj.weight").T
        h2 = rms(x, f(pf + "post_attention_layernorm.weight"))
        x = x + (nn.silu(h2 @ f(pf + "mlp.gate_proj.weight").T) * (h2 @ f(pf + "mlp.up_proj.weight").T)) @ f(pf + "mlp.down_proj.weight").T
    hid = rms(x, f("norm.weight"))
    logits = hid @ f("lm_head.weight").T
    prev, toks, conf = last, [], []
    for k in range(B):
        e = f("markov_head.markov_w1.weight")[prev]
        sc = logits[k] + f("markov_head.markov_w2.weight") @ e
        prev = int(mx.argmax(sc))
        toks.append(prev)
        conf.append(float(mx.sigmoid((f("confidence_head.proj.weight")[0] * mx.concatenate([hid[k], e])).sum() + f("confidence_head.proj.bias")[0])))
    cum, acc = [], 1.0
    for c in conf:
        acc *= c
        cum.append(acc)
    return hid, toks, cum


if a.dspark:
    print("DSpark")
    p = Path(a.dspark).expanduser()
    d = json.loads((p / "config.json").read_text())
    d["transformer_layer_config"]["sliding_window"] = a.window
    cfg = GB.Cfg(d, "dspark")
    w = GB._load(p)
    mine = GB.DSpark(p, cfg, w, 0)
    T, n = len(cfg.tap_ids), a.n
    taps = mk_taps(T, n)
    last, p0 = 777, n
    st = GB.BlockDraft(cfg.window)
    st.append(taps, None, 0)
    t0 = time.perf_counter()
    toks = mine.draft(st, last, p0, 8)
    mx.eval(toks, st.conf)
    print(f"  draft {time.perf_counter() - t0:.2f}s tokens {toks.tolist()} cum conf {[round(x, 3) for x in st.conf.tolist()]}")
    hid_mine = mine.block_hidden(st, last, p0)
    hid_ref, tref, cref = naive_dspark(w, cfg, taps, last, p0, a.window)
    cmp("block hidden vs naive fp32", hid_mine, hid_ref, 5e-2)
    cmp("cum confidence vs naive", mx.array(st.conf.tolist()), mx.array(cref), 5e-2)
    print(f"  naive tokens {tref}")
    same = toks.tolist() == tref
    print(f"  chain tokens identical: {same}")
    rep["dspark tokens identical"] = (same, 0.0)
    st2 = GB.BlockDraft(cfg.window)
    for s in range(0, n, 9):
        st2.append(taps[s:s + 9], None, s)
        if s % 18 == 0:
            mine.feed(st2)
    mine.feed(st2)
    cmp("ring incremental vs one-shot (K l0)", st2.ks[0], st.ks[0], 1e-3)

bad = [k for k, (ok, _) in rep.items() if not ok]
print("RESULT", "FAIL " + ", ".join(bad) if bad else "all ok")
sys.exit(1 if bad else 0)
