"""DS4.1 cache quantization SIMULATION on a slice (ds2 / ds2e, real DS4.1 dims). Opt-in tool, touches no engine code: hooks are installed on the
mlx-vlm module in this process only.

Per-token cache of a DS4.1 compression-source layer (one entry per `ratio` tokens): `cache.compress` = RoPE'd compressed KV pool [D=512], already
fake-quantized to FP4 e2m1 / e4m3 scales / blocks of 16 by the model; `cache.keys` = indexer keys [128], already FP4 e2m1 / ue8m0 / blocks of 32.
Fixed size, not per token: `cache.window` ring [128, 512] (FP8 e4m3, ue8m0, blocks of 32), compressor state.

Stored tensors stay fp16 (hp); a "view" = quantize->dequantize of the whole table, applied where the cache is READ (decode-faithful: query q sees
the entries [0, sink) and [n_q - win, n_q) at full precision, n_q = number of compressed entries visible to q; the others through the view).
  kinds:  ["fp4"]                      native format (idempotent on the stored values -> must be bit-exact)
          ["aff", bits, group, sink, win]   mx.quantize affine, bits 4/5/6/8
  cfg = {"name":..., "pool": spec|null, "idx": spec|null}      spec null = fp16 as stored.

  ds41_cache_sim.py PACK N CFG_JSON OUT_JSONL [E=256]    (PACK = ds2 / ds2e dir; random token ids, prefill chunks of 512, last E rows scored)
Per cfg: top-k overlap (layer-0 indexer, last E rows, mean |A & B| / k), attention-output rel L2 (layers 0 / 1, last E rows), logits rel L2
(layer_gate metric), KL / top-1 of the logits vs the fp16-cache run."""
import json, sys, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import patch as P
import mlx_vlm.models.deepseek_v41.language as L
from mlx_vlm.models.deepseek_v41 import fakequant as FQ

CH = 512
SIM = dict(pool=None, idx=None, clens=None, rec=None, calls=dict(pool=0, idx=0))


def view(x, spec, kind):
    if spec[0] == "fp4":
        return FQ.fake_quant_fp4_e4m3(x) if kind == "pool" else FQ.fake_quant_fp4_ue8m0(x)
    _, bits, gs = spec[:3]
    w, s, b = mx.quantize(x, group_size=gs, bits=bits)
    return mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(x.dtype)


def hp_mask(T, clens, spec):
    """[S, T] bool: entries that stay at full precision for each query (sink + recent window)."""
    if spec[0] == "fp4":
        return mx.zeros((clens.shape[0], T), dtype=mx.bool_)
    sink, win = spec[3], spec[4]
    j = mx.arange(T)[None, :]
    return (j < sink) | (j >= clens - win)


_isc = L._index_scores
_spa = L._sparse_pooled_attention
_att = L.DeepseekV41Attention._attention
_idx_call = L.Indexer.__call__


def isc(q, index_k, weights, compress_lens=None, candidates=None):
    s = _isc(q, index_k, weights, compress_lens, candidates)
    sp = SIM["idx"]
    if sp is None:
        return s
    SIM["calls"]["idx"] += 1
    sq = _isc(q, view(index_k, sp, "idx"), weights, compress_lens, candidates)
    return mx.where(hp_mask(index_k.shape[1], compress_lens, sp), s, sq)


def spa(q, window, pool, indices, window_mask, pooled_mask, scale, sinks):
    sp = SIM["pool"]
    if sp is None:
        return _spa(q, window, pool, indices, window_mask, pooled_mask, scale, sinks)
    SIM["calls"]["pool"] += 1
    T = pool.shape[1]
    cat = mx.concatenate([pool, view(pool, sp, "pool")], axis=1)
    lq = ~hp_mask(T, SIM["clens"], sp)[None, :, :]                       # [1, S, T] -> per (query, entry); gather the flag with the index
    flag = mx.take_along_axis(mx.broadcast_to(lq, (indices.shape[0],) + lq.shape[1:]), mx.maximum(indices, 0), axis=-1) & (indices >= 0)
    return _spa(q, window, cat, mx.where(flag, indices + T, indices), window_mask, pooled_mask, scale, sinks)


def att(self, cache, x):
    SIM["clens"] = (mx.arange(cache.offset + 1, cache.offset + x.shape[1] + 1) // self.compress_ratio)[:, None] if self.compress_ratio else None
    return _att(self, cache, x)


def idx_call(self, *a, **k):
    out = _idx_call(self, *a, **k)
    if SIM["rec"] is not None:
        SIM["rec"]["topk"] = out
    return out


L._index_scores, L._sparse_pooled_attention = isc, spa
L.DeepseekV41Attention._attention = att
L.Indexer.__call__ = idx_call


def run(model, ids, E):
    lm = model.language_model
    cache = lm.make_cache()
    rec = {}
    outs = {}
    N = len(ids)
    for s in range(0, N, CH):
        last = s + CH >= N
        SIM["rec"] = rec if last else None
        h = []
        if last:
            for i, layer in enumerate(lm.layers):
                orig = layer.attn.__class__

                def call(self, *a, _o=orig, _i=i, **k):
                    o = _o.__call__(self, *a, **k)
                    outs[_i] = o[0] if isinstance(o, tuple) else o
                    return o
                layer.attn.__class__ = type(orig.__name__, (orig,), {"__call__": call})
        lg = lm(mx.array(ids[s:s + CH])[None], cache=cache).logits
        if last:
            lg = lg[:, -E:]
            for i, layer in enumerate(lm.layers):
                layer.attn.__class__ = layer.attn.__class__.__mro__[1]
            res = dict(logits=np.array(lg[0].astype(mx.float32)), topk=np.array(rec["topk"][0, -E:]),
                       attn={i: np.array(o[0, -E:].astype(mx.float32)) for i, o in outs.items()})
        mx.eval(lg) if not last else None
        mx.clear_cache()
    return res


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / np.linalg.norm(b.astype(np.float64)))


def klt(ref, got):
    r = ref - ref.max(-1, keepdims=True); g = got - got.max(-1, keepdims=True)
    lr = r - np.log(np.exp(r).sum(-1, keepdims=True)); lg = g - np.log(np.exp(g).sum(-1, keepdims=True))
    return float((np.exp(lr) * (lr - lg)).sum(-1).mean()), float((ref.argmax(-1) == got.argmax(-1)).mean())


def overlap(a, b):
    o = []
    for x, y in zip(a, b):
        x, y = x[x >= 0], y[y >= 0]
        o.append(len(np.intersect1d(x, y)) / max(len(x), 1))
    return float(np.mean(o)), float(np.min(o))


def main():
    pack, N, cfgs, out = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3])), open(sys.argv[4], "a")
    E = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    V = cfg["text_config"]["vocab_size"]
    ids = np.random.default_rng(5).integers(16, V - 16, N).tolist()
    t0 = time.time(); SIM["pool"] = SIM["idx"] = None
    ref = run(model, ids, E)
    print(f"ref N={N} {time.time()-t0:.0f}s peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
    for c in cfgs:
        SIM["pool"], SIM["idx"] = c.get("pool"), c.get("idx")
        SIM["calls"] = dict(pool=0, idx=0)
        t0 = time.time(); got = run(model, ids, E)
        kl, top1 = klt(ref["logits"], got["logits"])
        ov, ovmin = overlap(ref["topk"], got["topk"])
        rec = dict(cfg=c["name"], N=N, E=E, topk_overlap=ov, topk_overlap_min=ovmin, attn_rel={i: rel(got["attn"][i], ref["attn"][i]) for i in ref["attn"]},
                   logits_rel=rel(got["logits"], ref["logits"]), kl=kl, top1=top1, bitexact=bool(np.array_equal(got["logits"], ref["logits"])),
                   hook_calls=SIM["calls"], sec=time.time() - t0, peak_gib=mx.get_peak_memory() / 2**30)
        rec["attn_rel"] = {str(k): v for k, v in rec["attn_rel"].items()}
        print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()


if __name__ == "__main__":
    main()
