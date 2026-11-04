"""KVWIN measurement on a Qwen3.5 mlx-lm model: KL / top-1 of a quantized KV cache (+ fp16 sink/window overlay) against the fp16 cache,
and decode tok/s. Every scored position is decode-faithful (WinKVCache builds the window per row, whatever the chunk size).

  kvwin_kl.py kl    MODEL N EVAL CONFIGS_JSON OUT_JSONL [TEXT_OFFSET]   # logits of the last EVAL positions of an N-token context
  kvwin_kl.py bench MODEL N STEPS CONFIGS_JSON OUT_JSONL [REPEATS]      # prefill N tokens, then STEPS greedy decode steps, interleaved A/B
CONFIGS_JSON: list of [kbits, vbits, group, sink, window]; [0,0,0,0,0] = fp16 KVCache."""
import json, os, sys, time
import mlx.core as mx
import pydoc_data.topics as T
from mlx_lm import load
from kyojin_mlx import kvwin as KW

CHUNK = 256
mx.set_cache_limit(1 << 30)                                   # keep the footprint (top) close to the live arrays: Air budget 13 GiB


def tokens(tok, n, off):
    path = os.environ.get("KVWIN_TEXT")                       # a UTF-8 file with >= off + n tokens of natural text (128K points on the Ultra)
    txt = open(path, encoding="utf-8", errors="ignore").read() if path else "\n\n".join(T.topics[k] for k in sorted(T.topics))
    ids = tok.encode(txt)
    if len(ids) < off + n:
        raise SystemExit(f"text has {len(ids)} tokens, need {off + n}: set KVWIN_TEXT=<bigger file>")
    return ids[off:off + n]


def cache_for(model, c):
    if c[0] and c[2] == 0:                                    # group 0 = TurboQuant pool (kyojin_mlx/tq.py): [kbits, vbits, 0, sink, window]
        from kyojin_mlx import tq
        KW.install()
        return [tq.TQKVCache(c[0], c[1], c[3], c[4]) if type(x) is KW.KVCache else x for x in KW._mlx_make_prompt_cache(model)]
    return KW.make_prompt_cache(model, None if c[0] == 0 else dict(kbits=c[0], vbits=c[1], group_size=c[2], sink=c[3], window=c[4], f32=len(c) < 6 or bool(c[5]))) \
        if c[0] else KW._mlx_make_prompt_cache(model)


def head(lm):
    return (lambda h: lm.model.embed_tokens.as_linear(h)) if lm.args.tie_word_embeddings else lm.lm_head


def run_logits(model, ids, cache, n_eval, CHUNK=CHUNK):
    lm = model.language_model
    hd = head(lm)
    out = []
    N = len(ids)
    for s in range(0, N, CHUNK):
        x = mx.array(ids[s:s + CHUNK])[None]
        h = lm.model(x, cache)
        e = max(s + x.shape[1] - (N - n_eval), 0)             # eval positions inside this chunk
        if e:
            lg = hd(h[:, -e:])[0]
            mx.eval(lg)
            out.append(lg)
        else:
            mx.eval(h)
        mx.clear_cache()
    return mx.concatenate(out, axis=0)


def kl_top1(ref, got, step=128):
    """Mean KL(ref || got), top-1 agreement and p99 KL over the rows; fp32 math on `step` rows at a time (keeps the peak small)."""
    kls, tops = [], []
    for i in range(0, ref.shape[0], step):
        r = ref[i:i + step].astype(mx.float32); g = got[i:i + step].astype(mx.float32)
        lr = r - mx.logsumexp(r, axis=-1, keepdims=True); lg = g - mx.logsumexp(g, axis=-1, keepdims=True)
        kl = mx.sum(mx.exp(lr) * (lr - lg), axis=-1)
        top = (mx.argmax(r, -1) == mx.argmax(g, -1)).astype(mx.float32)
        mx.eval(kl, top); kls += kl.tolist(); tops += top.tolist()
        mx.clear_cache()
    k = sorted(kls)
    return sum(kls) / len(kls), sum(tops) / len(tops), k[int(.99 * (len(k) - 1))]


def name(c):
    return "fp16" if not c[0] else (f"tq{c[0]}v{c[1]}" if c[2] == 0 else f"k{c[0]}v{c[1]}g{c[2]}") + (f"+s{c[3]}" if c[3] else "") + (f"+w{c[4]}" if c[4] else "") + ("/h" if len(c) > 5 and not c[5] else "")


def main():
    mode, mdir = sys.argv[1], sys.argv[2]
    model, tok = load(mdir)
    if mode == "kl":
        N, E = int(sys.argv[3]), int(sys.argv[4]); cfgs = json.load(open(sys.argv[5])); out = open(sys.argv[6], "a")
        off = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        ids = tokens(tok, N, off)
        t0 = time.time(); ref = run_logits(model, ids, cache_for(model, [0] * 5), E); mx.eval(ref)
        print(f"ref N={N} {time.time()-t0:.1f}s peak {mx.get_peak_memory()/2**30:.2f} GiB", flush=True)
        for c in cfgs:
            t0 = time.time(); cache = cache_for(model, c)
            got = run_logits(model, ids, cache, E)
            kl, top, p99 = kl_top1(ref, got)
            attn = [x for x in cache if hasattr(x, "nbytes") and x.__class__.__name__ in ("KVCache", "WinKVCache", "TQKVCache")]
            bpt = sum(x.nbytes for x in attn) / (N * len(attn)) / 2 if attn else 0          # bytes per token per layer per kv head (2 heads)
            rec = dict(cfg=name(c), c=c, N=N, E=E, off=off, kl=kl, top1=top, kl_p99=p99, bytes_tok_layer=bpt * 2, sec=time.time() - t0,
                       peak_gib=mx.get_peak_memory() / 2**30)
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
            del cache, got; mx.clear_cache()
    else:
        N, steps = int(sys.argv[3]), int(sys.argv[4]); cfgs = json.load(open(sys.argv[5])); out = open(sys.argv[6], "a")
        reps = int(sys.argv[7]) if len(sys.argv) > 7 else 3
        ids = tokens(tok, N, 0); lm = model.language_model; hd = head(lm)
        res = {name(c): [] for c in cfgs}
        for r in range(reps):
            for c in cfgs:
                cache = cache_for(model, c)
                for s in range(0, N, 512):
                    h = lm.model(mx.array(ids[s:s + 512])[None], cache); mx.eval(h)
                y = mx.argmax(hd(h[:, -1:])[0], axis=-1)
                ts = []
                for i in range(steps):
                    t0 = time.perf_counter(); h = lm.model(y[None], cache); y = mx.argmax(hd(h)[0], axis=-1); mx.eval(y); ts.append(time.perf_counter() - t0)
                ts = sorted(ts)[:max(len(ts) // 2, 1)]                  # fastest half: warm steps
                res[name(c)].append(len(ts) / sum(ts))
                del cache; mx.clear_cache()
        for c in cfgs:
            v = sorted(res[name(c)]); rec = dict(cfg=name(c), c=c, N=N, tok_s_max=v[-1], tok_s_p25=v[len(v) // 4], reps=reps, peak_gib=mx.get_peak_memory() / 2**30)
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()


if __name__ == "__main__":
    main()
