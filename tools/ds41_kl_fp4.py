#!/usr/bin/env python3
"""DS4.1 KL protocol against the release fp4 checkpoint (the TensorFold bar: 4 prompts x 64 tokens teacher-forced, KL vs release fp4 0.138, top-1 0.909).
  gen   PACK --tok DIR --out SEQ.json          the 4 bar prompts (tools/ds41_bar.py PROMPTS, DS4.1 chat template, thinking off) + --new (64) greedy
                                              tokens of PACK: the fixed token sequences both sides are teacher-forced on
  ref   RELEASE --seq SEQ.json --out REF.npz  the release checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017, fp4 experts / fp8 rest, as shipped), run
                                              LAYER BY LAYER: load one layer shard, run every sequence through it, free it, next layer (peak RAM ~ one
                                              layer, not the 475 GiB checkpoint). mlx-vlm deepseek_v41 numerics (fp8/fp4 activation fake-quant); the
                                              routed experts stay mxfp4 (the source bytes, gather_qmm), the rest is dequantized to bf16 exactly as
                                              mlx-vlm's sanitize does. Writes the full-vocab log-probs at the 64 continuation positions (fp32).
                                              --layers N: the first N layers only (plumbing check on a slice).
  score PACK --seq SEQ.json --ref REF.npz     our pack teacher-forced on the same sequences (one prefill pass each, the patched model): per position
                                              KL(ref || ours) over the full vocabulary, top-1 agreement (argmax ref == argmax ours); mean over 4 x 64.
                                              --dump OUT.npz writes our log-probs in the REF format (score against it = 0: the self check).
A slice pack with a cut vocabulary folds the prompt ids into it (as tools/ds41_bar.py): numbers there are plumbing.
usage: ds41_kl_fp4.py {gen,ref,score} MODEL [--tok DIR] [--seq F] [--out F] [--ref F] [--new 64] [--layers N] [--json F] [--dump F]"""
import argparse, gc, json, os, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

THEIRS = dict(kl=0.138, top1=0.909)
PROMPTS = {                                   # = tools/ds41_bar.py PROMPTS (TensorFold tools/zig/xpu_draft_receipt.py prompts + our structured one)
    "prose": "Describe a quiet harbour town at dawn in about two hundred words.",
    "code": "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and tests.",
    "structured": "Return a JSON object describing three fictional employees with the fields name, age, department, skills (a list) and salary.",
    "list": "List the planets of the solar system in order, one per line, with one fact about each.",
}
GiB = 2 ** 30

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=("gen", "ref", "score"))
ap.add_argument("model")
ap.add_argument("--tok")
ap.add_argument("--seq")
ap.add_argument("--out")
ap.add_argument("--ref")
ap.add_argument("--new", type=int, default=64)
ap.add_argument("--layers", type=int, default=0)
ap.add_argument("--fetch", help="ref: HF repo (deepseek-ai/DeepSeek-V4.1-Flash-1017); a missing shard is fetched just before its layer and deleted after it")
ap.add_argument("--engram-dir", help="ref: a dir that already holds the Engram shards model-00047/00048 (the pack dir ~/ds41/model): symlinked, not fetched")
ap.add_argument("--json")
ap.add_argument("--dump")
a = ap.parse_args()


def load_pack(path):
    """Our pack, patched exactly as tools/check_ds41_ref.py loads it (Engram tables through the parallel gather)."""
    from kyojin_mlx import patch as P
    model, spec, cfg = P.build_model(path)
    tables = None
    try:
        from kyojin_mlx import engram_rows as ER
        es = ER.EngramSpec.from_config(cfg)
        if es is not None:
            tables = ER.EngramTables(path, es)
            bad = tables.validate()
            assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
    except ImportError:
        pass
    P.patch_model(model, path, spec, cfg)
    if tables is not None:
        from kyojin_mlx import engram_mlx
        engram_mlx.install_parallel_gather(model, tables)
    return model.language_model, cfg.get("text_config", cfg)["vocab_size"]


def pack_logits(lm, ids):
    out = lm(mx.array([ids]), cache=lm.make_cache())
    return (out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32)


def logp_at(lg, lo, hi):
    x = lg[lo:hi]
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def load_seq():
    s = json.load(open(a.seq))
    return s, s["seqs"]


def seq_key(seqs):
    """Hash of every prompt and continuation id: a REF.npz is only valid for the exact sequences it was computed on."""
    import hashlib
    return hashlib.sha256(json.dumps([[x["name"], x["prompt_ids"], x["cont_ids"]] for x in seqs]).encode()).hexdigest()[:16]


if a.mode == "gen":
    from transformers import PreTrainedTokenizerFast
    from kyojin_mlx.serve.ds41_chat import render_chat
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.tok or a.model, "tokenizer.json"))
    lm, vocab = load_pack(a.model)
    folded = vocab < tok.vocab_size
    seqs = []
    for name, text in PROMPTS.items():
        ids = tok.encode(render_chat([{"role": "user", "content": text}], thinking=False), add_special_tokens=False)
        ids = [t % (vocab - 16) + 16 if t >= vocab else t for t in ids] if folded else ids
        cache, out = lm.make_cache(), []
        lg = lm(mx.array([ids]), cache=cache)
        for _ in range(a.new):                                                  # greedy, EOS ignored: always --new tokens
            lg = (lg.logits if hasattr(lg, "logits") else lg)[:, -1]
            out.append(int(mx.argmax(lg, axis=-1).item()))
            lg = lm(mx.array([[out[-1]]]), cache=cache)
        seqs.append(dict(name=name, prompt_ids=ids, cont_ids=out))
        print(f"gen {name}: prompt {len(ids)} + {len(out)} tok  {tok.decode(out)[:60]!r}", flush=True)
    json.dump(dict(pack=a.model, new=a.new, vocab=vocab, folded=folded, seqs=seqs), open(a.out, "w"))
    print("KLGEN OK", a.out)

elif a.mode == "score":
    meta, seqs = load_seq()
    t0 = time.time()
    lm, vocab = load_pack(a.model)
    ref = np.load(a.ref) if a.ref else None
    if ref is not None:
        rm = json.loads(str(ref["__meta__"])) if "__meta__" in ref else {}
        assert rm.get("seq") == seq_key(seqs), f"{a.ref} was computed on other sequences than {a.seq} (rerun ref, or gen into a new file)"
        assert rm.get("full", True), f"{a.ref} is a --layers {rm.get('layers')} plumbing reference, not the full model"
    kls, top1, dump = [], [], {}
    for s in seqs:
        full = s["prompt_ids"] + s["cont_ids"]
        lo, hi = len(s["prompt_ids"]) - 1, len(full) - 1                        # the positions that predict the continuation tokens
        mine = logp_at(pack_logits(lm, full), lo, hi)
        mx.eval(mine)
        dump[s["name"]] = np.array(mine, dtype=np.float32)
        if ref is not None:
            r = mx.array(ref[s["name"]])
            assert r.shape == mine.shape, f"{s['name']}: ref {r.shape} vs ours {mine.shape} (same SEQ.json and vocabulary?)"
            kl = mx.sum(mx.exp(r) * (r - mine), axis=-1)                         # KL(ref || ours) per position, full vocabulary
            t1 = mx.argmax(r, axis=-1) == mx.argmax(mine, axis=-1)
            kls.append(np.array(kl)); top1.append(np.array(t1))
            print(f"score {s['name']:10s}: KL {float(kl.mean()):.4f}  top-1 {float(t1.mean()):.3f}  ({kl.size} positions)", flush=True)
    if a.dump:
        np.savez(a.dump, __meta__=json.dumps(dict(seq=seq_key(seqs), full=True, src=a.model)), **dump)
    res = dict(pack=a.model, seq=a.seq, ref=a.ref, s=round(time.time() - t0, 1), peak_gib=round(mx.get_peak_memory() / GiB, 2), theirs=THEIRS)
    if kls:
        k, t = np.concatenate(kls), np.concatenate(top1)
        res.update(kl=round(float(k.mean()), 4), top1=round(float(t.mean()), 4), positions=int(k.size),
                   per_prompt={s["name"]: dict(kl=round(float(x.mean()), 4), top1=round(float(y.mean()), 4)) for s, x, y in zip(seqs, kls, top1)})
        print(f"KLFP4 kl {res['kl']} top1 {res['top1']} over {res['positions']} positions   theirs kl {THEIRS['kl']} top1 {THEIRS['top1']}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)

else:                                                                           # ref: the release checkpoint, layer streamed
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from mlx_vlm.models.deepseek_v41.config import ModelConfig
    from mlx_vlm.models.deepseek_v41.deepseek_v41 import Model, _pack_source_weight
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41Block, make_identity_pre_mix

    meta, seqs = load_seq()
    t0 = time.time()
    rel = os.path.abspath(a.model)
    os.makedirs(rel, exist_ok=True)
    fetched = set()

    def ensure(f):
        """The path of source file f in RELEASE: present, symlinked from --engram-dir, or fetched from --fetch (then deleted after use)."""
        p = os.path.join(rel, f)
        if os.path.exists(p):
            return p
        if a.engram_dir and os.path.exists(os.path.join(a.engram_dir, f)):
            os.symlink(os.path.join(os.path.abspath(a.engram_dir), f), p)
            return p
        assert a.fetch, f"{p} missing (give --fetch REPO)"
        from huggingface_hub import hf_hub_download
        t = time.time()
        hf_hub_download(a.fetch, f, local_dir=rel)
        if f.endswith(".safetensors"):
            fetched.add(f)
            print(f"  fetched {f}: {os.path.getsize(p) / GiB:.2f} GiB in {time.time() - t:.0f} s", flush=True)
        return p

    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        ensure(f)
    cfgd = json.load(open(os.path.join(rel, "config.json")))
    if a.layers:
        cfgd.get("text_config", cfgd)["num_hidden_layers"] = a.layers
    cfgd["model_path"] = rel
    config = ModelConfig.from_dict(cfgd)
    model = Model(config)                                                       # lazy parameters: nothing is materialized but what we load
    lm = model.language_model
    wmap = json.load(open(os.path.join(rel, "model.safetensors.index.json")))["weight_map"]
    shards = {}
    for k, f in wmap.items():
        shards.setdefault(f, []).append(k)

    def raw(prefix):
        """Source tensors whose key starts with prefix, from the shards that hold them (lazy until evaluated)."""
        out = {}
        for f in sorted({f for k, f in wmap.items() if k.startswith(prefix)}):
            w = mx.load(ensure(f))
            out.update({k: v for k, v in w.items() if k.startswith(prefix)})
        return out

    def install(prefix):
        w = raw(prefix)
        n = 0
        if isinstance(prefix, str) and prefix.startswith("layers."):
            i = int(prefix.split(".")[1])
            ffn = lm.layers[i].ffn
            for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                ws, ss = [], []
                for e in range(config.n_routed_experts):
                    pw, ps, mode = _pack_source_weight(w.pop(f"{prefix}ffn.experts.{e}.{src}.weight"), w.pop(f"{prefix}ffn.experts.{e}.{src}.scale"))
                    assert mode == "mxfp4", f"routed expert {e}.{src}: {mode}"
                    ws.append(pw); ss.append(ps)
                old = getattr(ffn.switch_mlp, dst)
                q = QuantizedSwitchLinear(old.input_dims, old.output_dims, config.n_routed_experts, bias=False, group_size=32, bits=4, mode="mxfp4")
                q.weight, q.scales = mx.stack(ws), mx.stack(ss)
                setattr(ffn.switch_mlp, dst, q)
                mx.eval(q.weight, q.scales)
                n += 2 * config.n_routed_experts
        n += len(w)
        sw = model.sanitize(w)                                                  # the rest: mlx-vlm's own key map and exact bf16 dequantization
        model.load_weights(list(sw.items()), strict=False)
        mx.eval([v for k, v in sw.items() if ".engram.embed." not in k])
        del w, sw
        for f in [f for f in fetched if not any(wmap[k] == f for k in wmap if ".engram." in k)]:    # a fetched trunk shard goes once loaded
            os.remove(os.path.join(rel, f)); fetched.discard(f)
        return n

    install("embed."); install(("norm.", "head."))
    hs, entries, masks, hashes = [], [], [], []
    for s in seqs:
        ids = mx.array([s["prompt_ids"] + s["cont_ids"]])
        entry = lm.make_cache()[0]
        image_mask = ids == config.image_token_id
        h = lm.embed_tokens(ids)
        h = mx.broadcast_to(h[..., None, :], (*h.shape[:2], config.hc_mult, config.hidden_size))
        lm._ensure_engram_hash()
        eh = None
        if lm.engram_hash is not None:
            eh = entry.map(lambda row, x, m: lm.engram_hash(x, row.offset, row, token_mask=m), ids, ~image_mask)
        pm = make_identity_pre_mix(*h.shape[:2], config.hc_mult)
        mx.eval(h, pm, *([eh] if eh is not None else []))
        hs.append([h, pm]); entries.append(entry); masks.append(image_mask); hashes.append(eh)
    print(f"ref: {len(seqs)} sequences, {config.num_hidden_layers} layers, engram {lm.engram_hash is not None}, setup {time.time() - t0:.1f} s", flush=True)
    for i in range(config.num_hidden_layers):
        t1 = time.time()
        n = install(f"layers.{i}.")
        layer = lm.layers[i]
        for j in range(len(seqs)):
            h, pm = hs[j]
            if layer.engram is not None and hashes[j] is not None:
                h = layer.engram(h, hashes[j][:, :, layer.engram.layer_hash_index, :], ~masks[j])
            h, pm = layer(h, pm, masks[j], entries[j])
            mx.eval(h, pm)
            hs[j] = [h, pm]
        lm.layers[i] = nn.Module()                                              # free the layer: weights go, the cache entries keep its state
        del layer
        gc.collect(); mx.clear_cache()
        print(f"  layer {i:2d}: {n} tensors, {time.time() - t1:.1f} s, peak {mx.get_peak_memory() / GiB:.1f} GiB", flush=True)
    out = {}
    for s, (h, pm) in zip(seqs, hs):
        lo, hi = len(s["prompt_ids"]) - 1, len(s["prompt_ids"]) + len(s["cont_ids"]) - 1
        lg = lm.head(lm.norm(DeepseekV41Block.hc_pre(h[:, lo:hi], pm[:, lo:hi])))[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        out[s["name"]] = np.array(lp, dtype=np.float32)
    meta_out = dict(seq=seq_key(seqs), layers=config.num_hidden_layers, full=not a.layers, src=rel)
    np.savez(a.out, __meta__=json.dumps(meta_out), **out)
    print(f"KLREF OK {a.out}: {sum(v.shape[0] for v in out.values())} positions x {config.vocab_size}, {time.time() - t0:.0f} s, peak {mx.get_peak_memory() / GiB:.1f} GiB")
