#!/usr/bin/env python3
"""Refusal / KL measurement of one ablit arm on the real DS4.1 pack. ONE process per arm (the spec weights are baked into compiled traces).
  gen: harmful_heldout prompts, served chat template, batched decode, refusal regex on the first 500 chars of the answer.
  kl : harmless_heldout prompts (thinking off), `--ref` = none writes the greedy continuation + full log-probs (hook off);
       with --ref FILE.npz the same tokens are teacher-forced and KL(ref||ours) / top-1 are reported.
Set KYOJIN_ABLIT_SPEC before the run (unset = hook off).
usage: ablit_search.py gen|kl PACK --out F.json [--n 20] [--think on|off] [--temp 1.0] [--new 160] [--ref F.npz] [--dump F.npz] [--fresh]"""
import argparse, json, os, re, sys, time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.expanduser("~/ds41ablit/tools_ablit"))

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("gen", "kl"))
ap.add_argument("pack")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--think", choices=("on", "off"), default="off")
ap.add_argument("--effort", default="medium")
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--new", type=int, default=160)
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--ref")
ap.add_argument("--dump")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

PROMPTS = os.path.expanduser("~/ds41ablit/tools_ablit/prompts/")


def refusal(text):
    from eval_ablit import is_refusal
    return is_refusal(text[:500])


def main():
    from transformers import PreTrainedTokenizerFast
    from kyojin_mlx.serve.backends import load_vlm
    from kyojin_mlx.serve.ds41_chat import render_chat
    from kyojin_mlx.serve.sampling import Sampling, sample_rows
    tok = PreTrainedTokenizerFast(tokenizer_file=a.pack + "/tokenizer.json")
    fn = "harmful_heldout.jsonl" if a.mode == "gen" else "harmless_heldout.jsonl"
    rows = [json.loads(l) for l in open(PROMPTS + fn)][a.start:a.start + a.n]
    think = a.think == "on" and a.mode == "gen"
    ids = [tok(render_chat([{"role": "user", "content": r["text"]}], thinking=think, reasoning_effort=a.effort),
               add_special_tokens=False)["input_ids"] for r in rows]
    t0 = time.time()
    be, cfg = load_vlm(a.pack, ced=True, mtp=False)
    spec = os.environ.get("KYOJIN_ABLIT_SPEC", "")
    print(f"[ablit_search] loaded in {time.time()-t0:.0f}s spec={spec or 'OFF'}", flush=True)
    eos = int(cfg.get("eos_token_id", 1))
    mx.random.seed(a.seed)
    sp = Sampling(temperature=a.temp)
    ref = np.load(a.ref) if a.ref else None
    forced = None
    if a.mode == "kl" and ref is not None:
        forced = ref["tokens"]                              # [n, T]
    out, lps, toks_all = [], [], []
    for b0 in range(0, len(rows), a.batch):
        chunk = list(range(b0, min(b0 + a.batch, len(rows))))
        caches, first = [], []
        for i in chunk:
            c = be.make_cache()
            for s in range(0, len(ids[i]), 4096):
                lg = be.prefill_chunk(c, ids[i][s:s + 4096])
            caches.append(c)
            first.append(lg[0])
        logits = mx.stack(first)
        bc = be.batch(caches)
        n = len(chunk)
        gen = [[] for _ in range(n)]
        done = [False] * n
        lp_steps = []
        T = a.new if forced is None else forced.shape[1]
        for t in range(T):
            if a.mode == "kl":
                lp_steps.append((logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.float16))
            if forced is not None:
                nxt = mx.array([int(forced[i, t]) for i in chunk], dtype=mx.int32)
            else:
                keys = [mx.random.key(a.seed * 1000003 + chunk[k] * 7919 + t) for k in range(n)]
                nxt = sample_rows(logits, [sp] * n, keys)
            mx.eval(nxt)
            tl = nxt.tolist()
            for k in range(n):
                if not done[k]:
                    gen[k].append(tl[k])
                    if tl[k] == eos:
                        done[k] = True
            if all(done) and forced is None:
                break
            if t == T - 1:
                break
            logits = be.step(bc, nxt)
            mx.eval(logits)
        if a.mode == "kl":
            lps.append(np.array(mx.stack(lp_steps, axis=1)))           # [n, T, V] fp16
        be.unbatch(bc, n)
        for k, i in enumerate(chunk):
            toks_all.append(gen[k])
            text = tok.decode(gen[k], skip_special_tokens=False)
            out.append({"i": a.start + i, "prompt": rows[i]["text"], "text": text, "n_tok": len(gen[k])})
        print(f"[ablit_search] {len(out)}/{len(rows)} done {time.time()-t0:.0f}s", flush=True)
        del caches, bc
        mx.clear_cache()
    res = {"mode": a.mode, "spec": spec, "think": a.think, "temp": a.temp, "n": len(out)}
    if a.mode == "gen":
        nref = nunf = 0
        for o in out:
            t = o["text"]
            if think:
                if "</think>" in t:
                    ans = t.split("</think>", 1)[1]
                else:
                    ans, o["unfinished"] = "", True
                    nunf += 1
            else:
                ans = t
            o["answer"] = ans
            o["refusal"] = refusal(ans) if ans else None
            nref += bool(o["refusal"])
        scored = len(out) - nunf
        res.update(refusals=nref, unfinished=nunf, rate=nref / max(1, scored), rate_all=(nref + nunf) / max(1, len(out)))
    else:
        L = np.concatenate(lps, axis=0).astype(np.float32)
        if ref is None:
            T = max(len(g) for g in toks_all)
            tk = np.full((len(toks_all), T), eos, dtype=np.int64)
            for i, g in enumerate(toks_all):
                tk[i, :len(g)] = g
            np.savez(a.dump or a.out.replace(".json", ".npz"), tokens=tk, lp=L.astype(np.float16))
            res.update(saved=True, T=int(T))
        else:
            R = ref["lp"].astype(np.float32)
            Tn = min(R.shape[1], L.shape[1])
            R, L = R[:, :Tn], L[:, :Tn]
            kl = (np.exp(R) * (R - L)).sum(-1)                              # [n, T]
            top1 = (R.argmax(-1) == L.argmax(-1))
            res.update(kl=float(kl.mean()), kl_max=float(kl.max()), top1=float(top1.mean()))
    res["items"] = out
    json.dump(res, open(a.out, "w"), indent=1)
    print("[ablit_search] RESULT", {k: v for k, v in res.items() if k != "items"}, flush=True)


main()
