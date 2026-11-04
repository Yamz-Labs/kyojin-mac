#!/usr/bin/env python3
"""CED prefill quality gate (day 1 on the Ultra, or any slice): is the CED prefill (kyojin_mlx.ced, on by default in the server, --no-ced turns it off) good enough to
become a default? Same prompt through the exact prefill and through CED on the SAME chunk grid, per length:
  KL          KL(exact || CED) of the next-token distribution at the last row and the mean over the last 16 rows (nats)
  top-1       argmax agreement over those 16 rows (rows where the exact top-1 leads by < 0.25 nats are near ties, not counted)
  answer      greedy continuation of --new tokens from each cache (the decode path is exact in both): matching prefix length
  needle      a 5-digit code hidden at depth 10 / 50 / 90 % of the prompt, asked at the end: exact and CED must both print it
              (needs a tokenizer; SKIP otherwise)
  floor       KL between the exact prefill and the exact prefill on a different chunk grid = the rounding noise of this model; CED passes the KL
              test at max(--kl-max, 2 x floor)
and the prefill time (tokens/s) of both paths. Tails tried: --tails 128,exact ("exact" = receptive field D * (w - 1) + w, see kyojin_mlx/ced.py).
VERDICT PASS needs, for the first tail in --tails and every length: KL last row and mean <= bound, top-1 >= 15/16, answer prefix >= 0.75 of
--new, needle CED ok whenever exact ok. Exit 0 on PASS, 1 on FAIL, 2 when CED does not apply to the pack. CED is on by default (owner decision
24/10: vendor-designed, "negligible degradation"); this gate is the bug check of OUR implementation: a FAIL means a bug to fix, not a reason to switch CED off.

usage: ced_gate.py MODEL_DIR [--lengths 4096,16384,65536] [--chunk 2048] [--tails 128,exact] [--new 32] [--kl-max 0.02]
                   [--tokenizer DIR] [--text FILE ...] [--no-needle] [--no-floor] [--mid N] [--text-overrides JSON] [--dump OUT.json]
  --mid / --text-overrides: shallow slices (tests): the 2-layer ds2 slice runs as an encoder-less CED model with
     --mid 0 --text-overrides '{"compress_ratios":[1,1],"kv_source_layer_ids":[0],"index_source_layer_ids":[0],"candidate_source_layer_id":0}'
     (the machinery and the timings are real, the quality number says little: the decoder half is one layer deep)."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model")
ap.add_argument("--lengths", default="4096,16384,65536")
ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk = the serving grid (CED cuts every chunk; the result depends on it)")
ap.add_argument("--tails", default="128,exact")
ap.add_argument("--new", type=int, default=32)
ap.add_argument("--kl-max", type=float, default=0.02)
ap.add_argument("--tokenizer", help="directory with tokenizer.json (default: the model dir, then models/tok_ds41)")
ap.add_argument("--text", nargs="*", help="text files for the prompts (default: the repo's .md and kyojin_mlx sources)")
ap.add_argument("--no-needle", action="store_true")
ap.add_argument("--no-floor", action="store_true")
ap.add_argument("--mid", type=int)
ap.add_argument("--text-overrides", default="")
ap.add_argument("--dump")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-peak-gib", type=float, default=0.0, help="abort a length when the MLX peak memory passes this (0 = no limit)")
a = None

from kyojin_mlx import ced as C  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------- model
def load(model_dir, overrides):
    from kyojin_mlx import patch as P
    model, spec, cfg = P.build_model(model_dir, overrides or None)
    tables = None
    try:
        from kyojin_mlx import engram_rows as ER
        es = ER.EngramSpec.from_config(cfg)
        if es is not None:
            tables = ER.EngramTables(model_dir, es)
            bad = tables.validate()
            assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
    except ImportError:
        pass
    P.patch_model(model, model_dir, spec, cfg)
    if tables is not None:
        from kyojin_mlx import engram_mlx
        engram_mlx.install_parallel_gather(model, tables)
    return model, spec, cfg, tables


# ---------------------------------------------------------------------------------------------------------------- tokens
def find_tokenizer():
    for d in ([a.tokenizer] if a.tokenizer else []) + [a.model, os.path.join(REPO, "models", "tok_ds41")]:
        if d and os.path.exists(os.path.join(d, "tokenizer.json")):
            try:
                from transformers import PreTrainedTokenizerFast
                return PreTrainedTokenizerFast(tokenizer_file=os.path.join(d, "tokenizer.json"))
            except Exception as e:                                           # noqa: BLE001
                print(f"tokenizer at {d} not usable: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return None


def corpus_text():
    files = a.text or sorted(glob.glob(os.path.join(REPO, "*.md")) + glob.glob(os.path.join(REPO, "kyojin_mlx", "*.py")) + glob.glob(os.path.join(REPO, "tools", "*.py")))
    return "\n\n".join(open(f, errors="replace").read() for f in files)


class Prompts:
    def __init__(self, vocab, tok):
        self.vocab, self.tok = vocab, tok
        self.rng = np.random.default_rng(a.seed)
        self.base = None
        if tok is not None:
            ids = tok.encode(corpus_text(), add_special_tokens=False)
            self.base = [int(i) for i in ids if int(i) < vocab]

    def filler(self, n, offset=0):
        if self.base:
            reps = -(-(n + offset) // len(self.base)) + 1
            return (self.base * reps)[offset:offset + n]
        return [int(x) for x in self.rng.integers(3, self.vocab, n)]

    def needle(self, n, depth, code):
        enc = lambda s: [int(i) for i in self.tok.encode(s, add_special_tokens=False)]
        pre = enc(" The secret passcode for the vault is %s. Remember it. " % code)
        post = enc("\n\nQuestion: What is the secret passcode for the vault?\nAnswer: The secret passcode for the vault is")
        body = self.filler(n - len(pre) - len(post), offset=int(depth * 1000))
        k = int(len(body) * depth)
        return body[:k] + pre + body[k:] + post


# ---------------------------------------------------------------------------------------------------------------- runs
def prefill(lm, ids, chunk):
    """(cache, fp32 logits of the last 16 rows [r, V], seconds). Only the last chunk evaluates the head."""
    cache = lm.make_cache()
    t0 = time.time()
    lg = None
    n = len(ids)
    for s in range(0, n, chunk):
        out = lm(mx.array([ids[s:s + chunk]]), cache=cache)
        if s + chunk >= n:
            lg = out.logits[0, -16:].astype(mx.float32)
            mx.eval(lg, *cache[0].state)
        else:
            mx.eval(*cache[0].state)
    return cache, lg, time.time() - t0


def greedy(lm, cache, lg_last, new):
    out = []
    lg = lg_last
    for _ in range(new):
        t = int(mx.argmax(lg).item())
        out.append(t)
        lg = lm(mx.array([[t]]), cache=cache).logits[0, -1].astype(mx.float32)
    return out


def kl_rows(p_lg, q_lg):
    """KL(softmax(p) || softmax(q)) per row, nats."""
    n = min(p_lg.shape[0], q_lg.shape[0])
    p_lg, q_lg = p_lg[-n:], q_lg[-n:]
    lp = p_lg - mx.logsumexp(p_lg, axis=-1, keepdims=True)
    lq = q_lg - mx.logsumexp(q_lg, axis=-1, keepdims=True)
    return np.array(mx.sum(mx.exp(lp) * (lp - lq), axis=-1))


def agree(p_lg, q_lg, min_margin=0.25):
    """(rows where CED's argmax equals the exact one, rows counted). Rows where the exact top-1 leads the runner-up by less than `min_margin`
    nats are near ties and not counted (a rounding-level change may flip them; random-id prompts have many)."""
    n = min(p_lg.shape[0], q_lg.shape[0])
    p, q = p_lg[-n:], q_lg[-n:]
    top2 = np.array(mx.topk(p, 2, axis=-1))
    margin = np.abs(top2[:, 0] - top2[:, 1])
    ok = np.array(mx.argmax(p, axis=-1)) == np.array(mx.argmax(q, axis=-1))
    sel = margin >= min_margin
    return int((ok & sel).sum()), int(sel.sum())


def lcp(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v:
            break
        n += 1
    return n


def run_pair(lm, st, ids, chunk, tail, new, floor):
    """Exact vs CED on one prompt."""
    with C.exact(lm):
        ce, le, t_ex = prefill(lm, ids, chunk)
        ans_e = greedy(lm, ce, le[-1], new)
        del ce
        fl = None
        if floor:
            c2, l2, _ = prefill(lm, ids, chunk * 2 if chunk * 2 < len(ids) else max(1, chunk // 2))
            fl = kl_rows(le, l2)
            del c2
    st.tail, st.on = (C.receptive_rows(st.lay) if tail == "exact" else int(tail)), True
    st.stats.update(chunks=0, rows=0, rows_cut=0)
    cc, lc, t_ced = prefill(lm, ids, chunk)
    ans_c = greedy(lm, cc, lc[-1], new)
    del cc
    k = kl_rows(le, lc)
    top, rows = agree(le, lc)
    return dict(n=len(ids), tail=st.tail, chunk=chunk, kl_last=float(k[-1]), kl_mean=float(k.mean()), kl_max=float(k.max()),
                top1=top, top1_rows=rows, answer_prefix=lcp(ans_e, ans_c), answer_new=new, sec_exact=round(t_ex, 2), sec_ced=round(t_ced, 2),
                tok_s_exact=round(len(ids) / t_ex, 1), tok_s_ced=round(len(ids) / t_ced, 1), rows_cut=st.stats["rows_cut"],
                floor_kl_last=None if fl is None else float(fl[-1]), floor_kl_mean=None if fl is None else float(fl.mean()),
                peak_gib=round(mx.get_peak_memory() / 2 ** 30, 2)), ans_e, ans_c


def needle_run(lm, st, prompts, n, depth, chunk, tail, new):
    code = "%05d" % int(prompts.rng.integers(10000, 99999))
    ids = prompts.needle(n, depth, code)
    with C.exact(lm):
        ce, le, _ = prefill(lm, ids, chunk)
        out_e = prompts.tok.decode(greedy(lm, ce, le[-1], new))
        del ce
    st.tail, st.on = (C.receptive_rows(st.lay) if tail == "exact" else int(tail)), True
    cc, lc, _ = prefill(lm, ids, chunk)
    out_c = prompts.tok.decode(greedy(lm, cc, lc[-1], new))
    del cc
    return dict(n=n, depth=depth, code=code, exact_ok=code in out_e, ced_ok=code in out_c, exact_out=out_e[:40], ced_out=out_c[:40])


def floor_noisy(r) -> bool:
    """The exact prefill on another chunk grid already differs from itself by > 0.1 nats at the last row (a near-tie prompt, e.g. 4096 tokens of the day-1
    text: floor KL 0.25): a greedy continuation diverges after a token between two exact prefills too, so the answer-prefix test says nothing there. The KL
    bound already scales with the floor."""
    f = r.get("floor_kl_last")
    return f is not None and f > 0.1


def main(argv=None, loader=None):
    global a
    a = ap.parse_args(argv)
    overrides = json.loads(a.text_overrides) if a.text_overrides else None
    model, spec, cfg, tables = (loader or load)(a.model, overrides)
    lm = model.language_model
    if spec.ced is None:
        print(f"CED: {spec.name} has no CED description (ArchSpec.ced): not applicable")
        sys.exit(2)
    why = C.explain(lm.config, a.mid)
    if why:
        print(f"CED: not applicable to this pack: {why}")
        sys.exit(2)
    tails = [t if t == "exact" else int(t) for t in a.tails.split(",")]
    C.enable(model, spec, tail=tails[0] if tails[0] != "exact" else "exact", mid=a.mid)
    st = C.state(lm)
    vocab = cfg.get("text_config", cfg)["vocab_size"]
    tok = find_tokenizer()
    prompts = Prompts(vocab, tok)
    print(f"ced_gate: {spec.name} mid={st.lay.mid}/{st.lay.n_layers} window={st.lay.window} receptive={C.receptive_rows(st.lay)} chunk={a.chunk} "
          f"tails={tails} tokens={'tokenizer text' if prompts.base else 'synthetic ids'} needle={'yes' if tok and not a.no_needle else 'SKIP'}", flush=True)
    lengths = [int(x) for x in a.lengths.split(",") if x]
    rows, needles, ok = [], [], True
    for ti, tail in enumerate(tails):
        for n in lengths:
            if a.max_peak_gib and mx.get_peak_memory() / 2 ** 30 > a.max_peak_gib:
                print(f"peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB over --max-peak-gib: stop", flush=True)
                break
            ids = prompts.filler(n)
            r, ae, ac = run_pair(lm, st, ids, a.chunk, tail, a.new, not a.no_floor)
            r["tail_arg"] = str(tail)
            bound = a.kl_max if r["floor_kl_mean"] is None else max(a.kl_max, 2 * r["floor_kl_mean"], 2 * r["floor_kl_last"])
            r["kl_bound"] = bound
            r["pass"] = bool(r["kl_last"] <= bound and r["kl_mean"] <= bound and r["top1"] >= r["top1_rows"] - 1 and (r["answer_prefix"] >= 0.75 * a.new or floor_noisy(r)))
            rows.append(r)
            if ti == 0:
                ok &= r["pass"]
            print(json.dumps(r), flush=True)
            if tok and not a.no_needle and ti == 0:
                for depth in (0.1, 0.5, 0.9):
                    nr = needle_run(lm, st, prompts, n, depth, a.chunk, tail, 12)
                    needles.append(nr)
                    if nr["exact_ok"] and not nr["ced_ok"]:
                        ok = False
                    print(json.dumps(nr), flush=True)
    last = rows[0] if rows else None
    if last:
        sp = {r["n"]: round(r["tok_s_ced"] / r["tok_s_exact"], 2) for r in rows if r["tail_arg"] == str(tails[0])}
        print(f"prefill speed ratio CED/exact (tail {tails[0]}): {sp}", flush=True)
    peak = mx.get_peak_memory() / 2 ** 30
    print(f"VERDICT {'PASS' if ok and rows else 'FAIL'}: tail {tails[0]}, {len([r for r in rows if r['tail_arg'] == str(tails[0])])} lengths, "
          f"{sum(1 for r in rows if r['tail_arg'] == str(tails[0]) and r['pass'])} pass, needles {sum(1 for x in needles if x['ced_ok'])}/{len(needles)} "
          f"(exact {sum(1 for x in needles if x['exact_ok'])}), peak {peak:.1f} GiB", flush=True)
    if a.dump:
        json.dump(dict(rows=rows, needles=needles, verdict=bool(ok and rows), model=a.model, window=st.lay.window, mid=st.lay.mid), open(a.dump, "w"), indent=1)
    if tables is not None:
        tables.close()
    sys.exit(0 if ok and rows else 1)


if __name__ == "__main__":
    main()
