"""Streamed BF16 reference for the Qwen3.5 hybrid (GDN + full attention): the BF16 checkpoint is loaded by mlx-lm lazily and run doc by doc;
only the top-256 log-probs per position are kept (macair41's conv harness: KL on the renormalised top-256, top-1, NLL).
usage: qwen35.py REF_DIR DOCS.json OUTDIR   -> OUTDIR/ref_<doc>.npz {idx, lp, top1, nll}
Needs the whole BF16 model in memory: the qualify planner refuses it above the footprint cap (the Ultra day-1 adapter for bigger models streams layers)."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model

TOP = 256
CH = 256


def head_of(lm):
    return (lambda h: lm.model.embed_tokens.as_linear(h)) if getattr(lm, "lm_head", None) is None else lm.lm_head


def doc_ref(model, ids):
    lm = model.language_model
    hd = head_of(lm)
    x = mx.array(ids)[None]
    h = lm.model(x)[0]
    out = {"idx": [], "lp": [], "nll": [], "top1": []}
    for s in range(0, len(ids) - 1, CH):
        e = min(s + CH, len(ids) - 1)
        lg = hd(h[s:e]).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(ids[s + 1:e + 1])
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        ti = mx.argpartition(-lp, TOP, axis=-1)[:, :TOP]
        tl = mx.take_along_axis(lp, ti, axis=-1)
        am = mx.argmax(lp, axis=-1)
        mx.eval(ti, tl, nll, am)
        out["idx"].append(np.array(ti)); out["lp"].append(np.array(tl)); out["nll"].append(np.array(nll)); out["top1"].append(np.array(am))
        mx.clear_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


def main(ref_dir, docs, outdir):
    d = json.loads(Path(docs).read_text())
    model, _ = load_model(Path(ref_dir), lazy=False)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for n, ids in zip(d["names"], d["ids"]):
        r = doc_ref(model, ids)
        np.savez(Path(outdir) / f"ref_{n}.npz", **r)
        print(n, len(ids), "nll", float(r["nll"].mean()), "peak GiB", round(mx.get_peak_memory() / 2**30, 2), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
