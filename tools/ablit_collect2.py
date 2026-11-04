#!/usr/bin/env python3
"""ablit_collect2.py -- second refusal direction, read where the served refusal is decided.

The shipped spec reads one direction at the last prompt token (before any thinking). Under served conditions (long
system prompt, thinking on) the reply can still decline *after* the thinking block. This tool reads the same model
with the current spec already active and captures the residual state at the first answer token -- i.e. after the
stored reasoning and the closing `</think>`, at the position that predicts the first answer token. What remains of
the decline - comply difference there is the second direction.

Per sample: render the server chat template (thinking on, reasoning medium), append the stored reasoning and
`</think>`, and run one forward pass (chunks through the model cache) with the current spec armed through
`kyojin_mlx.ablit`. The readout is the output of block L, averaged over the mHC streams (the collect_ds41 readout).

Direction: mean(decline) - mean(comply) per layer; the layer with the largest |Cohen d| of the projection wins;
the vector is Gram-Schmidt orthogonalised against the current spec's rows. The result is written as
<out>/edit_spec.json + <out>/edit_spec.safetensors with "R" [K+1, hidden] (K = rows of the current spec; K=1 -> 2).

Weights: attn_w / mlp_w are copied unchanged from the current spec (first direction, same per-layer layout).
The second direction's own profile is written as attn_w2 / mlp_w2 = --scale * attn_w / mlp_w (scale defaults to 1).
Note: the shipped `kyojin_mlx.ablit.load` still applies attn_w / mlp_w to every row of R, so a run with the K=2
spec uses the first direction's strength for both; the attn_w2 / mlp_w2 pair is the hook a per-row runtime needs.

No sample text, prompt or system prompt is ever printed or written: only counts, ids and norms.

  ablit_collect2.py --samples S.jsonl --prompts P.jsonl --sys sys.txt --labels labels.json \\
      --spec final/edit_spec.json --model ~/ds41/model --out out/ablit2 --layers 22,26,30 --limit 400 \\
      [--chunk 2048] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kyojin_mlx.serve.ds41_chat import THINK_CLOSE, render_chat  # noqa: E402

LOG = lambda m: print(f"[{time.strftime('%H:%M:%S')}] [ablit_collect2] {m}", flush=True)


# ---------------------------------------------------------------------------------------------------------------- io
def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _key(value):
    """Sample ids as written in the labels file, tolerant of "12" vs 12."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    s = str(value)
    return int(s) if s.lstrip("-").isdigit() else s


def load_labels(path: str, all_ids: list | None = None) -> tuple[list, list, list]:
    """-> (decline ids, comply ids, excluded ids). JSON object {"decline": [...], "comply": [...]}; other id lists
    in the object (e.g. "partial") are excluded from comply. With no "comply" key, comply = all_ids - decline - excluded.
    Also accepts a JSON list of two lists (decline, comply)."""
    d = json.load(open(path))
    if isinstance(d, dict):
        decline = d.get("decline") or d.get("refuse") or []
        comply = d.get("comply") or d.get("compliance") or []
        excluded = set()
        for k, v in d.items():
            if k in ("decline", "comply", "refuse", "compliance"):
                continue
            if isinstance(v, list):
                excluded |= {_key(x) for x in v}
        decline = [_key(x) for x in decline]
        comply = [_key(x) for x in comply]
        if not comply and all_ids is not None:
            ds = set(decline)
            comply = [i for i in all_ids if i not in ds and i not in excluded]
        return decline, comply, sorted(excluded, key=str)
    if isinstance(d, (list, tuple)) and len(d) == 2:
        return [_key(x) for x in d[0]], [_key(x) for x in d[1]], []
    raise ValueError("labels: expected a JSON object with decline/comply id lists")


def render_prompt(sys_text: str, prompt_row: dict, reasoning: str | None) -> str:
    """Server template, thinking on, reasoning medium; then the stored reasoning and the closing block tag. The last
    token is the one that predicts the first answer token."""
    text = prompt_row.get("text", prompt_row.get("prompt", prompt_row.get("content", "")))
    head = render_chat([{"role": "system", "content": sys_text}, {"role": "user", "content": text}],
                       thinking=True, reasoning_effort="medium")
    r = reasoning or ""
    if r.endswith(THINK_CLOSE):
        r = r[:-len(THINK_CLOSE)]
    return head + r + THINK_CLOSE


# ------------------------------------------------------------------------------------------------------------ fitting
def fit_direction(decline: np.ndarray, comply: np.ndarray) -> tuple[np.ndarray, float]:
    """mean(decline) - mean(comply) and the Cohen d of the samples projected on it (pooled std, ddof 1)."""
    D = np.asarray(decline, dtype=np.float64)
    C = np.asarray(comply, dtype=np.float64)
    if D.ndim != 2 or C.ndim != 2 or D.shape[1] != C.shape[1] or len(D) < 1 or len(C) < 1:
        raise ValueError("fit_direction: need two non-empty [n, hidden] matrices")
    delta = D.mean(0) - C.mean(0)
    nrm = float(np.linalg.norm(delta))
    if nrm < 1e-12:
        return delta.astype(np.float32), 0.0
    u = delta / nrm
    pd, pc = D @ u, C @ u
    nd, nc = len(D), len(C)
    vd = pd.var(ddof=1) if nd > 1 else 0.0
    vc = pc.var(ddof=1) if nc > 1 else 0.0
    dof = max(nd + nc - 2, 1)
    sp = float(np.sqrt(((nd - 1) * vd + (nc - 1) * vc) / dof))
    diff = float(pd.mean() - pc.mean())
    d = diff / sp if sp > 1e-12 else (float(np.sign(diff)) * 1e6 if diff else 0.0)
    return delta.astype(np.float32), float(d)


def choose_layer(hidden: dict[int, dict], decline_ids: list, comply_ids: list, min_n: int = 2):
    """hidden[L][sample_id] -> vector [hidden]. -> (layer, delta, cohen_d, per-layer stats)."""
    ds, cs = set(decline_ids), set(comply_ids)
    stats, best = [], None
    for L in sorted(hidden):
        h = hidden[L]
        D = [h[k] for k in decline_ids if k in h and k in ds]
        C = [h[k] for k in comply_ids if k in h and k in cs]
        st = {"layer": int(L), "n_decline": len(D), "n_comply": len(C)}
        if len(D) < min_n or len(C) < min_n:
            st["skipped"] = True
            stats.append(st)
            continue
        delta, d = fit_direction(np.stack(D), np.stack(C))
        st.update({"cohen_d": float(d), "separation": float(np.linalg.norm(delta)),
                   "mean_decline_norm": float(np.linalg.norm(np.mean(D, 0))),
                   "mean_comply_norm": float(np.linalg.norm(np.mean(C, 0)))})
        stats.append(st)
        if best is None or abs(d) > abs(best[1]):
            best = (int(L), float(d), delta)
    if best is None:
        raise ValueError("choose_layer: no layer has enough labelled samples on both sides")
    return best[0], best[2], best[1], stats


def orthonormal_rows(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 1:
        R = R[None]
    n = np.linalg.norm(R, axis=1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("R has a zero row")
    return R / n


def gram_schmidt(direction: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Direction minus its components on every current row, normalised."""
    d = np.asarray(direction, dtype=np.float64).copy()
    for row in orthonormal_rows(R):
        d -= float(d @ row) * row
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        raise ValueError("the new direction lies in the span of the current R: nothing left to project")
    return (d / n).astype(np.float32)


# --------------------------------------------------------------------------------------------------------- spec io
def _resolve_r_file(spec: dict, spec_path: str) -> str:
    rf = spec.get("r_file")
    if rf:
        return rf if os.path.isabs(rf) else os.path.join(os.path.dirname(os.path.abspath(spec_path)), rf)
    return os.path.splitext(spec_path)[0] + ".safetensors"


def read_current_spec(spec_path: str) -> tuple[dict, np.ndarray]:
    """-> (spec dict, R [K, hidden] normalised rows)."""
    from safetensors.numpy import load_file
    spec = json.load(open(spec_path))
    t = load_file(_resolve_r_file(spec, spec_path))
    R = t["R"] if "R" in t else t["r"]
    if R.ndim == 1:
        R = R[None]
    if R.shape[-1] != int(spec["hidden"]):
        raise ValueError(f"current spec: direction {R.shape} does not fit hidden {spec['hidden']}")
    return spec, orthonormal_rows(R).astype(np.float32)


def write_spec(out_dir: str, spec: dict, R_new: np.ndarray, layer: int, cohen_d: float, scale: float,
               layer_stats: list, n_decline: int, n_comply: int) -> tuple[str, str]:
    """Copy the current per-layer layout, append the new row, write edit_spec.json + .safetensors."""
    from safetensors.numpy import save_file
    os.makedirs(out_dir, exist_ok=True)
    hidden, n_layers = int(spec["hidden"]), int(spec["n_layers"])
    if len(spec["attn_w"]) < n_layers or len(spec["mlp_w"]) < n_layers:
        raise ValueError("current spec: attn_w / mlp_w shorter than n_layers")
    out = dict(spec)
    out.pop("r_file", None)
    out["hidden"], out["n_layers"] = hidden, n_layers
    out["attn_w"] = [float(x) for x in spec["attn_w"][:n_layers]]
    out["mlp_w"] = [float(x) for x in spec["mlp_w"][:n_layers]]
    out["attn_w2"] = [scale * float(x) for x in out["attn_w"]]
    out["mlp_w2"] = [scale * float(x) for x in out["mlp_w"]]
    out["direction_mode"] = "k2-projected"
    out["direction_index"] = int(layer)
    out["direction_index2"] = int(layer)
    out["cohen_d2"] = float(cohen_d)
    out["scale2"] = float(scale)
    out["readout"] = (f"output of block {int(layer)} (mean over the mHC streams) at the first answer token, "
                      "after the stored reasoning and </think>, with the current spec active")
    out["n_decline"] = int(n_decline)
    out["n_comply"] = int(n_comply)
    out["candidates"] = layer_stats
    json_path = os.path.join(out_dir, "edit_spec.json")
    st_path = os.path.join(out_dir, "edit_spec.safetensors")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
    save_file({"R": np.asarray(R_new, dtype=np.float32)}, st_path)
    return json_path, st_path


# ----------------------------------------------------------------------------------------------------------- capture
def _arrays(obj, seen=None) -> list:
    """Every mx.array reachable from obj (lists, dicts, objects); bounds the lazy graph between chunks."""
    import mlx.core as mx
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        return [obj]
    out = []
    if hasattr(obj, "raw_arrays"):
        return list(obj.raw_arrays())
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out += _arrays(v, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _arrays(v, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for v in vars(obj).values():
            out += _arrays(v, seen)
    return out


class _Tap:
    """Wrap one block instance: record the block output (mean over the mHC streams) when armed."""

    def __init__(self, block, layer: int, store: dict):
        self.block, self.layer, self.store, self.armed = block, layer, store, False

    @property
    def engram(self):
        return getattr(self.block, "engram", None)

    def __call__(self, h, pre_mix, image_mask, cache):
        out = self.block(h, pre_mix, image_mask, cache)
        if self.armed:
            hh = out[0] if isinstance(out, tuple) else out
            if getattr(hh, "ndim", 0) != 4:
                raise RuntimeError(f"capture: block {self.layer} output rank {getattr(hh, 'ndim', None)} != 4")
            self.store[self.layer] = hh.mean(axis=2)[:, -1]          # [batch, hidden] of the last token
        return out


def capture(lm, ids: list[int], layers, chunk: int = 2048) -> dict[int, np.ndarray]:
    """One forward pass over ids (chunks through the cache, no generation); -> {layer: hidden [H] at the last token}."""
    import mlx.core as mx
    layers = sorted({int(x) for x in layers})
    if not ids:
        raise ValueError("capture: empty token sequence")
    n_lay = len(lm.layers)
    if layers[0] < 0 or layers[-1] >= n_lay:
        raise ValueError(f"capture: layers {layers} outside model ({n_lay} blocks)")
    original = [lm.layers[L] for L in layers]
    taps = {L: None for L in layers}
    for L, blk in zip(layers, original):
        lm.layers[L] = _Tap(blk, L, taps)
    try:
        cache = lm.make_cache()
        n = len(ids)
        step = max(1, int(chunk) if chunk else n)
        i = 0
        while i < n:
            j = min(n, i + step)
            last = j >= n
            for L in layers:
                lm.layers[L].armed = last
            lm(mx.array(ids[i:j])[None], cache=cache)
            if not last:
                mx.eval(*_arrays(cache))
            i = j
        vals = {}
        for L in layers:
            v = taps[L]
            if v is None:
                raise RuntimeError(f"capture: no readout at block {L}")
            vals[L] = v[0]
        mx.eval(*vals.values(), *_arrays(cache))
        return {L: np.array(vals[L], dtype=np.float32) for L in layers}
    finally:
        for L, blk in zip(layers, original):
            lm.layers[L] = blk


# ------------------------------------------------------------------------------------------------------------ collect
def collect(lm, encode, samples: list[dict], prompts: list[dict], sys_text: str, labels: tuple,
            spec_path: str, out_dir: str, layers=(22, 26, 30), limit: int | None = None, chunk: int = 2048,
            scale: float = 1.0, log=LOG) -> dict:
    """Run the collection and write the K+1 spec. labels = (decline ids, comply ids, excluded ids)."""
    t0 = time.time()
    samples = samples[:limit] if limit else samples
    decline, comply, excluded = labels
    ds, cs, xs = set(decline), set(comply), set(excluded)
    if ds & cs:
        raise ValueError(f"labels: {len(ds & cs)} sample ids are in both decline and comply")

    from kyojin_mlx import ablit as AB
    if not AB._installed:                                         # main: patch_model installed it before fuse/rows
        AB.install_block()
    armed = AB.attach(lm, spec_path)
    log(f"current spec armed on {armed} blocks; {len(samples)} samples, layers {list(layers)}")

    hidden = {int(L): {} for L in layers}
    used = skipped_prompt = skipped_unlabelled = no_reasoning = 0
    for row in samples:
        key = _key(row.get("i", row.get("id")))
        if key in xs or (key not in ds and key not in cs):
            skipped_unlabelled += 1
            continue
        pi = row.get("prompt-index", row.get("prompt_index"))
        if pi is None or not (0 <= int(pi) < len(prompts)):
            skipped_prompt += 1
            continue
        reasoning = row.get("reasoning")
        if not reasoning:
            no_reasoning += 1
        ids = encode(render_prompt(sys_text, prompts[int(pi)], reasoning))
        if not ids:
            skipped_prompt += 1
            continue
        vals = capture(lm, ids, layers, chunk=chunk)
        for L in layers:
            hidden[int(L)][key] = vals[int(L)]
        used += 1
        if used % 10 == 0:
            log(f"{used} samples captured ({time.time() - t0:.0f}s)")
        del vals
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
    log(f"captured {used} samples (skipped: {skipped_unlabelled} unlabelled/excluded, {skipped_prompt} prompt, "
        f"{no_reasoning} without reasoning)")
    if used == 0:
        raise SystemExit("ablit_collect2: no usable sample")

    layer, delta, cohen_d, stats = choose_layer(hidden, decline, comply)
    cur_spec, R_cur = read_current_spec(spec_path)
    new_row = gram_schmidt(delta, R_cur)
    R_new = np.vstack([orthonormal_rows(R_cur).astype(np.float32), new_row[None]])
    if not np.allclose(R_new @ R_new.T, np.eye(len(R_new)), atol=1e-5):
        raise RuntimeError("ablit_collect2: R rows are not orthonormal")
    n_decl = sum(1 for k in decline if k in hidden[layer])
    n_comp = sum(1 for k in comply if k in hidden[layer])
    json_path, st_path = write_spec(out_dir, cur_spec, R_new, layer, cohen_d, scale, stats, n_decl, n_comp)
    for st in stats:
        if not st.get("skipped"):
            log(f"layer {st['layer']}: d={st['cohen_d']:.3f} |delta|={st['separation']:.2f} "
                f"(decline {st['n_decline']}, comply {st['n_comply']})")
    log(f"chosen layer {layer}: Cohen d {cohen_d:.3f}; new row |r|={float(np.linalg.norm(new_row)):.6f}, "
        f"R {R_new.shape}; scale2 {scale}")
    return {"layer": layer, "cohen_d": cohen_d, "R": list(R_new.shape), "used": used,
            "skipped_unlabelled": skipped_unlabelled, "skipped_prompt": skipped_prompt,
            "no_reasoning": no_reasoning, "out": os.path.dirname(json_path), "stats": stats}


# ---------------------------------------------------------------------------------------------------------------- cli
def parse_layers(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(" ", "").split(",") if x != ""]


def make_encode(model_dir: str):
    """The server tokenizer: <pack>/tokenizer.json (fast), else the model directory's tokenizer."""
    tj = os.path.join(model_dir, "tokenizer.json")
    if os.path.isfile(tj):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=tj)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)

    def enc(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False)["input_ids"])
    return enc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect a second (first answer token) refusal direction; write a K=2 spec")
    ap.add_argument("--samples", required=True, help="JSONL: i, prompt-index, reasoning, content")
    ap.add_argument("--prompts", required=True, help="JSONL prompts (row field 'text')")
    ap.add_argument("--sys", required=True, help="system prompt file")
    ap.add_argument("--labels", required=True, help="JSON with decline / comply id lists")
    ap.add_argument("--spec", required=True, help="current spec (json, .safetensors next to it)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default=os.environ.get("KYOJIN_MODEL_DIR", ""), help="pack directory (or KYOJIN_MODEL_DIR)")
    ap.add_argument("--layers", default="22,26,30", help="readout layers to compare")
    ap.add_argument("--limit", type=int, default=0, help="at most this many samples (0 = all)")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk (tokens) per forward call")
    ap.add_argument("--scale", type=float, default=1.0, help="weights of the second direction (x attn_w / mlp_w)")
    a = ap.parse_args(argv)
    if not a.model:
        ap.error("--model PACK_DIR is required (or set KYOJIN_MODEL_DIR)")

    samples = read_jsonl(a.samples, a.limit or None)
    prompts = read_jsonl(a.prompts)
    sys_text = open(a.sys).read()
    labels = load_labels(a.labels, [_key(s.get("i", s.get("id"))) for s in samples])
    LOG(f"samples {len(samples)}, prompts {len(prompts)}, labels: {len(labels[0])} decline, "
        f"{len(labels[1])} comply, {len(labels[2])} excluded")

    os.environ["KYOJIN_ABLIT_SPEC"] = os.path.abspath(a.spec)    # must be set before patch_model / fuse install
    from kyojin_mlx import ablit as AB
    from kyojin_mlx.serve.backends import load_backend
    t0 = time.time()
    be, _cfg = load_backend(a.model, ced=False, mtp=False, log=LOG)   # exact prefill, no CED, no draft head
    lm = getattr(be, "lm", None) or be.model.language_model
    LOG(f"model loaded in {time.time() - t0:.0f}s, spec {AB.spec_path()}")
    encode = make_encode(a.model)
    out_dir = a.out
    summary = collect(lm, encode, samples, prompts, sys_text, labels, a.spec, out_dir,
                      layers=parse_layers(a.layers), limit=a.limit or None, chunk=a.chunk, scale=a.scale)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "collect2_stats.json"), "w") as f:
        json.dump(summary, f, indent=1)
    LOG(f"wrote {os.path.join(out_dir, 'edit_spec.json')} and edit_spec.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
