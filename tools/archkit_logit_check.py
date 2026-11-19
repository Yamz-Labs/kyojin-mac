#!/usr/bin/env python3
"""Logit check of a GGUF loaded through the architecture kit (CPU) against llama.cpp.

oracle (once, per file):  llama-perplexity -m M.gguf -f prompts.txt -c 64 --chunks 3 -ngl 0 -nr -b 64 --kl-divergence-base ref.bin
  -nr (no weight repack) with a batch of >= 32 tokens sends the matmuls to llama.cpp's BLAS backend: dequantized weights, F32 activations (exact arithmetic).
  Without -nr, K-quant files run the Q8_K-activation dot products, which add pseudo-random noise (KL ~0.003, top-1 ~94 %) that no other engine can reproduce.
check:                    python tools/archkit_logit_check.py M.gguf ref.bin [--q8k] [--skeleton] [--ties [--tie-thr 1e-3]]
--ties (MoE near-tie awareness): the router linears of the model loaded through the kit are tapped; for every position the smallest gap between the k-th and (k+1)-th largest router logit
over all MoE layers is recorded (k = expert_used_count of the layer). A top-k choice with a gap below --tie-thr (default 1e-3) can flip between two engines for arithmetic reasons alone, and the
flip then travels through the KV cache. The comparison is printed twice: all positions, and without the positions at / after the first near-tie of their chunk (the first tie is looked for over
the whole chunk, the unscored first half included, because it reaches the scored half through the KV cache). A real error shows in the second line, a tie does not.
Gaps are measured on the router logits (ranking of softmax / sigmoid gates is the same); a router with a selection bias added after the gate (e_score_correction_bias) is UNVERIFIED here.

ref.bin holds, for the second half of each chunk, the log-probabilities llama.cpp computed, quantised to 16 bits over [max-16, max] nats
(tools/perplexity/perplexity.cpp: log_softmax / process_logits). The chunk tokens are in the file, so no tokenizer is involved on our side."""
from __future__ import annotations

import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_default_device(mx.cpu)
from kyojin_mlx import foreign_archkit_cpu as C  # noqa: E402


def read_ref(path):
    b = open(path, "rb").read()
    assert b[:8] == b"_logits_", "not a --kl-divergence-base file"
    n_ctx, n_vocab, n_chunk = struct.unpack_from("<iii", b, 8)
    p = 8 + 12
    toks = np.frombuffer(b, dtype=np.int32, count=n_ctx * n_chunk, offset=p).reshape(n_chunk, n_ctx)
    p += toks.nbytes
    nv = 2 * ((n_vocab + 1) // 2) + 4
    first = n_ctx // 2
    n_pos = n_ctx - 1 - first
    ref = []
    for c in range(n_chunk):
        rows = np.frombuffer(b, dtype=np.uint16, count=n_pos * nv, offset=p).reshape(n_pos, nv)
        p += n_pos * nv * 2
        sm = rows[:, :4].copy().view(np.float32)                       # scale, min_log_prob
        q = rows[:, 4:4 + n_vocab].astype(np.float32)
        ref.append((q * sm[:, :1] + sm[:, 1:2], q > 0))
    assert p == len(b), f"{len(b) - p} trailing bytes"
    return n_ctx, n_vocab, toks, first, ref


def q8k_round(x):
    """ggml quantize_row_q8_K_ref (ggml-quants.c): per 256 block, iscale = -127 / (signed element of largest |x|), q = min(127, rint(iscale * x)), value = q / iscale.
    Returns the dequantized activations (what the integer dot product of the K-quant CPU kernels effectively multiplies by)."""
    sh = x.shape
    b = x.astype(mx.float32).reshape(-1, sh[-1] // 256, 256)
    i = mx.argmax(mx.abs(b), axis=-1, keepdims=True)
    mx_ = mx.take_along_axis(b, i, axis=-1)
    safe = mx.where(mx_ == 0, 1.0, mx_)
    iscale = -127.0 / safe
    q = mx.minimum(mx.round(iscale * b), 127.0)                       # mx.round = half to even = ggml nearest_int
    y = mx.where(mx_ == 0, 0.0, q / iscale)
    return y.reshape(sh)


def wrap_q8k(model):
    """Replace every nn.Linear by a wrapper that rounds its input as ggml does before a K-quant x Q8_K dot product."""
    import mlx.nn as nn

    class Q8KLin(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin

        def __call__(self, x):
            return self.lin(q8k_round(x))

    n = 0

    def walk(m):
        nonlocal n
        for k, v in m.children().items():
            for key, child in ([(k, v)] if not isinstance(v, list) else [(None, e) for e in v]):
                if isinstance(child, nn.Linear):
                    setattr(m, k, Q8KLin(child)); n += 1
                elif isinstance(child, nn.Module):
                    walk(child)
                elif isinstance(child, list):
                    for e in child:
                        if isinstance(e, nn.Module): walk(e)
    walk(model)
    return n


def bos_first(gguf, toks):
    """llama-perplexity decodes BOS in place of the first token of every chunk when the vocabulary adds BOS (perplexity.cpp:387-398) but stores the original token in the
    --kl-divergence-base file. Same rule here: the GGUF key tokenizer.ggml.add_bos_token, default true for a SentencePiece (`llama`) tokenizer, false otherwise (llama-vocab.cpp)."""
    from kyojin_mlx import foreign_gguf as GF
    g = GF.GgufFile(gguf)
    try:
        add = bool(g.kv.get("tokenizer.ggml.add_bos_token", g.kv.get("tokenizer.ggml.model") == "llama"))
        bos = g.kv.get("tokenizer.ggml.bos_token_id")
    finally:
        g.close()
    toks = toks.copy()
    if add and bos is not None:
        toks[:, 0] = int(bos)
    return toks


class RouterTap:
    """Records the output logits of the router linears: one entry per tapped module per forward."""

    def __init__(self):
        self.layers = {}                     # tapped module path -> list of [T, E] float32 arrays (one per forward since reset)

    def reset(self):
        for v in self.layers.values():
            v.clear()


def tap_routers(model, n_expert, top_k=None):
    """Wrap every router linear of `model` (an nn.Linear whose name holds 'gate' or 'router', not '*_proj', with n_expert outputs) so that its logits are recorded.
    n_expert: int, or the per-layer list of expert counts. Returns (tap, [(path, layer index or None)])."""
    import mlx.nn as nn
    tap = RouterTap()
    found = []
    want = set(n_expert) if isinstance(n_expert, (list, tuple)) else {int(n_expert)}

    class Tapped(nn.Module):
        def __init__(self, lin, path):
            super().__init__()
            self.lin, self.path = lin, path

        def __call__(self, x):
            y = self.lin(x)
            tap.layers[self.path].append(np.array(y.astype(mx.float32)).reshape(-1, y.shape[-1]))
            return y

    def is_router(path, m):
        last = path.rsplit(".", 1)[-1]
        out = m.weight.shape[0] if hasattr(m, "weight") else None
        return (isinstance(m, nn.Linear) or type(m).__name__ == "QuantizedLinear") and out in want and ("gate" in path or "router" in path) and not last.endswith("_proj")

    def walk(m, prefix):
        for k in list(m.keys()):
            v = m[k]
            items = [(f"{prefix}{k}.{i}", i, e) for i, e in enumerate(v)] if isinstance(v, list) else [(f"{prefix}{k}", None, v)]
            for path, i, child in items:
                if not isinstance(child, nn.Module):
                    continue
                if is_router(path, child):
                    w = Tapped(child, path)
                    tap.layers[path] = []
                    if i is None:
                        m[k] = w
                    else:
                        v[i] = w
                    lm = re.search(r"(?:layers|blocks|h)\.(\d+)\.", path)
                    found.append((path, int(lm.group(1)) if lm else None))
                else:
                    walk(child, path + ".")

    walk(model, "")
    return tap, found


def position_gaps(tap, found, top_k):
    """Per position of the last forward: the smallest gap between the k-th and (k+1)-th largest router logit over the tapped layers. top_k: int, or the per-layer list."""
    gaps = None
    for path, li in found:
        arrs = tap.layers[path]
        if not arrs:
            continue
        lg = arrs[-1]
        k = (top_k[li] if isinstance(top_k, (list, tuple)) and li is not None and li < len(top_k) else max(top_k) if isinstance(top_k, (list, tuple)) else int(top_k))
        if k <= 0 or k >= lg.shape[-1]:
            continue
        sr = -np.sort(-lg, axis=-1)
        g = sr[:, k - 1] - sr[:, k]
        gaps = g if gaps is None else np.minimum(gaps, g)
    return gaps


def first_tie(gaps, thr):
    """Index of the first position whose smallest gap is below thr, or None."""
    if gaps is None:
        return None
    i = np.nonzero(np.asarray(gaps) < thr)[0]
    return int(i[0]) if i.size else None


def kept_before_tie(pos, ft):
    """A position stays in the second comparison when its chunk has no near-tie, or it lies before the first one (the flip changes that position and, via the KV cache, every later one)."""
    return ft is None or pos < ft


class Stats:
    def __init__(self):
        self.top1 = self.n = 0
        self.dmax = 0.0
        self.dmean, self.kl, self.miss = [], [], []

    def add(self, ours, r, valid):
        self.n += 1
        self.top1 += int(ours.argmax() == r.argmax())
        if ours.argmax() != r.argmax():                       # how far apart the two top candidates are for llama.cpp itself
            s2 = np.sort(r)[-2:]
            self.miss.append(round(float(s2[1] - s2[0]), 3))
        d = np.abs(ours[valid] - r[valid])
        self.dmax = max(self.dmax, float(d.max()))
        self.dmean.append(float(d.mean()))
        pr = np.exp(r[valid])                                 # tokens inside llama.cpp's [max-16, max] window; the rest is clamped there
        self.kl.append(float((pr * (r[valid] - ours[valid])).sum()))

    def line(self, label="positions"):
        if not self.n:
            return f"{label} 0: nothing to compare"
        return (f"{label} {self.n}: top-1 agreement {self.top1}/{self.n}, max |dlogp| {self.dmax:.4f}, mean |dlogp| {np.mean(self.dmean):.5f}, "
                f"mean KL(ref||ours) {np.mean(self.kl):.6f}, max KL {np.max(self.kl):.6f}")


def moe_counts(gguf):
    """(arch, expert_count, expert_used_count) of the file; each a scalar or a per-layer list."""
    from kyojin_mlx import foreign_gguf as GF
    g = GF.GgufFile(gguf)
    try:
        a = g.kv.get("general.architecture")
        return a, g.kv.get(f"{a}.expert_count"), g.kv.get(f"{a}.expert_used_count")
    finally:
        g.close()


def main():
    argv = sys.argv[1:]
    thr = 1e-3
    if "--tie-thr" in argv:
        i = argv.index("--tie-thr")
        thr = float(argv[i + 1])
        del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv
    args = [a for a in argv if not a.startswith("--")]
    gguf, refp = args[0], args[1]
    n_ctx, n_vocab, toks, first, ref = read_ref(refp)
    toks = bos_first(gguf, toks)
    model, rep = C.load_cpu(gguf, skeleton="--skeleton" in sys.argv)      # --skeleton: the config-driven skeleton even when an mlx-lm class exists
    if "--q8k" in sys.argv:
        print("Q8_K activation rounding on", wrap_q8k(model), "linears")
    print(f"arch {rep['arch']} -> mlx-lm {rep['model_type']}, {rep['n_tensors']} tensors, config {rep['config']}")
    allp, clean = Stats(), Stats()
    tap = found = None
    n_tied_chunks = 0
    if "--ties" in sys.argv:
        _, ne, nu = moe_counts(gguf)
        lo, hi, vals = C.GF.kv_counts(ne)
        top_k = C.GF.kv_counts(nu)[2] or C.GF.kv_counts(nu)[1]
        if hi:
            tap, found = tap_routers(model, vals if vals is not None else hi)
        print(f"near-tie check: {len(found) if found else 0} router linear(s) tapped, threshold {thr:g}" + ("" if found else " (none found: UNVERIFIED, no tie information)"))
    for c in range(toks.shape[0]):
        x = mx.array(toks[c][None].astype(np.int32))
        if tap:
            tap.reset()
        lg = model(x)[0].astype(mx.float32)
        lp = np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True))   # [n_ctx, V]
        ours = lp[first:n_ctx - 1]
        r, valid = ref[c]
        gaps = position_gaps(tap, found, top_k) if tap and found else None
        ft = first_tie(gaps, thr)
        n_tied_chunks += ft is not None
        if gaps is not None:
            print(f"prompt {c}: smallest top-k gap {float(gaps.min()):.2e} at position {int(gaps.argmin())}; first near-tie (< {thr:g}) at position {ft if ft is not None else '-'}")
        for j in range(ours.shape[0]):
            allp.add(ours[j], r[j], valid[j])
            if kept_before_tie(first + j, ft):               # absolute position in the chunk; positions at / after the first tie are left out of the second line
                clean.add(ours[j], r[j], valid[j])
        last = ours[-1]
        print(f"prompt {c}: last stored position, next-token top5 ours {np.argsort(-last)[:5].tolist()} llama.cpp {np.argsort(-r[-1])[:5].tolist()}; "
              f"logp of llama.cpp top1: ours {last[r[-1].argmax()]:.4f} ref {r[-1].max():.4f}")
    print(allp.line())
    if allp.miss:
        print(f"top-1 misses: llama.cpp's own gap between its top two tokens (nats): {allp.miss}")
    if tap and found:
        print(clean.line("positions before the first near-tie (ties and what follows them excluded):"), f"[{n_tied_chunks} of {toks.shape[0]} chunk(s) hold a near-tie; {allp.n - clean.n} position(s) excluded]")


if __name__ == "__main__":
    main()
