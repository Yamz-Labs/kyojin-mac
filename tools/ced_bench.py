#!/usr/bin/env python3
"""CED prefill timing, per layer, exact against CED, and a 40-layer / 384-expert extrapolation.

On a 2-layer slice (models/ds2, run as an encoder-less CED model: --mid 0 and the ratio-1 overrides of tools/ced_gate.py) the decoder half is one layer
deep, so only per-layer numbers are meaningful. They are measured with a synchronising wrapper around every block and every MoE (so a layer's time is
its own), then combined:
  F(R)   MoE time of a chunk of R rows. Two paths (patch.py _moe_rows): below KYOJIN_PF_CROSS * E / k rows (ds2: 74, the real 384-expert model: 896)
         the routed experts run in chunks of 4 rows through the decode kernels (c4 ms per 4 rows, measured on the slice at R < 74: the experts are the
         real size, 6 per token, so the cost per 4 rows is the real one); from there on the S3 path: F = a + b R, a = decode of all the slice's experts
         (fit on R >= 128, all touched). Real model: F_real(R) = c4 R / 4 below 896 rows, else a / E_slice * touched(R) + b R with
         touched(R) = E (1 - (1 - k/E)^R). A 128-row tail therefore costs 32 decode-kernel calls per layer, not an S3 prefill.
  rest(R) block time minus MoE time (attention, compressor, hyper-connections), linear fit on the measured R >= 128.
  exact chunk  = 40 layers x (rest(C) + F_real(C))
  CED chunk    = 20 encoder layers x (rest(C) + F_real(C)) + midpoint (measured, MoE on `tail` rows re-priced) + 19 x (rest(tail) + F_real(tail))
The encoder layers are priced like a decoder layer of the slice (the slice has no ratio-2 layer; they do more compressor work, so the exact figure is a
floor on cost and the gain estimate is slightly conservative). This is a model of the cost, not a measurement of the 40-layer pack: the day-1 number
is the gate's tok/s (tools/ced_gate.py) on the real pack.

usage: ced_bench.py MODEL_DIR --mid 0 --text-overrides JSON [--T 4096] [--chunk 2048] [--tail 128] [--curve 16,64,128,256,512,1024] [--reps 3] [--model-chunk 4096]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from kyojin_mlx import ced as C  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model")
ap.add_argument("--mid", type=int)
ap.add_argument("--text-overrides", default="")
ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--tail", type=int, default=None)
ap.add_argument("--curve", default="16,64,128,256,512,1024")
ap.add_argument("--model-chunk", type=int, default=4096, help="serving chunk the 40-layer model is priced at")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--experts", type=int, default=384, help="routed experts of the real model")
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--dump")


def instrument(lm, rec):
    """Synchronising timers on every block and every MoE."""
    def timed(name):
        def fn(self, orig, *a, **k):
            mx.synchronize()
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            mx.eval(out)
            mx.synchronize()
            rec.setdefault(name, []).append(time.perf_counter() - t0)
            return out
        return fn

    for i, blk in enumerate(lm.layers):
        C._override(blk, "__call__", timed(f"blk{i}"))
        C._override(blk.ffn, "__call__", timed(f"ffn{i}"))


def run(lm, ids, chunk, ced_on):
    cache = lm.make_cache()
    t0 = time.perf_counter()
    for s in range(0, len(ids), chunk):
        out = lm(mx.array([ids[s:s + chunk]]), cache=cache)
        mx.eval(out.logits, *cache[0].state)
    mx.synchronize()
    return time.perf_counter() - t0


def median(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    a = ap.parse_args()
    import ced_gate as G
    G.a = argparse.Namespace(model=a.model)
    overrides = json.loads(a.text_overrides) if a.text_overrides else None
    model, spec, cfg, tables = G.load(a.model, overrides)
    lm = model.language_model
    tc = cfg.get("text_config", cfg)
    if overrides:
        tc = {**tc, **overrides}
    vocab, E_slice, k = tc["vocab_size"], tc["n_routed_experts"], tc["num_experts_per_tok"]
    C.enable(model, spec, tail=a.tail, mid=a.mid)
    st = C.state(lm)
    mid, n_layers, tail = st.lay.mid, st.lay.n_layers, st.tail
    print(f"load peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB; mid={mid}/{n_layers} window={st.lay.window} tail={tail} chunk={a.chunk} T={a.T} E={E_slice} k={k}", flush=True)
    mx.reset_peak_memory()
    rng = np.random.default_rng(0)
    ids_all = [int(x) for x in rng.integers(3, vocab, max(a.T, 4096))]
    rec: dict = {}

    # ---- uninstrumented totals, exact and CED
    tot = {}
    for name, on in (("exact", False), ("ced", True)):
        C.set_on(lm, on)
        ts = [run(lm, ids_all[:a.T], a.chunk, on) for _ in range(a.reps + 1)][1:]
        tot[name] = min(ts)
    print(f"total prefill T={a.T} chunk={a.chunk}: exact {tot['exact']:.2f} s ({a.T / tot['exact']:.0f} tok/s)  ced {tot['ced']:.2f} s ({a.T / tot['ced']:.0f} tok/s)  "
          f"ratio {tot['exact'] / tot['ced']:.2f}  peak {mx.get_peak_memory() / 2 ** 30:.2f} GiB", flush=True)

    # ---- instrumented per-layer
    instrument(lm, rec)
    per = {}
    for name, on in (("exact", False), ("ced", True)):
        C.set_on(lm, on)
        rec.clear()
        for _ in range(a.reps + 1):
            run(lm, ids_all[:a.T], a.chunk, on)
        per[name] = {k_: median(v[len(v) // (a.reps + 1):]) for k_, v in rec.items()}      # drop the warm-up repetition
    print("per-layer ms per chunk (median):", flush=True)
    for name in ("exact", "ced"):
        print(f"  {name:5s}", {k_: round(v * 1000, 1) for k_, v in sorted(per[name].items())}, flush=True)

    # ---- MoE curve (exact, one chunk of R rows), minimum over the repetitions
    C.set_on(lm, False)
    last = n_layers - 1
    curve = {}
    for R in [int(x) for x in a.curve.split(",") if x]:
        rec.clear()
        for _ in range(a.reps + 1):
            run(lm, ids_all[:R], R, False)
        curve[R] = dict(ffn=min(rec[f"ffn{last}"][1:]), blk=min(rec[f"blk{last}"][1:]))
    pf_slice = int(float(os.environ.get("KYOJIN_PF_CROSS", "14")) * E_slice / k)          # rows from which the slice runs the S3 prefill path
    pf_real = int(float(os.environ.get("KYOJIN_PF_CROSS", "14")) * a.experts / k)
    small = sorted(r for r in curve if r < pf_slice)
    big = sorted(r for r in curve if r >= pf_slice)
    c4 = float(np.median([curve[r]["ffn"] / (r / 4) for r in small])) if small else float("nan")
    b_, a_ = np.polyfit(np.array(big, dtype=float), np.array([curve[r]["ffn"] for r in big]), 1) if len(big) > 1 else (0.0, curve[big[0]]["ffn"])
    print("MoE F(R) ms:", {int(r): round(curve[r]["ffn"] * 1000, 2) for r in sorted(curve)}, flush=True)
    print(f"  chunk path (rows < {pf_slice} on the slice, < {pf_real} on the real model): {c4 * 1000:.2f} ms per 4 rows; S3 path fit a={a_ * 1000:.1f} ms "
          f"(all {E_slice} experts decoded) b={b_ * 1e6:.1f} us/row", flush=True)
    big_r = np.array([r for r in sorted(curve) if r >= 128], dtype=float)
    rest_pts = np.array([curve[int(r)]["blk"] - curve[int(r)]["ffn"] for r in big_r])
    rb, ra = np.polyfit(big_r, rest_pts, 1)
    print("rest(R) ms:", {int(r): round(x * 1000, 2) for r, x in zip(big_r, rest_pts)}, f"linear fit {ra * 1000:.1f} + {rb * 1e6:.1f} us/row", flush=True)

    def touched(E, R):
        return E * (1 - (1 - k / E) ** R)

    def F_real(R):
        if R < pf_real:
            return c4 * R / 4
        return a_ / E_slice * touched(a.experts, R) + b_ * R

    def rest_at(R):
        return ra + rb * R

    C_ = a.model_chunk
    N = a.layers
    exact_chunk = N * (rest_at(C_) + F_real(C_))
    half = N // 2
    # measured midpoint block (MoE on the tail rows, priced as on the slice) re-priced for the real model: swap its MoE for F_real(tail)
    slice_tail_ffn = per["ced"][f"ffn{mid}"]
    mid_blk = per["ced"][f"blk{mid}"] - slice_tail_ffn + F_real(tail)
    tail_layer = rest_at(tail) + F_real(tail)
    ced_chunk = half * (rest_at(C_) + F_real(C_)) + mid_blk + (N - half - 1) * tail_layer
    print(f"MODEL ({N} layers, {a.experts} experts, chunk {C_}, tail {tail}): exact {exact_chunk * 1000:.0f} ms/chunk = {C_ / exact_chunk:.0f} tok/s ; "
          f"CED {ced_chunk * 1000:.0f} ms/chunk = {C_ / ced_chunk:.0f} tok/s ; gain x{exact_chunk / ced_chunk:.2f}", flush=True)
    print(f"  one decoder layer: exact at {C_} rows {1000 * (rest_at(C_) + F_real(C_)):.1f} ms (rest {rest_at(C_) * 1000:.1f} + MoE {F_real(C_) * 1000:.1f}); "
          f"CED at {tail} rows {tail_layer * 1000:.1f} ms (rest {rest_at(tail) * 1000:.1f} + MoE {F_real(tail) * 1000:.1f}); midpoint {mid_blk * 1000:.1f} ms", flush=True)
    out = dict(total=tot, per_layer=per, curve={int(r): v for r, v in curve.items()}, model=dict(
        chunk=C_, tail=tail, layers=N, experts=a.experts, exact_ms=exact_chunk * 1000, ced_ms=ced_chunk * 1000, gain=exact_chunk / ced_chunk,
        c4_ms=c4 * 1000, s3_a_ms=a_ * 1000, s3_b_us=b_ * 1e6, rest_a_ms=ra * 1000, rest_b_us=rb * 1e6))
    if a.dump:
        json.dump(out, open(a.dump, "w"), indent=1)
    if tables is not None:
        tables.close()


main()
