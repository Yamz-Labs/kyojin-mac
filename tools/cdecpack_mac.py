#!/usr/bin/env python3
"""cdecpack on Metal: KYOJIN_CDECPACK 0 against 1 on the tiny random DS4.1 model (tests/mini_ds41.py) built with the DS4.1 attention shape (head_dim 512, 8 heads, fp16, so the
packed pool is D=512: 256 B codes + 32 B scales = 288 B per row; the index keys are D=32 here, 68 B per row only at the real D=128).
Needs mlx + mlx-vlm, Metal for the fused paths. Bounded inside the tool: mx.set_memory_limit(3.5 GiB), the run stops if the peak passes it, --tokens <= 4096, --rows <= 8.

    python3 tools/cdecpack_mac.py [--tokens 2048] [--steps 12] [--rows 4] [--cpu]

Per arm (flag 0, flag 1), a fresh prefilled cache (256-row chunks, fp4pool.MEMO_BYTES = 0 so every pool layer reads back as a PackedView whatever its size):
  A. single-session decode, --steps tokens through the model (fused core: the packed attend, which already reads the pool without a mirror: a control, the flag must not move it);
  B. chained verify of --rows tokens (spec.verify, --steps times from the same base cache): the reuse layers go through fuse_attn.core_rows, where the flag acts. Metal only.
Prints per arm: pool mirror bytes retained in the cache (what a dense() read keeps), codes + scales bytes, bytes per pool row, depthflat.HITS, cdecpack.HITS, MLX peak, median op ms
(a tiny model on Metal: read it as 'no slowdown', never as a gain). Exit 1 if any logits row differs bitwise between the two arms, 2 (NOT ENGAGED) if arm B ran with the flag on
and cdecpack.compact never fired (the lever does nothing on this build: report that), 3 if arm B raised. --cpu runs arm A only on the stock path (equality only, nothing fused).
Nothing here is a claim about the real DS4.1 pack: the formula for it is in docs/reports/cdecpack_REPORT.md."""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tests"), os.path.join(HERE, "..")]
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

GIB = 2 ** 30
LIMIT = int(3.5 * GIB)


def guard():
    if mx.get_peak_memory() > LIMIT:
        sys.exit("peak above 3.5 GiB: stopped")


class Wrap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


def retained(cache):
    """Bytes the pool slots of `cache` keep: packed codes + scales (valid rows), dense mirror, memo; and the pool rows."""
    out = {"packed": 0, "mirror": 0, "memo": 0, "rows": 0}
    for c in cache:
        sl = getattr(c, "compress", None)
        if sl is None:
            continue
        for e in list.__iter__(sl):
            if hasattr(e, "codes"):
                out["packed"] += e.codes[:, :e.n].nbytes + e.scales[:, :e.n].nbytes
                out["rows"] += e.n
                out["mirror"] += 0 if e.mirror is None else e.mirror.nbytes
                out["memo"] += 0 if e.memo is None else e.memo.nbytes
    return out


def arm(flag, args, metal, M, fp4pool, DF, CD, FL, A, R, S, lm, model):
    FL.set("cdecpack", str(flag))
    for k in DF.HITS:
        DF.HITS[k] = 0
    for k in CD.HITS:
        CD.HITS[k] = 0
    out = {"logits_a": [], "logits_b": [], "ms_a": [], "ms_b": [], "err": None}
    cache = fp4pool.wrap_cache(lm.make_cache())
    ids = M.tokens(args.tokens, seed=1)
    for s in range(0, args.tokens, 256):
        mx.eval(lm(mx.array([ids[s:s + 256]]), cache=cache).logits, *cache[0].state)
        guard()
    out["after_prefill"] = retained(cache)
    if args.rows and metal:
        try:
            for i in range(args.steps):
                toks = [5 + i + j for j in range(args.rows)]
                t = time.perf_counter()
                lg, _, _ = S.verify(model, cache, toks)
                mx.eval(lg)
                out["ms_b"].append((time.perf_counter() - t) * 1e3)
                out["logits_b"].append(np.array(lg))
                guard()
        except Exception as ex:                                       # noqa: BLE001  (report, do not hide)
            out["err"] = repr(ex)
    out["after_verify"] = retained(cache)
    tok = 5
    for _ in range(args.steps):
        t = time.perf_counter()
        lg = lm(mx.array([[tok]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg)
        out["ms_a"].append((time.perf_counter() - t) * 1e3)
        out["logits_a"].append(np.array(lg))
        tok = int(mx.argmax(lg).item())
        guard()
    out["after_decode"] = retained(cache)
    out["df"], out["cd"] = dict(DF.HITS), dict(CD.HITS)
    out["peak"] = mx.get_peak_memory()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if not (0 < args.tokens <= 4096 and 0 <= args.rows <= 8 and args.steps > 0):
        sys.exit("--tokens <= 4096, --rows <= 8")
    mx.set_memory_limit(LIMIT)
    metal = mx.default_device() == mx.gpu and mx.metal.is_available()
    import mini_ds41 as M
    from kyojin_mlx import archs as A, cdecpack as CD, depthflat as DF, flags as FL, fp4pool, rows as R, spec as S
    fp4pool.MEMO_BYTES = 0
    lm, _ = M.make(dtype=mx.float16, sliding_window=16, head_dim=512, num_attention_heads=8)
    model = Wrap(lm)
    if metal:
        from kyojin_mlx import fuse
        print("fuse.install:", fuse.install(A.DS41))
        R.enable(model, A.DS41, grouped=False)
        S.install(model, A.DS41)
    res = {}
    for flag in (0, 1):
        mx.clear_cache()
        mx.reset_peak_memory()
        res[flag] = arm(flag, args, metal, M, fp4pool, DF, CD, FL, A, R, S, lm, model)
    rc = 0
    for flag, r in res.items():
        print("\nflag %d" % flag)
        for k in ("after_prefill", "after_verify", "after_decode"):
            b = r[k]
            print("  %-13s pool rows %-7d packed %9d B (%5.1f B/row)  mirror %10d B (%6.1f B/row)  memo %d B" % (
                k, b["rows"], b["packed"], b["packed"] / max(b["rows"], 1), b["mirror"], b["mirror"] / max(b["rows"], 1), b["memo"]))
        print("  depthflat.HITS", r["df"], " cdecpack.HITS", r["cd"], " MLX peak %.1f MiB" % (r["peak"] / 2 ** 20))
        for n in ("a", "b"):
            if r["ms_" + n]:
                print("  op ms median arm %s: %.2f over %d" % (n.upper(), statistics.median(r["ms_" + n][2:] or r["ms_" + n]), len(r["ms_" + n])))
        if r["err"]:
            print("  ARM B RAISED:", r["err"])
    for n in ("a", "b"):
        for i, (x, y) in enumerate(zip(res[0]["logits_" + n], res[1]["logits_" + n])):
            if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
                print("DIFFERENT logits arm %s step %d: max |d| %.3g" % (n.upper(), i, float(np.abs(x - y).max())))
                rc = 1
                break
    if rc == 0:
        print("\nlogits bit-identical flag 0 against 1: arm A %d steps, arm B %d steps" % (len(res[0]["logits_a"]), len(res[0]["logits_b"])))
    if any(r["err"] for r in res.values()):
        rc = rc or 3
    elif args.rows and metal and res[1]["cd"]["compact"] == 0:
        print("NOT ENGAGED: cdecpack.compact never fired on arm B with the flag on")
        rc = rc or 2
    elif args.rows and metal:
        d = res[0]["after_verify"]["mirror"] - res[1]["after_verify"]["mirror"]
        print("pool mirror kept after the verify steps: flag 0 %d B, flag 1 %d B (saved %d B)" % (res[0]["after_verify"]["mirror"], res[1]["after_verify"]["mirror"], d))
    sys.exit(rc)


if __name__ == "__main__":
    main()
