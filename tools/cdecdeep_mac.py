"""cdecdeep: before / after of dec_keys_codes on Metal, one index layer of the decode core at depth, synthetic packed KV, no model, no tokens (cdecdeep).

What it times (the term of `dsv4_core.core` that a packed key layer pays at every step from ISCORE_MMA_MIN = 262144 keys on):
  off     flag 0 in a packed configuration: `CD.dense_of(keys)` (fp4pack.unpack of the WHOLE layer, all slots) + `dsa_decode.scores_rows` (the NAX form when the device has it)
  on      flag 1: `keyscodes.scores` -> `dsa_decode.scores_rows_codes` (codes read in the kernel, 8-key tile unpacked in threadgroup memory; the first call runs the self check)
  dense   `scores_rows` over a dense layer that already exists (what a dense stack / mirrorq pays: no unpack at all) - the floor the unpack-free path is measured against
For every cell it prints ms (median, min of --reps, the three arms interleaved inside every repetition), and checks
  on == scores_rows(nax=False) over the unpacked keys, BIT FOR BIT, on the columns < lens   (the kernel's contract; a mismatch prints MISMATCH and the run exits 3)
  on vs off: largest |difference|, share of equal scores, and the overlap of the top-k (idxfast.topk_mt, k = --topk) the two score sets select (the head-sum order of the
             NAX and the simdgroup forms differs, so a near tie may flip: rounding-level drift, expected small, never a different row layout)
Rows: R rows of every session (a plain step R = 1, a verify round R > 1); sessions S > 1 live in slots S-1 .. 0 of an S + 1 slot stack (mu_keep: a reversed, non-identity slot map
and a spare slot, `sess` = the slot of the row like Plan.sess), S = 1 is the cdecpack view (one slot). kind `plain` scores every key, kind `mask` is a consumer layer (a
random 1/8 of the 8-key blocks kept, like the candidate source's mask). Keys per session = depth (the ratio-1 source layer holds one key per token).

Bounded: mx.set_memory_limit(3.5 GiB) and the peak (mx.get_peak_memory) is checked after every cell; a cell whose modelled peak (2.4 KB per key of the off arm's unpack
temporaries + the codes) would not fit is skipped with a printed line, not run. Tokens: none are generated; R rows stand for the <= 24 verify rows of a round.

usage (Mac, GPU idle):
  PYTHONPATH=. python tools/cdecdeep_mac.py --json scratch/cdecdeep_mac.json
  PYTHONPATH=. python tools/cdecdeep_mac.py --depths 262144,524288 --rows 1,6 --sessions 1,2 --kinds plain,mask --reps 7
  PYTHONPATH=. python tools/cdecdeep_mac.py --smoke        # any device: builds the data and checks the script, runs no kernel without Metal
Read it: `off` is today's cost in a packed configuration, `on` the lever, `dense` the unpack-free floor; the lever pays when on < off and on is not far above dense.
Nothing here is a served rate. The lever only engages where the core holds a PACKED key layer: one session with KYOJIN_CDECPACK=1 KYOJIN_MU_KEEP=0 (cdecpack needs a non-slot
stack), or KYOJIN_STACKPACK=1 with 2+ sessions (any mu_keep). Served A/B at the same depth: KYOJIN_DEC_KEYS_CODES=0 against 1 on that configuration; it engaged when the fired counter
`dec_keys_codes` and keyscodes.HITS["codes"] are > 0 with HITS["selfcheck_fail"] == 0 (cdecpack.HITS["key_dense"] keeps counting: the DCS.layer argument builds a lazy, never evaluated unpack)."""
import argparse
import gc
import json
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

LIMIT = int(3.5 * 2 ** 30)
NH, DI = 32, 128
ap = argparse.ArgumentParser()
ap.add_argument("--depths", default="262144,524288,1048576", help="keys per session (= tokens of context for the ratio-1 source layer)")
ap.add_argument("--rows", default="1,6", help="rows per session in the pass")
ap.add_argument("--sessions", default="1,2")
ap.add_argument("--kinds", default="plain,mask")
ap.add_argument("--topk", type=int, default=512)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--no-lock", action="store_true")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()

METAL = mx.metal.is_available()
if a.smoke:
    a.depths, a.rows, a.sessions, a.reps, a.warm, a.topk = "4096,8192", "1,2", "1,2", 2, 1, 32
mx.set_memory_limit(LIMIT)
RESULTS = []
BAD = []
LOCK = None
if METAL and not a.no_lock and not a.smoke:
    import fcntl
    LOCK = open("/tmp/ultra_heavy.lock", "w")
    fcntl.flock(LOCK, fcntl.LOCK_EX)


def peak_check(tag):
    p = mx.get_peak_memory()
    if p > LIMIT:
        print(f"ABORT {tag}: peak {p / 2 ** 30:.2f} GiB above the 3.5 GiB bound", flush=True)
        sys.exit(2)
    return p


def clean():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def modelled_peak(slots, depth):
    """Bytes: codes + scales (68 B per key per slot), the dense layer the off arm builds (256 B) and its elementwise temporaries (~2.2 KB per key: int32 index, fp32 magnitude,
    value, product), the dense floor arm's layer (256 B)."""
    return slots * depth * (68 + 256 + 2200 + 256)


def make_codes(slots, depth, seed):
    rng = np.random.default_rng(seed)
    codes = mx.array(rng.integers(0, 256, size=(slots, depth, DI // 2), dtype=np.uint8))
    scales = mx.array(rng.integers(120, 128, size=(slots, depth, DI // 32), dtype=np.uint8))       # 2^-7 .. 1 : fp16-safe products
    mx.eval(codes, scales)
    return codes, scales


def make_rows(S, R, depth, seed, kind):
    rng = np.random.default_rng(seed)
    n = S * R
    q = mx.array((0.5 * rng.standard_normal((n, NH, DI))).astype(np.float16))
    w = mx.array((0.05 * np.abs(rng.standard_normal((n, NH)))).astype(np.float16))
    lens = mx.array([depth - R + 1 + (u % R) for u in range(n)], dtype=mx.int32)
    slots = list(range(S - 1, -1, -1))                                       # session s lives in slot S - 1 - s (non-identity for S > 1); slot S is the spare
    sess = mx.array([slots[u // R] for u in range(n)], dtype=mx.uint32)
    cand = None
    if kind == "mask":
        blocks = rng.random((n, depth // 8 + 1)) < 0.125
        cand = mx.array(np.repeat(blocks, 8, axis=1)[:, :depth])
    mx.eval(q, w, lens, sess, *([] if cand is None else [cand]))
    return q, w, lens, sess, cand


def timed(fns, reps, warm):
    """{name: (median ms, min ms)}; the arms alternate inside every repetition so drift hits all of them alike."""
    for _ in range(warm):
        for f in fns.values():
            mx.eval(f())
    ts = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            t = time.perf_counter()
            mx.eval(f())
            ts[k].append((time.perf_counter() - t) * 1e3)
    return {k: (statistics.median(v), min(v)) for k, v in ts.items()}


def valid_mask(lens, T):
    ln = np.array(lens)
    return np.arange(T)[None, :] < ln[:, None]


def cell(S, R, depth, kind):
    from kyojin_mlx import dsa_decode as DSA, fp4pack, keyscodes as KC, stackpack as SP
    from kyojin_mlx.fp4pool import PackedView
    NS = 1 if S == 1 else S + 1
    est = modelled_peak(NS, depth)
    tag = f"S={S} R={R} L={depth} {kind}"
    if est > LIMIT * 0.85:
        print(f"skip {tag}: modelled peak {est / 2 ** 30:.2f} GiB would not fit the 3.5 GiB bound (not run)", flush=True)
        return
    clean()
    codes, scales = make_codes(NS, depth, 1)
    q, w, lens, sess, cand = make_rows(S, R, depth, 2, kind)
    keys = PackedView(codes, scales, depth, mx.float16, None, 32, False) if NS == 1 else SP.PackedStack(codes, scales, mx.float16, 32, False)
    KC.ENABLED = True
    if not METAL:
        print(f"{tag}: no Metal, data built ({est / 2 ** 20:.0f} MiB modelled), no kernel run", flush=True)
        return
    on_fn = lambda: KC.scores(q, w, keys, lens, depth, cand, sess)
    if on_fn() is None:
        print(f"{tag}: keyscodes.scores returned None (self check failed or gate closed): {KC.HITS}", flush=True)
        BAD.append(tag + " gate")
        return
    off_fn = lambda: DSA.scores_rows(q, w, fp4pack.unpack(codes, scales, 32, False, mx.float16), lens, depth, cand, sess)
    dense = fp4pack.unpack(codes, scales, 32, False, mx.float16)
    mx.eval(dense)
    dense_fn = lambda: DSA.scores_rows(q, w, dense, lens, depth, cand, sess)
    ms = timed({"off": off_fn, "on": on_fn, "dense": dense_fn}, a.reps, a.warm)
    # exactness of the kernel against the simdgroup form over the same unpacked keys, then the drift against the flag-0 (NAX) values
    on, off = on_fn(), off_fn()
    ref = DSA.scores_rows(q, w, dense, lens, depth, cand, sess, nax=False)
    mx.eval(on, off, ref)
    vm = valid_mask(lens, depth)
    onn, offn, refn = (np.array(x.astype(mx.float32)) for x in (on, off, ref))
    bit = bool(np.array_equal(np.where(vm, onn, 0), np.where(vm, refn, 0)))
    both = vm & np.isfinite(onn) & np.isfinite(offn)
    mx_diff = float(np.abs(onn - offn)[both].max()) if both.any() else 0.0
    same = float((onn == offn)[vm].mean())
    from kyojin_mlx import idxfast as IX
    k = min(a.topk, depth // 4)
    ti, to = np.array(IX.topk_mt(on, lens, k)), np.array(IX.topk_mt(off, lens, k))
    ov = float(np.mean([len(set(x[x >= 0]) & set(y[y >= 0])) / max(1, len(set(y[y >= 0]))) for x, y in zip(ti, to)]))
    peak = peak_check(tag)
    row = dict(S=S, R=R, L=depth, kind=kind, off_ms=ms["off"][0], off_min=ms["off"][1], on_ms=ms["on"][0], on_min=ms["on"][1], dense_ms=ms["dense"][0], dense_min=ms["dense"][1],
               on_bit_equal_simdgroup=bit, max_abs_diff_vs_off=mx_diff, equal_share_vs_off=same, topk_overlap_vs_off=ov, peak_gib=peak / 2 ** 30)
    RESULTS.append(row)
    print(f"{tag:<26} off {ms['off'][0]:9.3f} (min {ms['off'][1]:8.3f})  on {ms['on'][0]:9.3f} (min {ms['on'][1]:8.3f})  dense {ms['dense'][0]:9.3f} (min {ms['dense'][1]:8.3f}) ms"
          f"  on/off {ms['on'][0] / ms['off'][0]:.2f}  bit-equal to simdgroup: {bit}  |d|max vs off {mx_diff:.4g}  equal {same:.4f}  top{k} overlap {ov:.4f}  peak {peak / 2 ** 30:.2f} GiB",
          flush=True)
    if not bit:
        print(f"MISMATCH {tag}: scores_rows_codes differs from scores_rows(nax=False) over the unpacked keys", flush=True)
        BAD.append(tag + " mismatch")
    del codes, scales, keys, dense, on, off, ref


def main():
    print(f"cdecdeep_mac metal={METAL} limit={LIMIT / 2 ** 30:.1f} GiB depths={a.depths} rows={a.rows} sessions={a.sessions} kinds={a.kinds}", flush=True)
    for depth in [int(x) for x in a.depths.split(",")]:
        for S in [int(x) for x in a.sessions.split(",")]:
            for R in [int(x) for x in a.rows.split(",")]:
                for kind in a.kinds.split(","):
                    cell(S, R, depth, kind)
    from kyojin_mlx import keyscodes as KC
    print("keyscodes.HITS", KC.HITS, flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(results=RESULTS, bad=BAD, hits=KC.HITS), f, indent=1)
    if BAD:
        print("FAIL", BAD, flush=True)
        sys.exit(3)


main()
