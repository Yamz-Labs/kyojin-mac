"""Prefill flatness past 128K: correctness gate first, then ms per op of the old and the new indexer path at real geometry (32 heads x 128, top-k 512, 2048 candidate
blocks of 8 keys), synthetic fp16 data, one layer, `rows` query rows (scaled to an 8192-row chunk in the last table). Needs the Metal GPU and NAX (like fscore).

    python tools/bench_pf_flat.py                       # Tc 128K, 256K, 512K, 1M; < 60 s; peak memory printed and asserted < 3.5 GiB
    python tools/bench_pf_flat.py --tc 262144,1048576 --reps 5
    python tools/bench_pf_flat.py --check-only

Arms (each flag is flipped on the module literal, interleaved, one process):
  old     fscore -> [rows, Tc] scores -> topk_block / select_candidate_blocks (or dsv4_core.candidate_mask_rt without the vendor module) -> E1 cand_topk
  bmax    KYOJIN_PF_IDX_BMAX: block-max kernel -> pick_blocks -> gathered rescoring
  +gscore KYOJIN_PF_IDX_GSCORE: the rescoring in one kernel
Gate (PASS / FAIL, exit status 1 on any FAIL): bmax kernel == fscore scores reshaped (blocks, 8) and maxed, bitwise; gscore == fscore columns, bitwise; plain, source and
consumer indices == the old path; source block mask == the old mask; cand ids from the mask == the old consumer's blocks; select_fused == vendor select + topk."""
import argparse
import sys
import time

import numpy as np

import mlx.core as mx

from kyojin_mlx import ds_pf_idx as PI
from kyojin_mlx import idx_topk as IT
from kyojin_mlx import pf_flat as PF

NH, D, BS, TOPK, KB = 32, 128, 8, 512, 2048
GIB = 2 ** 30
FAILS = []


def gate(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def med(fn, reps):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def data(rows, Tc, seed):
    mx.random.seed(seed)
    q = mx.random.normal((1, rows, NH, D)).astype(mx.float16)
    k = mx.random.normal((1, Tc, D)).astype(mx.float16)
    w = (mx.random.normal((1, rows, NH)) * 0.05).astype(mx.float16)
    mx.eval(q, k, w)
    return q, k, w


def vendor_select():
    try:
        import importlib
        return importlib.import_module("mlx_vlm.models.deepseek_v41.language").select_candidate_blocks
    except Exception:
        return None


def old_select(s, clen, vendor):
    """(mask [n, Tc] bool, idx [n, TOPK]) of the stock source path on one score tile."""
    if vendor is not None:
        return vendor(s[None], clen[:, None], KB, BS)[0], PI.topk_block(s, TOPK)
    from kyojin_mlx import dsv4_core as C
    return C.candidate_mask_rt(s, clen.astype(mx.int32), KB, BS), PI.topk_block(s, TOPK)


def rows_for(Tc):
    per_row = Tc * 8 + KB * BS * 256 * 2                        # scores + where copy, gathered keys + product of the old consumer
    return int(max(32, min(256, (2.4 * GIB) // per_row)) // 8 * 8)


def correctness(q, k, w, q2, k2, w2, start, ratio, rows, Tc, vendor):
    print(f" gate, rows={rows}, Tc={Tc}", flush=True)
    clen = (start + mx.arange(rows) + 1) // ratio
    nblk = -(-Tc // BS)
    s = PI.scores_range(q, k, w, start, ratio, 0, rows)
    bm = PF.bmax_scores(q, k, w, start, ratio, 0, rows)
    ref_bm = s.reshape(rows, nblk, BS).max(-1) if Tc % BS == 0 else None
    gate("bmax kernel == fscore reshaped (blocks, 8) max, bitwise", ref_bm is not None and bool(mx.array_equal(bm, ref_bm).item()))
    ref_mask, ref_idx = old_select(s, clen, vendor)
    ids = PF.pick_blocks(bm, clen, KB, True)
    got_mask = PF.mask_from_ids(ids, nblk, Tc)
    gate("source block mask == " + ("vendor select_candidate_blocks" if vendor is not None else "candidate_mask_rt"), bool(mx.array_equal(got_mask, ref_mask).item()))
    PF.GSCORE = True
    sel = PF.select_from_blocks(q, k, PF.key_blocks(k), w, ids, start, ratio, 0, TOPK)
    gate("source top-k (bmax + gscore) == stock top-k", bool(mx.array_equal(sel, ref_idx).item()))
    gs = PF.gathered_scores(q, k, PF.key_blocks(k), w, ids, start, ratio, 0)
    cols = mx.minimum(PF._positions(ids), Tc - 1).astype(mx.int32)
    full = mx.take_along_axis(s, cols, axis=1)
    full = mx.where(PF._positions(ids) < Tc, full, -mx.inf)
    gate("gscore == fscore columns, bitwise", bool(mx.array_equal(gs, full).item()), f"(gscore used: {PF.HITS['gscore'] > 0}, fallback: {PF.HITS['gscore_fallback']})")
    PF.GSCORE = False
    sel0 = PF.select_from_blocks(q, k, PF.key_blocks(k), w, ids, start, ratio, 0, TOPK)
    gate("gathered-copy rescoring == fscore columns, bitwise", bool(mx.array_equal(PF.gathered_scores(q, k, PF.key_blocks(k), w, ids, start, ratio, 0), full).item()))
    gate("source top-k (bmax, gathered copy) == stock top-k", bool(mx.array_equal(sel0, ref_idx).item()))
    PF.BMAX, PF.BMAX_MIN = True, 0
    plain_new = PF.plain_topk(q, k, w, start, ratio, TOPK, rows)
    plain_old = IT.scan(lambda t0, n: PI.scores_range(q, k, w, start, ratio, t0, n), rows, IT.qblock(rows), lambda t0, sc: PI.topk_block(sc, TOPK))
    gate("plain layer top-k (bmax) == stock", bool(mx.array_equal(plain_new, plain_old).item()))
    ids_old = PF.ids_from_mask(ref_mask[:, ::BS], KB)
    gate("cand ids from the stock mask == the blocks the consumer's argsort keeps", bool(mx.array_equal(ids_old, mx.where(ids >= 0, ids, -1)).item()))
    cand = ref_mask[None]
    cons_old = PI.cand_topk(q2, k2, w2, start, ratio, 0, rows, cand, BS, KB, TOPK)
    for tag, g in (("gathered copy", False), ("gscore", True)):
        PF.GSCORE = g
        gate(f"consumer top-k from ids ({tag}) == E1 cand_topk", bool(mx.array_equal(PF.consumer_topk(q2, k2, w2, start, ratio, TOPK, rows, ids), cons_old).item()))
    PF.GSCORE = False
    if PF.METAL_OPT:
        from kyojin_mlx import idxfast as IX
        fi, fm = IX.select_fused(s, clen.astype(mx.int32), TOPK, KB, BS)
        gate("select_fused (selfuse) mask + idx == stock", bool(mx.array_equal(fm, ref_mask).item()) and bool(mx.array_equal(fi, ref_idx).item()))
    del s, bm, full
    mx.clear_cache()


def timing(q, k, w, q2, k2, w2, start, ratio, rows, Tc, reps, vendor):
    clen = (start + mx.arange(rows) + 1) // ratio
    nblk = -(-Tc // BS)
    kb = PF.key_blocks(k)
    out = {}
    s = PI.scores_range(q, k, w, start, ratio, 0, rows)
    mx.eval(s)
    out["fscore (scores to memory)"] = med(lambda: PI.scores_range(q, k, w, start, ratio, 0, rows), reps)
    out["bmax kernel (block maxima only)"] = med(lambda: PF.bmax_scores(q, k, w, start, ratio, 0, rows), reps)
    out["old plain select: topk_block"] = med(lambda: PI.topk_block(s, TOPK), reps)
    out["old source select: vendor/rt mask + topk_block"] = med(lambda: old_select(s, clen, vendor), reps)
    bm = PF.bmax_scores(q, k, w, start, ratio, 0, rows)
    mx.eval(bm)
    out["new plain pick_blocks(512)"] = med(lambda: PF.pick_blocks(bm, clen, TOPK, False), reps)
    out["new source pick_blocks(2048, forced last)"] = med(lambda: PF.pick_blocks(bm, clen, KB, True), reps)
    ids_p = PF.pick_blocks(bm, clen, TOPK, False)
    ids_s = PF.pick_blocks(bm, clen, KB, True)
    mx.eval(ids_p, ids_s)
    for tag, g in (("gathered copy", False), ("gscore", True)):
        PF.GSCORE = g
        out[f"new rescoring 512 blocks ({tag})"] = med(lambda: PF.select_from_blocks(q, k, kb, w, ids_p, start, ratio, 0, TOPK), reps)
        out[f"new rescoring 2048 blocks ({tag})"] = med(lambda: PF.select_from_blocks(q2, k2, PF.key_blocks(k2), w2, ids_s, start, ratio, 0, TOPK), reps)
    PF.GSCORE = False
    mask, _ = old_select(s, clen, vendor)
    cand = mask[None]
    mx.eval(cand)
    out["old consumer: E1 cand_topk (mask strided read + argsort Tc/8 + gather)"] = med(lambda: PI.cand_topk(q2, k2, w2, start, ratio, 0, rows, cand, BS, KB, TOPK), reps)
    out["new make_ids from the mask (once per source layer)"] = med(lambda: PF.make_ids(mask, KB), reps)
    if PF.METAL_OPT:
        from kyojin_mlx import idxfast as IX
        out["selfuse: select_fused (top-k + mask)"] = med(lambda: IX.select_fused(s, clen.astype(mx.int32), TOPK, KB, BS), reps)
    del s, mask, cand
    mx.clear_cache()
    # one layer, end to end, per arm
    lay = {}
    lay["plain old"] = med(lambda: IT.scan(lambda t0, n: PI.scores_range(q, k, w, start, ratio, t0, n), rows, IT.qblock(rows), lambda t0, sc: PI.topk_block(sc, TOPK)), reps)
    PF.BMAX, PF.BMAX_MIN = True, 0
    for tag, g in (("gathered copy", False), ("gscore", True)):
        PF.GSCORE = g
        lay[f"plain bmax ({tag})"] = med(lambda: PF.plain_topk(q, k, w, start, ratio, TOPK, rows), reps)
    PF.GSCORE = True
    lay["source old"] = med(lambda: IT.scan(lambda t0, n: PI.scores_range(q, k, w, start, ratio, t0, n), rows, IT.qblock(rows),
                                            lambda t0, sc: old_select(sc, (start + t0 + mx.arange(sc.shape[0]) + 1) // ratio, vendor)), reps)
    lay["source bmax+gscore"] = med(lambda: PF.source_topk(q, k, w, start, ratio, TOPK, rows, KB), reps)
    idsf = PF.source_topk(q, k, w, start, ratio, TOPK, rows, KB)[1]
    mx.eval(idsf)
    mask, _ = old_select(PI.scores_range(q, k, w, start, ratio, 0, rows), clen, vendor)
    mx.eval(mask)
    lay["consumer old (E1)"] = med(lambda: PI.cand_topk(q2, k2, w2, start, ratio, 0, rows, mask[None], BS, KB, TOPK), reps)
    for tag, g in (("gathered copy", False), ("gscore", True)):
        PF.GSCORE = g
        lay[f"consumer ids ({tag})"] = med(lambda: PF.consumer_topk(q2, k2, w2, start, ratio, TOPK, rows, idsf), reps)
    PF.GSCORE, PF.BMAX = False, False
    del mask
    mx.clear_cache()
    return out, lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", default="131072,262144,524288,1048576")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not (mx.metal.is_available() and PI.NX.nax_available()):
        print("needs the Metal GPU with neural accelerators (fscore, matmul2d)")
        return 2
    vendor = vendor_select()
    print(f"candidate mask reference: {'vendor select_candidate_blocks' if vendor else 'dsv4_core.candidate_mask_rt (vendor module absent)'}", flush=True)
    t_all = time.perf_counter()
    per_key = {}
    layer_tab = {}
    for Tc in [int(x) for x in a.tc.split(",")]:
        rows = rows_for(Tc)
        ratio = 1
        start = ratio * (Tc - rows)
        q, k, w = data(rows, Tc, 1)
        q2, k2, w2 = data(rows, Tc, 2)
        mx.reset_peak_memory()
        correctness(q, k, w, q2, k2, w2, start, ratio, rows, Tc, vendor)
        if not a.check_only:
            ops, lay = timing(q, k, w, q2, k2, w2, start, ratio, rows, Tc, a.reps, vendor)
            scale = 8192 / rows
            print(f" ms per op at Tc={Tc}, rows={rows} (x{scale:.1f} for an 8192-row chunk in brackets)")
            for n, v in ops.items():
                print(f"   {n:72s} {v:9.2f}  [{v * scale:9.1f}]")
            print(f" ms per layer, end to end")
            for n, v in lay.items():
                print(f"   {n:72s} {v:9.2f}  [{v * scale:9.1f}]")
            per_key[Tc] = ops["fscore (scores to memory)"] / Tc / rows * 1e6
            layer_tab[Tc] = {n: v * scale for n, v in lay.items()}
        peak = mx.get_peak_memory() / GIB
        print(f" peak memory {peak:.2f} GiB", flush=True)
        gate(f"peak memory {peak:.2f} GiB < 3.5 GiB", peak < 3.5)
        del q, k, w, q2, k2, w2
        mx.clear_cache()
    if per_key:
        print("fscore ns per (row x key) across Tc (a rising value = memory thrash / cache misses, not FLOPs): " + "  ".join(f"{t}: {v:.4f}" for t, v in per_key.items()))
        print("per 8192-row chunk, ms per layer (rows scaled); chunk model at a past of N tokens = 3 plain layers at Tc = N/2 + source at N + 4 consumers at N:")
        for Tc, lay in layer_tab.items():
            print(f"   Tc={Tc}: " + "; ".join(f"{n} {v:.0f}" for n, v in lay.items()))
    print(f"total {time.perf_counter() - t_all:.1f} s; {'ALL GATES PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
