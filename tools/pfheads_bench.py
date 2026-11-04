"""pf_idx_heads (prefill indexer first pass on a query-head subset): the ms of every piece of the indexer at the served chunk geometry, and the per-chunk
model they add up to at a 983K-token past.

Every row scores every visible key, so the first pass and the block ranking are linear in the visible key count: the numbers measured here at 64K keys scale
by 491520 / 65536 = 7.5 for a plain layer at a 983K-token past (ratio 2) and by 983040 / 65536 = 15 for the candidate source (ratio 1). The exact
rescoring is flat in the past (it gathers pf_idx_heads_m * 8 keys per row). Those products are MODEL: the harness caps the key count at 64K and a Metal
array at 1 GiB, so no run here holds the 251 MB key mirror of a real 983K past.

    python tools/pfheads_bench.py               # 8192 rows, 64K keys, 10 interleaved reps, < 180 s, peak memory printed
    python tools/pfheads_bench.py --reps 3 --rows 4096
"""
import argparse
import sys
import time

import numpy as np

import mlx.core as mx

from kyojin_mlx import pf_flat as PF

ROWS, TC, TOPK, CANDS, RATIO = 8192, 65536, 512, 2048, 1
_MM = """
    // the matmul2d tile alone: 32 keys x N head slots x 128 per op.run, no post, no store but one value per row (what a narrower head tile costs)
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x;
    const uint Tc = prm[0], TPT = prm[1];
    threadgroup int8_t Qs[32 * 128];
    threadgroup float acc[4];
    for (uint i = tid; i < 32u * 128u; i += 128u) Qs[i] = Q[ulong(t) * 32 * 128 + i];
    if (tid < 4u) acc[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto desc = matmul2d_descriptor(32, @N@, 128, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tQ((threadgroup int8_t*)Qs, dextents<int32_t, 2>(128, @N@));
    float mine = 0.0f;
    for (uint j = 0; j < TPT; ++j) {
        const uint k0 = (threadgroup_position_in_grid.y * TPT + j) * 128 + sg * 32;
        if (k0 >= Tc) break;
        const uint ks = k0 + 32u > Tc ? Tc - 32u : k0;
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tK((device int8_t*)(K + ulong(ks) * 128), dextents<int32_t, 2>(128, 32));
        auto cT = op.get_destination_cooperative_tensor<decltype(tK), decltype(tQ), int32_t>();
        op.run(tK, tQ, cT);
        mine += float(cT[0]) + float(cT[cT.get_capacity() - 1]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) acc[sg] += mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) S[t] = acc[0] + acc[1] + acc[2] + acc[3];
"""


def tile_only(slots, rows, tc):
    """ms of a pure matmul2d loop over `slots` head slots (no head sum, no block max): what a narrower tile of the same 32-key tile costs."""
    from kyojin_mlx import metal_nax as NX
    k = mx.fast.metal_kernel(name=f"kyojin_pfheads_mm{slots}", header=NX.HEADER_NAX, input_names=["Q", "K", "prm"], output_names=["S"],
                             source=_MM.replace("@N@", str(slots)))
    prm = mx.array([tc, 32] + [0] * 14, dtype=mx.uint32)
    q = mx.zeros((rows * 32, 128), dtype=mx.int8)
    kk = mx.zeros((tc, 128), dtype=mx.int8)
    mx.eval(q, kk)
    fn = lambda: k(inputs=[q, kk, prm], grid=(rows * 128, -(-tc // (128 * 32)), 1), threadgroup=(128, 1, 1), output_shapes=[(rows,)], output_dtypes=[mx.float32])[0]
    mx.eval(fn())
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t) * 1e3)
    del q, kk
    mx.clear_cache()
    return float(np.median(ts))
QB = 512                                   # rows of one indexer reduction block (idx_topk.qblock)
GIB = 2 ** 30
X_PLAIN, X_SOURCE = 491520 / 65536, 983040 / 65536
TAIL = 128                                  # CED rows of a source / consumer layer (ced.py: the midpoint and everything after it run on the tail)


def data(rows, tc):
    """On the int8 grid (integers in [-120, 120]) so the pack is lossless: the first pass then ranks on its own head subset alone, as in tests/test_pfheads.py."""
    rng = np.random.default_rng(1)
    q = np.rint(rng.standard_normal((1, rows, 32, 128)) * 30).clip(-120, 120).astype(np.float16)
    k = np.rint(rng.standard_normal((1, tc, 128)) * 30).clip(-120, 120).astype(np.float16)
    w = (rng.standard_normal((1, rows, 32)) * 0.05).astype(np.float16)
    return mx.array(q), mx.array(k), mx.array(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--tc", type=int, default=TC)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    rows, tc = a.rows, a.tc
    q, k, w = data(rows, tc)
    start = RATIO * (tc - rows)
    PF.BMAX, PF.BMAX_MIN, PF.I8, PF.I8_REG, PF._I8_GRID = True, 0, True, True, True
    ki, ke = PF.i8pack(k[0])
    qi, qe = PF.i8pack(q[0].reshape(-1, 128))
    w8 = w[0].astype(mx.float32) * qe.reshape(rows, 32)
    mx.eval(ki, ke, qi, qe, w8)
    packs = {}
    for nh in (8, 16):
        packs[nh] = PF.head_rows(qi, w8, PF.head_sel(q, w, nh), nh)
        mx.eval(*packs[nh])
    nb = -(-tc // 8)
    clen = (start + mx.arange(QB) + 1) // RATIO
    bm0 = PF.bmax_i8(qi, ki, w8, ke, start, RATIO, 0, QB)
    ids = {m: PF.pick_blocks(bm0, clen, m, False) for m in (TOPK + PF.I8_MARGIN, 1024, CANDS)}
    kb = PF.key_blocks(k)
    mx.eval(bm0, *ids.values())
    arms = {
        "i8 pack, keys (grows with the past)": lambda: PF.i8pack(k[0]),
        "i8 pack, queries (flat in the past)": lambda: PF.i8pack(q[0].reshape(-1, 128)),
        "first pass 32 heads (today)": lambda: PF.bmax_i8(qi, ki, w8, ke, start, RATIO, 0, rows),
        "first pass 16 heads": lambda: PF.bmax_i8_rows(packs[16][0], ki, packs[16][1], ke, start, RATIO, 0, rows, 16),
        "first pass 8 heads": lambda: PF.bmax_i8_rows(packs[8][0], ki, packs[8][1], ke, start, RATIO, 0, rows, 8),
        "head subset (select + pack, 16 heads)": lambda: PF.head_rows(qi, w8, PF.head_sel(q, w, 16), 16),
    }
    for m in (TOPK + PF.I8_MARGIN, 1024, CANDS):
        arms[f"rank {m} blocks (one {QB}-row block)"] = (lambda m=m: PF._blk_topk(bm0, clen.astype(mx.int32), m))
    for m in (TOPK + PF.I8_MARGIN, 1024, CANDS):
        arms[f"exact rescoring {m} blocks (one {QB}-row block)"] = (
            lambda m=m: PF.select_from_blocks(q, k, kb, w, ids[m], start, RATIO, 0, TOPK))
    res = {n: [] for n in arms}
    for _ in range(a.reps):
        for n, fn in arms.items():
            t = time.perf_counter()
            mx.eval(fn())
            res[n].append((time.perf_counter() - t) * 1e3)
    med = {n: float(np.median(v)) for n, v in res.items()}

    def show(name, extra=""):
        v = np.array(res[name])
        print(f"  {name:46s} {np.median(v):8.2f} +- {v.std(ddof=1) / len(v) ** 0.5:5.2f} ms{extra}")

    print(f"rows={rows} keys={tc} topk={TOPK} candidate blocks={CANDS}: {a.reps} interleaved reps, median +- SE")
    for n in arms:
        show(n, f"   x{rows // QB} = {med[n] * rows / QB:6.1f} ms per chunk" if "block (" in n else "")
    print(f"  peak memory {mx.get_peak_memory() / GIB:.2f} GiB")
    print("a narrower matmul2d tile is not cheaper: the same 32-key tile, head slots 32 / 16 / 8, no head sum and no block max")
    for slots in (32, 16, 8):
        v = tile_only(slots, rows, tc)
        print(f"  {slots:2d} head slots {v:6.1f} ms  ({rows * tc * slots * 128 * 2 / v * 1e3 / 1e12:5.1f} TFLOP/s)")

    first = {h: med[f"first pass {h} heads{'' if h < 32 else ' (today)'}"] for h in (32, 16, 8)}
    prep = med["head subset (select + pack, 16 heads)"]
    pk, pq = med["i8 pack, keys (grows with the past)"], med["i8 pack, queries (flat in the past)"]
    rank = {m: med[f"rank {m} blocks (one {QB}-row block)"] * rows / QB for m in (TOPK + PF.I8_MARGIN, 1024, CANDS)}
    resc = {m: med[f"exact rescoring {m} blocks (one {QB}-row block)"] * rows / QB for m in (TOPK + PF.I8_MARGIN, 1024, CANDS)}
    m0 = TOPK + PF.I8_MARGIN
    print("model, ms per layer: the pieces that grow with the past (key pack, first pass, block ranking) scale with the key count, the query pack, the head")
    print("subset and the exact rescoring are flat in it. MODEL: the x7.5 / x15 key scaling and the source layer's 128-row tail are not measured here.")
    print(f"  {'':40s} {'64K keys':>9s} {'plain 983K':>11s} {'source 983K':>12s} {'source tail':>12s}")
    rows_out = []
    for tag, h, m in (("plain today", 32, m0), ("plain 16 heads, M=544", 16, m0), ("plain 16 heads, M=1024", 16, 1024), ("plain 16 heads, M=2048", 16, CANDS)):
        grow = pk + first[h] + rank[m]
        flat = grow + pq + (prep if h < 32 else 0) + resc[m]
        pl = grow * X_PLAIN + pq + (prep if h < 32 else 0) + resc[m]
        src = grow * X_SOURCE + pq + (prep if h < 32 else 0) + resc[m]
        tail = grow * X_SOURCE * TAIL / rows + pq * TAIL / rows + (prep if h < 32 else 0) * TAIL / rows + resc[m] * TAIL / rows
        rows_out.append((tag, flat, pl, src, tail))
        print(f"  {tag:40s} {flat:9.1f} {pl:11.0f} {src:12.0f} {tail:12.0f}")
    print("  a source / consumer layer that CED cut runs its indexer on the last 128 rows only (ced.py:_attention_mid, the served prefill): 'source tail'")
    print("  is that layer, 'source 983K' the same layer on a whole chunk (the last chunk of a prompt, or ced_skip_dec off)")
    today = 3 * rows_out[0][2] + rows_out[0][4]
    for tag, _, pl, _, tail in rows_out[1:]:
        new = 3 * pl + tail
        print(f"  chunk of 8192 rows at 983K tokens, {tag:26s}: indexer {new:6.0f} ms against {today:6.0f} ms today ({new - today:+6.0f} ms), "
              f"a {100 * (today - new) / 3970:+.1f} % chunk time on the measured 3970 ms (MODEL: {2065 * 3970 / (3970 + new - today):.0f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
