# SPDX-License-Identifier: Apache-2.0
"""KYOJIN_HY4_HISA=1: a hierarchical, candidate-restricted indexer for PROMPT chunks (L > 16 rows) at long context (default off).

The exact indexer scores every pooled key of the context for every row: L x J x H x d x 2 FLOP per DSA layer (J = ctx / 4), ~4.4 TFLOP per 4096-row chunk and layer at 1M.
Here the J pools are cut into blocks of KYOJIN_HY4_HISA_BLOCK pools (default 32 = 128 tokens). Each block keeps its per-channel max and min (Quest bound), so
    bound(q, block) = relu(q)^+ . kmax - relu(-q) . kmin  >=  q . k   for every key k of the block   (and the mirrored bound from below)
and, with the head weights w of either sign, the block score  sum_h [w_h >= 0 ? w_h relu(UB_h) : w_h relu(LB_h)]  is an upper bound of the best exact score inside the block: a lone
needle pool cannot be diluted the way a mean-pooled summary would. One candidate set per TILE of consecutive rows (KYOJIN_HY4_HISA_TILE, default 64): the tile's max block score
(PIVOT-style), KYOJIN_HY4_HISA_BLOCKS blocks (default 64 = 2048 pools) + the first KYOJIN_HY4_HISA_FIRST (2) blocks + the last KYOJIN_HY4_HISA_LAST (2) blocks the tile can see.
Every row of the tile then scores exactly (the production relu(iq . pk) * w sum) the candidate pools only and takes its own top n_top among them, causally masked per row.
FLOP per chunk and layer at 1M: block bound 2 x 4096 x 8192 x 256 x 2 ~ 0.03 TFLOP + exact 4096 x 2048 x 16 x 128 x 2 ~ 0.03 TFLOP, against 4.4. With every block kept it equals the
exact selection (tests/test_hy4_hisa.py). Quality gates: top-n overlap vs exact (KYOJIN_HY4_HISA_AUDIT=1 logs it), passkey at 256K / 512K / 1M, NLL.
The flags are read at CALL time (admin flag switch, one boot); decode / verify rows (L <= 16) never take this path (the summaries would be rebuilt from all of pk per call)."""
import os
import sys

import mlx.core as mx

_ENV = os.environ.get
STATS = {"calls": 0, "ovl_sum": 0.0, "ovl_min": 1.0, "rows": 0}


def _i(name, default):
    return int(_ENV(name, str(default)) or default)


def wanted(L, B, J, n_top):
    """Parameters dict when the hierarchical selection applies to this call, else None. L rows, J pools, n_top pools kept per row."""
    if _ENV("KYOJIN_HY4_HISA", "0") != "1" or B != 1 or L <= 16:
        return None
    bp, m, tile = _i("KYOJIN_HY4_HISA_BLOCK", 32), _i("KYOJIN_HY4_HISA_BLOCKS", 64), _i("KYOJIN_HY4_HISA_TILE", 64)
    if J < _i("KYOJIN_HY4_HISA_MIN_POOLS", 32768) or J % bp or m * bp < n_top or m >= J // bp or tile < 1:      # no saving (or no room for n_top) below that
        return None
    return dict(bp=bp, m=m, tile=tile, first=_i("KYOJIN_HY4_HISA_FIRST", 2), last=_i("KYOJIN_HY4_HISA_LAST", 2))


def _block_bounds(pk, bp):
    """pk [J, d] -> the stacked keys (kub, klb) [Jb, 2d]: relu(q), relu(-q) concatenated times kub gives the upper bound, times klb the lower bound."""
    Jb, d = pk.shape[0] // bp, pk.shape[1]
    r = pk.astype(mx.float32).reshape(Jb, bp, d)
    kmax, kmin = r.max(axis=1), r.min(axis=1)
    return mx.concatenate([kmax, -kmin], axis=-1), mx.concatenate([kmin, -kmax], axis=-1)


def top(iq, w, pk, valid, offv, kp, n_top, select_tail, prm, rows_blk=512):
    """iq [1, H, L, d] (roped), w [1, L, H], pk [1, J, d], valid [1, J], offv int32 [1] -> (top int32 [1, L, n_top] pool ids ascending (empty slots 0), ok_pool bool [1, L, n_top])."""
    _, H, L, d = iq.shape
    J = pk.shape[1]
    bp, m, T = prm["bp"], prm["m"], prm["tile"]
    Jb = J // bp
    Lp = -(-L // T) * T
    nt = Lp // T
    qpos = offv[:, None] + mx.arange(Lp, dtype=mx.int32)[None, :]                                  # [1, Lp]
    nvis = mx.minimum(qpos // kp if select_tail else (qpos + 1) // kp, J)[0]                       # [Lp] pools visible to each row (monotone in the row)
    row_ok = mx.arange(Lp) < L
    q32 = iq[0].astype(mx.float32)                                                                 # [H, L, d]
    w32 = w[0].astype(mx.float32)                                                                  # [L, H]
    if Lp > L:
        q32 = mx.pad(q32, [(0, 0), (0, Lp - L), (0, 0)])
        w32 = mx.pad(w32, [(0, Lp - L), (0, 0)])
    kub, klb = _block_bounds(pk[0], bp)
    bidx = mx.arange(Jb, dtype=mx.int32)
    rb = max(T, rows_blk // T * T)
    aggs = []
    for a in range(0, Lp, rb):                                                                       # block bound of every row, max over the rows of each tile
        b = min(Lp, a + rb)
        q = q32[:, a:b]
        q2 = mx.concatenate([mx.maximum(q, 0.0), mx.maximum(-q, 0.0)], axis=-1)                    # [H, n, 2d]
        wr = w32[a:b].T[..., None]                                                                   # [H, n, 1]
        bs = (mx.where(wr >= 0, wr * mx.maximum(q2 @ kub.T, 0.0), wr * mx.maximum(q2 @ klb.T, 0.0))).sum(axis=0)        # [n, Jb]
        vis = (bidx[None, :] * bp < nvis[a:b, None]) & row_ok[a:b, None]
        bs = mx.where(vis, bs, -mx.inf)
        aggs.append(bs.reshape((b - a) // T, T, Jb).max(axis=1))
    agg = mx.concatenate(aggs, axis=0) if len(aggs) > 1 else aggs[0]                                 # [nt, Jb]
    lastrow = mx.minimum(mx.arange(nt) * T + T - 1, L - 1)
    lb = (mx.maximum(nvis[lastrow], 1) - 1) // bp                                                    # last block the tile sees
    forced = (bidx[None, :] < prm["first"]) | ((bidx[None, :] <= lb[:, None]) & (bidx[None, :] > lb[:, None] - prm["last"]))
    agg = mx.where(forced, mx.inf, agg)
    blocks = mx.sort(mx.argpartition(-agg, m - 1, axis=-1)[..., :m], axis=-1).astype(mx.int32)       # [nt, m]
    cand = (blocks[..., None] * bp + mx.arange(bp, dtype=mx.int32)).reshape(nt, m * bp)              # [nt, C] candidate pool ids, ascending
    C = m * bp
    pkc = pk[0][cand]                                                                                # [nt, C, d]
    vc = valid[0][cand]                                                                              # [nt, C]
    qt = q32.reshape(H, nt, T, d)
    wt = w32.reshape(nt, T, H)
    nv = nvis.reshape(nt, T)
    tg = max(1, (1 << 26) // (H * T * C))
    tops, oks = [], []
    for g0 in range(0, nt, tg):
        g1 = min(nt, g0 + tg)
        qg = qt[:, g0:g1].transpose(1, 0, 2, 3).reshape(g1 - g0, H * T, d)
        sc = mx.maximum(qg @ pkc[g0:g1].astype(mx.float32).swapaxes(-1, -2), 0.0).reshape(g1 - g0, H, T, C)
        sc = (sc * wt[g0:g1].transpose(0, 2, 1)[..., None]).sum(axis=1)                              # [tg, T, C]
        vis = (cand[g0:g1][:, None, :] < nv[g0:g1][..., None]) & vc[g0:g1][:, None, :]
        sc = mx.where(vis, sc, -mx.inf)
        sel = mx.argpartition(-sc, n_top - 1, axis=-1)[..., :n_top]
        ok = mx.take_along_axis(vis, sel, axis=-1)
        pid = mx.take_along_axis(mx.broadcast_to(cand[g0:g1][:, None, :], sc.shape), sel, axis=-1)
        pid = mx.sort(mx.where(ok, pid, J), axis=-1)
        tops.append(pid)
    pid = mx.concatenate(tops, axis=0) if len(tops) > 1 else tops[0]
    pid = pid.reshape(Lp, n_top)[:L][None]
    ok = pid < J
    return mx.where(ok, pid, 0).astype(mx.int32), ok


def audit(iq, w, pk, valid, offv, kp, n_top, select_tail, top_, ok_, n_rows=32):
    """KYOJIN_HY4_HISA_AUDIT=1: on ~n_rows sampled rows, the fraction of the exact top-n_top pools the hierarchical selection kept. Host sync, so only outside a compiled program;
    logs one line per 12 calls (one chunk of the 12 DSA layers) on stderr."""
    L = iq.shape[2]
    J = pk.shape[1]
    rows = mx.arange(0, L, max(1, L // n_rows))[:n_rows]
    qpos = offv[:, None] + rows[None, :]
    nvis = mx.minimum(qpos // kp if select_tail else (qpos + 1) // kp, J)[0]
    kq = mx.maximum(iq[0][:, rows].astype(mx.float32) @ pk[0].astype(mx.float32).T, 0.0)           # [H, n, J]
    sc = (kq * w[0][rows].astype(mx.float32).T[..., None]).sum(axis=0)
    vis = (mx.arange(J)[None, :] < nvis[:, None]) & valid[0][None, :]
    sc = mx.where(vis, sc, -mx.inf)
    ex = mx.argpartition(-sc, n_top - 1, axis=-1)[..., :n_top]
    exok = mx.take_along_axis(vis, ex, axis=-1)
    h, hok = top_[0][rows], ok_[0][rows]
    hit = ((ex[:, :, None] == h[:, None, :]) & hok[:, None, :]).any(axis=-1) & exok
    n_ex = exok.sum(axis=-1)
    ovl = hit.sum(axis=-1).astype(mx.float32) / mx.maximum(n_ex, 1).astype(mx.float32)
    ovl = mx.where(n_ex >= n_top, ovl, 1.0)                                                          # rows with fewer visible pools than n_top are exact by construction
    mx.eval(ovl)
    v = ovl.tolist()
    STATS["calls"] += 1
    STATS["ovl_sum"] += sum(v)
    STATS["rows"] += len(v)
    STATS["ovl_min"] = min(STATS["ovl_min"], min(v))
    if STATS["calls"] % 12 == 0:
        print(f"HISA audit: calls {STATS['calls']} rows {STATS['rows']} top-{n_top} overlap mean {STATS['ovl_sum'] / STATS['rows']:.4f} min {STATS['ovl_min']:.3f} (J {J})", file=sys.stderr, flush=True)
