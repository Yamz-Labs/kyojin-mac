# SPDX-License-Identifier: Apache-2.0
"""keyscodes (flag dec_keys_codes, KYOJIN_DEC_KEYS_CODES, default 0): the decode index scores of a packed-key stack read the CODES, not a whole-layer unpack.

The depth term. A packed key layer reaches the batched core as a `fp4pool.PackedView` (cdecpack, one session) or a `stackpack.PackedStack` (stackpack, 2+ sessions,
any mu_keep slots). Every score branch of `dsv4_core.core` that is not the fused / fast kernel (those two are gated to n * keys <= IDX_FUSE_MAX = 32768 and keys <=
IDX_SCORE_FAST_MAX = 32768) starts with `CD.dense_of(hand.index_k)`: `fp4pack.unpack` of the WHOLE layer ([S, cap, 128], ~12 un-fused elementwise ops) for each index
layer and each step. `depthflat.py` prices that at ~650 MB of traffic per 128K-key layer, i.e. ~5 KB per key against the 256 B the score kernel then reads, and it does not
depend on the number of rows. From ISCORE_MMA_MIN = 262144 keys on (the width at which the core takes `dsa_decode.scores_rows`) it is the largest per-round term that
grows with the depth in a packed configuration, paid by every layer that owns index keys (2, 8, 14 at ratio 2 and the ratio-1 source; the consumers 24 .. 36 score the
source's keys through its block mask).

The lever. `dsa_decode.scores_rows_codes` unpacks only the 8-key tile a simdgroup is about to multiply, in threadgroup memory, and runs the simdgroup-matrix arithmetic of
`scores_rows(nax=False)` on it: 68 B per key read from the codes, nothing written, no dense layer. A masked 8-key block (consumer layers) is skipped before its codes are
read. Row u addresses the buffer slot `Plan.sess[u]` (under mu_keep that is the slot, NOT the session rank `Plan.rank[u]`), `Ck` = codes.shape[1] rows per slot.

Exactness. Same values as `scores_rows(unpack(codes), nax=False)` bit for bit (same tile arithmetic, same head reduction, same fp16 rounding points; `fp4_key` / the
unpack formula is the one fp4pack.unpack uses). Against the flag-0 path, which takes the neural-accelerator form (`nax=None`) of the same call when the device has it, the
head-sum order differs (the NAX form sums 32 heads in order, the simdgroup form sums 4 partial sums), so a near tie in a score can flip: rounding-level drift, the one
`iscore_mma` already carries and `decsel` / `idxblocks` / `dec_idx_i8` choose on purpose. Flag 0: this module is never reached (the core tests `KC.ENABLED` first).

Scope. fp16 queries (the first-use self check runs fp16) of 32 heads x 128 channels, the key codec of KeySlots (block 32, ue8m0 scales), the runtime-width path (`rt`), t_max >= ISCORE_MMA_MIN, a
PackedView / PackedStack as `hand.index_k`. Anything else, a device without Metal, or a first-use self check that does not agree: `scores` returns None and the core runs the
path it ran before (`HITS["skip"]`, one printed line for a failed self check).
"""
from __future__ import annotations

import os
import sys

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_DEC_KEYS_CODES", "0") not in ("0", "")
HITS = {"codes": 0, "skip": 0, "selfcheck": 0, "selfcheck_fail": 0}
_state = {"ok": None}


def parts(keys):
    """(codes, scales) of a packed key layer (a PackedView or a PackedStack), or None for a dense array / a layer without a packed form."""
    if keys is None or isinstance(keys, mx.array):
        return None
    c, s = getattr(keys, "codes", None), getattr(keys, "scales", None)
    if not isinstance(c, mx.array) or not isinstance(s, mx.array) or c.ndim != 3 or s.ndim != 3:
        return None
    return c, s


def applies(keys, q, t_max: int) -> bool:
    """Can the score call of this layer read the codes? Geometry, codec and width only (the flag and the device are `scores`' business)."""
    from . import dsa_decode as DSA
    p = parts(keys)
    if p is None or q.ndim != 3:
        return False
    c, s = p
    if q.dtype != mx.float16 or getattr(keys, "dtype", None) != q.dtype or not DSA.rows_codes_supported(q.shape[1], q.shape[2], q.dtype, getattr(keys, "block", 0), bool(getattr(keys, "e4m3", True))):
        return False
    return c.dtype == mx.uint8 and s.dtype == mx.uint8 and c.shape[-1] == 64 and s.shape[-1] == 4 and c.shape[:2] == s.shape[:2] and 0 < t_max <= c.shape[1]


def _selfcheck() -> bool:
    """Once per process: a small call of the kernel against the dense `scores_rows(nax=False)` of the unpacked keys, two slots in reversed order and a masked row. A kernel that
    does not compile or does not agree bit for bit switches the lever off with one printed line; the core keeps its path."""
    if _state["ok"] is not None:
        return _state["ok"]
    HITS["selfcheck"] += 1
    ok = False
    try:
        import numpy as np
        from . import dsa_decode as DSA, fp4pack
        rng = np.random.default_rng(7)
        S, Ck, N, T = 2, 256, 3, 200
        k = mx.array(rng.standard_normal((S, Ck, 128)).astype(np.float32))
        codes, scales = fp4pack.pack(k, 32, False)
        dense = fp4pack.unpack(codes, scales, 32, False, mx.float16)
        q = mx.array(rng.standard_normal((N, 32, 128)).astype(np.float16))
        w = mx.array(np.abs(rng.standard_normal((N, 32))).astype(np.float16))
        lens = mx.array([T, 77, 130], dtype=mx.int32)
        sess = mx.array([1, 0, 1], dtype=mx.uint32)
        blocks = rng.random((N, T // 8 + 1)) < 0.5
        cand = mx.array(np.repeat(blocks, 8, axis=1)[:, :T])
        mx.eval(dense, q, w, lens, sess, cand)
        ok = True
        for cm in (None, cand):
            got = DSA.scores_rows_codes(q, w, codes, scales, lens, T, cm, sess)
            ref = DSA.scores_rows(q, w, dense, lens, T, cm, sess, nax=False)
            mx.eval(got, ref)
            g, r = np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32))
            for u in range(N):
                n = int(lens[u])
                ok = ok and bool(np.array_equal(g[u, :n], r[u, :n]))
    except Exception as e:                                              # no Metal, a kernel that does not compile
        print(f"keyscodes: self check raised {type(e).__name__}: {str(e)[:160]}; dec_keys_codes stays off", file=sys.stderr, flush=True)
        ok = False
    if not ok:
        HITS["selfcheck_fail"] += 1
        print("keyscodes: self check failed; dec_keys_codes stays off", file=sys.stderr, flush=True)
    _state["ok"] = ok
    return ok


def scores(q, w, keys, lens, t_max: int, cand, sess):
    """[N, t_max] scores of every row over the codes of `keys`, or None when the layer / the device / the self check keeps the unpack path. `sess` is Plan.sess_arr (slots)."""
    if not ENABLED or not mx.metal.is_available() or not applies(keys, q, t_max):
        HITS["skip"] += 1
        return None
    if cand is not None and (cand.ndim != 2 or cand.shape[1] < t_max):
        HITS["skip"] += 1
        return None
    if not _selfcheck():
        HITS["skip"] += 1
        return None
    from . import dsa_decode as DSA
    c, s = parts(keys)
    HITS["codes"] += 1
    return DSA.scores_rows_codes(q, w, c, s, lens, t_max, cand, sess)
