# SPDX-License-Identifier: Apache-2.0
"""Fused routed-MoE decode (moefuse): one MoE layer in five launches at any row count.

Result of the moefuse2 3-way (stock gather / custom election kernel / union
election, DS4.1 dims, interleaved, outputs verified bit-identical first):
the custom trellis-sharing kernel (gather-shaped + expert election) trails the
proven union path on real traffic (R=4 distinct: +7 % vs -3 %; R=1: +6 % vs -6 %;
all within/near noise except the custom kernel's R=4-distinct loss, replicated
2/2) and only beats it under full expert sharing (-39 % vs -36 %). The extra
Metal (per-count member cases, on-demand x) costs more I-cache than it saves.
So the variant kept behind this flag is the existing union election path
(MoeGather grouped=False, union=True): same 5 launches/layer, bit-identical to
the stock path, no new Metal. R=1 is unchanged (election is a no-op there).

Behind KYOJIN_MOEFUSE=1 (default 0 until the lead's real-pack gate).
"""
from __future__ import annotations

import os

from . import moe as M

MOEFUSE = os.environ.get("KYOJIN_MOEFUSE", "0") == "1"


class MoeFused:
    """Election-sharing MoE decode wrapping a MoeGather: an inner gather with
    grouped=False, union=True (pairs routed to the same expert share one
    trellis decode; per-pair results bit-identical to the plain gather path).
    """

    def __init__(self, gather: M.MoeGather):
        m = gather
        self.m = m
        self._u = M.MoeGather(m.g, m.u, m.d, m.topk, limit=m.limit, wpre=m.wpre, fq=m.fq,
                              tgk_gu=m.tgk_gu, tgk_dn=m.tgk_dn, alpha=m.alpha, beta=m.beta,
                              union=True, grouped=False, pin_rows=m.pin_rows)
        self._u._is_fused_inner = True  # the inner gather never re-enters the flag branch

    def __call__(self, x, ids, wts, fq8: bool = False):
        rows = x.shape[0]
        return self._u(x, ids.reshape(rows, -1), wts.reshape(rows, -1), fq8)


def wrap(gather: M.MoeGather) -> MoeFused:
    return MoeFused(gather)
