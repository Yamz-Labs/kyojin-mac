# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 verify rows (flag glm_dense_rows, KYOJIN_GLM_DENSE_ROWS, default 0): a dense FP8 projection of 2..4 rows is ONE launch that reads its weights once.

mlx_vlm's `linear(module, x)` sends a 3-D input of 2..8 rows (one session's MTP verify rows, [1, L, H]) through `native_batch_linear`, whose fallback for a module that
is not an MLX QuantizedLinear is `tokenwise(module, x)`: L calls on [1, 1, H] slices, L launches, the whole weight read L times. Every ForeignLinear (qkv_proj, o_proj,
f_b / g_b, q_a / kv_a, q_b, shared experts, lm_head...) took that road: G8d ladder on the cut, lm_head 0.56 / 1.16 / 1.59 / 2.14 ms and KDA qkv_proj +0.2 ms per layer for
R = 1..4 (bandwidth-bound: R = 4 should cost ~ R = 1). With the flag, a ForeignLinear whose matrix takes the G3 GEMV (`DevPack.g3_ok`: fp8 / fp8b, 1..DENSE_GEMV_ROWS rows,
KYOJIN_GLM_DENSE_GEMV on) is called once with all its rows: the G3 kernel holds R x NR accumulators per simdgroup, stages x once and reads each weight unit once. The per-row
sum order of that kernel does not depend on R (same lane / unit walk, same xor-butterfly reduce, see tests/test_dense_rows.py: row r == the R = 1 result bit for bit), so
the output equals the tokenwise one. Anything else (other formats, > DENSE_GEMV_ROWS rows, a resident fp16 copy) keeps the stock path. Read per call.

N1-VROWS (flag glm_vrows, KYOJIN_GLM_VROWS, default 0): the same route for every verify shape the engine serves: [B, L, H] with B * L <= VROWS_MAX (8) rows, B sessions of L
rows each (mlx_vlm's tokenwise sends [B, L, H] as L calls of B rows, i.e. 4 sessions x 2 rows = two R = 4 launches, the weights read twice), and the 6-bit copy of
KYOJIN_GLM_DENSE_Q serves up to VROWS_MAX rows (stock: <= 4) so every row keeps the weights it has at R = 1 / 4 (the sym6 kernel's rows are independent of R, same bits).
Beyond VROWS_MAX rows, or with the flag off, nothing changes. Without VROWS, KYOJIN_GLM_DENSE_ROWS keeps its G3e2 meaning (one session, <= DENSE_GEMV_ROWS rows)."""
from __future__ import annotations

import os

from . import fired as FD

ENABLED = os.environ.get("KYOJIN_GLM_DENSE_ROWS", "0") not in ("0", "")
VROWS = os.environ.get("KYOJIN_GLM_VROWS", "0") not in ("0", "")
VROWS_MAX = int(os.environ.get("KYOJIN_GLM_VROWS_MAX", "8"))


def cap() -> int:
    """Rows a verify call may carry in ONE launch (0 = route off)."""
    from . import foreign_linear as FL
    if VROWS:
        return max(VROWS_MAX, FL.DENSE_GEMV_ROWS)
    return FL.DENSE_GEMV_ROWS if ENABLED else 0


def why(module, x) -> str:
    """'' when `module(x)` can take all the rows of x [B, L, H] in one launch, else the reason (flag audit)."""
    from . import foreign_linear as FL
    if not isinstance(module, FL.ForeignLinear):
        return "not_foreign_linear"
    if x.ndim != 3 or (x.shape[0] != 1 and not VROWS):
        return "shape"
    if not FL.DENSE_GEMV:
        return "dense_gemv_off"
    c, rows = cap(), x.shape[0] * x.shape[1]
    if rows > c:
        return f"rows>{c}"
    FL.ROWS_CAP[0] = c
    try:
        if not module.dev.g3_ok(rows):
            return module.dev.g3_why(rows)
    finally:
        FL.ROWS_CAP[0] = 0
    return ""


def route(module, x) -> bool:
    """True: call `call(module, x)` once (all rows)."""
    if not (ENABLED or VROWS):
        return False
    r = why(module, x)
    if r:
        if FD.AUDIT and r != "not_foreign_linear":
            FD.fb("glm_vrows" if VROWS else "dense_rows", r)
        return False
    FD.fire("glm_vrows" if VROWS else "dense_rows")
    return True


def call(module, x):
    """module(x) with the GEMV row cap of the route (set for this call only; nothing nests another route inside a ForeignLinear)."""
    from . import foreign_linear as FL
    FL.ROWS_CAP[0] = cap()
    try:
        return module(x)
    finally:
        FL.ROWS_CAP[0] = 0
