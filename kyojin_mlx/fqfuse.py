# SPDX-License-Identifier: Apache-2.0
"""Folds the DS4.1 fp8 activation round trip (`fake_quant_fp8_ue8m0`, one launch) into the rot kernel of the linear that reads it.

Every EXL3 linear of a DS4.1 layer reads `fq(x)`: fq launch + rot + GEMV + finalize. The rot kernel loads x anyway, so it can apply the round trip while it loads
(metal_src.with_fq8): the fq node is then referenced by nobody and is never launched. The memoised fq wrapper (fused_group._memo) tells this module which array is
the output of which input; a linear that receives such an output with fp16 activations reads the INPUT of the fq instead and runs its rot with the fold on. The values are
bit-identical (the folded arithmetic is the fq kernel's, stored as half in between). Linears that do not know the fold (dense prefill, bias, other paths) simply use the fq output,
which then is launched as before. 6 launches fewer per reuse layer.

Switch: module attribute ENABLED (env KYOJIN_FQ_FUSE=0); a traced `mx.compile` region keeps the choice it was traced with."""
from __future__ import annotations

import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_FQ_FUSE", "1") != "0"
_pairs: list = []          # (input, output) of the last fq calls, newest last; both kept alive so that `is` tests stay valid


def note(x: mx.array, y: mx.array) -> None:
    _pairs.append((x, y))
    del _pairs[:-8]


def source(y: mx.array) -> mx.array | None:
    """The input x of the fq call that produced `y`, if the fold applies (fp16, last dim a multiple of 128), else None."""
    if not ENABLED or y.dtype != mx.float16 or y.shape[-1] % 128:
        return None
    for x, yi in reversed(_pairs):
        if yi is y:
            return x if x.dtype == mx.float16 and x.shape == y.shape else None
    return None
