# SPDX-License-Identifier: Apache-2.0
"""Flag KYOJIN_FOREIGN_ATTN_FAST (default off): the sparse-attention mask of a foreign pack's QSA indexer as ONE kernel at decode.

At decode past the indexer budget (qwen4_exp, 4K+ context) the full-attention layer runs `indexer.build_mask(selection)`: ~25 small ops (token-index scatter, arange,
three comparisons, where, put_along_axis ...) per layer per token, then a dense SDPA with that boolean mask. Here one launch writes the same boolean mask:

  token t (left padding 0, one query row per sequence):  blk = t / ratio
    sparse (complete_count > block_topk):  mask[t] = (blk is one of the selected blocks) or (complete_count * ratio <= t < query_end)
    otherwise                              mask[t] = t < query_end                       (plain causal)
A boolean result has no rounding: the kernel is bit-equal to the stock mask by construction, and a module only qualifies when a probe (random selections, sparse and
dense rows, stock vs kernel) is `array_equal`. A module qualifies by structure (attributes block_topk / compress_ratio / build_mask), never by name.
Anything else (S > 1, left padding, a mask shape the stock code would build differently) runs the module's own `build_mask`.
The class method is replaced once; `ACTIVE` switches the fast path at run time (a profile cell A/Bs in one process)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx

ACTIVE = os.environ.get("KYOJIN_FOREIGN_ATTN_FAST", "0") not in ("0", "")
STATS = {"fast": 0, "stock": 0}
_KERNEL = None

_SRC = """
    uint t = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    int K = P[0], ratio = P[1], topk = P[2], key_len = P[3];
    if ((int)t >= key_len) return;
    int cc = C[b];
    int qe = Q[b];
    bool causal = (int)t < qe;
    bool r;
    if (cc > topk) {
        int blk = (int)t / ratio;
        bool sel = false;
        for (int j = 0; j < K; j++) { if (S[b * K + j] == blk) { sel = true; break; } }
        bool tail = ((int)t >= cc * ratio) && ((int)t < qe);
        r = sel || tail;
    } else {
        r = causal;
    }
    out[b * key_len + t] = r;
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(name="kyojin_qsa_mask", input_names=["S", "C", "Q", "P"], output_names=["out"], source=_SRC)
    return _KERNEL


def fused_mask(selected_blocks, complete_counts, query_ends, key_len, ratio, topk):
    """selected_blocks (B, 1, K) int32, complete_counts (B, 1), query_ends (B, 1) -> bool (B, 1, 1, key_len)."""
    B, _, K = selected_blocks.shape
    P = mx.array([K, ratio, topk, key_len], dtype=mx.int32)
    out = _kernel()(inputs=[selected_blocks.astype(mx.int32), complete_counts.astype(mx.int32).reshape(B), query_ends.astype(mx.int32).reshape(B), P],
                    output_shapes=[(B, 1, 1, key_len)], output_dtypes=[mx.bool_], grid=(key_len, B, 1), threadgroup=(min(256, key_len), 1, 1))
    return out[0]


def _ok(self, sel):
    sb = sel.selected_blocks
    return (ACTIVE and mx.default_device() == mx.gpu and sb.ndim == 3 and sb.shape[1] == 1 and sel.zero_padding and sel.key_len >= 1 and sb.shape[2] >= 1
            and tuple(sel.complete_counts.shape) == tuple(sb.shape[:2]) and tuple(sel.query_ends.shape) == tuple(sb.shape[:2]) and sb.dtype == mx.int32)


def _probe(cls, stock) -> bool:
    """Stock vs kernel on random selections, sparse and dense rows, awkward key lengths."""
    mx.random.seed(11)
    for ratio, topk, key_len, B in ((4, 8, 301, 2), (4, 8, 40, 1), (16, 4, 1024, 1), (4, 6, 33, 3)):
        me = SimpleNamespace(compress_ratio=ratio, block_topk=topk)
        for cc_list in ([key_len // ratio] * B, [max(1, topk - 1)] * B, [(key_len // ratio) - (i % 2) for i in range(B)]):
            cc = mx.array(cc_list, dtype=mx.int32).reshape(B, 1)
            sb = (mx.random.randint(0, 1 << 20, (B, 1, topk)) % cc[..., None]).astype(mx.int32)      # decode picks only complete blocks (no -1 in a sparse row)
            qe = (cc * ratio + (mx.arange(B)[:, None] % ratio)).astype(mx.int32)
            qe = mx.minimum(qe + 1, key_len)
            sel = SimpleNamespace(selected_blocks=sb, query_ends=qe, complete_counts=cc, left_padding=mx.zeros((B,), dtype=mx.int32), key_len=key_len, zero_padding=True)
            ref = stock(me, sel)
            got = fused_mask(sb, cc, qe, key_len, ratio, topk)
            if ref.shape != got.shape or ref.dtype != got.dtype or not mx.array_equal(ref, got).item():
                return False
    return True


def qualifies(m) -> bool:
    return all(hasattr(m, a) for a in ("build_mask", "compress_ratio", "block_topk", "head_dim", "index_qk_proj")) and isinstance(m.compress_ratio, int) and isinstance(m.block_topk, int)


def _wrap(stock):
    def build_mask(self, selection):
        if _ok(self, selection):
            STATS["fast"] += 1
            return fused_mask(selection.selected_blocks, selection.complete_counts, selection.query_ends, selection.key_len, self.compress_ratio, self.block_topk)
        STATS["stock"] += 1
        return stock(self, selection)
    build_mask._kyojin_attn_fast = True
    build_mask._stock = stock
    return build_mask


def install(model, log=print) -> int:
    """Replace `build_mask` once on the class of every qualifying indexer when the probe passes. Returns the number of modules armed (0 with the flag unset)."""
    if not ACTIVE:
        return 0
    n = 0
    done = {}
    for _, m in model.named_modules():
        if not qualifies(m):
            continue
        cls = type(m)
        if cls not in done:
            stock = cls.build_mask
            if getattr(stock, "_kyojin_attn_fast", False):
                done[cls] = True
            else:
                try:
                    ok = _probe(cls, stock)
                except Exception as e:
                    log(f"[foreign] attn_fast: probe error {type(e).__name__}: {e}")
                    ok = False
                done[cls] = ok
                if ok:
                    cls.build_mask = _wrap(stock)
                else:
                    log(f"[foreign] attn_fast: {cls.__name__} refused (kernel mask differs from the module's own)")
        if done[cls]:
            n += 1
    log(f"[foreign] attn_fast: {n} QSA indexers build their decode mask in one launch")
    return n
