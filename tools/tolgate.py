"""Tolerance class of the day-1 NAX rows (KYOJIN_ROWS_NAX=1, the default), shared by spec_check, mdraft_check and serve_spec_check.

Rows above 4 run the fused NAX kernel, whose fp16 accumulation order differs from the solo GEMV: the products differ at fp16 rounding, so the
bit-exact comparisons of those gates become tolerance tests (still bitwise with KYOJIN_ROWS_NAX=0: `tolerant()` is False).
Measured on the 2-layer slice packs (random-ish weights amplify rounding; real DS4.1 will be tighter), spec_check --depth 5, ds2 ds2m ds2e, 3 seeds:
  * logit row vs solo row, max|d| / max|logit| : NAX rows 0.037-0.049 (rms(d) ~ 1.5 % of rms(logit)). The independent control "a different valid
    fp16 order", `spec_check --neg stock_matmul` (stock x @ w.T, no NAX), gives 0.080: the NAX deviation is inside that class. REL_LOGIT = 0.10.
  * cache / ring arrays vs solo, max|d| / max|array| : 0.028-0.036, and identical (0.0278) with stock_matmul: fp8 fake-quant boundary flips of the
    cached kv (one e4m3 ulp of an element is at most 2^-3 of the array max). REL_STATE = 2^-3.
  * a greedy token may flip only at a near tie: the solo margin top1 - top2 (or solo top1 - the flipped token) is within what the rounding can
    move, 2 x max|d| of that row (spec_check, measured per row), or TIE_REL x max|logit| where only tokens are seen (serve_spec_check).
  * 99.5 % of the greedy tokens must agree with plain decode.
"""
import numpy as np

REL_LOGIT = 0.10          # max |logit - solo logit| / max |solo logit| of the row
REL_STATE = 0.125         # same for cache / ring arrays, per array (2^-3: one fp8 e4m3 ulp)
TIE_REL = 0.10            # token-level near tie: margin <= TIE_REL x max|logit|  (2 x the measured NAX class 0.05, both sides move)
MIN_AGREE = 0.995         # greedy top-1 agreement with plain decode
MIN_ROWS = 200            # below this many compared rows one flip is already under MIN_AGREE: a run only fails on unexplained flips, the pool (tolpool.py) on the rate


def tolerant() -> bool:
    """True when the NAX rows are on (exl3_linear.ROWS_NAX, set by KYOJIN_ROWS_NAX / rows.enable(nax=)) AND the device runs NAX (metal_nax probe: KYOJIN_NAX=0 or a
    GPU without neural accelerators keeps the solo bits): comparisons use the tolerance class."""
    from kyojin_mlx import exl3_linear as EL, metal_nax as NX
    return bool(EL.ROWS_NAX) and NX.nax_available()


def maxrel(x, y) -> float:
    """max |x - y| / max |y| (float64)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    den = float(np.abs(y).max())
    return float(np.abs(x - y).max()) / den if den > 0 else float(np.abs(x - y).max())


def margin(ref_logits) -> float:
    """top1 - top2 of a logit row."""
    t = np.partition(np.asarray(ref_logits, np.float64), -2)[-2:]
    return float(t[1] - t[0])


def near_tie(ref_logits, got_token=None, dmax=None) -> bool:
    """A flip at this row is explained by the rounding class: the solo margin (to the runner-up, or to `got_token` when given) is within
    2 x dmax (the largest logit deviation measured on the row) or, without dmax, TIE_REL x max|logit|."""
    r = np.asarray(ref_logits, np.float64)
    gap = margin(r) if got_token is None else float(r.max() - r[got_token])
    return gap <= (2 * dmax if dmax is not None else TIE_REL * float(np.abs(r).max()))
