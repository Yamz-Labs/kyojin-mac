"""Compact, kernel-readable form of the NVFP4 expert block scales (flag KYOJIN_MOE_SCALE_COMPACT, default off).

The csf32 checkpoint stream is tiled and indexed per block, so a kernel cannot address (row, col) in O(1); the loader used to
expand it to one E4M3 byte per 16-wide block (`meta`). The compact form built here keeps the same values at ~0.6 byte/scale:

  codes uint8  [rows, cols/2]    4-bit code per scale, row-major, low nibble = even col; byte j of a row = the ushort j of the expanded `meta`
  base  uint8  [rows]            per-row base byte; scale = base[row] + code (base <= 240, so no wrap)
  ovl   uint8  [rows, OB]        1 bit per 16-col group (OB = ceil(cols / 16 / 8) bytes per row): set when the group holds an exception
  eptr  uint16 [rows + 1]        per matrix: index of the row's first exception (end = next entry), exceptions sorted by (row, col)
  ecol  uint16 [n_exc]           column of each exception
  evl   uint8  [n_exc]           replacement scale byte
  ebase uint32 [E]               stacks only: first exception of each expert matrix (global index = ebase[ex] + eptr[ex * (rows + 1) + row])

scale(row, col) = ovl bit of its group set and (col in the row's exception list) ? evl : base[row] + nibble(codes): O(1) on the common path
(one code byte, one base byte, one overlay byte), a scan of <= 5 entries (measured max per row) only in the 16-col groups that hold an exception.
The expanded matrix is reproduced exactly (see `expand`, asserted in tests/test_moe_scale_compact.py).
"""
from __future__ import annotations

import os

import numpy as np

from . import csf_scale as cs

FLAG = "KYOJIN_MOE_SCALE_COMPACT"


def enabled():
    return os.environ.get(FLAG, "0") not in ("", "0", "false", "False")


def csf_planes(bases, codes, page_bases, offsets, exceptions):
    """(base[rows], code[rows, cols] 0..15 as stored, exception mask[rows, cols], full scale matrix[rows, cols])."""
    full = cs._decode_csf_32x2(bases, codes, page_bases, offsets, exceptions)
    rows, cols = full.shape
    _, _, _, _, cells = cs._csf32_blocks(rows, cols)
    packed = codes.reshape(-1)
    code = np.empty(rows * cols, dtype=np.uint8)
    code[cells] = np.stack((packed & 0xF, packed >> 4), -1).ravel()
    code = code.reshape(rows, cols)
    base = bases.reshape(-1)
    mask = full != (base[:, None] + code)
    return base, code, mask, full


def best_base(full):
    """Per row, the base in 0..240 whose 16-wide window base..base+15 holds the most scales (fewest exceptions)."""
    rows, cols = full.shape
    hist = np.bincount((np.arange(rows, dtype=np.int64)[:, None] * 256 + full).ravel(), minlength=rows * 256).reshape(rows, 256)
    cs_ = np.concatenate((np.zeros((rows, 1), np.int64), np.cumsum(hist, 1)), 1)
    win = cs_[:, 16:257] - cs_[:, 0:241]
    return win.argmax(1).astype(np.uint8)


def ovl_bytes(cols):
    return (-(-cols // 16) + 7) // 8


def pack_compact(full, base=None):
    """Compact arrays of one expanded matrix `full` [rows, cols] (uint8 E4M3 bytes). base: per-row bases (default best_base)."""
    full = np.asarray(full, dtype=np.uint8)
    rows, cols = full.shape
    if cols % 2:
        raise ValueError("cols must be even")
    base = best_base(full) if base is None else np.asarray(base, np.uint8).reshape(-1)
    if (base > 240).any():
        raise ValueError("bases must be in 0..240")
    d = full.astype(np.int16) - base[:, None]
    mask = (d < 0) | (d > 15)
    code = np.where(mask, 0, d).astype(np.uint8)
    codes = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    ng = -(-cols // 16)
    gm = np.zeros((rows, ng * 16), bool)
    gm[:, :cols] = mask
    gm = gm.reshape(rows, ng, 16).any(2)
    ovl = np.packbits(np.pad(gm, ((0, 0), (0, ovl_bytes(cols) * 8 - ng))), axis=1, bitorder="little")
    cnt = mask.sum(1)
    eptr = np.concatenate(([0], np.cumsum(cnt)))
    if int(eptr[-1]) > 65535:
        raise ValueError("too many exceptions for uint16 eptr")
    r, c = np.nonzero(mask)                                          # row-major: sorted by (row, col)
    return dict(codes=np.ascontiguousarray(codes), base=np.ascontiguousarray(base), ovl=np.ascontiguousarray(ovl), eptr=eptr.astype(np.uint16),
                ecol=c.astype(np.uint16), evl=full[r, c].astype(np.uint8), rows=rows, cols=cols)


def csf32_bases(streams):
    return np.asarray(streams[0], np.uint8).reshape(-1)


def from_csf32(streams, use_csf_bases=True):
    """Compact arrays straight from the csf32 streams (the checkpoint's own row bases: its code cells and exceptions are kept)."""
    full = cs._decode_csf_32x2(*streams)
    return pack_compact(full, csf32_bases(streams) if use_csf_bases else None)


def stack(cs_list):
    """Stack E compact matrices of one shape: arrays concatenated expert by expert (+ ebase uint32 [E]); every array has at least one element."""
    c0 = cs_list[0]
    assert all((c["rows"], c["cols"]) == (c0["rows"], c0["cols"]) for c in cs_list)
    cat = lambda k: np.concatenate([c[k].reshape(-1) for c in cs_list])
    n = np.array([c["ecol"].size for c in cs_list], np.int64)
    ebase = np.concatenate(([0], np.cumsum(n)[:-1])).astype(np.uint32)
    ebase = np.pad(ebase, (0, max(0, 16 - ebase.size)))                # mlx passes arrays of <= 8 elements as `constant` scalars, not device pointers
    pad = lambda a: a if a.size else np.zeros(1, a.dtype)
    return dict(codes=cat("codes"), base=cat("base"), ovl=cat("ovl"), eptr=cat("eptr"), ecol=pad(cat("ecol")), evl=pad(cat("evl")), ebase=ebase,
                rows=c0["rows"], cols=c0["cols"], E=len(cs_list))


ARRAYS = ("codes", "base", "ovl", "eptr", "ecol", "evl", "ebase")
NAMES = ("cd", "bs", "ov", "ep", "ec", "ev", "eb")                   # kernel input name prefixes, in ARRAYS order


def expand(c, e=0):
    """Reference decode (numpy) of expert `e` of a stack (or of one compact matrix) to the [rows, cols] uint8 matrix a kernel must read."""
    rows, cols = c["rows"], c["cols"]
    ob = ovl_bytes(cols)
    codes = c["codes"].reshape(-1, cols // 2)[e * rows:(e + 1) * rows]
    base = c["base"].reshape(-1)[e * rows:(e + 1) * rows]
    ovl = c["ovl"].reshape(-1, ob)[e * rows:(e + 1) * rows]
    eptr = c["eptr"].reshape(-1)[e * (rows + 1):(e + 1) * (rows + 1)].astype(np.int64)
    eb = int(c["ebase"][e]) if "ebase" in c else 0
    out = np.empty((rows, cols), np.uint8)
    out[:, 0::2] = base[:, None] + (codes & 0xF)
    out[:, 1::2] = base[:, None] + (codes >> 4)
    for r in np.flatnonzero(eptr[1:] != eptr[:-1]):                  # rows with exceptions only (a Python loop over ~half the rows is fine for a reference)
        for k in range(eb + eptr[r], eb + eptr[r + 1]):
            out[r, c["ecol"][k]] = c["evl"][k]
    return out


def nbytes(c):
    return sum(c[k].nbytes for k in ARRAYS if k in c)


# ----------------------------------------------------------------------------------------------
# Metal side: the scale-read helper and the source transform of the decode MoE kernels (variant scm; with the flag off no source changes)
# ----------------------------------------------------------------------------------------------
import re

CSC_HDR = r"""
struct CSC { device const uchar* cd; device const uchar* bs; device const uchar* ov; device const ushort* ep; device const ushort* ec; device const uchar* ev; device const uint* eb; };

// The two scale bytes (low byte = col 2 j, high byte = col 2 j + 1) of ushort j of row n of expert ex: the ushort the expanded `meta` holds at [(ex * N + n) * COLS / 2 + j].
// COLS = scales per row (KIN / 16), N = rows per expert matrix. Layout: see kyojin_mlx/moe_scale_compact.py.
inline ushort csc_pair(CSC s, uint ex, uint N, uint COLS, uint n, uint j) {
  const ulong row = ulong(ex) * N + n;
  const uint cb = uint(s.cd[row * (COLS >> 1) + j]);
  const uint b = uint(s.bs[row]);
  uint lo = b + (cb & 15u), hi = b + (cb >> 4);
  const uint g = j >> 3;                                              // 16-col group of cols 2 j and 2 j + 1
  const uint ob = ((COLS + 15u) / 16u + 7u) >> 3;
  if (((uint(s.ov[row * ob + (g >> 3)]) >> (g & 7u)) & 1u) != 0u) {
    const ulong er = ulong(ex) * (N + 1u) + n;
    const uint e0 = s.eb[ex];
    uint a = e0 + uint(s.ep[er]);
    const uint e = e0 + uint(s.ep[er + 1]);
    for (; a < e; ++a) {
      const uint c = uint(s.ec[a]);
      if (c == 2u * j) lo = uint(s.ev[a]); else if (c == 2u * j + 1u) hi = uint(s.ev[a]);
    }
  }
  return ushort(lo | (hi << 8));
}
inline ushort csc_byte(CSC s, uint ex, uint N, uint COLS, uint n, uint c) {
  const ushort v = csc_pair(s, ex, N, COLS, n, c >> 1);
  return (c & 1u) != 0u ? ushort(v >> 8) : ushort(v & 0xFFu);
}
"""
_CSP = "csc_pair({s}, ex, uint(N), uint(KIN) / 16u, {n}, {j})"
_CSB = "csc_byte({s}, ex, uint(N), uint(KIN) / 16u, {n}, {j})"

# kernel family -> [(pointer variable, input prefix, {read text: (helper, row expr, col expr)})]
_RULES = {
    "inl": [("ma", "A", {"ma[o]": (_CSP, "rbase", "lane + 32u * i")}), ("mb", "B", {"mb[o]": (_CSP, "rbase", "lane + 32u * i")})],
    "g2m": [("ma", "A", {"ma[r * UPR + 32u * i]": (_CSP, "n0 + r", "lane + 32u * i")}), ("mb", "B", {"mb[r * UPR + 32u * i]": (_CSP, "n0 + r", "lane + 32u * i")})],
    "g2mdc": [("ma", "", {"ma[r * UPR + 32u * i]": (_CSP, "n0 + r", "lane + 32u * i")})],
    "dc": [("MTP", "", {"MTP[o]": (_CSP, "n", "lane + 32u * i")})],
    "omx": [("ma", "A", {"ma[r * (uint(KIN) / 16u) + b * 32u]": (_CSB, "n0 + r", "b * 32u + lane")}),
            ("mb", "B", {"mb[r * (uint(KIN) / 16u) + b * 32u]": (_CSB, "n0 + r", "b * 32u + lane")})],
    "omxdc": [("ma", "", {"ma[r * (uint(KIN) / 16u) + b * 32u]": (_CSB, "n0 + r", "b * 32u + lane")})],
}


def scm_source(src, kind):
    """The kernel source `src` of family `kind` with every scale read going through csc_pair / csc_byte (inputs cd<P> bs<P> ov<P> ep<P> ec<P> ev<P> eb<P> replace mt<P>)."""
    for var, pre, reads in _RULES[kind]:
        line = re.compile(r"^[ \t]*device const (?:ushort|uchar)\* %s = .*;[ \t]*$" % var, re.M)
        assert len(line.findall(src)) == 1, (kind, var)
        init = "const CSC c%s = {%s};" % (var, ", ".join(n + pre for n in NAMES))
        src = line.sub(lambda m: init, src)
        for text, (h, n, j) in reads.items():
            assert src.count(text) >= 1, (kind, text)
            src = src.replace(text, h.format(s="c" + var, n=n, j=j))
    assert not re.search(r"\b(ma|mb|MTP)\[", src), kind
    return src


def scm_names(input_names):
    """input_names with every scale input `mt<P>` replaced by its seven compact arrays."""
    out = []
    for n in input_names:
        if n in ("mt", "mtA", "mtB"):
            out += [p + n[2:] for p in NAMES]
        else:
            out.append(n)
    return out


def cm_list(d):
    """The seven compact arrays of a DevPack, in kernel input order."""
    return [d.cm[k] for k in ARRAYS]


def usable(*ds):
    return all(getattr(d, "cm", None) is not None for d in ds)


def mt(d, scm):
    """The scale input(s) of a stacked expert DevPack: its expanded `meta`, or with scm the seven compact arrays."""
    return cm_list(d) if scm else [d.meta]


def build_dev(ps):
    """dict of mx arrays (ARRAYS order + rows / cols) of the stacked compact scales of Packs `ps` (nvfp4: meta uint16 [n, k / 32] = the [n, k / 16] scale bytes)."""
    import mlx.core as mx
    cl = [pack_compact(np.ascontiguousarray(p.meta).view(np.uint8).reshape(p.n, p.k // 16)) for p in ps]
    st = stack(cl)
    out = {k: mx.array(st[k]) for k in ARRAYS}
    out["rows"], out["cols"], out["E"] = st["rows"], st["cols"], st["E"]
    return out


# ----------------------------------------------------------------------------------------------
# Transient expand (GPU): compact -> the expanded `meta` layout (uint16 [ne, rows, cols / 2]) for the paths that still read `meta` (prefill GEMMs, gather, slices).
# The result is never stored on the DevPack: it lives as long as the call that holds it.
# ----------------------------------------------------------------------------------------------
KEEP_FLAG = "KYOJIN_MOE_SCALE_KEEP_META"                              # 1: keep the expanded `meta` resident next to the compact form (staging / A-B runs)


def drop_meta():
    return enabled() and os.environ.get(KEEP_FLAG, "0") != "1"


_EXP_SRC = """
    const uint j = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const CSC s = {cdv, bsv, ovv, epv, ecv, evv, ebv};
    const uint ex = e0[0] + r / uint(N);
    y[ulong(r) * (uint(COLS) / 2u) + j] = csc_pair(s, ex, uint(N), uint(COLS), r % uint(N), j);
"""
_exp_k = None


def expand_dev(cm, rows, cols, e0=0, ne=1):
    """uint16 [ne, rows, cols / 2] = the expanded `meta` of experts e0 .. e0 + ne - 1 of the compact stack `cm` (cols = scales per row = k / 16)."""
    import mlx.core as mx
    global _exp_k
    if _exp_k is None:
        _exp_k = mx.fast.metal_kernel(name="kyojin_csc_expand", input_names=["cdv", "bsv", "ovv", "epv", "ecv", "evv", "ebv", "e0"], output_names=["y"], header=CSC_HDR, source=_EXP_SRC)
    assert cm is not None, "compact scales missing: this path needs `meta` and the DevPack has neither (KYOJIN_MOE_SCALE_COMPACT)"
    return _exp_k(inputs=[cm[a] for a in ARRAYS] + [mx.array([e0], dtype=mx.uint32)], template=[("N", rows), ("COLS", cols)], grid=(cols // 2, ne * rows, 1), threadgroup=(min(cols // 2, 32), 1, 1),
                  output_shapes=[(ne, rows, cols // 2)], output_dtypes=[mx.uint16])[0]
