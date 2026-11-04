# SPDX-License-Identifier: Apache-2.0
"""Load-time restack of a per-expert EXL3 pack (the layout of the HF pack yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz) into the stacked layout of
tools/convert_stack.py, on lazy mx arrays: same names, dtypes, shapes and bytes, so the patcher, the MoE kernels and the grouped wo_a see no
difference and no conversion pass (and no second copy of the pack on disk) is needed.

  <p>.experts.<E>.<proj>.{trellis,suh,svh,mul1|mcg}  ->  <p>.experts.<proj>.{trellis U32 [W], offset U32 [n+1], k U8 [n], suh F16 [n,k],
                                                        svh F16 [n,nout], mul1|mcg I32 [], shape I32 [2]}
  <p>.slice.<S>.{trellis,suh,svh,mul1|mcg}           ->  <p>.slices.{trellis I16 [S,kt,nt,16K], suh F16 [S,k], svh F16 [S,nout], mul1|mcg I32 []}

Offsets and bits come from the lazy shapes (nothing is read for them). The per-expert entries are removed from the dict: once a stack is
evaluated, the only reference to the per-expert buffers was the concatenate graph, so memory holds one copy of the weights, not two."""
from __future__ import annotations

import re

import mlx.core as mx

EXP = re.compile(r"^(.*)\.experts\.(\d+)\.(.+)\.(trellis|suh|svh|mul1|mcg)$")
SLC = re.compile(r"^(.*)\.slice\.(\d+)\.(trellis|suh|svh|mul1|mcg)$")


def needed(names) -> bool:
    """True when the names hold a per-expert or per-slice layout."""
    return any(EXP.match(k) or SLC.match(k) for k in names)


def _flag(t: dict, names: list[str], where: str) -> mx.array:
    """The codebook flag of a group (one value for every member, checked with one stacked compare)."""
    f = [t[n].reshape(()) for n in names]
    st = mx.stack(f)
    ok = mx.all(st == st[0])
    mx.eval(ok, st)
    assert bool(ok.item()), f"{where}: codebook flag differs between members"
    return mx.array(st[0].item(), dtype=mx.int32)


def _key(parts: dict, where: str) -> str:
    key = "mul1" if "mul1" in parts else "mcg" if "mcg" in parts else None
    assert key, f"{where}: codebook flag tensor missing"
    return key


def _experts(t: dict, prefix: str, proj: str, parts: dict, out: dict) -> None:
    n = len(parts)
    where = f"{prefix}.experts.*.{proj}"
    assert sorted(parts) == list(range(n)), f"{where}: expert ids not contiguous"
    for e in range(n):
        assert {"trellis", "suh", "svh"} <= set(parts[e]), f"{prefix}.experts.{e}.{proj}: missing parts"
    fk = _key(parts[0], where)
    flag = _flag(t, [parts[e][fk] for e in range(n)], where)
    ks, off, words, kt, nt = [], [0], [], None, None
    s_suh, s_svh = tuple(t[parts[0]["suh"]].shape), tuple(t[parts[0]["svh"]].shape)
    for e in range(n):
        tr = t[parts[e]["trellis"]]
        assert tr.dtype == mx.int16 and tr.ndim == 3 and tr.shape[2] % 16 == 0, f"{parts[e]['trellis']}: {tr.dtype} {tr.shape}"
        if kt is None:
            kt, nt = tr.shape[0], tr.shape[1]
        assert (tr.shape[0], tr.shape[1]) == (kt, nt), f"{parts[e]['trellis']}: tile grid {tr.shape[:2]} differs from {(kt, nt)}"
        su, sv = t[parts[e]["suh"]], t[parts[e]["svh"]]
        assert tuple(su.shape) == s_suh and tuple(sv.shape) == s_svh and su.dtype == mx.float16 and sv.dtype == mx.float16, \
            f"{where}: suh/svh differ at expert {e}"
        ks.append(tr.shape[2] // 16)
        off.append(off[-1] + tr.size // 2)
        words.append(tr.reshape(-1).view(mx.uint32))
    assert off[-1] < 2 ** 32, f"{where}: {off[-1]} words does not fit a uint32 offset"
    base = f"{prefix}.experts.{proj}"
    out[base + ".trellis"] = mx.concatenate(words)
    out[base + ".offset"] = mx.array(off, dtype=mx.uint32)
    out[base + ".k"] = mx.array(ks, dtype=mx.uint8)
    out[base + ".suh"] = mx.stack([t[parts[e]["suh"]] for e in range(n)])
    out[base + ".svh"] = mx.stack([t[parts[e]["svh"]] for e in range(n)])
    out[base + "." + fk] = flag
    out[base + ".shape"] = mx.array([kt, nt], dtype=mx.int32)


def _slices(t: dict, prefix: str, parts: dict, out: dict) -> None:
    n = len(parts)
    where = f"{prefix}.slice.*"
    assert sorted(parts) == list(range(n)), f"{where}: ids not contiguous"
    fk = _key(parts[0], where)
    flag = _flag(t, [parts[s][fk] for s in range(n)], where)
    base = f"{prefix}.slices"
    for k in ("trellis", "suh", "svh"):
        arrs = [t[parts[s][k]] for s in range(n)]
        assert len({(a.dtype, tuple(a.shape)) for a in arrs}) == 1, f"{where}.{k}: shapes differ"
        out[f"{base}.{k}"] = mx.stack(arrs)
    assert out[f"{base}.trellis"].dtype == mx.int16
    out[f"{base}.{fk}"] = flag


def restack(t: dict) -> dict:
    """New dict: every per-expert / per-slice group replaced by its stacked tensors (lazy), all other entries as they are."""
    eg, sg, used = {}, {}, set()
    for k in t:
        m = EXP.match(k)
        if m:
            eg.setdefault((m[1], m[3]), {}).setdefault(int(m[2]), {})[m[4]] = k
            used.add(k)
            continue
        m = SLC.match(k)
        if m:
            sg.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = k
            used.add(k)
    if not used:
        return t
    out = {k: v for k, v in t.items() if k not in used}
    for (prefix, proj), parts in sorted(eg.items()):
        _experts(t, prefix, proj, parts, out)
    for prefix, parts in sorted(sg.items()):
        _slices(t, prefix, parts, out)
    return out
