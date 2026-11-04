"""Per-array ledger of one session cache (read only, no eval, never builds a mirror): which array holds which bytes, and whether the engine's bill
(`Backend.nbytes` -> `arrays_of`) counts it. A test or a tool reads `itemize(cache)` (one row per array) or `summary(cache)` (bytes by role).

Roles:
    packed      codes / scales of the fp4 pool and index-key layers, the n valid rows (counted)
    dense_layer a pool / key layer that fell off the fp4 grid and stays fp16 (counted)
    state       window ring, compressor kv_state / score_state, Engram history (counted)
    handoff     index_k, candidates, compress_kv, topk_idxs: the cross-layer handoff a forward pass publishes for its consumer layers. Every pass republishes it
                (owner layers run before their consumers), so after the pass it is dead weight that the bill still counts (counted)
    slack       the capacity of a code buffer past its n valid rows (NOT counted)
    mirror/memo dense depthflat mirror / memo of a packed layer (NOT counted)
    segment     sealed kvseg segments a live layer keeps beside its buffer (counted by `seg_bytes`)
    other       any other array reachable from the cache (counted)
`counted` = the bytes `Backend.nbytes` bills for the row. Aliased arrays (a handoff that is a slice of a mirror) are listed under their own name: the bytes of a slice are
its own, the parent buffer is listed as the mirror. For exact "what is resident" read `retained_walk` / `Engine.kv_true`."""
from __future__ import annotations

import os

import mlx.core as mx

HANDOFF = ("index_k", "candidates", "compress_kv", "topk_idxs")


def enabled() -> bool:
    """Flag kv_handoff_free (default 1, read per prefill chunk)."""
    return os.environ.get("KYOJIN_KV_HANDOFF_FREE", "1") not in ("0", "")


def _row(path, role, a, counted=True):
    return {"path": path, "role": role, "shape": tuple(a.shape), "dtype": str(a.dtype), "nbytes": int(a.nbytes), "counted": bool(counted)}


def _slot_rows(path, slots):
    from . import fp4pool as FP
    out = []
    for i in range(len(slots)):
        e = list.__getitem__(slots, i)
        p = f"{path}[{i}]"
        if e is None:
            continue
        if isinstance(e, FP._Pk):
            n_c = e.codes.shape[0] * e.n * e.codes.shape[2]
            n_s = e.scales.shape[0] * e.n * e.scales.shape[2]
            out.append({"path": p + ".codes", "role": "packed", "shape": (e.codes.shape[0], e.n, e.codes.shape[2]), "dtype": "uint8", "nbytes": n_c, "counted": True})
            out.append({"path": p + ".scales", "role": "packed", "shape": (e.scales.shape[0], e.n, e.scales.shape[2]), "dtype": "uint8", "nbytes": n_s, "counted": True})
            slack = e.codes.nbytes + e.scales.nbytes - n_c - n_s
            if slack:
                out.append({"path": p + ".slack", "role": "slack", "shape": (), "dtype": "uint8", "nbytes": int(slack), "counted": False})
            for nm in ("mirror", "memo"):
                m = getattr(e, nm)
                if m is not None:
                    out.append(_row(p + "." + nm, nm, m, False))
            if e.segs:
                out.append({"path": p + ".segs", "role": "segment", "shape": (len(e.segs),), "dtype": "uint8", "nbytes": int(e.sb), "counted": True})
        elif hasattr(e, "segs") and not hasattr(e, "codes"):                       # kvseg snapshot layer: its arrays are its segments + tail, all counted
            for j, a in enumerate(e.arrays()):
                out.append(_row(f"{p}.part{j}", "packed", a))
        elif isinstance(e, mx.array):
            out.append(_row(p, "dense_layer", e))
    return out


def itemize(cache) -> list:
    """One dict per array of `cache` (a list of cache objects or one): path, role, shape, dtype, nbytes, counted."""
    out = []
    for li, c in enumerate(cache if isinstance(cache, (list, tuple)) else [cache]):
        pre = f"c{li}"
        for k, v in list(vars(c).items()):
            if isinstance(v, mx.array):
                out.append(_row(f"{pre}.{k}", "handoff" if k in HANDOFF else "other", v))
            elif hasattr(v, "raw_arrays") and isinstance(v, list):
                out += _slot_rows(f"{pre}.{k}", v)
            elif isinstance(v, (list, tuple)):
                for j, a in enumerate(v):
                    if isinstance(a, mx.array):
                        out.append(_row(f"{pre}.{k}[{j}]", "state" if k in ("window", "kv_state", "score_state") else "other", a, k not in ("_compress_buffers", "_key_buffers")))
                    elif a is not None and hasattr(a, "__dict__"):
                        for kk, vv in vars(a).items():
                            if isinstance(vv, mx.array):
                                out.append(_row(f"{pre}.{k}[{j}].{kk}", "other", vv))
            elif v is not None and hasattr(v, "__dict__") and not isinstance(v, type):
                for kk, vv in vars(v).items():
                    if isinstance(vv, mx.array):
                        out.append(_row(f"{pre}.{k}.{kk}", "state", vv))
                    elif isinstance(vv, (list, tuple)):
                        out += [_row(f"{pre}.{k}.{kk}[{j}]", "state", a) for j, a in enumerate(vv) if isinstance(a, mx.array)]
    return out


def summary(cache, tokens: int | None = None) -> dict:
    """{role: bytes} plus `counted` (what the bill sums) and, with `tokens`, the same per token."""
    rows = itemize(cache)
    by: dict = {}
    for r in rows:
        by[r["role"]] = by.get(r["role"], 0) + r["nbytes"]
    by["counted"] = sum(r["nbytes"] for r in rows if r["counted"])
    by["uncounted"] = sum(r["nbytes"] for r in rows if not r["counted"])
    if tokens:
        by["per_token"] = {k: round(v / tokens, 2) for k, v in by.items() if k != "per_token"}
    return by


def handoff_bytes(cache) -> int:
    """Bytes the bill counts for the four handoff fields of every layer cache."""
    return sum(r["nbytes"] for r in itemize(cache) if r["role"] == "handoff")


def free_handoff(cache) -> int:
    """Drop the four cross-layer handoff fields of every cache object; returns the bytes the bill counted for them. Exact: every forward pass republishes the handoff in
    layer order before a consumer reads it (vendor `compress_kv` / `index_k` / `topk_idxs` are written by the owner layers each pass; dsv4_core clears and rebuilds them)."""
    n = 0
    for c in cache if isinstance(cache, (list, tuple)) else [cache]:
        if not hasattr(c, "reset_handoff"):
            continue
        for k in HANDOFF:
            v = getattr(c, k, None)
            if isinstance(v, mx.array):
                n += v.nbytes
        c.reset_handoff()
    return n
