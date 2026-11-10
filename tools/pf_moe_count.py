# SPDX-License-Identifier: Apache-2.0
"""CPU-side counter of the prefill MoE path (cmoe8k): launches, syncs, allocations, threadgroups and weight decodes per chunk.

Runs the REAL `MoePrefill` code (kyojin_mlx/moe_prefill.py, metal_nax.py, metal_nax_gu.py) at the served shape (hidden 5120, inter 2304, 384 experts,
top-6, 2-4 bitrate classes) with `mx.fast.metal_kernel` replaced by a recorder that returns lazy zeros, and `moe_sort.plan_device` replaced by a
numpy twin. Nothing is timed and no kernel runs: the numbers are COUNTS of what the host path asks the GPU to do, plus a model of the MLX buffer
cache. The weights are tiny stand-ins (the byte counts come from the shapes and bitrates, not from the arrays).

  python tools/pf_moe_count.py [--layers 20] [--rows 8192] [--arms ...] [--ks 2,3,3,4] [--json out.json]

Rows per expert come from `served_counts`: a model fitted to the numbers measured on the lane (oppfcfg,
opskew): per 4096-row pass 355 of 384 experts hold <= 64 rows, those hold ~54.5 % of the pairs, the 29 others hold the rest, the busiest holds ~1400.
A call of 8192 rows is two independent 4096-row draws from the same expert popularity (not a rescaling).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mlx.core as mx  # noqa: E402

from kyojin_mlx import metal_nax as NX, metal_nax_gu as GU, moe as M, moe_prefill as MP, moe_sort as MS  # noqa: E402

E_SERVED, TOPK, HIDDEN, INTER = 384, 6, 5120, 2304
MLX_PAGE = 16384                                   # vm_page_size of Apple silicon; MLX BufferCache reuses a cached buffer c for a request s iff s <= c < min(2 s, s + 2 pages)


# ------------------------------------------------------------------------------------------------------------- served routing model
def served_targets(E: int = E_SERVED, topk: int = TOPK, T: int = 4096, n_hot: int | None = None, cold_share: float = 0.545, hot_max: int | None = None, seed: int = 0, cold_cap: float = 44.0):
    """Target rows per expert for one T-row pass: E - n_hot cold experts of 1..64 rows holding cold_share of the pairs, n_hot hot experts (zipf, busiest hot_max)."""
    rng = np.random.default_rng(seed)
    P = T * topk
    n_hot = max(2, round(29 * E / E_SERVED)) if n_hot is None else n_hot       # a smaller E (tests) keeps the served shares: hot fraction, busiest = 34 % of the rows
    hot_max = round(1402 * T / 4096) if hot_max is None else hot_max
    nc = E - n_hot
    c = rng.gamma(1.2, 1.0, nc)
    for _ in range(40):                              # rescale to the cold total with every expert in [1, 64]
        c = np.clip(c * (cold_share * P / c.sum()), 1.0, cold_cap)
    lo, hi = 0.0, 3.0
    for _ in range(60):                              # zipf exponent s with the busiest = hot_max and the hot total = P - cold total
        s = (lo + hi) / 2
        w = np.arange(1, n_hot + 1, dtype=np.float64) ** -s
        top = (P - c.sum()) * w[0] / w.sum()
        lo, hi = (s, hi) if top < hot_max else (lo, s)
    w = np.arange(1, n_hot + 1, dtype=np.float64) ** -s
    h = (P - c.sum()) * w / w.sum()
    return np.concatenate([h, c])


def _draw(pop: np.ndarray, T: int, topk: int, rng) -> np.ndarray:
    """T rows of topk DISTINCT experts each, expert e drawn with weight pop[e] (Gumbel top-k)."""
    keys = np.log(pop)[None, :] + rng.gumbel(size=(T, len(pop)))
    return np.argpartition(-keys, topk - 1, axis=1)[:, :topk].astype(np.int32)


def served_popularity(E: int = E_SERVED, topk: int = TOPK, T: int = 4096, seed: int = 0, iters: int = 14) -> np.ndarray:
    """Expert popularity whose T-row draws land on `served_targets` (iterative proportional fitting: a row cannot hit an expert twice, so a hot expert saturates)."""
    tgt = served_targets(E, topk, T, seed=seed)
    pop = tgt / tgt.sum()
    rng = np.random.default_rng(seed + 1)
    for _ in range(iters):
        cnt = np.bincount(_draw(pop, T, topk, rng).reshape(-1), minlength=E).astype(np.float64)
        pop = pop * np.clip(tgt / np.maximum(cnt, 1.0), 0.5, 2.0) ** 0.8
        pop /= pop.sum()
    return pop


def route(pop: np.ndarray, T: int, topk: int, seed: int, layer: int, piece: int = 4096) -> np.ndarray:
    """ids [T, topk] for one layer: the popularity is permuted per layer (a different set of experts is hot at each layer); T > 4096 = independent 4096-row draws."""
    E = len(pop)
    perm = np.random.default_rng(10_000 + layer).permutation(E)
    p = pop[perm]
    rng = np.random.default_rng(seed * 1000 + layer)
    parts = [_draw(p, min(piece, T - a), topk, rng) for a in range(0, T, piece)]
    return np.concatenate(parts, axis=0)


def hist_stats(cnt: np.ndarray) -> dict:
    act = cnt[cnt > 0]
    return dict(active=int(len(act)), le64=int((act <= 64).sum()), pairs_le64=float(act[act <= 64].sum() / act.sum()), max=int(act.max()),
                blocks64=int(np.ceil(act / 64).sum()))


# ------------------------------------------------------------------------------------------------------------------------ recorder
class Rec:
    """Kernel launches, host syncs, uploads and concatenations of one `MoePrefill` call, in order."""

    def __init__(self):
        self.launches: list[dict] = []
        self.syncs = 0
        self.uploads = 0
        self.concats: list[int] = []
        self.concat_epochs: list[int] = []                 # number of syncs before each concatenation (buffers die at the next sync, see cache_sim)

    def kernel_factory(self, name="", header="", input_names=(), output_names=(), source="", **kw):
        meta = dict(name=name, inputs=list(input_names), source=source)
        rec = self

        class K:
            def __call__(self, inputs, grid, threadgroup, output_shapes, output_dtypes, **kw2):
                outs = [mx.zeros(tuple(s), dtype=d) for s, d in zip(output_shapes, output_dtypes)]
                rec.launches.append(dict(meta, epoch=rec.syncs, ins=dict(zip(meta["inputs"], inputs)), grid=tuple(grid), tg=tuple(threadgroup),
                                         out_bytes=[int(np.prod(s)) * d.size for s, d in zip(output_shapes, output_dtypes)]))
                return outs

        return K()


class Proxy:
    """`mx` as seen by moe_prefill: counts eval / async_eval, host uploads (mx.array of host data) and concatenations."""

    def __init__(self, rec: Rec):
        self._rec = rec

    def eval(self, *a):
        self._rec.syncs += 1
        return mx.eval(*a)

    def async_eval(self, *a):
        self._rec.syncs += 1
        return mx.async_eval(*a)

    def array(self, *a, **k):
        self._rec.uploads += 1
        return mx.array(*a, **k)

    def concatenate(self, arrs, axis=0):
        out = mx.concatenate(arrs, axis=axis)
        self._rec.concats.append(int(np.prod(out.shape)) * out.dtype.size)
        self._rec.concat_epochs.append(self._rec.syncs)
        return out

    def __getattr__(self, k):
        return getattr(mx, k)


def plan_twin(ids, wts, experts, topk, rank=None):
    """numpy twin of moe_sort.plan_device (same outputs, same rank order)."""
    ids_np, wts_np = np.array(ids).astype(np.int64), np.array(wts)
    r = np.arange(experts) if rank is None else np.array(rank).astype(np.int64)
    pr = r[ids_np]
    cnt = np.bincount(pr, minlength=experts)
    cum = np.concatenate([[0], np.cumsum(cnt)]).astype(np.uint32)
    order = np.argsort(pr, kind="stable")
    inv = np.empty(order.size, np.uint32)
    inv[order] = np.arange(order.size, dtype=np.uint32)
    return (mx.array((order // topk).astype(np.uint32)), mx.array(ids_np[order].astype(np.int32)), mx.array(wts_np[order].astype(np.float32)),
            mx.array(inv), mx.array(cum))


# --------------------------------------------------------------------------------------------------------------------- served stand-in
def make_moe(E: int = E_SERVED, ks=(2, 3, 3, 4), hidden: int = HIDDEN, inter: int = INTER, seed: int = 0):
    """MoeGather with the served shapes and tiny words. Bitrates dealt by a seeded shuffle (3 projections share one K per expert, so the classes are the K values)."""
    rng = np.random.default_rng(seed)
    kk = np.array([ks[i % len(ks)] for i in range(E)], np.uint8)
    rng.shuffle(kk)

    def stack(kin, nout):
        kt, nt = kin // 16, nout // 16
        return M.Stack(mx.zeros((64,), dtype=mx.uint32), mx.zeros((E + 1,), dtype=mx.uint32), mx.array(kk), mx.zeros((E, kin), dtype=mx.float16),
                       mx.zeros((E, nout), dtype=mx.float16), kt, nt, "mul1")
    g, u, d = stack(hidden, inter), stack(hidden, inter), stack(inter, hidden)
    return M.MoeGather(g, u, d, TOPK)


ARMS = {                                          # name -> (chunk rows per pass, module settings)
    "base4096": (4096, {}),
    "chunk8192": (8192, {}),
    "nosync4096": (4096, dict(MOE_NOSYNC=True)),
    "nosync8192": (8192, dict(MOE_NOSYNC=True)),
    "nosync_rowend4096": (4096, dict(MOE_NOSYNC=True, NOSYNC_ROWEND=True)),
    "nosync_rowend8192": (8192, dict(MOE_NOSYNC=True, NOSYNC_ROWEND=True)),
    "rowmap": (4096, dict(MOE_ROWMAP=True)),
    "epi": (4096, dict(MOE_EPI=True)),
    "epi_cap": (4096, dict(MOE_EPI=True, EPI_CAP=True)),
    "rowmap+epi": (4096, dict(MOE_ROWMAP=True, MOE_EPI=True)),
    "rowmap_rm64": (4096, dict(MOE_ROWMAP=True, RM_ROWS=64)),
}


# ----------------------------------------------------------------------------------------------------------------------- launch stats

def classify(name: str) -> str:
    if name.startswith("kyojin_rotpf"):
        return "rot"
    if name.startswith("kyojin_midpf"):
        return "mid"
    if name.startswith("kyojin_finpf"):
        return "final"
    if name.startswith("kyojin_sort"):
        return "sort"
    if name.startswith("kyojin_gu_"):
        return "gemm_gu_epi"
    if name.startswith("kyojin_nax_"):
        return "gemm"
    return "other"


def gemm_stats(L: dict) -> dict:
    """Threadgroups launched / live, row blocks (= weight decodes of the whole matrix), longest serial block chain and decoded weight bytes of one GEMM launch.
    Reads the row ranges the host passed (`rowoff`) and applies the kernel's own indexing (read from its source text)."""
    name, src, ins = L["name"], L["source"], L["ins"]
    persist = re.search(r"_pl(?=_)", name) is not None
    gu = name.startswith("kyojin_gu_")
    eids = np.array(ins["eids"]).astype(np.int64)
    rowoff = np.array(ins["rowoff"]).astype(np.int64)
    GL = len(eids)
    tgx, gy, gz = L["grid"][0] // L["tg"][0], L["grid"][1], L["grid"][2]
    if gu:
        m = re.match(r"kyojin_gu_\w+?_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(rm|a)_\d+$", name)
        Kg, Ku, bmu, nmb = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        nproj, z_units, kt, nt = 2, GL, int(np.array(ins["params"])[0]), int(np.array(ins["params"])[1])
        Kz = [(Kg, Ku)] * GL
    else:
        m = re.match(r"kyojin_nax_\w+?_p(\d)_(\d+)_(\d+)_(\d+)_(\d+)_", name)
        nproj, bmu, nmb = int(m[1]), int(m[2]), int(m[3])
        kt, nt = int(np.array(ins["params"])[0]), int(np.array(ins["params"])[1])
        kk = [np.array(ins[f"kk{i}"]).astype(np.int64) for i in range(nproj)]
        Kz = [tuple(int(k[e]) for k in kk) for e in eids]
    rows = bmu * nmb
    rowend = "const uint zo = z + (z >= GL" in src
    assert gz == (GL if gu else nproj * GL)
    M_ = np.zeros(gz, np.int64)
    for z in range(gz):
        zo = z + (1 if rowend and z >= GL else 0)
        M_[z] = rowoff[zo + 1] - rowoff[zo]
    M_ = np.maximum(M_, 0)
    blocks = -(-M_ // rows)                           # row blocks per z
    if gu:
        per_z_bytes = np.array([kt * nt * 32 * (Kz[z][0] + Kz[z][1]) for z in range(gz)])
    else:
        per_z_bytes = np.array([kt * nt * 32 * Kz[z % GL][z // GL] for z in range(gz)])
    tot_tg = tgx * gy * gz
    if persist:
        live_tg = int((np.minimum(blocks, gy)).sum()) * tgx           # a TG with no block to run returns at once; the others loop
        chain = int((-(-blocks // gy)).max()) if gz else 0
    else:
        live_tg = int(blocks.sum()) * tgx
        chain = 1 if blocks.max() > 0 else 0
    return dict(tgx=tgx, rm="_rm" in name[-12:] or name.endswith("rm") or "_rm_" in name, max_blocks=int(blocks.max()) if gz else 0, tg=tot_tg, live_tg=live_tg, dead_tg=tot_tg - live_tg, row_blocks=int(blocks.sum()), chain=chain, rows=int(M_.sum()),
                useful_rows=int(M_.sum()), decoded_bytes=int((blocks * per_z_bytes).sum()), sweep_bytes=int(per_z_bytes[(M_ > 0)].sum()),
                persist=persist, rowend=rowend, row_block_rows=rows)


def summarize(rec: Rec) -> dict:
    out = dict(launches=len(rec.launches), by={}, syncs=rec.syncs, uploads=rec.uploads, concat_bytes=sum(rec.concats), concat_n=len(rec.concats),
               alloc_bytes=0, allocs=0, gemm=dict(tg=0, live_tg=0, dead_tg=0, row_blocks=0, chain_max=0, chain_sum=0, decoded_bytes=0, sweep_bytes=0, n=0, max_blocks=0, xform_x=0),
               mid_dead_rows=0)
    for L in rec.launches:
        c = classify(L["name"])
        out["by"][c] = out["by"].get(c, 0) + 1
        out["alloc_bytes"] += sum(L["out_bytes"])
        out["allocs"] += len(L["out_bytes"])
        if c in ("gemm", "gemm_gu_epi"):
            s = gemm_stats(L)
            g = out["gemm"]
            for k in ("tg", "live_tg", "dead_tg", "row_blocks", "decoded_bytes", "sweep_bytes"):
                g[k] += s[k]
            g["n"] += 1
            g["max_blocks"] = max(g["max_blocks"], s["max_blocks"])
            if s["rm"]:
                g["xform_x"] = max(g["xform_x"], s["tgx"])         # N-group threadgroups that each redo the input transform of a row block
            g["chain_max"] = max(g["chain_max"], s["chain"])
            g["chain_sum"] += s["chain"]
        elif c == "mid" and "rows" in L["ins"]:
            lo_hi = np.array(L["ins"]["rows"]).astype(np.int64)
            out["mid_dead_rows"] += int(L["grid"][1] - (lo_hi[1] - lo_hi[0]))
    out["alloc_bytes"] += rec.concats and sum(rec.concats) or 0
    out["allocs"] += len(rec.concats)
    out["sizes"] = [b for L in rec.launches for b in L["out_bytes"]] + list(rec.concats)
    last = rec.syncs
    ep: dict = {}
    for L in rec.launches:                                  # per sync interval: (buffers freed at the next sync, buffers that live to the end of the call)
        keep = classify(L["name"]) == "final" and L["epoch"] < last
        ep.setdefault(L["epoch"], ([], []))[1 if keep else 0].extend(L["out_bytes"])
    for b, e in zip(rec.concats, rec.concat_epochs):
        ep.setdefault(e, ([], []))[0].append(b)
    out["epochs"] = [ep[k] for k in sorted(ep)]
    return out


def cache_sim(layers: list, min_bytes: int = 1 << 20, tail: int = 20) -> dict:
    """MLX BufferCache replayed over the layers of a chunk. A layer is a list of sync intervals (freed_at_next_sync, kept_to_layer_end) size lists (or a flat
    size list = one interval): the sync path evals every pass, so a pass's intermediates are back in the cache before the next pass allocates, while the
    pass output waits for the concatenation. A request s reuses the smallest cached c >= s with c < min(2 s, s + 2 pages). Fresh allocations / bytes in
    layer 1, in layers 2.. (steady state) and in the last `tail` layers (the chunk after the cache has seen the earlier ones)."""
    free: list[int] = []
    res = dict(first_bytes=0, first_n=0, steady_bytes=0, steady_n=0, steady_hit_n=0, tail_bytes=0, tail_n=0)
    cur = [0]

    def take(s, li):
        if s < min_bytes:
            return None
        cand = [c for c in free if s <= c < min(2 * s, s + 2 * MLX_PAGE)]
        if cand:
            c = min(cand)
            free.remove(c)
            if li:
                res["steady_hit_n"] += 1
            return c
        k = "first" if li == 0 else "steady"
        res[f"{k}_bytes"] += s
        res[f"{k}_n"] += 1
        if cur[0] >= len(layers) - tail:
            res["tail_bytes"] += s
            res["tail_n"] += 1
        return s

    for li, layer in enumerate(layers):
        cur[0] = li
        epochs = [(layer, [])] if layer and isinstance(layer[0], int) else layer
        kept = []
        for tmp, keep in epochs:
            got = [c for c in (take(s, li) for s in tmp) if c is not None]
            kept += [c for c in (take(s, li) for s in keep) if c is not None]
            free += got
        free += kept
    return res


# ------------------------------------------------------------------------------------------------------------------------------- driver
def run_arm(arm: str, rows: int = 8192, layers: int = 20, ks=(2, 3, 3, 4), seed: int = 0, moe=None, pop=None, base_pass: int = 4096) -> dict:
    """One arm over `layers` layers of a `rows`-row call. base_pass = the rows of a default MoE pass (4096 served; tests use a smaller shape and scale `rows` with it)."""
    chunk, sets = ARMS[arm]
    chunk = chunk * base_pass // 4096
    moe = moe or make_moe(ks=ks)
    E = moe.g.experts
    hidden = moe.g.kin
    pop = served_popularity(E, TOPK, base_pass, seed) if pop is None else pop
    saved = {k: getattr(MP, k) for k in ("MOE_ROWMAP", "MOE_EPI", "MOE_NOSYNC", "NOSYNC_ROWEND", "RM_ROWS", "HOIST", "ONE_PASS", "EPI_CAP")}
    saved_site = (NX.site_on, NX.nax_available)
    saved_ps = MS.plan_device
    saved_mk = mx.fast.metal_kernel
    rec = Rec()
    try:
        for k, v in {**dict(MOE_ROWMAP=False, MOE_EPI=False, MOE_NOSYNC=False, NOSYNC_ROWEND=False, RM_ROWS=0, HOIST=False, ONE_PASS=False, EPI_CAP=False), **sets}.items():
            setattr(MP, k, v)
        NX.site_on, NX.nax_available = (lambda s: True), (lambda: True)
        MS.plan_device = plan_twin
        mx.fast.metal_kernel = rec.kernel_factory
        for mod in (MP, NX, GU, MS):
            getattr(mod, "_kc").clear()
        NX._params.clear()
        MP.mx = Proxy(rec)
        pf = MP.MoePrefill(moe, chunk=chunk)
        if chunk != 4096 * base_pass // 4096 and chunk > MP.PASS_CAP:
            raise ValueError("chunk above PASS_CAP")
        assert pf.fused and pf.h
        x = mx.zeros((rows, hidden), dtype=mx.float16)
        wts = mx.zeros((rows, TOPK), dtype=mx.float32) + 0.25
        per_layer, counts_hist = [], []
        tot = None
        for l in range(layers):
            rec.launches.clear(), rec.concats.clear(), rec.concat_epochs.clear()
            rec.syncs = rec.uploads = 0
            ids_np = route(pop, rows, TOPK, seed, l, base_pass)
            counts_hist.append(np.bincount(ids_np.reshape(-1), minlength=len(pop)))
            y = pf(x, mx.array(ids_np), wts)
            s = summarize(rec)
            s["bailed"] = pf.bailed
            per_layer.append(s)
        sizes = [s.pop("sizes") for s in per_layer]
        epochs = [s.pop("epochs") for s in per_layer]
        agg = dict(arm=arm, rows=rows, layers=layers, chunk=chunk)
        for k in ("launches", "syncs", "uploads", "alloc_bytes", "allocs", "concat_bytes", "concat_n", "mid_dead_rows"):
            agg[k] = sum(s[k] for s in per_layer)
        agg["by"] = {}
        for s in per_layer:
            for k, v in s["by"].items():
                agg["by"][k] = agg["by"].get(k, 0) + v
        agg["gemm"] = {k: sum(s["gemm"][k] for s in per_layer) for k in ("tg", "live_tg", "dead_tg", "row_blocks", "decoded_bytes", "sweep_bytes", "n", "chain_sum")}
        agg["gemm"]["chain_max"] = max(s["gemm"]["chain_max"] for s in per_layer)
        agg["gemm"]["max_blocks"] = max(s["gemm"]["max_blocks"] for s in per_layer)
        agg["gemm"]["xform_x"] = max(s["gemm"]["xform_x"] for s in per_layer)
        agg["max_buffer"] = max(max(sz) for sz in sizes)
        agg["cache"] = cache_sim(epochs)
        agg["bailed"] = per_layer[-1]["bailed"]
        agg["hist"] = hist_stats(counts_hist[0])
        return agg
    finally:
        for k, v in saved.items():
            setattr(MP, k, v)
        NX.site_on, NX.nax_available = saved_site
        MS.plan_device = saved_ps
        mx.fast.metal_kernel = saved_mk
        MP.mx = mx


def table(res: list[dict]) -> str:
    G = 2 ** 30
    cols = ["arm", "launches", "syncs", "uploads", "allocs", "alloc GiB", "concat GiB", "fresh GiB (steady)", "fresh GiB (last 20 layers)", "tg", "dead tg", "row blocks", "decoded GiB", "chain max", "chain sum", "max blk/exp", "xform x", "bailed"]
    rows = [cols]
    for r in res:
        g = r["gemm"]
        rows.append([r["arm"], r["launches"], r["syncs"], r["uploads"], r["allocs"], f"{r['alloc_bytes'] / G:.2f}", f"{r['concat_bytes'] / G:.2f}",
                     f"{r['cache']['steady_bytes'] / G:.2f}", f"{r['cache']['tail_bytes'] / G:.2f}", g["tg"], g["dead_tg"], g["row_blocks"], f"{g['decoded_bytes'] / G:.1f}", g["chain_max"], g["chain_sum"], g["max_blocks"], g["xform_x"], r["bailed"]])
    w = [max(len(str(x[i])) for x in rows) for i in range(len(cols))]
    return "\n".join("  ".join(str(c).rjust(w[i]) for i, c in enumerate(r)) for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=20, help="MoE layers per chunk (the CED encoder half)")
    ap.add_argument("--rows", type=int, default=8192, help="rows of the MoE call (one CED chunk)")
    ap.add_argument("--arms", default="base4096,chunk8192,nosync4096,nosync_rowend4096,nosync8192,nosync_rowend8192,rowmap,epi,rowmap+epi,rowmap_rm64")
    ap.add_argument("--ks", default="2,3,3,4", help="bitrates dealt to the experts (classes = distinct values)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ks = tuple(int(v) for v in a.ks.split(","))
    moe = make_moe(ks=ks)
    pop = served_popularity(seed=a.seed)
    h = hist_stats(np.bincount(route(pop, 4096, TOPK, a.seed, 0).reshape(-1), minlength=E_SERVED))
    print(f"routing model, one 4096-row pass: {h}")
    res = [run_arm(arm, a.rows, a.layers, ks, a.seed, moe, pop) for arm in a.arms.split(",")]
    print(table(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
