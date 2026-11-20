# SPDX-License-Identifier: Apache-2.0
"""A skeleton model (foreign_skeleton.Model) on a real GGUF without decoding the file whole.

`load_cpu` decodes every tensor to one dtype: right for the synthetic checks, impossible on a big MoE (Hy4-flash: 41 x 3 expert stacks of
512 x 1536 x 1536 = 590 GB in f32). `load_stored` builds the same Model, the same parameter names (param_name) and the same strict coverage
(every tensor of a decoded layer has a parameter, every parameter a tensor), and differs only in where the weights live:

  metal=True   2-D linears of a Metal kernel type (foreign_linear rules: Q4_K / Q4_0 / IQ4_NL as stored, Q8_0 with fp8stored, Q3_K / Q5_K / Q6_K /
               IQ* with kstored, k % 256 == 0) become ForeignLinear (nn.Linear modules) or MoeFFN stored callables (shared / latent projections);
               the routed expert stacks become ForeignSwitchGLU (one stacked DevPack per projection, the projections may differ in type); every
               other tensor is decoded to `dtype`.
  metal=False  (CPU check tier) every non-expert tensor decoded to `dtype`; the routed experts are RowExperts: each chosen expert's matrices are
               decoded from the mapped file when the router first picks it (LRU bounded in bytes), so the RSS stays the dense part plus the cache.

The Engram table stays in the file, read by row (foreign_skeleton.engram_source_of), on both tiers. `layers=n` keeps the first n decoded layers
(the others are mapped and checked, not read). No architecture name appears here."""
from __future__ import annotations

import collections
import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import foreign_gguf as GF
from . import foreign_skeleton as SK
from . import foreign_skeleton_moe as SM
from .foreign_skeleton_rows import ROWS

EXPERT_KINDS = {"ffn_gate_exps": "g", "ffn_up_exps": "u", "ffn_down_exps": "d"}


def read_raw(g, name, lead=None):
    """Copy of g.raw(name) (or of its slice [lead] on the first axis) read with os.pread: no page of the mapping is touched, so the RSS holds only
    what is decoded (the CPU tier on a file larger than the memory budget)."""
    import os
    ti = g.tensors[name]
    if not ti.known:
        raise ti.type_error()
    _, be, bb = GF.TYPES_BY_NAME[ti.tname]
    shape = ti.shape[:-1] + (ti.ne[0] // be * bb,)
    src = g._owner.get(name, g)
    start, n = src.data_start + ti.offset, ti.nbytes
    if lead is not None:
        per = ti.nbytes // shape[0]
        start, n, shape = start + int(lead) * per, per, shape[1:]
    b = os.pread(src._f.fileno(), n, start)
    if len(b) != n:
        raise GF.GgufError(f"{name}: short read {len(b)} of {n} bytes")
    return np.frombuffer(b, np.uint8).reshape(shape)


def _f32(g, name, metal):
    if metal:
        return np.asarray(g.tensor_f32(name), np.float32)
    ti = g.tensors[name]
    if ti.tname in GF.ROW_ONLY:
        raise GF.GgufTypeError(f"tensor {name}: GGUF type {ti.tname} is read by row only")
    return np.asarray(GF.dequant(ti.tname, read_raw(g, name), ti.ne[0]), np.float32).reshape(ti.shape)


def stored_ok(ti) -> bool:
    """The tensor stays as stored on Metal: a type the kernels read (same flags as foreign_linear._read_gguf), k a multiple of 256."""
    from . import foreign_linear as FL
    if len(ti.ne) not in (2, 3) or ti.ne[0] % 256:
        return False
    return (ti.tname in GF.KERNEL_TYPES or (ti.tname in GF.KERNEL_TYPES8 and FL.fp8stored())
            or (ti.tname in GF.KERNEL_TYPESK and FL.kstored()))


class _Lru:
    def __init__(self, cap_bytes):
        self.d, self.cap, self.used, self.miss, self.hit = collections.OrderedDict(), int(cap_bytes), 0, 0, 0

    def get(self, key, make):
        if key in self.d:
            self.hit += 1
            self.d.move_to_end(key)
            return self.d[key]
        self.miss += 1
        v = make()
        self.d[key] = v
        self.used += sum(int(a.nbytes) for a in v)
        while self.used > self.cap and len(self.d) > 1:
            _, old = self.d.popitem(last=False)
            self.used -= sum(int(a.nbytes) for a in old)
        return v


class RowExperts:
    """CPU switch_mlp for foreign_moe_generic.GenericMoE: (x [.., H], inds [.., K]) -> [.., K, W], un-weighted. Expert e's gate / up / down are
    decoded (f32) from the mapped GGUF when first chosen; slots are grouped by expert (one matmul per chosen expert). Same act / clamp as _Stack."""

    def __init__(self, g, il, spec, lru):
        self.g, self.il, self.spec, self.lru = g, il, spec, lru
        self.names = {p: f"blk.{il}.{k}.weight" for k, p in EXPERT_KINDS.items()}
        self.gated = "g" in self.names and self.names["g"] in g.tensors

    def _dec(self, p, e):
        n = self.names[p]
        ti = self.g.tensors[n]
        return mx.array(np.asarray(GF.dequant(ti.tname, read_raw(self.g, n, e), ti.ne[0]), np.float32).reshape(ti.ne[1], ti.ne[0]))

    def weights(self, e):
        return self.lru.get((self.il, e), lambda: [self._dec(p, e) for p in (("g", "u", "d") if self.gated else ("u", "d"))])

    def __call__(self, x, inds):
        from . import foreign_moe_generic as G
        lead, H = x.shape[:-1], x.shape[-1]
        K = inds.shape[-1]
        x2 = x.reshape(-1, H).astype(mx.float32)
        ind = np.array(inds).reshape(-1, K)
        flat = ind.reshape(-1)
        tok = np.arange(flat.size) // K
        order = np.argsort(flat, kind="stable")
        ys, s = [], self.spec
        for e in np.unique(flat):
            sl = order[flat[order] == e]
            xe = x2[mx.array(tok[sl].astype(np.int32))]
            w = self.weights(int(e))
            gw, uw, dw = (w if self.gated else (None, *w))
            h = G.apply_act(s.act, None if gw is None else xe @ gw.T, xe @ uw.T, s, s.clamp, s.clamp_mode)
            ys.append(h @ dw.T)
        y = mx.concatenate(ys, axis=0)[mx.array(np.argsort(order, kind="stable").astype(np.int32))]
        return y.reshape(*lead, K, y.shape[-1]).astype(x.dtype)


def _module(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if isinstance(m, list) else (m[p] if isinstance(m, dict) and p in m else getattr(m, p))
    return m


def _switch_limit(spec):
    """ForeignSwitchGLU.limit = the pre-activation clamp (min(gate, c), clip(up, -c, c), then silu): exact for clamp_mode pre_act or no clamp."""
    if spec.clamp <= 1e-6:
        return None
    if spec.clamp_mode != "pre_act":
        raise SK.SkeletonError(f"post-activation expert clamp {spec.clamp}: ForeignSwitchGLU has the pre-activation form only")
    return float(spec.clamp)


def load_stored(path, metal=True, layers=None, dtype=mx.float32, row=None, log=print, expert_cache_gib=4.0):
    """GGUF -> (model, report). See the module docstring. The file stays open with the model (model.__dict__['_gguf']): experts / Engram rows read it."""
    from mlx.utils import tree_flatten, tree_unflatten
    from . import foreign_linear as FL
    from .foreign_skeleton_kpool import fires as kp_fires
    t0 = time.time()
    g = GF.GgufFile(GF.find_gguf(path))
    arch = g.kv["general.architecture"]
    r = ROWS.get(arch) if row is None else row
    ne = {k: ti.ne for k, ti in g.tensors.items()}
    moe = SM.detect(arch, g.kv, ne, r) if r is not None else None
    extra = moe.names if moe is not None else None
    sp = SK.build_spec(arch, g.kv, ne, row=r, allow_moe=moe is not None)
    model = SK.Model(sp, moe.factory if moe is not None else None)
    nl = sp.num_hidden_layers if layers is None else min(int(layers), sp.num_hidden_layers)
    if nl < sp.num_hidden_layers:
        model.model.layers = model.model.layers[:nl]
    kpl = {il for il, t in enumerate(sp.layer_tensors) if kp_fires(t)}
    rep = collections.Counter({k: 0 for k in ("mapped", "mapped_not_loaded", "decoded", "stored_linears", "expert_blocks", "nextn_skipped", "engram_table_rows")})
    fm = collections.Counter()
    weights, mods, experts, skipped = [], [], {}, []
    for name, ti in g.tensors.items():
        mm = re.match(r"blk\.(\d+)\.(\w+)\.", name)
        il = int(mm[1]) if mm else None
        if il is not None and il >= sp.num_hidden_layers:              # MTP / nextn blocks: not decoded layers
            rep["nextn_skipped"] += 1
            continue
        if name == SK.ENGRAM_TABLE and sp.engram_layers:
            rep["engram_table_rows"] += 1
            continue
        pn = SK.param_name(name, extra, kpl)
        if pn is None:
            skipped.append(name)
            continue
        rep["mapped"] += 1
        if il is not None and il >= nl:
            rep["mapped_not_loaded"] += 1
            continue
        kind = mm[2] if mm else None
        if kind in EXPERT_KINDS:
            experts.setdefault(il, {})[EXPERT_KINDS[kind]] = name
            continue
        mpath, leaf = pn.rsplit(".", 1)
        if leaf == "weight" and len(ti.ne) == 2 and ti.tname in ("BF16", "F16") and FL.SHARED_GEMV and not (kind in EXPERT_KINDS):    # flag shared_input_gemv: a 16-bit linear stays 16-bit (not a float32 matrix)
            mod = _module(model, mpath)
            if isinstance(mod, nn.Linear) and "bias" not in mod:
                raw = np.ascontiguousarray(g.raw(name))
                w = mx.array(raw.view(np.uint16)) if ti.tname == "BF16" else mx.array(raw.view(np.float16))
                mods.append((mpath, FL.DenseLinear(w.view(mx.bfloat16) if ti.tname == "BF16" else w, out_dtype=dtype)))
                fm["dense_" + ti.tname.lower()] += 1
                rep["stored_linears"] += 1
                continue
        if metal and leaf == "weight" and len(ti.ne) == 2 and stored_ok(ti):
            mod = _module(model, mpath)
            owner_path, attr = mpath.rsplit(".", 1)
            owner = _module(model, owner_path)
            if isinstance(mod, nn.Linear) or isinstance(owner, SM.MoeFFN):
                if "bias" in mod:
                    raise SK.SkeletonError(f"{name}: a stored linear with a bias parameter is not wired here")
                lin = FL.make_linear(GF.to_pack(ti.tname, g.raw(name), ti.ne[0]), None, out_dtype=dtype)
                fm[lin.fmt] += 1
                if isinstance(owner, SM.MoeFFN):
                    owner.set_stored(attr, lin)
                else:
                    mods.append((mpath, lin))
                rep["stored_linears"] += 1
                continue
        weights.append((pn, mx.array(_f32(g, name, metal)).astype(dtype)))
        rep["decoded"] += 1
    if skipped:
        raise SK.SkeletonError(f"{arch}: tensors with no skeleton parameter: {skipped[:8]}")
    if mods:
        model.update_modules(tree_unflatten(mods))
    lru = _Lru(expert_cache_gib * (1 << 30))
    for il, names in sorted(experts.items()):
        ffn = model.model.layers[il].mlp
        if not isinstance(ffn, SM.MoeFFN):
            raise SK.SkeletonError(f"layer {il}: expert tensors on a layer whose FFN is {type(ffn).__name__}")
        spec = ffn.moe_spec
        if metal:
            tis = {p: g.tensors[n] for p, n in names.items()}
            bad = [n for p, n in names.items() if not stored_ok(tis[p])]
            if bad:
                raise SK.SkeletonError(f"layer {il}: experts {bad} have no Metal kernel type (foreign_gguf.KERNEL_TYPES*) or k % 256")
            if spec.act != "silu" or "g" not in names:
                raise SK.SkeletonError(f"layer {il}: ForeignSwitchGLU takes gated silu experts (act {spec.act})")
            dps = []
            for p in ("g", "u", "d"):
                ti, raw = tis[p], g.raw(names[p])
                packs = FL.pmap(lambda e, raw=raw, ti=ti: GF.to_pack(ti.tname, raw[e], ti.ne[0]), range(ti.ne[2]))
                dps.append(FL.DevPack(packs))
                del packs
                fm[f"experts_{dps[-1].fmt}"] += 1
            sw = FL.ForeignSwitchGLU(*dps, limit=_switch_limit(spec))
            mx.eval([a for dp in dps for a in (dp.qs, dp.meta, dp.gs) if isinstance(a, mx.array)])
        else:
            sw = RowExperts(g, il, spec, lru)
        ffn.set_stored("switch_mlp", sw)
        rep["expert_blocks"] += 1
    have = {k for k, _ in weights}
    missing = [k for k, _ in tree_flatten(model.parameters()) if k not in have]
    if missing:
        raise SK.SkeletonError(f"{arch}: {len(missing)} parameters without a tensor, e.g. {missing[:4]}")
    model.load_weights(weights, strict=True)
    del weights
    mx.eval(model.parameters())
    if sp.engram_layers:
        model.engram_source = SK.engram_source_of(g)
    if FL.SHARED_GEMV:                                             # flag shared_input_gemv: same-input groups (decided from the module graph)
        FL.install_shared_input(model, log)
    model.__dict__["_gguf"] = g
    model.__dict__["_expert_lru"] = lru
    rep = dict(rep, arch=arch, layers_loaded=nl, layers_total=sp.num_hidden_layers, n_tensors=len(g.tensors), formats=dict(fm),
               metal=bool(metal), load_s=round(time.time() - t0, 1))
    log(f"[skeleton-stored] {arch}: {rep}")
    return model, dict(rep, spec=sp)
