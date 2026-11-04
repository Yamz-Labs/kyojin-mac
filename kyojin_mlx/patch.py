# SPDX-License-Identifier: Apache-2.0
"""S1 module surgery: `patch_model(model, pack_dir)` turns an mlx-vlm / mlx-lm model built from an EXL3 pack config into a model that
runs on the Kyojin Metal kernels. Rule based on the module type found in the model tree; the per-architecture table is in archs.py.

  nn.Linear                       -> Exl3Linear (GEMV 1-4 rows, decode + matmul above)
  head class (ParallelHead)       -> Exl3Linear with fp32 output
  MultiLinear (DS4.1 wo_a slices) -> Exl3MultiLinear (batch 1, <= 4 rows: grouped GEMV, 3 launches for all groups; else one Exl3Linear per group)
  fused gate_up / qkv_a           -> ConcatLinear of the split pack linears (each member has its own suh / svh)
  routed experts                  -> MoeGather (5 launches per layer), wrapped for the model's calling convention
  dense tensors                   -> model.sanitize + load_weights (bf16 -> fp16), every parameter must be covered
Pack tensors are never touched: sidecars (.suh .svh .mul1 .mcg .offset .k .shape) are consumed with their trellis, the vision
tower and MTP heads are skipped (counted in the report). The patched model computes in fp16 (Exl3Linear output) with fp32 routing,
MoE reduce and (DS4.1) head. Prefill beyond 4 rows: dense linears use decode + matmul, MoE runs in chunks of 4 rows (correct,
slow) below KYOJIN_PF_MIN_ROWS (default 14 * experts / top-k rows, 2.5 * experts / top-k with the fused NAX path of metal_nax) rows; from there on the routed experts run the S3 prefill path (moe_prefill.py: tokens grouped by
expert, decode to fp16 per expert group + gather_mm)."""
from __future__ import annotations

import glob
import importlib
import json
import os
import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from . import moe_sumoe as SU, exl3_linear as EL, headlast as HL, fqfuse as FQF, kernels as KN, metal_nax as NX, moe as M, tuning as TN
from . import fired as FD
from . import archs as _AR
from .archs import ArchSpec, FUSED, FUSED_ORDER, activate, arch_of, package, target
from .exl3_linear import Exl3Linear, GEMV_MAX_ROWS
from .grouped import GroupedGemv

SIDE = (".suh", ".svh", ".mul1", ".mcg", ".offset", ".k", ".shape", ".bias")
MOE_MAX_ROWS = int(os.environ.get("KYOJIN_MOE_ROWS", "4"))
PF_MIN_ROWS = os.environ.get("KYOJIN_PF_MIN_ROWS")                    # rows in one MoE call from which S3 runs; default = PF_CROSS * experts / top-k
GROUPED_WOA = os.environ.get("KYOJIN_GROUPED_WOA", "1") != "0"           # DS4.1 wo_a: one grouped GEMV instead of one per slice
PF_CROSS = float(os.environ.get("KYOJIN_PF_CROSS", "14"))             # Air (M5, 10 cores): S3 = decode chunks at about 14 * E / topk rows (DS4.1 900, GLM 500)
_CROSS_NAX = os.environ.get("KYOJIN_PF_CROSS_NAX")
# with the fused EXL3 + NAX path (metal_nax) S3 wins from about 2.5 rows per expert on the Air (DS4.1 160 tokens, GLM 80) and 1.75 on the 64-core Ultra
# (tools/pfcross.py vs the MTP decode calls of MAX_PAIRS // top-k rows; DS4.1: tie at 1.5, x1.19 at 2; GLM-5.3: x0.94 at 1.5, x1.07 at 2)
PF_CROSS_NAX = float(_CROSS_NAX) if _CROSS_NAX else (1.75 if TN.gpu_cores() >= 32 else 2.5)


# --------------------------------------------------------------------------------------------------------------- modules
class ConcatLinear(nn.Module):
    """Stands for a fused projection (gate_up_proj, qkv_a_proj): runs the split members and concatenates on the last axis."""

    def __init__(self, members: list):
        super().__init__()
        self.members = members
        self.out_features = sum(m.out_features for m in members)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([m(x) for m in self.members], axis=-1)


class Exl3MultiLinear(nn.Module):
    """mlx-vlm MultiLinear (x [B, G, S, in] -> [B, G, S, out], one weight per group G) from per-group EXL3 slices."""

    def __init__(self, trellis: mx.array, suh: mx.array, svh: mx.array, cb: str, fq_in: bool = False):
        super().__init__()
        self.fq_in = fq_in                               # fp8 round trip of the input first (the checkpoint quantizes the attention output before wo_a)
        G, kt, nt = trellis.shape[0], trellis.shape[1], trellis.shape[2]
        # the G groups of a decode step run in ONE launch (GroupedGemv), so the k-split fills the device divided by G: the per-group slices and the
        # grouped kernel get the same tgk (bit-identical to the per-group path), and the launch still reaches the fill target (DS4.1 wo_a: 8 groups x
        # 8 blocks x 8 = 512 threadgroups on a 64-core GPU, where the unsplit pick would give 8 x 8 x 32 = 2048).
        ent = TN.lookup(kt * 16, nt * 16, trellis.shape[3] // 16)
        tgk = ent.get("tgk") or KN.pick_tgk(kt, nt // 8, 4, max(1, TN.fill_tgs() // G), TN.fill_min_tiles())
        tune = dict(ent, tgk=tgk)
        self.groups = [Exl3Linear(trellis[g], suh[g], svh[g], cb=cb, tune=tune) for g in range(G)]
        self.num_groups, self.in_features, self.out_features = len(self.groups), self.groups[0].in_features, self.groups[0].out_features
        self.grouped = GroupedGemv(trellis, suh, svh, cb, tgk=tgk) if GROUPED_WOA else None            # 3 launches for all groups (batch 1, <= 4 rows)

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        assert transpose and x.ndim == 4 and x.shape[1] == self.num_groups, x.shape
        fold = (self.fq_in and FQF.ENABLED and self.grouped is not None and x.shape[0] == 1 and x.shape[2] <= GEMV_MAX_ROWS and x.dtype == mx.float16
                and x.shape[-1] % 128 == 0)
        if fold:                                                                        # the fp8 round trip runs inside the rot of the grouped GEMV: one launch fewer
            return self.grouped(x, fq8=True)
        nax = (self.grouped is not None and x.shape[0] == 1 and max(GEMV_MAX_ROWS, EL.ROW_SPLIT_MAX) < x.shape[2] <= EL.DENSE_FUSED_MAX and EL.DENSE_FUSED != "0"
               and (self.in_features // 16) % 2 == 0 and (self.out_features // 16) % 2 == 0 and NX.site_on(EL.DENSE_FUSED))
        if nax and self.fq_in and FQF.ENABLED and x.dtype == mx.float16 and x.shape[-1] % 128 == 0:
            return self.grouped.nax(x, fq8=True)                                # prompt rows: the round trip inside the rot of the fused NAX path (20 x 1 GiB of traffic per 8K chunk)
        if (self.fq_in and FQF.ENABLED and not nax and x.shape[0] == 1 and x.shape[2] > max(GEMV_MAX_ROWS, EL.ROW_SPLIT_MAX) and x.dtype == mx.float16
                and x.shape[-1] % 128 == 0):
            return mx.stack([self.groups[g](x[:, g], fq8=True) for g in range(self.num_groups)], axis=1)     # prompt rows: the round trip inside each group's rot
        if self.fq_in:
            x = _AR.fq(x)
        if self.grouped is not None and x.shape[0] == 1 and (x.shape[2] <= GEMV_MAX_ROWS or (KN.MAX_ROWS < x.shape[2] <= KN.fuse_rows() and self.grouped.fuses(x.shape[2]))):
            return self.grouped(x)                                              # KYOJIN_FUSE_ROWS: a 5 / 6 row verify block is ONE fused launch for the 8 slices, not 8 linears
        if self.grouped is not None and x.shape[2] == 1 and x.shape[0] <= max(GEMV_MAX_ROWS, EL.ROW_SPLIT_MAX):    # N users, one token each: users = rows of a group
            xt = x.transpose(2, 1, 0, 3)
            if EL.ROWS_NAX and x.shape[0] > KN.MAX_ROWS and (self.in_features // 16) % 2 == 0 and (self.out_features // 16) % 2 == 0 and NX.nax_available():
                return self.grouped.nax(xt).transpose(2, 1, 0, 3)               # one fused NAX launch for all groups, flat in rows
            return self.grouped(xt).transpose(2, 1, 0, 3)                       # any number of rows: one row-group launch set
        if nax:
            return self.grouped.nax(x)                                          # prompt rows above the multi-user row band (ROW_SPLIT_MAX: those stay GEMV launches, row invariant) up to the dense fused crossover: one fused launch set for the 8 groups (x1.3-2 at 512-1024 rows)
        return mx.stack([self.groups[g](x[:, g]) for g in range(self.num_groups)], axis=1)


def _head_last_row(model) -> None:
    """The LM head of a prompt chunk costs 2 * hidden * vocab flops per row (DS4.1: 1.3 GFLOP, 5 % of the whole prefill) and a [rows, vocab] logits tensor for
    nothing when the caller reads `logits[:, -1]` only. Inside `headlast.last_row()`, chunks above LAST_ROW_MIN rows get the logits of their last row only ([B, 1, vocab])."""
    head = getattr(model.language_model, "head", None)
    if head is None:                                                            # GLM / draft models: no DS4.1 head module, every row as before
        return
    cls = head.__class__
    orig = cls.__call__

    def __call__(self, x, *a, **k):
        if HL.active() and x.ndim == 3 and x.shape[1] > HL.LAST_ROW_MIN:
            x = x[:, -1:]
        return orig(self, x, *a, **k)

    head.__class__ = type(cls.__name__, (cls,), {"__call__": __call__})


def pf_min_rows(moe: M.MoeGather) -> int:
    """Rows in one MoE call from which the S3 prefill path runs instead of the 4-row decode chunks: KYOJIN_PF_MIN_ROWS, else a number of rows per
    expert (PF_CROSS_NAX with the fused NAX path, PF_CROSS without) times experts / top-k."""
    if PF_MIN_ROWS:
        return int(PF_MIN_ROWS)
    from .moe_prefill import fused_default
    n = int((PF_CROSS_NAX if fused_default() else PF_CROSS) * moe.g.experts / moe.topk)
    return max(n, EL.ROW_SPLIT_MAX + 1)                                         # multi-user decode / verify rows (rows.enable) keep the row-invariant decode path


def tail_band_takes(rows: int, topk: int, step: int, floor: int, grouped: bool, hotk: bool, pairs_cap: int, row_split_max: int | None = None, spec_rows: int | None = None) -> bool:
    """pf_tail_c4 gate: one grouped call for a row band that the four-row chunks would serve today, i.e. above every decode / verify row count and
    below the S3 floor (pf_min_rows). Never at or above the floor (S3 serves those: pf_tail_grp measured the grouped call slower there). rows is a shape,
    the bounds come from the live configuration: the chunk step, the multi-user row split, the longest verify round."""
    if row_split_max is None:
        row_split_max = EL.ROW_SPLIT_MAX
    if spec_rows is None:
        from . import pld
        spec_rows = pld.MAX_DRAFT + 1
    return grouped and not hotk and max(step, row_split_max, spec_rows) < rows < floor and rows * topk <= pairs_cap


def _moe_rows(moe: M.MoeGather, x2: mx.array, ids2: mx.array, w2: mx.array, xsrc: mx.array | None = None) -> mx.array:
    """xsrc: the input of the fp8 fake quant whose output is x2 (decode: folded into the rot of the routed GEMV); prompts and chunked rows use x2."""
    step = max(1, min(MOE_MAX_ROWS, M.MAX_PAIRS // moe.topk))
    if x2.shape[0] <= step:
        return moe(x2, ids2, w2) if xsrc is None else moe(xsrc, ids2, w2, fq8=True)
    if M.TAIL_GRP and moe.grouped and not moe.hotk and x2.shape[0] * moe.topk <= M.TAIL_PAIRS:    # pf_tail_grp: the whole band in ONE grouped call
        FD.hit("pf_tail_grp")
        return moe(x2, ids2, w2, max_pairs=M.TAIL_PAIRS) if xsrc is None else moe(xsrc, ids2, w2, fq8=True, max_pairs=M.TAIL_PAIRS)
    if M.TAIL_C4 and tail_band_takes(x2.shape[0], moe.topk, step, pf_min_rows(moe), bool(moe.grouped), bool(moe.hotk), M.TAIL_PAIRS):    # pf_tail_c4: only rows the chunks would serve
        FD.hit("pf_tail_c4")
        return moe(x2, ids2, w2, max_pairs=M.TAIL_PAIRS) if xsrc is None else moe(xsrc, ids2, w2, fq8=True, max_pairs=M.TAIL_PAIRS)
    if x2.shape[0] >= pf_min_rows(moe):
        return moe.prefill(x2, ids2, w2)
    return mx.concatenate([moe(x2[i:i + step], ids2[i:i + step], w2[i:i + step]) for i in range(0, x2.shape[0], step)], axis=0)


def _routed(self, x, ids, wts, src):
    """Routed experts of one KyojinMoe on x [..., h] -> [rows, h] fp32. With sumoe (self.msh) the decode rows get routed + shared from the same five launches."""
    k, h = ids.shape[-1], x.shape[-1]
    x2, ids2, w2 = x.reshape(-1, h), ids.reshape(-1, k), wts.reshape(-1, k)
    s2 = None if src is None else src.reshape(-1, h)
    if self.msh is not None and self.sumoe_takes(x2.shape[0]):
        return self.msh(x2 if s2 is None else s2, ids2, w2, fq8=s2 is not None)
    return _moe_rows(self.moe, x2, ids2, w2, s2)


class _SumoeMixin:
    def sumoe_takes(self, rows: int) -> bool:
        """Decode rows that the sumoe launches serve: the same band as the plain decode path (below the prefill cut, one launch group)."""
        return SU.SUMOE and rows < pf_min_rows(self.moe) and rows * self.msh.topk <= M.MAX_PAIRS and rows <= MOE_MAX_ROWS


class _CombineMixin:
    """SwitchGLU.__call__(x, indices, weights, shared, residual) -> routed (weighted fp32 sum) + shared (+ hyper-connection residual)."""

    def __call__(self, x, indices, weights=None, shared=None, residual=None):
        assert weights is not None, "the fused path needs the routing weights"
        k, h = indices.shape[-1], x.shape[-1]
        src = FQF.source(x)
        y = _routed(self, x, indices, weights, src).reshape(*x.shape[:-1], -1)
        out = y if shared is None else y + shared.astype(mx.float32)
        out = out.astype(x.dtype)
        if residual is not None:
            from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand
            out = hc_expand(out, *residual)
        return out


class _SlotMixin:
    """(x, inds, weights) -> [..., 1, hidden] fp32: the routed sum already done, the model sums the size-1 slot axis."""

    def __call__(self, x, inds, weights):
        k, h = inds.shape[-1], x.shape[-1]
        src = FQF.source(x)
        y = _routed(self, x, inds, weights, src)
        return y.reshape(*x.shape[:-1], 1, -1)


def _moe_module(old: nn.Module, moe: M.MoeGather, convention: str) -> nn.Module:
    mixin = {"combine": _CombineMixin, "slot": _SlotMixin}[convention]
    base = type(old) if convention == "combine" else nn.Module            # "combine": keeps isinstance(SwitchGLU) true for MoE.__call__
    cls = type("KyojinMoe", (mixin, _SumoeMixin, base), {})
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    obj.moe = moe
    return obj


_hooked = False


def _install_linear_hook():
    """mlx-vlm's `linear()` sends 2-8 rows through exact_quantized_linear / tokenwise helpers made for MLX quantized modules;
    Exl3 modules take any row count themselves."""
    global _hooked
    if _hooked:
        return
    L = importlib.import_module("mlx_vlm.models.linear")
    orig = L.native_batch_linear

    def native_batch_linear(module, x):
        if isinstance(module, (Exl3Linear, ConcatLinear)):
            return module(x)
        return orig(module, x)

    L.native_batch_linear = native_batch_linear
    _hooked = True


# --------------------------------------------------------------------------------------------------------------- pack I/O
def load_pack(pack_dir: str, prefix: str | None = None) -> dict:
    """All tensors of the pack (lazy mx arrays, nothing evaluated); with `prefix`, only the tensors whose name starts with it.
    A per-expert pack (the HF layout, one tensor per expert / wo_a slice) is restacked here on lazy arrays (kyojin_mlx/restack.py): no conversion."""
    ip = os.path.join(pack_dir, "model.safetensors.index.json")
    wmap = json.load(open(ip))["weight_map"] if os.path.exists(ip) else None
    if wmap:
        files = sorted({f for k, f in wmap.items() if prefix is None or k.startswith(prefix)})
    else:
        files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(pack_dir, "*.safetensors")))]
    t: dict = {}
    for f in files:
        # with an index, a tensor comes from the file the index names (the official Engram shards repeat q / k / wkv of their layer; the pack's copy wins)
        t.update({k: v for k, v in mx.load(os.path.join(pack_dir, f)).items()
                  if (wmap is None or wmap.get(k) == f) and (prefix is None or k.startswith(prefix))})
    from . import restack as RS
    return RS.restack(t)                                      # per-expert HF layout -> stacked layout (lazy, same bytes); a stacked pack is returned as is


def build_model(pack_dir: str, text_overrides: dict | None = None):
    """Model skeleton from the pack config (lazy random weights, nothing evaluated). Returns (model, spec, config).
    text_overrides: values replacing keys of text_config (tests: a smaller index_topk makes a short context exercise the indexer selection)."""
    from . import tuning
    tuning.install(pack_dir)                                  # <pack>/tuning/<device>.json, if any (no file: the constants)
    cfg = json.load(open(os.path.join(pack_dir, "config.json")))
    if text_overrides:
        cfg["text_config"] = {**cfg["text_config"], **text_overrides}
    spec = arch_of(cfg)
    activate(spec)                                            # the class-level hooks (ablit, ced, kvdisk, fused paths) resolve the vendor classes of THIS model
    mod = package(spec)
    # Engram (DS4.1): with model_path set, mlx-vlm sanitize swaps each `engram.embed` for an OffloadedEngramEmbedding (np.memmap of the rows
    # in the shard, 24 rows per token) and drops the 95 GiB tables from the weights, so nothing evaluates them. See kyojin_mlx/engram_rows.py.
    if cfg.get("text_config", cfg).get("engram_layer_ids"):
        cfg = {**cfg, "model_path": pack_dir}
    return mod.Model(mod.ModelConfig.from_dict(cfg)), spec, cfg


def _get(root, path: str):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def _set(root, path: str, new) -> None:
    parent_path, leaf = path.rsplit(".", 1) if "." in path else ("", path)
    parent = _get(root, parent_path) if parent_path else root
    if leaf.isdigit():
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)


def _has(root, path: str) -> bool:
    try:
        _get(root, path)
        return True
    except (AttributeError, IndexError, KeyError):
        return False


def _cb(t: dict, p: str, default: str) -> str:
    return "mcg" if f"{p}.mcg" in t else "mul1" if f"{p}.mul1" in t else default


# --------------------------------------------------------------------------------------------------------------- patch
class Plan:
    """What the pack holds, classified, after its dense tensors were loaded into the model."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def load_dense(model, pack_dir: str, spec: ArchSpec | None = None, config: dict | None = None, dense_dtype=None, prefix: str | None = None) -> Plan:
    """Classify the pack (quantized linears / grouped slices / expert stacks / dense), rename + cast + sanitize the dense tensors and
    load them into `model` (before any module is replaced: sanitize looks at the original classes).
    dense_dtype None: bf16 -> fp16, everything else as stored (Kyojin). A float dtype casts every floating dense tensor (reference)."""
    if spec is None or config is None:
        cfg = json.load(open(os.path.join(pack_dir, "config.json")))
        spec, config = arch_of(cfg), cfg
    t = load_pack(pack_dir, prefix)
    n_layers = len(model.language_model.layers)
    layer_re = re.compile(spec.layer_re)
    used: set = set()
    skipped: list = []

    def dropped(name: str) -> bool:
        m = layer_re.search(name)
        return bool(m) and int(m[1]) >= n_layers

    # ---- classify the quantized tensors of the pack (a prefix owns its .trellis and sidecars)
    by_prefix: dict = {}
    for k in t:
        pre, _, suf = k.rpartition(".")
        if suf == "trellis" or "." + suf in SIDE:
            by_prefix.setdefault(pre, []).append(k)
    linears, slices, stacks = [], [], []
    for p in sorted(by_prefix):
        if f"{p}.trellis" not in t:
            continue
        if any(p.startswith(s) for s in spec.skip_prefixes) or dropped(p) or target(spec, p) is None:
            skipped.extend(by_prefix[p])
            used.update(by_prefix[p])
            continue
        (stacks if f"{p}.offset" in t else slices if p.endswith(".slices") else linears).append(p)
        used.update(by_prefix[p])
    params_before = dict(tree_flatten(model.parameters()))
    dense = {}
    for k, v in t.items():
        if k in used:
            continue
        if any(k.startswith(s) for s in spec.skip_prefixes) or dropped(k):
            skipped.append(k)
            used.add(k)
            continue
        used.add(k)
        for pat, new, mv in spec.dense_rules:
            if re.match(pat, k):
                k = re.sub(pat, new, k)
                v = v.moveaxis(*mv) if mv else v
                break
        if dense_dtype is not None and mx.issubdtype(v.dtype, mx.floating):
            v = v.astype(dense_dtype)
        elif v.dtype == mx.bfloat16:
            v = v.astype(mx.float16)
        dense[k] = v
    sane = model.sanitize(dict(dense))
    if dense_dtype is None:                                       # sanitize expands fp8 block-scaled weights (Engram wkv) to bf16: fp16 like every other dense tensor,
        sane = {k: (v.astype(mx.float16) if v.dtype == mx.bfloat16 else v) for k, v in sane.items()}     # else fp16 activations x bf16 weight is an fp32 matmul (3x slower)
    unexpected = sorted(k for k in sane if k not in params_before)
    assert not unexpected, f"{len(unexpected)} sanitized tensors without a model parameter, e.g. {unexpected[:5]}"
    bad = [(k, tuple(v.shape), tuple(params_before[k].shape)) for k, v in sane.items() if tuple(v.shape) != tuple(params_before[k].shape)]
    assert not bad, f"shape mismatch: {bad[:5]}"
    model.load_weights(list(sane.items()), strict=False)
    left = sorted(k for k in t if k not in used)
    assert not left, f"{len(left)} pack tensors consumed by nothing, e.g. {left[:5]}"
    return Plan(t=t, spec=spec, config=config, linears=linears, slices=slices, stacks=stacks, skipped=skipped, loaded=set(sane),
                n_dense=len(dense), default_cb=config.get("quantization_config", {}).get("codebook", "mul1"))


def check_loaded(model, plan: Plan) -> None:
    """Every parameter still in the language model tree must have been loaded from the pack."""
    remaining = {k for k, _ in tree_flatten(model.language_model.parameters())}
    missing = sorted(k for k in remaining if f"language_model.{k}" not in plan.loaded)
    assert not missing, f"{len(missing)} model parameters never loaded, e.g. {missing[:5]}"


def _release_superseded_stacks(t: dict, roles: dict) -> None:
    """Drop a wired MoE layer's routed-only stack tensors from the pack dict. The extended stacks (build_moe, KYOJIN_PORT_SUMOE=1)
    supersede them, and the pack pins everything it holds to the end of patch_model: without this the load keeps two copies of every
    MoE weight (the 172 GiB DS4.1 pack does not fit twice). Only the routed-only sidecars go; the extended stacks never name them."""
    for r in roles.values():
        for suf in ("trellis", "offset", "k", "suh", "svh", "shape", "mcg", "mul1"):
            t.pop(f"{r}.{suf}", None)


def build_moe(old, moe_parent, st: dict, mspec, materialize: bool = True, rep: dict | None = None) -> nn.Module:
    """The KyojinMoe replacing the routed experts `old` of `moe_parent` (st = gate / up / down Stacks). With KYOJIN_PORT_SUMOE=1 and an ungated shared
    expert of Exl3Linear projections, the shared expert rides the routed launches on decode rows (moe_sumoe)."""
    limit = old
    for part in mspec.limit_path.split("."):
        limit = getattr(limit, part)
    limit = float(limit)
    topk = int(moe_parent.gate.top_k)
    kw = dict(limit=limit, wpre=mspec.weight_before_down, fq=mspec.fake_quant_fp8, alpha=mspec.alpha, beta=mspec.beta)
    new_moe = _moe_module(old, M.MoeGather(st["gate"], st["up"], st["down"], topk, **kw), mspec.convention)
    new_moe.msh = None
    if SU.SUMOE and hasattr(moe_parent, "shared_experts") and not hasattr(moe_parent, "shared_expert_gate"):
        ext = SU.wire_shared(st["gate"], st["up"], st["down"], moe_parent.shared_experts, limit, mspec.alpha, mspec.beta)
        if ext is not None:
            if materialize:
                mx.eval(*[a for s in ext for a in (s.words, s.offset, s.k, s.suh, s.svh)])
                mx.clear_cache()
            new_moe.moe = M.MoeGather(*ext, topk, **kw)                  # same extended stacks: the routed-only path (prefill, chunks) never names the last expert
            new_moe.msh = SU.MoeSharedFused(*ext, topk, **kw)
            stub = SU.SharedStub(moe_parent.shared_experts, mspec.convention == "combine")
            stub.takes = new_moe.sumoe_takes
            moe_parent.shared_experts = stub
            if rep is not None:
                rep["sumoe_layers"] = rep.get("sumoe_layers", 0) + 1
    return new_moe


def patch_model(model, pack_dir: str, spec: ArchSpec | None = None, config: dict | None = None, materialize: bool = True,
                prefix: str | None = None) -> dict:
    """Load the dense tensors of `pack_dir` into `model` and replace every quantized linear / expert stack by Kyojin modules.
    `model` comes from `build_model` (or the same constructor). Returns a report (counts, skipped names).
    prefix: only the pack tensors with this name prefix (the draft head: spec = archs.mtp_arch(trunk spec), prefix "mtp.")."""
    _install_linear_hook()
    from . import ablit as AB
    if prefix is None and AB.spec_path():                                       # trunk only: the draft head (prefix) is never edited
        AB.install_block()
    plan = load_dense(model, pack_dir, spec, config, prefix=prefix)
    spec, t, default_cb, linears, slices, stacks = plan.spec, plan.t, plan.default_cb, plan.linears, plan.slices, plan.stacks
    lm = model.language_model
    rep = dict(linears=0, heads=0, fused_splits=0, multilinear=0, moe_layers=0, dense_tensors=plan.n_dense, skipped=len(plan.skipped))
    fused: dict = {}
    # ---- quantized linears
    for p in linears:
        q = target(spec, p)
        cb = _cb(t, p, default_cb)
        bias = t.get(f"{p}.bias")
        lin_kw = dict(trellis=t[f"{p}.trellis"], suh=t[f"{p}.suh"], svh=t[f"{p}.svh"], bias=bias, cb=cb)
        if _has(model, q):
            old = _get(model, q)
            is_head = type(old).__name__ in spec.head_classes
            assert isinstance(old, nn.Linear) or is_head, f"{q}: {type(old).__name__} is not a Linear-like module"
            lin = Exl3Linear(**lin_kw, out_dtype=mx.float32 if is_head else mx.float16)
            assert (lin.out_features, lin.in_features) == tuple(old.weight.shape), (q, lin.out_features, lin.in_features, old.weight.shape)
            assert (bias is not None) == ("bias" in old), f"{q}: bias in pack {bias is not None}, in model {'bias' in old}"
            _set(model, q, lin)
            rep["heads" if is_head else "linears"] += 1
        else:
            base, leaf = q.rsplit(".", 1)
            fname = FUSED.get(leaf)
            assert fname and _has(model, f"{base}.{fname}"), f"{q}: no module and no fused parent"
            fused.setdefault((base, fname), {})[leaf] = Exl3Linear(**lin_kw)
    for (base, fname), members in fused.items():
        order = FUSED_ORDER[fname]
        assert set(members) == set(order), f"{base}.{fname}: pack has {sorted(members)}, expected {order}"
        old = _get(model, f"{base}.{fname}")
        cat = ConcatLinear([members[n] for n in order])
        assert cat.out_features == old.weight.shape[0] and members[order[0]].in_features == old.weight.shape[1], (base, fname)
        _set(model, f"{base}.{fname}", cat)
        rep["fused_splits"] += 1

    # ---- grouped slices (DS4.1 wo_a)
    for p in slices:
        q = target(spec, p)
        old = _get(model, q)
        ml = Exl3MultiLinear(t[f"{p}.trellis"], t[f"{p}.suh"], t[f"{p}.svh"], _cb(t, p, default_cb), fq_in=spec.fq_grouped_in)
        assert (old.weight.shape[0], old.weight.shape[1], old.weight.shape[2]) == (ml.num_groups, ml.out_features, ml.in_features), (q, old.weight.shape)
        _set(model, q, ml)
        rep["multilinear"] += 1

    # ---- routed experts
    groups: dict = {}
    for p in stacks:
        q = target(spec, p)
        parent, proj = q.rsplit(".", 1)
        groups.setdefault(parent, {})[spec.moe.roles[proj]] = p
    for parent, roles in groups.items():
        assert set(roles) == {"gate", "up", "down"}, (parent, sorted(roles))
        old = _get(model, parent)
        cbs = {_cb(t, roles[r], default_cb) for r in roles}
        assert len(cbs) == 1, cbs
        st = {r: M.Stack.from_arrays(t, roles[r], cbs.copy().pop()) for r in roles}
        if materialize:
            mx.eval(*[a for s in st.values() for a in (s.words, s.offset, s.k, s.suh, s.svh)])
            mx.clear_cache()                                                     # per-expert buffers of a restacked pack are free now: do not keep them cached
        moe_parent = _get(model, parent.rsplit(".", 1)[0])
        assert st["gate"].experts == old.gate_proj.weight.shape[0], (parent, st["gate"].experts, old.gate_proj.weight.shape[0])
        new_moe = build_moe(old, moe_parent, st, spec.moe, materialize, rep)
        if new_moe.msh is not None:
            _release_superseded_stacks(t, roles)
            if materialize:
                mx.clear_cache()
        _set(model, parent, new_moe)
        rep["moe_layers"] += 1

    check_loaded(model, plan)
    if prefix is None and AB.spec_path():
        rep["ablit_layers"] = AB.attach(model)
    model.eval()
    if materialize:
        mx.eval(lm.parameters())
    if HL.ENABLED:
        _head_last_row(model)
    from . import fuse as F
    if spec.fq_grouped_in:                                                       # DS4.1: fake_quant_fp8_ue8m0(x) once per input array (wq_a / wkv, routed / shared experts read the same x)
        from . import pf_glue as PG
        PG.memoize_fq()
    from . import gluefuse as GF
    rep["glue_fuse"] = GF.install()                                              # KYOJIN_GLUE_FUSE=1: mHC mix vector, norm into the mix kernel (0 when off)
    from . import engram_fuse as EF
    rep["engram_hash_fused"] = EF.install()                                    # n-gram hash of a decode step / verify row in one launch (stock code for prompts)
    rep["prefill_attn"] = F.install(spec, only=F.PREFILL_IMPLS)                  # prompt-time kernels (no decode-side replacement; rows.enable installs the rest)
    from . import ds_pf_attn
    rep["ds_pf"] = ds_pf_attn.install(model)                                    # DS4.1 prefill attention (S > 1); no-op for other architectures
    from . import bound_layers as BL
    rep["bound_layers"] = BL.install(model)                                     # prompt chunks: evaluate every few layers instead of one graph over the whole model
    from . import longctx
    rep["longctx"] = longctx.install(model)                                     # long-context exact levers (KYOJIN_LONGCTX*, default off)
    return rep
