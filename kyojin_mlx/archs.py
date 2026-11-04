# SPDX-License-Identifier: Apache-2.0
"""Per-architecture mapping tables for `patch_model` (kyojin_mlx.patch). The patcher is rule based on the module type found in the
model tree (Linear, MultiLinear, fused gate_up / qkv, head class, routed experts). A new architecture is one `ArchSpec`: how pack
tensor names map onto module paths, which dense tensors need a rename, which pack linears mlx-vlm / mlx-lm fuse, and how the
routed-expert module is called (activation, where the routing weight is applied, output convention).

Arch = the model package: `glm5_next` (GLM-5.3) and `deepseek_v41` (DS4.1) on mlx-vlm 0.7.10. MiniMax-M3.1: see ULTRA.md."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# pack members of a module that the model class fuses, in the fused output order (callers split the fused output)
FUSED = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "q_a_proj": "qkv_a_proj", "kv_a_proj_with_mqa": "qkv_a_proj",
         "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj"}
FUSED_ORDER = {"gate_up_proj": ("gate_proj", "up_proj"), "qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
               "qkv_proj": ("q_proj", "k_proj", "v_proj")}
W2 = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


@dataclass
class MoeSpec:
    """Routed experts. `convention`: "combine" = the replacement subclasses SwitchGLU and returns routed (weighted sum) + shared
    (+ hyper-connection residual) like mlx-vlm's `SwitchGLU.__call__(x, inds, weights, shared, residual)`; "slot" = it returns
    [..., 1, hidden] fp32 and the model sums the slot axis itself (DeepseekV41MoE)."""
    roles: dict                                   # expert projection name in the model -> "gate" | "up" | "down"
    convention: str = "combine"
    weight_before_down: bool = False              # routing weight multiplies the activation (DS4.1) instead of the output (GLM)
    fake_quant_fp8: bool = False                  # DS4.1: fp8 e4m3 / ue8m0 block-32 fake quant of the down-projection input
    alpha: float = 1.0                            # act = min(g, L) * sigmoid(alpha * min(g, L)) * (clip(u, -L, L) + beta)
    beta: float = 0.0                             # silu: alpha 1, beta 0; swigluoai (MiniMax-M3): alpha 1.702, beta 1
    limit_path: str = "activation.limit"          # on the replaced switch module


@dataclass
class RowsSpec:
    """Multi-user decode (kyojin_mlx.rows): N users, one token each, one forward per step.
    adapter     "batch_cache" = the model's own batch caches (merge) already run one attention over the rows;
                "dsv4_core"   = projections and attention core for all users at once (kyojin_mlx.dsv4_core: window ring, compressor state,
                indexer and sparse attention over stacked ragged per-user state, one dispatch per op);
                "dsv4_hoist"  = the projections of all users run in one launch each, only the attention core runs once per user, with
                the model's own code (enable(core="user"); uses the hoist / sink tables).
    attn_attr   layer attribute holding the attention module (adapter "dsv4_hoist").
    hoist       linears of the attention module (dotted path) whose output the adapter computes for all users at once;
    sink        linears at the end of the attention (their input is collected from the users, the output computed at once).
    sites       (module class, method, impl) overrides that make a decode-time matmul row-invariant (see rows.SITES)."""
    adapter: str
    attn_attr: str = ""
    hoist: tuple = ()
    sink: tuple = ()
    sites: tuple = ()


@dataclass
class FuseSpec:
    """Launch cuts (kyojin_mlx.fuse): module-level functions or class attributes of the model code replaced by fused kernels that keep the stock
    arithmetic bit for bit (each impl falls back to the original for shapes it does not cover).
    funcs   ("module.path[:Class]", attribute, impl name in fuse.IMPLS)."""
    funcs: tuple = ()


@dataclass
class MtpSpec:
    """Draft head stored in the pack next to the trunk (the checkpoint's MTP stages under `prefix`<stage>.), see kyojin_mlx.dspark.
    kind          "dspark" = block drafter: ONE forward over [seed, noise x (block - 1)] through `stages` blocks, non-causal attention over
                  the last `sliding_window` main-kv rows (projected tap states of the trunk) ++ the block, markov-biased greedy
                  sampling of the block, confidence head for the draft length.
    taps_at_input the drafter reads the INPUT of its target layers (after the layer's Engram), mean over the hyper-connection streams.
    name_rules / dense_rules   like ArchSpec, for the drafter model (`language_model.layers.<stage>.*` plus its extra modules)."""
    kind: str
    prefix: str = "mtp."
    keep: int = 128                                      # draft-ring rows the taps keep: max(keep, drafter window) (and the CED tail floor)
    taps_at_input: bool = True
    name_rules: list = field(default_factory=list)
    dense_rules: list = field(default_factory=list)
    keys: dict = field(default_factory=dict)             # config key -> role (stages, block, noise, taps, rank, experts, topk)


@dataclass(frozen=True)
class Vendor:
    """The model code an ArchSpec runs on: the module that defines the classes (mlx-vlm `language` file) and the roles Kyojin patches or tests for.
    A role left "" is not used by that architecture (a check on it is skipped, never guessed). `cls(role)` resolves a role once (the import is cached by
    Python); the hot paths call it at install time and keep the class."""
    module: str
    block: str = ""                                   # one decoder layer (hyper-connection block); the ablit hook and the block fusions patch its __call__
    attn: str = ""                                    # the attention module of a layer
    moe: str = ""                                     # the routed + shared MoE module of a layer
    cache: str = ""                                   # the per-model KV cache class (the kvdisk adapter, CED, the fp4 pool match it by name through the MRO)
    batch_cache: str = ""                             # its batched form (rows)

    def lang(self):
        import importlib
        return importlib.import_module(self.module)

    def package(self):
        """The model package (`mlx_vlm.models.<x>`) that holds `Model` / `ModelConfig`: the parent of `module`."""
        import importlib
        return importlib.import_module(self.module.rsplit(".", 1)[0])

    def sub(self, name: str):
        """A sibling module of the model package (`fakequant`, `engram`): the package of `module` plus `name`."""
        import importlib
        return importlib.import_module(self.module.rsplit(".", 1)[0] + "." + name)

    def cls(self, role: str):
        name = getattr(self, role)
        if not name:
            raise ValueError(f"{self.module}: no class named for role {role!r}")
        try:
            return getattr(self.lang(), name)
        except AttributeError as e:
            raise ValueError(f"{self.module} has no class {name!r} (vendor role {role!r}); the model code changed or the ArchSpec is wrong") from e


@dataclass
class CedSpec:
    """CED prefill (kyojin_mlx.ced): the model's layer layout splits into an encoder half and a decoder half that reads one shared global KV;
    a prompt chunk then runs the decoder half on its last `sliding_window` rows only. `adapter` names the installer in ced.ADAPTERS; whether a
    given config qualifies is `ced.layout(config)` (keyed on the layer layout, not on the model)."""
    adapter: str
    mid: int | None = None            # first decoder layer; None = the even split n // 2 (DS4.1: 20 of 40). An explicit layer lifts the even-layer-count rule
    tail: int | None = None           # decoder-half rows per chunk; None = sliding_window (DS4.1: 128; never lower, docs/cloud/rival/WORK_LEDGER.md)


@dataclass
class ArchSpec:
    name: str
    name_rules: list = field(default_factory=list)       # (regex, replacement) applied in order to the pack prefix of a linear
    skip_prefixes: tuple = ()                            # pack tensors the text path does not use (vision, MTP)
    dense_rules: list = field(default_factory=list)      # (key regex, new key, moveaxis(a, b) | None) for dense tensors
    head_classes: tuple = ("ParallelHead",)              # module classes that return fp32 logits
    layer_re: str = r"layers\.(\d+)\."                   # layer id in a pack name (layers >= num_hidden_layers are dropped: MTP)
    moe: MoeSpec | None = None
    rows: RowsSpec | None = None
    fuse: FuseSpec | None = None
    mtp: MtpSpec | None = None
    fq_grouped_in: bool = False                          # DS4.1: the attention output goes through the fp8 round trip (blocks of 32) before the grouped wo_a, as in the checkpoint's model.py
    ced: CedSpec | None = None                           # CED prefill (on by default, --no-ced), None = the architecture has no encoder / decoder split
    vendor: Vendor | None = None                         # the model code (module + class roles); None = the architecture has no Kyojin class-level hook
    trunk: str = "language_model"                        # attribute of the loaded model that holds the decoder (`.layers`); the ablit hook arms its blocks
    index_patch: str = ""                                # "glm": KYOJIN_DSA_INDEX_GLM installs kyojin_mlx.glm_index (the pool selection of the GLM indexer)


GLM5 = ArchSpec(
    name="glm5_next",
    name_rules=[(r"^model\.language_model\.", "language_model.model."), (r"^lm_head", "language_model.lm_head"),
                (r"\.mlp\.experts\.(\w+)$", r".mlp.switch_mlp.\1")],
    skip_prefixes=("model.visual.",), index_patch="glm",
    vendor=Vendor(module="mlx_vlm.models.glm5_next.language", attn="Glm5NextAttention"),
    # the pack stores the KDA short conv as one fused [C, 1, 4] tensor; the model keeps [C, 4, 1] (same move as q/k/v_conv1d)
    dense_rules=[(r"(.*\.self_attn)\.conv1d\.weight$", r"\1.qkv_conv.conv.weight", (2, 1))],
    layer_re=r"model\.language_model\.layers\.(\d+)\.",
    moe=MoeSpec(roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}, convention="combine"),
    rows=RowsSpec(adapter="batch_cache",
                  sites=(("HyperConnection", "_mix", "hc_branch_mix"), ("MoEGate", "__call__", "router_sigmoid_group"),
                         ("Glm5NextIndexer", "_project_keys", "index_gate_keys"),
                         ("Glm5NextAttention", "_attend", "mla_attend_rows"))),
)

DS41 = ArchSpec(
    name="deepseek_v41",
    name_rules=[(r"^mtp\..*$", None), (r"^", "language_model."), (r"\.ffn\.experts\.(\w+)$", lambda m: ".ffn.switch_mlp." + W2.get(m[1], m[1])),
                (r"\.shared_experts\.(w\d)$", lambda m: ".shared_experts." + W2[m[1]]), (r"\.attn\.wo_a\.slices$", ".attn.wo_a")],
    skip_prefixes=("vision.", "aligner.", "image_start", "image_end", "image_newline", "mtp."),
    moe=MoeSpec(roles={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}, convention="slot", weight_before_down=True,
                fake_quant_fp8=True),
    rows=RowsSpec(adapter="dsv4_core", attn_attr="attn",
                  hoist=("wq_a", "wq_b", "wkv", "compressor.wkv", "compressor.wgate", "indexer.wq_b", "indexer.weights_proj"),
                  sink=("wo_a", "wo_b"),
                  sites=(("DeepseekV41Block", "hc_mixes", "hc_block_mixes"), ("DeepseekV41MoEGate", "__call__", "router_sqrtsoftplus"),
                         ("Engram", "__call__", "engram_rows"))),
)

DS41.fq_grouped_in = True
DS41.vendor = Vendor(module="mlx_vlm.models.deepseek_v41.language", block="DeepseekV41Block", attn="DeepseekV41Attention", moe="DeepseekV41MoE",
                     cache="DeepseekV41Cache", batch_cache="BatchDeepseekV41Cache")

DS41.fuse = FuseSpec(funcs=(("mlx_vlm.models.deepseek_v41.language", "hc_expand", "mhc_expand"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Block", "hc_pre", "mhc_pre"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Attention", "_attention", "attn_decode"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Block", "__call__", "block_decode"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Attention", "_compress_part", "compress_part"),
                            ("mlx_vlm.models.deepseek_v41.language", "_apply_rope_at_positions", "rope_positions"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Attention", "_sparse_attention", "sparse_prefill"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Attention", "_attention", "local_prefill"),
                            ("mlx_vlm.models.deepseek_v41.language", "hc_mix_coeffs", "mix_prompt"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41MoE", "__call__", "moe_prompt"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41MLP", "__call__", "shared_mlp"),
                            ("mlx_vlm.models.deepseek_v41.language:DeepseekV41Block", "__call__", "block_prompt"),
                            ("mlx_vlm.models.deepseek_v41.language:Indexer", "__call__", "indexer")))
DS41.ced = CedSpec(adapter="dsv4")                         # mid / tail None: n // 2 and sliding_window, the values the lane runs

DS41.mtp = MtpSpec(
    kind="dspark",
    name_rules=[(r"^mtp\.\d+\.(main_proj)$", r"language_model.\1"), (r"^mtp\.(\d+)\.", r"language_model.layers.\1."),
                (r"\.ffn\.experts\.(\w+)$", lambda m: ".ffn.switch_mlp." + W2.get(m[1], m[1])),
                (r"\.shared_experts\.(w\d)$", lambda m: ".shared_experts." + W2[m[1]]), (r"\.attn\.wo_a\.slices$", ".attn.wo_a")],
    dense_rules=[(r"^mtp\.\d+\.(main_norm|norm)\.weight$", r"language_model.\1.weight", None),
                 (r"^mtp\.\d+\.markov_head\.embed\.weight$", "language_model.markov_embed.weight", None),
                 (r"^mtp\.\d+\.markov_head\.head\.weight$", "language_model.markov_head.weight", None),
                 (r"^mtp\.\d+\.confidence_head\.proj\.weight$", "language_model.confidence.weight", None),
                 (r"^mtp\.(\d+)\.(.*)$", r"language_model.layers.\1.\2", None)],
    keys=dict(stages="num_nextn_predict_layers", block="dspark_block_size", noise="dspark_noise_token_id", taps="dspark_target_layer_ids",
              rank="dspark_markov_rank", experts="dspark_n_routed_experts", topk="dspark_num_experts_per_tok"))

ARCHS = {a.name: a for a in (GLM5, DS41)}                  # a patched architecture = an ArchSpec here + the `arch` field of its modeldesc row


def arch_of(config: dict) -> ArchSpec:
    from . import modeldesc
    mt = config.get("model_type", "")
    d = modeldesc.by_model_type(mt)
    if d is None or d.arch is None or d.arch not in ARCHS:
        raise ValueError(f"unsupported model_type {mt!r}; known: {sorted(ARCHS)}")
    return ARCHS[d.arch]


# ---------------------------------------------------------------------------------------------------------- the active architecture
# One model per process. `patch.build_model` calls `activate(spec)`; before that (tests, tools that never load a pack) the DS4.1 spec answers, which
# is what every module-level literal it replaces said.
_ACTIVE: ArchSpec = DS41


def activate(spec: ArchSpec) -> ArchSpec:
    global _ACTIVE
    _ACTIVE = spec
    return spec


def active() -> ArchSpec:
    return _ACTIVE


def vendor(spec: ArchSpec | None = None) -> Vendor:
    v = (spec or _ACTIVE).vendor
    if v is None:
        raise ValueError(f"{(spec or _ACTIVE).name}: ArchSpec.vendor is not set")
    return v


def lang(spec: ArchSpec | None = None):
    """The model-code module of the active architecture (replaces the literal `mlx_vlm.models.deepseek_v41.language` imports)."""
    return vendor(spec).lang()


_CLS: dict = {}


def sub(name: str, spec: ArchSpec | None = None):
    """A sibling module of the active model package (cached): `sub("fakequant")`, `sub("engram")`."""
    v = vendor(spec)
    m = _CLS.get((v.module, name))
    if m is None:
        m = _CLS[(v.module, name)] = v.sub(name)
    return m


def fake_quant_fp8_ue8m0(x):
    """The model package's fp8 e4m3 / ue8m0 block-32 round trip (the stock op the fused kernels reproduce)."""
    return sub("fakequant").fake_quant_fp8_ue8m0(x)


def cls(role: str, spec: ArchSpec | None = None):
    """The class of a vendor role of the active architecture ("block", "attn", "moe", "cache", "batch_cache"); resolved once per (module, name)."""
    v = vendor(spec)
    key = (v.module, getattr(v, role))
    c = _CLS.get(key)
    if c is None:
        c = _CLS[key] = v.cls(role)
    return c


def cache_class_names() -> frozenset:
    """Class names of the per-model KV caches Kyojin has an adapter for (every ArchSpec with a vendor.cache)."""
    return frozenset(a.vendor.cache for a in ARCHS.values() if a.vendor is not None and a.vendor.cache)


def is_model_cache(obj) -> bool:
    """The model's own cache or a subclass of it (fp4pool.PackedDsCache keeps the name in its MRO). Pure name test: no mlx import."""
    names = _CACHE_NAMES if _CACHE_NAMES is not None else cache_class_names()
    return any(k.__name__ in names for k in type(obj).__mro__)


def package(spec: ArchSpec):
    """Model package of an ArchSpec: its Vendor.module's parent, else mlx-vlm's package named like the spec (the old rule)."""
    import importlib
    return spec.vendor.package() if spec.vendor is not None else importlib.import_module(f"mlx_vlm.models.{spec.name}")


# ----------------------------------------------------------------------------------------------- per-call bindings (flag vendor_bind)
# What the generic tables cost on the served hot paths: the fp8 round trip of a MoE / grouped linear input, the draft head's block collapse, the
# cache-name test. Off (the default) every call resolves through the tables above (one tuple get + one getattr on the vendor module); on, flags.set()
# pins the model's own function / constant once, so the call is the vendor function itself. Both forms return the same object: the flag measures the
# indirection and changes nothing else (exactness is tested in tests/test_vendor_bind.py).
_FQ = None                     # pinned fakequant.fake_quant_fp8_ue8m0
_FQ_DISABLE = None             # pinned fakequant.DISABLE
_HC_PRE = None                 # pinned block.hc_pre (the draft head's collapse)
_CACHE_NAMES = None            # pinned frozenset of cache class names


def fq(x):
    """The model package's fp8 e4m3 / ue8m0 block-32 round trip, per call (fake_quant_fp8_ue8m0)."""
    f = _FQ
    return f(x) if f is not None else sub("fakequant").fake_quant_fp8_ue8m0(x)


def fq_on() -> bool:
    """Whether the fp8 fake quant is on (mlx-vlm fakequant.DISABLE: the accuracy gate's --no-fq turns it off)."""
    d = _FQ_DISABLE
    return not (d if d is not None else sub("fakequant").DISABLE)


def hc_pre(h, pre):
    """The block's collapse of the hyper-connection state (`block.hc_pre`), which the draft head runs on every pass."""
    f = _HC_PRE
    return f(h, pre) if f is not None else cls("block").hc_pre(h, pre)


def pin_fq(v):
    """flag vendor_bind: pin / unpin the fp8 round trip (returns the attribute flags.set writes)."""
    global _FQ
    _FQ = sub("fakequant").fake_quant_fp8_ue8m0 if v == "1" else None
    return _FQ


def pin_fq_disable(v):
    global _FQ_DISABLE
    _FQ_DISABLE = sub("fakequant").DISABLE if v == "1" else None
    return _FQ_DISABLE


def pin_hc_pre(v):
    global _HC_PRE
    _HC_PRE = cls("block").hc_pre if v == "1" else None
    return _HC_PRE


def pin_cache_names(v):
    global _CACHE_NAMES
    _CACHE_NAMES = cache_class_names() if v == "1" else None
    return _CACHE_NAMES


def mtp_arch(spec: ArchSpec) -> ArchSpec:
    """The ArchSpec `patch_model` uses for the draft head of `spec` (pack names `mtp.<stage>.*` -> the drafter model's modules)."""
    assert spec.mtp is not None, f"{spec.name}: no draft head described"
    from dataclasses import replace
    return replace(spec, name_rules=spec.mtp.name_rules, dense_rules=spec.mtp.dense_rules, skip_prefixes=(), rows=None, mtp=None, fq_grouped_in=False, ced=None)


def target(arch: str | ArchSpec, p: str) -> str | None:
    """Module path in the model for the pack linear prefix `p` (None = dropped by the model's sanitize)."""
    spec = ARCHS[arch] if isinstance(arch, str) else arch
    q = p
    for pat, rep in spec.name_rules:
        if rep is None:
            if re.search(pat, q):
                return None
            continue
        q = re.sub(pat, rep, q)
    return q
