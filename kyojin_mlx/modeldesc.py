# SPDX-License-Identifier: Apache-2.0
"""Model description: ONE small declarative table row per architecture, and the geometry read from a config.json.

Nothing here imports mlx, so the table and `geometry()` are usable by the converter, `tools/qualify`, the server entry points and the tests.
Adding a model that reuses a known block = one `ModelDesc` row in `DESCS`. A new block adds one `ArchSpec` (kyojin_mlx.archs, if it is patched
through `patch_model`) and, for a new chat format, one `Family` class in serve/chat.py.

Two layers, kept apart on purpose:
  * `ModelDesc`   what the model IS (block layout, MoE, attention kind, KV kind, hyper-connections) and which server pieces it selects.
  * `geometry()`  the numbers of THIS checkpoint (layers, experts, widths, hc streams), read from its config. Nothing is a model name.
`KERNEL` is the geometry the hand-written Metal sources are compiled for. A gate that guards such a kernel compares the pack's geometry with
`KERNEL`; an unseen geometry falls back to the stock op (never a wider gate into a kernel that was not written for it). `fast_paths()` lists,
for a config, which of those paths apply and which fall back, so the speed a new model loses is visible before it runs
(`python -m kyojin_mlx.modeldesc CONFIG.json`).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelDesc:
    name: str
    model_types: tuple                    # exact config.json "model_type" values
    architectures: dict                   # config.json "architectures"[0] -> has a vision tower
    block: str                            # layer layout: "hybrid_gdn" (linear attention + full attention every N) | "csa_moe_hc" | "mla_dsa_moe_hc"
    attn: str                             # "gdn+gqa" | "csa" (compressed sparse + sliding window) | "mla_dsa"
    kv: str                               # "kvwin" (quantized pool on the full-attention layers) | "native" (the model's own cache, no KV bits to choose) | "fp16" (plain cache: attention sinks)
    moe: str                              # "yes" | "no" | "config" (decided by the experts count of the config)
    hc: str                               # hyper-connections: "yes" | "no" | "config" (hc_mult in the config)
    backend: str                          # serve.backends loader: "q35" | "vlm" | "generic" (stock mlx-lm class + multiuser plumbing)
    server: str                           # serve.chat family: "qwen" | "glm" | "ds41" (also the `thinking` flag spelling in chat)
    family: str                           # qualify family name
    arch: str | None = None               # kyojin_mlx.archs.ARCHS key (None = not patched through patch_model)
    ref: str | None = None                # streamed BF16 reference adapter in tools/qualify/refs
    chat_prefixes: tuple = ()             # unlisted model_types that still pick this chat family (startswith)
    backend_prefixes: tuple = ()          # unlisted model_types that still pick this backend (startswith)
    mtp_depth: int = 0                    # --mtp-depth qualify uses when the pack carries draft-head tensors
    chunk: int = 512                      # prefill chunk qualify runs the server with (2048 where CED applies)
    one_buffer_gate: bool = False         # the Engram gate (KYOJIN_ENGRAM_GATE=1) needs one command buffer per step
    own_tokenizer: str = ""               # qualify: default tokenizer dir (models/<name>) when the pack has no tokenizer_config.json
    builtin_template: str = ""            # serve module (kyojin_mlx.serve.<name>) that renders the chat prompt when the pack has no jinja template
    thinking_key: str = "thinking"        # chat template kwarg that switches thinking
    stop_tokens: tuple = ()               # token STRINGS (besides the config / generation_config / tokenizer eos) that end a turn; ids are looked up in the tokenizer
    mtp_prefix: str = "mtp."              # tensor-name prefix of the draft head in the pack (has_mtp_tensors; ArchSpec.mtp.prefix is the patched-model twin)
    mtp_auto: bool = False                # load_backend turns the draft head on by itself when the pack carries its tensors (DS4.1 / GLM keep it opt-in: --mtp-depth)


DESCS = (
    ModelDesc(
        name="qwen35", model_types=("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"),
        architectures={"Qwen3_5ForConditionalGeneration": True, "Qwen3_5ForCausalLM": False,
                       "Qwen3_5MoeForConditionalGeneration": True, "Qwen3_5MoeForCausalLM": False},
        block="hybrid_gdn", attn="gdn+gqa", kv="kvwin", moe="config", hc="no", backend="q35", server="qwen", family="hybrid_gdn",
        ref="qwen35", chat_prefixes=("qwen",), backend_prefixes=("qwen3",), thinking_key="enable_thinking",
        stop_tokens=("<|im_end|>", "<|endoftext|>"), mtp_auto=True),
    ModelDesc(
        name="deepseek_v41", model_types=("deepseek_v41",), architectures={"DeepseekV41ForCausalLM": False},
        block="csa_moe_hc", attn="csa", kv="native", moe="yes", hc="yes", backend="vlm", server="ds41", family="ds41",
        arch="deepseek_v41", chat_prefixes=("deepseek",), mtp_depth=3, chunk=2048, one_buffer_gate=True, own_tokenizer="tok_ds41", builtin_template="ds41_chat",
        stop_tokens=("<｜end▁of▁sentence｜>",)),
    ModelDesc(
        name="glm5_next", model_types=("glm5_next",), architectures={"Glm5NextForConditionalGeneration": True},
        block="mla_dsa_moe_hc", attn="mla_dsa", kv="native", moe="yes", hc="yes", backend="vlm", server="glm", family="glm5_next",
        arch="glm5_next", chat_prefixes=("glm",), stop_tokens=("<|user|>", "<|observation|>", "<|endoftext|>")),
    ModelDesc(                                      # OpenAI gpt-oss (MXFP4 experts read natively by mlx-lm, attention with learned sinks: fp16 KV, no TurboQuant pool)
        name="gpt_oss", model_types=("gpt_oss",), architectures={"GptOssForCausalLM": False},
        block="moe", attn="gqa+sinks", kv="fp16", moe="yes", hc="no", backend="generic", server="harmony", family="gpt_oss",
        stop_tokens=("<|return|>", "<|call|>")),
    ModelDesc(                                      # Baidu ERNIE 4.5 MoE family (mlx-lm ernie4_5 / ernie4_5_moe; the prefix takes a later ernie* model_type that mlx-lm learns): stock class,
        name="ernie", model_types=("ernie4_5", "ernie4_5_moe"),   # per-user caches + multi-user rows of GenericBackend, the pack's own chat template (no speed path yet)
        architectures={"Ernie4_5_ForCausalLM": False, "Ernie4_5_MoeForCausalLM": False},
        block="moe", attn="gqa", kv="fp16", moe="config", hc="no", backend="generic", server="qwen", family="ernie",
        chat_prefixes=("ernie",), backend_prefixes=("ernie",)),
)

_BY_TYPE = {t: d for d in DESCS for t in d.model_types}
_BY_ARCH = {a: d for d in DESCS for a in d.architectures}


def by_model_type(model_type: str) -> ModelDesc | None:
    """Exact match only (what `archs.arch_of` always did)."""
    return _BY_TYPE.get(model_type)


def by_architecture(architecture: str) -> ModelDesc | None:
    return _BY_ARCH.get(architecture)


def _prefix(model_type: str, field_name: str) -> ModelDesc | None:
    d = _BY_TYPE.get(model_type)
    if d is not None:
        return d
    for d in DESCS:
        if any(model_type.startswith(p) for p in getattr(d, field_name)):
            return d
    return None


def chat_desc(model_type: str) -> ModelDesc | None:
    """The row `serve.chat.family_of` picks: exact model_type, else the family prefixes ("qwen", "glm", "deepseek")."""
    return _prefix(model_type, "chat_prefixes")


def backend_desc(model_type: str) -> ModelDesc | None:
    """The row `serve.backends.load_backend` picks: exact model_type, else the backend prefixes ("qwen3")."""
    return _prefix(model_type, "backend_prefixes")


GENERIC_STOPS = ("<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
                 "<end_of_utterance>", "<eos>", "<turn|>")      # the subset of llama-vocab.cpp:3270-3290 that is a plain end-of-generation text (no pre-specific entry, no harmony / solar-open)


def stop_tokens_of(server: str) -> tuple:
    """Stop-token strings of the chat family `server` ("qwen" / "glm" / "ds41"): the row that selects that family."""
    if server in ("generic", "jinja"):         # a pack with no row (Jinja fallback family), or a bare GGUF of a mapped architecture: the ids of the file's own metadata (generation_config.json) plus the end-of-turn tokens llama.cpp treats as
        return GENERIC_STOPS                   # end of generation by their text (llama-vocab.cpp special_eog_ids, lines 3266-3296), which the metadata does not always carry (gemma, GLM)
    for d in DESCS:
        if d.server == server:
            return d.stop_tokens
    raise KeyError(server)


def text_config(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


# ------------------------------------------------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class Geometry:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int                        # qk_rope_head_dim (0 = none)
    hc_mult: int                          # hyper-connection streams (0 = none)
    n_mix: int                            # mixes per token of a hyper-connection: (2 + hc) * hc  (hc = 4 -> 24)
    o_groups: int                         # grouped output projection (wo_a) groups (0 = plain wo)
    index_heads: int                      # sparse-attention indexer (0 = none)
    index_dim: int
    experts: int                          # routed experts (0 = dense)
    topk: int
    moe_layers: int
    window: int                           # sliding window (0 = none)
    kv_layers: int                        # layers with a growing KV cache
    compress_ratios: tuple = field(default_factory=tuple)
    mtp_layers: int = 0


def _first(t: dict, *names, default=0):
    for n in names:
        v = t.get(n)
        if v is not None:
            return v
    return default


def geometry(cfg: dict) -> Geometry:
    """Numbers of this checkpoint from its config.json (or text_config). Same keys the model code reads (ced._g, dspark, mini_ds41.CFG)."""
    t = text_config(cfg)
    n = int(t["num_hidden_layers"])
    heads = int(t["num_attention_heads"])
    hidden = int(t["hidden_size"])
    hc = int(_first(t, "hc_mult", default=0) or 0)
    experts = int(_first(t, "n_routed_experts", "num_experts", "num_local_experts", default=0) or 0)
    dense_first = int(_first(t, "first_k_dense_replace", default=0) or 0) if experts else 0
    ratios = tuple(int(r) for r in (t.get("compress_ratios") or ())[:n])
    lt = t.get("layer_types")
    if lt:
        kvl = sum(1 for x in lt if x in ("full_attention", "attention"))
    elif t.get("full_attention_interval"):
        kvl = n // int(t["full_attention_interval"])
    else:
        kvl = n
    return Geometry(
        layers=n, hidden=hidden, heads=heads, kv_heads=int(t.get("num_key_value_heads") or heads),
        head_dim=int(t.get("head_dim") or hidden // heads), rope_dims=int(_first(t, "qk_rope_head_dim", "rope_head_dim", default=0) or 0),
        hc_mult=hc, n_mix=(2 + hc) * hc, o_groups=int(_first(t, "o_groups", default=0) or 0),
        index_heads=int(_first(t, "index_n_heads", default=0) or 0), index_dim=int(_first(t, "index_head_dim", default=0) or 0),
        experts=experts, topk=int(_first(t, "num_experts_per_tok", default=0) or 0), moe_layers=(n - dense_first) if experts else 0,
        window=int(_first(t, "sliding_window", default=0) or 0), kv_layers=kvl, compress_ratios=ratios,
        mtp_layers=int(_first(t, "mtp_num_hidden_layers", "num_nextn_predict_layers", default=0) or 0))


def describe(cfg: dict) -> tuple[ModelDesc | None, Geometry]:
    """(table row or None for an unknown model_type, geometry). The row is found by model_type, then by architectures[0]."""
    d = by_model_type(text_config(cfg).get("model_type", "") or cfg.get("model_type", ""))
    if d is None:
        d = by_architecture((cfg.get("architectures") or ["?"])[0])
    return d, geometry(cfg)


# ------------------------------------------------------------------------------------------------------------------ kernel envelope
class KERNEL:
    """Geometry the hand-written Metal sources are written for. One place per number; the gates in the kernel modules read these."""
    HC = 4                                # hyper-connection streams (mix kernels, decode block fusion)
    MIXES = 24                            # (2 + HC) * HC
    HEAD_DIM = 512                        # attention head width of the rope / kv / fq / wo_a kernels
    ROPE_DIMS = 64
    O_GROUPS = 8                          # wo_a groups
    IDX_HEADS = 32                        # indexer heads of the fused score kernel
    IDX_DIM = 128
    HIDDEN_MULT = 128                     # hidden % 128 (Hadamard / rope tiles)
    STAGE_MIN_HIDDEN = 4096               # suglue.stage_ok
    PF_NORM_HIDDEN = 5120                 # pf_glue.pre_norm_fq: one 1024-thread row, 4 lanes per thread over hidden = 5120 (written for exactly this width)


def fast_paths(g: Geometry) -> dict:
    """{path: (applies, why)}: which fused paths cover this geometry. False = the stock op runs (the speed a new model loses)."""
    def need(ok, msg):
        return bool(ok), ("ok" if ok else msg)
    hc = g.hc_mult
    hid = g.hidden % KERNEL.HIDDEN_MULT == 0
    return {
        "hc_block_decode (decstage, glue_fuse, gluefuse, fuse_layer.fold)": need(
            hc == KERNEL.HC and hid, f"hc_mult={hc} (kernels are written for {KERNEL.HC}) / hidden % {KERNEL.HIDDEN_MULT} = {g.hidden % KERNEL.HIDDEN_MULT}"),
        "hc_stage (suglue)": need(hc == KERNEL.HC and hid and g.hidden >= KERNEL.STAGE_MIN_HIDDEN,
                                  f"hc_mult={hc}, hidden={g.hidden} (needs {KERNEL.HC}, a multiple of {KERNEL.HIDDEN_MULT}, >= {KERNEL.STAGE_MIN_HIDDEN})"),
        "hc_mix_prefill (mhc_prefill)": need(hc == KERNEL.HC and g.n_mix == KERNEL.MIXES and (hc * g.hidden) % 256 == 0,
                                             f"hc_mult={hc}, mixes={g.n_mix} (kernel: {KERNEL.HC} streams, {KERNEL.MIXES} mixes)"),
        "hc_pre_norm_prefill (pf_glue.pre_norm_fq)": need(hc == KERNEL.HC and g.hidden == KERNEL.PF_NORM_HIDDEN,
                                                          f"hc_mult={hc}, hidden={g.hidden} (kernel: {KERNEL.HC} streams, hidden exactly {KERNEL.PF_NORM_HIDDEN})"),
        "attn_rope_kv_chain (dsv4_core, fuse_attn, pf_glue)": need(
            g.head_dim == KERNEL.HEAD_DIM and g.rope_dims == KERNEL.ROPE_DIMS,
            f"head_dim={g.head_dim}, rope_dims={g.rope_dims} (kernels: {KERNEL.HEAD_DIM} / {KERNEL.ROPE_DIMS})"),
        "wo_a_grouped_prefill (pf_glue)": need(g.o_groups == KERNEL.O_GROUPS and g.head_dim == KERNEL.HEAD_DIM,
                                               f"o_groups={g.o_groups}, head_dim={g.head_dim} (kernel: {KERNEL.O_GROUPS} groups of {KERNEL.HEAD_DIM})"),
        "indexer_score (ds_pf_idx, glm_index)": need(g.index_heads == KERNEL.IDX_HEADS and g.index_dim == KERNEL.IDX_DIM,
                                                     f"index heads/dim {g.index_heads}/{g.index_dim} (kernel: {KERNEL.IDX_HEADS}/{KERNEL.IDX_DIM})"),
        "rows_router (decrows)": need(g.experts == 0 or (g.experts <= 1024 and 1 <= g.topk <= 32 and g.topk < g.experts),
                                      f"experts={g.experts}, topk={g.topk} (router: experts <= 1024, 1 <= topk <= 32)"),
    }


def report(cfg: dict) -> str:
    d, g = describe(cfg)
    lines = [f"row: {d.name if d else 'NONE (model_type not in DESCS: add a row)'}"]
    if d:
        lines.append(f"  block={d.block} attn={d.attn} kv={d.kv} moe={d.moe} hc={d.hc} backend={d.backend} server={d.server} arch={d.arch}")
    lines.append(f"geometry: {g}")
    for k, (ok, why) in fast_paths(g).items():
        lines.append(f"  {'covered ' if ok else 'FALLBACK'} {k}: {why}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "config.json"))))
