# SPDX-License-Identifier: Apache-2.0
"""GGUF (llama.cpp) tensor names -> Hugging Face names of the architectures the engine runs, and the config.json a bare GGUF file can stand in for (numpy-free, mlx-free)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

from . import foreign_gguf as GF

# ---------------------------------------------------------------------------------------------------------- one table per architecture
# Source of every name below: llama.cpp gguf-py/gguf/tensor_mapping.py + constants.py (MODEL_TENSORS) at LLAMA_CPP_COMMIT, checked against the HF names of
# the mlx-lm class by tests/test_gguf_arch.py (which also replays the converter's transforms in the other direction). A GGUF kind is the tensor name
# after `blk.N.` without its `.weight` / `.bias`; the HF side is the module path under `model.layers.N.` (the suffix is put back). `exact` keys carry
# their suffix (or have none: `ssm_a`) and give the whole HF tensor path.
LLAMA_CPP_COMMIT = "29c720eaf72fcf8e4e61a56a88b994d92668261d"

_TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
_BLK0 = {   # llama.cpp llama / qwen2 / qwen3 attention block: pre-attention norm, q k v o, pre-ffn norm
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
}
_BLK = {**_BLK0, "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
_MLP = {"ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
_MOE = {"ffn_gate_inp": "mlp.gate", "ffn_gate_exps": "mlp.experts.gate_proj", "ffn_up_exps": "mlp.experts.up_proj", "ffn_down_exps": "mlp.experts.down_proj"}
# Qwen3.5 (gated delta net trunk): llama.cpp src/models/qwen35.cpp load_arch_tensors; no ffn_norm (post_attention_norm), no MoE here
_BLK35 = {
    "attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj", "ssm_norm": "linear_attn.norm", "ssm_conv1d": "linear_attn.conv1d",
}
_EXACT35 = {"ssm_a": "linear_attn.A_log", "ssm_dt.bias": "linear_attn.dt_bias"}
# the MTP block (blk.<n_trunk>): conversion/qwen.py _QwenMtpMixin.filter_tensors renames mtp.fc / pre_fc_norm_embedding / pre_fc_norm_hidden / norm to
# eh_proj / enorm / hnorm / shared_head.norm of block n_trunk; tensor_mapping.py gives their nextn.* names. Target: the mtp.* names of the q35 backend.
_MTP35 = {"nextn.eh_proj.weight": "mtp.fc.weight", "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight", "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
          "nextn.shared_head_norm.weight": "mtp.norm.weight"}


@dataclass(frozen=True)
class Arch:
    model_type: str                       # config["model_type"]: picks the mlx-lm class and the serving backend
    architectures: tuple
    blk: dict = field(default_factory=lambda: dict(_BLK))
    exact: dict = field(default_factory=dict)
    top: dict = field(default_factory=lambda: dict(_TOP))
    mtp: dict | None = None               # nextn tensors of the draft block -> mtp.* names (None: the architecture has no draft head here)
    dense: tuple = ()                     # HF suffixes that are never a linear (kept as stored tensors)
    skip_nextn: bool = False              # blocks past the trunk exist in the file (llama.cpp loads them with TENSOR_SKIP) and the mlx-lm class has no use for them
    xform: str = ""                       # name of the converter's transform set (foreign_xform.py)
    config: str = "qwen3"                 # config builder (_CONFIG)


_SHEXP = {"ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj", "ffn_down_shexp": "mlp.shared_expert.down_proj",
          "ffn_gate_inp_shexp": "mlp.shared_expert_gate"}
# GLM-4-0414 (llama.cpp src/models/glm4.cpp): ffn_up holds the fused gate+up projection [n_embd, 2 * n_ff] = HF mlp.gate_up_proj; two extra post norms
_GLM4 = {**_BLK0, "ffn_up": "mlp.gate_up_proj", "ffn_down": "mlp.down_proj", "post_attention_norm": "post_self_attn_layernorm", "post_ffw_norm": "post_mlp_layernorm"}
# Gemma 3 (src/models/gemma3.cpp): ffn_norm is the PRE-ffn norm here, post_attention_norm the HF post_attention_layernorm
_GEMMA3 = {**_BLK, "ffn_norm": "pre_feedforward_layernorm", "post_attention_norm": "post_attention_layernorm", "post_ffw_norm": "post_feedforward_layernorm", **_MLP}

# Qwen3.5 MoE (src/models/qwen35moe.cpp load_arch_tensors): the qwen35 trunk with routed + shared experts in place of the dense ffn. A file that stores the experts fused
# (`ffn_gate_up_exps`, create_tensor_gate_up_exps) is refused by name. The MoE draft block is skipped (the q35 draft head is the dense block).
_BLK35MOE = {**{k: v for k, v in _BLK35.items() if not k.startswith("ffn_")}, **_MOE, **_SHEXP}

# GLM-4.5 / 4.6 (src/models/glm4-moe.cpp load_arch_tensors): attn_post_norm (post_attention_norm) is the norm BEFORE the ffn (the graph norms ffn_inp with it), optional q / k
# norms, q / k / v bias, dense ffn on the first leading_dense_block_count blocks, routed experts + `ffn_exp_probs_b` (conversion/base.py:763 renames e_score_correction_bias
# to e_score_correction.bias) + a shared expert of n_ff_exp * n_expert_shared; nextn blocks follow the trunk (TENSOR_SKIP when not loaded).
_GLM4MOE = {**_BLK, **_MLP, "post_attention_norm": "post_attention_layernorm", **_MOE, "ffn_gate_shexp": "mlp.shared_experts.gate_proj",
            "ffn_up_shexp": "mlp.shared_experts.up_proj", "ffn_down_shexp": "mlp.shared_experts.down_proj"}
del _GLM4MOE["ffn_norm"]

# DeepSeek V2 / V3 family (src/models/deepseek2.cpp load_arch_tensors, MLA): conversion/deepseek.py DeepseekV2Model.modify_tensors splits kv_b_proj into `attn_k_b` (transposed to
# [H, kv_lora, nope]) and `attn_v_b` ([H, v_dim, kv_lora]); mlx-lm deepseek_v3 sanitize makes embed_q / unembed_out of kv_b_proj with exactly those two shapes, so the tensors map
# one to one and stay 3-D dense (Arch.dense). The legacy unsplit `attn_kv_b` (non-MLA files) is refused by name. Sigmoid router + exp_probs_b only (see _cfg_deepseek2).
_DEEPSEEK2 = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q_a_norm": "self_attn.q_a_layernorm", "attn_kv_a_norm": "self_attn.kv_a_layernorm",
              "attn_q_a": "self_attn.q_a_proj", "attn_q_b": "self_attn.q_b_proj", "attn_q": "self_attn.q_proj", "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa",
              "attn_k_b": "self_attn.embed_q", "attn_v_b": "self_attn.unembed_out", "attn_output": "self_attn.o_proj", **_MLP, **_MOE,
              "ffn_gate_shexp": "mlp.shared_experts.gate_proj", "ffn_up_shexp": "mlp.shared_experts.up_proj", "ffn_down_shexp": "mlp.shared_experts.down_proj"}

ARCHS: dict[str, Arch] = {
    "qwen3": Arch("qwen3", ("Qwen3ForCausalLM",), blk={**_BLK, **_MLP}),
    "qwen2": Arch("qwen2", ("Qwen2ForCausalLM",), blk={**_BLK0, **_MLP}, config="dense"),
    "llama": Arch("llama", ("LlamaForCausalLM",), blk={**_BLK0, **_MLP}, xform="llama", config="dense"),
    # mistral3 (src/models/mistral3.cpp; conversion/mistral3.py Ministral3Model(LlamaModel), undo_permute = True): the llama blocks and the same q / k permute undone. Served as the
    # llama class when the file has no rope scaling and no attention temperature (Mistral-Small-3.1 text); Ministral-3 files carry yarn + llama-4 scaling: refused (their config.json has another model_type, so they stay unmapped).
    "mistral3": Arch("llama", ("LlamaForCausalLM",), blk={**_BLK0, **_MLP}, xform="mistral3", config="dense"),
    "qwen2moe": Arch("qwen2_moe", ("Qwen2MoeForCausalLM",), blk={**_BLK0, **_MOE, **_SHEXP}, xform="qwen2moe", config="qwen2moe"),
    "glm4": Arch("glm4", ("Glm4ForCausalLM",), blk=dict(_GLM4), skip_nextn=True, config="glm4"),
    "gemma3": Arch("gemma3_text", ("Gemma3ForCausalLM",), blk=dict(_GEMMA3), xform="gemma3", config="gemma3"),
    "qwen3moe": Arch("qwen3moe", ("Qwen3MoeForCausalLM",), blk={**_BLK, **_MOE}, config="qwen3moe"),
    "qwen35": Arch("qwen3_5", ("Qwen3_5ForConditionalGeneration",), blk=dict(_BLK35), exact=dict(_EXACT35), mtp=dict(_MTP35),
                   dense=("conv1d.weight",), xform="qwen35", config="qwen35"),
    "qwen35moe": Arch("qwen3_5_moe", ("Qwen3_5MoeForConditionalGeneration",), blk=dict(_BLK35MOE), exact=dict(_EXACT35), dense=("conv1d.weight",), skip_nextn=True,
                      xform="qwen35moe", config="qwen35moe"),
    "glm4moe": Arch("glm4_moe", ("Glm4MoeForCausalLM",), blk=dict(_GLM4MOE), exact={"exp_probs_b.bias": "mlp.gate.e_score_correction_bias"}, skip_nextn=True,
                    config="glm4moe"),
    "deepseek2": Arch("deepseek_v3", ("DeepseekV3ForCausalLM",), blk=dict(_DEEPSEEK2), exact={"exp_probs_b.bias": "mlp.gate.e_score_correction_bias"},
                      dense=("embed_q.weight", "unembed_out.weight"), skip_nextn=True, config="deepseek2"),
}
GGUF_ARCHS = tuple(ARCHS)


def _arch(arch: str) -> Arch:
    if arch not in ARCHS:
        raise GF.GgufError(f"GGUF architecture {arch!r}: no tensor map in kyojin_mlx.foreign_names (mapped: {', '.join(ARCHS)}). "
                           f"DeepSeek V4 and GLM-5 GGUF files are not mapped")
    return ARCHS[arch]


def gguf_name(name: str, arch: str, n_trunk: int | None = None) -> str | None:
    """llama.cpp tensor name -> HF name of the same model. Unknown names raise: a tensor is never dropped silently. n_trunk (block_count minus
    nextn_predict_layers): blocks from there on are the draft head and come back as `mtp.*` (mtp.fc.weight, mtp.layers.0.<hf>, ...), or None
    for an architecture whose mlx-lm class has no draft head (llama.cpp loads those blocks with TENSOR_SKIP too)."""
    A = _arch(arch)
    if name in A.top:
        return A.top[name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        raise GF.GgufError(f"GGUF tensor {name!r}: no mapping to the {arch} architecture")
    i, rest = int(m[1]), m[2]
    head = n_trunk is not None and i >= n_trunk
    if head and A.skip_nextn:
        return None
    if head:
        if A.mtp is None:
            raise GF.GgufError(f"GGUF tensor {name!r}: block {i} is past the {n_trunk} trunk blocks and {arch} has no draft head mapping")
        if rest in A.mtp:
            return A.mtp[rest]
    pre = f"mtp.layers.{i - n_trunk}." if head else f"model.layers.{i}."
    if rest in A.exact:
        return pre + A.exact[rest]
    kind, _, sfx = rest.rpartition(".")
    if kind not in A.blk or sfx not in ("weight", "bias"):
        raise GF.GgufError(f"GGUF tensor {name!r}: no mapping to the {arch} architecture")
    return f"{pre}{A.blk[kind]}.{sfx}"


def is_linear(hf: str, ndim: int, arch: str) -> bool:
    """A weight matrix the loader may hand to a quantised kernel (not a norm, a router, a convolution or the embedding table)."""
    A = _arch(arch)
    return (hf.endswith(".weight") and ndim >= 2 and not hf.endswith("norm.weight") and not hf.endswith("mlp.gate.weight") and not hf.endswith(A.dense)
            and not hf.endswith("embed_tokens.weight"))


def n_trunk_of(kv: dict, arch: str) -> int | None:
    """Trunk blocks: block_count minus the draft-head blocks (llama_hparams::n_layer() = n_layer_all - n_layer_nextn), for architectures with a draft head."""
    A = _arch(arch)
    if A.mtp is None and not A.skip_nextn:
        return None
    return int(kv[f"{arch}.block_count"]) - int(kv.get(f"{arch}.nextn_predict_layers", 0))


def gguf_has_mtp(path) -> bool:
    """The GGUF carries a draft head the q35 backend can load (a mapped architecture with nextn blocks and their tensors)."""
    try:
        f = GF.find_gguf(path)
        if not f:
            return False
        g = GF.GgufFile(f)
    except (OSError, GF.GgufError):
        return False
    try:
        arch = g.kv.get("general.architecture")
        if arch not in ARCHS or ARCHS[arch].mtp is None:
            return False
        nt = n_trunk_of(g.kv, arch)
        return int(g.kv.get(f"{arch}.nextn_predict_layers", 0)) > 0 and f"blk.{nt}.nextn.eh_proj.weight" in g.tensors
    finally:
        g.close()


# ---------------------------------------------------------------------------------------------------------- config from metadata
def _hint(arch: str) -> str:
    """The remedy for a refused rope scaling: a config.json next to the file works only where the HF model_type is one the table serves (not Ministral-3 / DeepSeek-V2)."""
    return "; for a DeepSeek-V3 file the model's config.json next to it serves it" if arch == "deepseek2" else "" if arch == "mistral3" else "; put the model's config.json next to the file"


def _kv_getter(kv, arch):
    return lambda k, d=None: kv.get(f"{arch}.{k}", d)


def _cfg_qwen3(g: GF.GgufFile, arch: str) -> dict:
    a = _kv_getter(g.kv, arch)
    rs = a("rope.scaling.type")
    if rs not in (None, "none"):                                   # yarn / linear / ...: the config built here has no rope_scaling, the model would run unscaled
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={rs!r} is not mapped{_hint(arch)}")
    hd = a("attention.key_length") or (a("embedding_length") // a("attention.head_count"))
    if a("rope.dimension_count", hd) != hd:                        # partial rotary: not what the mapped classes do
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_count={a('rope.dimension_count')} != head size {hd}: partial rotary is not mapped; put config.json next to the file")
    cfg = dict(hidden_size=a("embedding_length"),
               num_hidden_layers=a("block_count"), intermediate_size=a("feed_forward_length", 0), num_attention_heads=a("attention.head_count"),
               num_key_value_heads=a("attention.head_count_kv"), head_dim=a("attention.key_length"), rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-6),
               rope_theta=a("rope.freq_base", 1e6), max_position_embeddings=a("context_length"),
               vocab_size=g.tensors["token_embd.weight"].ne[1],
               tie_word_embeddings="output.weight" not in g.tensors, hidden_act="silu", attention_bias=False)
    if arch == "qwen3moe":
        cfg.update(num_experts=a("expert_count"), num_experts_per_tok=a("expert_used_count"), moe_intermediate_size=a("expert_feed_forward_length"),
                   norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], shared_expert_intermediate_size=0)
    return cfg


def _cfg_qwen35(g: GF.GgufFile, arch: str) -> dict:
    """Qwen3.5 dense text model. Keys read as llama.cpp src/models/qwen35.cpp load_arch_hparams reads them; the HF names are those of the config of the
    same model in safetensors (linear_* = ssm.*, partial_rotary_factor = rope.dimension_count / head size, mrope_section = rope.dimension_sections[:3])."""
    kv = g.kv
    a = _kv_getter(kv, arch)
    bad = sorted(k for k in kv if k.startswith("prism."))
    if bad:
        raise GF.GgufError(f"GGUF {bad[0]}: a Hadamard-folded file keeps the linear-attention out_proj in grouped V order and folds the weights; not mapped")
    rs = a("rope.scaling.type")
    if rs not in (None, "none"):
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={rs!r} is not mapped{_hint(arch)}")
    nl, nx = int(a("block_count")), int(a("nextn_predict_layers", 0))
    nt = nl - nx
    nk, nv, ds = int(a("ssm.group_count")), int(a("ssm.time_step_rank")), int(a("ssm.state_size"))
    inner = int(a("ssm.inner_size"))
    if inner % nv or inner // nv != ds or nv % nk:
        raise GF.GgufError(f"GGUF {arch}.ssm: inner_size {inner} / time_step_rank {nv} != state_size {ds} (or {nv} % {nk} heads): llama.cpp assumes equal key / value head size")
    iv = int(a("full_attention_interval", 4))
    rec = a("attention.recurrent_layers")
    if rec is not None and [bool(x) for x in rec[:nt]] != [(i + 1) % iv != 0 for i in range(nt)]:
        raise GF.GgufError(f"GGUF {arch}.attention.recurrent_layers does not follow full_attention_interval={iv}: the mlx-lm class only has that pattern")
    hd = int(a("attention.key_length"))
    if int(a("attention.value_length", hd)) != hd:
        raise GF.GgufError(f"GGUF {arch}: key_length != value_length")
    rd = int(a("rope.dimension_count"))
    sec = list(a("rope.dimension_sections", [11, 11, 10, 0]))
    if len(sec) != 4 or sec[3] != 0 or 2 * sum(sec[:3]) != rd:
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_sections={sec} does not match rope.dimension_count={rd}")
    rope = dict(rope_type="default", rope_theta=a("rope.freq_base", 1e7), partial_rotary_factor=rd / hd, mrope_section=sec[:3], mrope_interleaved=True)
    return dict(hidden_size=a("embedding_length"), num_hidden_layers=nt, intermediate_size=a("feed_forward_length"), num_attention_heads=a("attention.head_count"),
                num_key_value_heads=a("attention.head_count_kv"), head_dim=hd, rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-6), rope_theta=rope["rope_theta"],
                partial_rotary_factor=rope["partial_rotary_factor"], rope_parameters=rope, max_position_embeddings=a("context_length"),
                vocab_size=g.tensors["token_embd.weight"].ne[1], tie_word_embeddings="output.weight" not in g.tensors, hidden_act="silu", attention_bias=False,
                attn_output_gate=True, full_attention_interval=iv, linear_conv_kernel_dim=a("ssm.conv_kernel"), linear_key_head_dim=ds, linear_value_head_dim=inner // nv,
                linear_num_key_heads=nk, linear_num_value_heads=nv, mtp_num_hidden_layers=nx)


def _cfg_qwen35moe(g: GF.GgufFile, arch: str) -> dict:
    """qwen35moe = the qwen35 trunk config + routed experts (softmax router, top-k renormalised: build_moe_ffn(.., norm_w = true, SOFTMAX), qwen35moe.cpp:500-502) and the
    shared expert. n_ff_exp falls back to n_ff / n_expert_used as llama.cpp does (qwen35moe.cpp:57-58); the draft block is not loaded (mtp_num_hidden_layers 0)."""
    a = _kv_getter(g.kv, arch)
    if a("expert_weights_scale") not in (None, 0, 0.0, 1, 1.0):
        raise GF.GgufError(f"GGUF {arch}.expert_weights_scale={a('expert_weights_scale')}: the qwen3_5_moe class does not scale the routing weights")
    if a("expert_gating_func") not in (None, 0, 1):                 # LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX = 1 (NONE = 0 resolves to softmax in this graph)
        raise GF.GgufError(f"GGUF {arch}.expert_gating_func={a('expert_gating_func')}: only the softmax router is mapped")
    ne, nu = int(a("expert_count")), int(a("expert_used_count"))
    ff = a("feed_forward_length", 0) or 0
    cfg = _cfg_qwen35(type("G", (), {"kv": {**g.kv, f"{arch}.nextn_predict_layers": 0, f"{arch}.block_count": int(a("block_count")) - int(a("nextn_predict_layers", 0))},
                                     "tensors": g.tensors})(), arch)
    cfg.update(num_experts=ne, num_experts_per_tok=nu, moe_intermediate_size=int(a("expert_feed_forward_length") or (ff // nu if ff else 0)),
               shared_expert_intermediate_size=int(a("expert_shared_feed_forward_length") or ff), norm_topk_prob=True, decoder_sparse_step=1,
               intermediate_size=ff or int(a("expert_shared_feed_forward_length") or 0), mtp_num_hidden_layers=0)
    return cfg


def _cfg_glm4moe(g: GF.GgufFile, arch: str) -> dict:
    """GLM-4.5 family (glm4_moe class): partial NEOX rotary (the converter does not permute q / k, glm.py "unlike GLM4V non-MoE"), sigmoid router with the correction bias,
    optional top-k renormalisation then expert_weights_scale (build_moe_ffn: scale applied when not 0 / 1), shared experts. llama.cpp has no group-limited routing, so n_group = 1
    (mlx-lm then does the same plain top-k)."""
    a = _kv_getter(g.kv, arch)
    _guard_common(a, arch)
    if a("expert_gating_func") not in (None, 0, 2):                # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID = 2; NONE becomes sigmoid (glm4-moe.cpp:16-18)
        raise GF.GgufError(f"GGUF {arch}.expert_gating_func={a('expert_gating_func')}: only the sigmoid router is mapped")
    sec = a("rope.dimension_sections")
    if sec is not None and any(int(x) for x in list(sec)[1:]):
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_sections={list(sec)}: multi-section rope is not mapped")
    nh = int(a("attention.head_count"))
    hd = int(a("attention.key_length") or (a("embedding_length") // nh))
    nx = int(a("nextn_predict_layers", 0))
    nu = int(a("expert_used_count"))
    norm = bool(a("expert_weights_norm", False))
    if nu == 1 and norm:
        raise GF.GgufError(f"GGUF {arch}: expert_used_count 1 with expert_weights_norm is not what the glm4_moe class computes (it skips the norm for one expert)")
    scale = float(a("expert_weights_scale") or 1.0)
    ns = int(a("expert_shared_count", 0) or 0)
    return dict(hidden_size=a("embedding_length"), num_hidden_layers=int(a("block_count")) - nx, intermediate_size=a("feed_forward_length"), num_attention_heads=nh,
                num_key_value_heads=a("attention.head_count_kv", nh), head_dim=hd, rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-5),
                rope_theta=a("rope.freq_base", 10000.0), rope_scaling=None, partial_rotary_factor=int(a("rope.dimension_count", hd)) / hd,
                max_position_embeddings=a("context_length"), vocab_size=g.tensors["token_embd.weight"].ne[1],
                attention_bias=any(n.endswith(".attn_q.bias") for n in g.tensors), use_qk_norm=any(n.endswith(".attn_q_norm.weight") for n in g.tensors),
                tie_word_embeddings="output.weight" not in g.tensors, n_routed_experts=int(a("expert_count")), num_experts_per_tok=nu,
                moe_intermediate_size=int(a("expert_feed_forward_length") or (a("feed_forward_length") // nu)), n_shared_experts=ns or None,
                first_k_dense_replace=int(a("leading_dense_block_count", 0) or 0), routed_scaling_factor=scale, norm_topk_prob=norm, n_group=1, topk_group=1,
                scoring_func="sigmoid", topk_method="noaux_tc")


def _cfg_deepseek2(g: GF.GgufFile, arch: str) -> dict:
    """DeepSeek-V3 class MLA model from metadata (conversion/deepseek.py set_gguf_parameters). Refused rather than approximated: yarn / any rope scaling (its mscale enters the
    attention scale and the rotary tables twice, deepseek2.cpp:302-309; for a V3 file the model's config.json next to the file serves it), a softmax router (V2 / V2-Lite need the deepseek_v2
    class, which holds an unsplit kv_b_proj), files without exp_probs_b. llama.cpp has no group-limited routing, so n_group = 1."""
    a = _kv_getter(g.kv, arch)
    _guard_common(a, arch)
    if a("rope.scaling.yarn_log_multiplier") is not None or a("attention.temperature_scale"):
        raise GF.GgufError(f"GGUF {arch}: yarn log multiplier / attention temperature scale are not mapped (a DeepSeek-V3 config.json next to the file does serve it)")
    if a("expert_gating_func") != 2 or not any(n.endswith(".exp_probs_b.bias") for n in g.tensors):
        raise GF.GgufError(f"GGUF {arch}: only the sigmoid router with exp_probs_b (DeepSeek-V3 class) is mapped, not softmax gating (DeepSeek-V2 / V2-Lite need the deepseek_v2 class: not mapped)")
    if not any(n.endswith(".attn_k_b.weight") for n in g.tensors):
        raise GF.GgufError(f"GGUF {arch}: no attn_k_b / attn_v_b tensors (legacy unsplit attn_kv_b file): not mapped")
    rope = int(a("rope.dimension_count"))
    nh = int(a("attention.head_count"))
    nx = int(a("nextn_predict_layers", 0))
    nu = int(a("expert_used_count"))
    ns = int(a("expert_shared_count", 0) or 0)
    return dict(hidden_size=a("embedding_length"), num_hidden_layers=int(a("block_count")) - nx, intermediate_size=a("feed_forward_length"),
                moe_intermediate_size=a("expert_feed_forward_length"), num_attention_heads=nh, num_key_value_heads=nh, n_shared_experts=ns or None,
                n_routed_experts=int(a("expert_count")), routed_scaling_factor=float(a("expert_weights_scale") or 1.0), kv_lora_rank=int(a("attention.kv_lora_rank")),
                q_lora_rank=int(a("attention.q_lora_rank")) if a("attention.q_lora_rank") else None, qk_rope_head_dim=rope,
                v_head_dim=int(a("attention.value_length_mla")), qk_nope_head_dim=int(a("attention.key_length_mla")) - rope, topk_method="noaux_tc", scoring_func="sigmoid",
                norm_topk_prob=bool(a("expert_weights_norm", False)), n_group=1, topk_group=1, num_experts_per_tok=nu, moe_layer_freq=1,
                first_k_dense_replace=int(a("leading_dense_block_count", 0) or 0), max_position_embeddings=a("context_length"), rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-6),
                rope_theta=a("rope.freq_base", 10000.0), rope_scaling=None, vocab_size=g.tensors["token_embd.weight"].ne[1],
                attention_bias=any(n.endswith(".attn_q_a.bias") for n in g.tensors), tie_word_embeddings="output.weight" not in g.tensors)


def _guard_common(a, arch, rope=True):
    rs = a("rope.scaling.type")
    if rope and rs not in (None, "none"):
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={rs!r} is not mapped{_hint(arch)}")
    if a("attn_logit_softcapping") or a("final_logit_softcapping") or a("logit_scale") or a("residual_scale") or a("embedding_scale"):
        raise GF.GgufError(f"GGUF {arch}: soft-capping / logit, residual or embedding scales are not mapped for this architecture")


def _cfg_dense(g: GF.GgufFile, arch: str) -> dict:
    """qwen2 / llama: the plain decoder (GQA, SwiGLU, full rotary). Anything the mlx-lm class would run differently is refused, not approximated."""
    kv = g.kv
    a = _kv_getter(kv, arch)
    rs = a("rope.scaling.type")
    if rs not in (None, "none"):
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={rs!r} is not mapped{_hint(arch)}")
    if a("expert_count"):
        raise GF.GgufError(f"GGUF {arch}.expert_count={a('expert_count')}: a mixture-of-experts file of this architecture is not mapped")
    hd = a("attention.key_length") or (a("embedding_length") // a("attention.head_count"))
    if a("rope.dimension_count", hd) != hd:
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_count={a('rope.dimension_count')} != head size {hd}: partial rotary is not mapped; put config.json next to the file")
    if a("attention.temperature_scale") or a("rope.scaling.yarn_log_multiplier"):
        raise GF.GgufError(f"GGUF {arch}: attention temperature scaling / yarn log multiplier (Ministral-3 class) is not mapped")
    if a("attention.sliding_window") or a("attn_logit_softcapping") or a("final_logit_softcapping") or a("logit_scale") or a("residual_scale") or a("embedding_scale"):
        raise GF.GgufError(f"GGUF {arch}: sliding window / soft-capping / logit, residual or embedding scales are not mapped for this architecture")
    cfg = dict(hidden_size=a("embedding_length"), num_hidden_layers=a("block_count"), intermediate_size=a("feed_forward_length", 0), num_attention_heads=a("attention.head_count"),
               num_key_value_heads=a("attention.head_count_kv", a("attention.head_count")), head_dim=hd, rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-6),
               rope_theta=a("rope.freq_base", 10000.0), max_position_embeddings=a("context_length"), vocab_size=g.tensors["token_embd.weight"].ne[1],
               tie_word_embeddings="output.weight" not in g.tensors, hidden_act="silu", mlp_bias=False,
               attention_bias=any(n.endswith(".attn_q.bias") for n in g.tensors))
    return cfg


def _cfg_qwen2moe(g: GF.GgufFile, arch: str) -> dict:
    """qwen2moe = the dense config + the routed experts and the shared expert (mlx-lm qwen2_moe: softmax router, no top-k renormalisation, sigmoid-gated shared expert)."""
    a = _kv_getter(g.kv, arch)
    expert_count = a("expert_count")
    cfg = _cfg_dense(type("G", (), {"kv": {**g.kv, f"{arch}.expert_count": None}, "tensors": g.tensors})(), arch)
    if not expert_count or a("leading_dense_block_count") or a("expert_weights_norm") or a("expert_shared_count", 1) not in (1, None):
        raise GF.GgufError(f"GGUF {arch}: expert_count / leading dense blocks / expert_weights_norm / shared expert count do not match the qwen2_moe class")
    cfg.update(num_experts=expert_count, num_experts_per_tok=a("expert_used_count"), moe_intermediate_size=a("expert_feed_forward_length"),
               shared_expert_intermediate_size=a("expert_shared_feed_forward_length"))
    return cfg


def _cfg_glm4(g: GF.GgufFile, arch: str) -> dict:
    """GLM-4-0414 (glm4): partial interleaved rotary (mlx-lm glm4 rope_traditional=True); a multi-section rope means the converter permuted q / k (Glm4Model.normal_to_neox): not mapped."""
    a = _kv_getter(g.kv, arch)
    _guard_common(a, arch)
    sec = a("rope.dimension_sections")
    if sec is not None and len(sec) >= 2 and sec[0] > 0 and sec[1] > 0:
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_sections={list(sec)}: the converter permuted q / k for M-RoPE (normal_to_neox); not mapped")
    hd = a("attention.key_length") or (a("embedding_length") // a("attention.head_count"))
    nx = int(a("nextn_predict_layers", 0))
    return dict(hidden_size=a("embedding_length"), num_hidden_layers=int(a("block_count")) - nx, intermediate_size=a("feed_forward_length"),
                num_attention_heads=a("attention.head_count"), num_key_value_heads=a("attention.head_count_kv", a("attention.head_count")), head_dim=hd,
                rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-5), rope_theta=a("rope.freq_base", 10000.0), partial_rotary_factor=a("rope.dimension_count", hd // 2) / hd,
                max_position_embeddings=a("context_length"), vocab_size=g.tensors["token_embd.weight"].ne[1], attention_bias=any(n.endswith(".attn_q.bias") for n in g.tensors),
                rope_traditional=True)


def _cfg_gemma3(g: GF.GgufFile, arch: str) -> dict:
    """Gemma 3 text (src/models/gemma3.cpp load_arch_hparams): sliding window with a pattern of 6, local rope base from rope.freq_base_swa, linear rope scaling on the
    global layers only, attention scale 1/sqrt(n_embd / n_head) for the 62-layer model else 1/sqrt(head size)."""
    a = _kv_getter(g.kv, arch)
    _guard_common(a, arch, rope=False)
    if a("rope.scaling.type") not in (None, "none", "linear"):
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={a('rope.scaling.type')!r} is not mapped (only linear, on the global layers)")
    if a("attention.sliding_window_pattern") is not None and not isinstance(a("attention.sliding_window_pattern"), int):
        raise GF.GgufError(f"GGUF {arch}.attention.sliding_window_pattern is a per-layer array; the mlx-lm class only has the every-N pattern")
    hd = a("attention.key_length")
    nl, nh = int(a("block_count")), int(a("attention.head_count"))
    sw = a("attention.sliding_window") or 0
    cfg = dict(hidden_size=a("embedding_length"), num_hidden_layers=nl, intermediate_size=a("feed_forward_length"), num_attention_heads=nh,
               num_key_value_heads=a("attention.head_count_kv", nh), head_dim=hd, rms_norm_eps=a("attention.layer_norm_rms_epsilon", 1e-6),
               rope_theta=a("rope.freq_base", 10000.0), rope_local_base_freq=a("rope.freq_base_swa", 10000.0), max_position_embeddings=a("context_length"),
               vocab_size=g.tensors["token_embd.weight"].ne[1], sliding_window=sw or 512, sliding_window_pattern=int(a("attention.sliding_window_pattern", 6)) if sw else 1,
               query_pre_attn_scalar=int(a("embedding_length") // nh) if nl == 62 else int(hd))     # int: transformers' Gemma3TextConfig is a strict dataclass (float 256.0 refused at load)
    if a("rope.scaling.type") == "linear":
        cfg["rope_scaling"] = {"type": "linear", "rope_type": "linear", "factor": float(a("rope.scaling.factor"))}
    return cfg


_CONFIG = {"qwen3": _cfg_qwen3, "qwen3moe": _cfg_qwen3, "qwen35": _cfg_qwen35, "qwen35moe": _cfg_qwen35moe, "glm4moe": _cfg_glm4moe, "deepseek2": _cfg_deepseek2, "dense": _cfg_dense, "qwen2moe": _cfg_qwen2moe, "glm4": _cfg_glm4, "gemma3": _cfg_gemma3}


def gguf_config(g: GF.GgufFile) -> dict:
    """HF-style config from GGUF metadata, used only when the directory has no config.json (the tokenizer / template / lane choices never come from here)."""
    arch = g.kv.get("general.architecture")
    if arch not in ARCHS:
        raise GF.GgufError(f"GGUF architecture {arch!r} is not mapped (mapped: {', '.join(ARCHS)}); put the model's config.json next to the file")
    A = ARCHS[arch]
    return dict(model_type=A.model_type, architectures=list(A.architectures), **_CONFIG[A.config](g, arch))


# ---------------------------------------------------------------------------------------------------------- a GGUF as a servable directory
_QWEN2_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_SERVE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


# Qwen's own tokenizer.json normalises NFC; llama.cpp does not normalise at all, so no other pre gets a normalizer.
_NFC_PRES = ("qwen2", "qwen35")
# the pre whose regex this file has always built from the tokenizer.json text ((?i:...) instead of llama.cpp's [sS] spelling): output stays byte-identical
_HF_SPLIT = {"QWEN2": [_QWEN2_SPLIT]}
_SAMPLING = {"qwen2": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}}      # GGUF has no sampling values: the Qwen3 card ones, for the pre they were chosen with (no other model gets invented ones)


def _spm_merges(vocab: dict, scores: dict) -> list:
    """SentencePiece BPE merges from the pieces and their scores: every split of a piece into two pieces of the vocabulary, ordered by the score of the merged piece (then by length),
    the same construction as transformers' SpmConverter.generate_merges (which is how HF builds llama / gemma tokenizer.json from a .model file). llama.cpp's llm_tokenizer_spm
    (llama-vocab.cpp) merges the adjacent pair whose merged piece has the highest score, which this ordering reproduces; ties between equal scores are broken by the pair's
    token ids here (llama.cpp: by the left position), so a vocabulary with many equal scores is not proven identical."""
    merges = []
    for piece, sc in scores.items():
        local = []
        for i in range(1, len(piece)):
            l, r = piece[:i], piece[i:]
            if l in vocab and r in vocab:
                local.append((l, r, sc))
        local.sort(key=lambda x: (vocab[x[0]], vocab[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda m: (m[2], len(m[0]), len(m[1])), reverse=True)
    return [(l, r) for l, r, _ in merges]


def _write_spm_tokenizer(kv: dict, out: str) -> None:
    """tokenizer.json for `tokenizer.ggml.model = llama` (SentencePiece: llama-2, mistral-7b, gemma): pieces with scores and types, byte fallback tokens <0xNN>, the
    space prefix `tokenizer.ggml.add_space_prefix` (default true; gemma: false), BOS/EOS as the file asks (add_bos defaults to true for this model, llama-vocab.cpp)."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, processors
    toks = [str(t) for t in kv["tokenizer.ggml.tokens"]]
    scores = [float(x) for x in kv["tokenizer.ggml.scores"]] if "tokenizer.ggml.scores" in kv else [0.0] * len(toks)
    types = list(kv["tokenizer.ggml.token_type"]) if "tokenizer.ggml.token_type" in kv else [1] * len(toks)
    if not (len(toks) == len(scores) == len(types)):
        raise GF.GgufError("GGUF tokenizer: tokens / scores / token_type lengths differ")
    vocab = {t: i for i, t in enumerate(toks)}
    # llama.cpp token types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte
    norm = {t: sc for t, sc, ty in zip(toks, scores, types) if ty in (1, 4)}
    merges = _spm_merges(vocab, norm)
    unk_id = next((i for i, ty in enumerate(types) if ty == 2), kv.get("tokenizer.ggml.unknown_token_id"))
    unk = toks[int(unk_id)] if unk_id is not None else None
    byte_fallback = any(ty == 6 for ty in types)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=unk, fuse_unk=True, byte_fallback=byte_fallback))
    prefix = bool(kv.get("tokenizer.ggml.add_space_prefix", True))
    tok.normalizer = normalizers.Sequence(([normalizers.Prepend("\u2581")] if prefix else []) + [normalizers.Replace(" ", "\u2581")])
    tok.decoder = decoders.Sequence([decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse()] + ([decoders.Strip(" ", 1, 0)] if prefix else []))
    tok.add_special_tokens([AddedToken(t, special=ty == 3, normalized=False) for t, ty in zip(toks, types) if ty in (3, 4)])
    bos, eos = kv.get("tokenizer.ggml.bos_token_id"), kv.get("tokenizer.ggml.eos_token_id")
    add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", True)) and bos is not None
    add_eos = bool(kv.get("tokenizer.ggml.add_eos_token", False)) and eos is not None
    if add_bos or add_eos:
        single = (f"{toks[int(bos)]} " if add_bos else "") + "$A" + (f" {toks[int(eos)]}" if add_eos else "")
        sp = ([(toks[int(bos)], int(bos))] if add_bos else []) + ([(toks[int(eos)], int(eos))] if add_eos else [])
        tok.post_processor = processors.TemplateProcessing(single=single, pair=None, special_tokens=sp)
    tok.save(os.path.join(out, "tokenizer.json"))
    tc = {"tokenizer_class": "PreTrainedTokenizerFast", "clean_up_tokenization_spaces": False, "add_bos_token": add_bos, "add_eos_token": add_eos}
    if eos is not None:
        tc["eos_token"] = toks[int(eos)]
    if bos is not None:
        tc["bos_token"] = toks[int(bos)]
    if unk is not None:
        tc["unk_token"] = unk
    if "tokenizer.ggml.padding_token_id" in kv:
        tc["pad_token"] = toks[int(kv["tokenizer.ggml.padding_token_id"])]
    if kv.get("tokenizer.chat_template"):
        tc["chat_template"] = str(kv["tokenizer.chat_template"])
    with open(os.path.join(out, "tokenizer_config.json"), "w") as fh:
        json.dump(tc, fh)
    stops = ([int(eos)] if eos is not None else []) + [int(kv[k]) for k in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id") if k in kv and int(kv[k]) != eos]
    with open(os.path.join(out, "generation_config.json"), "w") as fh:
        json.dump({"eos_token_id": stops}, fh)


def write_gguf_tokenizer(kv: dict, out: str) -> None:
    """tokenizer.json, tokenizer_config.json and generation_config.json from GGUF metadata, for a `gpt2` (byte-level BPE) vocabulary whose `tokenizer.ggml.pre`
    llama.cpp knows (foreign_pre.PRE_NAMES / PRE_REGEX, generated from src/llama-vocab.cpp): the split regexes in order, the byte-level map, ignore_merges,
    the BOS the model adds. Anything else is refused by name."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from . import foreign_pre as PRE

    model, pre = kv.get("tokenizer.ggml.model"), kv.get("tokenizer.ggml.pre")
    if model == "llama":
        return _write_spm_tokenizer(kv, out)
    if model != "gpt2":
        raise GF.GgufError(f"GGUF tokenizer {(model, pre)} is not mapped (gpt2 byte-level BPE and llama (SentencePiece) are built; llama.cpp wpm / ugm / rwkv / plamo2 are not); put tokenizer.json next to the file")
    if pre not in PRE.PRE_NAMES:
        raise GF.GgufError(f"GGUF tokenizer.ggml.pre={pre!r} is not a pre-tokenizer llama.cpp {PRE.LLAMA_CPP_COMMIT[:8]} knows; put tokenizer.json next to the file")
    ptype, ignore_merges, add_bos, _clean, _nobos = PRE.PRE_NAMES[pre]
    byte_encode, exprs = PRE.PRE_REGEX.get(ptype, PRE.PRE_REGEX["DEFAULT"])
    exprs = _HF_SPLIT.get(ptype, exprs) if pre == "qwen2" else exprs
    if not byte_encode:
        raise GF.GgufError(f"GGUF tokenizer.ggml.pre={pre!r} does not use the GPT-2 byte map (llama.cpp byte_encode=false): not built here; put tokenizer.json next to the file")
    try:
        splits = [pre_tokenizers.Split(Regex(e), behavior="isolated") for e in exprs]
    except Exception as e:                                          # e.g. afmoe: a custom \p{...} class the regex engine of `tokenizers` does not have
        raise GF.GgufError(f"GGUF tokenizer.ggml.pre={pre!r}: a split regex of llama.cpp is not supported by the tokenizers library ({e}); put tokenizer.json next to the file")
    toks = [str(t) for t in kv["tokenizer.ggml.tokens"]]
    merges = [tuple(str(m).split(" ", 1)) for m in kv["tokenizer.ggml.merges"]]
    types = list(kv["tokenizer.ggml.token_type"])
    vocab = {t: i for i, t in enumerate(toks)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=True) if ignore_merges else models.BPE(vocab=vocab, merges=merges))
    if pre in _NFC_PRES:
        tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(splits + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(t, special=ty == 3, normalized=False) for t, ty in zip(toks, types) if ty in (3, 4)])
    bos = kv.get("tokenizer.ggml.bos_token_id")
    add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", add_bos))
    add_eos = bool(kv.get("tokenizer.ggml.add_eos_token", False))
    eos = int(kv["tokenizer.ggml.eos_token_id"])
    if add_bos and bos is None:
        raise GF.GgufError(f"GGUF tokenizer.ggml.pre={pre!r} adds a BOS but the file has no tokenizer.ggml.bos_token_id")
    if add_bos or add_eos:                                          # llama_vocab::tokenize adds them when the vocab says so (add_bos from the pre default, tokenizer.ggml.add_bos_token overrides)
        single = ("<bos> " if add_bos else "") + "$A" + (" <eos>" if add_eos else "")
        sp = ([("<bos>", int(bos))] if add_bos else []) + ([("<eos>", eos)] if add_eos else [])
        tok.post_processor = processors.TemplateProcessing(single=single.replace("<bos>", toks[int(bos)] if add_bos else "").replace("<eos>", toks[eos]),
                                                           pair=None, special_tokens=[(toks[i], i) for _, i in sp])
    tok.save(os.path.join(out, "tokenizer.json"))
    tc = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": toks[eos], "clean_up_tokenization_spaces": False}
    if add_bos:
        tc["bos_token"] = toks[int(bos)]
    if "tokenizer.ggml.padding_token_id" in kv:
        tc["pad_token"] = toks[int(kv["tokenizer.ggml.padding_token_id"])]
    if kv.get("tokenizer.chat_template"):
        tc["chat_template"] = str(kv["tokenizer.chat_template"])
    with open(os.path.join(out, "tokenizer_config.json"), "w") as fh:
        json.dump(tc, fh)
    stops = [eos] + [int(kv[k]) for k in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id") if k in kv and int(kv[k]) != eos]     # llama_vocab special_eog_ids from metadata
    with open(os.path.join(out, "generation_config.json"), "w") as fh:
        json.dump({"eos_token_id": stops, **_SAMPLING.get(pre, {})}, fh)


def gguf_serve_dir(path: str, cache: str | None = None) -> str:
    """The directory the server reads for a GGUF model. A file, or a directory that lacks config or tokenizer files, gets a cached directory: config and
    tokenizer built from the metadata, the GGUF itself linked, files already next to the GGUF kept. Anything else comes back unchanged."""
    p = str(path)
    d = p if os.path.isdir(p) else os.path.dirname(p) or "."
    if os.path.isdir(p) and (not GF.find_gguf(p) or all(os.path.isfile(os.path.join(p, f)) for f in _SERVE_FILES[:2])):
        return p
    if not os.path.isdir(p) and not (os.path.isfile(p) and p.endswith(".gguf")):
        return p
    src = os.path.realpath(GF.find_gguf(p))
    st = os.stat(src)
    side = "".join(f"|{f}:{os.stat(os.path.join(d, f)).st_size}:{os.stat(os.path.join(d, f)).st_mtime_ns}"          # the cached directory copies these: an edited one must not hit a stale build
                    for f in _SERVE_FILES + ("generation_config.json",) if os.path.isfile(os.path.join(d, f)))
    key = hashlib.sha1(f"v5|{src}|{st.st_size}|{st.st_mtime_ns}{side}".encode()).hexdigest()[:16]
    root = cache or os.environ.get("KYOJIN_GGUF_CACHE") or os.path.expanduser("~/.cache/kyojin_mlx/gguf")
    out = os.path.join(root, f"{os.path.basename(src)[:-5]}-{key}")
    if os.path.isdir(out):
        return out
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".build-", dir=root)
    try:
        g = GF.GgufFile(src)
        try:
            if not os.path.isfile(os.path.join(d, "tokenizer.json")):
                write_gguf_tokenizer(g.kv, tmp)
            if os.path.isfile(os.path.join(d, "config.json")):
                shutil.copyfile(os.path.join(d, "config.json"), os.path.join(tmp, "config.json"))
            else:
                with open(os.path.join(tmp, "config.json"), "w") as fh:
                    json.dump(gguf_config(g), fh)
        finally:
            g.close()
        for f in _SERVE_FILES[1:] + ("generation_config.json",):
            if os.path.isfile(os.path.join(d, f)):
                shutil.copyfile(os.path.join(d, f), os.path.join(tmp, f))
        os.symlink(src, os.path.join(tmp, os.path.basename(src)))
        os.rename(tmp, out)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if os.path.isdir(out):      # another start built it first
            return out
        raise
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out
