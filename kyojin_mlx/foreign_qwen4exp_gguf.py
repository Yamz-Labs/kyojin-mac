# SPDX-License-Identifier: Apache-2.0
"""GGUF side of Qwen3.8-Flash-Next (llama.cpp arch `qwen4exp`): rows only, no model code. Unproven: the rows exist only while KYOJIN_KIT_UNPROVEN_ROWS is on.

The file reaches mlx-vlm's `qwen4_exp` class (foreign_linear.VLM_FAMILIES) under the HF names of its checkpoint. What llama.cpp's converter did on the way
(conversion/qwen4exp.py, then the inherited conversion/qwen.py, lines cited per rule):

  qwen.py:425-428    A_log -> -exp(A_log) (`ssm_a`), dt_bias renamed                                      -> POST "a_log" (foreign_xform), exact names
  qwen.py:431-432    every `*norm.weight` but linear_attn.norm gets +1 (hc norms, q / k norms, hc head)     -> POST "minus1"
  qwen4exp.py:163-166 ple norm_key / norm_query / norm_conv and indexer q / k layernorm get +1               -> POST "minus1"
  qwen4exp.py:168    ple conv1d squeezed [C, 1, K] -> [C, K]                                                   -> POST "conv" (the class's sanitize moves it to [C, K, 1])
  qwen.py (V reorder) linear-attention V heads grouped -> tiled (nk 16 != nv 48)                            -> rows / cols of foreign_xform._qwen35 (keys `qwen4exp.ssm.*`)
  qwen4exp.py:156-159 indexer q and k projections split out of `index_qk_proj` (q rows first)               -> Arch.concat: row-concatenated back (q then k)
  qwen4exp.py:174-181 shared-input merges (hc down|inject, qkv|gate, beta|alpha): NOT applied in the real files (all parts are separate tensors in both headers)
  qwen4exp.py:185-212 the 128 PLE shards concatenated by rows into `per_layer_token_embd` [rows, 160]    -> Arch.unserved (no loader path for a 28.8 GB gathered table yet)
Routed experts stay stacked ([E, n, k] per projection, one tensor each); the MoE block is mlx-vlm's Qwen3_5MoeSparseMoeBlock (softmax, top-k, renormalised, sigmoid-gated shared expert).
The MTP block (nextn_*) is skipped (skip_nextn); neither real file carries one."""
from __future__ import annotations

from . import foreign_gguf as GF
from . import foreign_names as FN
from . import foreign_xform as XF

LM = "model.language_model."
HC = "hyper_connection."
_HC = {"down": "input_mix_weight_down", "up": "input_mix_weight_up", "inject": "block_inject_weight", "norm": "hc_norm"}
BLK = {
    # gated delta net layers (names as the qwen35 row)
    "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj", "ssm_norm": "linear_attn.norm", "ssm_conv1d": "linear_attn.conv1d",
    # QSA layers (full attention + indexer)
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "indexer.q_proj": "self_attn.indexer.q_proj", "indexer.k_proj": "self_attn.indexer.k_proj",
    "indexer.q_norm": "self_attn.indexer.q_layernorm", "indexer.k_norm": "self_attn.indexer.k_layernorm",
    # per-layer embedding block (one layer)
    "ple_key": "ple.key_proj", "ple_value": "ple.value_proj", "ple_norm_key": "ple.norm_key", "ple_norm_query": "ple.norm_query", "ple_norm_conv": "ple.norm_conv",
    "ple_conv1d": "ple.conv1d",
    # MoE: router, stacked routed experts, shared expert and its gate
    **FN._MOE, **FN._SHEXP,
}
for _s, _h in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
    for _k, _v in _HC.items():
        BLK[f"hc_{_s}_{_k}"] = f"{_h}.{_v}"
EXACT = dict(FN._EXACT35)
TOP = {"token_embd.weight": LM + "embed_tokens.weight", "output.weight": "lm_head.weight",
       "output_hc_down.weight": LM + "hyper_connection_mixer.input_mix_weight_down.weight", "output_hc_up.weight": LM + "hyper_connection_mixer.input_mix_weight_up.weight",
       "output_hc_norm.weight": LM + "hyper_connection_mixer.hc_norm.weight",
       "per_layer_token_embd.weight": LM + "ple_table.weight"}                 # placeholder name: the class holds 128 shards per PLE layer (see unserved)
CONCAT = {"self_attn.indexer.index_qk_proj": ("self_attn.indexer.q_proj", "self_attn.indexer.k_proj")}
UNSERVED = ("per_layer_token_embd.weight",)
# the PLE n-gram table: the class's `ple.ple_embedding.ngram_embedding` looks up (B, T, heads) row ids; with KYOJIN_MAPPED_ROW_TABLE=1 it reads them from the mapped file (foreign_mapped_rows)
MAPPED_ROWS = {"per_layer_token_embd.weight": ".ple.ple_embedding.ngram_embedding"}
# never a quantised linear: convolutions, and the four-row hc injection matrix (F32 in the file, the class reads it as a plain linear)
DENSE = ("conv1d.weight", "block_inject_weight.weight")
ARCH = FN.Arch("qwen4_exp", ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"), blk=dict(BLK), exact=dict(EXACT), top=dict(TOP), dense=DENSE, skip_nextn=True,
               xform="qwen4exp", config="qwen4exp", layer_prefix=LM + "layers.", concat=dict(CONCAT), unserved=UNSERVED, mapped_rows=dict(MAPPED_ROWS))
_PLE_NORMS = ("ple_norm_key", "ple_norm_query", "ple_norm_conv")


def xform(name: str, kv: dict) -> XF.Xf | None:
    """The qwen35 set under the `qwen4exp.` keys (norm -1, A_log, V-head reorder, conv) plus the norms it does not know (hc head, ple) and the ple conv."""
    if name == "output_hc_norm.weight":
        return XF.Xf(post="minus1")
    if name.startswith("blk."):
        rest = name.split(".", 2)[2]
        kind = rest.rpartition(".")[0] if rest.endswith((".weight", ".bias")) else rest
        if kind in _PLE_NORMS:
            return XF.Xf(post="minus1")
        if kind == "ple_conv1d":
            return XF.Xf(post="conv")
        if kind == "ffn_gate_inp_shexp":
            return XF.Xf(post="row")
    return XF._qwen35(name, kv, "qwen4exp.")


def config(g: GF.GgufFile, arch: str) -> dict:
    """mlx-vlm qwen4_exp config (text_config + top level) from the metadata. Refused: rope scaling, another router than softmax, a routing scale, key != value length,
    a compress ratio list that is not 0 on linear layers and one ratio on the full-attention layers. UNVERIFIED against a Metal run."""
    a = FN._kv_getter(g.kv, arch)
    if a("rope.scaling.type") not in (None, "none"):
        raise GF.GgufError(f"GGUF {arch}.rope.scaling.type={a('rope.scaling.type')!r} is not mapped")
    if a("expert_weights_scale") not in (None, 0, 0.0, 1, 1.0) or a("expert_gating_func") not in (None, 0, 1):
        raise GF.GgufError(f"GGUF {arch}: expert_weights_scale / expert_gating_func: only the plain softmax top-k router is mapped")
    nx = int(a("nextn_predict_layers", 0) or 0)
    nt = int(a("block_count")) - nx
    nk, nv, ds, inner = int(a("ssm.group_count")), int(a("ssm.time_step_rank")), int(a("ssm.state_size")), int(a("ssm.inner_size"))
    if inner % nv or inner // nv != ds or nv % nk:
        raise GF.GgufError(f"GGUF {arch}.ssm: inner_size {inner} / time_step_rank {nv} != state_size {ds}")
    hd = int(a("attention.key_length"))
    if int(a("attention.value_length", hd)) != hd:
        raise GF.GgufError(f"GGUF {arch}: key_length != value_length")
    iv = int(a("full_attention_interval", 4))
    cr = list(a("attention.compress_ratios"))[:nt]
    full = [(i + 1) % iv == 0 for i in range(nt)]
    ratios = {int(x) for x, f in zip(cr, full) if f}
    if len(cr) != nt or len(ratios) != 1 or any((int(x) != 0) == (not f) for x, f in zip(cr, full)):
        raise GF.GgufError(f"GGUF {arch}.attention.compress_ratios does not follow full_attention_interval={iv}")
    rd = int(a("rope.dimension_count"))
    sec = list(a("rope.dimension_sections", [11, 11, 10, 0]))
    if len(sec) != 4 or sec[3] != 0 or 2 * sum(sec[:3]) != rd:
        raise GF.GgufError(f"GGUF {arch}.rope.dimension_sections={sec} does not match rope.dimension_count={rd}")
    ple = [int(i) + 1 for i in (a("ple.layers") or [])]                      # the file counts from 0, the class from 1
    tc = dict(model_type="qwen4_exp", hidden_size=int(a("embedding_length")), num_hidden_layers=nt, num_attention_heads=int(a("attention.head_count")),
              num_key_value_heads=int(a("attention.head_count_kv")), head_dim=hd, linear_num_value_heads=nv, linear_num_key_heads=nk, linear_key_head_dim=ds,
              linear_value_head_dim=inner // nv, linear_conv_kernel_dim=int(a("ssm.conv_kernel")), num_experts=int(a("expert_count")), num_experts_per_tok=int(a("expert_used_count")),
              moe_intermediate_size=int(a("expert_feed_forward_length")), shared_expert_intermediate_size=int(a("expert_shared_feed_forward_length")),
              rms_norm_eps=float(a("attention.layer_norm_rms_epsilon", 1e-6)), vocab_size=int(g.tensors["token_embd.weight"].ne[1]), max_position_embeddings=int(a("context_length")),
              hc_count=int(a("hyper_connection.count")), hc_lowrank=int(a("hyper_connection.low_rank")), full_attention_interval=iv,
              indexer_n_heads=int(a("attention.indexer.head_count")), indexer_kv_heads=1, indexer_head_dim=int(a("attention.indexer.key_length")), indexer_budget=int(a("attention.indexer.top_k")),
              indexer_compress_ratio=ratios.pop(), norm_topk_prob=True, mtp_num_hidden_layers=0, attention_bias=False, hidden_act="silu",
              tie_word_embeddings="output.weight" not in g.tensors,
              rope_parameters=dict(type="default", mrope_section=sec[:3], rope_theta=float(a("rope.freq_base", 1e7)), partial_rotary_factor=rd / hd))
    if ple:
        tc.update(ple_layer_ids=ple, ngram_size=int(a("ple.ngram_size")), heads_per_ngram=int(a("ple.heads_per_ngram")), ple_conv_kernel_size=int(a("ple.conv_kernel")),
                  eos_token_id=int(a("ple.eos_token_id")))
        row = a("embedding_length_per_layer_input")
        if row is not None and int(row) * int(a("ple.heads_per_ngram")) * (int(a("ple.ngram_size")) - 1) != int(a("embedding_length")):
            raise GF.GgufError(f"GGUF {arch}: PLE row {row} x heads does not give the hidden size (the class sets ple_embed_dim = hidden_size)")
    return dict(model_type="qwen4_exp", text_config=tc, vision_config=dict(model_type="qwen4_exp"), tie_word_embeddings=tc["tie_word_embeddings"],
                vocab_size=tc["vocab_size"], hidden_size=tc["hidden_size"])


def register() -> None:
    """Idempotent sync with KYOJIN_KIT_UNPROVEN_ROWS (no Metal proof yet): on = the rows of qwen4exp are in foreign_names.ARCHS / _CONFIG and foreign_xform._SETS; off = none."""
    from . import foreign_unproven as UP
    if UP.on("qwen4exp"):
        FN.ARCHS.setdefault("qwen4exp", ARCH)
        FN._CONFIG.setdefault("qwen4exp", config)
        XF._SETS.setdefault("qwen4exp", xform)
    else:
        FN.ARCHS.pop("qwen4exp", None)
        FN._CONFIG.pop("qwen4exp", None)
        XF._SETS.pop("qwen4exp", None)


register()
