# SPDX-License-Identifier: Apache-2.0
"""Config rows: the HF-style config of the architectures whose metadata the generated tables cannot place by key name alone, as DATA (strings of a small expression
language) instead of a Python builder per architecture. The hand builders `foreign_names._cfg_*` stay as the reference (tests/test_arch_kit_cfgrows.py: equal configs on the
synthetic files, equal refusals).

A row is  {"model_type", "architectures", "trunk", "let": [(name, expr)], "guards": [(refuse_when_expr, message)], "fields": [(config field, expr)], "extends": row name}.
Expressions are Python expressions parsed with `ast` and evaluated by a whitelist walker (nothing is exec'd or imported). What they may use:
  K('key', default=None)   metadata value `<arch>.key` (missing -> default)       T('.attn_q.bias')   some tensor name ends with it      HAS('prism.')  some metadata key starts with it
  VOCAB  rows of token_embd                                                       TIED   the file has no output.weight                  names bound by `let`
  literals, list / tuple / dict, + - * / // %, and / or / not, comparisons (in, not in, is None ...), `a if c else b`, x[i], x[i:j]
  int float bool len sum list str  IDX(list, v)  AT(list_or_scalar, i)  NZIDX(list) (indices of the non-zero entries)  MAX(list_or_scalar)  NE('tensor', i) (dim i, None if absent)  ROTATED('rope_freqs.weight')  and  REC_OK(recurrent_layers, n_trunk, interval)  (the one helper that is not a plain operator)
`extends` = the parent's let / guards / fields first, then this row's (a field with the same name replaces the parent's).
`trunk` = the number of trunk blocks (block_count minus the draft-head blocks): the loader skips the blocks past it."""
from __future__ import annotations

import ast
import operator
import os

from . import foreign_gguf as GF

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}


def _rec_ok(rec, nt, iv):
    """attention.recurrent_layers follows `full_attention_interval` (the only layer pattern of the mlx-lm class); None = the file does not carry it."""
    return rec is None or [bool(x) for x in list(rec)[:nt]] == [(i + 1) % iv != 0 for i in range(nt)]


def _rotated(g, name):
    """Rotated dimension pairs of a rope frequency-factor tensor: `n` entries 1.0 followed by entries >= 1e9 (the converter's way to leave the tail of the head unrotated);
    any other content raises. None when the file has no such tensor."""
    if g is None or name not in g.tensors:
        return None
    a = g.tensor_f32(name).reshape(-1)
    n = int((a < 1e9).sum())
    if not (a[:n] == 1.0).all() or not (a[n:] >= 1e9).all():
        raise GF.GgufError(f"{name}: not of the form [1.0] * n + [>= 1e9] * m: a rope scaling this loader does not map")
    return n


class _Env:
    def __init__(self, arch, kv, tensors, g=None):
        self.names = {}
        self.funcs = {"K": lambda k, d=None: (GF.kv_count_uniform(kv.get(f"{arch}.{k}", d), f"{arch}.{k}") if k in ("expert_count", "expert_used_count") and isinstance(kv.get(f"{arch}.{k}", d), (list, tuple)) else kv.get(f"{arch}.{k}", d)), "T": lambda s: any(n.endswith(s) for n in tensors), "HAS": lambda p: any(k.startswith(p) for k in kv),
                      "int": int, "float": float, "bool": bool, "len": len, "sum": sum, "list": list, "str": str, "REC_OK": _rec_ok,
                      "IDX": lambda a, v: list(a).index(v), "NZIDX": lambda a: [i for i, x in enumerate(a) if x],
                      "HYBRID": lambda hk, ff, n: ["*" if hk[i] else ("E" if ff[i] else "M") for i in range(n)],
                      "RANGE": lambda n: list(range(int(n))), "FALSEIDX": lambda a: [i for i, x in enumerate(a) if not x],
                      "LAYER_TYPES": lambda a: ["sliding_attention" if x else "full_attention" for x in a], "MAX": lambda a: max(a) if isinstance(a, (list, tuple)) else a, "AT": lambda a, i: a[i] if isinstance(a, (list, tuple)) else a,
                      "NE": lambda n, i: tensors[n].ne[i] if n in tensors else None, "ROTATED": lambda n: _rotated(g, n)}
        self.consts = {"VOCAB": tensors["token_embd.weight"].ne[1] if "token_embd.weight" in tensors else None, "TIED": "output.weight" not in tensors,
                       "True": True, "False": False, "None": None}


def _ev(n, e: _Env):
    t = type(n)
    if t is ast.Constant:
        return n.value
    if t is ast.Name:
        if n.id in e.names:
            return e.names[n.id]
        if n.id in e.consts:
            return e.consts[n.id]
        raise GF.GgufError(f"config row: unknown name {n.id!r}")
    if t is ast.BinOp and type(n.op) in _BIN:
        return _BIN[type(n.op)](_ev(n.left, e), _ev(n.right, e))
    if t is ast.UnaryOp and type(n.op) in (ast.Not, ast.USub):
        v = _ev(n.operand, e)
        return (not v) if type(n.op) is ast.Not else -v
    if t is ast.BoolOp:                                            # python's value semantics: `a or b` is a if a else b
        v = None
        for x in n.values:
            v = _ev(x, e)
            if (type(n.op) is ast.Or) == bool(v):
                return v
        return v
    if t is ast.Compare:
        left = _ev(n.left, e)
        for op, c in zip(n.ops, n.comparators):
            right = _ev(c, e)
            if type(op) not in _CMP or not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if t is ast.IfExp:
        return _ev(n.body, e) if _ev(n.test, e) else _ev(n.orelse, e)
    if t is ast.Subscript:
        sl = n.slice
        idx = slice(*[(_ev(x, e) if x is not None else None) for x in (sl.lower, sl.upper, sl.step)]) if type(sl) is ast.Slice else _ev(sl, e)
        return _ev(n.value, e)[idx]
    if t in (ast.List, ast.Tuple):
        v = [_ev(x, e) for x in n.elts]
        return v if t is ast.List else tuple(v)
    if t is ast.Dict:
        return {_ev(k, e): _ev(v, e) for k, v in zip(n.keys, n.values)}
    if t is ast.Call and type(n.func) is ast.Name and n.func.id in e.funcs and not n.keywords:
        return e.funcs[n.func.id](*[_ev(a, e) for a in n.args])
    raise GF.GgufError(f"config row: unsupported expression {ast.dump(n)[:80]}")


def ev(expr: str, env: _Env):
    return _ev(ast.parse(expr, mode="eval").body, env)


# ------------------------------------------------------------------------------------------------------------------------------------------ rows
_NOROPE = ("rs not in (None, 'none')", "GGUF {arch}.rope.scaling.type={rs!r} is not mapped (the config built here has no rope_scaling, the model would run unscaled)")
_NOSCALE = ("K('attn_logit_softcapping') or K('final_logit_softcapping') or K('logit_scale') or K('residual_scale') or K('embedding_scale')",
            "GGUF {arch}: soft-capping / logit, residual or embedding scales are not mapped for this architecture")
_TRUNK = "int(K('block_count')) - int(K('nextn_predict_layers', 0))"

ROWS = {
    # Qwen3.5 dense text model: keys as src/models/qwen35.cpp load_arch_hparams reads them; HF names those of the safetensors config (linear_* = ssm.*, partial_rotary_factor = rope.dimension_count / head size)
    "qwen35": dict(
        model_type="qwen3_5", architectures=("Qwen3_5ForConditionalGeneration",), trunk=_TRUNK,
        let=[("rs", "K('rope.scaling.type')"), ("nx", "int(K('nextn_predict_layers', 0))"), ("nt", "int(K('block_count')) - nx"), ("nk", "int(K('ssm.group_count'))"),
             ("nv", "int(K('ssm.time_step_rank'))"), ("ds", "int(K('ssm.state_size'))"), ("inner", "int(K('ssm.inner_size'))"), ("iv", "int(K('full_attention_interval', 4))"),
             ("hd", "int(K('attention.key_length'))"), ("rd", "int(K('rope.dimension_count'))"), ("sec", "list(K('rope.dimension_sections', [11, 11, 10, 0]))"),
             ("rope", "{'rope_type': 'default', 'rope_theta': K('rope.freq_base', 1e7), 'partial_rotary_factor': rd / hd, 'mrope_section': sec[:3], 'mrope_interleaved': True}")],
        guards=[("HAS('prism.')", "GGUF prism.*: a Hadamard-folded file keeps the linear-attention out_proj in grouped V order and folds the weights; not mapped"),
                _NOROPE,
                ("inner % nv or inner // nv != ds or nv % nk", "GGUF {arch}.ssm: inner_size {inner} / time_step_rank {nv} != state_size {ds} (or {nv} % {nk} heads): llama.cpp assumes equal key / value head size"),
                ("not REC_OK(K('attention.recurrent_layers'), nt, iv)", "GGUF {arch}.attention.recurrent_layers does not follow full_attention_interval={iv}: the mlx-lm class only has that pattern"),
                ("int(K('attention.value_length', hd)) != hd", "GGUF {arch}: key_length != value_length"),
                ("len(sec) != 4 or sec[3] != 0 or 2 * sum(sec[:3]) != rd", "GGUF {arch}.rope.dimension_sections={sec} does not match rope.dimension_count={rd}")],
        fields=[("hidden_size", "K('embedding_length')"), ("num_hidden_layers", "nt"), ("intermediate_size", "K('feed_forward_length')"), ("num_attention_heads", "K('attention.head_count')"),
                ("num_key_value_heads", "K('attention.head_count_kv')"), ("head_dim", "hd"), ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-6)"), ("rope_theta", "rope['rope_theta']"),
                ("partial_rotary_factor", "rope['partial_rotary_factor']"), ("rope_parameters", "rope"), ("max_position_embeddings", "K('context_length')"), ("vocab_size", "VOCAB"),
                ("tie_word_embeddings", "TIED"), ("hidden_act", "'silu'"), ("attention_bias", "False"), ("attn_output_gate", "True"), ("full_attention_interval", "iv"),
                ("linear_conv_kernel_dim", "K('ssm.conv_kernel')"), ("linear_key_head_dim", "ds"), ("linear_value_head_dim", "inner // nv"), ("linear_num_key_heads", "nk"),
                ("linear_num_value_heads", "nv"), ("mtp_num_hidden_layers", "nx")]),
    # qwen35moe = the qwen35 trunk + routed experts (softmax router, top-k renormalised, qwen35moe.cpp:500-502) and the shared expert; n_ff_exp falls back to n_ff / n_expert_used (qwen35moe.cpp:57-58); draft block not loaded
    "qwen35moe": dict(
        extends="qwen35", model_type="qwen3_5_moe", architectures=("Qwen3_5MoeForConditionalGeneration",),
        let=[("ne", "int(K('expert_count'))"), ("nu", "int(K('expert_used_count'))"), ("ff", "K('feed_forward_length', 0) or 0")],
        guards=[("K('expert_weights_scale') not in (None, 0, 0.0, 1, 1.0)", "GGUF {arch}.expert_weights_scale: the qwen3_5_moe class does not scale the routing weights"),
                ("K('expert_gating_func') not in (None, 0, 1)", "GGUF {arch}.expert_gating_func: only the softmax router is mapped")],
        fields=[("num_experts", "ne"), ("num_experts_per_tok", "nu"), ("moe_intermediate_size", "int(K('expert_feed_forward_length') or (ff // nu if ff else 0))"),
                ("shared_expert_intermediate_size", "int(K('expert_shared_feed_forward_length') or ff)"), ("norm_topk_prob", "True"), ("decoder_sparse_step", "1"),
                ("intermediate_size", "ff or int(K('expert_shared_feed_forward_length') or 0)"), ("mtp_num_hidden_layers", "0")]),
    # GLM-4.5 family (glm4_moe class): partial NEOX rotary, sigmoid router with correction bias, optional top-k renormalisation then expert_weights_scale, shared experts; no group-limited routing (n_group 1)
    "glm4moe": dict(
        model_type="glm4_moe", architectures=("Glm4MoeForCausalLM",), trunk=_TRUNK,
        let=[("rs", "K('rope.scaling.type')"), ("sec", "K('rope.dimension_sections')"), ("nh", "int(K('attention.head_count'))"),
             ("hd", "int(K('attention.key_length') or (K('embedding_length') // nh))"), ("nx", "int(K('nextn_predict_layers', 0))"), ("nu", "int(K('expert_used_count'))"),
             ("norm", "bool(K('expert_weights_norm', False))"), ("ns", "int(K('expert_shared_count', 0) or 0)")],
        guards=[_NOROPE, _NOSCALE,
                ("K('expert_gating_func') not in (None, 0, 2)", "GGUF {arch}.expert_gating_func: only the sigmoid router is mapped"),
                ("sec is not None and sum(sec[1:]) != 0", "GGUF {arch}.rope.dimension_sections={sec}: multi-section rope is not mapped"),
                ("nu == 1 and norm", "GGUF {arch}: expert_used_count 1 with expert_weights_norm is not what the glm4_moe class computes (it skips the norm for one expert)")],
        fields=[("hidden_size", "K('embedding_length')"), ("num_hidden_layers", "int(K('block_count')) - nx"), ("intermediate_size", "K('feed_forward_length')"), ("num_attention_heads", "nh"),
                ("num_key_value_heads", "K('attention.head_count_kv', nh)"), ("head_dim", "hd"), ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-5)"),
                ("rope_theta", "K('rope.freq_base', 10000.0)"), ("rope_scaling", "None"), ("partial_rotary_factor", "int(K('rope.dimension_count', hd)) / hd"),
                ("max_position_embeddings", "K('context_length')"), ("vocab_size", "VOCAB"), ("attention_bias", "T('.attn_q.bias')"), ("use_qk_norm", "T('.attn_q_norm.weight')"),
                ("tie_word_embeddings", "TIED"), ("n_routed_experts", "int(K('expert_count'))"), ("num_experts_per_tok", "nu"),
                ("moe_intermediate_size", "int(K('expert_feed_forward_length') or (K('feed_forward_length') // nu))"), ("n_shared_experts", "ns or None"),
                ("first_k_dense_replace", "int(K('leading_dense_block_count', 0) or 0)"), ("routed_scaling_factor", "float(K('expert_weights_scale') or 1.0)"), ("norm_topk_prob", "norm"),
                ("n_group", "1"), ("topk_group", "1"), ("scoring_func", "'sigmoid'"), ("topk_method", "'noaux_tc'")]),
    # DeepSeek-V3 class MLA model (conversion/deepseek.py set_gguf_parameters). Refused rather than approximated: any rope scaling, yarn multiplier / temperature scale, a softmax router, a file without exp_probs_b / split attn_k_b
    "deepseek2": dict(
        model_type="deepseek_v3", architectures=("DeepseekV3ForCausalLM",), trunk=_TRUNK,
        let=[("rs", "K('rope.scaling.type')"), ("rope", "int(K('rope.dimension_count'))"), ("nh", "int(K('attention.head_count'))"), ("nx", "int(K('nextn_predict_layers', 0))"),
             ("nu", "int(K('expert_used_count'))"), ("ns", "int(K('expert_shared_count', 0) or 0)")],
        guards=[_NOROPE, _NOSCALE,
                ("K('rope.scaling.yarn_log_multiplier') is not None or K('attention.temperature_scale')", "GGUF {arch}: yarn log multiplier / attention temperature scale are not mapped"),
                ("K('expert_gating_func') != 2 or not T('.exp_probs_b.bias')", "GGUF {arch}: only the sigmoid router with exp_probs_b (DeepSeek-V3 class) is mapped"),
                ("not T('.attn_k_b.weight')", "GGUF {arch}: no attn_k_b / attn_v_b tensors (legacy unsplit attn_kv_b file): not mapped")],
        fields=[("hidden_size", "K('embedding_length')"), ("num_hidden_layers", "int(K('block_count')) - nx"), ("intermediate_size", "K('feed_forward_length')"),
                ("moe_intermediate_size", "K('expert_feed_forward_length')"), ("num_attention_heads", "nh"), ("num_key_value_heads", "nh"), ("n_shared_experts", "ns or None"),
                ("n_routed_experts", "int(K('expert_count'))"), ("routed_scaling_factor", "float(K('expert_weights_scale') or 1.0)"), ("kv_lora_rank", "int(K('attention.kv_lora_rank'))"),
                ("q_lora_rank", "int(K('attention.q_lora_rank')) if K('attention.q_lora_rank') else None"), ("qk_rope_head_dim", "rope"), ("v_head_dim", "int(K('attention.value_length_mla'))"),
                ("qk_nope_head_dim", "int(K('attention.key_length_mla')) - rope"), ("topk_method", "'noaux_tc'"), ("scoring_func", "'sigmoid'"),
                ("norm_topk_prob", "bool(K('expert_weights_norm', False))"), ("n_group", "1"), ("topk_group", "1"), ("num_experts_per_tok", "nu"), ("moe_layer_freq", "1"),
                ("first_k_dense_replace", "int(K('leading_dense_block_count', 0) or 0)"), ("max_position_embeddings", "K('context_length')"),
                ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-6)"), ("rope_theta", "K('rope.freq_base', 10000.0)"), ("rope_scaling", "None"), ("vocab_size", "VOCAB"),
                ("attention_bias", "T('.attn_q_a.bias')"), ("tie_word_embeddings", "TIED")]),
    # LFM2-MoE (src/models/lfm2moe.cpp, conversion/lfm2.py LFM2MoeModel) -> the mlx-lm lfm2_moe class: short-conv blocks where head_count_kv is 0, attention (NEOX rope, q / k RMS norm) elsewhere;
    # leading dense blocks, then sigmoid router with a selection bias (exp_probs_b), top-k weights renormalised (lfm-base.cpp build_lfm_moe_ffn passes norm_w = true), weights scale
    # = expert_weights_scale (0 / absent = unscaled). The head is tied (embed_tokens.as_linear); head size = n_embd / n_head.
    "lfm2moe": dict(
        proven=False,       # no Metal identity proof yet: in force only with KYOJIN_KIT_UNPROVEN_ROWS (foreign_unproven)
        model_type="lfm2_moe", architectures=("Lfm2MoeForCausalLM",), trunk="int(K('block_count'))",
        let=[("rs", "K('rope.scaling.type')"), ("nt", "int(K('block_count'))"), ("nh", "int(K('attention.head_count'))"), ("hk", "K('attention.head_count_kv')"), ("att", "NZIDX(hk)"), ("nl", "int(K('leading_dense_block_count', 0) or 0)"),
             ("nu", "int(K('expert_used_count'))")],
        guards=[_NOROPE, _NOSCALE,
                ("len(hk) != nt or not att or sum(hk) != MAX(hk) * len(att)", "GGUF {arch}.attention.head_count_kv is not [0 or n] per block with a single n: the mlx-lm class has one kv head count"),
                ("not TIED", "GGUF {arch}: an output.weight head: the class always ties the head to the embeddings"),
                ("K('expert_gating_func') != 2", "GGUF {arch}.expert_gating_func: only the sigmoid router is mapped"),
                ("not T('.exp_probs_b.bias')", "GGUF {arch}: no exp_probs_b selection bias (the class would need use_expert_bias false: not checked against the graph)"),
                ("K('attention.key_length') not in (None, K('embedding_length') // nh)", "GGUF {arch}.attention.key_length is not n_embd / n_head: the class derives the head size that way"),
                ("K('attention.sliding_window') is not None or K('rope.dimension_count') not in (None, K('embedding_length') // nh)", "GGUF {arch}: sliding window / partial rope are not mapped"),
                ("K('expert_weights_scale') not in (None, 0, 0.0)", "GGUF {arch}.expert_weights_scale: the lfm2_moe class carries it as routed_scaling_factor, not verified against the graph"),
                ("T('.shortconv.conv.bias')", "GGUF {arch}: a conv bias is not mapped")],
        fields=[("vocab_size", "VOCAB"), ("hidden_size", "K('embedding_length')"), ("intermediate_size", "K('feed_forward_length')"), ("moe_intermediate_size", "K('expert_feed_forward_length')"),
                ("num_hidden_layers", "nt"), ("num_experts", "int(K('expert_count'))"), ("num_experts_per_tok", "nu"), ("norm_topk_prob", "True"), ("num_attention_heads", "nh"),
                ("num_key_value_heads", "MAX(hk)"), ("max_position_embeddings", "K('context_length')"), ("use_expert_bias", "True"), ("num_dense_layers", "nl"),
                ("norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-5)"), ("conv_bias", "False"), ("conv_L_cache", "int(K('shortconv.l_cache'))"), ("rope_theta", "K('rope.freq_base', 10000.0)"),
                ("routed_scaling_factor", "1.0"), ("full_attn_idxs", "att")]),
    # Cohere2-MoE (src/models/cohere2moe.cpp, conversion/command_r.py Cohere2MoeModel; North-Mini-Code) -> the mlx-vlm cohere2_moe text class (mlx-lm has none: KYOJIN_KIT_VLM_CLASSES).
    # Per-block sliding_window_pattern (True = sliding window, rope; False = global, no rope) -> layer_types; the leading dense blocks are global WITH rope in both (force_rope / prefix dense).
    # Norm type as llama.cpp: an RMS epsilon in the file = RMSNorm, else LayerNorm without bias. The head is tied (the class has no lm_head) and logits are scaled by logit_scale.
    # Routed experts only unless expert_shared_count is set (shared FFN = intermediate x count, averaged with the routed sum).
    "cohere2moe": dict(
        proven=False,       # no Metal identity proof yet, and the class is mlx-vlm's
        model_type="cohere2_moe", architectures=("Cohere2MoeForCausalLM",), trunk="int(K('block_count'))",
        let=[("rs", "K('rope.scaling.type')"), ("nt", "int(K('block_count'))"), ("sw", "K('attention.sliding_window_pattern')"), ("ns", "int(K('expert_shared_count', 0) or 0)")],
        guards=[_NOROPE,
                ("K('nextn_predict_layers', 0) or K('expert_weights_scale') not in (None, 0, 0.0) or K('rope.freq_base_swa') is not None", "GGUF {arch}: MTP blocks, expert_weights_scale or a separate sliding rope base are not mapped"),
                ("not TIED", "GGUF {arch}: an output.weight head: the class always ties the head to the embeddings"),
                ("K('expert_gating_func', 2) not in (0, 2)", "GGUF {arch}.expert_gating_func: only the sigmoid router is mapped"),
                ("sw is None or len(sw) != nt", "GGUF {arch}.attention.sliding_window_pattern is not one flag per block"),
                ("K('attention.key_length') != K('rope.dimension_count')", "GGUF {arch}: a partial rotary dimension is not mapped (the class rotates the whole head)"),
                ("T('.attn_q_norm.weight') or T('.attn_q.bias') or T('.ffn_exp_probs_b.bias') or T('.exp_probs_b.bias')", "GGUF {arch}: q / k norms, attention biases or a router bias are not mapped")],
        fields=[("vocab_size", "VOCAB"), ("hidden_size", "K('embedding_length')"), ("head_dim", "K('attention.key_length')"), ("num_hidden_layers", "nt"), ("intermediate_size", "K('expert_feed_forward_length')"),
                ("prefix_dense_intermediate_size", "K('feed_forward_length')"), ("num_attention_heads", "int(K('attention.head_count'))"), ("num_key_value_heads", "int(K('attention.head_count_kv'))"),
                ("rope_theta", "K('rope.freq_base', 10000.0)"), ("layer_norm_eps", "K('attention.layer_norm_epsilon', 1e-5)"), ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon')"),
                ("logit_scale", "K('logit_scale', 1.0)"), ("attention_bias", "False"), ("layer_norm_bias", "False"), ("sliding_window", "int(K('attention.sliding_window'))"),
                ("layer_types", "LAYER_TYPES(sw)"), ("num_experts", "int(K('expert_count'))"), ("num_experts_per_tok", "int(K('expert_used_count'))"), ("norm_topk_prob", "bool(K('expert_weights_norm', False))"),
                ("moe_num_shared_experts", "ns"), ("moe_gate_act", "'sigmoid'"), ("first_k_dense_replace", "int(K('leading_dense_block_count', 0) or 0)"), ("prefix_dense_sliding_window_pattern", "1")]),
    # NemotronH-MoE (src/models/nemotron-h.cpp, conversion/nemotron.py NemotronHModel; Nemotron-3.5-Lightning) -> the mlx-lm nemotron_h class (model_type alias, foreign_unproven). Blocks are
    # Mamba-2 (no kv heads, no ffn length), attention (kv heads > 0, no rope in the class) or routed MoE (ffn length > 0); the pattern string comes from the two per-block arrays. One shared
    # expert, sigmoid router with a selection bias, renormalised and scaled, non-gated relu^2 experts (fc1 / fc2). The trailing nextn block (MTP) is past the trunk and not loaded.
    "nemotron_h_moe": dict(
        proven=False,       # no Metal identity proof yet
        model_type="nemotron_h", architectures=("NemotronHForCausalLM",), trunk=_TRUNK,
        let=[("nt", "int(K('block_count')) - int(K('nextn_predict_layers', 0))"), ("hk", "K('attention.head_count_kv')"), ("ff", "K('feed_forward_length')")],
        guards=[("len(hk) != int(K('block_count')) or len(ff) != int(K('block_count'))", "GGUF {arch}: head_count_kv / feed_forward_length are not one value per block"),
                ("K('moe_latent_size') is not None", "GGUF {arch}: a latent projection around the experts is not mapped"),
                ("K('expert_gating_func') not in (None, 0, 2)", "GGUF {arch}.expert_gating_func: only the sigmoid router is mapped"),
                ("T('.ssm_in.bias') or T('.attn_q.bias') or T('.ssm_out.bias') or T('.ffn_up_shexp.bias')", "GGUF {arch}: projection biases are not mapped"),
                ("K('ssm.inner_size') % K('ssm.time_step_rank') != 0", "GGUF {arch}: ssm.inner_size is not a multiple of the head count")],
        fields=[("vocab_size", "VOCAB"), ("hidden_size", "K('embedding_length')"), ("intermediate_size", "MAX(ff)"), ("max_position_embeddings", "K('context_length')"),
                ("num_attention_heads", "int(K('attention.head_count'))"), ("num_key_value_heads", "MAX(hk)"), ("attention_bias", "False"), ("mamba_num_heads", "int(K('ssm.time_step_rank'))"),
                ("mamba_head_dim", "int(K('ssm.inner_size')) // int(K('ssm.time_step_rank'))"), ("mamba_proj_bias", "False"), ("ssm_state_size", "int(K('ssm.state_size'))"),
                ("conv_kernel", "int(K('ssm.conv_kernel'))"), ("n_groups", "int(K('ssm.group_count'))"), ("mlp_bias", "False"), ("layer_norm_epsilon", "K('attention.layer_norm_rms_epsilon', 1e-5)"),
                ("use_bias", "False"), ("use_conv_bias", "T('.ssm_conv1d.bias')"), ("hybrid_override_pattern", "HYBRID(hk, ff, nt)"), ("head_dim", "K('attention.key_length')"),
                ("moe_intermediate_size", "K('expert_feed_forward_length')"), ("moe_shared_expert_intermediate_size", "K('expert_shared_feed_forward_length')"),
                ("n_routed_experts", "int(K('expert_count'))"), ("n_shared_experts", "int(K('expert_shared_count', 0)) or None"), ("n_group", "int(K('expert_group_count', 1))"),
                ("topk_group", "int(K('expert_group_used_count', 1))"), ("num_experts_per_tok", "int(K('expert_used_count'))"), ("norm_topk_prob", "bool(K('expert_weights_norm', False))"),
                ("routed_scaling_factor", "float(K('expert_weights_scale', 1.0))")]),
    # flint (src/models/flint.cpp, conversion/flint.py FlintModel(Qwen2MoeModel); RecursiveSI/flint) -> the mlx-vlm flint text class (mlx-lm has none: KYOJIN_KIT_VLM_CLASSES).
    # q / k RMS norms, per-block sliding window (True = sliding) and rope (False = no rope) patterns, learned attention sinks per head (`attn_sinks` = the class's softmax_offset, passed to
    # mx.fast.scaled_dot_product_attention as `sinks`), leading dense blocks, then sigmoid-routed experts (selection bias optional, weights always renormalised, scale 1.0) + shared expert(s).
    "flint": dict(
        proven=False,       # no Metal identity proof yet, and the class is mlx-vlm's
        model_type="flint", architectures=("FlintForCausalLM",), trunk="int(K('block_count'))",
        let=[("nt", "int(K('block_count'))"), ("sw", "K('attention.sliding_window_pattern')"), ("rp", "K('attention.rope_pattern')"), ("nd", "int(K('leading_dense_block_count', 0) or 0)")],
        guards=[("sw is None or rp is None or len(sw) != nt or len(rp) != nt", "GGUF {arch}: sliding_window_pattern / rope_pattern are not one flag per block"),
                ("not K('expert_weights_norm', False) or K('expert_weights_scale', 1.0) not in (None, 1.0)", "GGUF {arch}: the class always renormalises the routed weights and does not scale them"),
                ("K('expert_gating_func') not in (None, 0, 2)", "GGUF {arch}.expert_gating_func: only the sigmoid router is mapped"),
                ("K('rope.scaling.type') not in (None, 'none')", "GGUF {arch}.rope.scaling.type is not mapped (the config built here has no rope_scaling)"),
                ("K('attention.key_length') != K('attention.value_length') or K('nextn_predict_layers', 0)", "GGUF {arch}: unequal key / value head size or MTP blocks are not mapped"),
                ("T('.attn_q.bias') or T('.ffn_up_shexp.bias')", "GGUF {arch}: projection biases are not mapped"),
                ("K('expert_shared_feed_forward_length') != K('expert_feed_forward_length') * K('expert_shared_count', 0)", "GGUF {arch}: shared expert width is not shared_count x expert width")],
        fields=[("vocab_size", "VOCAB"), ("hidden_size", "K('embedding_length')"), ("intermediate_size", "K('feed_forward_length')"), ("num_hidden_layers", "nt"),
                ("num_attention_heads", "int(K('attention.head_count'))"), ("num_key_value_heads", "int(K('attention.head_count_kv'))"), ("head_dim", "K('attention.key_length')"),
                ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-6)"), ("max_position_embeddings", "K('context_length')"), ("rope_theta", "K('rope.freq_base', 10000.0)"),
                ("layer_types", "LAYER_TYPES(sw)"), ("no_rope_layer_indices", "FALSEIDX(rp)"), ("use_sliding_window", "True"), ("sliding_window", "int(K('attention.sliding_window'))"),
                ("softmax_type", "'learnable'"), ("attention_bias", "False"), ("hidden_act", "'silu'"), ("mlp_only_layers", "RANGE(nd)"), ("decoder_sparse_step", "1"),
                ("num_experts", "int(K('expert_count'))"), ("num_experts_per_tok", "int(K('expert_used_count'))"), ("moe_intermediate_size", "K('expert_feed_forward_length')"),
                ("shared_expert_intermediate_size", "K('expert_shared_feed_forward_length')"), ("router_score_function", "'sigmoid'"), ("norm_topk_prob", "True"), ("tie_word_embeddings", "TIED")]),
    # gemma4 (src/models/gemma4.cpp, conversion/gemma.py Gemma4Model) -> the mlx-lm gemma4_text class (dense, per-layer-embedding E2B / E4B, and the 26B-A4B MoE).
    # Per-layer lists of the file (head_count_kv, feed_forward_length, sliding_window_pattern) are folded into the class's scalar fields; a guard refuses what the class cannot express.
    # rope_freqs.weight = the converter's [1.0] * n_rot + [1e30] * rest for the full-attention layers: the class builds that rope itself from partial_rotary_factor, so it is read here, not loaded.
    "gemma4": dict(
        proven=False,       # no Metal identity proof yet: in force only with KYOJIN_KIT_UNPROVEN_ROWS or KYOJIN_GEMMA4_ROWS (foreign_unproven)
        model_type="gemma4_text", architectures=("Gemma4ForCausalLM",), trunk="int(K('block_count'))",
        let=[("nt", "int(K('block_count'))"), ("pat", "list(K('attention.sliding_window_pattern'))"), ("gi", "IDX(pat, False)"), ("si", "IDX(pat, True)"), ("sp", "gi + 1"),
             ("ff", "K('feed_forward_length')"), ("ns", "int(K('attention.shared_kv_layers', 0) or 0)"), ("hk", "K('attention.head_count_kv')"),
             ("hd", "int(K('attention.key_length_swa'))"), ("ghd", "int(K('attention.key_length'))"), ("rot", "ROTATED('rope_freqs.weight')"),
             ("grot", "2 * rot if rot is not None else int(K('rope.dimension_count'))"), ("srot", "int(K('rope.dimension_count_swa'))"),
             ("ne", "int(K('expert_count', 0) or 0)"), ("wide", "AT(ff, nt - 1) != AT(ff, 0)"), ("kv_eq_v", "not T('blk.' + str(gi) + '.attn_v.weight')")],
        guards=[("len(pat) != nt or pat != (([True] * (sp - 1) + [False]) * (nt // sp + 1))[:nt]", "GGUF {arch}.attention.sliding_window_pattern is not the regular {sp}-block pattern the mlx-lm class derives"),
                ("K('attention.value_length') != ghd or K('attention.value_length_swa') != hd", "GGUF {arch}: value_length != key_length"),
                ("wide and (ns == 0 or AT(ff, nt - 1) != 2 * AT(ff, 0) or ff != [AT(ff, 0)] * (nt - ns) + [AT(ff, nt - 1)] * ns)", "GGUF {arch}.feed_forward_length is not [n] * (blocks - shared) + [2n] * shared"),
                ("AT(hk, 0) != AT(hk, si)", "GGUF {arch}.attention.head_count_kv of block 0 is not that of the first sliding block"),
                ("AT(hk, gi) != AT(hk, si) and not kv_eq_v", "GGUF {arch}: global layers have another kv head count and a V projection: the class only has that with K = V"),
                ("rot is not None and ghd != int(K('rope.dimension_count'))", "GGUF {arch}: rope_freqs.weight with rope.dimension_count != key_length"),
                ("srot != hd", "GGUF {arch}.rope.dimension_count_swa is not the sliding head size: partial rope on sliding layers is not mapped"),
                ("K('rope.scaling.type') not in (None, 'none')", "GGUF {arch}.rope.scaling.type is not mapped"),
                ("ne and not T('.ffn_gate_inp.scale')", "GGUF {arch}: MoE without the router scale tensor")],
        fields=[("hidden_size", "K('embedding_length')"), ("num_hidden_layers", "nt"), ("intermediate_size", "AT(ff, 0)"), ("num_attention_heads", "K('attention.head_count')"),
                ("head_dim", "hd"), ("global_head_dim", "ghd"), ("num_key_value_heads", "AT(hk, si)"), ("num_global_key_value_heads", "AT(hk, gi)"),
                ("global_partial_rotary_factor", "grot / ghd"), ("rms_norm_eps", "K('attention.layer_norm_rms_epsilon', 1e-6)"), ("vocab_size", "VOCAB"),
                ("vocab_size_per_layer_input", "NE('per_layer_token_embd.weight', 1) or VOCAB"), ("num_kv_shared_layers", "ns"),
                ("hidden_size_per_layer_input", "int(K('embedding_length_per_layer_input', 0) or 0)"), ("sliding_window", "K('attention.sliding_window')"),
                ("sliding_window_pattern", "sp"), ("max_position_embeddings", "K('context_length')"), ("attention_k_eq_v", "kv_eq_v"),
                ("final_logit_softcapping", "K('final_logit_softcapping')"), ("use_double_wide_mlp", "wide"), ("enable_moe_block", "ne > 0"), ("num_experts", "ne or None"),
                ("top_k_experts", "int(K('expert_used_count')) if ne else None"), ("moe_intermediate_size", "K('expert_feed_forward_length') if ne else None"),
                ("tie_word_embeddings", "TIED"),
                ("rope_parameters", "{'full_attention': {'partial_rotary_factor': grot / ghd, 'rope_theta': K('rope.freq_base'), 'rope_type': 'proportional'}, 'sliding_attention': {'partial_rotary_factor': srot / hd, 'rope_theta': K('rope.freq_base_swa'), 'rope_type': 'default'}}")]),
}


def gated_off(arch: str) -> bool:
    """A row marked proven=False is in force only while KYOJIN_KIT_UNPROVEN_ROWS is on (foreign_unproven; gemma4 also by its old KYOJIN_GEMMA4_ROWS)."""
    from . import foreign_unproven as UP
    return UP.gated_off(arch)


def has_rows(arch: str) -> bool:
    return arch in ROWS and not gated_off(arch)


def _chain(arch):
    r = ROWS[arch]
    return (_chain(r["extends"]) if r.get("extends") else []) + [r]


def build(arch: str, g) -> dict:
    """HF-style config (model_type, architectures + fields) of a GGUF (anything with .kv and .tensors) from the rows. Raises GgufError where the hand builder refuses."""
    if gated_off(arch):
        raise GF.GgufError(f"GGUF {arch}: the config rows are unproven and off (KYOJIN_KIT_UNPROVEN_ROWS=1 turns them on)")
    chain = _chain(arch)
    e = _Env(arch, g.kv, g.tensors, g)
    fields = {}
    try:
        for r in chain:
            for name, expr in r.get("let", ()):
                e.names[name] = ev(expr, e)
            for cond, msg in r.get("guards", ()):
                if ev(cond, e):
                    raise GF.GgufError(msg.format(arch=arch, **e.names))
            for name, expr in r.get("fields", ()):
                fields[name] = ev(expr, e)
    except (TypeError, KeyError) as ex:                              # int(None) and the like: a key the file must carry is missing
        raise GF.GgufError(f"GGUF {arch}: a metadata key the config row needs is missing or malformed ({ex})") from ex
    top = chain[-1]
    return dict(model_type=top["model_type"], architectures=list(top["architectures"]), **fields)


def n_trunk(arch: str, kv: dict) -> int:
    """Trunk blocks of a file (the blocks past it are the draft head, skipped as llama.cpp does)."""
    return int(ev(ROWS[arch]["trunk"] if "trunk" in ROWS[arch] else _chain(arch)[0]["trunk"], _Env(arch, kv, {})))
