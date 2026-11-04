# SPDX-License-Identifier: Apache-2.0
"""What the llama.cpp converter did to a tensor besides renaming it, and the way back (numpy only). One entry per architecture that transforms.

qwen35 (conversion/qwen.py at foreign_names.LLAMA_CPP_COMMIT):
  Qwen3NextModel.modify_tensors      A_log -> -exp(A_log) (`ssm_a`), dt_bias renamed (`ssm_dt.bias`), conv1d squeezed to [C, K],
                                     every `*norm.weight` except linear_attn.norm gets +1 (HF RMSNorm computes x * (1 + w))
  _LinearAttentionVReorderBase       when linear_num_key_heads != linear_num_value_heads the V heads go from grouped [K0v0 K0v1 K0v2 K1v0 ..] to tiled
                                     [K0v0 K1v0 .. K0v1 ..] (`_reorder_v_heads`: reshape (nk, r, hd) -> swap axes 0 and 1): rows of in_proj_qkv (V part
                                     only), in_proj_z, in_proj_a / in_proj_b (hd = 1), A_log, dt_bias, the V channels of conv1d, the COLUMNS of out_proj
  _QwenMtpMixin.filter_tensors       runs in index_tensors, so modify_tensors sees the draft block under its block-N names and adds the +1 to
                                     enorm / hnorm / shared_head.norm and to the block's own norms as well
The row permutations are applied to the stored blocks (rows are independent in every GGUF type). The column permutation of out_proj cannot be: its
128-column groups cross the 256-element blocks of the K-quants. It is undone on the activation instead (`cols`): W_hf x_hf = W_gguf x_hf[idx].
Nothing is re-quantised and no weight is converted."""
from __future__ import annotations

import numpy as np

from . import foreign_gguf as GF


class Xf:
    """rows: HF row r = stored row rows[r] (None: untouched). cols: input activation gather, x_stored = x_hf[..., cols] (None: untouched).
    post: name of the elementwise fix applied to the decoded float32 values (POST)."""
    __slots__ = ("rows", "cols", "post")

    def __init__(self, rows=None, cols=None, post=None):
        self.rows, self.cols, self.post = rows, cols, post


def v_tiled_index(nk: int, nv: int, hd: int) -> np.ndarray:
    """fwd[j] = grouped position of tiled position j: `_reorder_v_heads` is new = old[fwd]."""
    return np.arange(nv * hd).reshape(nk, nv // nk, hd).transpose(1, 0, 2).reshape(-1)


def v_grouped_index(nk: int, nv: int, hd: int) -> np.ndarray:
    """The inverse: old = new[inv]. Applying `_reorder_v_heads` again does NOT undo it (it swaps (nk, r) the same way, not (r, nk))."""
    return np.argsort(v_tiled_index(nk, nv, hd), kind="stable")


def _a_log(a):
    if not np.all(a < 0):
        raise GF.GgufError("ssm_a holds non-negative values: not a -exp(A_log) tensor")
    return np.log(-a.astype(np.float32))


POST = {
    "minus1": lambda w: w.astype(np.float32) - 1.0,       # norm weight: the converter stored w + 1
    "a_log": _a_log,                                      # ssm_a = -exp(A_log)
    "conv": lambda w: w.reshape(w.shape[0], 1, w.shape[1]).astype(np.float16),   # [C, K] -> HF [C, 1, K] fp16 (the safetensors path casts bf16 to fp16): mlx-lm sanitize moves it to its own layout and adds the norm +1
}


def _qwen35(name: str, kv: dict, p: str = "qwen35.") -> Xf | None:
    if name == "output_norm.weight":
        return Xf(post="minus1")                              # model.norm.weight: same +1
    if not name.startswith("blk."):
        return None
    rest = name.split(".", 2)[2]
    nk, nv, ds, inner = int(kv[p + "ssm.group_count"]), int(kv[p + "ssm.time_step_rank"]), int(kv[p + "ssm.state_size"]), int(kv[p + "ssm.inner_size"])
    vd = inner // nv
    tiled = nk != nv                                         # llama.cpp reorders only then (modify_tensors condition)
    kind = rest.rpartition(".")[0] if rest.endswith((".weight", ".bias")) else rest
    if kind.endswith("norm") and kind != "ssm_norm":
        return Xf(post="minus1")                              # attn_norm, post_attention_norm, attn_q_norm, attn_k_norm, nextn.enorm / hnorm / shared_head_norm
    if kind == "ssm_a":
        return Xf(rows=v_grouped_index(nk, nv, 1) if tiled else None, post="a_log")
    if kind == "ssm_dt" and rest.endswith(".bias"):
        return Xf(rows=v_grouped_index(nk, nv, 1) if tiled else None)
    if kind == "ssm_conv1d":
        r = np.concatenate([np.arange(2 * ds * nk), 2 * ds * nk + v_grouped_index(nk, nv, vd)]) if tiled else None
        return Xf(rows=r, post="conv")
    if not tiled:
        return None
    if kind == "attn_qkv":
        return Xf(rows=np.concatenate([np.arange(2 * ds * nk), 2 * ds * nk + v_grouped_index(nk, nv, vd)]))
    if kind == "attn_gate":
        return Xf(rows=v_grouped_index(nk, nv, vd))
    if kind in ("ssm_alpha", "ssm_beta"):
        return Xf(rows=v_grouped_index(nk, nv, 1))
    if kind == "ssm_out":
        return Xf(cols=v_tiled_index(nk, nv, vd))
    return None


def rope_pairs_index(rows: int, h: int) -> np.ndarray:
    """LlamaModel.permute (conversion/llama.py): w.reshape(h, 2, rows // h // 2, ..).swapaxes(1, 2): new = old[this]."""
    return np.arange(rows).reshape(h, 2, rows // h // 2).transpose(0, 2, 1).reshape(-1)


def _llama(name: str, kv: dict, arch: str = "llama") -> Xf | None:
    """LlamaModel.modify_tensors with undo_permute (llama.py:261-265): q_proj with n_head heads, k_proj with n_kv_head (permute() keeps the kv count when the two differ),
    weight AND bias. The rows of a stored tensor are `new`; the HF rows are the inverse."""
    if not name.startswith("blk.") or not name.split(".", 2)[2].startswith(("attn_q.", "attn_k.")):
        return None
    nh = int(kv[f"{arch}.attention.head_count"])
    h = nh if name.split(".", 2)[2].startswith("attn_q.") else int(kv.get(f"{arch}.attention.head_count_kv", nh))
    # rows = h * head_dim, known only from the tensor: the loader passes it through `rows_of` below
    return Xf(rows=("rope_pairs", h))


def _gemma3(name: str, kv: dict) -> Xf | None:
    """Gemma3Model.norm_shift (conversion/gemma.py:132, applied at :178-180): every `*norm.weight` is stored +1. (It also truncates embed_tokens to the real vocabulary;
    the config takes its vocab_size from the file's table.)"""
    return Xf(post="minus1") if name.endswith("norm.weight") else None


def _qwen2moe(name: str, kv: dict) -> Xf | None:
    """The shared-expert gate is [1, n_embd] in HF; a producer that stores it as a vector ([n_embd]) is read back as one row."""
    return Xf(post="row") if name.endswith("ffn_gate_inp_shexp.weight") else None


POST["row"] = lambda w: w.reshape(1, -1) if w.ndim == 1 else w


def _qwen35moe(name: str, kv: dict) -> Xf | None:
    """qwen35moe (src/models/qwen35moe.cpp) has the qwen35 trunk (same converter class chain, conversion/qwen.py Qwen3_5MoeTextModel) under the `qwen35moe.` keys,
    plus the shared-expert gate stored as a vector."""
    x = _qwen35(name, kv, "qwen35moe.")
    if x is None and name.endswith("ffn_gate_inp_shexp.weight"):
        return Xf(post="row")
    return x


_SETS = {"qwen35": _qwen35, "qwen35moe": _qwen35moe, "llama": _llama, "mistral3": lambda name, kv: _llama(name, kv, "mistral3"), "gemma3": _gemma3, "qwen2moe": _qwen2moe}


def plan(name: str, xform: str, kv: dict, rows: int | None = None) -> Xf | None:
    """Transform of GGUF tensor `name`, or None. rows: the tensor's output rows (needed by transforms that depend on them: llama's q / k permute)."""
    x = _SETS[xform](name, kv) if xform else None
    if x is not None and isinstance(x.rows, tuple):             # ("rope_pairs", heads): resolved with the tensor's row count
        x.rows = np.argsort(rope_pairs_index(rows, x.rows[1]), kind="stable")
    return x
