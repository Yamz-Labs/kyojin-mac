# SPDX-License-Identifier: Apache-2.0
"""The converter value transforms undone at load (foreign_xform.py), as DATA: named operations with expression arguments (no Python objects, no numpy).
foreign_xform stays the reference of the Mac loader; tools/archkit_export.py writes these rows into the neutral JSON and holds a numpy reference interpreter,
tests/test_arch_kit_export.py proves that interpreter equal to foreign_xform on every tensor name of the sets.

A set is {"extends": set, "let": [(name, expr)], "rules": [rule]}. A rule is {"match": regex (full match on the GGUF tensor name), "ops": [op]}; the FIRST rule whose
regex matches decides (a rule whose ops are all disabled = the tensor is untouched). An op may carry "when": expr. `let` names and K('<key>') (metadata `<arch>.<key>`)
are the expression language of foreign_cfgrows; a `let` that cannot be evaluated (a key the file lacks) is left undefined and only fails if a rule uses it.
Operations, applied in this order to the decoded tensor (the stored blocks of a K-quant file may be permuted along ROWS before decoding: rows are independent in every GGUF type):
  cols_inverse_gather {index}   W <- W[:, argsort(index)]      (the converter permuted the INPUT columns: the activation is gathered instead in a stored-kernel loader)
  gather_rows {index}           W <- W[index]                  (also for a 1-D tensor)
  post {fn}                     elementwise / reshape on the float32 values: minus1 (w - 1: the converter stored w + 1), a_log (log(-w); refuses a value >= 0),
                                conv_unsqueeze (reshape [C, K] -> [C, 1, K], cast fp16), conv_unsqueeze_f32 (the same reshape, values kept), row (1-D [n] -> [1, n])
Index kinds ({"kind": ..., args as expressions}):
  v_grouped {nk, nv, hd, skip=0}   skip leading rows kept in place, then the V heads from the converter's tiled order back to grouped:
                                   idx = argsort(arange(nv*hd).reshape(nk, nv // nk, hd).transpose(1, 0, 2).reshape(-1))
  v_tiled   {nk, nv, hd}           idx = arange(nv*hd).reshape(nk, nv // nk, hd).transpose(1, 0, 2).reshape(-1)
  rope_unpermute {heads}           the converter's LlamaModel.permute: p = arange(rows).reshape(heads, 2, rows // heads // 2).transpose(0, 2, 1).reshape(-1) (new = old[p]);
                                   idx = argsort(p), rows = size of the tensor's first axis"""
from __future__ import annotations

_NORM = r"blk\.\d+\.(?!ssm_norm\.)[a-z0-9_.]*norm\.(?:weight|bias)"


def _vg(hd, skip="0"):
    return {"kind": "v_grouped", "nk": "nk", "nv": "nv", "hd": hd, "skip": skip}


_PRE = "2 * ds * nk"
_V = {"let": [("nk", "int(K('ssm.group_count'))"), ("nv", "int(K('ssm.time_step_rank'))"), ("ds", "int(K('ssm.state_size'))"), ("inner", "int(K('ssm.inner_size'))"),
              ("vd", "inner // nv"), ("tiled", "nk != nv")]}                       # llama.cpp reorders the V heads only when the key and value head counts differ

SETS = {
    "qwen35": dict(**_V, rules=[
        dict(match=r"output_norm\.weight", ops=[dict(op="post", fn="minus1")]),
        dict(match=_NORM, ops=[dict(op="post", fn="minus1")]),
        dict(match=r"blk\.\d+\.ssm_a", ops=[dict(op="gather_rows", index=_vg("1"), when="tiled"), dict(op="post", fn="a_log")]),
        dict(match=r"blk\.\d+\.ssm_dt\.bias", ops=[dict(op="gather_rows", index=_vg("1"), when="tiled")]),
        dict(match=r"blk\.\d+\.ssm_conv1d\.(?:weight|bias)", ops=[dict(op="gather_rows", index=_vg("vd", _PRE), when="tiled"), dict(op="post", fn="conv_unsqueeze")]),
        dict(match=r"blk\.\d+\.attn_qkv\.(?:weight|bias)", ops=[dict(op="gather_rows", index=_vg("vd", _PRE), when="tiled")]),
        dict(match=r"blk\.\d+\.attn_gate\.(?:weight|bias)", ops=[dict(op="gather_rows", index=_vg("vd"), when="tiled")]),
        dict(match=r"blk\.\d+\.ssm_(?:alpha|beta)\.(?:weight|bias)", ops=[dict(op="gather_rows", index=_vg("1"), when="tiled")]),
        dict(match=r"blk\.\d+\.ssm_out\.(?:weight|bias)", ops=[dict(op="cols_inverse_gather", index={"kind": "v_tiled", "nk": "nk", "nv": "nv", "hd": "vd"}, when="tiled")]),
    ]),
    # the same trunk under `qwen35moe.` keys (K is relative to the architecture) + the shared-expert gate stored as a vector
    "qwen35moe": dict(extends="qwen35", rules=[dict(match=r".*ffn_gate_inp_shexp\.weight", ops=[dict(op="post", fn="row")])]),
    "llama": dict(rules=[
        dict(match=r"blk\.\d+\.attn_q\.(?:weight|bias)", ops=[dict(op="gather_rows", index={"kind": "rope_unpermute", "heads": "int(K('attention.head_count'))"})]),
        dict(match=r"blk\.\d+\.attn_k\.(?:weight|bias)", ops=[dict(op="gather_rows", index={"kind": "rope_unpermute", "heads": "int(K('attention.head_count_kv', K('attention.head_count')))"})]),
    ]),
    "gemma3": dict(rules=[dict(match=r".*norm\.weight", ops=[dict(op="post", fn="minus1")])]),
    "qwen2moe": dict(rules=[dict(match=r".*ffn_gate_inp_shexp\.weight", ops=[dict(op="post", fn="row")])]),
    "nemotron_h_moe": dict(rules=[dict(match=r"blk\.\d+\.ssm_a", ops=[dict(op="post", fn="a_log_flat")]), dict(match=r"blk\.\d+\.ssm_d|blk\.\d+\.ssm_norm\.weight", ops=[dict(op="post", fn="flat")]),
                                      dict(match=r"blk\.\d+\.ssm_conv1d\.weight", ops=[dict(op="post", fn="conv_unsqueeze_f32")])]),
    "lfm2moe": dict(rules=[dict(match=r"blk\.\d+\.shortconv\.conv\.weight", ops=[dict(op="post", fn="conv_unsqueeze_f32")])]),
}
# foreign_xform._SETS name -> the set of rows ("llama" rows read `<arch>.attention.head_count`, so mistral3 / granitemoe share them)
ALIASES = {"mistral3": "llama", "granitemoe": "llama", "gemma2": "gemma3"}


def rules(name: str) -> tuple:
    """(lets, rules) of a set with `extends` resolved (parent rules first)."""
    name = ALIASES.get(name, name)
    s = SETS[name]
    if s.get("extends"):
        pl, pr = rules(s["extends"])
        return pl + list(s.get("let", ())), pr + list(s["rules"])
    return list(s.get("let", ())), list(s["rules"])
