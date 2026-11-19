# SPDX-License-Identifier: Apache-2.0
"""Architecture import kit, run-time side (numpy / mlx-free): the tables generated from llama.cpp (foreign_arch_gen.py, tools/gen_arch_tables.py)
turned into (a) a GGUF tensor name -> parameter name map of a given model class and (b) the config of that class from the GGUF metadata.

Nothing here names a model. Both directions use the same trick: llama.cpp's converter lists every Hugging Face name / config key it maps onto a GGUF
tensor / key; the model class (an mlx-lm module) says which of them it really has. A tensor or key with no taker is reported, never dropped:
`resolve` raises on a required tensor without a parameter, `kit_config` raises on a file key it cannot place and on a converter transform
(permute, split, norm shift ...) nobody has vetted for the architecture."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import foreign_arch_gen as G


class KitError(ValueError):
    pass


# converter transforms that change tensor VALUES (or split / merge tensors): a file of such an architecture is only served through a vetted hand transform
_VALUE_FLAGS = ("permute", "cat", "chunk", "split", "stack", "transpose", "reshape", "view", "flip", "repeat", "plus1", "rename", "own_yield", "rope_freqs", "squeeze", "unsqueeze", "arith", "slice", "torch_op", "norm_shift")
_BLK = re.compile(r"blk\.(\d+)\.(.+)")


def have(arch: str) -> bool:
    return arch in G.ARCH_TENSORS


def candidates(arch: str, key: str) -> tuple:
    """Hugging Face name templates ("model.layers.{bid}.self_attn.q_proj") the converter maps onto tensor `key` of `arch`: global list, then the block list
    (replaced by the architecture's own list when it has one, as TensorNameMap.__init__ does)."""
    return tuple(G.HF_GLOBAL.get(key, ())) + tuple(G.HF_OVER.get(arch, {}).get(key, G.HF_BLOCK.get(key, ())))


def _kind(template: str) -> str:
    return template.replace("blk.{bid}.", "", 1)


def tensor_status(arch: str) -> dict:
    """tensor key -> 'required' | 'optional' | 'skip' | 'listed' (in gguf-py MODEL_TENSORS but not named by the C++ loader: qkv / expert helpers, or not read)."""
    ld = G.LOADER.get(arch, {}).get("tensors", {})
    return {k: ld.get(k, "listed") for k in G.ARCH_TENSORS[arch]}


def converter_flags(arch: str) -> tuple:
    return tuple(sorted({f for c in G.CONVERTER.get(arch, ()) for f in c["flags"]}))


def value_transforms(arch: str) -> tuple:
    return tuple(f for f in converter_flags(arch) if f in _VALUE_FLAGS)


# Hand-vetted rows (the only per-architecture hand-written lines of the kit): arch -> dict(flags=converter value transforms it may carry, xform=foreign_xform set that undoes
# them, keys=file keys the model class cannot use (the caveat is in the comment), cfg={config field: expression} for values llama.cpp's C++ loader derives that the file does not carry
# (the expression language of foreign_cfgrows, K('key') = `<arch>.key`: data, not a Python function, so the table can be exported).
VETTED = {
    # conversion/gemma.py Gemma2Model.modify_tensors: `*norm.weight` + 1. mlx-lm gemma2 has no sliding window: exact up to the window (4096 tokens), UNVERIFIED beyond.
    # src/models/gemma2.cpp: attention scale 1/sqrt(n_embd_head_k), the 27B (46 blocks) 1/sqrt(n_embd / n_head); mlx-lm takes query_pre_attn_scalar ** -0.5.
    "gemma2": dict(flags=("arith", "plus1"), xform="gemma2", keys=("gemma2.attention.sliding_window",),
                   cfg={"query_pre_attn_scalar": "K('embedding_length') // K('attention.head_count') if K('block_count') == 46 else K('attention.key_length')"}),
    # The four architectures of the tiny-config round trip (tests/test_arch_kit_roundtrip.py): the hand-written path already undoes each converter transform (foreign_xform) and its tests
    # replay the converter forward; the kit rows only name that same inverse. qwen35 / qwen35moe: conversion/qwen.py (A_log, dt_bias, conv1d, `*norm.weight` + 1, v-head reorder).
    "qwen35": dict(flags=("arith", "cat", "permute", "plus1", "slice", "split", "squeeze", "stack", "torch_op", "unsqueeze", "view"), xform="qwen35",
                   exact={"blk.{bid}.ssm_a": "model.layers.{bid}.linear_attn.A_log", "blk.{bid}.ssm_dt.bias": "model.layers.{bid}.linear_attn.dt_bias"}),
    "qwen35moe": dict(flags=("arith", "cat", "permute", "plus1", "slice", "split", "squeeze", "stack", "torch_op", "unsqueeze", "view"), xform="qwen35moe",
                      exact={"blk.{bid}.ssm_a": "model.layers.{bid}.linear_attn.A_log", "blk.{bid}.ssm_dt.bias": "model.layers.{bid}.linear_attn.dt_bias"}),
    # conversion/lfm2.py LFM2MoeModel: conv.conv weight squeezed (xform unsq1), experts stacked per layer (the file holds the stacked tensor = the class's switch_mlp after sanitize),
    # `.expert_bias` renamed `.expert_bias.bias` by filter_tensors (GGUF blk.N.exp_probs_b.bias = the class's `expert_bias` array, which has no .weight / .bias form)
    "lfm2moe": dict(proven=False, flags=("squeeze", "stack"), xform="lfm2moe", exact={"blk.{bid}.exp_probs_b.bias": "model.layers.{bid}.feed_forward.expert_bias"}),
    # conversion/command_r.py Cohere2MoeModel.modify_tensors: routed experts stacked per layer (= the class's switch_mlp after its sanitize); a zero bias tensor is dropped (`torch.any(bias != 0)`
    # raises otherwise: no value change). The text class is mlx-vlm's cohere2_moe (KYOJIN_KIT_VLM_CLASSES), so the line is unproven like the class lookup.
    "cohere2moe": dict(proven=False, flags=("stack", "torch_op")),
    # conversion/nemotron.py NemotronHModel.modify_tensors (MoE branch): A_log -> -exp(A_log) reshaped [heads, 1], D reshaped [heads, 1], mamba conv weight squeezed, gated-norm weight
    # reshaped [n_group, -1] (foreign_xform "nemotron_h_moe" undoes each); dt_bias -> `ssm_dt.bias` and e_score_correction.bias -> `exp_probs_b.bias` are renames; routed experts stacked per layer.
    "nemotron_h_moe": dict(proven=False, flags=("arith", "permute", "rename", "reshape", "rope_freqs", "split", "squeeze", "stack", "torch_op"), xform="nemotron_h_moe",
                       exact={"blk.{bid}.ssm_a": "backbone.layers.{bid}.mixer.A_log", "blk.{bid}.ssm_d": "backbone.layers.{bid}.mixer.D", "blk.{bid}.ssm_dt.bias": "backbone.layers.{bid}.mixer.dt_bias",
                              "blk.{bid}.exp_probs_b.bias": "backbone.layers.{bid}.mixer.gate.e_score_correction_bias",
                              "blk.{bid}.ffn_up_exps.weight": "backbone.layers.{bid}.mixer.switch_mlp.fc1.weight", "blk.{bid}.ffn_down_exps.weight": "backbone.layers.{bid}.mixer.switch_mlp.fc2.weight"}),
    # conversion/flint.py FlintModel(Qwen2MoeModel): `.softmax_offset` -> `attn_sinks.weight` (rename), `.gate.expert_bias` -> `exp_probs_b.bias` only when router_expert_bias (the class then
    # fills a zero `expert_bias` itself), routed experts stacked per layer (conversion/qwen.py Qwen2MoeModel), no value change. Class: mlx-vlm flint (KYOJIN_KIT_VLM_CLASSES).
    "flint": dict(proven=False, flags=("arith", "slice", "stack"), names={"ATTN_SINKS": "model.layers.{bid}.self_attn.softmax_offset"},
                  exact={"blk.{bid}.attn_sinks.weight": "model.layers.{bid}.self_attn.softmax_offset", "blk.{bid}.exp_probs_b.bias": "model.layers.{bid}.mlp.expert_bias"}),
    "glm4moe": dict(flags=("stack",), exact={"blk.{bid}.exp_probs_b.bias": "model.layers.{bid}.mlp.gate.e_score_correction_bias"}),     # conversion/base.py:763 rename; experts stacked as in sanitize
    # conversion/deepseek.py: kv_b_proj split into attn_k_b (transposed) / attn_v_b = mlx-lm sanitize's embed_q / unembed_out
    "deepseek2": dict(flags=("stack", "rename", "split", "torch_op", "transpose", "view"),
                      names={"ATTN_K_B": "model.layers.{bid}.self_attn.embed_q", "ATTN_V_B": "model.layers.{bid}.self_attn.unembed_out"},
                      exact={"blk.{bid}.exp_probs_b.bias": "model.layers.{bid}.mlp.gate.e_score_correction_bias"}),
    # conversion/granite.py GraniteMoeModel.modify_tensors: `input_linear` split at intermediate_size into gate (first half) and up (= mlx-lm sanitize); GraniteModel(LlamaModel): the q / k permute
    "granitemoe": dict(flags=("permute", "rope_freqs", "split", "stack"), xform="granitemoe",
                       names={"FFN_GATE_EXP": "model.layers.{bid}.block_sparse_moe.switch_mlp.gate_proj", "FFN_UP_EXP": "model.layers.{bid}.block_sparse_moe.switch_mlp.up_proj",
                              "FFN_DOWN_EXP": "model.layers.{bid}.block_sparse_moe.switch_mlp.down_proj"}),
    # conversion/gemma.py Gemma4Model: norm_shift() = 0.0 (NO +1 on the norms, unlike gemma2 / gemma3), the K / V / dense tensors keep the HF layout. What the converter does besides renaming:
    # layer_scalar -> layer_output_scale.weight; router.scale -> ffn_gate_inp.scale; router.per_expert_scale -> ffn_down_exps.scale (renames); experts.gate_up_proj / down_proj keep their
    # [expert, out, in] layout (the class's sanitize splits gate_up_proj); rope_freqs.weight = [1.0] * n_rot + [1e30] * rest, read by the config row (foreign_cfgrows "gemma4") and not loaded.
    "gemma4": dict(proven=False, flags=("arith", "norm_shift", "own_yield", "rope_freqs", "slice"), consumed=("rope_freqs.weight",),
                   exact={"blk.{bid}.layer_output_scale.weight": "model.layers.{bid}.layer_scalar",
                          "blk.{bid}.ffn_gate_inp.scale": "model.layers.{bid}.router.scale",
                          "blk.{bid}.ffn_down_exps.scale": "model.layers.{bid}.router.per_expert_scale",
                          "blk.{bid}.ffn_gate_up_exps.weight": "model.layers.{bid}.experts.gate_up_proj",
                          "blk.{bid}.ffn_down_exps.weight": "model.layers.{bid}.experts.down_proj"}),
}


def vetted(arch: str, what: str, default=()):
    from . import foreign_unproven as UP
    if UP.gated_off(arch):              # a vetted line marked proven=False does not exist while KYOJIN_KIT_UNPROVEN_ROWS is off
        return default
    return VETTED.get(arch, {}).get(what, default)


@dataclass
class NameMap:
    arch: str
    top: dict = field(default_factory=dict)         # "token_embd.weight" -> HF name
    blk: dict = field(default_factory=dict)         # kind ("attn_q") -> HF template with {bid}
    skipped: dict = field(default_factory=dict)     # tensor key -> why it has no parameter
    n_trunk: int | None = None
    ambiguous: dict = field(default_factory=dict)
    exact: dict = field(default_factory=dict)       # "blk.{bid}.ssm_a" -> parameter template, for tensors whose name has no .weight / .bias form

    def hf(self, name: str) -> str | None:
        """GGUF tensor name -> parameter name; None for a block past the trunk (draft head: llama.cpp loads those with TENSOR_SKIP). Unknown names raise."""
        for sfx in (".weight", ".bias"):
            if name.endswith(sfx) and name[:-len(sfx)] in self.top:
                return self.top[name[:-len(sfx)]] + sfx
        m = _BLK.fullmatch(name)
        if not m:
            raise KitError(f"GGUF tensor {name!r}: no mapping to the {self.arch} architecture")
        i, rest = int(m[1]), m[2]
        if self.n_trunk is not None and i >= self.n_trunk:
            return None
        if "blk.{bid}." + rest in self.exact:
            return self.exact["blk.{bid}." + rest].format(bid=i)
        kind, _, sfx = rest.rpartition(".")
        if kind not in self.blk or sfx not in ("weight", "bias"):
            raise KitError(f"GGUF tensor {name!r}: no mapping to the {self.arch} architecture")
        return self.blk[kind].format(bid=i) + "." + sfx


# mlx-lm stacks `mlp.experts.N.<proj>` into one `mlp.switch_mlp.<proj>` in sanitize(); converter candidates are in the pre-stacking form
_STACK = (".switch_mlp.", ".experts.")


def resolve(arch: str, names, n_trunk: int | None = None, strict: bool = True, check: bool = True) -> NameMap:
    """Pick, for every tensor of `arch`, the converter candidate that the model class really has. `names`: the parameter names of the class
    (mlx-lm: tree_flatten(model.parameters()) keys). A candidate that two tensors could take goes to the tensor that has no other choice
    (GLM-4: post_attention_layernorm is the ffn norm, so the attention post norm takes post_self_attn_layernorm); what stays ambiguous raises when
    `strict`. A required tensor with no taker raises (the class lacks an op: new model code is needed) unless check=False (coverage census: the NameMap.skipped
    dict then lists them)."""
    if not have(arch):
        raise KitError(f"GGUF architecture {arch!r} is not in the generated llama.cpp tables")
    have_names = {}                                            # candidate-form name -> the model's own name (layer index and expert stacking normalised)
    for n in names:
        for f in (n, re.sub(r"^layers\.\d+\.", "layers.{bid}.", n), re.sub(r"(?<![^.])\d+(?=\.)", "{bid}", n, count=1)):       # the block list is `layers.N` in mlx-lm, `h.N` / `blocks.N` elsewhere: the first numeric path part
            f = re.sub(r"\.layers\.\d+\.", ".layers.{bid}.", f)
            have_names[f] = f
            have_names[f.replace(_STACK[0], _STACK[1])] = f
    present = {}
    for key in tensor_status(arch):
        cs = []
        for c in candidates(arch, key):
            for sfx in (".weight", ".bias"):
                if c + sfx in have_names:
                    cs.append(have_names[c + sfx][:-len(sfx)])
                    break
        present[key] = list(dict.fromkeys(cs))
    chosen, changed = {}, True
    while changed:
        changed = False
        for key, cs in present.items():
            if key in chosen:
                continue
            free = [c for c in cs if c not in chosen.values()]
            if len(free) == 1:
                chosen[key] = free[0]
                changed = True
    out = NameMap(arch, n_trunk=n_trunk)
    amb = {}
    for key, status in tensor_status(arch).items():
        tpl = G.GGUF_NAME[key]
        free = [c for c in present[key] if c not in chosen.values() or chosen.get(key) == c]
        if key not in chosen and len(free) > 1:
            amb[key] = free
        hit = chosen.get(key) or (free[0] if free else None)
        if hit is None:
            out.skipped[key] = "required by llama.cpp, no parameter of this class carries it" if status == "required" else f"{status}, no parameter"
            continue
        if tpl.startswith("blk."):
            out.blk[_kind(tpl)] = hit
        else:
            out.top[tpl] = hit
        if key in amb:
            chosen[key] = hit
    for key, hit in vetted(arch, "names", {}).items():            # tensors the converter renames itself (modify_tensors yields GGUF names): no tensor_mapping candidate exists
        tpl = G.GGUF_NAME[key]
        if not any(hit + sfx in have_names for sfx in (".weight", ".bias") + (("",) if VETTED.get(arch, {}).get("proven") is False else ())):      # "" = a bare array parameter (flint softmax_offset): its tensor is renamed by an `exact` line
            raise KitError(f"{arch}: vetted name {hit!r} for {key} is not a parameter of the model class")
        (out.blk.__setitem__(_kind(tpl), hit) if tpl.startswith("blk.") else out.top.__setitem__(tpl, hit))
        out.skipped.pop(key, None)
    out.exact = dict(vetted(arch, "exact", {}))
    bad = [k for k, v in out.skipped.items() if v.startswith("required")]
    if bad and check:
        raise KitError(f"{arch}: llama.cpp requires {bad} and the model class has no parameter for them (a new op: hand-written model code)")
    if amb and strict and check:
        raise KitError(f"{arch}: tensors with several candidate parameters and no way to tell: {amb}")
    out.ambiguous = amb
    return out


# ---------------------------------------------------------------------------------------------------------- metadata -> config
def _rows(arch: str):
    seen = []
    for c in G.CONVERTER.get(arch, ()):
        for cls in c["chain"]:
            if cls not in seen:
                seen.append(cls)
                for method, args, cond in G.ROWS.get(cls, ()):
                    yield cls, method, args, cond


def metadata_rows(arch: str) -> list:
    """The converter's metadata, one dict per writer call: gguf key, source (hparam keys / rope parameter / block count / constant / expression), conditional."""
    out = []
    for cls, method, args, cond in _rows(arch):
        ks = G.WRITER_KEYS.get(method)
        if not ks:
            continue
        out.append(dict(cls=cls, method=method, key=ks[0].replace("{arch}", arch), src=args[0] if args else ("none",), cond=cond))
    return out


def snake(hf_arch: str) -> str:
    s = re.sub(r"(ForCausalLM|ForConditionalGeneration|LMHeadModel|Model)$", "", hf_arch)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s).lower()


# GGUF architectures whose mlx-lm / mlx-vlm model_type is not the snake case of a converter class the generated table lists for them (checked against the model's own
# config.json `model_type` and the module's ModelArgs). `minimax-m3`: the converter names MiniMaxM3Sparse*, the config of the model declares `minimax_m3_vl` (snake() splits
# "MiniMax" into mini_max). `nemotron_h_moe`: llama.cpp's NemotronHForCausalLM picks this GGUF architecture when the config has routed experts (conversion/nemotron.py), but the
# table lists only the Puzzle class under it; the model's config declares `nemotron_h`.
MODEL_TYPE_ALIASES = {}           # proven aliases (none yet); the two below have no Metal proof: foreign_unproven.UNPROVEN_ALIASES, in force with KYOJIN_KIT_UNPROVEN_ROWS


def model_type_candidates(arch: str) -> list:
    out = [snake(h) for c in G.CONVERTER.get(arch, ()) for h in c["hf_arch"]]
    from . import foreign_unproven as UP
    return out + [m for m in MODEL_TYPE_ALIASES.get(arch, ()) + UP.aliases(arch) if m not in out]


_HP = re.compile(r"(?:self\.)?hparams\[['\"](\w+)['\"]\]")


def _plain_hparams(expr: str) -> tuple:
    """`hparams['a']` or `hparams['a'] if 'a' in hparams else hparams['b']`: nothing but hparam lookups, so the file key is that hparam. () for any other expression."""
    keys = _HP.findall(expr)
    rest = re.sub(r"\bif\b|\belse\b|\bin\b|'\w+'|\"\w+\"|\bhparams\b|\s", "", _HP.sub("", expr))
    return tuple(dict.fromkeys(keys)) if keys and not rest else ()


def _hparam_arith(expr: str, hp: dict):
    """Value of an expression made only of hparams lookups, numbers and + - * / // (ast, nothing is executed); None when it is anything else or a lookup is missing."""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str) and n.slice.value in hp:
            v = n.value
            if (isinstance(v, ast.Name) and v.id == "hparams") or (isinstance(v, ast.Attribute) and v.attr == "hparams"):
                return hp[n.slice.value]
        raise ValueError
    try:
        return ev(ast.parse(expr, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return None


def kit_config(arch: str, kv: dict, fields=None, vocab: int | None = None, tied: bool | None = None, allow_keys=(), allow_flags=(), tensors=()) -> tuple:
    """(config dict, report). `fields`: names the model's args class accepts (an HF key is used only if the class has it; None = every first candidate).
    A file key under `<arch>.` that no direct converter row explains, and a value-changing converter transform not in allow_flags, raise."""
    cfg, used, unmapped = {}, set(), []
    allow_keys = tuple(allow_keys) + vetted(arch, "keys")
    flags = [f for f in value_transforms(arch) if f not in tuple(allow_flags) + vetted(arch, "flags")]
    if flags:
        raise KitError(f"{arch}: the llama.cpp converter changes tensor values / layout ({', '.join(flags)}): needs a vetted transform before a file of it is served")
    for r in metadata_rows(arch):
        src = r["src"]
        v = kv.get(r["key"])
        if v is None:
            continue
        if src[0] == "block_count":
            keys = G.BLOCK_COUNT_KEYS
        elif src[0] == "hparam":
            keys = src[1]
        elif src[0] == "rope" and src[1] in ("rope_theta",):
            keys = ("rope_theta",)
        elif src[0] == "expr" and _plain_hparams(src[1]):
            keys = _plain_hparams(src[1])
        else:
            unmapped.append((r["key"], src[0] + ":" + str(src[1] if len(src) > 1 else "")))
            continue
        k = next((k for k in keys if fields is None or k in fields), None)
        if k is None and r["key"].endswith(".context_length"):         # the trained length: no effect on logits, and mlx-lm classes without the field do not need it
            used.add(r["key"])
            continue
        if k is None:
            unmapped.append((r["key"], f"no config field among {list(keys)}"))
            continue
        cfg.setdefault(k, v)
        used.add(r["key"])
    for key, why in list(unmapped):                                  # `rope.dimension_count <- hidden_size // num_attention_heads`: a value the class derives itself; the file must agree
        expr = why[5:] if why.startswith("expr:") else ""
        want = _hparam_arith(expr, cfg) if expr else None
        if want is not None and key in kv:
            if kv[key] != want:
                raise KitError(f"{arch}: {key} = {kv[key]} but the converter derives {want} from {expr} (a value the model class would compute differently)")
            used.add(key)
            unmapped.remove((key, why))
    rest = sorted(k for k in kv if k.startswith(arch + ".") and k not in used and k not in allow_keys)
    if rest:
        raise KitError(f"GGUF keys not explained by a direct converter row for {arch}: {rest} (unmapped rows: {unmapped[:6]})")
    for k, expr in vetted(arch, "cfg", {}).items():
        if fields is None or k in fields:
            from . import foreign_cfgrows as CR
            cfg[k] = CR.ev(expr, CR._Env(arch, kv, {}))
    for k, kinds in (("attention_bias", ("attn_q.bias", "attn_k.bias", "attn_v.bias", "attn_output.bias")), ("mlp_bias", ("ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"))):
        if tensors and fields is not None and k in fields and k not in cfg:          # a flag the HF config carries and the GGUF does not: the file's own tensor list says
            cfg[k] = any(t.endswith(kinds) for t in tensors)
    if vocab is not None and (fields is None or "vocab_size" in fields):
        cfg["vocab_size"] = vocab
    if tied is not None and (fields is None or "tie_word_embeddings" in fields):
        cfg["tie_word_embeddings"] = tied
    return cfg, dict(used=sorted(used), unmapped=unmapped)


# ---------------------------------------------------------------------------------------------------------- hand-written tables vs generated rows
def compare_hand(arch: str, hand) -> dict:
    """Differences between a hand-written foreign_names.Arch and the generated rows of the same llama.cpp architecture.
    not_candidate: hand rows whose HF name the converter never maps onto that GGUF tensor (a rename the model class or a converter transform does);
    unknown_kind: hand GGUF names with no tensor of that architecture in gguf-py; not_in_hand: tensors llama.cpp's loader requires or optionally reads
    that the hand table has no row for."""
    rows = {}                                         # gguf name template (with suffix part stripped) -> tensor key
    for key in G.ARCH_TENSORS[arch]:
        rows[G.GGUF_NAME[key]] = key
    out = dict(not_candidate=[], unknown_kind=[], not_in_hand=[])
    seen = set()

    def check(tpl, hf_tpl, label):
        key = rows.get(tpl)
        if key is None:
            out["unknown_kind"].append(label)
            return
        seen.add(key)
        if hf_tpl not in candidates(arch, key):
            out["not_candidate"].append((label, hf_tpl, [c for c in candidates(arch, key)][:4]))

    for gname, hf in hand.top.items():
        stem, _, sfx = gname.rpartition(".")
        check(stem, hf[:-len(sfx) - 1], gname)
    for kind, hf in hand.blk.items():
        check("blk.{bid}." + kind, "model.layers.{bid}." + hf, kind)
    for kind, hf in hand.exact.items():
        stem, _, sfx = kind.rpartition(".")
        check("blk.{bid}." + (stem if sfx in ("weight", "bias") else kind), "model.layers.{bid}." + (hf[:-len(sfx) - 1] if sfx in ("weight", "bias") and hf.endswith(sfx) else hf), kind)
    if hand.mtp:
        for gname, hf in hand.mtp.items():
            out["not_candidate"].append((f"blk.<nextn>.{gname}", hf, ["(draft block: renamed by the converter's filter_tensors, not by a mapping row)"]))
    st = tensor_status(arch)
    out["not_in_hand"] = sorted(k for k, s in st.items() if s in ("required", "optional") and k not in seen)
    return out
