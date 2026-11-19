"""Coverage census of the architecture import kit (CPU, no weights): for every generated architecture, find a model class of the same family in the installed
mlx-lm / mlx-vlm, build it with a tiny config, and measure how far the generated tables resolve it with ZERO hand-written lines.
Strict: a name that is only ambiguous / guessed, a config the tiny build could not be made for, an inverse nobody wrote = NOT resolved.

  PYTHONPATH=. python tools/archkit_census.py [--md OUT.md] [--json OUT.json] [--only a,b]

Per architecture:
  class      mlx-lm / mlx-vlm module (how it was found: hf = converter's HF architecture name, name = the GGUF architecture name)
  params%    parameters of the tiny class that some GGUF tensor of the architecture reaches through resolve() (strict load needs 100)
  req        llama.cpp-required tensors with no parameter in the class (a new op: model code)
  flags      converter value transforms (kit refuses a file until they are vetted)
  cfg        required args fields the metadata rows do not give / converter rows that place no field (file keys the kit would refuse)
  verdict    zero-line | needs N vetted lines (lower bound) | needs model code | unverified (tiny config not built)
Shapes: a GGUF shape check needs a file (or a converter replay: done for the 6 round-trip architectures only); here a rename-only architecture is shape-consistent by construction."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import pkgutil
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)
from mlx.utils import tree_flatten  # noqa: E402

from kyojin_mlx import foreign_arch_gen as G, foreign_archkit as K  # noqa: E402

# tiny values by field name (generic names of HF configs, no model names)
TINY = dict(hidden_size=64, d_model=64, n_embd=64, dim=64, model_dim=64, hidden_dim=64, block_dim=64,
            num_hidden_layers=2, n_layer=2, num_layers=2, n_layers=2, num_transformer_layers=2, num_dense_layers=1,
            num_attention_heads=4, n_head=4, num_heads=4, n_heads=4, num_key_value_heads=2, n_kv_heads=2, num_kv_heads=2, head_dim=16,
            intermediate_size=128, ffn_hidden_size=128, n_inner=128, ff_intermediate_size=128, shared_intermediate_size=32, intermediate_size_mlp=128,
            vocab_size=64, rms_norm_eps=1e-6, norm_eps=1e-6, layer_norm_eps=1e-5, layer_norm_epsilon=1e-5, layernorm_epsilon=1e-5,
            max_position_embeddings=128, n_positions=128, n_ctx=128, model_max_length=128, rope_theta=10000.0, rotary_emb_base=10000.0, partial_rotary_factor=1.0,
            rotary_pct=1.0, rotary_dim=16, num_experts=4, num_local_experts=4, n_routed_experts=4, moe_num_experts=4, num_experts_per_tok=2, moe_topk=2,
            moe_intermediate_size=32, expert_ffn_hidden_size=32, n_shared_experts=1, num_shared_experts=1, num_shared_expert=1, shared_expert_intermediate_size=32,
            kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, first_k_dense_replace=0, n_group=1, topk_group=1,
            routed_scaling_factor=1.0, norm_topk_prob=True, attention_bias=False, mlp_bias=False, use_bias=False, tie_word_embeddings=False, sliding_window=16,
            hidden_act="silu", conv_kernel=4, use_conv_bias=False, conv_bias=False, state_size=16, time_step_rank=4, n_groups=1, decoder_sparse_step=1,
            mlp_only_layers=[], use_qk_norm=False, qk_norm=False, logits_scaling=1.0, attention_multiplier=1.0, embedding_multiplier=1.0, residual_multiplier=1.0,
            dim_model_base=64, scale_depth=1.0, scale_emb=1.0, rope_scaling=None, topk_method="greedy", scoring_func="softmax", moe_layer_freq=1,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4)
BY_TYPE = {"int": 2, "float": 1e-6, "bool": False, "str": "", "list": [], "dict": {}, "List": [], "Dict": []}


def modules(pkg):
    out = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        out.setdefault(m.name.replace("_", ""), m.name)
    return out


def find_classes(arch, lm, vl):
    """[(source, module name, how)] in the order load_cpu would try: converter HF names in mlx-lm, the GGUF name in mlx-lm, then mlx-vlm."""
    out = []
    from kyojin_mlx import foreign_cfgrows as CR
    if CR.has_rows(arch) and CR.ROWS[arch]["model_type"].replace("_", "") in lm:             # a config row names the class it builds (load_cpu uses the same): it goes first
        out.append(("mlx_lm", lm[CR.ROWS[arch]["model_type"].replace("_", "")], "row"))
    if arch.replace("_", "") in lm:                                  # the module named like the GGUF architecture first (the family of that very name)
        out.append(("mlx_lm", lm[arch.replace("_", "")], "name"))
    for mt in K.model_type_candidates(arch):
        if mt.replace("_", "") in lm:
            out.append(("mlx_lm", lm[mt.replace("_", "")], "hf"))
    for mt in K.model_type_candidates(arch) + [arch]:
        if mt.replace("_", "") in vl:
            out.append(("mlx_vlm", vl[mt.replace("_", "")], "hf" if mt != arch else "name"))
    seen, res = set(), []
    for o in out:
        if o[:2] not in seen:
            seen.add(o[:2])
            res.append(o)
    return res


def _req(f):
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def tiny(args_cls, mod_name):
    """Tiny config of an args class: the tiny value (by field name) for every required field and every field whose default is None; the rest keeps the class default
    (so a dense class stays dense). Required fields with no name entry get a value by annotation."""
    d = {}
    for f in dataclasses.fields(args_cls):
        if f.name == "model_type":
            d[f.name] = mod_name
        elif f.name == "tie_word_embeddings":
            d[f.name] = False                                      # a separate head parameter: an untied file carries output.weight
        elif f.name in TINY and (_req(f) or f.default is None):
            d[f.name] = TINY[f.name]
        elif f.name == "layer_types" and _req(f):
            d[f.name] = ["full_attention"] * 2
        elif _req(f):
            ty = str(f.type).replace("typing.", "").split("[")[0].lower()
            d[f.name] = {"int": 2, "float": 1.0, "bool": False, "str": "", "list": [], "dict": {}}.get(ty, 2)
    lt = next((f for f in dataclasses.fields(args_cls) if f.name == "layer_types"), None)
    from kyojin_mlx import foreign_unproven as UP
    if UP.on() and lt is not None and not _req(lt) and lt.default is None and "layer_types" not in d and isinstance(d.get("num_hidden_layers"), int):
        d["layer_types"] = ["full_attention"] * d["num_hidden_layers"]       # None default but the layer count was set by the tiny value: a list of that length (a class that derives its own keeps its default)
    return d


def text_args_class(mod):
    return next((v for n, v in vars(mod).items() if isinstance(v, type) and dataclasses.is_dataclass(v) and "Text" in n and ("Args" in n or "Config" in n)), None)


def build(src, mod_name):
    """(model, args class, config) with the tiny config, or raises."""
    mod = importlib.import_module(("mlx_lm.models." if src == "mlx_lm" else "mlx_vlm.models.") + mod_name)
    if src == "mlx_vlm":
        mod_args = getattr(mod, "ModelConfig", None) or getattr(mod, "ModelArgs")
    else:
        mod_args = mod.ModelArgs
    names = {f.name for f in dataclasses.fields(mod_args)}
    cfg = tiny(mod_args, mod_name)
    if "text_config" in names:                                     # wrapper args: the text trunk is the nested dict
        tcls = text_args_class(mod)
        cfg["text_config"] = tiny(tcls, mod_name) if tcls is not None else {"model_type": mod_name, **{k: v for k, v in TINY.items()}}
        if "vision_config" in names:
            cfg["vision_config"] = {"model_type": mod_name}
    a = mod_args.from_dict(cfg) if hasattr(mod_args, "from_dict") else mod_args(**{k: v for k, v in cfg.items() if k in names})
    return mod.Model(a), mod_args, cfg, (text_args_class(mod) if "text_config" in names else None)


def norm_names(names):
    pre = "language_model." if names and all(n.startswith("language_model.") for n in names if not n.startswith(("vision_tower", "multi_modal_projector", "audio"))) else ""
    out = []
    for n in names:
        if n.startswith(("vision_tower", "multi_modal_projector", "audio_tower", "embed_vision", "embed_audio")):
            continue
        out.append(n[len(pre):] if pre and n.startswith(pre) else n)
    return out


def _norm(n):
    return re.sub(r"(?<![^.])\d+(?=\.)", "{bid}", n, count=1)


def covered(nm, names):
    """normalised parameter names of the class that the name map reaches (a tensor kind hits its .weight and .bias)."""
    hits = set(nm.top.values()) | set(nm.blk.values()) | {v.rpartition(".")[0] if v.rpartition(".")[2] in ("weight", "bias") else v for v in nm.exact.values()}
    hits |= {h.replace(".switch_mlp.", ".experts.") for h in hits}
    ps = {_norm(n) for n in names}
    cov = set()
    for p in ps:
        stem = p.rpartition(".")[0] if p.rpartition(".")[2] in ("weight", "bias") else p
        if stem in hits or stem.replace(".switch_mlp.", ".experts.") in hits or p in hits or p in {v.replace("{bid}", "{bid}") for v in nm.exact.values()}:
            cov.add(p)
    return cov, ps


def config_report(arch, Args, tiny_cfg):
    """Which required args fields the metadata rows give, and which converter rows place no field. Synthetic file: every placeable row carries the tiny value."""
    fields = {f.name for f in dataclasses.fields(Args)}
    rows = K.metadata_rows(arch)
    kv, unmapped = {}, []
    for r in rows:
        src = r["src"]
        if src[0] == "block_count":
            ks = G.BLOCK_COUNT_KEYS
        elif src[0] == "hparam":
            ks = src[1]
        elif src[0] == "rope" and src[1] in ("rope_theta",):
            ks = ("rope_theta",)
        elif src[0] == "expr" and K._plain_hparams(src[1]):
            ks = K._plain_hparams(src[1])
        else:
            ks = None
        tgt = next((k for k in (ks or ()) if k in fields), None)
        if tgt is not None:
            kv[r["key"]] = tiny_cfg.get(tgt, TINY.get(tgt, 2))
        elif ks and r["key"].endswith(".context_length"):
            continue
        elif r["key"].startswith(arch + "."):
            unmapped.append((r["key"], src[0] + (":" + str(src[1])[:60] if len(src) > 1 else ""), bool(r["cond"])))
    cfg, rep = K.kit_config(arch, kv, fields, allow_flags=K.value_transforms(arch), allow_keys=tuple(kv))
    # arithmetic rows (`rope.dimension_count <- hidden // heads`): satisfied when the expression evaluates over the placed values
    rest, rest_cond = [], []
    for key, why, cond in unmapped:
        expr = why[5:] if why.startswith("expr:") else ""
        if expr and K._hparam_arith(expr, cfg) is not None:
            continue
        (rest_cond if cond else rest).append(key.replace(arch + ".", "", 1))
    miss = [f.name for f in dataclasses.fields(Args) if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            and f.name not in cfg and f.name not in ("model_type", "vocab_size", "tie_word_embeddings", "attention_bias", "mlp_bias")]     # the last four the kit takes from the file's own tensor list
    return miss, rest, rest_cond


def vetted_lines(arch):
    """Code lines of the hand rows of an architecture that already has them: the VETTED entry (comments excluded) + one xform-set entry per `xform`."""
    import inspect
    src = inspect.getsource(K).split("VETTED = {")[1].split("\n}\n")[0]
    m = re.search(r'^    "%s": dict\(.*?(?=^    "\w+": dict\(|^    #|\Z)' % re.escape(arch), src, re.S | re.M)
    if not m:
        return 0
    n = sum(1 for l in m.group(0).splitlines() if l.strip() and not l.strip().startswith("#"))
    return n + (1 if K.vetted(arch, "xform", "") else 0)


def run(only=None):
    import mlx_lm.models as pl
    import mlx_vlm.models as pv
    lm, vl = modules(pl), modules(pv)
    from kyojin_mlx import foreign_names as N
    hand = set(N.ARCHS)
    rows = []
    for arch in sorted(G.ARCH_TENSORS):
        if only and arch not in only:
            continue
        r = dict(arch=arch, cls="-", how="", params=None, req=[], flags=list(K.value_transforms(arch)), miss=[], unmapped=[], unmapped_cond=[], note="", n_lines=0, verdict="", hand=arch in hand,
                 has_loader=arch in G.LOADER, has_converter=bool(G.CONVERTER.get(arch)))
        cl = find_classes(arch, lm, vl)
        if not cl:
            r.update(verdict="needs model code", note="no model class of this family in mlx-lm / mlx-vlm" + ("" if r["has_converter"] else "; no converter class in llama.cpp (no HF names to map)"))
            rows.append(r)
            continue
        tries = []
        for src, mn, how in cl:
            try:
                model, Args, tcfg, tcls = build(src, mn)
                names = norm_names([k for k, _ in tree_flatten(model.parameters())])
                break
            except BaseException as e:                               # SystemExit included: one module exits when its optional dependency is missing
                tries.append(f"{src}.{mn}: {type(e).__name__}: {str(e)[:70]}")
        else:
            r.update(cls=f"{cl[0][0]}.{cl[0][1]}", how=cl[0][2], verdict="unverified", note="tiny config not built (" + "; ".join(tries)[:200] + ")")
            rows.append(r)
            continue
        r.update(cls=f"{src}.{mn}", how=how, tries=tries)
        if not r["has_converter"]:
            r.update(verdict="needs model code", note="class found, but llama.cpp has no converter class for this architecture (no HF names to resolve against)")
            rows.append(r)
            continue
        nm = K.resolve(arch, names, check=False)
        cov, ps = covered(nm, names)
        r["params"] = (len(cov), len(ps))
        r["unresolved_params"] = sorted(ps - cov)[:6]
        r["req"] = sorted(k for k, v in nm.skipped.items() if v.startswith("required"))
        r["amb"] = sorted(nm.ambiguous)
        st = K.tensor_status(arch)
        r["optcond_unresolved"] = sum(1 for k, v in nm.skipped.items() if not v.startswith("required") and st.get(k) in ("optional", "conditional"))
        try:
            r["miss"], r["unmapped"], r["unmapped_cond"] = config_report(arch, tcls or Args, tcfg.get("text_config", tcfg) if tcls else tcfg)
        except K.KitError as e:
            r["note"] += f" kit_config: {str(e)[:100]}"
            r["miss"], r["unmapped"], r["unmapped_cond"] = ["?"], [], []
        # verdict
        n = vetted_lines(arch)
        r["n_lines"] = n
        have_all = len(cov) == len(ps)
        # how many class parameters have no tensor: a required tensor with no parameter AND a parameter with no tensor, in equal number, is a naming difference (vetted `names` rows);
        # a required tensor with no parameter and no stray parameter is an op the class lacks
        stray = len(ps) - len(cov)
        need = (3 if r["flags"] else 0) + (1 if (r["unmapped"] and not r["flags"]) else 0) + (1 if r["miss"] else 0)
        if r["amb"]:
            r["verdict"], r["n_min"] = "needs vetted lines", len(r["amb"]) + need
            r["note"] = "ambiguous names: " + ", ".join(r["amb"][:4])
        elif r["req"] and stray < len(r["req"]):
            r["verdict"] = "needs model code"
            r["note"] = "required tensors with no parameter: " + ", ".join(r["req"][:5])
        elif r["req"] or not have_all:
            r["verdict"], r["n_min"] = "needs vetted lines", max(len(r["req"]), stray) + need
            r["note"] = (f"{len(r['req'])} required tensors without a parameter, {stray} parameters without a tensor (naming difference; UNVERIFIED that shapes agree): " + ", ".join(r["req"][:3] + r["unresolved_params"][:2]))
        elif n > 0:
            r["verdict"], r["n_min"] = "needs vetted lines", n
            r["note"] = f"hand rows exist today ({n} lines)"
        elif need:
            r["verdict"], r["n_min"] = "needs vetted lines", need
        else:
            r["verdict"] = "zero-line"
        rows.append(r)
    return rows


def totals(rows):
    from collections import Counter
    c = Counter(r["verdict"] for r in rows)
    return c


def md(rows):
    c = totals(rows)
    n = len(rows)
    z = [r for r in rows if r["verdict"] == "zero-line"]
    out = [f"# Architecture kit census (objective 1 coverage), generated by tools/archkit_census.py\n",
           f"Generated tables: {n} architectures. Strict: only a tiny class that resolves 100 % of its parameters, no required tensor missing, no ambiguity, no converter value transform, no config gap counts as zero-line.\n",
           "## Totals", f"- zero-line: **{c['zero-line']}** of {n}",
           f"  of which a file may carry ANY converter key without refusal (rows under an `if`: rope scaling, sliding window, attention.causal ... are refused by kit_config today): {sum(1 for r in z if not r['unmapped_cond'])}",
           f"- needs N vetted lines (lower bound): **{c['needs vetted lines']}**", f"- needs model code (no class, new op, or no converter): **{c['needs model code']}**",
           f"- unverified (class found, tiny config not built): **{c['unverified']}**",
           f"- class of the same family found: {sum(1 for r in rows if r['cls'] != '-')}; tiny class built: {sum(1 for r in rows if r['params'] is not None or r['verdict'] == 'needs model code' and r['cls'] != '-' and r['note'].startswith('class found'))}",
           f"- hand-written Mac architectures (foreign_names.ARCHS): {sum(1 for r in rows if r['hand'])}; the kit reproduces them with VETTED rows, so they are counted by what the TABLES alone give.\n",
           "Reading: zero-line = resolves from the tables alone for a file WITHOUT the conditional keys above (the usual case); verified end to end on a real file only for olmo2 (item 2 of the A2 child), the others by resolution of the tiny class. "
           "`needs vetted lines` for the 12 hand architectures counts the VETTED rows that exist today (their hand Arch rows and `_cfg_*` builders are not counted).\n",
           "Zero-line list: " + (", ".join(r["arch"] for r in z) or "none") + "\n",
           "Definitions: params% = parameters of the tiny class reached by some GGUF tensor (resolve, strict, ambiguity = not resolved); req = tensors llama.cpp requires that no parameter carries; flags = converter value transforms (the kit refuses a file of the architecture until they are vetted); "
           "cfg = required args fields the metadata rows do not give / converter rows that place no field (the kit refuses a file carrying such a key). N lines is a LOWER bound: 3 per flagged architecture (VETTED row 2 + xform-set 1, as gemma2 / granitemoe) + 1 for keys / cfg; "
           "whether the inverse of a flag exists in foreign_xform is UNVERIFIED unless the architecture is in the hand set. Shapes of GGUF tensors: not checkable without a file; a rename-only architecture is consistent by construction.\n",
           "| architecture | class (how) | params % | req missing | transforms (flags) | config gaps (missing fields / unplaced keys) | verdict | note |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pc = "-" if r["params"] is None else f"{100 * r['params'][0] / max(1, r['params'][1]):.0f}% ({r['params'][0]}/{r['params'][1]})"
        v = r["verdict"] if r["verdict"] != "needs vetted lines" else f"needs >= {r.get('n_min', 1)} vetted lines"
        fl = ",".join(r["flags"]) or "-"
        cg = (",".join(r["miss"]) or "-") + " / " + (",".join(r["unmapped"][:5]) + ("..." if len(r["unmapped"]) > 5 else "") if r["unmapped"] else "-") + (f" (+{len(r['unmapped_cond'])} cond)" if r["unmapped_cond"] else "")
        note = (r["note"] or "") + (" [first class failed: " + r["tries"][0][:70] + "]" if r.get("tries") and r["params"] is not None else "") + (" [hand Arch row exists]" if r["hand"] else "")
        out.append(f"| {r['arch']} | {r['cls']} ({r['how']}) | {pc} | {','.join(r['req'][:4]) or '-'} | {fl} | {cg} | {v} | {note[:160]} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--only")
    a = ap.parse_args()
    rows = run(set(a.only.split(",")) if a.only else None)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=str)
    text = md(rows)
    if a.md:
        open(a.md, "w").write(text)
    print("\n".join(text.splitlines()[:14]))
