"""Day-0 helpers for tools/day0_inspect.py: safetensors packs, tokenizer + chat template, vision side files, KV / draft terms of the ceiling, serve command.
Every answer comes from the engine's own loaders and tables (imported, not copied); header-only, CPU, no weight byte is read.
Rules that are NOT a table of the engine but a reading of serve/backends.py load_backend are marked 'rule' in the text of the answer."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
GIB = 1 << 30
SYS_MSG = "You are a careful assistant."
USER_MSG = "What is the weather in Paris?"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
VISION_NAME = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision_encoder|multi_modal_projector|mm_projector|image_encoder)(\.|$)")
AUDIO_NAME = re.compile(r"(^|\.)(audio_tower|audio_encoder|audio_model)(\.|$)")
MTP_NAME = re.compile(r"(^|\.)(mtp|nextn_predict_layers?|model_mtp)(\.|$|_)|\.nextn\.")
EXPERT_NAME = re.compile(r"\.(experts|switch_mlp|block_sparse_moe)\b|\bexperts\.(\d+\.|gate_up|down|gate_proj)")


# ------------------------------------------------------------------------------------------------------------------------------ safetensors headers
def read_headers(path: str) -> tuple[dict, list[str]]:
    """{tensor: (dtype, shape, nbytes)} and the file list, from the 8-byte length + JSON header of each shard. The shard list is the weight map of
    model.safetensors.index.json when there is one (as vendor_input.Checkpoint), else every *.safetensors."""
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        files = [p]
    else:
        idx = os.path.join(p, "model.safetensors.index.json")
        names = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.isfile(idx) else sorted(f for f in os.listdir(p) if f.endswith(".safetensors") and not f.startswith("._"))
        files = [os.path.join(p, f) for f in names]
    out: dict = {}
    for f in files:
        size = os.path.getsize(f)
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 100 * 1024 ** 2 or 8 + n > size:
                raise ValueError(f"{f}: bad safetensors header length {n}")
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, m in h.items():
            b, e = m["data_offsets"]
            out[k] = (m["dtype"], tuple(m["shape"]), e - b)
    return out, files


def role_of(name: str, shape: tuple, vocab: int | None) -> str:
    """trunk | routed experts | embeddings | head | mtp | vision | audio."""
    if VISION_NAME.search(name):
        return "vision"
    if AUDIO_NAME.search(name):
        return "audio"
    if MTP_NAME.search(name):
        return "mtp"
    if name.endswith(("lm_head.weight", "lm_head.scales", "lm_head.biases", "lm_head.weight_scale")) or re.match(r"lm_head\.", name) or name.startswith("output."):
        return "head"
    if re.search(r"(embed_tokens|wte|tok_embeddings|word_embeddings)\.", name) and "per_layer" not in name:
        return "embeddings"
    if EXPERT_NAME.search(name) and "shared_expert" not in name:
        return "routed experts"
    return "trunk"


def formats_of(info: dict, tinfo: dict | None = None) -> dict:
    """{linear prefix: fmt} with the engine's own classifier (vendor_input.classify_linear); EXL3 / MLX layouts are named from their sibling tensors."""
    from kyojin_mlx import vendor_input as VI
    tinfo = tinfo or {k: (v[0], v[1]) for k, v in info.items()}
    fmt = {}
    for p in VI.linear_prefixes(tinfo):
        c = VI.classify_linear(tinfo, p)
        if c is not None:
            fmt[p] = c["fmt"]
    return fmt


def label_of(name: str, info: dict, fmt: dict, aux: frozenset = frozenset()) -> str:
    if name in aux:
        return "nvfp4 scale stream (csf)"
    pre = name.rpartition(".")[0]
    if pre in fmt:
        return fmt[pre]
    if pre + ".trellis" in info:
        return "exl3"
    if pre + ".scales" in info and pre + ".weight" in info:
        return "mlx_quant"
    return info[name][0] + (" (norm / bias / small)" if len(info[name][1]) <= 1 else "")


def pack_family(path: str, cfg: dict) -> dict:
    """Which loader family the engine would route this directory to: the same test as foreign_linear.is_foreign (exl3 / MLX packs are never foreign)."""
    from kyojin_mlx import foreign_linear as FL
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    if qc.get("quant_method") == "exl3":
        return dict(family="exl3", via="config quantization_config.quant_method == exl3")
    if "quantization" in cfg:
        return dict(family="mlx", via="config has a 'quantization' block (mlx-lm quantized pack)")
    kind = FL.is_foreign(path)
    if kind == "safetensors":
        return dict(family="foreign", via="foreign_linear.is_foreign == safetensors (NVFP4 / MXFP4 / FP8 linears, read as stored)")
    return dict(family="plain", via="no exl3 / mlx / foreign marker: dense safetensors or a format mlx-lm reads itself")


def has_mlx_lm_class(mt: str) -> bool | None:
    try:
        from mlx_lm import utils as U
    except Exception:  # noqa: BLE001
        return None
    mt = getattr(U, "MODEL_REMAPPING", {}).get(mt, mt)
    return importlib.util.find_spec(f"mlx_lm.models.{mt}") is not None


def inspect_pack(path: str, limit_gib: float = 236.0) -> dict:
    p = os.path.expanduser(path)
    d = p if os.path.isdir(p) else os.path.dirname(p)
    info, files = read_headers(p)
    cfg_path = os.path.join(d, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
    rep: dict = dict(kind="safetensors", path=path, files=[os.path.basename(f) for f in files], blockers=[], notes=[], cfg_present=bool(cfg))
    rep["file_bytes"] = sum(os.path.getsize(f) for f in files)
    rep["n_tensors"] = len(info)
    tc = cfg.get("text_config") or cfg
    mt = cfg.get("model_type", "")
    rep["model_type"] = mt
    rep["architectures"] = cfg.get("architectures")
    vocab = tc.get("vocab_size") or cfg.get("vocab_size")
    fam = pack_family(p if os.path.isdir(p) else d, cfg) if cfg else dict(family="unknown", via="no config.json")
    rep["pack_family"] = fam["family"]
    rep["pack_family_via"] = fam["via"]
    tinfo = {k: (v[0], v[1]) for k, v in info.items()}
    try:                                                     # the engine's own header view: CSF-encoded NVFP4 scale streams read as one F8_E4M3 scale (csf_scale.CsfCheckpoint)
        from kyojin_mlx.csf_scale import CsfCheckpoint
        ck = CsfCheckpoint(d)
        try:
            tinfo = {k: tuple(v) for k, v in ck.info().items()}
        finally:
            ck.close()
    except Exception:  # noqa: BLE001   (a lone .safetensors file, no index: the plain header view stays)
        pass
    fmt = formats_of(info, tinfo)
    try:
        from kyojin_mlx.csf_scale import csf_stream_names
        aux = frozenset(csf_stream_names(tinfo))
    except Exception:  # noqa: BLE001
        aux = frozenset()
    rows: dict = {}
    for n, (dt, sh, nb) in info.items():
        r = role_of(n, sh, vocab)
        lab = label_of(n, info, fmt, aux)
        e = rows.setdefault((r, lab), dict(role=r, format=lab, tensors=0, bytes=0))
        e["tensors"] += 1
        e["bytes"] += nb
    rep["formats"] = sorted(rows.values(), key=lambda e: (e["role"], -e["bytes"]))
    by_role: dict = {}
    for e in rows.values():
        by_role[e["role"]] = by_role.get(e["role"], 0) + e["bytes"]
    rep["bytes_by_role"] = by_role
    from kyojin_mlx import vendor_input as VI
    try:
        det = VI.detect_checkpoint(tinfo, cfg)
        rep["detected"] = dict(primary=det["primary"], counts=det["counts"], unsupported_integer_linears=det["unsupported_integer_linears"])
    except Exception as e:  # noqa: BLE001
        rep["detected"] = dict(primary="unknown", counts={}, unsupported_integer_linears=0)
        rep["notes"].append(f"detect_checkpoint failed: {type(e).__name__}")
    from kyojin_mlx import modeldesc
    row = modeldesc.backend_desc(mt) if mt else None
    rep["backend_row"] = None if row is None else dict(name=row.name, backend=row.backend, server=row.server, arch=row.arch)
    rep["mlx_lm_class"] = has_mlx_lm_class(mt) if mt else None
    unknown_dt = sorted({v[0] for v in info.values() if v[0] not in VI._NP})
    rep["unknown_dtypes"] = unknown_dt
    resident = sum(b for r, b in by_role.items() if r not in ("vision", "audio"))
    rep["resident_est_bytes"] = resident
    rep["limit_bytes"] = int(limit_gib * GIB)
    rep["fits"] = resident <= rep["limit_bytes"]
    rep["mtp_bytes"] = by_role.get("mtp", 0)
    rep["side_tensors"] = {k: by_role[k] for k in ("vision", "audio") if k in by_role}
    mtp_cfg = int(tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers") or 0)
    rep["mtp_layers"] = mtp_cfg
    # ---- verdict
    v, why = "UNVERIFIED", ""
    f = fam["family"]
    backend = None if row is None else row.backend
    if not cfg:
        v, why = "UNVERIFIED", "no config.json next to the safetensors: the loader needs it (model_type picks the backend)"
    elif unknown_dt:
        v, why = "TYPE NEEDED", f"dtype tag(s) {unknown_dt} unknown to vendor_input (no reader)"
    elif rep["detected"]["primary"] == "unsupported-integer-quant":
        v, why = "TYPE NEEDED", f"{rep['detected']['unsupported_integer_linears']} integer-packed linears (GPTQ / AWQ style) that no reader in vendor_input understands"
    elif not rep["fits"]:
        v, why = "DOES NOT FIT", f"resident estimate {resident / GIB:.1f} GiB against the {limit_gib:g} GiB limit"
    elif f == "foreign":
        if row is None:
            v, why = "ROW NEEDED", f"model_type {mt!r} has no modeldesc row: foreign safetensors are routed by the row (backend q35, or the glm5_next vlm branch)"
            from kyojin_mlx import foreign_qwen4exp_st as Q4
            from kyojin_mlx import foreign_unproven as UP
            if mt == Q4.ROW["model_type"] and not Q4.ROW["proven"]:                       # an unproven row exists: never READY before a Metal identity proof
                if UP.on(Q4.ROW["gate"]):
                    v, why = "UNVERIFIED", f"qwen4_exp safetensors row is in force (KYOJIN_KIT_UNPROVEN_ROWS) but unproven: " + "; ".join(Q4.ROW["unproven"])
                else:
                    why += f"; an unproven qwen4_exp row exists: set {UP.ENV} to try it (it will read UNVERIFIED, not READY)"
        elif backend == "generic_foreign":                                                 # an unproven row in force: loads through load_foreign, never READY before a Metal identity proof
            from kyojin_mlx import foreign_qwen4exp_st as Q4
            notes = Q4.ROW["unproven"] if mt == Q4.ROW["model_type"] else []
            v, why = "UNVERIFIED", f"foreign {rep['detected']['primary']} pack, unproven row {row.name} in force (backend generic_foreign via load_foreign): no Metal identity proof yet" + ("; " + "; ".join(notes) if notes else "")
        elif backend == "q35" or (backend == "vlm" and mt == "glm5_next"):
            v, why = "READY", f"foreign {rep['detected']['primary']} pack, loader foreign_linear.load_foreign via the {backend} backend (row {row.name}); end-to-end identity on Metal NOT measured by this tool"
        else:
            v, why = "MODEL CODE NEEDED", f"foreign safetensors reach load_foreign only through the q35 backend and the glm5_next vlm branch (rule, from load_backend); row {row.name} uses backend {backend}"
    elif f in ("mlx", "plain", "exl3"):
        if f == "exl3" and backend not in ("q35", "vlm"):
            v, why = "MODEL CODE NEEDED", f"EXL3 pack: the trellis linears are patched in by the q35 / vlm loaders only (rule); model_type {mt!r} has {'backend ' + backend if backend else 'no modeldesc row'}"
        elif backend in ("q35", "vlm", "generic"):
            v, why = "READY", f"{f} pack, backend {backend} (row {row.name})"
        elif rep["mlx_lm_class"]:
            v, why = "READY", f"{f} pack, no modeldesc row: generic fallback on the stock mlx-lm class for {mt!r} (no speed path, KYOJIN_BACKEND_FALLBACK default on)"
        elif rep["mlx_lm_class"] is None:
            v, why = "UNVERIFIED", "mlx_lm not importable here: cannot tell whether a model class exists"
        else:
            v, why = "MODEL CODE NEEDED", f"no modeldesc row and no mlx_lm.models.{mt} class"
    rep["verdict"], rep["why"] = v, why
    return rep


# ------------------------------------------------------------------------------------------------------------------------------ tokenizer / chat template
def check_tokenizer(model_dir: str | None, cfg: dict | None, gguf_kv: dict | None = None, gguf_path: str | None = None,
                    tokenizer_dir: str | None = None, family: str | None = None) -> dict:
    """Same lookup as serve.server.build_app: tokenizer_config.json -> AutoTokenizer, else a bare tokenizer.json; a GGUF without tokenizer files gets the
    tokenizer.json built from its metadata (foreign_names.write_gguf_tokenizer). The chat template is rendered by the family class the server would pick, with a system
    message and a tools list. Verdict: TOKENIZER OK | TOKENIZER NEEDED | TEMPLATE FAILS."""
    out: dict = dict(verdict=None, why="", steps=[])
    tmp = None
    try:
        tdir = tokenizer_dir or model_dir
        if gguf_kv is not None and not (tdir and os.path.isfile(os.path.join(tdir, "tokenizer.json"))):
            from kyojin_mlx import foreign_names as N
            from kyojin_mlx import foreign_gguf as GF
            tmp = tempfile.mkdtemp(prefix="day0tok")
            kv = {k: v for k, v in gguf_kv.items()}
            out["gguf_tokenizer"] = dict(model=kv.get("tokenizer.ggml.model"), pre=kv.get("tokenizer.ggml.pre"), chat_template=bool(kv.get("tokenizer.chat_template")),
                                          bos=kv.get("tokenizer.ggml.bos_token_id"), eos=kv.get("tokenizer.ggml.eos_token_id"), n_tokens=len(kv.get("tokenizer.ggml.tokens") or ()))
            if not kv.get("tokenizer.ggml.tokens"):
                out["verdict"], out["why"] = "TOKENIZER NEEDED", "no tokenizer.ggml.tokens in the file and no tokenizer.json next to it"
                return out
            try:
                N.write_gguf_tokenizer(kv, tmp)
            except (GF.GgufError, KeyError, ValueError) as e:
                out["verdict"], out["why"] = "TOKENIZER NEEDED", f"engine cannot build a tokenizer from this GGUF: {str(e)[:220]}"
                return out
            out["steps"].append("tokenizer.json built from the GGUF metadata by foreign_names.write_gguf_tokenizer")
            tdir = tmp
            for f in ("tokenizer_config.json",):          # files next to the GGUF win, as in gguf_serve_dir
                if model_dir and os.path.isfile(os.path.join(model_dir, f)):
                    import shutil
                    shutil.copyfile(os.path.join(model_dir, f), os.path.join(tmp, f))
        if not tdir or not (os.path.isfile(os.path.join(tdir, "tokenizer_config.json")) or os.path.isfile(os.path.join(tdir, "tokenizer.json"))):
            out["verdict"], out["why"] = "TOKENIZER NEEDED", "no tokenizer_config.json / tokenizer.json in the directory (the server takes --tokenizer DIR for this)"
            return out
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        try:
            if os.path.isfile(os.path.join(tdir, "tokenizer_config.json")):
                tok = AutoTokenizer.from_pretrained(tdir, trust_remote_code=False)
            else:
                tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tdir, "tokenizer.json"))
            ids = tok.encode("Hello, world", add_special_tokens=False)
            if not ids:
                raise ValueError("encode returned no ids")
            out["n_vocab"] = len(tok)
            out["eos"] = getattr(tok, "eos_token", None)
        except Exception as e:  # noqa: BLE001
            out["verdict"], out["why"] = "TOKENIZER NEEDED", f"tokenizer does not load ({type(e).__name__}: {str(e)[:160]})"
            return out
        vs = (cfg or {}).get("vocab_size") or ((cfg or {}).get("text_config") or {}).get("vocab_size")
        if vs and len(tok) > int(vs):
            out["notes"] = [f"tokenizer has {len(tok)} ids, config vocab_size {vs}"]
        # ---- chat template, rendered by the engine's family class
        from kyojin_mlx.serve import chat as C
        from kyojin_mlx.serve import server as S
        try:
            fam = family or S._family_name(model_dir or "", cfg or {})
            fcls = C.FAMILIES[fam]
        except Exception as e:  # noqa: BLE001
            out["verdict"], out["why"] = "TEMPLATE FAILS", f"no chat family for model_type {(cfg or {}).get('model_type')!r}: {type(e).__name__}: {str(e)[:120]}"
            return out
        out["family"] = fam
        try:
            fobj = fcls(tok, tdir, None, {})
            msgs = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": USER_MSG}]
            r1 = fobj.render({"messages": msgs, "tools": TOOLS})
            r0 = fobj.render({"messages": msgs})
        except Exception as e:  # noqa: BLE001
            out["verdict"], out["why"] = "TEMPLATE FAILS", f"{fam} family: {type(e).__name__}: {str(e)[:200]}"
            return out
        miss = [lab for lab, txt in (("system message", SYS_MSG), ("user message", USER_MSG)) if txt not in r0.prompt]
        if miss or not r0.prompt.strip():
            out["verdict"], out["why"] = "TEMPLATE FAILS", "rendered prompt lacks the " + ", ".join(miss or ["whole text"])
            return out
        out["prompt_chars"] = len(r0.prompt)
        out["tools_in_prompt"] = "get_weather" in r1.prompt
        out["thinking_open"] = bool(r0.thinking_first)
        out["verdict"] = "TOKENIZER OK"
        out["why"] = f"{fam} family renders system + user" + ("" if out["tools_in_prompt"] else "; the template ignores a tools list (get_weather not in the prompt)")
        return out
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------ vision / audio
def check_vision(kind: str, cfg: dict | None, names, side_files: list[str], dir_files: list[str], gguf_arch: str | None = None) -> dict:
    """'vision supported' only where the engine's tables carry it; every other case reads 'text only loads'."""
    cfg = cfg or {}
    tower = sorted({n.split(".")[0] if kind == "gguf" else VISION_NAME.search(n).group(2) for n in names if (VISION_NAME.search(n) if kind != "gguf" else n.startswith(("v.", "mm.")))})
    audio = sorted({"audio" for n in names if (AUDIO_NAME.search(n) if kind != "gguf" else n.startswith(("s.", "a.")))})
    pre = [f for f in dir_files if f in ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json")]
    has = bool(tower or side_files or cfg.get("vision_config") or pre)
    out = dict(vision_parts=dict(tower_tensors=tower, mmproj_files=side_files, config_vision=bool(cfg.get("vision_config")), preprocessor_files=pre, audio_tensors=bool(audio)))
    if not has:
        out["verdict"], out["why"] = "text only", "no vision tower, mmproj, vision_config or preprocessor file"
        return out
    from kyojin_mlx import modeldesc
    if kind == "gguf":
        out["verdict"] = "text only loads"
        out["why"] = "foreign_gguf skips mmproj*.gguf and foreign_linear installs no vision tower for a GGUF: images are not read (vision supported: NO)"
        return out
    row = modeldesc.by_architecture((cfg.get("architectures") or ["?"])[0]) or (modeldesc.backend_desc(cfg.get("model_type", "")) if cfg.get("model_type") else None)
    arch_vis = None if row is None else row.architectures.get((cfg.get("architectures") or ["?"])[0])
    if arch_vis:
        out["verdict"] = "vision supported (per modeldesc table)"
        out["why"] = f"modeldesc row {row.name} marks {cfg['architectures'][0]} as vision; end-to-end image request NOT run by this tool (UNVERIFIED)"
    elif row is not None and row.backend == "generic" and cfg.get("vision_config"):
        out["verdict"] = "text only loads"
        out["why"] = "generic backend: image input only with flag KYOJIN_VISIONGEN=1 (visiongen, default off, needs an mlx-vlm class); not claimed"
    else:
        out["verdict"] = "text only loads"
        out["why"] = "vision files present but the engine's tables do not mark this model as vision (no modeldesc vision row; visiongen is only for the generic backend)"
    return out


# ------------------------------------------------------------------------------------------------------------------------------ ceiling terms
def kv_term_pack(cfg: dict, ctx: int) -> dict:
    """Bytes of KV the attention reads per decode step at 1 session and context ctx (fp16 K and V), from config.json. Counted: the full-attention layers (all ctx
    tokens) and sliding-window layers (min(ctx, window)). Not counted: linear-attention / SSM state, quantized KV pools, indexer / compressed-sparse reads."""
    t = cfg.get("text_config") or cfg
    try:
        n = int(t["num_hidden_layers"])
        heads = int(t["num_attention_heads"])
        hidden = int(t["hidden_size"])
    except (KeyError, TypeError, ValueError):
        return dict(error="config lacks num_hidden_layers / num_attention_heads / hidden_size: UNVERIFIED")
    if t.get("index_n_heads") or t.get("compress_ratios") or t.get("index_head_dim"):
        return dict(error="sparse / compressed attention (indexer or compress_ratios in the config): the read per step is not derivable from the header, term left out: UNVERIFIED")
    lt = t.get("layer_types")
    if lt:
        full = sum(1 for x in lt if x in ("full_attention", "attention"))
        slide = sum(1 for x in lt if x == "sliding_attention")
    elif t.get("full_attention_interval"):
        full, slide = n // int(t["full_attention_interval"]), 0
    else:
        full, slide = n, 0
    if t.get("kv_lora_rank"):
        per_tok = (int(t["kv_lora_rank"]) + int(t.get("qk_rope_head_dim") or 0)) * 2
        shape = f"MLA latent {t['kv_lora_rank']}+{t.get('qk_rope_head_dim') or 0} fp16"
    else:
        kvh = int(t.get("num_key_value_heads") or heads)
        hd = int(t.get("head_dim") or hidden // heads)
        per_tok = 2 * kvh * hd * 2
        shape = f"{kvh} kv heads x {hd} dims, K+V fp16"
    win = int(t.get("sliding_window") or 0)
    toks = full * ctx + (slide * min(ctx, win) if win else slide * ctx)
    return dict(bytes=toks * per_tok, per_token_per_layer=per_tok, full_layers=full, sliding_layers=slide, shape=shape, ctx=ctx)


def kv_term_gguf(tensors: dict, kv: dict, arch: str, ctx: int) -> dict:
    """Same, from the GGUF tensor table: per block that has attn_k (+ attn_v) the output width of those projections is the K / V width per token (fp16 cache)."""
    per_layer: dict = {}
    for n, ti in tensors.items():
        m = re.match(r"blk\.(\d+)\.(attn_k|attn_v|attn_kv_a_mqa)\.weight$", n)
        if m and len(ti.ne) == 2:
            per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + int(ti.ne[1])
    if not per_layer:
        return dict(error="no blk.N.attn_k / attn_v / attn_kv_a_mqa tensors: attention shape not derivable: UNVERIFIED")
    if any(re.match(r"blk\.\d+\.(indexer|attn_indexer)", n) for n in tensors):
        return dict(error="indexer tensors present (sparse attention): term left out: UNVERIFIED")
    win = int(kv.get(f"{arch}.attention.sliding_window", 0) or 0)
    total = sum(w * 2 * ctx for w in per_layer.values())
    out = dict(bytes=total, full_layers=len(per_layer), shape="widths of attn_k + attn_v (or MLA latent) of each attention block, fp16", ctx=ctx)
    if win:
        out["note"] = f"{arch}.attention.sliding_window={win} is in the file but which blocks use it is not read: every attention block counted at full ctx (upper bound)"
    return out


def roofline_pack(info: dict, cfg: dict, bw_gbs: float, ctx: int | None) -> dict:
    """Header-only decode ceiling of a safetensors pack at 1 session. Counted: trunk (all tensors once, shared experts included), routed experts x topk / experts,
    one embedding row, the head in full (tied embedding when there is no head tensor), KV read when ctx is given. Draft head bytes (mtp) reported apart. Left out: vision, audio."""
    t = cfg.get("text_config") or cfg
    def cnt(v):
        """scalar or per-layer array -> (min, max, list or None)"""
        if isinstance(v, (list, tuple)):
            vals = [int(x or 0) for x in v]
            return (min(vals), max(vals), vals) if vals else (0, 0, None)
        return int(v or 0), int(v or 0), None

    ne_lo, ne, ne_vals = cnt(t.get("n_routed_experts") or t.get("num_experts") or t.get("num_local_experts") or t.get("moe_num_experts"))
    u_lo, used, u_vals = cnt(t.get("num_experts_per_tok") or t.get("moe_k") or t.get("top_k"))
    vocab = int(t.get("vocab_size") or 0)
    parts = dict(trunk=0.0, routed_experts=0.0, embedding_row=0.0, head=0.0)
    mtp = 0
    emb_bytes = 0
    has_head = False
    experts_seen = False
    routed_frac, routed_zero = 0.0, False
    for n, (dt, sh, nb) in info.items():
        r = role_of(n, sh, vocab)
        if r in ("vision", "audio"):
            continue
        if r == "mtp":
            mtp += nb
        elif r == "embeddings":
            emb_bytes += nb
        elif r == "head":
            parts["head"] += nb
            has_head = True
        elif r == "routed experts":
            experts_seen = True
            lm = re.search(r"(?:layers|blocks|h)\.(\d+)\.", n)
            li = int(lm.group(1)) if lm else -1
            e_l = ne_vals[li] if ne_vals is not None and 0 <= li < len(ne_vals) else ne
            u_l = u_vals[li] if u_vals is not None and 0 <= li < len(u_vals) else used      # per-layer array: this layer's entry; else the max (lower ceiling, never higher)
            if e_l and u_l:
                routed_frac += nb * u_l / e_l
            else:
                routed_zero = True
            parts["routed_experts"] += nb
        else:
            parts["trunk"] += nb
    if experts_seen:
        if not (ne and used) or routed_zero:
            return dict(error="routed-expert tensors present but expert count / experts per token not in the config (or 0 for a layer that holds them): UNVERIFIED")
        parts["routed_experts"] = routed_frac
    if emb_bytes:
        if has_head:
            parts["embedding_row"] = emb_bytes / vocab if vocab else None
            if parts["embedding_row"] is None:
                return dict(error="vocab_size missing from config: embedding row not derivable: UNVERIFIED")
        else:
            parts["head"] += emb_bytes
    total = sum(parts.values())
    res = dict(bytes_per_step=total, parts=parts, draft_head_bytes=mtp, experts_used=used, experts_used_min=u_lo, experts_total=ne, experts_total_min=ne_lo, per_layer_experts=bool(u_vals is not None or ne_vals is not None), tied_head=not has_head, bw_gbs=bw_gbs,
               ceiling_tok_s=(bw_gbs * 1e9 / total) if total else None)
    add_kv(res, kv_term_pack(cfg, ctx) if ctx else None, ctx, bw_gbs)
    return res


def add_kv(res: dict, kvt: dict | None, ctx: int | None, bw_gbs: float) -> None:
    """Attach the KV term and the lists of counted / uncounted terms to a roofline dict."""
    res["ctx"] = ctx
    res["kv"] = kvt
    counted = ["trunk weights once", "routed experts x used / total", "one embedding row", "head in full"]
    not_counted = ["activations traffic", "vision / audio towers", "linear-attention / SSM state", "quantized-KV pool (KV counted as fp16)", "attention FLOPs (read bytes only)"]
    if kvt and "bytes" in kvt:
        counted.append(f"KV read at ctx {ctx} (fp16 K+V)")
        res["bytes_per_step_with_kv"] = res["bytes_per_step"] + kvt["bytes"]
        res["ceiling_with_kv_tok_s"] = bw_gbs * 1e9 / res["bytes_per_step_with_kv"]
    elif ctx:
        not_counted.append(f"KV read at ctx {ctx}: {kvt['error'] if kvt else 'unknown'}")
    else:
        not_counted.append("KV read (no --ctx given)")
    if res.get("draft_head_bytes"):
        counted.append("draft head declared: bytes per draft call reported apart (not added to the plain-decode step)")
        res["draft_call_ceiling_per_s"] = bw_gbs * 1e9 / res["draft_head_bytes"]
    else:
        not_counted.append("draft head (none declared in the header)")
    res["terms_counted"], res["terms_not_counted"] = counted, not_counted


# ------------------------------------------------------------------------------------------------------------------------------ serve command
def serve_command(path: str, port: int, ctx: int | None, backend: str | None, verdict: str, tok: dict | None, tokenizer_dir: str | None = None) -> dict:
    """The command that serves `path` on this machine, printed not run. The flags are validated with the server's own parser; pack-derived values (sessions, ctx,
    chunk, MTP depth ...) are NOT put on the line: the server derives them (autoconfig), they are listed under `derived` when the pack has a config.json."""
    from kyojin_mlx.serve import server as S
    real = os.path.realpath(os.path.expanduser(path))
    argv = ["--model", real, "--port", str(port)]
    if ctx:
        argv += ["--ctx", str(ctx)]
    if tokenizer_dir:
        argv += ["--tokenizer", tokenizer_dir]
    S.build_parser().parse_args(argv)                                      # the server's own parser must accept exactly this
    unknown = []
    launcher = f"{sys.executable} -m kyojin_mlx.serve"
    prefix = f"PYTHONPATH={REPO} "
    if backend == "vlm":
        vlm = next((d for d in (os.environ.get("KYOJIN_VLM_PKG", ""), os.path.join(REPO, "vlm_pkg")) if d and os.path.isdir(os.path.join(d, "mlx_vlm"))), None)
        if vlm:
            prefix = f"PYTHONPATH={REPO}:{vlm} "
        else:
            unknown.append("vlm_pkg (mlx-vlm 0.7.10 directory: tools/mac_serve.sh looks for ./vlm_pkg or KYOJIN_VLM_PKG; not found here): UNKNOWN")
    if tok is not None and tok.get("verdict") == "TOKENIZER NEEDED":
        unknown.append("--tokenizer DIR (a directory with tokenizer.json for this model; none found next to it): UNKNOWN")
    res = dict(command=prefix + launcher + " " + " ".join(argv), unknown=unknown, derived=None, runnable=verdict.startswith("READY") and not unknown and (tok is None or tok.get("verdict") == "TOKENIZER OK"))
    d = real if os.path.isdir(real) else os.path.dirname(real)
    if os.path.isfile(os.path.join(d, "config.json")):
        try:
            from kyojin_mlx.serve import autoconfig as A
            a = S.build_parser().parse_args(argv + ["--print-config"])
            a.model = a.model_opt
            res["derived"] = A.effective_config(A.resolve_serve_args(a))
        except Exception as e:  # noqa: BLE001
            res["derived"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
    if not res["runnable"]:
        res["note"] = "the line above is what the server would be given, but the verdict is not READY / TOKENIZER OK or a piece is UNKNOWN: not claimed to serve"
    return res
