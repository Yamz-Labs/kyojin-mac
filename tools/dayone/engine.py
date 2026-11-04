# SPDX-License-Identifier: Apache-2.0
"""dayone engine: a config.json shape -> tiny random-weight pack on CPU -> every phase a served model goes through.

No family is named here. The model code is found by the config's model_type (mlx_lm.models.<mt>, else mlx_vlm.models.<mt>), the cache comes from
the model's own make_cache (else mlx-lm's default), the loader is the stock one of that package. A phase is a function (ctx) -> detail string; it
raises on failure and the runner records `pass` / `FAIL: ExcType: first line` / `n/a: reason`. Random weights: nothing here measures real numerics."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx
from mlx.utils import tree_flatten

from kyojin_mlx import conformance as C, synth

PHASES = ("load", "entry", "prefill", "decode", "snapshot", "disk", "sessions2", "tools", "vision", "draft")
FLAG_ENVS = ("KYOJIN_GENERIC_CACHE", "KYOJIN_CHAT_FALLBACK", "KYOJIN_BACKEND_FALLBACK")
TEMPLATE = ("{%- if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>{% endif %}"
            "{%- for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}"
            "{%- for c in m.get('tool_calls', []) %}<tool_call><function={{ c['function']['name'] }}>"
            "{%- for k, v in c['function']['arguments'].items() %}<parameter={{ k }}>{{ v }}</parameter>{% endfor %}</function></tool_call>{% endfor %}"
            "{%- endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")


@contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def code_of(cfg: dict):
    """(package, module, config class) of the model code for cfg['model_type']: mlx-lm first, mlx-vlm second."""
    mt = cfg["model_type"]
    order = (("mlx_vlm", "ModelConfig"), ("mlx_lm", "ModelArgs")) if "vision_config" in cfg else (("mlx_lm", "ModelArgs"), ("mlx_vlm", "ModelConfig"))
    for pkg, cls in order:
        try:
            mod = importlib.import_module(f"{pkg}.models.{mt}")
        except ModuleNotFoundError:
            continue
        return pkg, mod, getattr(mod, cls)
    raise LookupError(f"no model code for model_type {mt!r} in mlx_lm / mlx_vlm")


def model_config(pkg: str, mod, Cfg, cfg: dict):
    """Cfg.from_dict, and for an mlx-vlm class the sub-configs (text / vision) the way mlx_vlm.utils.load_model fills them (llava-style classes need them)."""
    mc = Cfg.from_dict(cfg)
    if pkg == "mlx_vlm":
        from mlx_vlm.utils import update_module_configs
        mc = update_module_configs(mc, mod, cfg, ["text", "vision", "perceiver", "projector", "audio"])
    return mc


def build(cfg: dict, seed: int = 0):
    """(model, lm, make_cache) with random float32 weights. The vlm wrappers carry a default vision tower; only the text trunk is initialised when
    the config has no vision_config of its own."""
    pkg, mod, Cfg = code_of(cfg)
    model = mod.Model(model_config(pkg, mod, Cfg, cfg))
    lm = getattr(model, "language_model", model)
    target = model if "vision_config" in cfg else lm
    target.update(synth._init(target.parameters(), seed))
    mx.eval(target.parameters())
    if hasattr(lm, "make_cache"):
        mk = lm.make_cache
    else:
        from mlx_lm.models.cache import make_prompt_cache
        mk = lambda: make_prompt_cache(lm)                                       # noqa: E731
    return model, lm, mk


def write_pack(cfg: dict, path: str, seed: int = 0):
    os.makedirs(path, exist_ok=True)
    model, lm, mk = build(cfg, seed)
    flat = {k: v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(os.path.join(path, "model.safetensors"), flat, {"format": "mlx"})
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    vocab = int(text_cfg(cfg)["vocab_size"])
    stops = ("<|endoftext|>",)
    tok = {f"t{i}": i for i in range(vocab - len(stops))}
    json.dump({"version": "1.0", "pre_tokenizer": {"type": "WhitespaceSplit"}, "added_tokens": [{"id": len(tok) + i, "content": t, "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False} for i, t in enumerate(stops)],
               "model": {"type": "WordLevel", "vocab": tok, "unk_token": "t0"}}, open(os.path.join(path, "tokenizer.json"), "w"))
    json.dump({}, open(os.path.join(path, "tokenizer_config.json"), "w"))
    open(os.path.join(path, "chat_template.jinja"), "w").write(TEMPLATE)
    return model, lm, mk


def load_pack(cfg: dict, path: str):
    """The stock loader of the package that owns the model code (strict: a missing or extra tensor raises)."""
    pkg, mod, _ = code_of(cfg)
    if pkg == "mlx_lm":
        from pathlib import Path
        from mlx_lm.utils import load_model
        model, _ = load_model(Path(path), strict=True)
    else:
        from pathlib import Path
        from mlx_vlm.utils import load_model
        model = load_model(Path(path), lazy=False, strict=True) if "strict" in load_model.__code__.co_varnames else load_model(Path(path), lazy=False)
    mx.eval(model.parameters())
    lm = getattr(model, "language_model", model)
    if hasattr(lm, "make_cache"):
        mk = lm.make_cache
    else:
        from mlx_lm.models.cache import make_prompt_cache
        mk = lambda: make_prompt_cache(lm)                                       # noqa: E731
    return model, lm, mk


@dataclass
class Ctx:
    name: str
    cfg: dict
    path: str
    vocab: int
    ref: tuple = None
    loaded: tuple = None
    notes: dict = field(default_factory=dict)

    @property
    def lm(self):
        return self.loaded[1]

    @property
    def mk(self):
        return self.loaded[2]


# ------------------------------------------------------------------------------------------------------------------ phases
def p_entry(x: Ctx):
    """The serve entry for this config: autoconfig.read_pack, the modeldesc row, the chat family, then serve.backends.load_backend and the Backend API the
    engine drives (make_cache, chunked prefill_chunk, fork, a 2-row batched step against the solo run)."""
    from kyojin_mlx import modeldesc
    from kyojin_mlx.serve import autoconfig, backends
    mt = x.cfg["model_type"]
    pack = autoconfig.read_pack(x.path)
    row = modeldesc.backend_desc(mt)
    if row is not None:
        return f"modeldesc row {row.name}, chat {pack.family}, backend {row.backend} (described: the Metal-bound backend load is not run on CPU)"
    missing = ["chat family"] * (pack.family is None) + ["modeldesc row"]
    try:
        be, _ = backends.load_backend(x.path, log=lambda *a: None)
    except Exception as e:
        raise RuntimeError(f"no {' + '.join(missing)} for model_type {mt!r}; load_backend: {type(e).__name__}: {str(e).splitlines()[0][:60]}") from None
    ids = [C.seq(21, x.vocab, seed=31), C.seq(25, x.vocab, seed=32)]
    caches, firsts = [], []
    for p in ids:
        c = be.make_cache()
        for s in range(0, len(p), 8):
            lg = be.prefill_chunk(c, p[s:s + 8])
        caches.append(c)
        firsts.append(int(mx.argmax(lg[0]).item()))
    solo = [int(mx.argmax(C.decode(x.lm, be.fork(c), t, 1)[0][1])) for c, t in zip(caches, firsts)]
    bc = be.batch([be.fork(c) for c in caches])
    got = [int(t) for t in mx.argmax(be.step(bc, mx.array(firsts)), axis=-1).tolist()]
    if got != solo:
        raise AssertionError(f"backend 2-row step {got} != solo {solo}")
    return f"no row, chat {pack.family}; {type(be).__name__}: make_cache, chunked prefill, fork, 2-row step == solo"


def p_load(x: Ctx):
    x.ref = write_pack(x.cfg, x.path)
    x.loaded = load_pack(x.cfg, x.path)
    probe = C.seq(9, x.vocab, seed=5)
    d = C.maxdiff(C.forward(x.lm, x.mk, probe)[1], C.forward(x.ref[1], x.ref[2], probe)[1])
    if d != 0.0:
        raise AssertionError(f"loaded model differs from the one the pack was written from: {d}")
    return f"{sum(v.size for _, v in tree_flatten(x.loaded[0].parameters()))} params, load diff 0"


def p_prefill(x: Ctx):
    ids = C.seq(37, x.vocab)
    ref = C.reference_last(x.lm, x.mk, ids)
    _, lg = C.prefill(x.lm, x.mk, ids, 8)
    d = C.maxdiff(lg, ref)
    if d > C.TOL or int(mx.argmax(lg)) != int(mx.argmax(ref)):
        raise AssertionError(f"chunked prefill vs one-shot: {d:.2e}")
    return f"chunks of 8 vs one-shot {d:.1e}"


def p_decode(x: Ctx):
    r = C.check_prefill_decode(x.lm, x.mk, x.vocab)
    if r["decode_vs_ref"] > C.TOL or not r["decode_argmax_equal"]:
        raise AssertionError(f"decode vs one-shot: {r}")
    return f"6 steps vs one-shot {r['decode_vs_ref']:.1e}"


def _fork():
    from kyojin_mlx import multiuser
    return multiuser.fork_cache


def p_snapshot(x: Ctx):
    s = C.check_snapshot(x.lm, x.mk, _fork(), x.vocab)
    if not all(s.values()):
        raise AssertionError(f"fork not bitwise: {s}")
    return "2 forks bitwise"


def p_disk(x: Ctx):
    k = C.check_disk(x.lm, x.mk, x.vocab, {"arch": x.name, "pack": os.path.basename(x.path)})
    if not (k["disk_hit_at"] > 0 and k["disk_bitwise"]):
        raise AssertionError(f"disk round trip: {k}")
    return f"{k['disk_saved_bytes']} B saved, hit at {k['disk_hit_at']}, bitwise"


def p_sessions2(x: Ctx):
    """N=2: two prompts, caches advanced in turn, one token each per round. Each session bitwise equals its solo run."""
    a_ids, b_ids = C.seq(29, x.vocab, seed=11), C.seq(33, x.vocab, seed=12)

    def solo(ids):
        cache, lg = C.prefill(x.lm, x.mk, ids, 8)
        return [t for t, _ in C.decode(x.lm, cache, int(mx.argmax(lg)), 5)]
    sa, sb = solo(a_ids), solo(b_ids)
    ca, la = C.prefill(x.lm, x.mk, a_ids, 8)
    cb, lb = C.prefill(x.lm, x.mk, b_ids, 8)
    ta, tb = [int(mx.argmax(la))], [int(mx.argmax(lb))]
    for _ in range(4):
        ta.append(int(mx.argmax(C.decode(x.lm, ca, ta[-1], 1)[0][1])))
        tb.append(int(mx.argmax(C.decode(x.lm, cb, tb[-1], 1)[0][1])))
    if (ta[:5], tb[:5]) != (sa[:5], sb[:5]):
        raise AssertionError("interleaved sessions differ from solo runs")
    from kyojin_mlx import multiuser
    note = "interleaved == solo"
    try:                                                                         # the one-forward-for-N path (multiuser.batch_caches)
        multiuser.install(x.loaded[0]); multiuser.install(x.lm)                                           # the per-user attention hook the served path installs (routes the model module's sdpa)
        want = [int(mx.argmax(C.decode(x.lm, multiuser.fork_cache(c), t[-1], 1)[0][1])) for c, t in ((ca, ta), (cb, tb))]
        bc = multiuser.batch_caches([multiuser.fork_cache(ca), multiuser.fork_cache(cb)])
        out = multiuser.decode_batched(x.loaded[0], bc, mx.array([ta[-1], tb[-1]]), 2, async_=False)
        got = [int(o[1]) for o in out]
        note += f"; batch_caches one-forward pass: tokens {'equal' if got == want else 'DIFFER'} to solo ({got} vs {want})"
    except Exception as e:                                                       # reported, not a failure of the phase: the interleaved run is the contract
        note += f"; batch_caches/decode_batched: {type(e).__name__}: {str(e).splitlines()[0][:70]}"
    return note


def p_tools(x: Ctx):
    """Chat family for the config, template render with tools, a CANNED <tool_call> string parsed. Random weights never emit a call."""
    from kyojin_mlx.serve import chat
    mt = x.cfg["model_type"]
    fam = chat.family_of(mt)
    tools = [{"type": "function", "function": {"name": "get_w", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}}]
    cls = chat.FAMILIES[fam]
    from transformers import AutoTokenizer
    tokz = None
    try:
        tokz = AutoTokenizer.from_pretrained(x.path)
    except Exception:
        pass
    f = cls(tokz, x.path)
    r = f.render({"messages": [{"role": "user", "content": "hi"}], "tools": tools})
    if "get_w" not in r.prompt:
        raise AssertionError("tools are not in the rendered prompt")
    calls = chat.parse_tool_calls("<tool_call><function=get_w><parameter=n>3</parameter></function></tool_call>", tools)
    if not calls or json.loads(calls[0]["function"]["arguments"]) != {"n": 3}:
        raise AssertionError(f"canned call not parsed: {calls}")
    return f"family {fam}: render + canned call (n=3, coerced int)"


def p_vision(x: Ctx):
    """Only when the config has a vision tower: image rows from the tower replace the image-token embeddings, the text trunk runs on them. The input
    layout (patch rows, grid) is derived from the tower's own config fields; a tower that does not publish them is reported, not guessed."""
    vc = x.cfg.get("vision_config")
    if vc is None:
        raise NotImplementedError("no vision_config")
    model, lm = x.loaded[0], x.lm
    if not hasattr(model, "get_input_embeddings"):
        raise AssertionError("model class has no get_input_embeddings (the loader built a text-only class for a config with vision_config)")
    need = ("in_channels", "patch_size", "temporal_patch_size", "spatial_merge_size")
    if any(k not in vc for k in need):
        raise AssertionError(f"tower config lacks {[k for k in need if k not in vc]}: input layout unknown")
    m, p = int(vc["spatial_merge_size"]), int(vc["patch_size"])
    patches = (2 * m) ** 2
    n_img = patches // (m * m)
    tok = int(x.cfg.get("image_token_id", x.cfg.get("image_token_index")))
    ids = [3, 4] + [tok] * n_img + [5, 6]
    px = mx.random.normal((patches, int(vc["in_channels"]) * int(vc["temporal_patch_size"]) * p * p))
    f = model.get_input_embeddings(mx.array([ids]), px, image_grid_thw=mx.array([[1, 2 * m, 2 * m]]))
    kw = {"inputs_embeds": f.inputs_embeds}
    if getattr(f, "position_ids", None) is not None:
        kw["position_ids"] = f.position_ids
    lg = C.last_row(lm(mx.array([ids]), cache=x.mk(), **kw))
    base = C.last_row(lm(mx.array([ids]), cache=x.mk()))
    mx.eval(lg, base)
    if not bool(mx.all(mx.isfinite(lg)).item()):
        raise AssertionError("non-finite logits with an image")
    d = C.maxdiff(lg, base)
    if d == 0.0:
        raise AssertionError("image rows did not reach the trunk (logits equal the text-only ones)")
    return f"{n_img} image rows through the tower into the trunk, logits finite, differ from text-only by {d:.2f}"


def p_draft(x: Ctx):
    """Draft head: (1) the verify-k contract any head needs from the trunk (k rows in one forward on a cache == k sequential steps, the cache rolled back by a
    fork taken before); (2) whether the config declares a head, and whether Kyojin has a loader for this model_type."""
    ids = C.seq(20, x.vocab, seed=21)
    k_toks = C.seq(4, x.vocab, seed=22)
    cache, _ = C.prefill(x.lm, x.mk, ids, 8)
    fork = _fork()
    seq_cache = fork(cache)
    rows = []
    for t in k_toks:
        rows.append(C.last_row(x.lm(mx.array([[t]]), cache=seq_cache)))
    out = getattr(x.lm(mx.array([k_toks]), cache=fork(cache)), "logits", None)
    out = x.lm(mx.array([k_toks]), cache=fork(cache))
    out = getattr(out, "logits", out)[0].astype(mx.float32)
    mx.eval(out, *rows)
    d = max(C.maxdiff(out[i], rows[i]) for i in range(len(rows)))
    contract = f"verify-4 vs 4 steps {d:.1e}"
    t = text_cfg(x.cfg)
    keys = [k for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", "dspark_block_size", "mtp_depth") if t.get(k) or x.cfg.get(k)]
    if not keys:
        raise NotImplementedError(f"no draft head in config ({contract}, {'ok' if d <= C.TOL else 'FAILS'})")
    if d > C.TOL:
        raise AssertionError(f"verify-k contract: {d:.2e}")
    if os.environ.get("KYOJIN_MTPGEN", "0") != "0":                       # flag mtpgen: the generic loader decides (a dayone pack carries no head tensors, so this reports its refusal reason)
        from kyojin_mlx import mtpgen
        try:
            mtpgen.load_head(x.loaded[0], x.path, x.cfg)
        except mtpgen.HeadRefused as e:
            return f"config declares {keys}; {contract}; mtpgen refuses: {e}"
        return f"config declares {keys}; {contract}; mtpgen loaded the head"
    from kyojin_mlx import modeldesc
    row = modeldesc.backend_desc(x.cfg["model_type"])
    if row is None or not (row.mtp_auto or row.mtp_depth):
        raise AssertionError(f"config declares {keys}; {contract}; Kyojin has no draft-head loader for model_type {x.cfg['model_type']!r}")
    return f"config declares {keys}; {contract}; loader of row {row.name} not run (the synthetic pack carries no mtp.* tensors)"


RUN = {"entry": p_entry, "load": p_load, "prefill": p_prefill, "decode": p_decode, "snapshot": p_snapshot, "disk": p_disk,
       "sessions2": p_sessions2, "tools": p_tools, "vision": p_vision, "draft": p_draft}


def run(name: str, cfg: dict, flags: str = "off", workdir: str | None = None) -> dict:
    """Run every phase for one shape; flags 'off' = every new flag at 0 (stock behaviour), 'on' = KYOJIN_GENERIC_CACHE=1, KYOJIN_CHAT_FALLBACK=1.
    Returns {phase: (status, detail)}; status in pass / FAIL / n/a. A phase after a failed `load` is skipped."""
    d = workdir or tempfile.mkdtemp(prefix=f"dayone_{name}_")
    x = Ctx(name, cfg, d, int(text_cfg(cfg)["vocab_size"]))
    res: dict = {}
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        with env(KYOJIN_WIRED_MB="0", **{k: "1" if flags == "on" else "0" for k in FLAG_ENVS}):         # WIRED_MB=0: no MLX residency set (wire_memory reads a Metal-only device field)
            os.makedirs(d, exist_ok=True)
            json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
            for ph in PHASES:
                if ph != "load" and x.loaded is None:
                    res[ph] = ("FAIL", "skipped: load failed")
                    continue
                try:
                    res[ph] = ("pass", RUN[ph](x))
                except NotImplementedError as e:
                    res[ph] = ("n/a", str(e))
                except Exception as e:
                    res[ph] = ("FAIL", f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}")
                    if os.environ.get("DAYONE_TRACE"):
                        traceback.print_exc()
    finally:
        mx.set_default_device(prev)
        if workdir is None:
            shutil.rmtree(d, ignore_errors=True)
    return res
