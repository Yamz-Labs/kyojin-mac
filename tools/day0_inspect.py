#!/usr/bin/env python3
"""Day-0 readiness report of a GGUF for the Mac engine. Reads ONLY the GGUF header (metadata + tensor table) of the file, or of every shard of a directory;
no weight byte is touched, no GPU is used (mlx is pinned to the CPU device for the census), so a multi-hundred-GB file is answered in seconds.

usage: python tools/day0_inspect.py FILE.gguf | FIRST_SHARD.gguf | DIR_OF_SHARDS | SAFETENSORS_DIR [--json] [--port 8391] [--limit-gib 236] [--no-census] [--roofline [--bw-gbs GBS] [--ctx N]] [--no-tokenizer] [--serve-cmd]
Safetensors packs, tokenizer / chat template, vision files, KV terms and the serve command live in tools/day0_pack.py (same verdict words; TOKENIZER NEEDED / TEMPLATE FAILS replace a READY).

Verdict (one line, last): READY | ROW NEEDED (about N lines) | MODEL CODE NEEDED | TYPE NEEDED (a weight type id unknown to the reader) | DOES NOT FIT | SHARDS INVALID (inconsistent split set: numbering, duplicates, tensor total) |
NOT A TEXT LLM / ARCH KEY MISSING (no general.architecture key) | UNVERIFIED.
A complete split shard set (llama.cpp gguf-split) is read natively by foreign_gguf.GgufFile: no merge. A set that is merely incomplete on disk is never SHARDS INVALID and never READY: the
verdict says 'incomplete shard set: k of N present' (and 'verdict from shard 1 of N: weights not seen' when the shards present hold no tensor table).
Expert counts (expert_count / expert_used_count) may be scalars or per-layer arrays; the roofline uses each layer's own value and the report shows min..max.
--roofline: header-only arithmetic, bytes of weights one decode step reads at 1 session, and the memory-bandwidth ceiling tok/s = bandwidth / bytes (not a prediction).
Weight type modes (taken from kyojin_mlx.foreign_gguf / the load loop of foreign_linear.load_foreign, not from a document):
  stored        2-D / 3-D linear weight of a type in KERNEL_TYPES (Q4_0 Q4_K IQ4_NL MXFP4 NVFP4), KERNEL_TYPES8 (Q8_0) or KERNEL_TYPESK (Q5_K Q6_K IQ4_XS
                Q3_K IQ3_S IQ3_XXS Q2_K IQ2_* IQ1_*; row length % 256 == 0), and Q4_1 Q5_0 Q5_1 only when KYOJIN_LEGACY5_STORED=1 (default off until
                their Metal test has passed): read as stored by a Metal kernel (prefill above DENSE_ROWS rows decodes into a
                TRANSIENT fp16 buffer, never resident).
  resident16    everything else readable (norms, biases, F16 / BF16 / F32 tensors, types with no kernel): decoded to fp16 at load and kept: 2 bytes / weight.
  unsupported   type id unknown to the reader (verdict TYPE NEEDED, per tensor, the file is not refused), or not in DEQUANT.
The resident estimate = stored bytes + 2 bytes per weight of the resident16 tensors (F32 vectors keep 4); it leaves out the KV cache and activations."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from kyojin_mlx import foreign_gguf as GF  # noqa: E402  (numpy only)
import day0_pack as PK  # noqa: E402

GIB = 1 << 30
# architectures whose greedy identity on Metal was measured (ARCH_KIT_STATUS.md items 2-3); every other generated row is UNVERIFIED end to end
KIT_VERIFIED = ("olmo2", "gemma2", "granitemoe")
SIDE_PREFIX = (("mtp.", "MTP draft head"), ("v.", "vision tower"), ("mm.", "vision projector"), ("s.", "audio tower"), ("a.", "audio tower"))


def serve_blocker(arch, tensors) -> str | None:
    """Why foreign_linear.load_foreign would raise for a file of a hand-written row, or None: a tensor the row lists in Arch.unserved, or a model class the serve loader's lookup
    (foreign_vlm_classes.resolve, the one the loader calls) does not find for the row's model_type."""
    from kyojin_mlx import foreign_names as N, foreign_vlm_classes as VC
    A = N.ARCHS[arch]
    mapped = set(N.mapped_tables(arch, tensors))                          # flag KYOJIN_MAPPED_ROW_TABLE: those tables are read from the mapped file, not refused
    hit = sorted(n for n in tensors if n in A.unserved and n not in mapped)
    if hit:
        return f"serve loader refuses tensor {hit[0]}: no loader path serves it yet for {arch} (Arch.unserved)"
    try:
        VC.resolve([A.model_type], config={"model_type": A.model_type, "architectures": list(A.architectures)})      # the loader's config (gguf_config) carries both keys
    except BaseException as e:  # noqa: BLE001
        return f"serve loader finds no model class for model_type {A.model_type!r} ({type(e).__name__}: {str(e)[:100]})"
    return None


def shard_files(path: str) -> tuple[list[str], list[str]]:
    """(model shards, side files such as mmproj*.gguf)."""
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        return [p], []
    if not os.path.isdir(p):
        raise SystemExit(f"{path}: not a file or directory")
    fs = sorted(f for f in os.listdir(p) if f.endswith(".gguf") and not f.startswith("._"))
    side = [os.path.join(p, f) for f in fs if "mmproj" in f.lower()]
    return [os.path.join(p, f) for f in fs if "mmproj" not in f.lower()], side


def shard_state(files: list[str], hs: list[dict]):
    """(hs with the sibling shards present on disk added, None | dict(present=sorted shard numbers, total=N, dir, base)) for a split set; None when the files are not a split set."""
    first = next((h for h in hs if int(h["kv"].get("split.count", 1) or 1) > 1), None)
    if first is None:
        return hs, None
    n = int(first["kv"]["split.count"])
    m = GF._SHARD_RE.match(os.path.basename(first["file"]))
    if not m or int(m.group(3)) != n:
        return hs, None                                   # name does not follow the set pattern: left to the strict validation
    d, base = os.path.dirname(os.path.abspath(first["file"])), m.group(1)
    have = {os.path.abspath(h["file"]) for h in hs}
    sib = {}
    for f in sorted(os.listdir(d)):
        mm = GF._SHARD_RE.match(f)
        if mm and mm.group(1) == base and int(mm.group(3)) == n and 1 <= int(mm.group(2)) <= n:
            sib[int(mm.group(2))] = os.path.join(d, f)
    extra = [f for i, f in sorted(sib.items()) if os.path.abspath(f) not in have]
    if extra:
        hs = sorted(hs + read_headers(extra), key=lambda h: h["file"])
    return hs, dict(present=sorted(sib), total=n, dir=d, base=base)


def read_headers(files: list[str]):
    """Header of each file via the engine's own reader (mmap, nothing is paged in but the header)."""
    out = []
    for f in files:
        try:
            g = GF.GgufFile(f, _shard=True)
        except GF.GgufTypeError as e:
            out.append(dict(file=f, error=str(e), kv={}, tensors={}, size=os.path.getsize(f)))
            continue
        out.append(dict(file=f, kv=g.kv, tensors=dict(g.tensors), size=os.path.getsize(f), version=g.version))
        g.close()
    return out


def is_weight(name: str, ti) -> bool:
    """Linear-weight test (approximation of foreign_linear.is_linear, which needs the HF name): >= 2-D `.weight`, not a norm / embedding gate."""
    return name.endswith(".weight") and len(ti.ne) in (2, 3) and "norm" not in name


def mode_of(name: str, ti) -> str:
    t = ti.tname
    if t not in GF.DEQUANT:
        return "unsupported"
    if is_weight(name, ti):
        if t in GF.KERNEL_TYPES:
            return "stored"
        if t in GF.KERNEL_TYPES8:
            return "stored"
        if t in GF.KERNEL_TYPESK and ti.ne[0] % 256 == 0:
            return "stored"
        if t in GF.KERNEL_TYPES5 and GF.legacy5_stored():       # behind KYOJIN_LEGACY5_STORED (default off): not claimed before the Metal test passed
            return "stored"
        if t in GF.KERNEL_TYPES2 and GF.q2_0_stored():          # behind KYOJIN_Q2_0_STORED (default off): with it unset type id 42 is unknown (TYPE NEEDED)
            return "stored"
    return "resident16"


LABEL = {"stored": "read as stored (kernel)", "resident16": "resident fp16 copy", "unsupported": "unsupported"}


def _default_bw():
    """Sustained read bandwidth for this machine: the M5 Ultra figure is tools/opprof.py B_REF (BW_MEASURED.md); any other machine must pass --bw-gbs."""
    try:
        import subprocess
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        cpu = ""
    return 1159.0 if "M5 Ultra" in cpu else None


DEFAULT_BW_GBS = _default_bw()


def roofline(tensors: dict, kv: dict, arch: str | None, bw_gbs: float, ctx: int | None = None) -> dict:
    """Weight bytes ONE decode step reads at 1 session, from the tensor table only. Counted: every trunk tensor once (norms, biases and gates included); routed-expert tensors
    (`_exps`) x expert_used / expert_count; token_embd = ONE row (row bytes), unless the head is tied (no output.weight): then the full token_embd is the head; output.weight in full.
    ctx given: the KV read of the attention blocks at that context is added (day0_pack.kv_term_gguf). MTP / nextn bytes are reported apart (draft_head_bytes), not added.
    Left out: vision / audio towers, activations traffic, linear-attention state. tok/s ceiling = bw / bytes."""
    ne_lo, n_exp, ne_vals = GF.kv_counts(kv.get(f"{arch}.expert_count"))          # scalar or per-layer array
    u_lo, used, u_vals = GF.kv_counts(kv.get(f"{arch}.expert_used_count"))        # used = the max over layers (the scalar of a scalar key)

    def per_layer(name: str):
        """(experts, used) of the layer a routed tensor sits in: the per-layer array entry, else the scalar (a tensor outside blk.N with an array key counts the max: a lower ceiling, never a higher one)."""
        m2 = re.match(r"blk\.(\d+)\.", name)
        li = int(m2.group(1)) if m2 else -1
        e = ne_vals[li] if ne_vals is not None and 0 <= li < len(ne_vals) else n_exp
        u = u_vals[li] if u_vals is not None and 0 <= li < len(u_vals) else used
        return e, u
    nblk = int(kv.get(f"{arch}.block_count", 0) or 0)
    n_vocab = len(kv.get("tokenizer.ggml.tokens") or ()) or int(kv.get(f"{arch}.vocab_size", 0) or 0)
    nxt = int(kv.get(f"{arch}.nextn_predict_layers", 0) or 0)
    parts = dict(embedding_row=0, head=0, trunk=0, routed_experts=0)
    skipped = 0
    mtp_bytes = 0
    has_head = "output.weight" in tensors
    for n, ti in tensors.items():
        m = re.match(r"blk\.(\d+)\.", n)
        if (m and nblk and int(m.group(1)) >= nblk - nxt > 0) or re.search(r"nextn_predict|\.nextn\.", n) or any(n.startswith(p) for p, _ in SIDE_PREFIX):
            skipped += ti.nbytes
            if not any(n.startswith(p) for p, lab in SIDE_PREFIX if lab != "MTP draft head"):
                mtp_bytes += ti.nbytes
        elif n == "token_embd.weight":
            if has_head:
                parts["embedding_row"] += ti.nbytes // ti.ne[1]
            else:
                parts["head"] += ti.nbytes
        elif len(ti.ne) == 2 and n != "output.weight" and ((n.endswith("token_embd.weight")) or (n_vocab and ti.ne[1] == n_vocab and n.endswith(".weight") and "_exps" not in n)):
            parts["embedding_row"] += ti.nbytes // ti.ne[1]             # lookup table [features, vocab] (token_embd, per_layer_token_embd): a decode step reads one row, never the table
        elif n == "output.weight":
            parts["head"] += ti.nbytes
        elif "_exps" in n or (n_exp > 1 and len(ti.ne) == 3 and ti.ne[2] == n_exp):          # routed experts: by name, or by the expert-count dimension of a 3-D tensor
            e_l, u_l = per_layer(n)
            if not (e_l and u_l):
                raise ValueError(f"tensor {n} is a routed-expert tensor but {arch}.expert_count / expert_used_count are not in the metadata (or are 0 for its layer): UNVERIFIED")
            parts["routed_experts"] += ti.nbytes * u_l / e_l
        else:
            parts["trunk"] += ti.nbytes
    total = sum(parts.values())
    res = dict(bytes_per_step=total, parts=parts, skipped_bytes=skipped, draft_head_bytes=mtp_bytes, experts_used=used, experts_used_min=u_lo, experts_total=n_exp, experts_total_min=ne_lo, per_layer_experts=bool(u_vals is not None or ne_vals is not None), tied_head=not has_head, bw_gbs=bw_gbs,
               ceiling_tok_s=(bw_gbs * 1e9 / total) if total else None)
    PK.add_kv(res, PK.kv_term_gguf(tensors, kv, arch, ctx) if ctx else None, ctx, bw_gbs)
    return res


def census(arch: str):
    """Row of tools/archkit_census.py for this architecture (tiny config, CPU), or None."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import archkit_census as C
    rows = C.run({arch})
    return rows[0] if rows else None


def inspect(path: str, port: int = 8391, limit_gib: float = 236.0, use_census: bool = True, bw_gbs: float | None = None, ctx: int | None = None,
            tokenizer: bool = False, serve_cmd: bool = False) -> dict:
    t0 = time.time()
    files, side = shard_files(path)
    if not files:
        raise SystemExit(f"{path}: no .gguf file")
    hs = read_headers(files)
    shard_err = None
    hs, sstate = shard_state(files, hs)
    incomplete = sstate is not None and len(sstate["present"]) < sstate["total"]
    if incomplete:
        pass                                              # missing shards are not an inconsistency: no validation, no SHARDS INVALID; the verdict below says what was not seen
    elif len(hs) > 1 or (hs and int(hs[0]["kv"].get("split.count", 1) or 1) > 1):
        try:
            if int(hs[0]["kv"].get("split.count", 1) or 1) <= 1:
                raise GF.GgufError(f"{len(hs)} model .gguf files but {os.path.basename(files[0])} has no split.count > 1: not a split shard set")
            g = GF.GgufFile(files[0])        # validates the whole set (count, numbering, duplicates, tensor total); headers only
            ms = [g] + g._members
            hs = [dict(file=m.path, kv=m.kv, tensors={n: t for n, t in g.tensors.items() if g._owner.get(n, g) is m}, size=os.path.getsize(m.path)) for m in ms]
            g.close()
        except GF.GgufError as e:
            shard_err = str(e)
    rep: dict = dict(path=path, files=[os.path.basename(h["file"]) for h in hs], side_files=[os.path.basename(s) for s in side], blockers=[], notes=[])
    bad = [h for h in hs if h.get("error")]
    if shard_err:
        rep["blockers"].append("shard set refused: " + shard_err)
    if bad:
        rep["blockers"].append(f"reader error: {bad[0]['error']}")
        rep["unsupported_types"] = True
    kv = {}
    for h in hs:
        for k, v in h["kv"].items():
            kv.setdefault(k, v)
    arch = kv.get("general.architecture")
    rep["architecture"] = arch
    rep["name"] = kv.get("general.name")
    rep["file_bytes"] = sum(h["size"] for h in hs)
    split = [int(kv[k]) for k in ("split.count",) if k in kv]
    rep["shards"] = len(hs)
    rep["incomplete_set"] = None
    if incomplete:
        rep["incomplete_set"] = dict(present=len(sstate["present"]), total=sstate["total"], shard1_present=1 in sstate["present"], tensors_seen=sum(len(h["tensors"]) for h in hs))
        rep["notes"].append(f"incomplete shard set: {len(sstate['present'])} of {sstate['total']} present on disk; the other shard(s) hold tensors that were not seen")
    if len(hs) > 1 or (split and split[0] > 1):
        rep["notes"].append(f"split GGUF, {len(hs)} shard file(s), split.count={split[0] if split else '?'}: read natively as one logical file (no merge); point serve at the first shard or the directory")
    tensors = {}
    for h in hs:
        for n, ti in h["tensors"].items():
            tensors[n] = ti
    rep["n_tensors"] = len(tensors)
    # ---- weight types
    types: dict = {}
    resident = 0
    side_bytes: dict = {}
    for n, ti in tensors.items():
        pre = next((lab for p, lab in SIDE_PREFIX if n.startswith(p)), None)
        if pre is None and re.search(r"\.nextn\.|nextn_predict", n):
            pre = "MTP draft head"
        if pre:
            side_bytes[pre] = side_bytes.get(pre, 0) + ti.nbytes
        m = mode_of(n, ti)
        e = types.setdefault((ti.tname, m), dict(type=ti.tname, mode=LABEL[m], tensors=0, bytes=0, weights=0))
        e["tensors"] += 1
        e["bytes"] += ti.nbytes
        nel = 1
        for d in ti.ne:
            nel *= d
        e["weights"] += nel
        if pre in ("vision tower", "vision projector", "audio tower"):
            continue                                                              # UNVERIFIED whether load_foreign reads them: left out of the estimate
        if m == "stored":
            resident += ti.nbytes
        elif m == "resident16":
            resident += ti.nbytes if (ti.tname == "F32" and len(ti.ne) == 1) else 2 * nel
    rep["types"] = sorted(types.values(), key=lambda e: -e["bytes"])
    unk: dict = {}
    for n, ti in tensors.items():
        if not ti.known:
            e = unk.setdefault(ti.type, dict(id=ti.type, tensors=0, bytes=0, names=[], upstream=(GF.UPSTREAM_ONLY.get(ti.type) or (None,))[0]))
            e["tensors"] += 1
            e["bytes"] += ti.nbytes
            e["names"].append(n)
    rep["unknown_type_ids"] = [dict(e, names=e["names"][:3]) for e in unk.values()]
    uns = [e for e in rep["types"] if e["mode"] == LABEL["unsupported"]]
    rep["unsupported_types"] = bool(uns) or rep.get("unsupported_types", False)
    rep["resident_est_bytes"] = resident
    rep["limit_bytes"] = int(limit_gib * GIB)
    rep["fits"] = resident <= rep["limit_bytes"] and rep["file_bytes"] <= rep["limit_bytes"] * 1.0 + 0
    rep["side_tensors"] = {k: v for k, v in side_bytes.items()}
    exp_lo, exp, exp_vals = GF.kv_counts(kv.get(f"{arch}.expert_count"))
    used_lo, used_hi, used_vals = GF.kv_counts(kv.get(f"{arch}.expert_used_count"))
    rep["moe"] = bool(exp) or any("_exps" in n for n in tensors)
    rep["expert_count"], rep["expert_count_min"] = exp, exp_lo
    rep["expert_used"], rep["expert_used_min"] = used_hi, used_lo          # a scalar key: min == max; a per-layer array key: the extremes over layers
    rep["expert_per_layer"] = used_vals is not None or exp_vals is not None
    rep["mtp_layers"] = int(kv.get(f"{arch}.nextn_predict_layers", 0) or 0)
    # ---- tables, rows, census
    from kyojin_mlx import foreign_archkit as K
    rep["in_tables"] = bool(arch) and K.have(arch)
    if rep["in_tables"]:
        rep["converter_flags"] = list(K.converter_flags(arch))
        rep["value_transforms"] = list(K.value_transforms(arch))
        rep["vetted_flags"] = list(K.vetted(arch, "flags"))
        rep["refusing_flags"] = [f for f in K.value_transforms(arch) if f not in K.vetted(arch, "flags")]
    from kyojin_mlx import foreign_names as N
    rep["hand_row"] = arch in N.ARCHS
    try:
        from kyojin_mlx import foreign_cfgrows as CR
        rep["cfg_row"] = bool(arch in N._CONFIG or CR.has_rows(arch)) if arch else False
    except Exception as e:  # noqa: BLE001
        rep["cfg_row"] = None
        rep["notes"].append(f"cfg row lookup failed: {type(e).__name__}")
    try:
        from kyojin_mlx import foreign_unproven as UP
        h = UP.hint(arch)
        if h:
            rep["notes"].append(h)         # the rows exist but are off (KYOJIN_KIT_UNPROVEN_ROWS): the verdict is the one without them
    except Exception as e:  # noqa: BLE001
        rep["notes"].append(f"unproven rows lookup failed: {type(e).__name__}")
    cen = None
    if rep["in_tables"] and use_census:
        try:
            cen = census(arch)
        except BaseException as e:  # noqa: BLE001   (SystemExit from an optional mlx-lm dependency included)
            rep["notes"].append(f"census failed: {type(e).__name__}: {str(e)[:100]}")
    rep["census"] = None if cen is None else dict(cls=cen["cls"], how=cen["how"], verdict=cen["verdict"], n_min=cen.get("n_min", 0), note=cen["note"][:300],
                                                  required_missing=cen["req"], flags=cen["flags"], config_missing=cen["miss"], hand=cen["hand"])
    rep["mlx_lm_class"] = None if cen is None or cen["cls"] == "-" else cen["cls"]
    # kit_config on the REAL metadata (no model is built): does the generated config row refuse this file?
    rep["config_refusal"] = None
    if rep["in_tables"] and not rep["hand_row"] and not rep["cfg_row"] and kv and rep["mlx_lm_class"] and not incomplete:         # a config row replaces kit_config (foreign_cfgrows.build)
        try:
            from kyojin_mlx import foreign_archkit_cpu as KC
            _, Args, _ = KC._classes(arch)
            import dataclasses
            emb = tensors.get("token_embd.weight")
            K.kit_config(arch, kv, {f.name for f in dataclasses.fields(Args)}, vocab=emb.ne[1] if emb else None, tied="output.weight" not in tensors, tensors=tuple(tensors))
        except BaseException as e:  # noqa: BLE001
            rep["config_refusal"] = f"{type(e).__name__}: {str(e)[:240]}"
    # ---- verdict
    rep["serve_blocker"] = serve_blocker(arch, tensors) if rep["hand_row"] else None          # a known reason the serve loader would raise for a hand row: never READY then
    rep["mapped_tables"] = N.mapped_tables(arch, tensors) if rep["hand_row"] else []
    rep["verified"] = bool(rep["hand_row"] or arch in KIT_VERIFIED)
    serve = f"python -m kyojin_mlx.serve --model {path} --port {port}"
    v, why = None, ""
    inc = rep["incomplete_set"]
    if not arch and inc and not inc["shard1_present"]:
        v, why = "UNVERIFIED", f"incomplete shard set: {inc['present']} of {inc['total']} present; shard 1 (the metadata) is missing, so no architecture is known"
    elif not arch:
        keys = sorted(kv)
        text_like = any(n == "token_embd.weight" or n.startswith("blk.") for n in tensors)
        found = f"{len(keys)} metadata key(s) found: " + (", ".join(keys[:12]) + (" ..." if len(keys) > 12 else "") if keys else "none") + f"; tensors {len(tensors)}, first: " + ", ".join(list(tensors)[:4])
        if tensors and not text_like:
            v, why = "NOT A TEXT LLM", f"no general.architecture key and no token_embd.weight / blk.N.* tensor (not the layout of a llama.cpp text model); {found}"
        else:
            v, why = "ARCH KEY MISSING", f"no general.architecture key, so no architecture row can be chosen; {found}"
    elif shard_err:
        v, why = "SHARDS INVALID", shard_err
    elif not rep["fits"]:      # on an incomplete set the sizes are a lower bound: exceeding the limit already is final
        v, why = "DOES NOT FIT", f"resident estimate {resident / GIB:.1f} GiB / file {rep['file_bytes'] / GIB:.1f} GiB against the {limit_gib:g} GiB limit"
    elif unk:
        v = "TYPE NEEDED"
        why = "; ".join(f"unknown weight type id {e['id']} ({e['tensors']} tensors, " + (f"{e['bytes'] / GIB:.2f} GiB" if e["bytes"] else "size unknown") + ")"
                        + (f", upstream ggml names it {e['upstream']}: no decoder and no kernel in this reader" if e["upstream"] else "") + f"; first: {e['names'][0]}" for e in unk.values())
    elif rep["blockers"] or rep["unsupported_types"]:
        v = "MODEL CODE NEEDED"
        why = "types with no decoder / kernel: " + ", ".join(sorted({e["type"] for e in uns})) if uns else "; ".join(rep["blockers"])
    elif not rep["in_tables"] and not rep["hand_row"]:
        v, why = "MODEL CODE NEEDED", f"architecture {arch!r} is in neither the hand rows nor the generated tables (169 llama.cpp architectures at the kit's pin): tensor map, config and graph all missing"
    elif rep["serve_blocker"]:
        v, why = "MODEL CODE NEEDED", rep["serve_blocker"]
    elif rep["hand_row"] and rep["mapped_tables"]:
        v, why = "UNVERIFIED", f"hand-written row for {arch}; {rep['mapped_tables'][0]} is served from the mapped file (KYOJIN_MAPPED_ROW_TABLE=1): CPU-tested only, no Metal proof of the row-gather path"
    elif rep["hand_row"]:
        v, why = "READY", f"hand-written row for {arch}"
    elif cen is None:
        v, why = "UNVERIFIED", "census did not run (see notes)"
    elif cen["verdict"] == "needs model code":
        v, why = "MODEL CODE NEEDED", cen["note"][:240]
    elif cen["verdict"] == "needs vetted lines" and cen["note"].startswith("hand rows exist today") and not rep["config_refusal"] and not rep["refusing_flags"]:
        v, why = "READY", f"vetted rows already in the kit for {arch} ({cen.get('n_lines', '?')} lines in foreign_archkit.VETTED)" + ("" if rep["verified"] else "; greedy identity on Metal NOT yet measured")
    elif cen["verdict"] == "needs vetted lines" or rep["config_refusal"] or rep["refusing_flags"]:
        n = max(int(cen.get("n_min") or 1), 1)
        pieces = []
        if rep["refusing_flags"]:
            pieces.append("a vetted-transform line for converter flags " + ",".join(rep["refusing_flags"]))
        if cen["req"]:
            pieces.append("name rows for " + ",".join(cen["req"][:4]))
        if cen["miss"]:
            pieces.append("config line for " + ",".join(cen["miss"][:4]))
        if rep["config_refusal"]:
            pieces.append("kit_config refuses this file: " + rep["config_refusal"][:120])
        v, why = "ROW NEEDED", f"about {n} line(s): " + ("; ".join(pieces) or cen["note"][:200])
    elif cen["verdict"] == "zero-line":
        v, why = "READY", "zero-line architecture: generated row (kit_config accepts this file's metadata)" + ("" if rep["verified"] else "; greedy identity on Metal NOT yet measured for this architecture")
    else:
        v, why = "UNVERIFIED", f"census verdict {cen['verdict']!r}: {cen['note'][:160]}"
    if inc and arch:
        what = f"incomplete shard set: {inc['present']} of {inc['total']} present"
        if not inc["tensors_seen"] or not inc["shard1_present"]:
            what += f"; verdict from shard 1 of {inc['total']}: weights not seen" if inc["shard1_present"] else "; shard 1 missing"
        if v == "READY":
            v, why = "UNVERIFIED", f"{what}; the tables would give READY ({why}) but the tensor table is not complete"
        elif v != "DOES NOT FIT":
            why = f"{what}; {why}"
        else:
            why = f"{what}; {why}"
    elif not tensors and arch and v == "READY":
        v, why = "UNVERIFIED", "no tensor table in the file: weights not seen"
    rep["verdict"], rep["why"] = v, why
    rep["kind"] = "gguf"
    names = list(tensors)
    d0 = os.path.dirname(os.path.abspath(files[0]))
    rep["vision"] = PK.check_vision("gguf", None, names, [os.path.basename(s) for s in side], os.listdir(d0))
    rep["gguf_dir"] = d0
    if tokenizer:
        cfgp = os.path.join(d0, "config.json")
        try:
            cfg = json.load(open(cfgp)) if os.path.isfile(cfgp) else __import__("kyojin_mlx.foreign_names", fromlist=["x"]).gguf_config(GF.GgufFile(files[0], _shard=True))
        except Exception:  # noqa: BLE001
            cfg = {"model_type": arch}
        try:
            from kyojin_mlx.serve import server as SV
            fam = SV._family_name(files[0], cfg)
        except Exception:  # noqa: BLE001
            fam = None
        rep["tokenizer"] = PK.check_tokenizer(d0 if os.path.isfile(os.path.join(d0, "tokenizer_config.json")) else None, cfg, gguf_kv=kv, family=fam)
        if v.startswith("READY") and rep["tokenizer"]["verdict"] != "TOKENIZER OK":
            rep["verdict"], rep["why"] = rep["tokenizer"]["verdict"], rep["tokenizer"]["why"] + f" (model rows: {why})"
    if serve_cmd:
        rep["serve"] = PK.serve_command(path, port, ctx, None, rep["verdict"], rep.get("tokenizer"))
    if bw_gbs is not None:
        try:
            rep["roofline"] = roofline(tensors, kv, arch, bw_gbs, ctx)
        except ValueError as e:
            rep["roofline"] = dict(error=str(e))
    rep["serve_command"] = serve if rep["verdict"].startswith("READY") else None
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def is_pack(path: str) -> bool:
    """A safetensors pack: a .safetensors file, or a directory with *.safetensors and no model .gguf (same precedence as foreign_linear.is_foreign)."""
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        return p.endswith(".safetensors")
    return os.path.isdir(p) and bool([f for f in os.listdir(p) if f.endswith(".safetensors") and not f.startswith("._")]) and not shard_files(p)[0]


def inspect_pack(path: str, port: int = 8391, limit_gib: float = 236.0, bw_gbs: float | None = None, ctx: int | None = None, tokenizer: bool = True,
                 serve_cmd: bool = False) -> dict:
    """Full day-0 report of a safetensors pack (EXL3, NVFP4 / ModelOpt / FP8 foreign, MLX, dense): format census per role, loader family, tokenizer + template, vision, ceiling."""
    t0 = time.time()
    rep = PK.inspect_pack(path, limit_gib)
    p = os.path.expanduser(path)
    d = p if os.path.isdir(p) else os.path.dirname(p)
    cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.isfile(os.path.join(d, "config.json")) else {}
    info, _ = PK.read_headers(p)
    dir_files = os.listdir(d)
    rep["vision"] = PK.check_vision("safetensors", cfg, list(info), [], dir_files)
    if tokenizer:
        rep["tokenizer"] = PK.check_tokenizer(d, cfg)
        if rep["verdict"].startswith("READY") and rep["tokenizer"]["verdict"] != "TOKENIZER OK":
            rep["why"] = rep["tokenizer"]["why"] + f" (loader: {rep['why']})"
            rep["verdict"] = rep["tokenizer"]["verdict"]
    if bw_gbs is not None:
        try:
            rep["roofline"] = PK.roofline_pack(info, cfg, bw_gbs, ctx)
        except Exception as e:  # noqa: BLE001
            rep["roofline"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
    if serve_cmd:
        row = rep.get("backend_row")
        rep["serve"] = PK.serve_command(path, port, ctx, row["backend"] if row else None, rep["verdict"], rep.get("tokenizer"))
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def render_common(r: dict, o: list) -> None:
    tk = r.get("tokenizer")
    if tk:
        g = tk.get("gguf_tokenizer")
        o.append(f"tokenizer {tk['verdict']}: {tk['why']}" + (f"  [gguf tokenizer.ggml.model={g['model']} pre={g['pre']} chat_template={'yes' if g['chat_template'] else 'no'} tokens={g['n_tokens']}]" if g else "")
                 + (f"  [family {tk['family']}, tools in prompt: {'yes' if tk.get('tools_in_prompt') else 'no'}]" if tk.get("family") else ""))
    vi = r.get("vision")
    if vi:
        pt = vi["vision_parts"]
        o.append(f"vision    {vi['verdict']}: {vi['why']}  [tower tensors {pt['tower_tensors'] or '-'}; mmproj {pt['mmproj_files'] or '-'}; vision_config {'yes' if pt['config_vision'] else 'no'}; preprocessor files {pt['preprocessor_files'] or '-'}; audio {'yes' if pt['audio_tensors'] else 'no'}]")
    sv = r.get("serve")
    if sv:
        o.append(f"serve     {sv['command']}")
        for u in sv["unknown"]:
            o.append(f"          {u}")
        if sv.get("derived"):
            o.append("          the server derives (not on the line): " + json.dumps(sv["derived"], default=str))
        if sv.get("note"):
            o.append(f"          note: {sv['note']}")


def render_pack(r: dict) -> str:
    o = [f"pack      {r['path']}  ({len(r['files'])} file(s), {r['file_bytes'] / GIB:.2f} GiB, {r['n_tensors']} tensors, read in {r['seconds']} s)"]
    o.append(f"model     model_type={r['model_type'] or '-'} architectures={r['architectures']}  loader family: {r['pack_family']} ({r['pack_family_via']})")
    br = r["backend_row"]
    o.append(f"engine    modeldesc row: {('%s (backend %s, chat family %s)' % (br['name'], br['backend'], br['server'])) if br else 'none'}; mlx-lm class for the model_type: {r['mlx_lm_class']}; detected linear formats: {r['detected']['counts'] or '-'}")
    o.append("formats   role              format                         tensors        GiB")
    for e in r["formats"]:
        o.append(f"          {e['role']:17s} {e['format']:30s} {e['tensors']:7d} {e['bytes'] / GIB:10.3f}")
    o.append(f"size      resident estimate {r['resident_est_bytes'] / GIB:.1f} GiB (vision / audio left out); file {r['file_bytes'] / GIB:.1f} GiB; limit {r['limit_bytes'] / GIB:g} GiB -> {'fits' if r['fits'] else 'does not fit'}; draft head (mtp.*) {r['mtp_bytes'] / GIB:.3f} GiB, config mtp layers {r['mtp_layers']}")
    for n in r["notes"] + r["blockers"]:
        o.append(f"note      {n}")
    render_roofline(r, o)
    render_common(r, o)
    o.append(f"VERDICT   {r['verdict']}: {r['why']}" + (f"  | {r['serve']['command']}" if r.get("serve") and r["serve"]["runnable"] else ""))
    return "\n".join(o)


def _fmt_range(lo, hi) -> str:
    return str(hi) if lo == hi else f"{lo}..{hi}"


def render_roofline(r: dict, o: list) -> None:
    rl = r.get("roofline")
    if rl and "error" in rl:
        o.append(f"roofline  UNVERIFIED: {rl['error']}")
    elif rl:
        pt = rl["parts"]
        o.append(f"roofline  {rl['bytes_per_step'] / 1e9:.3f} GB read per decode step at 1 session (trunk {pt['trunk'] / 1e9:.3f}, routed experts {pt['routed_experts'] / 1e9:.3f} "
                 f"[{_fmt_range(rl['experts_used_min'], rl['experts_used'])}/{_fmt_range(rl['experts_total_min'], rl['experts_total'])}{', per layer' if rl.get('per_layer_experts') else ''}], head {pt['head'] / 1e9:.3f}{' (tied to token_embd)' if rl['tied_head'] else ''}, embedding row {pt['embedding_row'] / 1e6:.3f} MB)")
        o.append(f"          ceiling {rl['ceiling_tok_s']:.1f} tok/s at {rl['bw_gbs']:g} GB/s. This is a memory-bandwidth ceiling for decode at 1 session (weights only), NOT a prediction of speed.")
        kv = rl.get("kv")
        if kv and "bytes" in kv:
            o.append(f"          + KV / attention read at ctx {kv['ctx']}: {kv['bytes'] / 1e9:.3f} GB ({kv['full_layers']} full-attention layers, {kv['shape']})" + (f"; {kv['note']}" if kv.get("note") else "")
                     + f" -> {rl['bytes_per_step_with_kv'] / 1e9:.3f} GB per step, ceiling {rl['ceiling_with_kv_tok_s']:.1f} tok/s")
        elif kv:
            o.append(f"          KV term not counted: {kv['error']}")
        if rl.get("draft_head_bytes"):
            o.append(f"          draft head declared: {rl['draft_head_bytes'] / 1e9:.3f} GB per draft call (ceiling {rl['draft_call_ceiling_per_s']:.0f} draft calls/s), not added to the plain-decode step")
        o.append("          counted: " + "; ".join(rl["terms_counted"]))
        o.append("          NOT counted: " + "; ".join(rl["terms_not_counted"]))


def render(r: dict) -> str:
    if r.get("kind") == "safetensors":
        return render_pack(r)
    o = [f"file      {r['path']}  ({len(r['files'])} file(s), {r['file_bytes'] / GIB:.2f} GiB, {r['n_tensors']} tensors, read in {r['seconds']} s)"]
    o.append(f"arch      {r['architecture']}  name={r.get('name')}  generated tables: {'yes' if r['in_tables'] else 'NO'}  hand row: {'yes' if r['hand_row'] else 'no'}  config: {'hand / cfgrows row' if r['cfg_row'] else 'kit_config from the generated tables'}")
    c = r["census"]
    if c:
        o.append(f"census    {c['verdict']}  class={c['cls']} ({c['how']})  n_min={c['n_min']}  required tensors without a parameter={c['required_missing'] or '-'}  config gaps={c['config_missing'] or '-'}")
        o.append(f"          note: {c['note'] or '-'}")
    else:
        o.append("census    not available (UNVERIFIED)")
    if r["in_tables"]:
        o.append(f"converter flags {r['converter_flags'] or '-'}; value transforms {r['value_transforms'] or '-'}; vetted {r['vetted_flags'] or '-'}; forcing a refusal: {r['refusing_flags'] or '-'}")
    if r["config_refusal"]:
        o.append(f"config    REFUSED: {r['config_refusal']}")
    o.append(f"model     {'routed MoE: %s experts, %s used%s' % (_fmt_range(r['expert_count_min'], r['expert_count']), _fmt_range(r['expert_used_min'], r['expert_used']), ' (per layer: min..max over layers)' if r['expert_per_layer'] else '') if r['moe'] else 'dense'}; MTP layers {r['mtp_layers']}; side tensors {{{', '.join(f'{k}: {v / GIB:.2f} GiB' for k, v in r['side_tensors'].items()) or '-'}}}; side files {r['side_files'] or '-'}")
    o.append("types     type      tensors        GiB  mode")
    for e in r["types"]:
        o.append(f"          {e['type']:9s} {e['tensors']:7d} {e['bytes'] / GIB:10.3f}  {e['mode']}")
    o.append(f"size      resident estimate {r['resident_est_bytes'] / GIB:.1f} GiB; file {r['file_bytes'] / GIB:.1f} GiB; limit {r['limit_bytes'] / GIB:g} GiB -> {('fits (partial set: lower bound)' if r.get('incomplete_set') else 'fits') if r['fits'] else 'does not fit'}")
    for n in r["notes"] + r["blockers"]:
        o.append(f"note      {n}")
    render_roofline(r, o)
    render_common(r, o)
    o.append(f"VERDICT   {r['verdict']}: {r['why']}" + (f"  | {r['serve_command']}" if r["serve_command"] else ""))
    return "\n".join(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--port", type=int, default=8391)
    ap.add_argument("--limit-gib", type=float, default=236.0)
    ap.add_argument("--no-census", action="store_true", help="skip the tiny-config census (faster, no mlx-lm import): the verdict becomes UNVERIFIED for non-hand rows")
    ap.add_argument("--roofline", action="store_true", help="add the decode bytes-per-step and the bandwidth ceiling (header arithmetic)")
    ap.add_argument("--ctx", type=int, default=None, help="context length N for the KV / attention read term of --roofline (and the --ctx of --serve-cmd)")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the tokenizer + chat template check")
    ap.add_argument("--serve-cmd", action="store_true", help="print the command that serves this model on this machine (not run)")
    ap.add_argument("--bw-gbs", type=float, default=DEFAULT_BW_GBS, help="memory bandwidth in GB/s for --roofline (default: 1159 on the M5 Ultra, tools/opprof.py B_REF; required on any other machine)")
    a = ap.parse_args(argv)
    if a.roofline and a.bw_gbs is None:
        ap.error("--roofline needs --bw-gbs on this machine (no measured bandwidth on file)")
    bw = a.bw_gbs if a.roofline else None
    if is_pack(a.path):
        r = inspect_pack(a.path, a.port, a.limit_gib, bw, a.ctx, not a.no_tokenizer, a.serve_cmd)
    else:
        r = inspect(a.path, a.port, a.limit_gib, not a.no_census, bw, a.ctx, not a.no_tokenizer, a.serve_cmd)
    print(json.dumps(r, indent=1, default=str) if a.json else render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
