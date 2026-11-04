#!/usr/bin/env python3
"""Convert an EXL3 pack (one safetensors tensor per expert) to the stacked Mac layout. CPU only, numpy, no safetensors/torch.
usage: python3 -I convert_stack.py SRC_PACK OUT_DIR [--only REGEX] [--verify N] [--dry-run]

What it changes (everything else is copied byte for byte, same names, same file names):
  routed experts   <p>.experts.<E>.<proj>.{trellis,suh,svh,mul1}   (E = 0..n-1, one tensor each, 46080 of them in DS4.1)
    ->  <p>.experts.<proj>.trellis  U32 [W]      all experts of one projection, concatenated in expert order (ragged: K may differ)
        <p>.experts.<proj>.offset   U32 [n+1]    word offset of expert e (tile (kt_i, nt_j) at offset[e] + (kt_i*nt + nt_j) * 8*K[e])
        <p>.experts.<proj>.k        U8  [n]      bits per weight of expert e (2 or 3 in our packs)
        <p>.experts.<proj>.suh      F16 [n, k]   input sign flips      <p>.experts.<proj>.svh  F16 [n, nout]  output scales
        <p>.experts.<proj>.mul1     I32 []       codebook flag (checked equal for every expert)
        <p>.experts.<proj>.shape    I32 [2]      [kt, nt] tiles of 16x16 (identical for every expert)
  grouped-output slices  <p>.slice.<S>.{trellis,suh,svh,mul1}  (DS4.1 attn.wo_a, 8 slices)
    ->  <p>.slices.trellis I16 [S, kt, nt, 16K]   <p>.slices.suh F16 [S, k]   <p>.slices.svh F16 [S, nout]   <p>.slices.mul1 I32 []
Stacked tensors go to stacked.<layer>.safetensors; a new model.safetensors.index.json lists every tensor. Word offsets must stay
below 2**32 (checked). The int16 trellis bytes are untouched: expert e = words[offset[e]:offset[e+1]] viewed as int16 [kt, nt, 16K].
Streaming: one tensor in memory at a time (at most the 1.2 GiB embedding when copied in 64 MiB chunks)."""
import argparse, glob, json, os, random, re, shutil, struct, sys

import numpy as np

EXP = re.compile(r"^(.*)\.experts\.(\d+)\.(.+)\.(trellis|suh|svh|mul1|mcg)$")
SLC = re.compile(r"^(.*)\.slice\.(\d+)\.(trellis|suh|svh|mul1|mcg)$")
LAYER = re.compile(r"^(.*?(?:layers|mtp)\.\d+)")
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


class Src:
    """All tensors of a pack: name -> (path, absolute offset, nbytes, dtype, shape)."""

    def __init__(self, d):
        self.dir, self.t, self.meta, self.fds = d, {}, None, {}
        self.files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        assert self.files, f"no .safetensors in {d}"
        for f in self.files:
            h, base, meta = read_header(f)
            self.meta = self.meta or meta
            for k, v in h.items():
                a, b = v["data_offsets"]
                assert k not in self.t, f"duplicate tensor {k}"
                self.t[k] = (f, base + a, b - a, v["dtype"], v["shape"])

    def read(self, name, start=0, n=None):
        f, off, nb, _, _ = self.t[name]
        n = nb - start if n is None else n
        fd = self.fds.get(f) or self.fds.setdefault(f, os.open(f, os.O_RDONLY))
        out, got = bytearray(), 0
        while got < n:
            b = os.pread(fd, n - got, off + start + got)
            assert b, f"short read in {name}"
            out += b; got += len(b)
        return bytes(out)

    def close(self):
        for fd in self.fds.values(): os.close(fd)


class Entry:
    def __init__(self, name, dtype, shape, nbytes, chunks):
        self.name, self.dtype, self.shape, self.nbytes, self.chunks = name, dtype, shape, nbytes, chunks


def copy_chunks(src, name):
    nb = src.t[name][2]
    def gen():
        for s in range(0, nb, CHUNK):
            yield src.read(name, s, min(CHUNK, nb - s))
    return gen


def write_st(path, entries, meta):
    hdr, off = {}, 0
    if meta: hdr["__metadata__"] = meta
    for e in entries:
        hdr[e.name] = {"dtype": e.dtype, "shape": list(e.shape), "data_offsets": [off, off + e.nbytes]}
        off += e.nbytes
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb))); fh.write(hb)
        for e in entries:
            w = 0
            for c in e.chunks():
                fh.write(c); w += len(c)
            assert w == e.nbytes, f"{e.name}: wrote {w} expected {e.nbytes}"
    os.replace(tmp, path)


def bytes_entry(name, dtype, shape, arr):
    b = np.ascontiguousarray(arr).tobytes()
    return Entry(name, dtype, shape, len(b), lambda b=b: iter([b]))


def flag_value(src, names):
    vals = set(int(np.frombuffer(src.read(n), "<i4")[0]) for n in names)
    assert len(vals) == 1, f"codebook flag differs between tensors: {vals}"
    return vals.pop()


def plan_expert_group(src, prefix, proj, parts):
    """parts: {e: {suffix: tensor name}} -> list[Entry]."""
    n = len(parts)
    assert sorted(parts) == list(range(n)), f"{prefix}.{proj}: expert ids not contiguous"
    flag = None
    for e in range(n):
        assert "trellis" in parts[e] and "suh" in parts[e] and "svh" in parts[e], f"{prefix}.experts.{e}.{proj}: missing parts"
    flag_names = [parts[e].get("mul1") or parts[e].get("mcg") for e in range(n)]
    assert all(flag_names), f"{prefix}.{proj}: codebook flag tensor missing"
    flag_key = "mul1" if "mul1" in parts[0] else "mcg"
    val = flag_value(src, flag_names)
    ks, kt, nt, words, off = [], None, None, [], [0]
    for e in range(n):
        _, _, nb, dt, shp = src.t[parts[e]["trellis"]]
        assert dt == "I16" and len(shp) == 3 and shp[2] % 16 == 0, f"{parts[e]['trellis']}: {dt} {shp}"
        if kt is None: kt, nt = shp[0], shp[1]
        assert (shp[0], shp[1]) == (kt, nt), f"{parts[e]['trellis']}: tile grid {shp[:2]} differs from {(kt, nt)}"
        ks.append(shp[2] // 16); assert nb % 4 == 0
        off.append(off[-1] + nb // 4)
    assert off[-1] < 2 ** 32, f"{prefix}.{proj}: {off[-1]} words does not fit a uint32 offset"
    suh_s, svh_s = src.t[parts[0]["suh"]][4], src.t[parts[0]["svh"]][4]
    for e in range(n):
        assert src.t[parts[e]["suh"]][4] == suh_s and src.t[parts[e]["svh"]][4] == svh_s and \
            src.t[parts[e]["suh"]][3] == "F16" and src.t[parts[e]["svh"]][3] == "F16", f"{prefix}.{proj}: suh/svh differ at expert {e}"
    base = f"{prefix}.experts.{proj}"

    def tr():
        for e in range(n): yield src.read(parts[e]["trellis"])

    def cat(key):
        def g():
            for e in range(n): yield src.read(parts[e][key])
        return g
    return [
        Entry(base + ".trellis", "U32", [off[-1]], off[-1] * 4, tr),
        bytes_entry(base + ".offset", "U32", [n + 1], np.array(off, "<u4")),
        bytes_entry(base + ".k", "U8", [n], np.array(ks, "u1")),
        Entry(base + ".suh", "F16", [n] + suh_s, n * int(np.prod(suh_s)) * 2, cat("suh")),
        Entry(base + ".svh", "F16", [n] + svh_s, n * int(np.prod(svh_s)) * 2, cat("svh")),
        bytes_entry(base + "." + flag_key, "I32", [], np.array(val, "<i4")),
        bytes_entry(base + ".shape", "I32", [2], np.array([kt, nt], "<i4")),
    ]


def plan_slices(src, prefix, parts):
    n = len(parts)
    assert sorted(parts) == list(range(n)), f"{prefix}.slice: ids not contiguous"
    names = {k: [parts[s][k] for s in range(n)] for k in ("trellis", "suh", "svh")}
    flag_key = "mul1" if "mul1" in parts[0] else "mcg"
    val = flag_value(src, [parts[s][flag_key] for s in range(n)])
    for k, ns in names.items():
        shp = {(src.t[x][3], tuple(src.t[x][4])) for x in ns}
        assert len(shp) == 1, f"{prefix}.slice.*.{k}: shapes differ {shp}"
    dt, shp = src.t[names["trellis"][0]][3], src.t[names["trellis"][0]][4]
    assert dt == "I16"
    base = f"{prefix}.slices"

    def cat(ns):
        def g():
            for x in ns: yield src.read(x)
        return g
    out = []
    for k in ("trellis", "suh", "svh"):
        d, s = src.t[names[k][0]][3], src.t[names[k][0]][4]
        out.append(Entry(f"{base}.{k}", d, [n] + s, sum(src.t[x][2] for x in names[k]), cat(names[k])))
    out.append(bytes_entry(f"{base}.{flag_key}", "I32", [], np.array(val, "<i4")))
    return out


def convert(src_dir, out_dir, only=None, dry=False):
    src = Src(src_dir)
    rx = re.compile(only) if only else None
    names = [k for k in src.t if not rx or rx.search(k)]
    egroups, sgroups, consumed = {}, {}, set()
    for k in names:
        m = EXP.match(k)
        if m:
            egroups.setdefault((m[1], m[3]), {}).setdefault(int(m[2]), {})[m[4]] = k; consumed.add(k); continue
        m = SLC.match(k)
        if m:
            sgroups.setdefault(m[1], {}).setdefault(int(m[2]), {})[m[3]] = k; consumed.add(k)
    os.makedirs(out_dir, exist_ok=True)
    wmap, total = {}, 0
    stacked = {}                                         # layer key -> [Entry]
    for (prefix, proj), parts in sorted(egroups.items()):
        key = (LAYER.match(prefix) or [None, prefix])[1]
        stacked.setdefault(key, []).extend(plan_expert_group(src, prefix, proj, parts))
    for prefix, parts in sorted(sgroups.items()):
        key = (LAYER.match(prefix) or [None, prefix])[1]
        stacked.setdefault(key, []).extend(plan_slices(src, prefix, parts))
    meta = src.meta or {"format": "pt"}
    nameset = set(names)
    nstack = sum(len(v) for v in stacked.values())
    print(f"{len(names)} tensors, {len(consumed)} stacked into {nstack} tensors in {len(stacked)} files, "
          f"{len(egroups)} expert groups, {len(sgroups)} slice groups")
    if dry: return src, wmap
    for f in src.files:                                   # pass-through: same file name, same tensor names
        ents = [Entry(k, v[3], v[4], v[2], copy_chunks(src, k)) for k, v in src.t.items()
                if v[0] == f and k in nameset and k not in consumed]
        if not ents: continue
        fn = os.path.basename(f); write_st(os.path.join(out_dir, fn), ents, meta)
        for e in ents: wmap[e.name] = fn; total += e.nbytes
        print("pass", fn, len(ents), "tensors", flush=True)
    for key, ents in sorted(stacked.items()):
        fn = f"stacked.{key}.safetensors"; write_st(os.path.join(out_dir, fn), ents, meta)
        for e in ents: wmap[e.name] = fn; total += e.nbytes
        print("stack", fn, len(ents), "tensors", round(sum(e.nbytes for e in ents) / 2 ** 30, 2), "GiB", flush=True)
    for f in glob.glob(os.path.join(src_dir, "*")):       # config, tokenizer, templates (no safetensors, no index, no dirs)
        b = os.path.basename(f)
        skip = b.endswith((".safetensors", ".st", ".done.json")) or b in ("model.safetensors.index.json", "plan.json", "quantization_config.json")
        if os.path.isfile(f) and not skip:
            if b == "config.json":
                c = json.load(open(f)); c.setdefault("quantization_config", {})["expert_layout"] = "ragged-u32-v1"
                json.dump(c, open(os.path.join(out_dir, b), "w"), indent=1)
            else:
                shutil.copy2(f, out_dir)
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=1)
    return src, wmap


def load_stack(out_dir, wmap, prefix, proj):
    """Read back one expert group from the converted pack with numpy (reference reader for tests and for the Mac loader)."""
    base = f"{prefix}.experts.{proj}"
    def get(sfx):
        fn = os.path.join(out_dir, wmap[f"{base}.{sfx}"]); h, b, _ = read_header(fn); v = h[f"{base}.{sfx}"]
        a, z = v["data_offsets"]
        arr = np.memmap(fn, dtype={"U32": "<u4", "U8": "u1", "F16": "<f2", "I32": "<i4"}[v["dtype"]], mode="r", offset=b + a, shape=tuple(v["shape"]))
        return arr
    return dict(words=get("trellis"), offset=get("offset"), k=get("k"), suh=get("suh"), svh=get("svh"), shape=get("shape"))


def expert_trellis(st, e):
    kt, nt = (int(x) for x in st["shape"]); k = int(st["k"][e])
    return np.asarray(st["words"][int(st["offset"][e]):int(st["offset"][e + 1])]).view("<i2").reshape(kt, nt, 16 * k)


def verify(src, out_dir, wmap, samples, seed=1):
    rnd, n_ok = random.Random(seed), 0
    groups = sorted({(m[1], m[3]) for k in src.t for m in [EXP.match(k)] if m and f"{m[1]}.experts.{m[3]}.trellis" in wmap})
    for prefix, proj in groups:
        st = load_stack(out_dir, wmap, prefix, proj); n = len(st["k"])
        assert st["offset"][-1] == len(st["words"]) and st["offset"][0] == 0
        for e in rnd.sample(range(n), min(samples, n)):
            nm = f"{prefix}.experts.{e}.{proj}"
            ref = np.frombuffer(src.read(nm + ".trellis"), "<i2").reshape(src.t[nm + ".trellis"][4])
            assert np.array_equal(expert_trellis(st, e), ref), nm
            assert np.array_equal(st["suh"][e], np.frombuffer(src.read(nm + ".suh"), "<f2")), nm
            assert np.array_equal(st["svh"][e], np.frombuffer(src.read(nm + ".svh"), "<f2")), nm
            n_ok += 1
    sg = sorted({m[1] for k in src.t for m in [SLC.match(k)] if m and f"{m[1]}.slices.trellis" in wmap})
    for prefix in sg:
        fn = os.path.join(out_dir, wmap[f"{prefix}.slices.trellis"]); h, b, _ = read_header(fn); v = h[f"{prefix}.slices.trellis"]
        arr = np.memmap(fn, dtype="<i2", mode="r", offset=b + v["data_offsets"][0], shape=tuple(v["shape"]))
        for s in range(v["shape"][0]):
            nm = f"{prefix}.slice.{s}.trellis"
            assert np.array_equal(arr[s], np.frombuffer(src.read(nm), "<i2").reshape(src.t[nm][4])), nm
            n_ok += 1
    print(f"verify: {n_ok} expert/slice tensors identical to the source, {len(groups)} expert groups, {len(sg)} slice groups")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--only"); ap.add_argument("--verify", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s, wm = convert(a.src, a.out, a.only, a.dry_run)
    if not a.dry_run and a.verify: verify(s, a.out, wm, a.verify)
    s.close()
