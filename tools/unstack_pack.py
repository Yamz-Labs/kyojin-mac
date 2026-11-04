#!/usr/bin/env python3
"""Inverse of tools/convert_stack.py, for tests on the Air: a stacked slice pack -> the per-expert layout of the HF pack (one tensor per expert
and per wo_a slice), so the load-time restack (kyojin_mlx/restack.py) runs on the existing slice packs. Layout only, bytes untouched, CPU numpy.
Files without stacked tensors are hard-linked (no extra disk); stacked.<layer>.safetensors becomes <layer>.raw.safetensors; new index.
usage: python3 -I unstack_pack.py STACKED_DIR OUT_DIR"""
import glob, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_stack as CS  # noqa: E402

STK = re.compile(r"^(.*)\.experts\.([^.]+)\.(trellis|offset|k|suh|svh|mul1|mcg|shape)$")
SLS = re.compile(r"^(.*)\.slices\.(trellis|suh|svh|mul1|mcg)$")


def arr(src, name):
    _, _, _, dt, shp = src.t[name]
    return np.frombuffer(src.read(name), {"U32": "<u4", "U8": "u1", "F16": "<f2", "I32": "<i4", "I16": "<i2"}[dt]).reshape(shp)


def main(sd, od):
    os.makedirs(od, exist_ok=True)
    src = CS.Src(sd)
    meta = src.meta or {"format": "pt"}
    wmap = {}
    for f in src.files:
        names = [k for k, v in src.t.items() if v[0] == f]
        if not any(STK.match(k) or SLS.match(k) for k in names):
            dst = os.path.join(od, os.path.basename(f))
            if not os.path.exists(dst):
                os.link(f, dst)
            wmap.update({k: os.path.basename(f) for k in names})
            continue
        ents = []
        groups = sorted({STK.match(k).group(1, 2) for k in names if STK.match(k)})
        for prefix, proj in groups:
            b = f"{prefix}.experts.{proj}"
            fk = "mul1" if f"{b}.mul1" in src.t else "mcg"
            off, ks, suh, svh = arr(src, b + ".offset"), arr(src, b + ".k"), arr(src, b + ".suh"), arr(src, b + ".svh")
            kt, nt = (int(x) for x in arr(src, b + ".shape"))
            flag = arr(src, f"{b}.{fk}").reshape(())
            for e in range(len(ks)):
                words = np.frombuffer(src.read(b + ".trellis", int(off[e]) * 4, int(off[e + 1] - off[e]) * 4), "<i2").reshape(kt, nt, 16 * int(ks[e]))
                p = f"{prefix}.experts.{e}.{proj}"
                ents += [CS.bytes_entry(p + ".trellis", "I16", list(words.shape), words), CS.bytes_entry(p + ".suh", "F16", list(suh[e].shape), suh[e]),
                         CS.bytes_entry(p + ".svh", "F16", list(svh[e].shape), svh[e]), CS.bytes_entry(f"{p}.{fk}", "I32", [], flag)]
        for prefix in sorted({SLS.match(k)[1] for k in names if SLS.match(k)}):
            b = f"{prefix}.slices"
            fk = "mul1" if f"{b}.mul1" in src.t else "mcg"
            tr, suh, svh, flag = arr(src, b + ".trellis"), arr(src, b + ".suh"), arr(src, b + ".svh"), arr(src, f"{b}.{fk}").reshape(())
            for s in range(tr.shape[0]):
                p = f"{prefix}.slice.{s}"
                ents += [CS.bytes_entry(p + ".trellis", "I16", list(tr[s].shape), tr[s]), CS.bytes_entry(p + ".suh", "F16", list(suh[s].shape), suh[s]),
                         CS.bytes_entry(p + ".svh", "F16", list(svh[s].shape), svh[s]), CS.bytes_entry(f"{p}.{fk}", "I32", [], flag)]
        for k in names:
            if not (STK.match(k) or SLS.match(k)):
                _, _, nb, dt, shp = src.t[k]
                ents.append(CS.Entry(k, dt, shp, nb, CS.copy_chunks(src, k)))
        fn = os.path.basename(f).replace("stacked.", "", 1).replace(".safetensors", ".raw.safetensors")
        CS.write_st(os.path.join(od, fn), ents, meta)
        wmap.update({e.name: fn for e in ents})
        print(f"{os.path.basename(f)} -> {fn}: {len(ents)} tensors", flush=True)
    for f in glob.glob(os.path.join(sd, "*")):
        b = os.path.basename(f)
        if os.path.isfile(f) and not b.endswith(".safetensors") and b != "model.safetensors.index.json":
            c = open(f, "rb").read()
            if b == "config.json":
                cfg = json.loads(c)
                cfg.get("quantization_config", {}).pop("expert_layout", None)
                c = json.dumps(cfg, indent=1).encode()
            open(os.path.join(od, b), "wb").write(c)
    json.dump({"metadata": {}, "weight_map": wmap}, open(os.path.join(od, "model.safetensors.index.json"), "w"), indent=1)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
