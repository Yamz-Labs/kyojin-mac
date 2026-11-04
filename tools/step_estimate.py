"""One DS4.1 decode token (40-layer mix) from tools/layer_cpu.py runs: launches and CPU ms per token, then the Ultra solo estimate with and without MTP.
usage: step_estimate.py DIR [--trees base,cur] [--gap-us 1.7,3.0] [--gpu-ms 9.1] [--mtp 0.98,1.13,1.28] [--json F]
DIR holds {tree}_{name}_r{rep}.json written by layer_cpu.py (several reps: the quietest one counts, min over reps):
  full    = layer_cpu.py ds2                       full layer (compressor + indexer) and reuse layer, fixed part (embed, head, final norm)
  reindex = layer_cpu.py ds2 --variant reindex     indexer without the compressor
  cand    = layer_cpu.py ds2 --variant cand        full + candidate-block selection (the layer that is the candidate source)
  local   = layer_cpu.py ds2 --ratios 2,0          layer 1 local (window only)
  engram  = layer_cpu.py ds2e --engram-resident    layer 1 reuse + Engram on a resident table (no host sync); minus the reuse layer = the first Engram layer,
            which also builds the n-gram hash of the token (HASH_LAUNCHES dispatches, shared by the Engram layers); the second one costs the layer without it
Layer mix of the real config (compress_ratios, kv / index / candidate source ids): 2 local, 4 full (one of them the candidate source, + the selection),
4 reindex, 30 reuse; 2 Engram layers. The window index math (about 9 dispatches) is built once per token and shared by all layers (fused_group.share_window_math):
the harness bills it to the first layer that calls `_window_part` (the full / reindex layer of the baseline tree, the local layer of ours), WIN_FIX removes the duplicates. CPU = python graph build + async_eval encode with one command buffer per step (lower bound of the default-buffer run: the Air GPU wait
is out of it). Ultra model (ULTRA.md s.12): CPU side = the Air figure; GPU side = --gpu-ms (routed experts ALU bound 3.9 + other weights 5.2 at 0.82 x 1.2 TB/s)
+ launches x dependent-kernel gap; step = max of the two sides."""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--trees", default="base,cur"); ap.add_argument("--gap-us", default="1.7,3.0")
ap.add_argument("--gpu-ms", type=float, default=9.1); ap.add_argument("--mtp", default="0.98,1.13,1.28"); ap.add_argument("--json")
ap.add_argument("--base-win-fix", type=float, default=-9 * 7, help="window-math duplicates removed from the base tree (-63 for the pre-share tree 2b76e28; -9 when base is the current tree with a switch off)")
ap.add_argument("--extra-launches", type=float, default=0.0, help="launches added to the token by hand, e.g. 35 = the ratio-1 compressor of layer 20 closes a group every token (121 -> 156)")
ap.add_argument("--sync-ms", type=float, default=0.0, help="serialisation of a disk-resident Engram table (host read of the hash ids at each Engram layer: about 0.33-0.5 ms each), added to the step")
a = ap.parse_args()
MIX = dict(reuse=30, full=4, cand=1, reindex=4, local=2)
HASH_LAUNCHES = int(os.environ.get("HASH_LAUNCHES", "56"))             # Engram n-gram hash of one token (graph census of the hash call: 56 stock, 1 with engram_fuse)
WIN_FIX = {"base": a.base_win_fix, "cur": -9}                           # shared window math billed 8 times (4 full + 4 reindex, one is real) / twice (2 local, one is real)


def load(tree, name):
    """Quietest rep per quantity: min over the reps of the min-based and of the median CPU figures; launches are equal across reps."""
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.dir, f"{tree}_{name}_r*.json")))]
    assert runs, (tree, name)
    def pick(kind):
        ents = [r[kind] for r in runs if kind in r]
        e = dict(launches=ents[0].get("launches", 0.0))
        if kind == "none":
            e["min"] = min(x["build_min"] + x["enc_min"] for x in ents); e["med"] = min(x["build"] + x["enc"] for x in ents)
        else:
            e["min"] = min(x["build_min_ms"] + x["enc_min_ms"] for x in ents); e["med"] = min(x["build_ms"] + x["enc_ms"] for x in ents)
        return e
    return {k: pick(k) for k in (("none",) if "build_min" in runs[0]["none"] else ()) + ("full", "reuse")}


def sub(x, y):
    return {k: x[k] - y[k] for k in x}


def kinds(tree):
    f, ri, ca, lo, en = (load(tree, n) for n in ("full", "reindex", "cand", "local", "engramres"))
    k = dict(reuse=f["reuse"], full=f["full"], cand=sub(ca["full"], f["full"]), reindex=ri["full"], local=lo["reuse"])
    first = sub(en["reuse"], f["reuse"])                        # first Engram layer + the hash
    share = HASH_LAUNCHES / first["launches"]
    second = {q: first[q] - (share * first[q]) for q in first}  # hash cost taken pro rata to its launches
    second["launches"] = first["launches"] - HASH_LAUNCHES
    return k, dict(en["none"]), dict(first=first, second=second)


def token(tree):
    k, fixed, en = kinds(tree)
    if tree != "cur":                                           # the candidate selection and the Engram layers are not touched by the cuts: their CPU is read on the last tree (noise-free there)
        k2, _, en2 = kinds("cur")
        for q in ("min", "med"):
            k["cand"][q] = k2["cand"][q]
            for e in ("first", "second"):
                en[e][q] = en2[e][q]
    tot = {q: fixed[q] + en["first"][q] + en["second"][q] for q in fixed}
    for n, c in MIX.items():
        for q in tot:
            tot[q] += c * k[n][q]
    tot["launches"] += WIN_FIX.get(tree, 0) + (a.extra_launches if tree == "cur" else 0)
    return k, fixed, en, tot


GPU_ROWS = [1.00, 1.36, 1.81, 2.27, 3.08]                     # GPU cost of a chained verify of n = 1..5 rows against one plain step (ULTRA.md s.11, Air GPU)
SYNC_MS = 1.5                                                  # draft tokens go back to python before the verify is built (s.12: d bracket 0.23-0.30)


def mtp(tree, cpu_ms, gpu_ms, mult):
    """MTP from the measured CPU cost of the verify rows and of the draft call (tools/spec_cpu.py on ds2m, json spec_{tree}.json, M10_LIGHT=16): verify of n rows
    takes max(ratio_n x CPU of a plain token, GPU rows factor x GPU side), a round adds the draft call (its CPU min + SYNC_MS). Returns {a: (best D, tok/s)}."""
    f = os.path.join(a.dir, f"spec_{tree}.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    ratio = [j[f"verify{n}"]["min"] / j["solo"]["min"] for n in range(1, 6)]
    dcpu = j["draft_cpu"]
    d = (dcpu.get("cpu_min") or (dcpu["build_med"] + dcpu["enc_med"])) + SYNC_MS
    solo = max(cpu_ms, gpu_ms)
    res = {}
    for acc in (0.6, 0.735, 0.85):
        best = (0, 1e3 / solo)
        for D in (1, 2, 3, 4):
            n = D + 1
            v = max(ratio[n - 1] * cpu_ms, GPU_ROWS[n - 1] * gpu_ms)
            tps = 1e3 * sum(acc ** i for i in range(n)) / (v + d)
            if tps > best[1]:
                best = (D, tps)
        res[acc] = best
    return res, d, ratio


out = {}
gaps = [float(v) for v in a.gap_us.split(",")]
mult = [float(v) for v in a.mtp.split(",")]
for tree in a.trees.split(","):
    k, fixed, en, tot = token(tree)
    print(f"== {tree}")
    print(f"{'kind':10s} {'count':>5s} {'launches':>9s} {'CPU min ms':>11s} {'CPU med ms':>11s}   per token: launches, CPU min ms")
    for n, c in MIX.items():
        print(f"{n:10s} {c:5d} {k[n]['launches']:9.1f} {k[n]['min']:11.3f} {k[n]['med']:11.3f}   {c * k[n]['launches']:8.0f} {c * k[n]['min']:8.2f}")
    print(f"{'fixed':10s} {'':5s} {fixed['launches']:9.1f} {fixed['min']:11.3f} {fixed['med']:11.3f}   (embed, head, final norm)")
    print(f"{'engram 1st':10s} {1:5d} {en['first']['launches']:9.1f} {en['first']['min']:11.3f} {en['first']['med']:11.3f}   (with the hash)")
    print(f"{'engram 2nd':10s} {1:5d} {en['second']['launches']:9.1f} {en['second']['min']:11.3f} {en['second']['med']:11.3f}")
    print(f"{'window fix':10s} {'':5s} {WIN_FIX.get(tree, 0):9.1f}   (shared window math billed once)")
    print(f"TOKEN: {tot['launches']:.0f} launches, CPU {tot['min']:.2f} ms (min-based) .. {tot['med']:.2f} ms (median)")
    rec = dict(kinds=k, fixed=fixed, engram=en, token=tot, ultra={})
    for g in gaps:
        gpu = a.gpu_ms + tot["launches"] * g * 1e-3
        for cname, cpu in (("cpu_min", tot["min"]), ("cpu_med", tot["med"])):
            step = max(cpu, gpu) + a.sync_ms
            solo = 1e3 / step
            rec["ultra"][f"gap{g}_{cname}"] = dict(gpu_ms=gpu, cpu_ms=cpu, step_ms=step, solo=solo, mtp=[round(solo * m, 1) for m in mult])
            m = mtp(tree, cpu, gpu, mult)
            if m and cname == "cpu_min":
                res, d, ratio = m
                print(f"      MTP with measured CPU rows (verify/solo {'/'.join(f'{r:.2f}' for r in ratio)}, draft+sync {d:.1f} ms): "
                      + "; ".join(f"a={k}: D={b[0]} {b[1]:.0f} tok/s (x{b[1] / solo:.2f})" for k, b in res.items()))
                rec["ultra"][f"gap{g}_{cname}"]["mtp_cpu_rows"] = {str(k): [b[0], round(b[1], 1)] for k, b in res.items()}
            print(f"  gap {g} us  {cname}: GPU side {gpu:5.2f} ms, CPU side {cpu:5.2f} ms -> step {step:5.2f} ms = {solo:5.1f} tok/s solo; MTP x{'/'.join(map(str, mult))}: "
                  + " / ".join(f"{solo * m:.0f}" for m in mult))
    out[tree] = rec
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
