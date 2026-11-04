"""Per-group table of one DS4.1 decode token (40-layer mix) from slice profiles, against the Air's rooflines.
usage: step_table.py RESULTS_DIR [--json F] [--bw 125] [--launch-us 1.7] [--alu 270] [--vocab 129280]
Reads results_mac/macair10_{count,sync}_{before,after}_{ds2,ds2e}.json written by tools/prof_groups.py:
  ds2  layer 0 = compress + indexer layer, layer 1 = reuse layer;  ds2e = ds2 + an Engram layer 1;  layer -1 = outside the layers.
Layer mix of the real config: 2 local, 4 full, 4 reindex, 30 reuse (local = reuse without the compressor groups, full = reindex without the indexer
groups), 2 Engram layers. Bytes = weight bytes one token reads (top-k of the experts; KV and the Engram table rows excluded; the slice vocab head
is scaled to --vocab). Time = exclusive group time of the synced profile minus the eval floor (small groups read 0: use launches x launch cost).
Roofs: BW = bytes / --bw GB/s; ALU = routed weights / --alu Gw/s (K <= 3 trellis decode); launch = launches x --launch-us."""
import argparse, collections, json, os

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--json")
ap.add_argument("--bw", type=float, default=125.0); ap.add_argument("--launch-us", type=float, default=1.7)
ap.add_argument("--alu", type=float, default=270.0); ap.add_argument("--vocab", type=int, default=129280)
ap.add_argument("--slice-vocab", type=int, default=4096)
ap.add_argument("--routed-weights", type=float, default=6 * 3 * 5120 * 2304, help="routed weights read per layer and token (top-k x 3 x hidden x inter)")
a = ap.parse_args()
MIX = dict(reindex=4, full=4, local=2, reuse=30)
EXTRA_L = {"idx.": "full", "comp.": "local"}                  # groups a kind lacks relative to its template


def load(v, m):
    """Launches and bytes from the count profile; time = per layer and group of the quietest synced round (lowest eval floor)."""
    import glob
    c = json.load(open(os.path.join(a.dir, f"macair10_count_{v}_{m}.json")))
    files = sorted(glob.glob(os.path.join(a.dir, f"macair10_sync_{v}_{m}_r*.json"))) or [os.path.join(a.dir, f"macair10_sync_{v}_{m}.json")]
    s = min((json.load(open(f)) for f in files), key=lambda j: j["floor_us"])      # the round with the lowest eval floor = the quietest Air
    cr, sr = c["runs"][0], s["runs"][0]
    out = {}
    for l, groups in cr["bl"].items():
        d = {}
        for g, n in groups.items():
            ms = 1e3 * max(0.0, sr["bt"].get(l, {}).get(g, 0.0) - s["floor_us"] * 1e-6 * sr["bn"].get(l, {}).get(g, 0))
            d[g] = dict(launch=n, mb=cr["bb"].get(l, {}).get(g, 0.0) / 1e6, ms=ms)
        out[l] = d
    return out


def kinds(v):
    d2, d2e = load(v, "ds2"), load(v, "ds2e")
    reindex, reuse = d2["0"], d2["1"]
    full = {g: x for g, x in reindex.items() if not g.startswith("idx.")}
    local = {g: x for g, x in reuse.items() if not g.startswith("comp.")}
    engram = {g: x for g, x in d2e["1"].items() if g.startswith("engram.") and g != "engram.hash"}
    fixed = d2e["-1"]
    return dict(reindex=reindex, full=full, local=local, reuse=reuse), engram, fixed


def token(v):
    ks, engram, fixed = kinds(v)
    tot = collections.defaultdict(lambda: dict(launch=0, mb=0.0, ms=0.0))
    def add(g, x, k):
        t = tot[g]; t["launch"] += k * x["launch"]; t["mb"] += k * x["mb"]; t["ms"] += k * x["ms"]
    for kind, n in MIX.items():
        for g, x in ks[kind].items():
            add(g, x, n)
    for g, x in engram.items():
        add(g, x, 2)
    for g, x in fixed.items():
        add(g, x, 1)
    h = tot["head"]; s = a.vocab / a.slice_vocab                  # the head scales with the vocabulary
    h["mb"] *= s; h["ms"] *= s
    return tot


before, after = token("before"), token("after")
names = sorted(set(before) | set(after), key=lambda g: -max(after.get(g, {}).get("ms", 0), after.get(g, {}).get("mb", 0) / a.bw))
hdr = f"{'group':18s} {'launch b>a':>11s} {'MB':>8s} {'ms before':>10s} {'ms after':>9s} {'BW roof':>8s} {'ALU roof':>9s} {'launch roof':>11s}  note"
print(hdr)
rows = []
tb = dict(launch=0, ms=0.0); ta = dict(launch=0, ms=0.0, mb=0.0, bw=0.0, lr=0.0)
for g in names:
    b, f = before.get(g, dict(launch=0, mb=0, ms=0)), after.get(g, dict(launch=0, mb=0, ms=0))
    if g in ("<top>", "<end>"):
        continue
    bw = f["mb"] * 1e6 / (a.bw * 1e9) * 1e3
    alu = 40 * a.routed_weights / (a.alu * 1e9) * 1e3 if g == "moe.routed" else 0.0
    lr = f["launch"] * a.launch_us * 1e-3
    note = ""
    if f["ms"] and max(bw, alu):
        note = f"{f['ms'] / max(bw, alu):.2f}x of roof"
    print(f"{g:18s} {b['launch']:4d}>{f['launch']:<5d} {f['mb']:8.1f} {b['ms']:10.3f} {f['ms']:9.3f} {bw:8.3f} {alu:9.3f} {lr:11.3f}  {note}")
    rows.append(dict(group=g, launch_before=b["launch"], launch_after=f["launch"], mb=f["mb"], ms_before=b["ms"], ms_after=f["ms"], bw_roof_ms=bw, alu_roof_ms=alu, launch_roof_ms=lr))
    tb["launch"] += b["launch"]; tb["ms"] += b["ms"]
    ta["launch"] += f["launch"]; ta["ms"] += f["ms"]; ta["mb"] += f["mb"]; ta["bw"] += bw; ta["lr"] += lr
print(f"{'TOTAL':18s} {tb['launch']:4d}>{ta['launch']:<5d} {ta['mb']:8.1f} {tb['ms']:10.3f} {ta['ms']:9.3f} {ta['bw']:8.3f} {'':9s} {ta['lr']:11.3f}")
if a.json:
    json.dump(dict(mix=MIX, rows=rows, total_before=tb, total_after=ta, bw=a.bw, launch_us=a.launch_us, alu=a.alu), open(a.json, "w"), indent=1)
