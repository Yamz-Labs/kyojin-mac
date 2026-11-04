"""Is the per-step variation of the recorded command stream affine in the step index (same parity)? usage: rp_affine.py DUMPDIR NSTEPS
For each command and each field (grid, threadgroup, binding size / offset, constant words) of steps t, t+2, t+4, ...: affine if the second difference is 0."""
import re, sys, collections, struct
d, n = sys.argv[1], int(sys.argv[2])
def parse(p):
    out = []
    for ln in open(p):
        t = ln.split(); f = {"k": t[1], "disp": t[2], "bar": t[5], "v": {}}
        f["v"]["grid"] = tuple(int(x) for x in t[3][5:].split(",")); f["v"]["tg"] = tuple(int(x) for x in t[4][3:].split(","))
        for x in t[6:]:
            if x.startswith("tgm"): continue
            m = re.match(r"\[(\d+):(.*)\]$", x); s = int(m.group(1)); b = m.group(2)
            if b.startswith("B"):
                ln_, hx = b[1:].split(":"); raw = bytes.fromhex(hx)
                for i in range(0, len(raw) - len(raw) % 4, 4): f["v"][f"b{s}.w{i // 4}"] = struct.unpack("<i", raw[i:i + 4])[0]
            else:
                mm = re.match(r"([nx])(\d+)\+(\d+)", b); f["v"][f"s{s}.size"] = int(mm.group(2)); f["v"][f"s{s}.off"] = int(mm.group(3)); f["v"][f"s{s}.role"] = mm.group(1)
        out.append(f)
    return out
steps = [parse(f"{d}/step{i}.txt") for i in range(n)]
print("cmds per step", [len(s) for s in steps])
for par in (0, 1):
    idx = [i for i in range(n) if i % 2 == par]
    S = [steps[i] for i in idx]
    if len({len(s) for s in S}) != 1: print("parity", par, "signature lengths differ", [len(s) for s in S]); continue
    tot = aff = const = 0; bad = collections.Counter(); badex = {}
    for c in range(len(S[0])):
        if len({s[c]["k"] for s in S}) != 1: bad["kernel-differs"] += 1; continue
        keys = set().union(*[set(s[c]["v"]) for s in S])
        for k in sorted(keys):
            vals = [s[c]["v"].get(k) for s in S]
            if any(isinstance(v, str) for v in vals):
                if len(set(vals)) > 1: bad["role"] += 1
                continue
            flat = []
            for v in vals:
                flat.append(v if isinstance(v, tuple) else (v,))
            if len(set(flat)) == 1: const += 1; continue
            tot += 1
            ok = True
            for j in range(len(flat[0])):
                seq = [f[j] for f in flat]
                if any(seq[i + 2] - 2 * seq[i + 1] + seq[i] != 0 for i in range(len(seq) - 2)): ok = False
            if ok: aff += 1
            else: bad[S[0][c]["k"][:44] + ":" + k] += 1; badex.setdefault(S[0][c]["k"][:44] + ":" + k, (c, flat[:6]))
    print(f"parity {par}: fields constant {const}, varying {tot}, affine in n {aff}, NOT affine {tot - aff}")
    for k, v in bad.most_common(14): print("   not affine:", k, v, badex.get(k))
