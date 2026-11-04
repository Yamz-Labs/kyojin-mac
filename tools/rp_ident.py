"""Buffer identity census between recorded steps i and i+P (usage: rp_ident.py DIR N P): how many (command, slot) bindings name another buffer / offset."""
import re, sys
d, n, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def parse(p):
    out = []
    for ln in open(p):
        t = ln.split(); b = {}
        for x in t[6:]:
            m = re.match(r"\[(\d+):([nx])(\d+)\+(\d+)@([0-9a-f]+)\]$", x)
            if m: b[int(m.group(1))] = (m.group(2), int(m.group(3)), int(m.group(4)), m.group(5))
        out.append((t[1], b))
    return out
S = [parse(f"{d}/step{i}.txt") for i in range(n)]
for i in range(0, n - P):
    a, b = S[i], S[i + P]
    if len(a) != len(b): print(f"step {i} vs {i + P}: lengths {len(a)} {len(b)}"); continue
    tot = same = diffbuf = diffoff = diffsize = 0
    for (ka, ba), (kb, bb) in zip(a, b):
        for k in ba:
            if k not in bb: continue
            tot += 1; x, y = ba[k], bb[k]
            if x[3] == y[3] and x[2] == y[2]: same += 1
            elif x[3] != y[3]: diffbuf += 1
            else: diffoff += 1
    print(f"step {i} vs {i + P}: bindings {tot}: same buffer+offset {same}, other buffer {diffbuf}, same buffer other offset {diffoff}")
