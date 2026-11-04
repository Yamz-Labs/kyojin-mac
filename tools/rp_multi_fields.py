"""N-session verify pass: are the varying stream fields additive in the per-session positions? Records passes with independently shifted session offsets (shifts keep the phase), compares fields."""
import ctypes, os, sys, re, struct, itertools
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay", "build", "librp.dylib")); lib.rp_install(1)
import numpy as np, mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import dsv4_core as DC
N = int(sys.argv[2]); D = 3
be, cfg = load_backend(sys.argv[1], mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; rng = np.random.default_rng(7); base = rng.integers(16, V - 16, 61).tolist()
caches = []
for u in range(N):
    c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16 + u] + (base * 4)[:100 + 10 * u], draft=d); caches.append(c)
bc = be.batch(caches); st = DC.stack_rows(bc[0])
toks = [[int(t) for t in rng.integers(16, V - 16, D + 1)] for _ in range(N)]
P0 = [r.offset for r in st.rows]; print("base offsets", P0)
def rec(shift):
    for r, p, d in zip(st.rows, P0, shift): r.offset = p + d
    st._plan = None
    mx.synchronize(); mx.clear_cache(); lib.rp_rec_begin(90); lib.rp_rec_snap_born(90, 1)
    ps = be.spec_verify_bc(bc, toks); pend = [a for pd in st.pend.values() for a in vars(pd).values() if isinstance(a, mx.array)]
    mx.eval(ps.logits, *pend, *list(ps.taps.values())); mx.synchronize(); n = lib.rp_rec_end()
    lib.rp_dump(90, b"/tmp/mf.txt"); lib.rp_hostconst_dump(90, b"/tmp/mfh.txt")
    st._counts = None; st.pend = {}
    for r, p in zip(st.rows, P0): r.offset = p
    f = {}
    names = []
    for i, ln in enumerate(open("/tmp/mf.txt")):
        t = ln.split(); names.append(t[1]); f[(i, "grid")] = tuple(int(x) for x in t[3][5:].split(",")); f[(i, "tg")] = tuple(int(x) for x in t[4][3:].split(","))
        for x in t[6:]:
            m = re.match(r"\[(\d+):B(\d+):([0-9a-f]*)\]$", x)
            if m:
                raw = bytes.fromhex(m.group(3))
                for j in range(0, len(raw) - len(raw) % 4, 4): f[(i, "b%s.w%d" % (m.group(1), j // 4))] = struct.unpack("<i", raw[j:j + 4])[0]
    for ln in open("/tmp/mfh.txt"):
        c, s, n, h = ln.split(); raw = bytes.fromhex(h)
        for j in range(0, len(raw) - len(raw) % 4, 4): f[(int(c), "hb%s.w%d" % (s, j // 4))] = struct.unpack("<i", raw[j:j + 4])[0]
    return n, names, f
S = 2
shifts = [[0] * N] + [[(S * 12 if s == u else 0) for s in range(N)] for u in range(N)] + [[(S * 12 if s in (u, (u + 1) % N) else 0) for s in range(N)] for u in range(N)]
for _ in range(2): rec([0] * N)
R = [rec(sh) for sh in shifts]
print("command counts", [r[0] for r in R], "same stream:", len({tuple(r[1]) for r in R}) == 1)
b = R[0][2]; keys = set(b)
for r in R[1:]: keys &= set(r[2])
vary = [k for k in keys if any(r[2][k] != b[k] for r in R[1:])]
print("fields", len(keys), "varying with the positions", len(vary))
def val(f, k):
    v = f[k]; return v if isinstance(v, tuple) else (v,)
bad = []
for u in range(N):
    for w in range(N):
        pass
for u in range(N):
    pair = R[1 + N + u][2]; a = R[1 + u][2]; w = R[1 + (u + 1) % N][2]
    for k in vary:
        x = val(pair, k); y = tuple(p + q - r for p, q, r in zip(val(a, k), val(w, k), val(b, k)))
        if x != y: bad.append((u, k, x, y))
print("non-additive field instances (pair shifts vs sum of single shifts):", len(bad), "of", len(vary) * N)
seen = set()
for u, k, x, y in bad:
    if k[0] not in seen and len(seen) < 8:
        seen.add(k[0]); print("   cmd", k[0], k[1], "pair", x, "additive", y)

# per-session response: the fields that move when only session u moves, and by how much (shift = S * 12 positions)
resp = []
for u in range(N):
    a = R[1 + u][2]
    ch = sorted(((k[0], str(k[1])), tuple(x - y for x, y in zip(val(a, k), val(b, k)))) for k in vary if a[k] != b[k])
    resp.append(ch)
print("fields changed by session shifts:", [len(r) for r in resp])
cmds = [[c for (c, _), _ in r] for r in resp]
print("same command sequence for all sessions:", all(c == cmds[0] for c in cmds), "| same deltas (as a multiset):", all(sorted(d for _, d in r) == sorted(d for _, d in resp[0]) for r in resp))
for u in range(1, N):
    diff = [(x, y) for x, y in zip(resp[0], resp[u]) if x[1] != y[1]]
    print("  session", u, "pairs with a different delta than session 0:", len(diff), diff[:3])
