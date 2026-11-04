"""Per-shape row windows of the few-row kernels vs MLX quantized_matmul, measured on THIS chip in the HOT regime (warm-up first), for every eligible
(N, K) of a pack: q4mma (simdgroup 8x8 MMA, 2..q4mma.MAX_ROWS rows) and, on a chip with neural accelerators, q4nax (NAX matmul2d, 2..32 rows, best
(u, nb, ks) launch shape of the shape). Each kernel gets the longest run of row counts where it beats MLX by --margin
(q4nax: also beats q4mma); the q4nax window is checked first. Writes "min_rows" {NxK: [lo, hi]} and "nax" {NxK: [lo, hi, u, nb, ks]} read by kyojin_mlx.q4mma.load_tuning.
usage: q4mma_tune.py PACK [--out FILE] [--warm 40] [--margin 0.97] [--rows 2..32]"""
import argparse, json, os, statistics, time
import mlx.core as mx
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--out"); ap.add_argument("--warm", type=float, default=40); ap.add_argument("--margin", type=float, default=0.97)
ap.add_argument("--passes", type=int, default=5); ap.add_argument("--rows", default="2,3,4,5,6,8,9,10,12,14,16,20,24,28,32"); a = ap.parse_args()
from kyojin_mlx import q4mma as Q, mlxpack
model, tok = mlxpack.load(a.pack); lm = model.language_model
import mlx.nn as nn
shapes = {}
for n, m in lm.named_modules():
    if Q._ok(m):
        shapes[(m.weight.shape[0], m.weight.shape[1] * 8)] = shapes.get((m.weight.shape[0], m.weight.shape[1] * 8), 0) + 1
emb = lm.model.embed_tokens
if isinstance(emb, nn.QuantizedEmbedding) and emb.bits == 4 and emb.group_size == 64:
    shapes[(emb.weight.shape[0], emb.weight.shape[1] * 8)] = 1
del model
z = mx.random.normal((2048, 2048)).astype(mx.float16); t0 = time.perf_counter()
while time.perf_counter() - t0 < a.warm: mx.eval([z @ z for _ in range(8)])
NAXC = [(2, 2, 1), (2, 4, 1), (2, 2, 2), (1, 2, 2)] if Q.nax_on() else []
ROWS = [int(v) for v in a.rows.split(",")]


def run(win, rows):
    """Longest run of consecutive measured row counts won (ties: the higher run) -> [lo, hi] (never: [hi + 1, hi])."""
    best, lo, prev = None, None, None
    for R in rows + [None]:
        if R is not None and win[R]:
            lo = R if lo is None else lo
        elif lo is not None:
            if best is None or prev - lo >= best[1] - best[0]: best = (lo, prev)
            lo = None
        prev = R
    if best is None: return [rows[-1] + 1, rows[-1]]
    hi = best[1]
    return [best[0], hi]


out, outn = {}, {}
for (N, K), cnt in sorted(shapes.items()):
    L = max(2, min(64, int(2000 / (N * K / 2 / 1e5)) + 1))   # >= ~2 ms of reads per timed batch (small shapes: launch noise)
    ws = [mx.random.randint(0, 2**31 - 1, (N, K // 8)).astype(mx.uint32) * mx.array(2654435761, dtype=mx.uint32) for _ in range(L)]
    scs = [mx.random.uniform(0.001, 0.02, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    bis = [mx.random.uniform(-0.1, 0.0, (N, K // 64)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ws, scs, bis)
    cf = [c for c in NAXC if K % (128 * c[0]) == 0 and N % 32 == 0]
    med = {}
    for R in ROWS:
        x = mx.random.normal((R, K)).astype(mx.bfloat16); mx.eval(x)
        f = {"mlx": lambda i: mx.quantized_matmul(x, ws[i % L], scs[i % L], bis[i % L], transpose=True, group_size=64, bits=4)}
        if R <= Q.MAX_ROWS:
            f["q"] = lambda i: Q.q4mma3(Q.prep_x(x), ws[i % L], scs[i % L], bis[i % L], R, Q.NT, Q.KS)   # the x prep launch is charged to each call
        for c in cf:
            f[c] = (lambda c: lambda i: Q.q4nax(Q.prep_xn(x, c[0]), ws[i % L], scs[i % L], bis[i % L], R, *c))(c)
        T = {k: [] for k in f}
        for p in range(a.passes + 1):
            for k, fn in f.items():
                t0 = time.perf_counter(); mx.eval([fn(i) for i in range(L)]); dt = (time.perf_counter() - t0) / L * 1e6
                if p: T[k].append(dt)
        med[R] = {k: min(v) for k, v in T.items()}                          # min over passes: robust to bursts of other GPU work
    c_best = min(cf, key=lambda c: sum(med[R][c] / med[R]["mlx"] for R in ROWS)) if cf else None   # one launch shape per (N, K)
    nax = {R: med[R][c_best] if c_best else 1e30 for R in ROWS}
    q = {R: med[R].get("q", 1e30) for R in ROWS}
    winq = {R: q[R] <= a.margin * med[R]["mlx"] for R in ROWS}             # vs MLX only: the nax window is checked first, and KYOJIN_Q4NAX=0 keeps a full q4mma window
    winn = {R: nax[R] <= a.margin * med[R]["mlx"] and nax[R] < q[R] for R in ROWS}
    out[f"{N}x{K}"] = run(winq, ROWS)
    if c_best:
        outn[f"{N}x{K}"] = run(winn, ROWS) + list(c_best)
    print(f"{N}x{K} (x{cnt}) mlx/q/nax{c_best}: " + " ".join(f"R{R} {med[R]['mlx']:.0f}/{q[R]:.0f}/{nax[R]:.0f}".replace(f"/{1e30:.0f}", "/-") for R in ROWS)
          + f" -> q4mma {out[f'{N}x{K}']} nax {outn.get(f'{N}x{K}')}", flush=True)
    del ws, scs, bis
dst = a.out or os.path.join(os.path.dirname(Q.__file__), "q4mma_tuning.json")
doc = json.load(open(dst)) if os.path.exists(dst) else {}
dev = doc.setdefault("devices", {}).setdefault(Q.device_name(), {})                                       # keyed by chip; other chips are untouched
dev.setdefault("min_rows", {}).update(out)
if outn: dev.setdefault("nax", {}).update(outn)
doc["note"] = "hot regime, x transpose charged to every call; per shape min_rows [lo, hi] = rows served by q4mma, nax [lo, hi, u, nb, ks] = rows served by q4nax, checked first (lo > hi: never); tools/q4mma_tune.py PACK [--out FILE] adds the shapes of a pack for THIS chip"
json.dump(doc, open(dst, "w"), indent=1)
print("wrote", dst)
