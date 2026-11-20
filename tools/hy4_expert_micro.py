"""Routed-expert block micro on Hy4 shapes (H 1536 latent, F 1536, K 16 slots/row), synthetic stored Q3_K / Q4_K stacks, as served (ForeignSwitchGLU, flags from env).
L layer stacks are cycled so the working set (> SLC) is read from DRAM like the model does. Prints ms per layer-call and GB/s of the expert bytes touched.
  python tools/hy4_expert_micro.py [rows list, default 1,4] [triples, default q3q3q4,q4q4q4] [layers, default 24]"""
import os, sys, time
import numpy as np
import mlx.core as mx
from kyojin_mlx import foreign_linear as FL, foreign_gguf as GF

BS = {"Q3_K": (110, 108), "Q4_K": (144, 0)}
H = F = 1536; E = 32; K = 16


def stack(tname, n, k, rng):
    bs, nb = BS[tname]; row = k // 256 * bs; packs = []
    for _ in range(E):
        raw = rng.integers(0, 256, (n, row), dtype=np.uint8)
        for b in range(k // 256):
            o = b * bs + nb; raw[:, o:o + 2] = np.array([0x00, 0x1c], np.uint8)
            if tname == "Q4_K": raw[:, o + 2:o + 4] = np.array([0x00, 0x18], np.uint8)
        packs.append(GF.to_pack(tname, raw, k))
    return FL.DevPack(packs)


rows_l = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,4").split(",")]
trip = (sys.argv[2] if len(sys.argv) > 2 else "q3q3q4,q4q4q4").split(",")
L = int(sys.argv[3]) if len(sys.argv) > 3 else 24
rng = np.random.default_rng(3)
print(f"cfg NBRK {FL.NBRK} NBR {FL.NBR} GATEUP_PAIR {FL.GATEUP_PAIR} L {L}", flush=True)
for tr in trip:
    t3 = [{"q3": "Q3_K", "q4": "Q4_K"}[tr[i:i + 2]] for i in (0, 2, 4)]
    sws = [FL.ForeignSwitchGLU(stack(t3[0], F, H, rng), stack(t3[1], F, H, rng), stack(t3[2], H, F, rng)) for _ in range(L)]
    per_expert = sum(BS[t][0] * (1536 // 256) * 1536 for t in t3)
    for rows in rows_l:
        xs = [mx.array(rng.standard_normal((rows, 1, H)).astype(np.float32) * 0.5).astype(mx.float16) for _ in range(L)]
        share = os.environ.get("HY4M_SHARE", "0") == "1"           # 1 = every row picks the same K experts (distinct fraction 1/rows): is a repeated expert read free (cache) or paid?
        ids = [mx.array(np.stack([(rng.choice(E, K, replace=False) if (not share or r == 0) else None) for r in range(rows)] if not share else [rng.choice(E, K, replace=False)] * rows)[:, None, :].astype(np.uint32)) for _ in range(L)]
        if os.environ.get("HY4M_SORT", "0") == "1":              # expert-major slot order (timing probe only: the row of a slot is then arbitrary): equal experts adjacent in the grid
            ids = [mx.array(np.sort(np.array(i).reshape(-1)).reshape(i.shape)) for i in ids]
        def run(n):
            y = None
            for i in range(n):
                y = sws[i % L](xs[i % L] if y is None else xs[i % L] + y[:, :1, 0, :1].astype(mx.float16) * 0, ids[i % L])   # the data dependence chains the layers like the model
            mx.eval(y)
        run(L); ts = []
        for _ in range(4):
            t = time.perf_counter(); run(2 * L); ts.append((time.perf_counter() - t) / (2 * L))
        ms = sorted(ts)[1] * 1e3
        gb = per_expert * rows * K / 1e9
        print(f"MICRO {tr} rows {rows} share {int(share)} NBRK {FL.NBRK}: {ms:.3f} ms/layer-call, expert bytes (slots, no dedup) {gb * 1e3:.1f} MB -> {gb / (ms / 1e3) / 1e3:.2f} TB/s", flush=True)
