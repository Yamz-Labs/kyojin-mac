"""Stage profile of quantize_big on the first ROWS rows of the real tied head: head_prof.py SRC ROWS CHUNK [K]"""
import sys, time, os, numpy as np, mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import conv_mlx as C
src, rows, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 6; SW = int(sys.argv[5]) if len(sys.argv) > 5 else 3
W = mx.load(f"{src}/model.safetensors-00001-of-00001.safetensors"); w = W["model.language_model.embed_tokens.weight"][:rows]
k = w.shape[1]
x = np.random.default_rng(1).standard_normal((4096, k)).astype(np.float32); H = mx.array(x.T @ x)
hd = C.finalize_H(H, 4096, 3)
C.STAGE.clear(); mx.reset_peak_memory(); t0 = time.perf_counter()
r = C.quantize_big(w, hd, K, 3, "/tmp/head_prof_scratch.npy", chunk, SW)
tot = time.perf_counter() - t0
print(f"sw={SW} g={r['g']:.5f} rows={rows} chunk={chunk} K={K} total {tot:.1f}s tiles={rows*k//256} -> {rows*k//256/tot:.0f} tiles/s  stages {({a: round(b, 1) for a, b in C.STAGE.items()})} peakGB {mx.get_peak_memory()/2**30:.2f} proxy {r['proxy']:.5f}")
import hashlib; print("sha", hashlib.sha1(np.ascontiguousarray(r["trellis"]).tobytes()).hexdigest()[:12])
