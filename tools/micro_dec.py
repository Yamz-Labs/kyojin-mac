"""Decode cost of the dense linears of each layer (the part the weight cache removes): min of N evals of kernels.decode_wr per Exl3Linear. usage: micro_dec.py PACK [reps]"""
import sys, time, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, kernels as KN, wcache as WC
pack = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
model, spec, cfg = P.build_model(pack); P.patch_model(model, pack, spec, cfg)
lm = model.language_model
for li, layer in enumerate(lm.layers):
    tot = 0.0; parts = []
    for lin in WC._linears(layer):
        mx.eval(KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)); ts = []
        for _ in range(reps):
            time.sleep(0.3); t0 = time.perf_counter(); mx.eval(KN.decode_wr(lin._words, lin.K, lin._kt, lin._nt, lin.cb)); ts.append(time.perf_counter() - t0)
        tot += min(ts); parts.append(f"{lin.in_features}x{lin.out_features}:{min(ts)*1e3:.2f}")
    print(f"L{li}: sum decode {tot*1e3:6.1f} ms over {len(parts)} linears | " + " ".join(parts), flush=True)
