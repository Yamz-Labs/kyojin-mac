import sys, numpy as np, mlx.core as mx
from kyojin_mlx import patch as P, spec as S, headlast as HL
model, spec, cfg = P.build_model(sys.argv[1]); P.patch_model(model, sys.argv[1], spec, cfg)
for T, ch in ((200, None), (1500, 700), (4096, 2048)):
    ids = np.random.default_rng(5).integers(1, 4000, T).tolist(); out = {}
    for on in (True, False):
        HL.ENABLED = on; c, lg, _, _ = S.prefill(model, ids, (), ch); out[on] = (np.array(lg), [np.array(x) for x in c[0].state] if hasattr(c[0], "state") else None)
    a, b = out[True][0], out[False][0]
    print(T, "bitwise last-row logits:", np.array_equal(a, b), "maxabs", float(np.abs(a-b).max()), "argmax", a.argmax() == b.argmax(), flush=True)
