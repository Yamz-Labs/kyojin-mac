import numpy as np, mlx.core as mx, os, sys
from kyojin_mlx import patch as P
model, spec, cfg = P.build_model(sys.argv[1]); P.patch_model(model, sys.argv[1], spec, cfg)
lm = model.language_model
ids = mx.array([np.random.default_rng(3).integers(1, 4000, 200).tolist()])
from kyojin_mlx import headlast as HL
with HL.last_row(): lg = lm(ids, cache=lm.make_cache()).logits
mx.eval(lg); print("KYOJIN_HEAD_LAST =", os.environ.get("KYOJIN_HEAD_LAST"), "logits", lg.shape)
np.save(sys.argv[2], np.array(lg[:, -1].astype(mx.float32)))
