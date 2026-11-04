import sys, collections, os
import mlx.core as mx, mlx.nn as nn
from kyojin_mlx import patch as P
from kyojin_mlx.exl3_linear import Exl3Linear
pack = sys.argv[1]
model, spec, cfg = P.build_model(pack)
rep = P.patch_model(model, pack, spec, cfg)
print(rep)
lm = model.language_model
c = collections.Counter()
for n, m in lm.named_modules():
    t = type(m).__name__
    if t in ("Linear",) or isinstance(m, Exl3Linear) and False:
        print("DENSE LINEAR", n, m.weight.shape, m.weight.dtype)
    c[t] += 1
print(c)
for k, v in lm.parameters().items() if False else []:
    pass
from mlx.utils import tree_flatten
for k, v in tree_flatten(lm.parameters()):
    if "layers.0." in k or "layers" not in k:
        print(k, tuple(v.shape), v.dtype)
