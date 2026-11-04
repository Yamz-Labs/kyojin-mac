"""Dispatch census by stage of ONE user's TQ attention step for one layer (4B shapes), evaluated state."""
import sys, os, re, collections, tempfile
import mlx.core as mx
from kyojin_mlx import tq, kvwin as KW, multiuser as MU
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
Hq, Hk, D = 16, 4, 256
KW.install()
c = tq.TQKVCache(6, 6, 4, 256)
for s in range(0, CTX, 1024):
    c.update_and_fetch(mx.random.normal((1, Hk, 1024, D)).astype(mx.float16), mx.random.normal((1, Hk, 1024, D)).astype(mx.float16))
c.update_and_fetch(mx.random.normal((1, Hk, 7, D)).astype(mx.float16), mx.random.normal((1, Hk, 7, D)).astype(mx.float16)); mx.eval(c.state)
k = mx.random.normal((1, Hk, 1, D)).astype(mx.float16); v = mx.random.normal((1, Hk, 1, D)).astype(mx.float16); q = mx.random.normal((1, Hq, 1, D)).astype(mx.float16)
mx.eval(k, v, q)
VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided", "StopGradient", "Depends"}
def census(name, outs):
    path = os.path.join(tempfile.gettempdir(), "c85.dot"); mx.export_to_dot(path, outs); txt = open(path).read()
    labels = re.findall(r'\[label ?= ?"([^"]*)", shape=rectangle\]', txt)
    cn = collections.Counter(labels)
    print(f"{name:22s} dispatches {sum(v for kk, v in cn.items() if kk not in VIEWS):3d}: " + ", ".join(f"{kk}:{vv}" for kk, vv in cn.most_common() if kk not in VIEWS))
H = Hk
kv32 = tq.rotate(mx.concatenate([k, v], axis=1), D)
census("rotate(K,V)", [kv32[:, :H].astype(mx.float16), kv32[:, H:].astype(mx.float16)])
ks, vs = c.update_and_fetch(k, v)
census("update_and_fetch", [*c.kq, *c.vq, c.rk, c.rv])
census("attend", [c.attend(q, D ** -0.5)])
