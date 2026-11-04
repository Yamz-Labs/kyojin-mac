import runpy, sys, os
sys.path.insert(0, os.getcwd())
from kyojin_mlx import fp4pool, dsa_attend
n = {"view": 0, "packed": 0, "dense_calls": 0}
o = fp4pool.PackedView.__init__
def init(self, *a): n["view"] += 1; o(self, *a)
fp4pool.PackedView.__init__ = init
op = dsa_attend.attend_packed
def ap(*a, **k): n["packed"] += 1; return op(*a, **k)
dsa_attend.attend_packed = ap
od = dsa_attend.attend
def ad(*a, **k): n["dense_calls"] += 1; return od(*a, **k)
dsa_attend.attend = ad
from kyojin_mlx import fuse_attn, dsv4_core
n["fa_core"]=0; n["fa_attend"]=0; n["rows_core"]=0
oc = fuse_attn.core
def c(*a, **k): n["fa_core"] += 1; return oc(*a, **k)
fuse_attn.core = c
oa = fuse_attn.attend
def fa(*a, **k): n["fa_attend"] += 1; n["dt"] = str(a[0].dtype); return oa(*a, **k)
fuse_attn.attend = fa
orc = dsv4_core.core
def rc(*a, **k): n["rows_core"] += 1; return orc(*a, **k)
dsv4_core.core = rc
import atexit; atexit.register(lambda: print("CNT", n))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
