import sys, time, os, mlx.core as mx
from kyojin_mlx import layer_pf as LP
from kyojin_mlx.serve.backends import load_backend
be,_=load_backend(sys.argv[1]); R=int(sys.argv[2])
orig=LP._layer_call; acc=[0.0,0]
def wrap(self,x,mask=None,cache=None):
    mx.eval(x); mx.synchronize(); t=time.perf_counter()
    o=orig(self,x,mask,cache); mx.eval(o); mx.synchronize(); acc[0]+=time.perf_counter()-t; acc[1]+=1; return o
LP._layer_call=wrap
from mlx_lm.models import qwen3_5 as Q5
Q5.DecoderLayer.__call__=wrap
ids=[(i*7919)%5000+100 for i in range(R)]
for _ in range(2):
    acc[:]=[0.0,0]; c=be.make_cache(); t=time.perf_counter(); mx.eval(be.prefill_chunk(c,ids)); tot=time.perf_counter()-t
print("S1",os.environ.get("S1"),"layers ms",round(acc[0]*1e3),"total ms",round(tot*1e3))
