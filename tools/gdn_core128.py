import sys,time,os
sys.path.insert(0,os.path.expanduser("~/kyojin-mlx"))
import mlx.core as mx
from kyojin_mlx import gdn_nax as G
Hk,Hv,D,T=16,32,128,2048
mx.random.seed(0)
q=(mx.random.normal((1,T,Hk,D))*0.1).astype(mx.bfloat16)
k=(mx.random.normal((1,T,Hk,D))*0.1).astype(mx.bfloat16)
v=mx.random.normal((1,T,Hv,D)).astype(mx.bfloat16)
g=mx.sigmoid(mx.random.normal((1,T,Hv))+3)
b=mx.sigmoid(mx.random.normal((1,T,Hv)))
s=mx.zeros((1,Hv,D,D),mx.float32)
def run(tg):
    G.gated_delta_nax.__globals__["_TG"]=tg
    ts=[]
    for i in range(8):
        mx.eval(q);t=time.time();y,so=G.gated_delta_nax(q,k,v,g,b,s);mx.eval(y,so);ts.append((time.time()-t)*1000)
    return min(ts),y
print("core ms best",run(None)[0])
