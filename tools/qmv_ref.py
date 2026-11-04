import time, statistics, numpy as np, mlx.core as mx
mx.set_cache_limit(256<<20)
POOL=24
for (kin,nout) in ((2560,9216),(9216,2560)):
  ws=[]
  for i in range(POOL):
    w=mx.random.normal((nout,kin),key=mx.random.key(i)).astype(mx.float16)
    ws.append(mx.quantize(w,group_size=64,bits=4)); mx.eval(*ws[-1])
  x=mx.random.normal((1,kin)).astype(mx.float16); mx.eval(x)
  nbytes=sum(a.nbytes for a in ws[0])
  f=lambda q: mx.quantized_matmul(x,q[0],q[1],q[2],transpose=True,group_size=64,bits=4)
  mx.eval(*[f(q) for q in ws[:3]]); ts=[]
  for r in range(9):
    t0=time.perf_counter(); mx.eval(*[f(ws[i%POOL]) for i in range(24)]); ts.append((time.perf_counter()-t0)/24*1e6)
  print(kin,nout,f"bytes {nbytes/1e6:.2f}MB min {min(ts):.1f} med {statistics.median(ts):.1f} -> {nbytes/min(ts)/1e3:.0f} GB/s",flush=True)
