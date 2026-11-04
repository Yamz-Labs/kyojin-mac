"""mlx-lm batch_generate reference for the multi-user bench: bench_mlxlm_batch.py MLX_PACK [PROMPTS.json]; prints generation tok/s (all users) for N=1,2,4,8."""
import json,sys,statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate
model,tok=load(sys.argv[1])
P=json.load(open(sys.argv[2] if len(sys.argv)>2 else "results_mac/macmu2/prompts8.json"))["prompts"]
def run(N):
    r=batch_generate(model,tok,P[:N],max_tokens=64)
    s=r.stats; return s.generation_tps, s.generation_tokens
run(2)
res={1:[],2:[],4:[],8:[]}
for _ in range(3):
    for N in res: res[N].append(run(N))
for N,v in res.items(): print("mlxlm N=%d gen_tps med %.1f toks %s"%(N,statistics.median(x[0] for x in v),v[0][1]),[round(x[0],1) for x in v])
