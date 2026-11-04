"""router_bench.py: MoE router seam, stock (softmax + argpartition + take_along_axis + sum + divide) vs decrows.router_topk, rows 1..8, isolated (graph built outside the timer, N independent copies)."""
import os, sys, time, json
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..", ".")]
import mlx.core as mx
from kyojin_mlx import decrows as DR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tests"))
from test_decrows import stock_router


def t(fn, N=100):
    mx.eval(fn(N))
    best = 1e9
    for _ in range(7):
        g = fn(N); t0 = time.perf_counter(); mx.eval(g); best = min(best, time.perf_counter() - t0)
    return best / N * 1e6


out = {}
for E, k in ((16, 2), (64, 6), (256, 8)):
    for R in (1, 2, 4, 8):
        lg = [(mx.random.normal((R, 1, E)) * 3).astype(mx.bfloat16) for _ in range(8)]; mx.eval(lg)
        a = t(lambda N: [stock_router(lg[i % 8], k, True) for i in range(N)])
        b = t(lambda N: [DR.router_topk(lg[i % 8], k, True) for i in range(N)])
        out[f"E{E}k{k}r{R}"] = (round(a, 1), round(b, 1))
print("ROUTER_BENCH " + json.dumps(out))
print("| router us/call | " + " | ".join(f"{r}r" for r in (1, 2, 4, 8)) + " |\n|---|---|---|---|---|")
for E, k in ((16, 2), (64, 6), (256, 8)):
    print(f"| E{E} k{k} stock | " + " | ".join(f"{out[f'E{E}k{k}r{R}'][0]}" for R in (1, 2, 4, 8)) + " |")
    print(f"| E{E} k{k} fused | " + " | ".join(f"{out[f'E{E}k{k}r{R}'][1]}" for R in (1, 2, 4, 8)) + " |")
