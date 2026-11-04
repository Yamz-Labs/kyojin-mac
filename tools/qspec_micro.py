"""ms of the pieces of one draft step (sync per call, min of 20) on Qwen3.5 EXL3 + MTP head. usage: qspec_micro.py PACK BF16"""
import sys, time
import mlx.core as mx
from kyojin_mlx import exl3_linear as EL, qspec as QS
model, _ = EL.load_exl3(sys.argv[1]); QS.row_linears(model); lm = model.language_model
def t(f, n=20):
    best = 1e9
    for i in range(n + 3):
        t0 = time.perf_counter(); mx.eval(f()); best = min(best, time.perf_counter() - t0)
    return best * 1e3
H = lm.args.hidden_size
o = mx.random.normal((1, 1, H)).astype(mx.float16)
print("lm_head exl3 1 row", round(t(lambda: lm.lm_head(o)), 2))
for kk in (16384, 32768, 65536):
    dh = QS.load_draft_head(sys.argv[2], kk); print(f"draft head {kk} 4bit", round(t(lambda: dh(o)), 2))
for qb in (0, 4):
    mtp = QS.load_mtp(model, sys.argv[2], quant_bits=qb)
    sp = QS.QSpec(model, mtp, depth=1); sp.mcache = QS.KVCache()
    sp.mtp_rows(mx.array([5] * 200), mx.random.normal((1, 200, H)).astype(mx.float16)); mx.eval(sp.mcache.keys)
    print(f"mtp layer q{qb} 1 row (200 ctx)", round(t(lambda: (sp.mtp_rows(mx.array([7]), o), sp.mcache.trim(1))[0]), 2))
    print(f"mtp layer q{qb} 4 rows", round(t(lambda: (sp.mtp_rows(mx.array([7] * 4), mx.concatenate([o] * 4, 1)), sp.mcache.trim(4))[0]), 2))
print("trunk 1 row", end=" ")
from mlx_lm.models.cache import make_prompt_cache
c = make_prompt_cache(model); lm.model(mx.array([[5] * 100]), c)
print(round(t(lambda: (lm.model(mx.array([[7]]), c)), 10), 2))
