"""Exactness + cost of a multi-row (S = 1..9) trunk forward on Qwen3.5 EXL3 vs sequential T=1 steps.
usage: qspec_probe.py PACK [--rowsplit 16] [--n 12]"""
import argparse, time, statistics
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from kyojin_mlx import exl3_linear as EL
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--rowsplit", type=int, default=16)
ap.add_argument("--rowinv", type=int, default=1); ap.add_argument("--maxrows", type=int, default=0); ap.add_argument("--n", type=int, default=12); ap.add_argument("--plen", type=int, default=300); a = ap.parse_args()
EL.ROW_SPLIT_MAX = a.rowsplit
from kyojin_mlx import kernels as KN
if a.maxrows: KN.MAX_ROWS = a.maxrows; EL.GEMV_MAX_ROWS = a.maxrows
model, _ = EL.load_exl3(a.pack); lm = model.language_model
from kyojin_mlx import qspec as QS
print('row linears', QS.row_linears(model) if a.rowinv else 0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.pack)
text = "The history of the Roman Empire begins with the founding of Rome in 753 BC. Rome grew from a small town on the Tiber into a vast empire. " * 40
ids = tok.encode(text)[:a.plen]
def fwd(x, cache):
    h = lm.model(x, cache); return lm.lm_head(h), h
def fresh():
    c = make_prompt_cache(model); lg, h = fwd(mx.array(ids)[None], c); mx.eval(lg, *[x for cc in c for x in (cc.state if hasattr(cc, "state") else [])]); return c, lg
# reference: greedy solo decode n tokens, keep logits
c, lg = fresh(); t = mx.argmax(lg[:, -1], -1); toks = [t.item()]; ref = []
for i in range(a.n + 9):
    lg, _ = fwd(t[:, None], c); mx.eval(lg); ref.append(lg[0, 0]); t = mx.argmax(lg[:, -1], -1); toks.append(t.item())
print("solo toks", toks[:10])
for S in range(2, 9):
    c, _ = fresh(); x = mx.array(toks[:S])[None]
    lg, _ = fwd(x, c); mx.eval(lg)
    eq = [bool(mx.array_equal(lg[0, j], ref[j])) for j in range(S)]
    mx.eval(); d = [float(mx.abs(lg[0, j].astype(mx.float32) - ref[j].astype(mx.float32)).max()) for j in range(S)]
    am = [int(mx.argmax(lg[0, j]).item()) == toks[j + 1] for j in range(S)]
    print(f"S={S} bitwise rows {''.join('1' if e else '0' for e in eq)} argmax_ok {''.join('1' if e else '0' for e in am)} maxdiff {max(d):.3g}")
# cost
def timeit(S, reps=8):
    c, _ = fresh(); x = mx.array(toks[:S])[None]; ts = []
    for _ in range(reps + 2):
        t0 = time.perf_counter(); lg, h = fwd(x, c); mx.eval(lg); ts.append(time.perf_counter() - t0)
        for cc in c:   # keep the cache length constant: undo
            if hasattr(cc, "trim"): cc.trim(S)
            else: pass
    return statistics.median(ts[2:]) * 1e3
print("ms per forward (KV trimmed, GDN state drifts: cost only):", {S: round(timeit(S), 1) for S in (1, 2, 4, 6, 8)})
