"""GPU time of the chained verify of n rows (chain), of the batched step of N users (users) and of the solo step: one command buffer per step, so the wait
after `async_eval` is the GPU time (plus the commit latency, the same for every mode). Modes are run interleaved round by round (the Air is shared).
usage: verify_gpu.py PACK [--len 135] [--reps 25] [--modes solo,c1,c2,c3,c4,c5,c8,u8,u16] [--json F]
  cN = chained verify of N rows of one sequence (spec.verify), uN = one batched decode step of N users (rows.step), advancing positions are not used (fixed position,
  every rep starts from the same cache).  Columns: build / encode CPU, GPU wait (min, p25, median), launches."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
import mlx.core as mx
import numpy as np
from kyojin_mlx import patch as P, rows as R, spec as S
from count_launches import graph_census

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=25)
ap.add_argument("--modes", default="solo,c1,c2,c3,c4,c5,c8,u8,u16"); ap.add_argument("--json")
ap.add_argument("--set", nargs="*", default=[], help="text_config overrides key=json")
ap.add_argument("--first", type=int, default=0)

a = ap.parse_args()
model, spec, cfg = P.build_model(a.pack, {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.set)})
P.patch_model(model, a.pack, spec, cfg)
R.enable(model, spec)
S.install(model, spec)
if a.first:
    model.language_model.layers = model.language_model.layers[:a.first]
tc = cfg["text_config"]
rng = np.random.default_rng(1)
V = tc["vocab_size"]
prompt = rng.integers(16, V - 16, a.len).tolist()
cache, lg, _, _ = S.prefill(model, prompt)
toks = rng.integers(16, V - 16, 32).tolist()
modes = a.modes.split(",")


ABL = set()
_moe_call = __import__("kyojin_mlx.moe", fromlist=["MoeGather"]).MoeGather.__call__
_core = __import__("kyojin_mlx.fuse_attn", fromlist=["core"]).core


def _moe(self, x, ids, wts, fq8=False):
    if "moe" in ABL:
        return mx.zeros((x.shape[0], self.d.nout), mx.float32)
    return _moe_call(self, x, ids, wts, fq8)


def _core_w(self, L, cache, x, qr, q, kv):
    if "core" in ABL:                                                        # no compressor / indexer / ring / attention kernel: zeros of the output shape
        return mx.zeros((1, self.o_groups, 1, (self.n_heads // self.o_groups) * self.head_dim), q.dtype), None
    if "glue" in ABL:                                                        # only the compressor + indexer (the attention kernels run on a fixed pool / idx)
        return _core(self, L, cache, x, qr, q, kv)
    return _core(self, L, cache, x, qr, q, kv)


_Lm = __import__("mlx_vlm.models.deepseek_v41.language", fromlist=["Indexer"])
_idx_call, _idx_last = _Lm.Indexer.__call__, {}


def _idx_w(self, *a, **k):
    if "idx" in ABL and self.layer_idx in _idx_last:                         # indexer ablated: the top-k of the previous call (same shape) stands in, no indexer launch
        return _idx_last[self.layer_idx]
    r = _idx_call(self, *a, **k)
    if "idx" not in ABL:
        _idx_last[self.layer_idx] = r
    return r


_Lm.Indexer.__call__ = _idx_w
import kyojin_mlx.moe as _M, kyojin_mlx.fuse_attn as _FA, kyojin_mlx.fuse_layer as _FL
_M.MoeGather.__call__ = _moe
_FA.core = _core_w


_kfl: dict = {}


def run(m):
    """mode[:ablation]: the ablation (moe, core, moe+core) is part of the mode, so the variants interleave rep by rep; each ablation keeps its own compiled regions."""
    m, _, ab = m.partition(":")
    ABL.clear(); ABL.update(ab.split("+") if ab else [])
    layers = model.language_model.layers
    for blk in layers:
        st = _kfl.get((id(blk), ab))
        if st is None:
            blk.__dict__.pop("_kfl", None)
        else:
            blk.__dict__["_kfl"] = st
    try:
        return _run(m)
    finally:
        for blk in layers:
            if blk.__dict__.get("_kfl") is not None:
                _kfl[(id(blk), ab)] = blk.__dict__["_kfl"]


def _run(m):
    if m == "solo":
        return R.step(model, R.fork(cache), mx.array([toks[0]]))
    n = int(m[1:])
    if m[0] == "c":
        lg, _, chain = S.verify(model, cache, toks[:n])
        return [lg] + [w for w in chain.rows[-1].window if w is not None]           # the logits and the ring state the next step starts from (a deferred ring write is paid here)
    bc = R.batch([R.fork(cache) for _ in range(n)])
    return R.step(model, bc, mx.array(toks[:n]))


res = {m: dict(build=[], enc=[], gpu=[]) for m in modes}
for m in modes:
    for _ in range(3):
        mx.eval(run(m))
for rep in range(a.reps):
    for m in modes:
        t0 = time.perf_counter(); out = run(m); t1 = time.perf_counter(); mx.async_eval(out); t2 = time.perf_counter(); mx.eval(out); t3 = time.perf_counter()
        r = res[m]; r["build"].append(t1 - t0); r["enc"].append(t2 - t1); r["gpu"].append(t3 - t2)
out = {}
print(f"{'mode':6s} {'launch':>7s} {'cpu min':>8s} | {'gpu min':>8s} {'p25':>8s} {'med':>8s}  (ms)")
for m in modes:
    r = res[m]
    cen = graph_census(run(m))
    if os.environ.get("CEN"): print(m, dict(list(cen["by_type"].items())[:30]))
    cpu = [b + e for b, e in zip(r["build"], r["enc"])]
    o = out[m] = dict(launches=cen["dispatches"], cpu_min=1e3 * min(cpu), gpu_min=1e3 * min(r["gpu"]), gpu_p25=1e3 * float(np.percentile(r["gpu"], 25)), gpu_med=1e3 * float(np.median(r["gpu"])))
    print(f"{m:6s} {o['launches']:7d} {o['cpu_min']:8.2f} | {o['gpu_min']:8.2f} {o['gpu_p25']:8.2f} {o['gpu_med']:8.2f}")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
