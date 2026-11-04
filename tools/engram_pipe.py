"""Engram without a host sync in the DS4.1 decode step: exactness gate, command buffers per step, pipelined wall time A/B (one process, interleaved).

usage: PYTHONPATH=$PWD:$HOME/kyojin-mlx/vlm_pkg ../../.venv/bin/python tools/engram_pipe.py ../../models/ds2e [--steps 24] [--reps 15] [--variants par,par_ahead,staged,dev]
       (clang++ -O2 -std=c++17 -fobjc-arc -dynamiclib -framework Metal -framework Foundation tools/cbcount/cbcount.mm -o tools/cbcount/libcbcount.dylib first)
       MLX_MAX_OPS_PER_BUFFER=100000 MLX_MAX_MB_PER_BUFFER=1000000 as in macair17's rp_chain.py puts the whole step in one command buffer (the counter's cleanest reading)

variants (the embed of every Engram layer; the loop is the one a server would run):
  par        ParallelRowEmbedding, mlx-lm style loop (step i + 1 built and submitted before token i is read)   = today
  par_ahead  ParallelRowEmbedding, pipelined_greedy without a stage (hash ids one step ahead)                  = control: ids alone do not help
  staged     StagedRowEmbedding + EngramStage, pipelined_greedy (disk table, rows read after the build)
  staged_early  same, and the layers before the first Engram layer are submitted to the GPU while the rows are read
  gate       same stage, commit gate: the step is encoded and submitted first, its command buffers are held until the rows are read (one buffer per step)
  dev        DeviceRowEmbedding (table resident), pipelined_greedy                                              = no host read at all
Gates: prefill (one shot and chunked), teacher-forced decode and pipelined decode logits bit for bit equal to `par`; Engram hash ids equal to the host RowMap;
poisoned slots show NaN when read before the fill; commit counts per phase."""
import argparse, ctypes, json, os, statistics, sys, time
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import mlx.core as mx
if os.environ.get("M10_LIGHT"):                                                         # GPU-light proxy (outputs are garbage: use --no-exact)
    sys.path.insert(0, here); import light  # noqa: F401
from kyojin_mlx import patch as P, rows as R, engram_mlx as EM, engram_rows as ER
import kyojin_mlx

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--variants", default="par,par_ahead,staged,dev"); ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--reps", type=int, default=15); ap.add_argument("--len", type=int, default=135); ap.add_argument("--forced", type=int, default=10)
ap.add_argument("--chunk-rows", type=int, default=0, help="DeviceRowEmbedding chunk rows (0 = device max buffer / 2; a small value exercises the multi-chunk path)")
ap.add_argument("--no-timing", action="store_true"); ap.add_argument("--no-exact", action="store_true", help="skip the exactness gates (M10_LIGHT runs give garbage outputs)"); ap.add_argument("--out", default="")
a = ap.parse_args()
print("kyojin_mlx from", kyojin_mlx.__file__, flush=True)
lib = ctypes.CDLL(os.path.join(here, "cbcount", "libcbcount.dylib"))
lib.cb_install(); lib.cb_times.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]; lib.cb_commits.restype = ctypes.c_long; lib.cb_gpu_us.restype = ctypes.c_double; lib.cb_completed.restype = ctypes.c_long
print("env MLX_MAX_OPS_PER_BUFFER =", os.environ.get("MLX_MAX_OPS_PER_BUFFER"), " MLX_MAX_MB_PER_BUFFER =", os.environ.get("MLX_MAX_MB_PER_BUFFER"), flush=True)

model, spec, cfg = P.build_model(a.pack); P.patch_model(model, a.pack, spec, cfg); R.enable(model, spec)
lm = model.language_model; V = cfg["text_config"]["vocab_size"]
es = ER.EngramSpec.from_config(cfg); assert es is not None, "pack has no Engram layer"
tables = ER.EngramTables(a.pack, es); assert tables.validate() == []
names = a.variants.split(",")
assert names[0] == "par", "par is the reference and must come first"
mods = {"par": [EM.ParallelRowEmbedding(tables, p) for p in range(len(es.layer_ids))]}
stage = EM.install_staged_gather(model, tables); mods["staged"] = stage.embeds
if "dev" in names:
    mods["dev"] = EM.install_device_gather(model, tables, a.chunk_rows or None)
    print(f"dev: table resident, {len(mods['dev'][0]._w)} chunk(s) of {mods['dev'][0]._cr} rows per layer", flush=True)
mods["par_ahead"] = mods["par"]; mods["staged_early"] = stage.embeds; mods["gate"] = stage.embeds
def use(v):
    for pos, lid in enumerate(es.layer_ids):
        lm.layers[lid].engram.embed = mods[v][pos]
loop_of = {"par": "mlxlm", "par_ahead": "ahead", "staged": "ahead", "staged_early": "ahead", "gate": "ahead", "dev": "ahead"}

rng = np.random.default_rng(0)
ids = rng.integers(16, V - 16, a.len).tolist()
use("par")
cache0, lg0 = R.prefill(model, ids)
tok0 = mx.argmax(lg0, axis=-1); mx.eval(tok0)
bad = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + info if info else ""), flush=True)
    if not ok: bad.append(name)
f32 = lambda x: np.array(x.astype(mx.float32))

# ------------------------------------------------------------------------------------------------------------------------------- loops
# st rows: (build s, wait + fill s, submit s, commits in build, commits in wait + fill, commits in submit, commit counter at the step's start). serial=True drains the stream before each phase
# reading (MLX encodes an async_eval on its stream thread, so a count taken right after the call would miss commits still to come).
def loop_mlxlm(cache, tok, n, st=None, serial=False):
    """mlx-lm generate_step shape: step i + 1 is built and submitted, then token i is read."""
    out = []
    lg = R.step(model, cache, tok); y = mx.argmax(lg, axis=-1); mx.async_eval(y)
    for i in range(n - 1):
        if serial: mx.synchronize()
        t1 = time.perf_counter(); b0 = lib.cb_commits(); cs = b0
        lg = R.step(model, cache, y); ny = mx.argmax(lg, axis=-1)
        t2 = time.perf_counter(); b1 = lib.cb_commits()
        mx.async_eval(ny)
        t3 = time.perf_counter()
        if serial: mx.synchronize()
        b2 = lib.cb_commits()
        if st is not None: st.append((t2 - t1, 0.0, t3 - t2, b1 - b0, 0, b2 - b1, cs))
        out.append(int(y.item())); y = ny
    out.append(int(y.item()))
    return out

def loop_ahead(cache, tok, n, st=None, trace=None, serial=False):
    stg = stage if mods_current.startswith("staged") or mods_current == "gate" else None
    if stg is not None: stg.early = mods_current == "staged_early"; stg.gate = mods_current == "gate"
    gate = stg is not None and stg.gate
    marks = {}
    def probe(tag):
        if tag == "start" and serial:
            mx.synchronize()
            if gate: stg.resync()
        if tag == ("released" if gate else "submitted") and serial:                          # a gated step cannot drain before its commits are released
            mx.synchronize()
            if gate: stg.resync()
        marks[tag] = (time.perf_counter(), lib.cb_commits())
        if tag == "submitted" and st is not None and not gate:
            s, b, f, d = marks["start"], marks["built"], marks["filled"], marks["submitted"]
            st.append((b[0] - s[0], f[0] - b[0], d[0] - f[0], b[1] - s[1], f[1] - b[1], d[1] - f[1], s[1]))
        if tag == "released" and st is not None:                                              # gated: build | submit | wait + fill + release
            s, b, f, d, r = marks["start"], marks["built"], marks["filled"], marks["submitted"], marks["released"]
            st.append((b[0] - s[0], r[0] - d[0], d[0] - f[0], b[1] - s[1], r[1] - d[1], d[1] - f[1], s[1]))
    try:
        return list(EM.pipelined_greedy(model, cache, tok, n, stage=stg, probe=probe, trace=trace))
    finally:
        if stg is not None: stg.gate = False

mods_current = "par"
def run(v, cache, tok, n, st=None, trace=None, serial=False):
    global mods_current
    use(v); mods_current = v
    return loop_mlxlm(cache, tok, n, st, serial) if loop_of[v] == "mlxlm" else loop_ahead(cache, tok, n, st, trace, serial)

# ------------------------------------------------------------------------------------------------------------------------------- 1. exactness
if not a.no_exact:
    print("\n== exactness (reference: par) ==")
    use("par")
    cpar, lpar = R.prefill(model, ids)
    ref_pre = f32(lpar)
    chunked = lambda: [ids[i:i + 40] for i in range(0, len(ids), 40)]
    def prefill_chunked(v):
        use(v); c = lm.make_cache(); lg = None
        for ch in chunked():
            lg = lm(mx.array(ch)[None], cache=c).logits[:, -1].astype(mx.float32); mx.eval(lg)
        return c, lg
    cc, lc = prefill_chunked("par"); ref_chunk = f32(lc)
    forced = rng.integers(16, V - 16, a.forced).tolist()
    def forced_logits(v, cache):
        use(v); out = []
        for t in forced:
            lg = R.step(model, cache, mx.array([t])); mx.eval(lg); out.append(f32(lg))
        return out
    ref_forced = forced_logits("par", R.fork(cache0))
    tokens_ref = None
    for v in names:
        use(v)
        c1, l1 = R.prefill(model, ids); check(f"{v}: prefill one shot (len {len(ids)}) logits == par", np.array_equal(f32(l1), ref_pre))
        c2, l2 = prefill_chunked(v); check(f"{v}: prefill in chunks of 40 logits == par chunked", np.array_equal(f32(l2), ref_chunk))
        fl = forced_logits(v, R.fork(cache0)); check(f"{v}: {a.forced} teacher-forced decode steps logits == par", all(np.array_equal(x, y) for x, y in zip(fl, ref_forced)))
        trace = []
        toks = run(v, R.fork(cache0), tok0, a.steps, trace=trace)
        if tokens_ref is None:
            tokens_ref = toks
        check(f"{v}: pipelined decode tokens ({a.steps}) == par", toks == tokens_ref)
        if loop_of[v] == "ahead":
            use("par"); c = R.fork(cache0); bad_l = 0; seq = [int(tok0.item())] + toks[:-1]
            for t, lt in zip(seq, trace):
                lg = R.step(model, c, mx.array([t])); mx.eval(lg); bad_l += not np.array_equal(f32(lg), f32(lt))
            check(f"{v}: pipelined decode logits == par stock steps on the same tokens", bad_l == 0, f"({bad_l} of {len(trace)} differ)")

    # hash ids against the host RowMap (numpy): the prompt (the model's own hash) and every position the pipelined loop hashes one step ahead
    tmap = json.load(open(os.path.join(a.pack, "engram_token_map.json")))
    if isinstance(tmap, dict): tmap = tmap.get("token_map", tmap) if "token_map" in tmap else [tmap[str(i)] for i in range(len(tmap))]
    rm = ER.RowMap(es, tmap)
    lm._ensure_engram_hash()
    NH = 8
    got = []
    orig_ahead = EM.hashes_ahead
    def rec_ahead(m, e, y):
        r = orig_ahead(m, e, y)
        if r is not None: got.append(np.asarray(r)[0, 0])                                   # [n_layers, n_cols] of position L + k
        return r
    EM.hashes_ahead = rec_ahead
    try:
        toks_h = run("staged", R.fork(cache0), tok0, NH)
    finally:
        EM.hashes_ahead = orig_ahead
    host = rm(np.array(ids + [int(tok0.item())] + toks_h))                                 # [L + 1 + NH, n_layers, n_cols]
    pre_ids = np.asarray(lm.engram_hash(mx.array(ids)[None], 0, lm.make_cache()[0], token_mask=None))[0]
    check("hash ids of the prompt (model's GPU hash) == host RowMap", np.array_equal(pre_ids, host[:len(ids)]))
    check(f"hash ids of {len(got)} pipelined positions (one step ahead on the GPU) == host RowMap", np.array_equal(np.stack(got), host[len(ids):len(ids) + len(got)]))

    # poisoned slots: an eval before the fill must show NaN (the slots really feed the kernel); after the fill the step is exact
    stage.poison = True
    use("staged"); c = R.fork(cache0)
    with stage.deferred():
        lg = R.step(model, c, tok0)
    early = f32(lg)
    stage.pending = []; stage.poison = False
    use("staged"); c = R.fork(cache0)
    with stage.deferred():
        lg = R.step(model, c, tok0)
    stage.fill(None)
    use("par"); c = R.fork(cache0); ref1 = f32(R.step(model, c, tok0))
    check("staged: logits after the fill == par", np.array_equal(f32(lg), ref1))
    check("staged: logits evaluated BEFORE the fill differ from par (poisoned slots feed the kernel, negative control)", not np.array_equal(early, ref1), f"({100 * float(np.isnan(early).mean()):.0f} % of the logits are NaN)")

# ------------------------------------------------------------------------------------------------------------------------------- 2. commit counts
print("\n== command buffers per step (MLX_MAX_OPS_PER_BUFFER=%s) ==" % os.environ.get("MLX_MAX_OPS_PER_BUFFER"))
summ = {}
for v in names:
    st = []
    run(v, R.fork(cache0), tok0, 8, st, serial=True)
    steady = st[2:] if len(st) > 3 else st
    med = lambda k: statistics.median(x[k] for x in steady)
    summ[v] = dict(build_commits=med(3), wait_fill_commits=med(4), submit_commits=med(5), total=med(3) + med(4) + med(5))
    print(f"{v:10s} commits per step: during graph build {med(3):.0f} | wait + fill {med(4):.0f} | async_eval {med(5):.0f} | total {med(3) + med(4) + med(5):.0f}")
if "staged" in summ: check("staged: 0 command buffers committed during the graph build", summ["staged"]["build_commits"] == 0 and summ["staged"]["wait_fill_commits"] == 0)
if "dev" in summ: check("dev: 0 command buffers committed during the graph build", summ["dev"]["build_commits"] == 0)
if "par" in summ: print("   (par: the baseline commits %d inside the build)" % summ["par"]["build_commits"])

def gaps_ms(st):
    """GPU idle time per step, ms, from the GPU timestamps of every command buffer (one queue, executed in commit order): sum over consecutive
    buffers of max(0, GPU start of j - GPU end of j - 1), from the first buffer of step 2 to the last one, divided by the steps in that range."""
    t0, t1 = ctypes.c_double(), ctypes.c_double()
    n = lib.cb_commits(); k0 = st[2][6]; tt = []
    for j in range(k0 - 1, n):
        lib.cb_times(j, ctypes.byref(t0), ctypes.byref(t1)); tt.append((t0.value, t1.value))
    if any(b == 0.0 for _, b in tt): return []                                           # a completion handler had not run yet: drop this run
    idle = sum(max(0.0, tt[j][0] - tt[j - 1][1]) for j in range(1, len(tt)))
    return [idle * 1e3 / (len(st) - 2)]

# ------------------------------------------------------------------------------------------------------------------------------- 3. pipelined wall A/B
res = {}
if not a.no_timing:
    print(f"\n== pipelined decode wall, {a.steps} steps per run, {a.reps} interleaved runs (first 2 dropped) ==")
    W = {v: [] for v in names}; G = {v: [] for v in names}; PH = {v: [] for v in names}; GP = {v: [] for v in names}; RD = {v: [] for v in names}
    for rep in range(a.reps + 2):
        for v in names:
            c = R.fork(cache0); mx.synchronize(); lib.cb_reset(); st = []; stage.read_log.clear()
            c_before = lib.cb_commits(); t0 = time.perf_counter()
            run(v, c, tok0, a.steps, st)
            dt = time.perf_counter() - t0
            mx.synchronize(); time.sleep(0.02)
            if rep >= 2:
                GP[v] += gaps_ms(st)
                RD[v] += [x * 1e3 for x in stage.read_log[2:]]
                W[v].append(dt / a.steps * 1e3); G[v].append(lib.cb_gpu_us() / a.steps / 1e3)
                PH[v].append((statistics.median(x[0] for x in st[2:]) * 1e3, statistics.median(x[1] for x in st[2:]) * 1e3, statistics.median(x[2] for x in st[2:]) * 1e3))
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(len(xs) * f))]
    for v in names:
        w, g = W[v], G[v]
        res[v] = dict(wall_min=min(w), wall_p25=q(w, .25), wall_med=q(w, .5), gpu_min=min(g), gpu_med=q(g, .5),
                      read_med=(statistics.median(RD[v]) if RD[v] else 0.0), read_max=(max(RD[v]) if RD[v] else 0.0), gap_min=min(GP[v] or [-1]), gap_p25=q(GP[v] or [-1], .25), gap_med=q(GP[v] or [-1], .5), build_ms=statistics.median(p[0] for p in PH[v]), wait_fill_ms=statistics.median(p[1] for p in PH[v]), encode_ms=statistics.median(p[2] for p in PH[v]))
        r = res[v]
        print(f"{v:10s} wall ms/step min {r['wall_min']:6.2f} p25 {r['wall_p25']:6.2f} med {r['wall_med']:6.2f} | GPU busy ms/step min {r['gpu_min']:5.2f} med {r['gpu_med']:5.2f} | "
              f"CPU phases (med): build {r['build_ms']:.2f} + wait/fill {r['wait_fill_ms']:.2f} + submit {r['encode_ms']:.2f} ms | GPU idle per step min {r['gap_min']:.2f} p25 {r['gap_p25']:.2f} med {r['gap_med']:.2f} ms"
              + (f" | row read per fill med {r['read_med']:.3f} max {r['read_max']:.3f} ms" if r['read_med'] else ""))
    if "par" in res:
        for v in names:
            if v != "par": print(f"   {v} vs par: min {100 * (1 - res[v]['wall_min'] / res['par']['wall_min']):+.1f} %   p25 {100 * (1 - res[v]['wall_p25'] / res['par']['wall_p25']):+.1f} %   med {100 * (1 - res[v]['wall_med'] / res['par']['wall_med']):+.1f} %")
    for v in names:
        r = res[v]; cpu = r['build_ms'] + r['wait_fill_ms'] + r['encode_ms']
        print(f"   {v:10s} model check: max(CPU, GPU) = {max(cpu, r['gpu_med']):.2f}  CPU + GPU = {cpu + r['gpu_med']:.2f}  measured med {r['wall_med']:.2f} ms/step")
print("peak Metal memory %.0f MB" % (mx.get_peak_memory() / 1e6))
if a.out:
    json.dump(dict(env=dict(ops=os.environ.get("MLX_MAX_OPS_PER_BUFFER"), mb=os.environ.get("MLX_MAX_MB_PER_BUFFER")), commits=summ, timing=res, failed=bad), open(a.out, "w"), indent=1)
tables.close()
print("\nFAILED: " + ", ".join(bad) if bad else "\nALL GATES PASS")
sys.exit(1 if bad else 0)
