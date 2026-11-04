"""Speculative decode inside the batched server and the KV memory budget, on a pack with a draft head (the DS4.1 slice with real mtp weights).
KYOJIN_ROWS_NAX=0: every comparison is bitwise (token streams == PLAIN solo). KYOJIN_ROWS_NAX=1 (default; verify rows above 4 run the NAX kernel, fp16
rounding class, see tolgate.py): a stream is compared to PLAIN solo up to its first divergence; the divergence must be a near tie of the PLAIN logits (the
logit of the PLAIN token minus the logit of the token taken, at that position, within TIE_REL x max|logit|; recomputed by decode steps), the tokens after it are
not comparable (a different context) and are excused; the greedy agreement of all the compared tokens must be >= 99.5 %. A session that left its reference
gets the real draft head's tokens (the oracle cannot follow it). The negative control must still differ.
usage: serve_spec_check.py PACK [--users 4] [--new 24] [--lens 41,67,90,120,58,75,101,83] [--json OUT] [--only exact,control,churn,bench]

One solo reference: PLAIN (mtp depth 0, one request at a time, the batched DS4.1 core, one row). The speculative server runs the same core
(rows = pending token + drafts of every session in one pass), so its tokens must equal the PLAIN solo whatever the depth, the sessions beside it
and the acceptance: no second reference, no near-tie exception.
exact    greedy tokens of every session == PLAIN solo, for N = 1, 2, 4 sessions, fixed depths 1..3 and
         the adaptive depth, joins staggered, two prompt groups. The real draft head of a 2-layer slice is
         noise (it accepts nothing), so a per-session ORACLE replaces its tokens: the true continuation with wrong tokens injected on a
         schedule (err = 0: always right, 3: one in three wrong, 1: all wrong). The real head still runs (ring, cost); only its tokens change.
control  draft-off controls (plain batched == PLAIN solo, spec server at D = 0 == PLAIN solo) and a negative control (the verify keeps every
         drafted row): the latter must differ. With err = 0 every drafted token must be accepted (the oracle really reaches the verify).
yield    8 requests through 2 slots, replies yield their slot every 6 tokens (no memory limit): outputs == PLAIN solo.
churn    8 sessions through 2 slots with a forced small --kv-budget: outputs == own-mode solo, tracked KV <= budget, drops and rebuilds happened
         (plain decode and speculative decode).
bench    step cost vs rows: N sessions x (1 + D) rows in one verify + commit (batched core, and the older chained verify, interleaved),
         against the batched plain step."""
import argparse, json, statistics, threading, time

import mlx.core as mx
import numpy as np

import tolgate as T

from kyojin_mlx.serve import backends as B
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling

ap = argparse.ArgumentParser()
ap.add_argument("pack"); ap.add_argument("--users", type=int, default=4); ap.add_argument("--new", type=int, default=24)
ap.add_argument("--lens", default="41,67,90,120,58,75,101,83"); ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--block", type=int, default=64)
ap.add_argument("--only", default="control,exact,yield,churn,churnbig,bench"); ap.add_argument("--big-len", type=int, default=1200); ap.add_argument("--big-mtps", default="0,2"); ap.add_argument("--quick", action="store_true", help="a short version of control + exact (about 30 s on the slice)"); ap.add_argument("--json"); ap.add_argument("--reps", type=int, default=8)
a = ap.parse_args()
mx.set_cache_limit(256 * 2 ** 20)
be, cfg = load_backend(a.pack, mtp=True)
V = cfg["text_config"]["vocab_size"]
rng = np.random.default_rng(5)
LENS = [int(v) for v in a.lens.split(",")]
PROMPTS = {f"u{i}": rng.integers(16, V - 16, n).tolist() for i, n in enumerate(LENS)}
PROMPTS["p1"] = PROMPTS["u3"][:100] + rng.integers(16, V - 16, 12).tolist()           # shares the first 64-token block with u3 (prefix hit)
REF: dict = {}                                                         # PLAIN solo, by prompt
REFC = REF                                                             # the true continuation the oracle drafts from (same dict)
TOL = T.tolerant()
ORACLE = {"err": 0, "on": False, "strict": not TOL, "calls": 0}
TSTAT = dict(tokens=0, flips=0, tie_flips=0, diverged=0, cases=0, unique=set(), unique_tokens=set())
MARGIN: dict = {}                                                      # (prompt key, index) -> (PLAIN logits row) cache for the near-tie check
real_many = B.VlmBackend.draft_many


def oracle_many(self, jobs):
    """The drafter API of the engine: the real head runs for every job (ring state, cost); only its tokens are replaced by the true
    continuation with wrong tokens injected on a schedule."""
    toks = real_many(self, jobs)
    if not ORACLE["on"]:
        return toks
    ORACLE["calls"] += 1
    out = []
    for (d, pending, p0, depth, history), t in zip(jobs, toks):
        ids, o = history
        ref = REFC[tuple(ids)]
        if not ORACLE["strict"] and not o[-1] == pending == ref[len(o) - 1]:
            out.append(t)                                              # negative control: the session already left its reference
            continue
        assert o[-1] == pending == ref[len(o) - 1], "session out of step with its reference"
        res = ref[len(o):len(o) + depth]
        e = ORACLE["err"]
        out.append([(x + 1) % V if e and (len(o) + i) % e == 0 else x for i, x in enumerate(res)])
    return out


B.VlmBackend.draft_many = oracle_many
real_lazy = B.VlmBackend.draft_lazy


def oracle_lazy(self, jobs):
    """The lazy drafter API with the oracle's tokens: they reach the verify as arrays (confidence 1 on every token the oracle has, 0 past the end of
    its reference), so the lazy verify at the planned depth and the confidence cut of the accept loop are exercised with accepted drafts."""
    if not ORACLE["on"]:
        return real_lazy(self, jobs)
    if not B.LAZY_DRAFT:
        return None
    got = oracle_many(self, jobs)
    live = [i for i, j in enumerate(jobs) if j[3] > 0]
    if not live:
        return None
    L = self.drafter.block
    toks = mx.array([got[i] + [0] * (L - len(got[i])) for i in live], dtype=mx.int32)
    conf = mx.array([[1.0] * len(got[i]) + [0.0] * (L - len(got[i])) for i in live])
    return B.LazyDrafts(self, jobs, live, toks, conf)


B.VlmBackend.draft_lazy = oracle_lazy


def serve(names, *, sessions, mtp=0, budget=None, err=0, floor=None, ctx=4096, stagger=0.0, override=None, prefix=0, oracle=True, n_new=None):
    """Run the requests `names` through one engine on this thread (the one that loaded the model). Returns (tokens by name, status, engine)."""
    ORACLE["on"], ORACLE["err"] = oracle and mtp > 0, err
    eng = Engine(be, sessions=sessions, ctx=ctx, chunk=a.chunk, block=a.block, prefix_bytes=prefix, mtp_depth=mtp, kv_budget=budget, rq_floor=floor,
                 depth_override=override, kv_refresh=1, replay_chunk=8, log=lambda *x: print("   [engine]", *x, flush=True))
    out, res = {}, {}

    def worker(i, name):
        time.sleep(stagger * i)
        h = eng.submit(PROMPTS[name], max_new=n_new or a.new)
        toks = []
        while True:
            ev = h.events.get()
            if ev[0] == "tok":
                toks.append(ev[1])
            else:
                out[name] = toks
                return

    def driver():
        ths = [threading.Thread(target=worker, args=(i, n)) for i, n in enumerate(names)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        res["st"] = eng.status()
        eng.stop()
    th = threading.Thread(target=driver)
    th.start()
    eng.run()
    th.join()
    return out, res["st"], eng


def solo_ref():
    for n, ids in PROMPTS.items():
        toks, _, _ = serve([n], sessions=1, mtp=0)
        REF[tuple(ids)] = toks[n]


def first_diff(got, names, ref=None):
    """{name: index of the first token that differs from solo} (empty when all equal)."""
    out = {}
    ref = REFC if ref is None else ref
    for n in names:
        r = ref[tuple(PROMPTS[n])]
        if got[n] != r:
            out[n] = next((i for i, (x, y) in enumerate(zip(got[n], r)) if x != y), min(len(got[n]), len(r)))
    return out


def plain_logits(name, i):
    """PLAIN decode logits that produced reference token i of prompt `name`: prefill, then decode steps through the reference tokens (cached)."""
    ids, r = PROMPTS[name], REF[tuple(PROMPTS[name])]
    if (name, i) not in MARGIN:
        c = be.make_cache()
        for j in range(0, len(ids), a.chunk):
            lg = be.prefill_chunk(c, ids[j:j + a.chunk])
        row = lg[0]
        if i > 0:
            bc = be.batch([c])
            for t in r[:i]:
                row = be.step(bc, mx.array([t], dtype=mx.int32))[0]
        MARGIN[(name, i)] = np.array(row.astype(mx.float32)).reshape(-1)
    return MARGIN[(name, i)]


def same(got, names, ref=None):
    """bitwise token equality; in tolerance mode: equal up to the first divergence, which must be a near tie of the PLAIN logits (see the header)"""
    ref = REFC if ref is None else ref
    if not TOL:
        return all(got[n] == ref[tuple(PROMPTS[n])] for n in names)
    good, TSTAT["last_diverged"] = True, False
    for n in names:
        r, g = ref[tuple(PROMPTS[n])], got[n]
        i = next((q for q, (x, y) in enumerate(zip(g, r)) if x != y), None)
        if i is None:
            good &= len(g) == len(r)
            TSTAT["tokens"] += min(len(g), len(r))
            TSTAT["unique_tokens"].update((n, q) for q in range(min(len(g), len(r))))
            continue
        TSTAT["tokens"] += i + 1
        TSTAT["unique_tokens"].update((n, q) for q in range(i + 1))
        TSTAT["flips"] += 1
        TSTAT["diverged"] += 1
        TSTAT["last_diverged"] = True
        TSTAT["unique"].add((n, i))
        row = plain_logits(n, i) if REF is ref else None
        if row is not None and int(np.argmax(row)) == r[i] and T.near_tie(row, g[i]):
            TSTAT["tie_flips"] += 1
        else:
            good = False
            print(f"   {n}: token {i} is {g[i]}, PLAIN {r[i]}: not a near tie of the PLAIN logits" + ("" if row is None else f" (gap {row.max() - row[g[i]]:.3g}, max|logit| {np.abs(row).max():.3g})"), flush=True)
    return good


report: dict = {"pack": a.pack, "lens": LENS, "new": a.new}
ok = True
only = set(a.only.split(","))
print("solo references (plain) ...", flush=True)
t0 = time.perf_counter()
solo_ref()
print(f"  {len(REF)} requests, {time.perf_counter() - t0:.1f} s; tokens of u0: {REF[tuple(PROMPTS['u0'])][:8]}", flush=True)
NAMES = [n for n in PROMPTS if n.startswith("u")]
GROUPS = [NAMES[:4], NAMES[4:8]][:1 if a.quick else 2]
QN, QD, QE = ((2,), (2, 3), (3, 1)) if a.quick else ((1, 2, 4), (1, 2, 3), (0, 3, 1))
mx.reset_peak_memory()


if "control" in only:
    rep = {}
    for gi, grp in enumerate(GROUPS):
        for n in QN:
            names = grp[:n]
            got, st, _ = serve(names, sessions=n, mtp=0)
            rep[f"plain_batched_g{gi}_N{n}"] = same(got, names, REF)
            got, st, _ = serve(names, sessions=n, mtp=2, override=0)
            rep[f"spec_server_D0_g{gi}_N{n}"] = same(got, names)
    print("  draft-off controls (plain batched == plain solo; spec server at D=0 == plain solo):", rep, flush=True)
    report["control"] = rep
    ok &= all(rep.values())

if "exact" in only:
    rep = {}
    for gi, grp in enumerate(GROUPS):
        for n in QN:
            names = grp[:n]
            for depth in QD:
                for err in QE:
                    got, st, _ = serve(names, sessions=n, mtp=depth, override=depth, err=err, stagger=0.0)
                    key = f"g{gi}_N{n}_D{depth}_err{err}"
                    rep[key] = dict(equal=same(got, names), rounds=st.get("spec_rounds"), rows=st.get("spec_rows"), drafted=st.get("spec_drafted"),
                                    accepted=st.get("spec_accepted"))
                    ok &= rep[key]["equal"]
                    if err == 0:                                       # the oracle's tokens reached the verify, and all were taken
                        rep[key]["oracle_all_accepted"] = bool(st.get("spec_drafted")) and (st.get("spec_accepted") == st.get("spec_drafted") or (TOL and TSTAT["last_diverged"]))   # NAX on: a session that flipped left its oracle
                        ok &= rep[key]["oracle_all_accepted"]
                    print(f"  {key}: equal {rep[key]['equal']}  rounds {rep[key]['rounds']} rows {rep[key]['rows']} accepted {rep[key]['accepted']}/{rep[key]['drafted']}", flush=True)
        for n in QN[-1:] if a.quick else (2, 4):                         # adaptive depth, staggered joins
            names = grp[:n]
            got, st, _ = serve(names, sessions=n, mtp=3, err=3, stagger=0.15)
            key = f"g{gi}_adaptive_N{n}_staggered"
            rep[key] = dict(equal=same(got, names), rounds=st.get("spec_rounds"), rows=st.get("spec_rows"), accepted=st.get("spec_accepted"), drafted=st.get("spec_drafted"))
            ok &= rep[key]["equal"]
            print(f"  {key}: equal {rep[key]['equal']}  rounds {rep[key]['rounds']} rows {rep[key]['rows']} accepted {rep[key]['accepted']}/{rep[key]['drafted']}", flush=True)
    for mtp in () if a.quick else (0, 2):                               # p1 joins after u3 filled the prefix cache: hit, same tokens
        got, st, _ = serve(["u3", "p1"], sessions=2, mtp=mtp, err=3, stagger=3.0, prefix=1 << 28)
        rep[f"prefix_mtp{mtp}"] = dict(equal=same(got, ["u3", "p1"], REF), hits=st["prefix_hits"])
        ok &= rep[f"prefix_mtp{mtp}"]["equal"] and st["prefix_hits"] >= 1
        print(f"  prefix_mtp{mtp}: {rep[f'prefix_mtp{mtp}']}", flush=True)
    report["exact"] = rep

if "control" in only:
    orig = B.BatchedPass.commit

    def keep_all(self, accepted, drafts=None, p0s=None):
        return orig(self, [c - 1 for c in self.ps.plan.counts], drafts, p0s)      # no rollback: the wrong drafts stay in the cache
    B.BatchedPass.commit, ORACLE["strict"] = keep_all, False
    try:
        got, st, _ = serve(NAMES[:2], sessions=2, mtp=3, override=3, err=3)
    finally:
        B.BatchedPass.commit, ORACLE["strict"] = orig, not TOL
    saved = {k: (set(v) if isinstance(v, set) else v) for k, v in TSTAT.items()}
    neg = not same(got, NAMES[:2])
    TSTAT.clear(); TSTAT.update(saved)                                 # the control is not part of the agreement figures
    report["control"]["negative_control_differs"] = neg
    ok &= neg
    print(f"  negative control (no rollback) differs from solo: {neg}", flush=True)

if "num" in only:
    def top(lg):
        v = np.array(lg.astype(mx.float32)).reshape(-1)
        o = np.argsort(-v)
        return int(o[0]), float(v[o[0]] - v[o[1]])
    f32 = lambda x: np.array(x.astype(mx.float32))
    for n in ("u4", "u0", "u5", "u6"):
        ids, ref = PROMPTS[n], REFC[tuple(PROMPTS[n])]
        c = be.make_cache()
        for i in range(0, len(ids), a.chunk):
            lg = be.prefill_chunk(c, ids[i:i + a.chunk])
        t0 = int(mx.argmax(lg[0]).item())
        bc = be.batch([be.fork(c)])
        lp = be.step(bc, mx.array([t0]))
        p1 = be.spec_verify([(be.fork(c), [t0])]).logits
        p3 = be.spec_verify([(be.fork(c), [t0, ref[1], ref[2]])]).logits
        mx.eval(lp, p1, p3)
        print(f"  {n}: plain top {top(lp)} | chain1 top {top(p1[0])} maxdiff vs plain {np.abs(f32(lp[0]) - f32(p1[0])).max():.4f} | "
              f"chain3 row0 maxdiff vs chain1 {np.abs(f32(p3[0]) - f32(p1[0])).max():.4f}", flush=True)

if "inflight" in only:
    # ticks of the budget churn (long prompts, speculative) whose end-of-tick live memory is high: in-flight GPU work (freed by synchronize), not garbage
    import gc, collections, time
    for i in range(8):
        PROMPTS[f"b{i}"] = rng.integers(16, V - 16, a.big_len + 37 * i).tolist()
    bn = [f"b{i}" for i in range(8)]
    _, _, e1 = serve([bn[-1]], sessions=1, mtp=0, ctx=2048, budget=1 << 40, n_new=16)
    free, _, _ = serve(bn, sessions=2, mtp=2, ctx=2048, floor=6, budget=1 << 40, n_new=16, stagger=0.02, oracle=False)
    for n in bn:
        REFC[tuple(PROMPTS[n])] = free[n]
    hits, _note = [], Engine._note_kv
    mx.clear_cache(); gc.collect(); base = mx.get_active_memory() / 2 ** 20

    def note(self):
        _note(self)
        live = mx.get_active_memory() / 2 ** 20 - base
        if live > 150 and len(hits) < 4:
            mx.synchronize()
            l1 = mx.get_active_memory() / 2 ** 20 - base
            time.sleep(0.5)
            l2 = mx.get_active_memory() / 2 ** 20 - base
            gc.collect()
            l3 = mx.get_active_memory() / 2 ** 20 - base
            hits.append((len(hits), round(live, 1), "after synchronize", round(l1, 1), "after 0.5 s", round(l2, 1), "after gc", round(l3, 1), "resume" if any(x.resume for x in self.prefilling) else "fresh"))
    Engine._note_kv = note
    got, st, eng = serve(bn, sessions=2, mtp=2, err=3, ctx=2048, floor=6, budget=int(e1.kv_peak * 2.3), n_new=16, stagger=0.02)
    Engine._note_kv = _note
    print(f"  baseline {base:.1f} MB; kv peak {st['kv_peak_bytes'] / 2 ** 20:.1f} MB; drops {st['kv_drops']} rebuilds {st['rebuilds']}; ticks over 150 MB at the end: "
          f"{len(hits) or 'none in this run (timing dependent)'}", flush=True)
    for h in hits:
        print("   ", h, flush=True)

if "yield" in only:
    rep = {}
    for mtp, ov in ((0, None), (2, 2), (3, None)):
        ref = REF
        got, st, _ = serve(NAMES[:8], sessions=2, ctx=256, mtp=mtp, override=ov, err=3, floor=6, stagger=0.02)
        key = f"yield8_mtp{mtp}{'' if ov else 'a'}"
        rep[key] = dict(equal=same(got, NAMES[:8], ref), diff=first_diff(got, NAMES[:8], ref), yields=st.get("yields", 0))
        ok &= rep[key]["equal"] and rep[key]["yields"] > 0
        print(f"  {key}: {rep[key]}", flush=True)
    report["yield"] = rep

if "churn" in only:
    rep = {}
    ids8 = NAMES[:8]
    longest = max(ids8, key=lambda n: len(PROMPTS[n]))                    # one session of the longest request, measured: the unit of the budget
    _, _, e1 = serve([longest], sessions=1, mtp=0, ctx=256, budget=1 << 33)
    unit = e1.kv_peak
    print(f"  cache bytes per token ~{e1.bpt:.0f}; longest session {unit / 2 ** 20:.2f} MB", flush=True)
    for mtp in (0, 2):
        ref = REF
        budget = int(unit * 2.3)                                          # two active sessions and a little room: parked sessions get dropped
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        got, st, eng = serve(ids8, sessions=2, mtp=mtp, err=3, ctx=256, floor=6, budget=budget, stagger=0.02)
        peak_delta = mx.get_peak_memory() - base
        key = f"churn8_mtp{mtp}"
        rep[key] = dict(equal=same(got, ids8, ref), diff=first_diff(got, ids8, ref), budget_mb=budget / 2 ** 20, kv_peak_mb=st["kv_peak_bytes"] / 2 ** 20,
                        within_budget=st["kv_peak_bytes"] <= budget, mx_peak_over_baseline_mb=peak_delta / 2 ** 20, mx_live_end_of_tick_over_baseline_mb=(eng.active_peak - base) / 2 ** 20, yields=st.get("yields", 0),
                        drops=st.get("kv_drops", 0), rebuilds=st.get("rebuilds", 0), replay_tokens=st.get("replay_tokens", 0),
                        rebuild_prefill_tokens=st.get("rebuild_prefill_tokens", 0))
        ok &= rep[key]["equal"] and rep[key]["within_budget"] and rep[key]["drops"] > 0 and rep[key]["rebuilds"] > 0
        print(f"  {key}: " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep[key].items()), flush=True)
    report["churn"] = rep

if "churnbig" in only:
    class SyncedLive:
        """Peak live memory over baseline measured at the end of each tick AFTER mx.synchronize() (no GPU work in flight)."""
        def __enter__(self):
            self.peak, self.orig, me = 0, Engine._note_kv, self

            def note(eng):
                me.orig(eng)
                mx.synchronize()
                me.peak = max(me.peak, mx.get_active_memory())
            Engine._note_kv = note
            return self

        def __exit__(self, *e):
            Engine._note_kv = self.orig

    # KV-sized prompts: the live memory of the process follows the budget, not the number of sessions. Reference = the same mode without a limit.
    for i in range(8):
        PROMPTS[f"b{i}"] = rng.integers(16, V - 16, a.big_len + 37 * i).tolist()
    bn = [f"b{i}" for i in range(8)]
    rep = {}
    for mtp in [int(v) for v in a.big_mtps.split(',')]:
        _, _, e1 = serve([bn[-1]], sessions=1, mtp=mtp, ctx=2048, budget=1 << 40, n_new=16)
        unit = e1.kv_peak
        mx.clear_cache(); base = mx.get_active_memory()
        mx.reset_peak_memory()
        with SyncedLive() as sl_free:
            free, stf, ef = serve(bn, sessions=2, mtp=mtp, ctx=2048, floor=6, budget=1 << 40, n_new=16, stagger=0.02, oracle=False)
        live_free, synced_free = (ef.active_peak - base) / 2 ** 20, (sl_free.peak - base) / 2 ** 20
        for n in bn:
            REFC[tuple(PROMPTS[n])] = free[n]
        budget = int(unit * 2.3)
        mx.clear_cache(); base = mx.get_active_memory()
        t0 = time.perf_counter()
        with SyncedLive() as sl:
            got, st, eng = serve(bn, sessions=2, mtp=mtp, ctx=2048, floor=6, budget=budget, n_new=16, stagger=0.02, err=3)
        key = f"churnbig8_mtp{mtp}"
        print(f"   live over baseline, synchronized at the end of each tick: {(sl.peak - base) / 2 ** 20:.1f} MB (budget {budget / 2 ** 20:.1f} MB, unlimited run {synced_free:.1f} MB); "
              f"unsynchronized {(eng.active_peak - base) / 2 ** 20:.1f} MB (unlimited {live_free:.1f})", flush=True)
        print(f"   live memory left after the run: {(mx.get_active_memory() - base) / 2 ** 20:.2f} MB", flush=True)
        rep[key] = dict(equal=same(got, bn), budget_mb=budget / 2 ** 20, kv_peak_mb=st["kv_peak_bytes"] / 2 ** 20, kv_peak_unlimited_mb=stf["kv_peak_bytes"] / 2 ** 20,
                        within_budget=st["kv_peak_bytes"] <= budget, live_over_baseline_mb=(eng.active_peak - base) / 2 ** 20, live_over_baseline_unlimited_mb=live_free, live_synced_over_baseline_mb=(sl.peak - base) / 2 ** 20, live_synced_unlimited_mb=synced_free,
                        yields=st.get("yields", 0), drops=st.get("kv_drops", 0), rebuilds=st.get("rebuilds", 0), replay_tokens=st.get("replay_tokens", 0),
                        rebuild_prefill_tokens=st.get("rebuild_prefill_tokens", 0), prefix_hit_tokens=st.get("prefix_hit_tokens", 0), wall_s=time.perf_counter() - t0)
        ok &= rep[key]["equal"] and rep[key]["within_budget"] and rep[key]["rebuilds"] > 0
        print(f"  {key}: " + " ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep[key].items()), flush=True)
    report["churnbig"] = rep

if "bench" in only:
    rep = {}
    R = be.R
    for n in (1, 2, 4):
        caches = []
        for i in range(n):
            c = be.make_cache()
            be.prefill_chunk(c, PROMPTS[NAMES[i]][:64])
            caches.append(c)
        # plain batched step (the draft-off path)
        bc = be.batch([be.fork(c) for c in caches])
        tok = mx.array([7] * n)
        for _ in range(2):
            mx.eval(be.step(bc, tok))
        ts = []
        for _ in range(a.reps):
            t = time.perf_counter()
            mx.eval(be.step(bc, tok))
            ts.append(time.perf_counter() - t)
        rep[f"N{n}_plain_step_ms"] = statistics.median(ts) * 1e3
        for d in (0, 1, 2, 3):
            rows = n * (1 + d)
            bc = be.batch([be.fork(c) for c in caches])                # the batched core: all rows of all sessions in one pass
            cs = [be.fork(c) for c in caches]                          # the chained verify (reference)
            ts, tc = [], []
            for r in range(a.reps + 4):
                lists = [[7] + [11 + j for j in range(d)] for _ in range(n)]
                t = time.perf_counter()
                ps = be.spec_verify_bc(bc, lists)
                ps.commit([d] * n)
                mx.eval(ps.logits, *[x for c in bc for x in c.stacked_arrays()])
                t_core = time.perf_counter() - t
                t = time.perf_counter()
                p = be.spec_verify([(c, lists[0]) for c in cs])
                new = [p.commit(k, d) for k in range(n)]
                mx.eval(p.logits, *[x for c in new for x in be.state(c)])
                t_chain = time.perf_counter() - t
                cs = new
                if r >= 4:
                    ts.append(t_core)
                    tc.append(t_chain)
            ms, msc = statistics.median(ts) * 1e3, statistics.median(tc) * 1e3
            rep[f"N{n}_D{d}"] = dict(rows=rows, ms=ms, ms_per_row=ms / rows, ms_min=min(ts) * 1e3, chain_ms=msc, chain_ms_min=min(tc) * 1e3)
            print(f"  N={n} D={d}: {rows:2d} rows  batched core verify+commit {ms:7.1f} ms (min {min(ts) * 1e3:.1f}, {ms / rows:5.1f} ms/row)   chained verify {msc:7.1f} ms   "
                  f"plain batched step {rep[f'N{n}_plain_step_ms']:.1f} ms", flush=True)
    ORACLE["on"] = False
    rings = []
    for i in range(4):                                                 # four sessions with their own draft rings
        c, d = be.make_cache(), be.new_draft()
        for j in range(0, 96, a.chunk):
            be.prefill_chunk(c, (PROMPTS[NAMES[i]] * 3)[j:j + a.chunk], d)
        rings.append(d)
    for n in (1, 2, 4):                                                # one drafter pass for n sessions against n serial calls
        jobs = [(rings[i], 7 + i, 96, 3, None) for i in range(n)]
        tb, ts = [], []
        for r in range(a.reps + 3):
            t = time.perf_counter()
            real_many(be, jobs)
            t_b = time.perf_counter() - t
            t = time.perf_counter()
            for j in jobs:
                real_many(be, [j])
            t_s = time.perf_counter() - t
            if r >= 3:
                tb.append(t_b)
                ts.append(t_s)
        rep[f"draft_S{n}"] = dict(batched_ms=statistics.median(tb) * 1e3, serial_ms=statistics.median(ts) * 1e3, batched_ms_min=min(tb) * 1e3, serial_ms_min=min(ts) * 1e3)
        print(f"  draft for {n} session(s) (real head, block of 5): one pass {rep[f'draft_S{n}']['batched_ms']:.1f} ms (min {min(tb) * 1e3:.1f}), "
              f"{n} serial calls {rep[f'draft_S{n}']['serial_ms']:.1f} ms (min {min(ts) * 1e3:.1f})", flush=True)
    report["bench"] = rep

if TOL:
    agree = 1 - TSTAT["flips"] / max(TSTAT["tokens"], 1)
    report["tolerance"] = dict(compared_tokens=TSTAT["tokens"], flips=TSTAT["flips"], tie_flips=TSTAT["tie_flips"], agreement=agree,
                               unique_flip_positions=len(TSTAT["unique"]), unique_positions=len(TSTAT["unique_tokens"]))
    print(f"tolerance: greedy agreement vs PLAIN solo {agree:.4f} over {TSTAT['tokens']} compared tokens ({TSTAT['flips']} flips, {TSTAT['tie_flips']} at a near tie; "
          f"{len(TSTAT['unique'])} distinct (prompt, position) among {len(TSTAT['unique_tokens'])}; >= {T.MIN_AGREE})", flush=True)
    ok &= agree >= T.MIN_AGREE and TSTAT["tie_flips"] == TSTAT["flips"]
report["ok"] = bool(ok)
report["mx_peak_gb"] = mx.get_peak_memory() / 2 ** 30
print(("ALL EXACT" if ok and not TOL else "ALL WITHIN TOLERANCE" if ok else "FAILED"), f"peak memory {report['mx_peak_gb']:.2f} GB")
if a.json:
    json.dump(report, open(a.json, "w"), indent=1, default=str)
raise SystemExit(0 if ok else 1)
