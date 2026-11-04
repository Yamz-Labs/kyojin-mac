"""Wall ms per phase of the served spec round (draft_lazy / spec_verify_lazy / sample+read / accept+commit) via monkeypatched timers; runs tools/mu117.py.
usage: mu3_phase.py <mu117 args>"""
import sys, os, time, runpy, collections
os.environ.setdefault("KYOJIN_TQ", "6:6:4:256")
from kyojin_mlx.serve import q35mlx as QM, engine as EN
T = collections.defaultdict(float); C = collections.Counter()
def wrap(cls, name):
    f = getattr(cls, name)
    def g(*a, **k):
        t0 = time.perf_counter(); r = f(*a, **k); T[name] += time.perf_counter() - t0; C[name] += 1; return r
    setattr(cls, name, g)
bk = [c for c in vars(QM).values() if isinstance(c, type) and hasattr(c, "draft_lazy")][0]
for n in ("draft_lazy", "spec_verify_lazy", "settle", "_draft_arrays", "_draft_arrays_batched", "_mtp_rows_multi", "_verify"):
    wrap(bk, n)
for n in ("_spec_tick",):
    wrap(EN.Engine, n)
orig_sr = EN.sample_rows
def sr(*a, **k):
    t0 = time.perf_counter(); r = orig_sr(*a, **k); T["sample_rows"] += time.perf_counter() - t0; C["sample_rows"] += 1; return r
EN.sample_rows = sr
sys.argv = ["mu117.py"] + sys.argv[1:]
try:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "mu117.py"), run_name="__main__")
finally:
    for k, v in T.items():
        print("PHASE %-18s n=%4d total %.2f s  per call %.1f ms" % (k, C[k], v, v / max(C[k], 1) * 1e3))
