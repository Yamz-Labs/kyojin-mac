import os, sys, runpy
from kyojin_mlx import exl3_linear as EL
cnt = {}
for n in ("_mma", "_ksplit", "_dense", "_many"):
    def mk(f, n=n):
        def g(self, x2, *a, **k):
            key = (n, x2.shape[0], self.K); cnt[key] = cnt.get(key, 0) + 1
            return f(self, x2, *a, **k)
        return g
    setattr(EL.Exl3Linear, n, mk(getattr(EL.Exl3Linear, n)))
sys.argv = ["spec_check.py"] + sys.argv[1:]
try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_check.py"), run_name="__main__")
except SystemExit as e:
    print("exit", e.code)
print("COUNTS", sorted(cnt.items()))
