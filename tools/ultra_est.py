"""Ultra estimate of the multi-user step from the Air slice measurements (bench_mu_vlm.py json, one file per attention-core variant).
Model: step(N) = work(N) / S + launches(N) * c.  work(N) = Air step(N) - launches(N) * c  (what is left after the launch floor), S = how much faster
the Ultra runs that work (bandwidth / cores), c = cost of one dependent launch (Air: 1.6 us for a tiny op in a chain, 3.4 us measured in the DS4.1
attention core). Launch cost does not scale with S. Launches and ms/step are the measured ones of each file (mean launches over consecutive steps).
usage: ultra_est.py mu_ds2_user.json [mu_ds2_batched.json ...]   (the label is the file's "core" field: user = per-user cores, batched = batched core)"""
import json, sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    d, kind = r["n"], r.get("core", "user")
    ns = sorted(int(k) for k in d)
    for S in (4, 8):
        for c in (1.6e-3, 3.4e-3):
            t = {n: (d[str(n)]["ms_step"] - d[str(n)]["launches"] * c) / S + d[str(n)]["launches"] * c for n in ns}
            row = "  ".join(f"N={n}: x{n * t[1] / t[n]:.2f} agg, {t[1] / t[n] * 100:.0f}% per user" for n in ns if n > 1)
            print(f"S={S} c={c * 1e3:.1f}us {kind:8s} solo {t[1]:.2f} ms/2 layers | {row}")
