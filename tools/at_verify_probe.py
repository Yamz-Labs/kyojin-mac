#!/usr/bin/env python3
"""autotune_mlx.tune_prefill on one shape, verify band only: prints the MMA / NAX picks and the entry. usage: at_verify_probe.py KIN NOUT K [rounds]"""
import sys
sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/tools/qualify")
import autotune_mlx as AT
kin, nout, K = (int(v) for v in sys.argv[1:4]); rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
AT.ROWS_GRID = (5, 8, 12, 16)
pf = AT.tune_prefill(kin, nout, K, "mul1", rounds)
for r, v in pf.items(): print(r, v["pick"], "ks", v.get("ks", {}).get("pick"), "mma", v.get("mma"))
print(AT.entry_from(dict(default=4, pick=4), pf))
