"""Pool the --json outputs of tolerance-mode runs (spec_check, mdraft_check, serve_spec_check) and judge the greedy top-1 agreement of the pool.
usage: tolpool.py F.json ...   (each file: rows, flips, tie_flips)   rc 0 iff agreement >= 99.5 % and every flip is at a near tie"""
import json, sys

import tolgate as T

rows = flips = tie = 0
for f in sys.argv[1:]:
    d = json.load(open(f))
    rows += d["rows"]; flips += d["flips"]; tie += d["tie_flips"]
agree = 1 - flips / max(rows, 1)
ok = agree >= T.MIN_AGREE and tie == flips and rows > 0
print(f"pool of {len(sys.argv) - 1} runs: top-1 agreement vs solo {agree:.4f} over {rows} rows ({flips} flips, {tie} at a near tie; >= {T.MIN_AGREE})  =>  {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
