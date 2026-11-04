"""tools/tolgate.py (tolerance class of the NAX rows) and tools/tolpool.py (pooled top-1 agreement)."""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tolgate as T                                                       # noqa: E402


def test_maxrel_and_margin():
    ref = np.array([1.0, -4.0, 2.0, 3.5])
    assert T.maxrel(ref, ref) == 0.0
    assert abs(T.maxrel(ref + np.array([0.0, 0.4, 0.0, 0.0]), ref) - 0.1) < 1e-12
    assert T.margin(ref) == 1.5 and T.margin(np.array([9.0, 9.0, 1.0])) == 0.0


def test_near_tie_bounds():
    r = np.array([10.0, 9.5, 2.0])
    assert T.near_tie(r, 1, dmax=0.3) and not T.near_tie(r, 1, dmax=0.2)      # gap 0.5 <= 2 x dmax
    assert T.near_tie(r, 1) and not T.near_tie(r, 2)                           # without dmax: TIE_REL x max|logit| = 1.0
    assert T.near_tie(r) and T.near_tie(np.array([10.0, 9.99, 0.0]))


def test_tolpool_verdict(tmp_path):
    def pool(rows, flips, tie):
        f = tmp_path / f"r{rows}_{flips}_{tie}.json"
        f.write_text(json.dumps(dict(rows=rows, flips=flips, tie_flips=tie)))
        return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tolpool.py"), str(f)], cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "tools")})
    assert pool(1200, 5, 5).returncode == 0                                   # 99.58 %
    assert pool(1200, 7, 7).returncode == 1                                   # 99.42 %: under the rate
    assert pool(1200, 1, 0).returncode == 1                                   # a flip that is not a near tie
