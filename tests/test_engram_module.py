"""DS4.1 Engram module (wkv + gate path) against the box CPU reference. Real data only (a few hundred MB, built by tools/engram_module_ref.py on the
box and tools/pack_slice_stream.py): KYOJIN_ENGRAM_MODULE=<dir with module_ref.npz and model/> (results_mac/engram_module on the Air); skipped otherwise."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("KYOJIN_ENGRAM_MODULE")


@pytest.mark.skipif(not D or not os.path.exists(os.path.join(D or "", "module_ref.npz")), reason="KYOJIN_ENGRAM_MODULE not set")
def test_engram_module_matches_box_reference():
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "tools", "engram_module_check.py"), os.path.join(D, "module_ref.npz"), os.path.join(D, "model")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1500:]
    assert "ALL PASS" in r.stdout
