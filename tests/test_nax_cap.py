"""NAX capability probe (metal_nax.nax_status / site_on): probe result and log line, KYOJIN_NAX=0|1 overrides, simulated old GPU, failing compile,
and the dense linear on both paths (NAX forced on vs forced off) within the fp16 rounding class. Each case runs in a fresh process (the probe is cached per process)."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal not available", allow_module_level=True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(code, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("KYOJIN_NAX") and k not in ("KYOJIN_PF_FUSED", "KYOJIN_DENSE_FUSED", "KYOJIN_ATTN_PF", "KYOJIN_MIX_PF", "KYOJIN_ROWS_NAX")}
    e.update(env, PYTHONPATH=REPO)
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=e, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout, r.stderr


SITES = """
import json
from kyojin_mlx import metal_nax as NX, attn_prefill as A, mhc_prefill as H, moe_prefill as MP
print(json.dumps(dict(st=NX.nax_status(), sites=[NX.nax_available(), A.available(), H.available(), MP.fused_default(), NX.site_on("1")])))
"""


def test_probe_on_the_air_is_logged():
    out, err = run(SITES)
    d = json.loads(out)
    if not d["st"]["arch_nax"]:
        pytest.skip("no neural accelerators on this GPU")
    assert d["st"]["on"] and d["st"]["compiles"] and all(d["sites"]), d
    assert "[kyojin] NAX ON" in err and d["st"]["arch"] in err


def test_off_switch_turns_every_site_off_even_forced():
    out, err = run(SITES, KYOJIN_NAX="0", KYOJIN_ATTN_PF="1", KYOJIN_PF_FUSED="1", KYOJIN_DENSE_FUSED="1", KYOJIN_MIX_PF="1")
    d = json.loads(out)
    assert not d["st"]["on"] and not any(d["sites"]), d
    assert "[kyojin] NAX OFF" in err


def test_site_off_wins_over_the_probe():
    out, _ = run(SITES + "\nprint(NX.site_on('0'))", KYOJIN_ATTN_PF="0", KYOJIN_PF_FUSED="0")
    d = json.loads(out.splitlines()[0])
    assert d["sites"][1] is False and d["sites"][3] is False and out.splitlines()[1] == "False"


OLD_GPU = """
import json
from kyojin_mlx import metal_nax as NX
NX._architecture = lambda: "applegpu_g14g"      # an M2 / M3 class GPU
from kyojin_mlx import attn_prefill as A, moe_prefill as MP
print(json.dumps(dict(st=NX.nax_status(), sites=[NX.nax_available(), A.available(), MP.fused_default()], forced=NX.site_on("1"))))
"""


def test_old_gpu_falls_back_without_compiling():
    out, err = run(OLD_GPU)
    d = json.loads(out)
    assert not d["st"]["on"] and not d["st"]["compiles"] and not any(d["sites"]) and "no neural accelerators" in d["st"]["reason"], d
    assert "NAX OFF" in err


def test_unknown_gpu_with_nax_forced_still_needs_the_probe():
    out, _ = run(OLD_GPU, KYOJIN_NAX="1")
    d = json.loads(out)
    assert d["st"]["on"] == d["st"]["compiles"]           # on exactly when matmul2d compiled and matched numpy here


BROKEN = """
import json
from kyojin_mlx import metal_nax as NX
NX.HEADER_NAX = NX.HEADER_NAX + "\\n#include <no_such_header_for_nax.h>\\n"
print(json.dumps(dict(st=NX.nax_status(), site=NX.site_on("1"), avail=NX.nax_available())))
"""


def test_failing_compile_probe_means_off_not_a_crash():
    out, err = run(BROKEN)
    d = json.loads(out)
    assert d["st"]["on"] is False and d["site"] is False and d["avail"] is False and "probe failed" in d["st"]["reason"], d
    assert "NAX OFF" in err


AB = """
import sys, numpy as np, mlx.core as mx
from kyojin_mlx import metal_nax as NX, exl3_linear as EL
rng = np.random.default_rng(7)
out = {}
for cb, K, kin, nout in (("mul1", 3, 1024, 384), ("mcg", 2, 512, 1024)):
    sh = (kin // 16, nout // 16, 16 * K)
    tr = rng.integers(-32768, 32768, sh, dtype=np.int16)
    suh = (rng.choice([-1.0, 1.0], kin) * rng.uniform(0.5, 1.5, kin)).astype(np.float16)
    svh = (rng.choice([-1.0, 1.0], nout) * rng.uniform(0.5, 1.5, nout)).astype(np.float16)
    lin = EL.Exl3Linear(mx.array(tr), mx.array(suh), mx.array(svh), cb=cb)
    for rows in (40, 300):
        x = mx.array((rng.standard_normal((rows, kin)) * 2).astype(np.float16))
        y = lin(x); mx.eval(y)
        out[f"{cb}_{rows}"] = np.array(y).astype(np.float32)
        out[f"{cb}_{rows}_fused"] = np.array(lin._fused_ok(rows))
np.savez(sys.argv[1], **out)
"""


def test_dense_linear_both_paths_agree(tmp_path):
    st, _ = run(SITES)
    if not json.loads(st)["st"]["on"]:
        pytest.skip("no neural accelerators on this GPU")
    res = {}
    for name, env in (("nax", {}), ("off", {"KYOJIN_NAX": "0"})):
        f = str(tmp_path / f"{name}.npz")
        subprocess.run([sys.executable, "-c", textwrap.dedent(AB), f], check=True, capture_output=True, text=True, timeout=600,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith("KYOJIN_NAX")}, **env, "PYTHONPATH": REPO})
        res[name] = np.load(f)
    for k in res["nax"].files:
        if k.endswith("_fused"):
            assert bool(res["nax"][k]) and not bool(res["off"][k]), k          # the two runs really took different paths
            continue
        a, b = res["nax"][k], res["off"][k]
        rel = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
        assert rel < 3e-3, (k, rel)
