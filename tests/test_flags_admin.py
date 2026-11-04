"""kyojin_mlx.flags (lever flags switched at call time) and the owner-only admin endpoint GET / POST /admin/flags of the server."""
import http.client
import json
import os
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import flags as FL                                           # noqa: E402

METAL = mx.metal.is_available()
needs_metal = pytest.mark.skipif(not METAL, reason="needs Metal")


@pytest.fixture(autouse=True)
def restore():
    """Every test leaves the env and the module attributes as it found them."""
    env = {f.env: os.environ.get(f.env) for f in FL.FLAGS.values()}
    env["KYOJIN_FUSE_MIXPRE"] = os.environ.get("KYOJIN_FUSE_MIXPRE")
    from kyojin_mlx import fuse as F, glue_fuse as GN, gluefuse as GF
    attrs = [(FL._mod(m), a) for f in FL.FLAGS.values() for m, a, _ in f.attrs] + [(F, "MIXPRE"), (GN, "ENABLED"), (GF, "ENABLED")]
    saved = [(o, a, getattr(o, a)) for o, a in attrs]
    yield
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    for o, a, v in saved:
        setattr(o, a, v)


def test_get_all_lists_every_lever_with_its_current_value(monkeypatch):
    monkeypatch.setenv("KYOJIN_MOEFUSE", "1")
    a = FL.get_all()
    assert set(a) == set(FL.FLAGS)
    assert a["moefuse"]["value"] == "1" and a["moefuse"]["runtime"]
    assert not a["engram_gate"]["runtime"] and "mlx" in a["engram_gate"]["reason"]


@pytest.mark.parametrize("name", [n for n, f in FL.FLAGS.items() if f.runtime])
def test_set_writes_env_and_module_attributes(name):
    f = FL.FLAGS[name]
    for v in ("1", "0") if f.kind == "bool" else ("7", "0"):
        g = FL.GEN
        FL.set(name, v)
        assert os.environ[f.env] == v and FL.GEN == g + 1 and FL.get_all()[name]["value"] == v
        for m, a, fn in f.attrs:
            assert getattr(FL._mod(m), a) == fn(v)


def test_set_values_and_errors():
    from kyojin_mlx import moe as M, moe_fused as MF
    FL.set("moe_fastdec", True)
    assert M.FASTDEC is True
    FL.set("moe_fastdec", "off")
    assert M.FASTDEC is False
    with pytest.raises(PermissionError):
        FL.set("engram_gate", 0)
    with pytest.raises(KeyError):
        FL.set("no_such_flag", 1)
    before = MF.MOEFUSE
    with pytest.raises(ValueError):                              # validated before anything is applied
        FL.set_many({"moefuse": 1, "pf_glu": "maybe"})
    assert MF.MOEFUSE == before
    with pytest.raises(ValueError):
        FL.set("idx_qblock", "x")


def test_glue_pair_is_applied_together():
    from kyojin_mlx import fuse as F, glue_fuse as GN, gluefuse as GF
    os.environ.pop("KYOJIN_FUSE_MIXPRE", None)
    FL.set_many({"glue_fuse": 0, "glue_fuse_norm": 1})
    assert not GF.ENABLED and not GN.ENABLED and F.MIXPRE          # the norm needs glue_fuse
    FL.set("glue_fuse", 1)
    assert GF.ENABLED and GN.ENABLED and F.MIXPRE is (not GF.MIXPREOFF)
    FL.set_many({"glue_fuse": 0, "glue_fuse_norm": 0})
    assert not GF.ENABLED and not GN.ENABLED and F.MIXPRE


def test_dspol_is_read_per_round():
    from kyojin_mlx.serve.engine import Engine
    eng = object.__new__(Engine)
    FL.set("dspol", 0)
    assert eng._dspol_on is False
    FL.set("dspol", 1)
    assert eng._dspol_on is True


@needs_metal
def test_moe_fastdec_codebook_follows_the_flag_at_call_time():
    from kyojin_mlx import moe as M
    from test_moe import KS, TOPK, make_stack
    st = [make_stack(256, 384, KS, "mul1", s)[0] for s in (1, 2)] + [make_stack(384, 256, KS, "mul1", 3)[0]]
    m = M.MoeGather(*st, TOPK)
    FL.set("moe_fastdec", 0)
    assert m.cbf == M._roof(m.cbg)
    FL.set("moe_fastdec", 1)
    assert m.cbf == M._roof("mul1fd")
    m.fastdec = False                                            # a per-instance pin wins
    assert m.cbf == M._roof(m.cbg)


@needs_metal
def test_dense_fast_switches_a_built_linear():
    from kyojin_mlx import kernels as KN, metal_src as S
    rng = np.random.default_rng(0)
    kin, nout, K = 512, 384, 3
    tr = rng.integers(-32768, 32768, (kin // 16, nout // 16, 16 * K), dtype=np.int16)
    su, sv = (rng.uniform(0.5, 1.5, n).astype(np.float16) for n in (kin, nout))
    g = KN.Mul1Gemv(tr, su, sv, rows=1, out_dtype=mx.float16, cb="mul1")
    x = mx.array(rng.standard_normal((1, kin)).astype(np.float16))
    FL.set("dense_fast", 0)
    a = np.array(g.fused(x))
    FL.set("dense_fast", 1)
    b = np.array(g.fused(x))
    assert {k[-1] for k in g._fc} == {S.TAIL_ATOM, S.TAIL_VEC}   # one cached entry per read mode: the switch reached the built linear
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


@needs_metal
def test_bound_layers_switch_on_an_installed_model():
    from kyojin_mlx import bound_layers as BL
    from test_bound_layers import Cache, make_model, run
    m, _ = make_model(8)
    FL.set("bound_layers", 0)
    BL.reset()
    BL.install(m)
    run(m.language_model, 128, Cache())
    assert BL.stats()["bounds"] == 0
    FL.set("bound_layers", 1)
    run(m.language_model, 128, Cache())
    assert BL.stats()["bounds"] == 2


# ------------------------------------------------------------------------------------------------ server endpoint
class _Prefix:
    def __init__(self):
        self.shrunk = []

    def shrink(self, n):
        self.shrunk.append(n)
        return True


class _Engine:
    def __init__(self):
        self.busy = 0
        self.prefix = _Prefix()

    def status(self):
        return {"waiting": 0, "prefilling": 0, "decoding": self.busy}


class _App:
    def __init__(self):
        self.engine = _Engine()
        self.logs = []

    def log(self, s):
        self.logs.append(s)


@pytest.fixture
def server():
    from kyojin_mlx.serve import server as SV
    app = _App()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), SV.make_handler(app))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield app, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def req(port, method, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = None if body is None else json.dumps(body).encode()
    c.request(method, "/admin/flags", body=data, headers={"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    out = r.status, json.loads(r.read())
    c.close()
    return out


def test_admin_get_and_post(server):
    from kyojin_mlx import moe_fused as MF
    app, port = server
    code, out = req(port, "GET")
    assert code == 200 and set(out["flags"]) == set(FL.FLAGS)
    code, out = req(port, "POST", {"moefuse": 1, "idx_qblock": 256})
    assert code == 200 and MF.MOEFUSE is True and out["flags"]["idx_qblock"]["value"] == "256" and out["gen"] == FL.GEN
    assert app.engine.prefix.shrunk == [0]                      # the memory prefix cache is dropped after a switch


def test_admin_refusals(server):
    app, port = server
    assert req(port, "POST", {"engram_gate": 0})[0] == 400
    assert req(port, "POST", {"nope": 1})[0] == 400
    assert req(port, "POST", {"pf_glu": "maybe"})[0] == 400
    app.engine.busy = 1
    assert req(port, "POST", {"moefuse": 1})[0] == 409
    assert app.engine.prefix.shrunk == []


def test_admin_is_localhost_only(server, monkeypatch):
    from kyojin_mlx.serve import server as SV
    assert SV.is_loopback("127.0.0.1") and SV.is_loopback("::1") and not SV.is_loopback("192.168.1.20")
    monkeypatch.setattr(SV, "is_loopback", lambda host: False)    # what a remote peer sees
    _, port = server
    assert req(port, "GET")[0] == 403
    assert req(port, "POST", {"moefuse": 1})[0] == 403


def test_admin_cache_limit_mb(server):
    app, port = server
    old = mx.set_cache_limit(64 * 2 ** 20)
    try:
        code, out = req(port, "POST", {"cache_limit_mb": 128})
        assert code == 200 and out["cache_limit_mb"] == 128
        assert mx.set_cache_limit(old) == 128 * 2 ** 20
        assert req(port, "POST", {"cache_limit_mb": -1})[0] == 400
        assert req(port, "POST", {"cache_limit_mb": "x"})[0] == 400
    finally:
        mx.set_cache_limit(old)


def test_cache_limit_default_follows_the_weights():
    from kyojin_mlx.serve.server import cache_limit_mb_for
    assert cache_limit_mb_for(5 * 2 ** 30) == 512
    assert cache_limit_mb_for(10 * 2 ** 20) == 256
    assert cache_limit_mb_for(172 * 2 ** 30) == 16384
    assert cache_limit_mb_for(10 * 2 ** 20, 243 * 2 ** 30) == 7776    # the floor follows the machine (a small model on a big Mac)
    assert cache_limit_mb_for(10 * 2 ** 20, 8 * 2 ** 30) == 256
    assert cache_limit_mb_for(172 * 2 ** 30, 243 * 2 ** 30) == 16384


def test_cache_limit_floor_when_the_weights_leave_little_room():
    from kyojin_mlx.serve.server import cache_limit_mb_for
    G = 2 ** 30
    assert cache_limit_mb_for(int(8.8 * G), 10 * G, int(1.2 * G)) == 320            # 16 GB Air, 27B GGUF: the floor (working set / 32), not weights / 10 (899 MB)
    assert cache_limit_mb_for(int(8.8 * G), 10 * G, 30 * G) == 901                  # same weights with room: weights / 10
    assert cache_limit_mb_for(172 * G, 243 * G, 39 * G) == 16384                    # DS4.1 on the Ultra keeps its cache
    assert cache_limit_mb_for(int(8.8 * G), 10 * G, None) == 901                    # room unknown: unchanged


def test_flag_table_defaults_reach_the_lever_modules():
    """A default flipped in kyojin_mlx.flags must change what the code reads (the modules read their variable once at import)."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("KYOJIN_")}
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    code = ("import kyojin_mlx\nfrom kyojin_mlx import suglue, gluefuse, glue_fuse, idxfast, depthflat, moe\n"
            "print(suglue.ENABLED, gluefuse.ENABLED, glue_fuse.ENABLED, idxfast.ENABLED, depthflat.ENABLED, moe.ROOF)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root)
    want = [FL.FLAGS[n].default == "1" for n in ("port_suglue", "glue_fuse", "glue_fuse_norm", "idx_fast", "depthflat", "moeroof")]
    assert r.stdout.split() == [str(w) for w in want], r.stderr[-400:]


@pytest.mark.parametrize("name", [n for n, f in FL.FLAGS.items() if f.runtime])
def test_every_runtime_flag_accepts_its_own_default(name):
    """A switchable flag must take its table default back through set() (gpu_keepalive's 0.5 was refused by the 0/1 parser)."""
    f = FL.FLAGS[name]
    assert FL._norm(f, f.default) == (f.default if f.kind == "int" else ("1" if FL._on(f.default) else "0"))
