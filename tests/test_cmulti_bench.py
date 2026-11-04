"""tools/bench_cmulti.py end to end on CPU: a stub HTTP server (the routes the script uses) in front of an Engine on the grid-sensitive fake. Checks the plumbing and
the arithmetic of the script, not any speed: the costs are simulated sleeps."""
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kyojin_mlx import flags as FL
from kyojin_mlx.serve.engine import Engine
from cmulti_fakes import GridFake

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_script():
    spec = importlib.util.spec_from_file_location("bench_cmulti", os.path.join(ROOT, "tools", "bench_cmulti.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def stub(monkeypatch):
    for f in ("mix_decode", "mix_slice_ms", "pf_share", "kv_admit"):
        monkeypatch.setenv(FL.FLAGS[f].env, "0")
    be = GridFake(pf_ms=0.02, step_ms=0.5, layers=8)
    eng = Engine(be, sessions=8, ctx=1 << 16, chunk=64, block=1 << 15, prefix_bytes=0, kv_budget=8 * 15000, kv_refresh=1)
    eng.start()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send(200, {**eng.status(), "kv_budget_gb": eng.kv_budget / 2 ** 30, "requests": 0, "timing": {}})   # unrounded: the budget of the stub is tiny

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/admin/flags":
                st = eng.status()
                if st["waiting"] or st["prefilling"] or st["decoding"]:
                    return self._send(409, {})
                FL.set_many(body)
                eng.prefix.shrink(0)
                return self._send(200, {"flags": {}})
            ids = body["prompt"]
            try:
                h = eng.submit(ids, max_new=body["max_tokens"])
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            n, reason, stats = 0, None, None
            try:
                while True:
                    ev = h.events.get(timeout=60)
                    if ev[0] == "tok":
                        self.wfile.write(f"data: {json.dumps({'choices': [{'text': chr(65 + ev[1] % 26), 'finish_reason': None}]})}\n\n".encode()); self.wfile.flush()
                    else:
                        reason, stats = ev[1], ev[2]
                        break
                self.wfile.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': reason}]})}\n\n".encode())
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': stats['prompt_tokens'], 'prompt_tokens_details': {'cached_tokens': stats['cached_tokens']}}})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                h.cancel()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", eng
    srv.shutdown()
    eng.stop()


def test_dry_run_plans_without_a_server(capsys):
    assert load_script().main(["--dry-run", "--url", "http://127.0.0.1:1"]) == 0
    assert "mix50+slice" in capsys.readouterr().out


COMMON = ["--id-lo", "1", "--id-hi", "90", "--reps", "1"]


def test_gap_and_children_scenarios_run_against_the_stub(stub, tmp_path):
    url, eng = stub
    out = tmp_path / "r.json"
    rc = load_script().main(["--url", url, "--scenarios", "gap,children", "--pf-tokens", "2000", "--slice-ms", "2", "--a-tokens", "4000", "--children", "3", "--prefix-tokens", "600",
                             "--suffix-tokens", "10", "--check-tokens", "6", "--json", str(out), *COMMON])
    r = json.load(open(out))
    assert rc == 0 and "aborted" not in r
    g = r["gap"]
    assert set(g) == {"off", "mix50", "slice", "mix50+slice"}
    assert g["slice"]["decoder_tokens_in_window"][0] > g["off"]["decoder_tokens_in_window"][0]          # the plumbing sees the lever (simulated costs)
    c = r["children"]["600"]
    assert c["texts_equal"] == [True]                                                                      # flag on and off give the same child texts (grid-sensitive fake)
    assert c["pf_share"][0]["server_prefill_tokens"] < c["off"][0]["server_prefill_tokens"]
    assert r["rss_max_mib"] < 3.5 * 1024
    for k in ("mix_decode", "mix_slice_ms", "pf_share", "kv_admit"):
        assert os.environ[FL.FLAGS[k].env] == "0"                                                           # the script puts the flags back


def test_memory_scenario_and_its_bound(stub, tmp_path):
    url, eng = stub
    mod = load_script()
    out = tmp_path / "m.json"
    rc = mod.main(["--url", url, "--scenarios", "memory", "--mem-sessions", "6", "--mem-tokens", "3000", "--mem-new", "400", "--mem-over", "100", "--mem-max-gib", "1", "--json", str(out), *COMMON])
    r = json.load(open(out))
    assert rc == 0, r
    arms = r["memory"]["arms"]
    assert arms["kv_admit"]["kv_peak_bytes"] <= 8 * 15000 < arms["off"]["kv_peak_bytes"]
    rc = mod.main(["--url", url, "--scenarios", "memory", "--mem-sessions", "60", "--mem-tokens", "3000", "--mem-over", "2.5", "--json", str(out), *COMMON])
    assert rc == 1 and "server runs" in json.load(open(out))["aborted"]                  # more sessions than the server runs: refused before anything is sent


def test_memory_auto_sizing_and_absolute_cap(stub, tmp_path):
    url, eng = stub
    mod = load_script()
    out = tmp_path / "a.json"
    rc = mod.main(["--url", url, "--scenarios", "memory", "--mem-tokens", "3000", "--mem-new", "400", "--mem-max-gib", "1", "--mem-target-x", "1.5", "--json", str(out), *COMMON])
    r = json.load(open(out))
    assert rc == 0, r
    assert 1.4 <= r["memory"]["need_over_budget"] <= 1.9                          # 3400 tokens x 8 B per session against a 120000 B budget: 5 sessions = 1.13x? derived to reach 1.5x
    rc = mod.main(["--url", url, "--scenarios", "memory", "--mem-tokens", "3000", "--mem-max-gib", "0.0000001", "--mem-target-x", "1.5", "--json", str(out), *COMMON])
    assert rc == 1 and "mem-max-gib" in json.load(open(out))["aborted"]
