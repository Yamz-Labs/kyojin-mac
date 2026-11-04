"""tools/decspeed2_ab.py against a stub of the serve endpoints: the harness itself, on CPU, with no model and no lane.

The tool is the owner's harness for the served A/B (fixed prose + code prompt, greedy, flags off against on, tok/s and GPU busy, the emitted tokens
compared between arms). What can be checked without a lane is the plumbing: the arm switching through POST /admin/flags, that `base` is every lever at 0,
that the ABBA order is used, that a repeat is dropped after a switch, that tok/s and the gpu_busy deltas are read from the fields the server really
sends, that a different text on an arm is reported as TEXT-CHANGED, and that the bounds and the loopback rule refuse.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import decspeed2_ab as AB

LEVERS = dict(AB.LEVERS)


class Stub:
    """A serve stand-in: /health, /admin/flags (GET+POST), /v1/completions, /v1/tokenize, /v1/stats."""

    def __init__(self, slow_arm=None, gain=1.35, text_break=None):
        self.flags = {k: "0" for k in LEVERS}
        self.flags.update({"half_rows": "4", "fuse_rows": "8", "gpu_span": "0"})
        self.calls = []                      # (path, flags snapshot at the request, prompt head)
        self.gpu = 0.0
        self.slow_arm, self.gain, self.text_break = slow_arm, gain, text_break
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def _handler(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/health":
                    return self._send({"status": "ok", "model": "ds41"})
                if p == "/admin/flags":
                    return self._send({"flags": dict(stub.flags), "gen": 1})
                if p == "/v1/stats":
                    return self._send({"timing": {"gpu_busy_r2": stub.gpu, "gpu_n_r2": stub.gpu / 10.0, "t_rbuild": 1.0}})
                self._send({"error": "no"}, 404)

            def do_POST(self):
                p = self.path.split("?")[0]
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if p == "/admin/flags":
                    for k, v in body.items():
                        if k == "mtp_fixed":
                            continue
                        if k not in stub.flags:
                            return self._send({"error": "unknown flag"}, 400)
                        stub.flags[k] = str(v)
                    return self._send({"flags": dict(stub.flags)})
                if p == "/v1/tokenize":
                    return self._send({"tokens": [ord(c) for c in body.get("prompt") or ""], "count": len(body.get("prompt") or "")})
                if p == "/v1/completions":
                    arm = tuple(sorted((k, v) for k, v in stub.flags.items() if k in LEVERS))
                    stub.calls.append((p, arm, (body.get("prompt") or "")[:16]))
                    slow = stub.slow_arm is not None and dict(arm).get(stub.slow_arm) == "1"
                    stub.gpu += 10.0
                    text = "the harbour at first light is a study in grey"
                    if stub.text_break and dict(arm).get(stub.text_break) == "1":
                        text = text.replace("grey", "gray")
                    tps = 50.0 * (stub.gain if slow else 1.0)
                    return self._send({"choices": [{"text": text, "finish_reason": "length"}],
                                       "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
                                       "timings": {"ttft_s": 0.4, "total_s": 0.6, "predicted_per_second": tps}})
                self._send({"error": "no"}, 404)

        return H


@pytest.fixture()
def stub():
    s = Stub(slow_arm="spec_ladder")
    yield s
    s.close()


def run(stub, tmp_path, argv):
    out = tmp_path / "ab.json"
    rc = AB.run(["--base", stub.url, "--json", str(out), "--reps", "2", "--max-tokens", "32"] + argv)
    return rc, out


def test_default_arms_are_the_track_levers_and_base_is_all_zero(stub, tmp_path):
    rc, out = run(stub, tmp_path, [])
    assert rc == 0
    j = json.loads(out.read_text())
    assert set(j["report"]["prose"]) == {"base", "spec_ladder=1", "cdecpack=1", "rg_half=1"}
    # base is every lever of the track at 0, and each arm switches exactly its own lever on
    seen = {arm for _, arm, _ in stub.calls}
    assert (("cdecpack", "0"), ("rg_half", "0"), ("spec_ladder", "0")) in seen
    assert (("cdecpack", "1"), ("rg_half", "0"), ("spec_ladder", "0")) in seen
    assert j["flags"]["half_rows"] == "4"


def test_abba_order_and_a_dropped_warmup_per_switch(stub, tmp_path):
    run(stub, tmp_path, ["--arms", "spec_ladder=1", "--reps", "2"])
    prose = [c for c in stub.calls if c[2].startswith("The harbour")]
    arms = [c[1] for c in prose]
    assert len(prose) == 6                                             # 2 arms x 2 reps + one dropped warm-up per arm
    measured = [arms[i] for i in (1, 3, 4, 5)]                         # the measured visits: base, arm, arm, base (ABBA)
    assert measured[0] == measured[3] and measured[1] == measured[2]   # base then arm, then arm then base: ABBA
    base0 = (("cdecpack", "0"), ("rg_half", "0"), ("spec_ladder", "0"))
    assert measured[0] == base0 and measured[1] != base0               # the first measured visit is the all-off base
    assert arms[0] == arms[1] and arms[2] == arms[3]                   # the warm-up carries the same flags as its measured twin
    assert prose[0][2] == prose[1][2]
    assert prose[0][2] == prose[1][2]                                  # the same prompt text is used in every visit


def test_toks_per_second_and_gpu_busy_are_read_from_the_server_fields(stub, tmp_path):
    rc, out = run(stub, tmp_path, ["--arms", "spec_ladder=1"])
    j = json.loads(out.read_text())
    base, arm = j["report"]["prose"]["base"], j["report"]["prose"]["spec_ladder=1"]
    assert base["tps"] == pytest.approx(50.0, abs=0.01) and arm["tps"] == pytest.approx(67.5, abs=0.01)
    assert arm["vs_base_pct"] == pytest.approx(35.0, abs=0.1)
    assert base["gpu_ms_round"] == pytest.approx(10.0, abs=0.01)       # 10 ms busy / 1 round
    assert base["gpu_ms_tok"] == pytest.approx(1.25, abs=0.01)         # 10 ms / 8 tokens
    assert base["tokens"] == 8 and base["text_same"] and base["token_same"]


def test_a_win_needs_same_tokens(tmp_path):
    s = Stub(slow_arm="rg_half", gain=1.35, text_break="rg_half")
    try:
        rc, out = run(s, tmp_path, ["--arms", "rg_half=1"])
        j = json.loads(out.read_text())
        assert rc == 1 and "TEXT-CHANGED" in j["verdict"]
        assert j["report"]["code"]["rg_half=1"]["text_same"] is False
        assert j["report"]["code"]["rg_half=1"]["vs_base_pct"] > 30.0   # the speed is reported, it is just not a win
    finally:
        s.close()


def test_a_same_text_gain_is_flagged_as_a_win_candidate(tmp_path):
    s = Stub(slow_arm="rg_half", gain=1.35)
    try:
        rc, out = run(s, tmp_path, ["--arms", "rg_half=1"])
        j = json.loads(out.read_text())
        assert rc == 0 and "WIN candidate: rg_half=1" in j["verdict"]
    finally:
        s.close()


def test_the_flags_are_left_as_they_were_found(stub, tmp_path):
    stub.flags["spec_ladder"] = "1"                                   # the lane was serving with the ladder on
    stub.flags["gpu_span"] = "0"
    run(stub, tmp_path, ["--arms", "rg_half=1"])
    assert stub.flags["spec_ladder"] == "1" and stub.flags["cdecpack"] == "0" and stub.flags["rg_half"] == "0"
    assert stub.flags["gpu_span"] == "0"


def test_bounds_and_loopback_are_refused(stub, tmp_path):
    assert AB.run(["--base", stub.url, "--reps", "9"]) == 2
    assert AB.run(["--base", stub.url, "--max-tokens", "4000"]) == 2
    assert AB.run(["--base", "http://10.0.0.9:8292"]) == 2
    assert AB.run(["--base", stub.url, "--arms", "not_a_lever=1"]) == 2
    assert not stub.calls                                            # nothing was sent


def test_unknown_flag_on_the_server_is_refused(tmp_path):
    s = Stub()
    try:
        s.flags.pop("cdepack", None)
        s.flags.pop("cdecpack", None)                                 # a build without the lever
        assert AB.run(["--base", s.url, "--reps", "1", "--max-tokens", "32"]) == 2
    finally:
        s.close()


def test_prompts_are_fixed_and_short():
    """The two classes are the same text in every run (a prompt edit changes the baseline), and short context."""
    for p in (AB.PROSE, AB.CODE):
        assert 200 < len(p) < 800
    assert AB.PROSE.startswith("The harbour at first light") and "merge_ranges" in AB.CODE


def test_source_has_no_prompt_or_row_count_special_case():
    """Generic only: the tool must not name a pack, a model or a row count (the common rules)."""
    src = open(AB.__file__).read()
    assert not re.search(r"\bds41|deepseek|qwen|glm", src, re.I)
