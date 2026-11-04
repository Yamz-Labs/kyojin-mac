"""The real HTTP handler (kyojin_mlx.serve.server) streaming bursts of tokens over a socket, flags off and on. mlx is stubbed (the HTTP threads never
touch it); the engine is a fake whose handle is filled with bursts like a speculative round commits them."""
import http.client
import json
import queue
import sys
import threading
import types

import pytest

pytest.importorskip("jinja2")

try:
    import mlx.core  # noqa: F401
except Exception:                                                    # no mlx here: server.py only needs the name `sampling` imports
    m = types.ModuleType("mlx")
    c = types.ModuleType("mlx.core")
    c.array = object
    m.core = c
    sys.modules.setdefault("mlx", m)
    sys.modules.setdefault("mlx.core", c)

from kyojin_mlx.serve import hostloop as HL                          # noqa: E402
from kyojin_mlx.serve import server as SV                            # noqa: E402


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"{i};" for i in ids)


class Handle:
    def __init__(self):
        self.events = HL.new_queue()
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()


class Engine:
    ctx = 4096
    max_step_gap = 0.0

    def __init__(self, bursts):
        self.bursts, self.handles = bursts, []

    def submit(self, ids, **kw):
        h = Handle()
        self.handles.append(h)
        threading.Thread(target=self._feed, args=(h,), daemon=True).start()
        return h

    def _feed(self, h):
        n = 0
        for b in self.bursts:                                        # one burst = the tokens of one round, put back to back
            for _ in range(b):
                n += 1
                h.events.put(("tok", n))
            import time
            time.sleep(0.01)
        h.events.put(("done", "length", {"prompt_tokens": 3, "completion_tokens": n, "cached_tokens": 0, "queue_s": 0.0, "ttft_s": 0.0, "total_s": 0.1}))


def serve(bursts):
    eng = Engine(bursts)
    app = SV.App(eng, Tok(), None, "m", {}, set())
    srv = SV.Server(("127.0.0.1", 0), SV.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, eng


def stream(srv, **body):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
    conn.request("POST", "/v1/completions", json.dumps({"prompt": [1, 2, 3], "stream": True, "stream_options": {"include_usage": True}, **body}),
                 {"Content-Type": "application/json"})
    raw = conn.getresponse().read().decode()
    frames = [f[len("data: "):] for f in raw.split("\n\n") if f.startswith("data: ")]
    return frames


def parsed(frames):
    text, finish, usage = "", None, None
    for f in frames:
        if f == "[DONE]":
            continue
        o = json.loads(f)
        if o["choices"]:
            text += o["choices"][0].get("text") or ""
            finish = o["choices"][0]["finish_reason"] or finish
        if "usage" in o:
            usage = o["usage"]
    return text, finish, usage


@pytest.fixture(autouse=True)
def flags():
    saved = (HL.COALESCE, HL.NODELAY)
    HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False
    yield
    HL.COALESCE, HL.NODELAY = saved


BURSTS = [1, 4, 3, 5, 2, 4, 4, 1, 3, 5]


@pytest.mark.parametrize("nodelay", [False, True])
def test_coalesced_stream_has_the_same_text_usage_and_frames(nodelay):
    out = {}
    for on in (False, True):
        HL.COALESCE, HL.NODELAY = on, nodelay
        srv, _ = serve(BURSTS)
        try:
            out[on] = stream(srv)
        finally:
            srv.shutdown()
    t0, f0, u0 = parsed(out[False])
    t1, f1, u1 = parsed(out[True])
    assert t0 == t1 == "".join(f"{i};" for i in range(1, sum(BURSTS) + 1))
    assert f0 == f1 == "length" and u0 == u1 and u1["completion_tokens"] == sum(BURSTS)
    assert out[False][-1] == out[True][-1] == "[DONE]"
    assert len(out[True]) == len(out[False])                         # one content frame per token (the burst is one socket write)


def test_stop_string_still_cancels_and_cuts_with_the_flag_on():
    for on in (False, True):
        HL.COALESCE = on
        srv, eng = serve([3, 4, 4, 4])
        try:
            text, finish, usage = parsed(stream(srv, stop=["5;6;"]))
        finally:
            srv.shutdown()
        assert text == "1;2;3;4;" and finish == "stop"
        assert usage["completion_tokens"] == 6                       # tokens up to the one that completed the stop string, as before
        assert eng.handles[0].cancelled.is_set()


def test_non_streaming_reply_with_the_flag_on():
    HL.COALESCE = True
    srv, _ = serve(BURSTS)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=20)
        conn.request("POST", "/v1/completions", json.dumps({"prompt": [1, 2, 3]}), {"Content-Type": "application/json"})
        o = json.loads(conn.getresponse().read())
    finally:
        srv.shutdown()
    assert o["choices"][0]["text"] == "".join(f"{i};" for i in range(1, sum(BURSTS) + 1)) and o["usage"]["completion_tokens"] == sum(BURSTS)
