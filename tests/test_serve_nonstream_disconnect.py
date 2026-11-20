"""A non-stream request stops generating when its client leaves (CPU, no Metal): real loopback server, fake engine."""
import http.client
import json
import socket
import threading
import time
import types

import pytest

from kyojin_mlx.serve import hostloop as HL
from kyojin_mlx.serve import server as SV


class Handle:
    def __init__(self):
        self.events = HL.new_queue()
        self.cancelled = threading.Event()
        self.session = None

    def cancel(self):
        self.cancelled.set()


class Engine:
    """submit() returns a handle; `tokens_per_s` > 0 feeds tokens until cancelled, 0 = a long prefill / slot wait (no event at all)."""
    ctx = 100000
    max_step_gap = 0.0

    def __init__(self, tokens_per_s):
        self.rate, self.handle = tokens_per_s, None

    def submit(self, ids, **kw):
        self.handle = h = Handle()
        if self.rate:
            threading.Thread(target=self._feed, args=(h,), daemon=True).start()
        return h

    def _feed(self, h):
        while not h.cancelled.is_set():
            h.events.put(("tok", 65))
            time.sleep(1.0 / self.rate)
        h.events.put(("done", "cancelled", {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0, "queue_s": 0.0, "ttft_s": 0.0, "total_s": 0.1}))


class Tok:
    def encode(self, text, **kw):
        return [1, 2, 3]

    def decode(self, ids, **kw):
        return "A" * len(ids)


class Parser:
    mode = "content"

    def feed(self, text):
        return [("content", text)] if text else []

    def finish(self):
        return []


class Family:
    name = "fake"

    def render(self, body):
        return types.SimpleNamespace(prompt="hi", images=None, cut_texts=[], thinking_first=False, parse_tools=False, tools=None)

    def parser(self, r):
        return Parser()


def serve(engine):
    app = SV.App(engine, Tok(), Family(), "m", {}, set(), log=lambda *a: None)
    srv = SV.Server(("127.0.0.1", 0), SV.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def send_and_leave(srv, **body):
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
    data = json.dumps({"messages": [{"role": "user", "content": "hi"}], **body}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(data) + data)
    return s


@pytest.fixture(autouse=True)
def fast_watch(monkeypatch):
    monkeypatch.setattr(HL, "DISCONNECT_PERIOD_S", 0.05)
    HL.VERIFY_LEFT, HL.BATCH_BAD = 64, False


@pytest.mark.parametrize("rate", [0, 200])               # 0: waiting for a slot or in prefill; 200: decoding
@pytest.mark.parametrize("coalesce", [False, True])
def test_client_leaving_cancels_a_non_stream_request(rate, coalesce):
    HL.COALESCE = coalesce
    eng = Engine(rate)
    srv = serve(eng)
    try:
        s = send_and_leave(srv)
        t0 = time.time()
        while eng.handle is None and time.time() - t0 < 5:
            time.sleep(0.01)
        assert eng.handle is not None and not eng.handle.cancelled.is_set()
        time.sleep(0.15)
        assert not eng.handle.cancelled.is_set()          # the client is still there
        s.close()
        assert eng.handle.cancelled.wait(5)
    finally:
        srv.shutdown()


def test_a_client_that_stays_gets_its_reply():
    eng = Engine(0)
    srv = serve(eng)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}), {"Content-Type": "application/json"})
        t0 = time.time()
        while eng.handle is None and time.time() - t0 < 5:
            time.sleep(0.01)
        time.sleep(0.2)                                    # several watch ticks pass with the client connected
        assert not eng.handle.cancelled.is_set()
        eng.handle.events.put(("tok", 65))
        eng.handle.events.put(("done", "stop", {"prompt_tokens": 3, "completion_tokens": 1, "cached_tokens": 0, "queue_s": 0.0, "ttft_s": 0.0, "total_s": 0.1}))
        out = json.loads(conn.getresponse().read().decode())
        assert out["choices"][0]["message"]["content"] == "A"
    finally:
        srv.shutdown()


def test_watch_is_rate_limited_and_sees_eof():
    a, b = socket.socketpair()
    now = [0.0]
    w = HL.ClientWatch(a, period=1.0, clock=lambda: now[0])
    assert not w.gone()
    b.close()
    assert not w.gone()                                    # inside the period: no syscall
    now[0] = 1.5
    assert w.gone()
    a.close()


def test_client_gone_ignores_pipelined_data():
    a, b = socket.socketpair()
    assert not HL.client_gone(a)
    b.sendall(b"GET / HTTP/1.1\r\n\r\n")
    assert not HL.client_gone(a)
    b.close()
    assert not HL.client_gone(a)                           # data still unread: not a hang-up yet
    a.recv(100)
    assert HL.client_gone(a)
    a.close()
