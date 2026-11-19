"""Hy4-flash delivery A, items 3 + 7a (CPU only): the synthetic 2-layer Hy4 GGUF of tests/test_skeleton_hy4_e2e.py (plus a small tokenizer) through the
real server in process (build_app + make_handler on a random loopback port, the engine loop on its own thread).
  routing: the modeldesc row of hy_v4_flash (keyed by the serve config's gguf_architecture) picks Hy4Backend, only with KYOJIN_SKELETON and
           KYOJIN_KIT_UNPROVEN_ROWS on; flags off = refused.
  serving: /v1/chat/completions greedy = a full-recompute greedy loop on the skeleton Model (no cache, fresh Engram history each call); a second live
           session is refused (503, the delivery-B text), never queued; a cancelled session frees the slot; the GGUF eos ends the reply."""
import importlib.util
import json
import os
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

HERE = os.path.dirname(__file__)
HY4_ONE_SESSION = "Hy4: one session only (Engram n-gram history is per model until delivery B)"
TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
PROMPT = "ab ab 1234!"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E2E = _load("hy4_e2e_helpers_b", "test_skeleton_hy4_e2e.py")
TOKT = _load("hy4_tok_helpers_b", "test_hy4_serve_tokenizer.py")


@pytest.fixture(scope="module")
def flags():
    keys = ("KYOJIN_SKELETON", "KYOJIN_KIT_UNPROVEN_ROWS", "KYOJIN_TQ", "KYOJIN_WIRED_MB")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({"KYOJIN_SKELETON": "1", "KYOJIN_KIT_UNPROVEN_ROWS": "1", "KYOJIN_TQ": "0",     # KYOJIN_TQ=0: what main() sets for the row's kv="fp16"
                       "KYOJIN_WIRED_MB": "0"})                                                      # CPU device: no residency set
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _write(path, eos=None):
    rng = np.random.default_rng(7)
    t = E2E._model_tensors(rng)
    table = rng.integers(0, 256, (sum(E2E.PRIMES), E2E.HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38
    kv = {**E2E._kv(), **TOKT._tok_kv(E2E.V), "tokenizer.chat_template": TEMPLATE}
    if eos is not None:
        kv["tokenizer.ggml.eos_token_id"] = int(eos)
    out = [(E2E.EG.TABLE, (E2E.HD, table.shape[0]), "F8_E4M3", table)] + [(n, a.shape[::-1], "F32", a.astype(np.float32)) for n, a in t.items()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    E2E.SY.write_gguf(path, kv, out)
    return path


def _reference(gguf, ids, n, stops):
    """Greedy on the skeleton Model, the whole sequence recomputed each step (offset 0: the Engram history restarts): no cache, no backend."""
    from kyojin_mlx import foreign_skeleton as SK
    model, _ = SK.load_cpu(gguf, row=E2E.R.ROWS[E2E.A])
    seq, out = list(ids), []
    for _ in range(n):
        t = int(mx.argmax(model(mx.array([seq]))[0, -1]).item())
        if t in stops:
            break
        out.append(t)
        seq.append(t)
    return out


class Lane:
    """The real server in process: build_app on this thread, the engine loop on its own (started by start())."""

    def __init__(self, gguf, cache):
        from kyojin_mlx.serve import server as SV
        from kyojin_mlx.foreign_names import gguf_serve_dir
        self.SV = SV
        d = gguf_serve_dir(os.path.dirname(gguf), cache=cache)
        args = SV.build_parser().parse_args([d, "--ctx", "64", "--prefill-chunk", "16", "--no-kv-disk", "--model-id", "Hy4-flash-Kyojin",
                                             "--kv-mode", "fp16", "--sessions", "4", "--prefix-cache-mb", "64"])
        self.log = []
        info = mx.device_info                                                  # CPU device: build_app / memfit read the GPU working-set keys
        mx.device_info = lambda: {"memory_size": 64 * 2 ** 30, "max_recommended_working_set_size": 48 * 2 ** 30, **info()}
        try:
            self.app = SV.build_app(args, log=lambda *a: self.log.append(" ".join(map(str, a))))
        finally:
            mx.device_info = info
        self.args = args
        self.srv = SV.Server(("127.0.0.1", 0), SV.make_handler(self.app))
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.th = None

    def start(self):
        if self.th is None:
            self.th = threading.Thread(target=self.app.engine.run, daemon=True)
            self.th.start()

    def close(self):
        if self.th is not None:
            self.app.engine.stop()
            self.th.join(10)
        self.srv.shutdown()
        self.srv.server_close()

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def prompt_ids(self):
        code, r = self.post("/v1/tokenize", {"messages": [{"role": "user", "content": PROMPT}]})
        assert code == 200, r
        return r["tokens"]


def _collect(h, timeout=60):
    toks = []
    while True:
        ev = h.events.get(timeout=timeout)
        if ev[0] == "tok":
            toks.append(ev[1])
        else:
            return toks, ev[1]


@pytest.fixture(scope="module")
def lane(flags, tmp_path_factory):
    root = tmp_path_factory.mktemp("hy4b")
    g = _write(str(root / "model" / "hy4_2l.gguf"))
    ln = Lane(g, str(root / "cache"))
    yield ln, g
    ln.close()


# ---------------------------------------------------------------------------------------------------------------------------- routing (item 3)
def test_row_keyed_by_gguf_architecture(flags):
    from kyojin_mlx import modeldesc
    cfg = {"model_type": "kyojin_skeleton", "gguf_architecture": "hy_v4_flash"}
    assert modeldesc.model_key(cfg) == "hy_v4_flash" and modeldesc.model_key({"model_type": "qwen3"}) == "qwen3"
    row = modeldesc.backend_desc(modeldesc.model_key(cfg))
    assert row is not None and row.backend == "hy4" and row.kv == "fp16"
    assert modeldesc.backend_desc("kyojin_skeleton") is None                 # not every skeleton architecture: the GGUF architecture names the row


def test_flags_off_refused(flags, tmp_path, monkeypatch):
    from kyojin_mlx import modeldesc
    from kyojin_mlx.serve import backends as B
    monkeypatch.delenv("KYOJIN_KIT_UNPROVEN_ROWS", raising=False)
    assert modeldesc.backend_desc("hy_v4_flash") is None
    d = tmp_path / "dir"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "kyojin_skeleton", "gguf_architecture": "hy_v4_flash"}))
    with pytest.raises(ValueError, match="KYOJIN_KIT_UNPROVEN_ROWS"):
        B.load_backend(str(d), log=lambda *a: None)
    monkeypatch.setenv("KYOJIN_KIT_UNPROVEN_ROWS", "1")                    # the row on, the skeleton off: refused by the loader
    monkeypatch.delenv("KYOJIN_SKELETON", raising=False)
    with pytest.raises(ValueError, match="KYOJIN_SKELETON"):
        B.load_backend(str(d), log=lambda *a: None)


def test_load_backend_picks_hy4(lane):
    ln, _ = lane
    be = ln.app.engine.backend
    assert be.kind == "hy4" and not be.one_session and be.stop_ids == {12}
    assert ln.args.sessions == 1                                              # one decoding session at a time (max_rows 1); further requests queue
    assert 12 in ln.app.stop_ids and ln.app.model_id == "Hy4-flash-Kyojin"


# ----------------------------------------------------------------------------------------------------------------------------- serving (7a)
def test_second_session_refused_then_slot_freed(lane, monkeypatch):
    """KYOJIN_HY4_ONE_SESSION=1 (delivery A): a second live session is a 503. Default: it queues (next test)."""
    ln, g = lane
    monkeypatch.setattr(ln.app.engine.backend, "one_session", HY4_ONE_SESSION, raising=False)
    ids = ln.prompt_ids()
    h = ln.app.engine.submit(ids, max_new=4)                                  # engine not running yet: this session stays live
    code, r = ln.post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "temperature": 0})
    assert code == 503 and r["error"]["message"] == "Hy4: one session only (Engram n-gram history is per model until delivery B)"
    ln.start()
    toks, reason = _collect(h)
    assert reason in ("length", "stop") and toks == _reference(g, ids, 4, {12})
    code, r = ln.post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4, "temperature": 0})
    assert code == 200, r                                                     # the first one finished: the slot is free again


def test_chat_greedy_equals_model(lane):
    ln, g = lane
    ln.start()
    ids = ln.prompt_ids()
    want = _reference(g, ids, 12, {12})
    code, r = ln.post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 12, "temperature": 0})
    assert code == 200, r
    tok = ln.app.tokenizer
    assert r["usage"]["completion_tokens"] == len(want)
    assert r["choices"][0]["message"]["content"] == tok.decode(want, skip_special_tokens=False)
    toks, _ = _collect(ln.app.engine.submit(ids, max_new=12))                # the same engine path, ids exact
    assert toks == want
    assert ln.app.engine.stats["finished_error"] == 0


def test_cancelled_session_frees_the_slot(lane):
    ln, g = lane
    ln.start()
    ids = ln.prompt_ids()
    h = ln.app.engine.submit(ids, max_new=40)
    h.cancel()
    toks, reason = _collect(ln.app.engine.submit(ids, max_new=5))            # waits for the cancelled one to end, then runs from a fresh history
    assert toks == _reference(g, ids, 5, {12})


def test_stop_id_honoured(flags, tmp_path):
    """The GGUF eos (Hy4-flash: 120025) ends the reply: a copy of the model whose eos is the 3rd greedy token stops after 2 tokens."""
    g0 = _write(str(tmp_path / "m0" / "hy4_2l.gguf"))
    ln0 = Lane(g0, str(tmp_path / "c0"))
    ids = ln0.prompt_ids()
    ln0.close()
    ref = _reference(g0, ids, 12, set())
    k = next(i for i in range(2, len(ref)) if ref[i] not in ref[:i])
    g = _write(str(tmp_path / "m1" / "hy4_2l.gguf"), eos=ref[k])
    ln = Lane(g, str(tmp_path / "c1"))
    try:
        assert ln.app.engine.backend.stop_ids == {ref[k]} and ref[k] in ln.app.stop_ids
        ln.start()
        code, r = ln.post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 12, "temperature": 0})
        assert code == 200, r
        assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == k
    finally:
        ln.close()


def test_sessions_own_their_engram_history(lane):
    """Interleaved sessions (each forward claims the model's history for its cache, parking the other's) equal the same sessions run alone; a fork continues like its source."""
    from kyojin_mlx.foreign_skeleton import SkeletonError
    ln, _ = lane
    be = ln.app.engine.backend
    assert not be.one_session

    def run(ids, steps, order=None):
        c = be.make_cache()
        lg = be.prefill_chunk(c, ids)
        out = [mx.argmax(lg, axis=-1).item()]
        return c, out

    a, ao = run([1, 2, 3], 0)
    b, bo = run([4, 5, 6, 7], 0)                                              # b took over: a's history is parked on its cache
    la = be.step(be.batch([a]), mx.array([ao[0]]))                           # a takes it back
    lb = be.step(be.batch([b]), mx.array([bo[0]]))
    a2, a2o = run([1, 2, 3], 0)
    b2, b2o = run([4, 5, 6, 7], 0)
    la2 = be.step(be.batch([a2]), mx.array([a2o[0]]))
    lb2 = be.step(be.batch([b2]), mx.array([b2o[0]]))
    assert float(mx.max(mx.abs(la - la2)).item()) == 0.0 and float(mx.max(mx.abs(lb - lb2)).item()) == 0.0
    snap = be.fork(a)                                                         # snapshot at a's position (after its decode step)
    nxt = mx.argmax(la, axis=-1)
    la3 = be.step(be.batch([a]), nxt)
    lf = be.step(be.batch([be.restore(snap)]), nxt)
    assert float(mx.max(mx.abs(la3 - lf)).item()) == 0.0
    lr = be.step(be.batch([be.restore(snap)]), nxt)                           # the snapshot is not consumed by a restore: a second one continues the same way
    assert float(mx.max(mx.abs(lr - lf)).item()) == 0.0
    with pytest.raises(ValueError, match="one session only"):
        be.batch([a, b])
    c = be.make_cache()
    be.prefill_chunk(c, [1, 2])
    c.hist = None
    be._owner = None
    d = be.make_cache()
    be.prefill_chunk(d, [3, 4])
    with pytest.raises(SkeletonError, match="no Engram history"):
        be.step(be.batch([c]), mx.array([5]))


def test_queued_second_request_runs_after_the_first(lane):
    """Default (no one-session gate): two requests at once both answer; the second waits for the slot, each from its own fresh history."""
    ln, g = lane
    ln.start()
    ids = ln.prompt_ids()
    h1 = ln.app.engine.submit(ids, max_new=6)
    h2 = ln.app.engine.submit(ids, max_new=6)
    t1, _ = _collect(h1)
    t2, _ = _collect(h2)
    assert t1 == t2 == _reference(g, ids, 6, {12})


def test_warm_main_leaves_a_clean_slot(lane):
    """The main-thread warm-up (run at load when step compile is on) is one throw-away session: it decodes, logs, and the next real session starts at offset 0 as owner."""
    ln, _ = lane
    be = ln.app.engine.backend
    lines = []
    dt = be.warm_main(lambda *a: lines.append(" ".join(str(x) for x in a)), n=5, steps=2)
    assert dt > 0 and any("main-thread warm-up 5 prompt rows + 2 steps" in l for l in lines)
    assert be._owner is None
    c = be.make_cache()
    be.prefill_chunk(c, [1, 2, 3])                                            # a new session claims the history without error
    be.step(be.batch([c]), mx.array([4]))
