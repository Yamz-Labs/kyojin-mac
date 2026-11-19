"""Hy4 MTP through the real engine (CPU, tiny synthetic Hy4 GGUF with a next-n block, in-process server): greedy tokens with KYOJIN_HY4_MTP=1 and the
engine's speculative rounds (mtp_depth 3) equal the full-recompute greedy reference; the engine really ran verify rounds."""
import importlib.util
import os

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)
HERE = os.path.dirname(__file__)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SB = _load("hy4_serve_backend_helpers_m", "test_hy4_serve_backend.py")
MT = _load("hy4_mtp_helpers_m", "test_hy4_mtp.py")
E2E = SB.E2E


def _write_mtp(path):
    rng = np.random.default_rng(11)
    t = E2E._model_tensors(rng)
    t.update(MT._head_tensors(rng, t))
    table = rng.integers(0, 256, (sum(E2E.PRIMES), E2E.HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38
    kv = {**MT._kv_mtp(), **SB.TOKT._tok_kv(E2E.V), "tokenizer.chat_template": SB.TEMPLATE}
    out = [(E2E.EG.TABLE, (E2E.HD, table.shape[0]), "F8_E4M3", table)] + [(n, a.shape[::-1], "F32", a.astype(np.float32)) for n, a in t.items()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    E2E.SY.write_gguf(path, kv, out)
    return path


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    keys = ("KYOJIN_SKELETON", "KYOJIN_KIT_UNPROVEN_ROWS", "KYOJIN_TQ", "KYOJIN_WIRED_MB", "KYOJIN_HY4_MTP")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({"KYOJIN_SKELETON": "1", "KYOJIN_KIT_UNPROVEN_ROWS": "1", "KYOJIN_TQ": "0", "KYOJIN_WIRED_MB": "0", "KYOJIN_HY4_MTP": "1"})
    root = tmp_path_factory.mktemp("hy4mtp")
    g = _write_mtp(str(root / "model" / "hy4_mtp.gguf"))
    ln = SB.Lane(g, str(root / "cache"))
    yield ln, g
    ln.close()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_engine_rounds_equal_plain_greedy(lane):
    ln, g = lane
    be = ln.app.backend if hasattr(ln.app, "backend") else ln.app.engine.backend
    assert be.supports_spec and ln.app.engine.mtp_depth > 0
    ln.start()
    ids = ln.prompt_ids()
    want = SB._reference(g, ids, 24, set())
    toks, _ = SB._collect(ln.app.engine.submit(ids, max_new=24))
    assert toks == want
    st = ln.app.engine.stats
    assert st.get("finished_error", 0) == 0
    from kyojin_mlx import hy4_mtp
    assert hy4_mtp.STATS["rounds"] > 0 and hy4_mtp.STATS["drafted"] > 0


def _driver_loop(model, ids, n, ub=512):
    """The benchmark cells' plain loop (c94 ON arm): model() on a KVCache list, prefill in chunks, argmax per step."""
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in model.layers]
    for a in range(0, len(ids), ub):
        lg = model(mx.array([ids[a:a + ub]]), cache=cache)
    tok = int(mx.argmax(lg[0, -1]).item())
    out = [tok]
    for _ in range(n - 1):
        tok = int(mx.argmax(model(mx.array([[tok]]), cache=cache)[0, -1]).item())
        out.append(tok)
    return out


def test_driver_loop_equals_engine_paths(lane):
    """The cell driver (direct model loop) must give the engine's own greedy ids: plain loop == engine with MTP rounds == engine with the exact commit."""
    from kyojin_mlx import hy4_mtp
    ln, g = lane
    be = ln.app.backend if hasattr(ln.app, "backend") else ln.app.engine.backend
    ln.start()
    ids = ln.prompt_ids()
    direct = _driver_loop(be.model, ids, 24)
    assert direct == SB._reference(g, ids, 24, set())
    toks, _ = SB._collect(ln.app.engine.submit(ids, max_new=24))
    assert toks == direct
    old, be.exact = be.exact, True
    try:
        toks, _ = SB._collect(ln.app.engine.submit(ids, max_new=24))
    finally:
        be.exact = old
    assert toks == direct
    # teacher forcing reproduces the free run when the "oracle" is the model's own ids (what the pre-flight relies on)
    from mlx_lm.models.cache import KVCache
    cache = [KVCache() for _ in be.model.layers]
    lg = be.model(mx.array([ids]), cache=cache)
    tf = [int(mx.argmax(lg[0, -1]).item())]
    for j in range(23):
        tf.append(int(mx.argmax(be.model(mx.array([[direct[j]]]), cache=cache)[0, -1]).item()))
    assert tf == direct


def test_cell_engine_construction(lane):
    """The c104 cell builds its own Engine on the loaded model (not through build_app): same ids as the lane's engine, rounds really ran."""
    import threading
    from kyojin_mlx import hy4_mtp
    from kyojin_mlx.serve import engine as EN
    ln, g = lane
    old_eng = ln.app.engine
    be = old_eng.backend
    ids = ln.prompt_ids()
    want = _driver_loop(be.model, ids, 24)
    old_eng.stop()
    if ln.th is not None:
        ln.th.join(10)
        ln.th = None
    eng = EN.Engine(be, sessions=1, ctx=8192, chunk=512, block=512, prefix_bytes=0, mtp_depth=3, log=lambda *a: None)
    th = threading.Thread(target=eng.run, daemon=True)
    th.start()
    try:
        r0 = hy4_mtp.STATS["rounds"]
        toks, _ = SB._collect(eng.submit(ids, max_new=24))
        assert toks == want
        assert hy4_mtp.STATS["rounds"] > r0
    finally:
        eng.stop()
        th.join(10)
