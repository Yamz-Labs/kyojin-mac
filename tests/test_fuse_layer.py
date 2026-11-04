"""Compiled decode regions (kyojin_mlx.fuse_layer) and the host-side compressor rope (fuse_attn._impl_compress_part): on real DS4.1 slices (KYOJIN_SHRUNK=<dir with
ds2/>, optionally ds2e/) the compiled block gives bit-identical logits to the same model with the compile switch off, for a prompt long enough that the pool holds rows
and over enough tokens to close several compression groups; the compiled path must not add launches, and a decode step must not read anything back from the GPU."""
import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
SHRUNK = os.environ.get("KYOJIN_SHRUNK")
pytestmark = [pytest.mark.skipif(not SHRUNK, reason="KYOJIN_SHRUNK not set"), pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")]


def load(name):
    from kyojin_mlx import patch as P, rows as R
    pack = os.path.join(SHRUNK, name)
    if not os.path.isdir(pack):
        pytest.skip(f"{pack} missing")
    model, spec, cfg = P.build_model(pack)
    P.patch_model(model, pack, spec, cfg)
    R.enable(model, spec)
    return model, cfg


@pytest.mark.parametrize("switch", ["fuse_layer", "fuse_index"])
@pytest.mark.parametrize("name", ["ds2", "ds2e"])
def test_compiled_block_bit_identical(name, switch):
    """switch fuse_layer: compiled regions vs eager; switch fuse_index: the trimmed + compiled indexer vs the stock `Indexer.__call__` (same logits to the bit)."""
    import importlib
    from kyojin_mlx import rows as R
    FL = importlib.import_module("kyojin_mlx." + switch)
    model, cfg = load(name)
    V = cfg["text_config"]["vocab_size"]
    ids = np.random.default_rng(5).integers(16, V - 16, 150).tolist()
    cache0, lg = R.prefill(model, ids)
    tok0 = int(mx.argmax(lg[0]).item())
    runs = {}
    try:
        for on in (False, True):
            FL.ENABLED = on
            c, tok, out = R.fork(cache0), tok0, []
            for _ in range(9):                                                     # 2 stock warm steps' worth + 7 compiled ones, 4 groups close
                lg = R.step(model, c, mx.array([tok]))
                mx.eval(lg)
                out.append(np.array(lg))
                tok = int(np.argmax(out[-1]))
            runs[on] = out
    finally:
        FL.ENABLED = True
    for a, b in zip(runs[False], runs[True]):
        assert np.array_equal(a, b)


def test_decode_step_has_no_host_sync_and_compiled_blocks_replay():
    """The step's token depends on a long GPU job; building the step must not wait for it (a `.item()` anywhere in the decode path would)."""
    import time
    from kyojin_mlx import rows as R
    model, cfg = load("ds2")
    V = cfg["text_config"]["vocab_size"]
    ids = np.random.default_rng(6).integers(16, V - 16, 150).tolist()
    cache, lg = R.prefill(model, ids)
    tok = mx.argmax(lg, axis=-1)
    for _ in range(4):                                                             # warm steps (stock), then compiled ones
        lg = R.step(model, cache, tok)
        tok = mx.argmax(lg, axis=-1)
        mx.eval(tok)
    a = mx.random.normal((3072, 3072))
    mx.eval(a)
    builds = []
    for _ in range(4):                                                             # both parities of the open compression group
        b = a
        for _ in range(24):
            b = (b @ a) * 1e-3
        mx.async_eval(b)
        gated = tok + (b.sum() > 3e38).astype(tok.dtype)                           # the token waits for the job
        t0 = time.perf_counter()
        lg = R.step(model, cache, gated)
        builds.append(time.perf_counter() - t0)
        tok = mx.argmax(lg, axis=-1)
        t1 = time.perf_counter()
        mx.eval(tok)
        job = time.perf_counter() - t1
    assert max(builds) < 0.5 * max(job, 0.05), (builds, job)
    assert any(getattr(l, "_kfl", None) is not None and len(l._kfl.fns) == 2 for l in model.language_model.layers)


@pytest.mark.parametrize("name", ["ds2", "ds2e"])
def test_fp8_fold_into_rot_bit_identical(name):
    """The fp8 activation round trip folded into the rot of the consuming linear (kyojin_mlx.fqfuse) gives the same logits to the bit as the separate fake-quant launch,
    in the eager path (a traced compiled region keeps the choice of its trace; tools/dump_logits.py checks the compiled path across processes)."""
    from kyojin_mlx import fqfuse, fuse_layer as FL, rows as R
    model, cfg = load(name)
    V = cfg["text_config"]["vocab_size"]
    ids = np.random.default_rng(6).integers(16, V - 16, 150).tolist()
    cache0, lg = R.prefill(model, ids)
    tok0 = int(mx.argmax(lg[0]).item())
    runs = {}
    try:
        FL.ENABLED = False
        for on in (False, True):
            fqfuse.ENABLED = on
            c, tok, out = R.fork(cache0), tok0, []
            for _ in range(7):
                lg = R.step(model, c, mx.array([tok]))
                mx.eval(lg)
                out.append(np.array(lg))
                tok = int(np.argmax(out[-1]))
            runs[on] = out
    finally:
        FL.ENABLED, fqfuse.ENABLED = True, True
    for a, b in zip(runs[False], runs[True]):
        assert np.array_equal(a, b)
