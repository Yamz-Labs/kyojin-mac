"""Runtime decensor hook (kyojin_mlx.ablit) on a tiny random DS4.1 stack (stock mlx-vlm modules, fp32): the hook equals the weight edit
W_out <- (I - w r r^T) W_out of the writers (wo_b, shared down_proj, every routed down_proj) on the eager decode path, the prompt path and the CED prefill;
the spec switch off is bit-identical to no hook; blocks that are not armed (draft head, layers >= n_layers, weight 0) are never touched."""
import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ablit as AB, ced as C  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402

LANG = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
STOCK_CALL = LANG.DeepseekV41Block.__dict__["__call__"]
N = M.CFG["num_hidden_layers"]
H = M.CFG["hidden_size"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """CPU stream: the GPU matmul of the reference and of the hook round differently enough to flip the router's top-k of a random model."""
    monkeypatch.delenv(AB.ENV, raising=False)
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)
    LANG.DeepseekV41Block.__call__ = STOCK_CALL
    AB._installed = False


def write_spec(tmp_path, k=1, per_layer=False, seed=3, scale=1.0, n_layers=N, zero_layers=()):
    rng = np.random.default_rng(seed)
    shape = (N, k, H) if per_layer else ((k, H) if k > 1 else (H,))
    R = rng.standard_normal(shape).astype(np.float32)
    if k > 1:                                                     # orthonormal rows
        R = np.linalg.qr(np.swapaxes(R, -1, -2))[0].swapaxes(-1, -2).astype(np.float32)
    from safetensors.numpy import save_file
    wa = [0.0 if i in zero_layers else float(scale * (0.4 + 0.5 * rng.random())) for i in range(N)]
    wm = [0.0 if i in zero_layers else float(scale * (0.3 + 0.6 * rng.random())) for i in range(N)]
    p = tmp_path / "edit_spec.json"
    json.dump({"hidden": H, "n_layers": n_layers, "attn_w": wa, "mlp_w": wm}, open(p, "w"))
    save_file({"R" if (per_layer or k > 1) else "r": R}, str(tmp_path / "edit_spec.safetensors"))
    return str(p)


def weight_edit(lm, path):
    """The reference: edit the writers of every armed layer (fp32 numpy-free math in mx)."""
    s = AB.load(path)
    R_all = mx.array(s.R)
    for L, blk in enumerate(lm.layers):
        if L >= s.n_layers:
            break
        R = R_all[L] if R_all.ndim == 3 else R_all                                    # [K, H]
        P = R.T @ R                                                                   # [H, H]

        def edit(W, w):                                                                # W [..., H(out), in]
            return W - w * mx.matmul(P, W)
        a, f = blk.attn.wo_b, blk.ffn
        a.weight = edit(a.weight, s.attn_w[L])
        f.shared_experts.down_proj.weight = edit(f.shared_experts.down_proj.weight, s.mlp_w[L])
        f.switch_mlp.down_proj.weight = edit(f.switch_mlp.down_proj.weight, s.mlp_w[L])
    mx.eval(lm.parameters())


def run(lm, ids, steps=6, chunks=None, ced=None):
    cache = lm.make_cache()
    out, s, logs = None, 0, []
    for c in (chunks or (len(ids),)):
        out = lm(mx.array([ids[s:s + c]]), cache=cache).logits
        mx.eval(out)
        s += c
    logs.append(out[:, -1].astype(mx.float32))
    tok = int(mx.argmax(logs[-1]).item())
    for _ in range(steps):
        out = lm(mx.array([[tok]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(out)
        logs.append(out)
        tok = int(mx.argmax(out).item())
    return logs


def close(a, b, tol=1e-5):
    a, b = np.array(a), np.array(b)
    return float(np.abs(a - b).max() / np.abs(b).max()) <= tol


def armed(tmp_path, **kw):
    path = write_spec(tmp_path, **kw)
    AB.install_block()
    lm, _ = M.make()
    assert AB.attach(lm, path) > 0
    return lm, path


@pytest.mark.parametrize("kw", [dict(), dict(k=3), dict(k=2, per_layer=True)], ids=["r", "R3", "R2-per-layer"])
def test_hook_equals_weight_edit_eager(tmp_path, kw):
    ids = M.tokens(40)
    lm, path = armed(tmp_path, **kw)
    got = run(lm, ids)
    ref_lm, _ = M.make()
    weight_edit(ref_lm, path)
    ref = run(ref_lm, ids)
    base_lm, _ = M.make()
    base = run(base_lm, ids)
    assert not close(base[0], ref[0], 1e-3), "the edit must change the logits (test is not vacuous)"
    for g, r in zip(got, ref):
        assert close(g, r)


def test_hook_equals_weight_edit_chunked_prompt_and_batch(tmp_path):
    ids = M.tokens(48)
    lm, path = armed(tmp_path, k=2)
    ref_lm, _ = M.make()
    weight_edit(ref_lm, path)
    for g, r in zip(run(lm, ids, chunks=(16, 16, 16)), run(ref_lm, ids, chunks=(16, 16, 16))):
        assert close(g, r)


def test_hook_equals_weight_edit_ced_prefill(tmp_path):
    ids = M.tokens(64)
    lm, path = armed(tmp_path)
    C.enable(lm, DS41, tail=24)
    got = run(lm, ids, chunks=(32, 32), steps=0)
    ref_lm, _ = M.make()
    weight_edit(ref_lm, path)
    C.enable(ref_lm, DS41, tail=24)
    ref = run(ref_lm, ids, chunks=(32, 32), steps=0)
    assert close(got[0], ref[0])
    assert C.state(lm).stats["rows_cut"] > 0                       # the CED path really cut rows
    base_lm, _ = M.make()
    C.enable(base_lm, DS41, tail=24)
    assert not close(run(base_lm, ids, chunks=(32, 32), steps=0)[0], ref[0], 1e-3)


def test_off_is_bit_identical(tmp_path):
    ids = M.tokens(40)
    lm0, _ = M.make()
    ref = run(lm0, ids)                                           # stock class code
    AB.install_block()
    lm1, _ = M.make()                                             # block replaced, nothing armed
    assert AB.attach(lm1) == 0                                    # env unset: no spec, off
    got = run(lm1, ids)
    lm2, _ = M.make()
    path = write_spec(tmp_path, scale=0.0)                        # spec present but every weight 0
    assert AB.attach(lm2, path) == 0
    got2 = run(lm2, ids)
    for a, b, c in zip(ref, got, got2):
        assert np.array_equal(np.array(a), np.array(b)) and np.array_equal(np.array(a), np.array(c))


def test_unarmed_blocks_untouched(tmp_path):
    path = write_spec(tmp_path, n_layers=5, zero_layers=(1,))
    AB.install_block()
    lm, _ = M.make()
    assert AB.attach(lm, path) == 4
    assert [("_ablit" in blk.__dict__) for blk in lm.layers] == [True, False, True, True, True, False, False, False]
    ids = M.tokens(30)
    got = run(lm, ids)
    ref_lm, _ = M.make()
    weight_edit_partial = AB.load(path)
    weight_edit(ref_lm, path)                                      # layers >= 5 and the zero-weight layer: weight 0 / skipped by n_layers
    assert close(got[0], run(ref_lm, ids)[0])


def test_env_switch(tmp_path, monkeypatch):
    path = write_spec(tmp_path)
    lm, _ = M.make()
    assert AB.attach(lm) == 0
    monkeypatch.setenv(AB.ENV, "off")
    assert AB.spec_path() is None and AB.attach(lm) == 0
    monkeypatch.setenv(AB.ENV, path)
    assert AB.spec_path() == path and AB.attach(lm) == N


def test_bad_spec_is_refused(tmp_path):
    path = write_spec(tmp_path)
    j = json.load(open(path))
    j["hidden"] = H + 1
    json.dump(j, open(path, "w"))
    with pytest.raises(ValueError):
        AB.load(path)
    with pytest.raises(FileNotFoundError):
        AB.load(str(tmp_path / "nope.json"))


def test_signature_off_and_content(tmp_path, monkeypatch):
    import json as _j
    from safetensors.numpy import save_file
    monkeypatch.delenv(AB.ENV, raising=False)
    assert AB.signature() is None
    p = tmp_path / "s.json"
    _j.dump({"hidden": 4, "n_layers": 1, "attn_w": [1.0], "mlp_w": [0.0]}, open(p, "w"))
    save_file({"r": np.ones(4, np.float32)}, str(tmp_path / "s.safetensors"))
    monkeypatch.setenv(AB.ENV, str(p))
    a = AB.signature()
    save_file({"r": np.arange(4, dtype=np.float32)}, str(tmp_path / "s.safetensors"))
    assert a is not None and AB.signature() != a
