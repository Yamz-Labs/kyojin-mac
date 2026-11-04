"""Runtime decensor hook (kyojin_mlx.ablit) on the engine's fast paths, on a tiny random DS4.1 stack shaped for them (head_dim 512, hc_mult 4, fp16, Metal):
compiled fused decode (fuse_layer solo B region) and the row-invariant verify (rows.verify). The compiled chained verify shares its hooked B region code with the solo path
(fuse_layer `_chain_regions`); the mini model has no EXL3 linears, which `spec.verify` needs, so a chain run waits for a real pack (step 3 compares chain on / off). The mini model is chaotic (router / top-k
flips amplify a 1e-7 rounding difference across layers), so the stack is cut after the first armed block: layers before it are untouched, hence identical, and the
logits differ only by the hook's own rounding. Each fast path is checked against the weight edit W_out <- (I - w r r^T) W_out run through the SAME path."""
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map, tree_unflatten

sys.path.insert(0, os.path.dirname(__file__))
pytest.importorskip("mlx_vlm.models.deepseek_v41")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
import mini_ds41 as M  # noqa: E402
from kyojin_mlx import ablit as AB, fuse_layer as FL, rows as R  # noqa: E402
from kyojin_mlx.archs import DS41  # noqa: E402

LANG = pytest.importorskip("mlx_vlm.models.deepseek_v41.language")
STOCK_CALL = LANG.DeepseekV41Block.__dict__["__call__"]
HID, ARMED, CUT = 128, 3, 4                       # hidden size, the one armed layer (a compressing layer: fused path), layers kept
OVER = dict(head_dim=512, hidden_size=HID, num_attention_heads=4, index_head_dim=128, q_lora_rank=64, o_lora_rank=64, hc_mult=4)


class Wrap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(AB.ENV, raising=False)
    yield
    from kyojin_mlx import fuse
    LANG.DeepseekV41Block.__call__ = STOCK_CALL                    # the block class goes back to stock: its process-wide installs are forgotten, the others stay
    AB._installed = False
    fuse._installed -= {k for k in fuse._installed if k[1] == "__call__" and k[0].endswith("DeepseekV41Block")}


def spec(tmp_path, seed=3, hid=HID):
    rng = np.random.default_rng(seed)
    R_ = rng.standard_normal((2, hid)).astype(np.float32)
    R_ = np.linalg.qr(R_.T)[0].T.astype(np.float32)
    n = M.CFG["num_hidden_layers"]
    wa = [0.0] * n
    wm = [0.0] * n
    wa[ARMED], wm[ARMED] = 0.8, 0.7
    p = tmp_path / "edit_spec.json"
    json.dump({"hidden": hid, "n_layers": n, "attn_w": wa, "mlp_w": wm}, open(p, "w"))
    from safetensors.numpy import save_file
    save_file({"R": R_}, str(tmp_path / "edit_spec.safetensors"))
    return str(p)


def make(path=None, mode="off", hid=HID):
    """mode: off | hook | edit. The model is cut after the armed block. The mHC mix parameters stay fp32 like the pack."""
    lm, _ = M.make(**{**OVER, "hidden_size": hid})
    lm.update(tree_unflatten([(k, (v * 8 if k.endswith("gate.weight") else v).astype(mx.float16) if v.dtype == mx.float32 and ".hc_" not in k else v)
                              for k, v in tree_flatten(lm.parameters())]))
    lm.layers = lm.layers[:CUT]
    mx.eval(lm.parameters())
    if mode == "hook":
        assert AB.attach(lm, path) == 1
    elif mode == "edit":
        s = AB.load(path)
        P = mx.array(s.R).T @ mx.array(s.R)
        blk = lm.layers[ARMED]
        for w, mod in ((s.attn_w[ARMED], blk.attn.wo_b), (s.mlp_w[ARMED], blk.ffn.shared_experts.down_proj), (s.mlp_w[ARMED], blk.ffn.switch_mlp.down_proj)):
            W = mod.weight.astype(mx.float32)
            mod.weight = (W - w * mx.matmul(P, W)).astype(mx.float16)
        mx.eval(lm.parameters())
    return Wrap(lm)


def err(a, b):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return float(np.abs(a - b).max() / np.abs(b).max())


def decode(m, ids, steps=8):
    lm = m.language_model
    cache = lm.make_cache()
    mx.eval(lm(mx.array([ids]), cache=cache).logits)
    out = []
    for i in range(steps):
        lg = lm(mx.array([[5 + i]]), cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg)
        out.append(lg)
    return mx.concatenate(out)


def test_fused_decode_hook_equals_weight_edit(tmp_path, monkeypatch):
    path = spec(tmp_path)
    AB.install_block()
    from kyojin_mlx import fuse
    assert fuse.install(DS41) > 0
    ids = M.tokens(150, vocab=96)
    FL.STATS.update({k: 0 for k in FL.STATS})
    got = decode(make(path, "hook"), ids)
    assert FL.STATS["call_solo"] > 0, "the compiled fused decode must run"
    ref = decode(make(path, "edit"), ids)
    base = decode(make(path, "off"), ids)
    assert err(got, ref) < 2e-2 and err(base, ref) > 5 * err(got, ref), (err(got, ref), err(base, ref))
    monkeypatch.setattr(FL, "ENABLED", False)                      # same hook, stock eager block: the compiled region agrees with it
    stock = decode(make(path, "hook"), ids)
    assert err(got, stock) < 2e-2, err(got, stock)


def test_rows_verify_hook_equals_weight_edit(tmp_path):
    path = spec(tmp_path)
    AB.install_block()
    ids, toks = M.tokens(40, vocab=96), [5, 6, 7, 8]

    def verify(mode):
        m = make(path, mode)
        R.enable(m, DS41, grouped=False)
        cache, _ = R.prefill(m, ids)
        p = R.verify(m, R.batch([cache]), [toks])
        mx.eval(p.logits)
        return p.logits

    got, ref, base = verify("hook"), verify("edit"), verify("off")
    assert err(got, ref) < 2e-2 and err(base, ref) > 5 * err(got, ref), (err(got, ref), err(base, ref))


def test_draft_head_blocks_untouched(tmp_path):
    path = spec(tmp_path)
    AB.install_block()
    m = make(path, "hook")
    assert [i for i, b in enumerate(m.language_model.layers) if getattr(b, "_ablit", None)] == [ARMED]
    draft, _ = M.make(**OVER)                                      # the draft head blocks are separate instances of the same class: never armed
    assert not any(getattr(b, "_ablit", None) for b in draft.layers)



@pytest.mark.parametrize("launchcut", [False, True])
def test_suglue_decode_hook_equals_weight_edit(tmp_path, monkeypatch, launchcut):
    """The served default decode (port_suglue stages A/B/E, optionally launchcut's stage-E combine) carries the hook too."""
    from kyojin_mlx import launchcut as LC, suglue as SG
    monkeypatch.setattr(SG, "ENABLED", True)
    monkeypatch.setattr(LC, "ENABLED", launchcut)
    path = spec(tmp_path, hid=4096)                                # suglue stages need D >= 4096
    AB.install_block()
    from kyojin_mlx import fuse
    assert fuse.install(DS41) > 0
    ids = M.tokens(150, vocab=96)
    seen = {"b": 0, "e": 0}
    sb, se = SG.stage_b, LC.ffn_stage_e
    monkeypatch.setattr(SG, "stage_b", lambda *a: (seen.__setitem__("b", seen["b"] + 1), sb(*a))[1])
    monkeypatch.setattr(LC, "ffn_stage_e", lambda *a: (seen.__setitem__("e", seen["e"] + 1), se(*a))[1])
    got = decode(make(path, "hook", 4096), ids)
    assert seen["b"] > 0 and seen["e"] > 0, seen
    base = decode(make(path, "off", 4096), ids)
    edit = decode(make(path, "edit", 4096), ids)                   # fp16 weight edit at D 4096: its own rounding, a looser bound
    assert err(got, edit) < 0.5 * err(base, edit), (err(got, edit), err(base, edit))
    monkeypatch.setattr(FL, "ENABLED", False)                      # same hook, eager suglue block (SG.block): another code path, same rounding
    ref = decode(make(path, "hook", 4096), ids)
    assert err(got, ref) < 2e-3 and err(base, ref) > 50 * err(got, ref), (err(got, ref), err(base, ref))
