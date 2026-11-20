"""NOTE: names, tensors and transforms come from the generated tables, the config from the declarative config rows (kyojin_mlx/foreign_cfgrows.py; the hand builders are the reference, tests/test_arch_kit_cfgrows.py).
The 4 architectures that had table-level coverage only (qwen35, qwen35moe, glm4moe, deepseek2): the synthetic tiny-config GGUF of the existing hand-path round-trip tests
(tests/test_gguf_arch.py, test_gguf_arch2.py: tiny random mlx-lm model -> the converter replayed forward, written from the hand tables) is loaded a second time through the
GENERATED llama.cpp tables (foreign_archkit_cpu.load_cpu: resolver, config rows, vetted rows) and its forward must equal the hand-written path's. CPU only."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
mx.set_default_device(mx.cpu)

import test_gguf_arch as T      # noqa: E402
import test_gguf_arch2 as T2    # noqa: E402
from kyojin_mlx import foreign_archkit_cpu as KC, foreign_linear as FL  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

CMP = ("from mlx.utils import tree_flatten\nP = dict(tree_flatten(m.parameters()))\nassert set(R.files) == set(P), (sorted(set(R.files) ^ set(P))[:4])\n"
       "for k in R.files:\n    e = float(np.abs(np.array(P[k].astype(mx.float32)) - R[k]).max()) if R[k].size else 0.0\n    assert e < 1e-3, ('tensor differs from the reference', k, e)\n")

CASES = {
    "qwen35": T.test_qwen35_gguf_forward_matches_the_hf_checkpoint,
    "qwen35moe": T2.test_qwen35moe_gguf_loads_to_the_same_parameters_and_logits,
    "glm4moe": T2.test_glm4moe_gguf_round_trip_with_expert_bias_shared_expert_and_dense_lead,
    "deepseek2": T2.test_deepseek2_gguf_round_trip_splits_kv_b_without_a_transpose,
}


# Reference of qwen35 / qwen35moe: the mlx-lm model built from the HF-layout tensors (T._ref_model), NOT the hand-loaded model. The first version took load_foreign's LAST return value:
# for qwen35 that is the model the hand test loads on purpose with the wrong V-head inverse (relative error 1.62 = the test's own negative control), for qwen35moe an fp16 hand
# model measured against fp32 (0.044 on 10 tokens). The kit's parameters equal the HF reference's, tensor by tensor (checked inside the clean interpreter below).
_PARAMS = list(CASES)


@pytest.mark.parametrize("arch", _PARAMS)
def test_generated_tables_give_the_same_forward_as_the_hand_tables(arch, tmp_path, monkeypatch):
    seen, refs = [], []
    real, real_tiny, real_ref = FL.load_foreign, T._tiny_dense, T._ref_model

    def spy(path, *a, **k):
        r = real(path, *a, **k)
        seen.append((path, r[0]))
        return r

    def spy_tiny(*a, **k):
        r = real_tiny(*a, **k)
        refs.append(r[0])
        return r

    def spy_ref(*a, **k):
        r = real_ref(*a, **k)
        refs.append(r[0])
        return r

    monkeypatch.setattr(T, "_ref_model", spy_ref)
    monkeypatch.setattr(FL, "load_foreign", spy)
    monkeypatch.setattr(T, "_tiny_dense", spy_tiny)
    CASES[arch](tmp_path)                                          # the hand path's own assertions run first
    path, hand = seen[-1]
    ids = [[3, 7, 1, 30, 22, 5, 9, 2, 11, 4]]
    # The loaders patch mlx-lm classes in place, and a MoE hand model cannot run a forward on dense tiny tensors afterwards: the reference of those is the ORIGINAL tiny mlx-lm
    # model (weights saved, rebuilt in a clean interpreter, like the kit side).
    ref_code = ""
    if refs:                                                       # an hf-layout reference model exists (T._ref_model / T._tiny_dense)
        import dataclasses
        from mlx.utils import tree_flatten
        ref = refs[-1]
        np.savez(tmp_path / "ref.npz", **{k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(ref.parameters())})
        ref_code = ("import importlib, dataclasses\nmod = importlib.import_module(%r)\nr = mod.Model(mod.ModelArgs.from_dict(%r))\n"
                    "r.load_weights([(k, mx.array(v)) for k, v in np.load(%r).items()])\nnp.save(%r, np.array(r(mx.array(%r)).astype(mx.float32)))\nR = np.load(%r)\n"
                    % (type(ref).__module__, dataclasses.asdict(ref.args), str(tmp_path / "ref.npz"), str(tmp_path / "ref_logits.npy"), ids, str(tmp_path / "ref.npz")))
        want_f = tmp_path / "ref_logits.npy"
    else:
        want_f = None
        a = np.array(hand(mx.array(ids)).astype(mx.float32))
    out = tmp_path / "kit_logits.npy"
    code = ("import sys, numpy as np; sys.path.insert(0, %r)\nimport mlx.core as mx; mx.set_default_device(mx.cpu)\n%s"
            "from kyojin_mlx import foreign_archkit_cpu as KC\nm, rep = KC.load_cpu(%r)\n"
            "assert rep['arch'] == %r and rep['names'].ambiguous == {}\n%snp.save(%r, np.array(m(mx.array(%r)).astype(mx.float32)))\n"
            % (ROOT, ref_code, str(path), arch, CMP if refs else "", str(out), ids))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-900:]
    b = np.load(out)
    if want_f is not None:
        a = np.load(want_f)
    assert a.shape == b.shape
    err = np.abs(a - b).max() / max(1.0, np.abs(a).max())
    assert err < 3e-2, f"{arch}: kit forward differs from the reference by {err:.4f} (relative to the logit scale)"
