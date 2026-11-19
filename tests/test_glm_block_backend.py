"""GlmBackend + the block drafters end to end on the tiny synthetic GLM pack (kyojin_mlx.synth glm_shared), CPU: enable_block loads both drafters, prefill / verify / commit /
plain step feed the per-session ring, the draft state stays lined up with the committed positions, glm_block_arm switches the drafter, and the backend is untouched when
KYOJIN_GLM_BLOCK is not used."""
import json
import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.glm5_next.language")
from kyojin_mlx import patch as P, synth as S  # noqa: E402
import test_glm_block as T  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_device():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "V", 96)
    d = tmp_path / "pack"
    d.mkdir()
    S.write_pack(S.SHAPES["glm_shared"], str(d))
    model, spec, cfg = P.build_model(str(d))
    plan = P.load_dense(model, str(d), spec, cfg)
    P.check_loaded(model, plan)
    mx.eval(model.language_model.parameters())
    from kyojin_mlx.serve.backends import GlmBackend
    be = GlmBackend(model, spec)
    dirs = {}
    for kind in ("dflash2", "dspark"):
        c = {}
        _, w, _ = T.weights(kind, hidden_cfg_out=c)
        p = tmp_path / kind
        p.mkdir()
        (p / "config.json").write_text(json.dumps(c))
        mx.save_safetensors(str(p / "model.safetensors"), w)
        dirs[kind] = str(p)
    return be, dirs


def test_untouched_without_block(backend):
    be, _ = backend
    assert be.block is None and be.blocks is None and be.new_draft() is None and be.draft_confs() is None


@pytest.mark.parametrize("arm", ["dflash2", "dspark"])
def test_rounds_keep_the_draft_state_lined_up(backend, monkeypatch, arm):
    be, dirs = backend
    monkeypatch.setenv("KYOJIN_GLM_BLOCK_Q", "0")
    monkeypatch.setenv("KYOJIN_GLM_TAP_CLAMP", "1")
    info = be.enable_block(f"dflash2:{dirs['dflash2']},dspark:{dirs['dspark']}")
    assert info["spec"] == "dflash2+dspark+lookup" and be.supports_spec and not be.lookup_only
    monkeypatch.setenv("KYOJIN_GLM_BLOCK_ARM", arm)
    assert be.block.kind == arm
    ids = [int(x) for x in mx.random.randint(1, 90, (20,)).tolist()]
    cache, dr = be.make_cache(), be.new_draft()
    assert dr.kind == arm
    lg = be.prefill_chunk(cache, ids[:12], draft=dr)
    lg = be.prefill_chunk(cache, ids[12:], draft=dr)
    assert dr.pos + dr.m == len(ids)                                   # every prompt position's taps are (or were) in the state
    seq, p0 = ids + [int(mx.argmax(lg, axis=-1)[0])], len(ids)
    bc = be.batch([cache])
    for _ in range(3):
        out = be.draft_many([(dr, seq[-1], p0, 3, None)])[0]
        assert 1 <= len(out) <= 3 and all(0 <= t < 96 for t in out)
        if arm == "dspark":
            conf = be.draft_confs()[0]
            assert len(conf) == len(out) and all(0.0 <= c <= 1.0 for c in conf)
        toks = [seq[-1]] + out
        pas = be.spec_verify_bc(bc, [toks])
        post = [int(mx.argmax(pas.logits[i])) for i in range(len(toks))]
        acc = 0
        while acc < len(toks) - 1 and toks[acc + 1] == post[acc]:
            acc += 1
        pas.commit([acc], [dr], [p0])
        seq += toks[1:1 + acc] + [post[acc]]
        p0 += acc + 1
        assert dr.pos + dr.m == p0                                      # the next draft call lines up with the pending token
    lg2 = be.step_feed(bc, mx.array([seq[-1]], dtype=mx.int32), [dr], [p0])
    mx.eval(lg2)
    assert dr.pos + dr.m == p0 + 1
    assert be.draft_many([(dr, 5, p0 + 7, 3, None)]) == [[]]            # a draft asked at a position the state does not line up with is empty
