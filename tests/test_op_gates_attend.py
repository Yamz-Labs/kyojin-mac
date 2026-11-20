"""Shape-only gate of the fused latent attention (glm_dsa.attend_usable): it must refuse an attention module with a rope tail or a sink. CPU only."""
import os
import sys
import types

import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)



def _attn(**kw):
    base = dict(num_heads=64, kv_lora_rank=512)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_attend_usable_refuses_rope_tail_and_sink(monkeypatch):
    from kyojin_mlx import glm_dsa as G
    monkeypatch.setattr(G, "ATTN_ENABLED", True)
    q = mx.zeros((1, 64, 1, 256), dtype=mx.float16)
    assert G.attend_usable(_attn(), q)                                              # the GLM layout: no rope attribute at all
    assert G.attend_usable(_attn(qk_rope_head_dim=0), q)
    assert not G.attend_usable(_attn(qk_rope_head_dim=64), q)                       # shape test passes (64 heads, rank 512), the math has no rope tail
    for n in G.SINK_ATTRS:
        assert not G.attend_usable(_attn(**{n: mx.zeros((64,))}), q), n
    assert G.attend_usable(_attn(sinks=None), q)
    assert not G.attend_usable(_attn(num_heads=32), q)                              # the old shape tests still hold


def test_fuse_dsa_guards_use_the_same_test():
    import inspect
    from kyojin_mlx import glm_fuse_dsa as F
    assert "plain_latent_attn(self)" in inspect.getsource(F.attend_rows)
    assert "plain_latent_attn(self)" in inspect.getsource(F.attend_batch_ok)
