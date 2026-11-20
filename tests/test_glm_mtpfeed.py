"""P2-MTPFEED: the prompt feed hides the head's projected K / V cache slot (flag glm_mtpfeed), nothing else changes."""
import mlx.core as mx

from kyojin_mlx import flags, glm_nextn as N


def test_noproj_view_hides_slot_3_only():
    cl = ["kv", "idx", "pool", "proj"]
    v = N._NoProj(cl)
    assert [v[0], v[1], v[2], v[3]] == ["kv", "idx", "pool", None]


def test_flag_registered_default_off_and_switchable():
    f = flags.FLAGS["glm_mtpfeed"]
    assert f.env == "KYOJIN_GLM_MTPFEED" and f.default == "0" and f.runtime


class _Head:
    def make_cache(self):
        return ["c"]


def _nextn():
    n = N.NextN.__new__(N.NextN)
    n.d = _Head()
    n.calls = []
    n._fwd = lambda cache, toks, hid, pos0, last_only, d=None, noproj=False: n.calls.append((pos0, int(toks.shape[1]), noproj)) or mx.zeros((1, 1, 4))
    return n


def _state(m, p=0):
    st = N.GlmDraft()
    st.cache, st.pos = ["c"], p
    st.hid, st.htok = mx.zeros((m, 4)), mx.arange(m, dtype=mx.int32)
    return st


def test_prompt_feed_only_uses_noproj(monkeypatch):
    n = _nextn()
    monkeypatch.setattr(N, "MTPFEED", True)
    n.feed(_state(1030), d=object())                   # prompt feed: 1029 pairs in 512-row pieces
    assert n.calls == [(0, 512, True), (512, 512, True), (1024, 5, True)]
    n.calls.clear()
    n.feed(_state(3), extra=7, d=object())             # draft feed (extra): stock path
    assert n.calls and all(not c[2] for c in n.calls)


def test_flag_off_is_stock(monkeypatch):
    n = _nextn()
    monkeypatch.setattr(N, "MTPFEED", False)
    n.feed(_state(600), d=object())
    assert all(not c[2] for c in n.calls)


# ---- the real head block at a small size (Metal): the three caches that matter are bit-identical with the projected cache hidden
import json
import os

import pytest

CFG = os.path.expanduser("/path/to/data")


def _head(dt, seed=0):
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
    from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
    if not os.path.exists(CFG):
        pytest.skip("no GLM config on this machine")
    c = json.load(open(CFG))["text_config"]
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        c[k] = c[k][:1]
    c.update(num_hidden_layers=1, hidden_size=512, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, intermediate_size=256, first_k_dense_replace=0)
    d = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict({"text_config": c}))
    mx.random.seed(seed)
    new = []
    for k, v in tree_flatten(d.parameters()):
        if v.ndim >= 2:
            new.append((k, (mx.random.normal(v.shape) * (1.0 / v.shape[-1] ** 0.5)).astype(mx.float32 if "compress" in k else dt)))
        else:
            new.append((k, (mx.zeros(v.shape) if "bias" in k else mx.ones(v.shape)).astype(dt)))
    d.update(tree_unflatten(new))
    mx.eval(d.parameters())
    d.eval()
    return d


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("dt", [mx.float16, mx.float32])
def test_head_caches_bit_identical_without_projected_cache(dt):
    mx.set_default_device(mx.gpu)
    d = _head(dt)
    n = 2100
    mx.random.seed(1)
    e = mx.random.normal((1, n, 512)).astype(dt)
    hid = mx.random.normal((1, n, 512)).astype(dt)
    out = {}
    for noproj in (False, True):
        cache = d.make_cache()
        for a in range(0, n, N.CHUNK):
            b = min(n, a + N.CHUNK)
            x = d.eh_proj(mx.concatenate([d.enorm(e[:, a:b]), d.hnorm(hid[:, a:b])], axis=-1))
            d.mtp_block(x, N._NoProj(cache[0]) if noproj else cache[0], last_only=True)      # output discarded, as in prefill_chunk
        subs = cache[0].caches
        arrs = [a for s in subs[:3] for a in (s.state if isinstance(s.state, (list, tuple)) else [s.state]) if isinstance(a, mx.array)]
        proj = [a for a in (subs[3].state if isinstance(subs[3].state, (list, tuple)) else [subs[3].state]) if isinstance(a, mx.array)]
        mx.eval(*arrs, *proj)
        out[noproj] = (arrs, proj)
    a0, a1 = out[False][0], out[True][0]
    assert len(a0) == len(a1) >= 3
    for x, y in zip(a0, a1):
        assert x.shape == y.shape and bool(mx.array_equal(x, y, equal_nan=True))
    assert sum(int(p.nbytes) for p in out[False][1]) > 0 and sum(int(p.nbytes) for p in out[True][1]) == 0
