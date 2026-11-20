"""GLM-5.3 shared DSA indexer layers (config `indexer_types`: "shared" = no indexer of its own, reuses the top-k of the nearest earlier "full" layer; the
list comes from `indexer_types`, else `index_topk_pattern` ("F"/"S" string or list), else `index_topk_freq` + `index_skip_topk_offset`).
Tiny random pack (synth shape glm_shared: 1 KDA layer + sparse layers full / full / shared / full / shared), loaded through the Kyojin loader, against the
stock mlx-vlm forward of the weights it was written from. CPU, fp32 (stock mlx-vlm disagrees with itself between bf16 and f32 on random tiny GLM packs)."""
import copy
import json

import mlx.core as mx
import pytest

L = pytest.importorskip("mlx_vlm.models.glm5_next.language")
from kyojin_mlx import conformance as C, synth as S  # noqa: E402

SH = S.SHAPES["glm_shared"]
TYPES = SH.text["indexer_types"]


@pytest.fixture(autouse=True)
def _cpu_device():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        d = str(tmp_path_factory.mktemp("glm_shared"))
        ref = S.write_pack(SH, d)
        return d, ref, S.load_pack(SH, d)
    finally:
        mx.set_default_device(prev)


def _record(monkeypatch):
    """(layer index, topk array object) of every sparse attention call, in call order."""
    calls = []
    stock = L.Glm5NextAttention._attend

    def _attend(self, q, latent, new_latent, topk, *a, **k):
        calls.append((self, topk))
        return stock(self, q, latent, new_latent, topk, *a, **k)

    monkeypatch.setattr(L.Glm5NextAttention, "_attend", _attend)
    return calls


def test_shared_layers_carry_no_indexer_tensors_and_the_caches_have_no_pool(pack):
    d, _, (_, lm, mk) = pack
    layers = lm.model.layers
    for i, t in enumerate(TYPES):
        sa = layers[i].self_attn
        if layers[i].block_type == "linear_attention":
            continue
        assert (sa.indexer is None) == (t == "shared")
    assert [len(c.caches) for c in mk()[1:]] == [4 if t == "full" else 2 for t in TYPES[1:]]
    names = json.load(open(f"{d}/config.json"))["text_config"]["indexer_types"]
    assert names == TYPES
    from safetensors import safe_open
    with safe_open(f"{d}/model.safetensors", "np") as f:
        have = {k.split(".layers.")[1].split(".")[0] for k in f.keys() if ".indexer." in k}
    assert have == {str(i) for i, t in enumerate(TYPES) if t == "full" and i > 0}


def test_a_shared_layer_attends_with_the_selection_of_the_previous_full_layer(pack, monkeypatch):
    _, _, (_, lm, mk) = pack
    calls = _record(monkeypatch)
    ids = C.seq(30, 96, seed=3)
    cache = mk()
    out = lm(mx.array([ids]), cache=cache)
    mx.eval(out.logits)
    sparse = [i for i, t in enumerate(TYPES) if i > 0]
    assert len(calls) == len(sparse)
    owner = None
    for (attn, topk), i in zip(calls, sparse):
        assert attn is lm.model.layers[i].self_attn
        if TYPES[i] == "full":
            owner = topk
            valid = int((topk[0, -1] >= 0).sum().item())
            assert 0 < valid <= SH.text["index_topk"] + SH.text["index_kpool"] - 1 < 30     # the selection is a real cut (top-k pools + open tail), not all 30 rows
        else:
            assert topk is owner                                                      # same array: reused, not recomputed
    calls.clear()
    out = lm(mx.array([[5]]), cache=cache)                                             # decode: same rule on the 1-row path
    mx.eval(out.logits)
    owner = None
    for (attn, topk), i in zip(calls, sparse):
        owner = topk if TYPES[i] == "full" else owner
        if TYPES[i] == "shared":
            assert topk is owner


@pytest.mark.parametrize("chunk", [30, 8])
def test_kyojin_logits_match_stock_on_the_same_weights(pack, chunk):
    _, (_, ref_lm, ref_mk), (_, lm, mk) = pack
    ids = C.seq(30, 96, seed=3)
    want = C.forward(ref_lm, ref_mk, ids)[1]
    cache = mk()
    got = C.prefill(lm, mk, ids, chunk, cache=cache) if chunk < len(ids) else C.forward(lm, mk, ids)[1]
    got = got[1] if isinstance(got, tuple) else got
    assert C.maxdiff(got.reshape(-1)[-96:], want.reshape(-1)[-96:]) <= C.TOL


def test_the_sharing_rule_follows_the_vendor_config():
    """indexer_types given > index_topk_pattern (string or list) > frequency / offset; one entry per layer."""
    T = pytest.importorskip("mlx_vlm.models.glm5_next.config").TextConfig
    base = {k: v for k, v in SH.text.items() if k != "indexer_types"}
    assert T.from_dict(dict(base, index_topk_pattern="FFSFS")).indexer_types == ["full", "full", "shared", "full", "shared"]
    assert T.from_dict(dict(base, index_topk_pattern=["full", "shared", "shared", "full", "full"])).indexer_types[1:3] == ["shared", "shared"]
    assert T.from_dict(dict(base, index_topk_freq=2, index_skip_topk_offset=2)).indexer_types == ["full", "full", "shared", "full", "shared"]
    bad = dict(base, index_topk_pattern="FS")
    with pytest.raises(ValueError):
        T.from_dict(bad)
