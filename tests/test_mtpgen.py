"""kyojin_mlx.mtpgen on CPU: tiny random trunks (llama, glm4_moe, qwen3_next) with synthetic draft heads written in the public tensor layouts.
Pins: the loader finds the head from config keys + weight names, builds the block from the trunk's own layer class, refuses what it cannot drive, and a
speculative decode with oracle drafts (all accepted / partial / none) gives the tokens of plain greedy decode through KV trim and through the
fork + rerun rollback of GDN state. Random weights: nothing here says anything about acceptance of a real head. No Metal."""
import copy
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")
from kyojin_mlx import mtpgen  # noqa: E402
from kyojin_mlx.serve import backends  # noqa: E402
from tools.dayone import engine, shapes  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("KYOJIN_MTPGEN", "1")
    monkeypatch.setenv("KYOJIN_GENERIC_CACHE", "1")
    monkeypatch.setenv("KYOJIN_WIRED_MB", "0")
    yield
    mx.set_default_device(prev)


def _unstack(w: dict) -> dict:
    out = {}
    for k, v in w.items():
        if ".switch_mlp." in k:
            pre, rest = k.split(".switch_mlp.")
            for e in range(v.shape[0]):
                out[f"{pre}.experts.{e}.{rest}"] = v[e]
        else:
            out[k] = v
    return out


def head_tensors(model, cfg, layout: str, j: int, L: int, *, scales=False) -> dict:
    """Raw head tensors in the public naming of `layout`; the block is a copy of trunk layer j (its own parameter names), perturbed."""
    lm = getattr(model, "language_model", model)
    H, V = lm.args.hidden_size, lm.args.vocab_size
    mx.random.seed(5)
    blk = {k: v + 0.01 * mx.random.normal(v.shape) for k, v in tree_flatten(lm.model.layers[j].parameters())}
    blk = _unstack(blk)
    nrm = lambda: 1.0 + 0.1 * mx.random.normal((H,))                                    # noqa: E731
    fc = 0.05 * mx.random.normal((H, 2 * H))
    if layout == "qwen":
        r = {"mtp.fc.weight": fc, "mtp.pre_fc_norm_embedding.weight": nrm(), "mtp.pre_fc_norm_hidden.weight": nrm(), "mtp.norm.weight": nrm()}
        r.update({f"mtp.layers.0.{k}": v for k, v in blk.items()})
    elif layout == "appended":
        p = f"model.layers.{L}."
        r = {p + "eh_proj.weight": fc, p + "enorm.weight": nrm(), p + "hnorm.weight": nrm(), p + "shared_head.norm.weight": nrm(),
             p + "shared_head.head.weight": 0.05 * mx.random.normal((V, H)), p + "embed_tokens.weight": lm.model.embed_tokens.weight}
        r.update({p + k: v for k, v in blk.items()})
    elif layout == "mimo":
        p = "model.mtp_layers.0."
        r = {p + "input_proj.weight": fc, p + "token_layernorm.weight": nrm(), p + "hidden_layernorm.weight": nrm(), p + "final_layernorm.weight": nrm()}
        r.update({p + k: v for k, v in blk.items()})
    elif layout == "longcat":
        p = "model.mtp.layers.0."
        r = {p + "eh_proj.weight": fc, p + "enorm.weight": nrm(), p + "hnorm.weight": nrm(), p + "final_layernorm.weight": nrm()}
        r.update({p + "mtp_block." + k: v for k, v in blk.items()})
    elif layout == "ernie_no_final_norm":
        r = {"model.mtp_linear_proj.0.weight": fc, "model.mtp_emb_norm.0.weight": nrm(), "model.mtp_hidden_norm.0.weight": nrm()}
        r.update({"model.mtp_block.0." + k: v for k, v in blk.items()})
    else:
        raise KeyError(layout)
    return r


def make(tmp_path, shape: str, layout: str | None, j: int = 0, keys=("num_nextn_predict_layers",), **kw):
    cfg = copy.deepcopy(shapes.SHAPES[shape])
    t = cfg.get("text_config") or cfg
    for k in keys:
        t[k] = 1
    path = str(tmp_path / shape)
    model, lm, mk = engine.write_pack(cfg, path)
    if layout:
        raw = head_tensors(model, cfg, layout, j, int(t["num_hidden_layers"]), **kw)
        mx.save_safetensors(os.path.join(path, mtpgen.HEAD_FILE), raw)
    return path, cfg


def load(path, cfg):
    be = backends.load_generic(path, log=lambda *a: None)
    rep = mtpgen.enable(be, path, cfg, log=lambda *a: None)
    return be, rep


# ------------------------------------------------------------------------------------------------ loader
@pytest.mark.parametrize("layout", ["qwen", "appended", "mimo", "longcat"])
def test_layouts_load_on_dense_trunk(tmp_path, layout):
    path, cfg = make(tmp_path, "dense_gqa", layout)
    be, rep = load(path, cfg)
    assert rep["block_class"] == "TransformerBlock" and rep["layers"] == 1 and rep["order"] == ("he" if layout == "mimo" else "eh")
    assert rep["own_head"] == (layout == "appended")
    assert be.supports_spec


def test_moe_head_picks_the_moe_layer_and_dense_picks_the_dense_one(tmp_path):
    path, cfg = make(tmp_path / "a", "dense_moe_mix", "appended", j=2)
    _, rep = load(path, cfg)
    assert rep["trunk_layer"] >= 1                                       # layer 0 is dense: a routed block cannot be built from it
    path, cfg = make(tmp_path / "b", "dense_moe_mix", "appended", j=0)
    _, rep = load(path, cfg)
    assert rep["trunk_layer"] == 0


def test_refusals_carry_a_reason(tmp_path):
    path, cfg = make(tmp_path / "a", "dense_gqa", None)
    with pytest.raises(mtpgen.HeadRefused, match="config declares no draft head"):
        mtpgen.load_head(backends.load_generic(path, log=lambda *a: None).model, path, copy.deepcopy(shapes.SHAPES["dense_gqa"]))
    path, cfg = make(tmp_path / "b", "dense_gqa", "ernie_no_final_norm")
    with pytest.raises(mtpgen.HeadRefused, match="final_norm"):
        load(path, cfg)
    path, cfg = make(tmp_path / "c", "dense_gqa", "qwen")
    raw = dict(mx.load(os.path.join(path, mtpgen.HEAD_FILE)))
    raw["mtp.layers.0.mystery.weight"] = mx.zeros((3,))
    mx.save_safetensors(os.path.join(path, mtpgen.HEAD_FILE), raw)
    with pytest.raises(mtpgen.HeadRefused, match="names differ"):
        load(path, cfg)


def test_config_keys(tmp_path):
    for k in mtpgen.CFG_KEYS:
        assert mtpgen.head_count({k: 2}) == (k, 2)
        assert mtpgen.head_count({"text_config": {k: 1}}) == (k, 1)
    assert mtpgen.head_count({"num_nextn_predict_layers": 0}) == ("", 0)


def test_flag_off_load_backend_is_unchanged(tmp_path, monkeypatch):
    path, cfg = make(tmp_path, "dense_gqa", "qwen")
    monkeypatch.setenv("KYOJIN_BACKEND_FALLBACK", "1")
    monkeypatch.setenv("KYOJIN_MTPGEN", "0")
    with pytest.raises(Exception):                                       # the head file is a strict-load extra for a stock loader: flag 0 = no special casing
        be, _ = backends.load_backend(path)
        assert not be.supports_spec
        raise RuntimeError("loaded")
    monkeypatch.setenv("KYOJIN_MTPGEN", "1")
    be, _ = backends.load_backend(path, log=lambda *a: None)
    assert be.supports_spec and type(be).__name__ == "MtpGenGenericBackend"


# ------------------------------------------------------------------------------------------------ the verify / rollback contract
def plain(be, ids, n):
    cache = be.make_cache()
    lg = be.prefill_chunk(cache, ids)
    bc = be.batch([cache])
    out = [int(mx.argmax(lg[0]).item())]
    while len(out) < n:
        lg = be.step(bc, mx.array([out[-1]], dtype=mx.int32))
        out.append(int(mx.argmax(lg[0]).item()))
    return out


def spec(be, ids, n, D, corrupt=None, oracle=None):
    """The engine's eager round: draft -> spec_verify_bc -> accept -> commit. oracle: the plain stream (drafts = its continuation), corrupt: index of the
    draft to spoil (None = exact). Returns (tokens, per-round accepted counts)."""
    cache, d = be.make_cache(), be.new_draft()
    lg = None
    for s in range(0, len(ids), 7):                                      # chunked prefill feeds the head across chunk ends
        lg = be.prefill_chunk(cache, ids[s:s + 7], draft=d)
    bc = be.batch([cache])
    out = [int(mx.argmax(lg[0]).item())]
    accs = []
    while len(out) < n:
        p0 = len(ids) + len(out) - 1
        if oracle is None:
            dr = be.draft_many([(d, out[-1], p0, D, None)])[0]
        else:
            dr = list(oracle[len(out):len(out) + D])
            if corrupt is not None and len(dr) > corrupt:
                dr[corrupt] = (dr[corrupt] + 1) % 90
            be.draft_many([(d, out[-1], p0, 0, None)])
        res = be.spec_verify_bc(bc, [[out[-1]] + dr])
        am = mx.argmax(res.logits, axis=-1).tolist()
        a = 0
        while a < len(dr) and am[a] == dr[a]:
            a += 1
        res.commit([a], [d], [p0])
        be.settle(bc, [d])
        out += dr[:a] + [am[a]]
        accs.append(a)
        assert d.hid is None or int(d.hid.shape[1]) == len(d.tok) + 1
    return out[:n], accs


@pytest.mark.parametrize("shape,layout,j", [("dense_gqa", "qwen", 0), ("dense_moe_mix", "appended", 2), ("gdn_moe", "qwen", 1)])
def test_spec_equals_plain_greedy_with_oracle_drafts(tmp_path, shape, layout, j):
    path, cfg = make(tmp_path, shape, layout, j)
    be, _ = load(path, cfg)
    ids = [3, 9, 27, 5, 11, 40, 2, 8, 19, 33, 60, 7, 1, 44, 23, 6, 12, 50]
    n, D = 14, 3
    ref = plain(be, ids, n)
    for c in (None, 1, 0):
        got, accs = spec(be, ids, n, D, corrupt=c, oracle=[None] * 0 + ref)
        assert got == ref, (c, got, ref)
        if c is None:
            assert max(accs) == D                                        # every draft kept: the all-accepted commit path ran
        if c == 0:
            assert set(accs) == {0}                                      # nothing accepted: full rollback every round
        if c == 1:
            assert 1 in accs                                             # partial acceptance
    got, accs = spec(be, ids, n, D)                                      # the head's own (random) drafts
    assert got == ref


def test_gdn_trunk_takes_the_fork_rerun_rollback(tmp_path):
    path, cfg = make(tmp_path, "gdn_moe", "qwen", 1)
    be, _ = load(path, cfg)
    cache = be.make_cache()
    be.prefill_chunk(cache, [3, 9, 27, 5, 11, 40])
    bc = be.batch([cache])
    res = be.spec_verify_bc(bc, [[4, 5, 6]])
    assert res.pre[0] is not None                                        # GDN layers cannot trim: a fork was taken
    d = be.new_draft()
    res.commit([0], [d], [6])
    assert d.hid.shape[1] == 1 and d.tok == []


def test_contract_check_reports_a_number(tmp_path):
    path, cfg = make(tmp_path, "dense_gqa", "qwen")
    be = backends.load_generic(path, log=lambda *a: None)
    assert mtpgen.contract_ok(be) < 2e-3
    with pytest.raises(mtpgen.HeadRefused, match="verify-4 contract"):
        mtpgen.contract_ok(be, tol=-1.0)


def test_draft_vocab_flag_keeps_the_output(tmp_path, monkeypatch):
    """KYOJIN_DRAFT_VOCAB: the draft argmax runs on a head slice, the verify decides: same tokens as plain greedy decode."""
    from kyojin_mlx import draftvocab
    path, cfg = make(tmp_path, "dense_gqa", "qwen", 0)
    be, _ = load(path, cfg)
    ids = [3, 9, 27, 5, 11, 40, 2, 8, 19, 33, 60, 7, 1, 44, 23, 6, 12, 50]
    ref = plain(be, ids, 14)
    monkeypatch.setattr(draftvocab, "ROUND", 8)
    monkeypatch.setenv("KYOJIN_DRAFT_VOCAB", "32")
    got, _ = spec(be, ids, 14, 3)
    assert got == ref
    assert be.__dict__.get("_draft_subheads"), "the sub-head was never used"
