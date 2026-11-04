"""Flag visiongen on tiny synthetic towers (tools/dayone), CPU only: a llava-style model (CLIP tower + projector + llama trunk, 1-D positions) for the
exact path, a Qwen3.5 tower (multimodal rope) for the refusal. Random weights: these tests say nothing about real numerics, only that the plumbing is exact."""
import base64
import io
import json
import os

import numpy as np
import pytest

pytest.importorskip("mlx_vlm")
pytest.importorskip("transformers")
pytest.importorskip("PIL")
os.environ.setdefault("KYOJIN_WIRED_MB", "0")
import mlx.core as mx                                                    # noqa: E402
from PIL import Image                                                    # noqa: E402

from kyojin_mlx import ds41_vision as DV, visiongen as VG                # noqa: E402
from kyojin_mlx.serve import backends as B, chat                         # noqa: E402
from tools.dayone import engine as E, shapes as S                        # noqa: E402

V, TOK = 96, 90
LLAVA = dict(model_type="llava", architectures=["LlavaForConditionalGeneration"], image_token_index=TOK, vision_feature_select_strategy="default",
             vision_feature_layer=-2, vocab_size=V,
             text_config=dict(model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                              rms_norm_eps=1e-6, vocab_size=V, max_position_embeddings=512, tie_word_embeddings=False),
             vision_config=dict(model_type="clip_vision_model", num_hidden_layers=2, hidden_size=32, intermediate_size=64, num_attention_heads=2,
                                image_size=16, patch_size=4, projection_dim=32, vocab_size=V))
TEMPLATE = ("{%- for m in messages %}t{{ 1 if m['role'] == 'user' else 2 }} {% if m['content'] is string %}{{ m['content'] }}{% else %}"
            "{%- for p in m['content'] %}{% if p['type'] == 'image' %}<image> {% else %}{{ p['text'] }} {% endif %}{% endfor %}{% endif %} {% endfor %}"
            "{%- if add_generation_prompt %}t2 {% endif %}")


def _pack(tmp_path, cfg, name):
    path = str(tmp_path / name)
    E.write_pack(cfg, path)
    tj = json.load(open(f"{path}/tokenizer.json"))
    tj["model"]["vocab"].pop(f"t{TOK}")
    tj["added_tokens"].append({"id": TOK, "content": "<image>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False})
    json.dump(tj, open(f"{path}/tokenizer.json", "w"))
    from transformers import PreTrainedTokenizerFast
    tid = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json").encode("<image>")[0]
    c = json.load(open(f"{path}/config.json"))
    c["image_token_index"] = tid                                     # whatever id the tokenizer gave the added token
    json.dump(c, open(f"{path}/config.json", "w"))
    json.dump(dict(image_processor_type="CLIPImageProcessor", size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}, do_center_crop=True,
                   do_normalize=True, image_mean=[.5, .5, .5], image_std=[.5, .5, .5], do_resize=True, resample=3), open(f"{path}/preprocessor_config.json", "w"))
    json.dump(dict(processor_class="LlavaProcessor", patch_size=4, vision_feature_select_strategy="default", image_token="<image>", num_additional_image_tokens=1),
              open(f"{path}/processor_config.json", "w"))
    open(f"{path}/chat_template.jinja", "w").write(TEMPLATE)
    return path


def _png(seed):
    rng = np.random.RandomState(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.randint(0, 255, (24, 40, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _rec(raw):
    return {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}


@pytest.fixture(scope="module")
def llava(tmp_path_factory):
    path = _pack(tmp_path_factory.mktemp("vg"), LLAVA, "llava")
    old = os.environ.get("KYOJIN_VISIONGEN")
    os.environ["KYOJIN_VISIONGEN"] = "1"
    try:
        be = B.load_generic(path, log=lambda *a: None)
    finally:
        os.environ.pop("KYOJIN_VISIONGEN") if old is None else os.environ.__setitem__("KYOJIN_VISIONGEN", old)
    return path, be


def _ids(vis, recs, head=(1, 5, 6), tail=(7, 8, 2)):
    plans = [vis.plan(r) for r in recs]
    toks = list(head) + [vis.image_token_id] * len(plans) + list(tail)
    ids, spans = vis.expand(toks, plans)
    return ids, spans, vis.real_ids(ids)


def _prefill(be, rids, spans, cuts, cache=None):
    cache = cache if cache is not None else be.make_cache()
    lg = None
    for a, b in zip(cuts, cuts[1:]):
        lg = be.prefill_chunk(cache, rids[a:b], vl=(spans, a))
    return cache, lg


def _stock(be, rids, vis, rec):
    """The model's own one-shot path: get_input_embeddings on the full sequence, then the trunk."""
    n, extra = vis._process(DV.decode_image(DV._record_bytes(rec)))
    arr = {k: mx.array(v) for k, v in extra.items()}
    pv = arr.pop("pixel_values")
    h = be.model.get_input_embeddings(mx.array([rids]), pv, **arr).inputs_embeds
    lm = be.model.language_model
    out = lm.model(mx.array([rids]), cache=be.make_cache(), inputs_embeds=h)
    return be.MU.head(lm, out[:, -1:])[:, -1].astype(mx.float32)


def test_flag_off_rejects_image_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    with pytest.raises(chat.ChatError):
        chat.render_jinja(TEMPLATE, msgs, None)
    f = chat.QwenFamily(None, "/nonexistent", template=TEMPLATE)
    assert f.vision is None
    with pytest.raises(chat.ChatError):
        f.render({"messages": msgs})


def test_vids_and_errors(llava):
    path, be = llava
    v = be.vision
    assert v.available(), v.reason
    a, b = _png(1), _png(2)
    ia, sa, _ = _ids(v, [_rec(a)])
    ia2, _, _ = _ids(v, [_rec(a)])
    ib, _, _ = _ids(v, [_rec(b)])
    assert ia == ia2 and ia != ib                                   # same bytes, same ids; other bytes, other ids
    assert sa[0].plan.n_tokens == 16 == sum(1 for t in ia if t >= DV.VID_BASE)
    with pytest.raises(DV.VisionError):                              # two images, one placeholder
        v.expand([1, v.image_token_id, 2], [v.plan(_rec(a)), v.plan(_rec(b))])
    with pytest.raises(DV.VisionError):
        v.plan({"url": "http://example.com/x.png"})                  # no fetching
    with pytest.raises(DV.VisionError):
        v.plan({"data": base64.b64encode(b"not an image").decode()})


def test_salt_depends_on_processor_files(llava, tmp_path):
    path, be = llava
    import shutil
    other = str(tmp_path / "p2")
    shutil.copytree(path, other)
    pc = json.load(open(f"{other}/preprocessor_config.json"))
    pc["image_mean"] = [.4, .5, .6]
    json.dump(pc, open(f"{other}/preprocessor_config.json", "w"))
    v2 = VG.GenVision(other, be.model)
    raw = _png(3)
    assert be.vision.plan(_rec(raw)).vids != v2.plan(_rec(raw)).vids


def test_chunked_overlay_equals_stock(llava):
    path, be = llava
    v = be.vision
    raw = _png(4)
    ids, spans, rids = _ids(v, [_rec(raw)])
    L = len(rids)
    ref = _stock(be, rids, v, _rec(raw))
    _, one = _prefill(be, rids, spans, [0, L])
    d1 = float(mx.abs(one - ref).max().item())
    _, cut = _prefill(be, rids, spans, [0, 7, 12, L])                # boundaries inside the span
    d2 = float(mx.abs(cut - ref).max().item())
    print(f"max |logit diff| vs the model's one-shot path: one chunk {d1:.2e}, chunks 7/12 {d2:.2e}")
    assert d1 < 1e-4 and d2 < 1e-4
    txt = mx.array([1, 5, 6, 7, 8, 2])
    base = be.prefill_chunk(be.make_cache(), [1, 5, 6, 7, 8, 2])
    assert float(mx.abs(base - one).max().item()) > 1e-3             # the image reached the trunk


def test_repeated_image_one_encode(llava):
    path, be = llava
    v = be.vision
    raw = _png(5)
    ids, spans, rids = _ids(v, [_rec(raw)])
    n0 = v.stats["encodes"]
    DV.prepare(spans, 0, len(ids))
    DV.prepare(spans, 0, len(ids))
    ids2, spans2, _ = _ids(v, [_rec(raw)])
    DV.prepare(spans2, 0, len(ids2))
    assert v.stats["encodes"] - n0 == 1 and ids == ids2


def test_two_sessions_and_fork(llava, monkeypatch):
    monkeypatch.setenv("KYOJIN_GENERIC_CACHE", "1")                  # batch_caches through the cache's own merge (cloud/cdayone)
    path, be = llava
    v = be.vision
    ia, sa, ra = _ids(v, [_rec(_png(6))])
    ib, sb, rb = _ids(v, [_rec(_png(7))], tail=(7, 8, 9, 2))
    La, Lb = len(ra), len(rb)
    ca_solo, la_solo = _prefill(be, ra, sa, [0, 9, La])
    cb_solo, lb_solo = _prefill(be, rb, sb, [0, 11, Lb])
    ca, cb = be.make_cache(), be.make_cache()                        # interleaved chunks of two sessions
    _prefill(be, ra, sa, [0, 9], ca)
    _prefill(be, rb, sb, [0, 11], cb)
    _, la = _prefill(be, ra, sa, [9, La], ca)
    _, lb = _prefill(be, rb, sb, [11, Lb], cb)
    assert float(mx.abs(la - la_solo).max().item()) == 0.0 and float(mx.abs(lb - lb_solo).max().item()) == 0.0
    # fork after the image prefill, continue both: same tokens, same logits
    fork = be.fork(ca)
    nxt = [3, 4]
    x = be.prefill_chunk(ca, nxt)
    y = be.prefill_chunk(fork, nxt)
    assert float(mx.abs(x - y).max().item()) == 0.0
    # the batched 2-row step equals the solo steps
    bc = be.batch([cb_solo, be.fork(cb)])
    st = be.step(bc, mx.array([3, 3]))
    s1 = be.step(be.batch([be.fork(cb)]), mx.array([3]))
    assert float(mx.abs(st[0] - s1[0]).max().item()) < 1e-4


def test_template_places_images_and_server_path(llava):
    path, be = llava
    f = chat.QwenFamily(None, path)
    f.vision = be.vision
    raw = _png(8)
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "t5 t6"}, {"type": "image_url", "image_url": {"url": _rec(raw)["url"]}}]}]}
    r = f.render(body)
    assert r.prompt.count("<image>") == 1 and len(r.images) == 1
    # a template that drops the image: count mismatch is a 400-class error, not a guess
    f2 = chat.QwenFamily(None, path, template="{% for m in messages %}{% if m['content'] is string %}{{ m['content'] }}{% else %}t1 {% endif %}{% endfor %}")
    f2.vision = be.vision
    r2 = f2.render(body)
    with pytest.raises(DV.VisionError):
        be.vision.expand([1, 2], [be.vision.plan(r2.images[0])])


def test_mrope_trunk_refused(tmp_path):
    cfg = dict(S.SHAPES["vl_gdn_mtp"])
    cfg["vision_config"] = dict(cfg["vision_config"])
    path = str(tmp_path / "q35")
    E.write_pack(cfg, path)
    json.dump(dict(image_processor_type="Qwen2VLImageProcessor"), open(f"{path}/preprocessor_config.json", "w"))
    os.environ["KYOJIN_VISIONGEN"] = "1"
    try:
        be = B.load_generic(path, log=lambda *a: None)
    finally:
        os.environ.pop("KYOJIN_VISIONGEN")
    assert not be.vision.available() and "multimodal rope" in be.vision.reason
    with pytest.raises(DV.VisionError):
        be.vision.plan({"data": base64.b64encode(_png(1)).decode()})


def test_mrope_overlay_would_differ(tmp_path):
    """Why Qwen3.5-style trunks are refused: the model's own positions over an image are not the 1-D offsets of the served step."""
    cfg = S.SHAPES["vl_gdn_mtp"]
    model, lm, mk = E.build(cfg)
    vc = cfg["vision_config"]
    m, p = vc["spatial_merge_size"], vc["patch_size"]
    n = (2 * m) ** 2 // (m * m)
    ids = [3, 4, cfg["vision_start_token_id"]] + [cfg["image_token_id"]] * n + [cfg["vision_end_token_id"], 5, 6]
    px = mx.random.normal(((2 * m) ** 2, vc["in_channels"] * vc["temporal_patch_size"] * p * p)) if "in_channels" in vc else \
        mx.random.normal(((2 * m) ** 2, 3 * vc["temporal_patch_size"] * p * p))
    f = model.get_input_embeddings(mx.array([ids]), px, image_grid_thw=mx.array([[1, 2 * m, 2 * m]]))
    own = lm(mx.array([ids]), cache=mk(), inputs_embeds=f.inputs_embeds, position_ids=f.position_ids).logits[:, -1]
    flat = lm(mx.array([ids]), cache=mk(), inputs_embeds=f.inputs_embeds).logits[:, -1]
    d = float(mx.abs(own - flat).max().item())
    print(f"qwen3_5 synthetic: max |logit diff| model positions vs 1-D offsets: {d:.3f}")
    assert d > 1e-3


def _collect(h):
    toks = []
    while True:
        ev = h.events.get(timeout=120)
        if ev[0] == "tok":
            toks.append(ev[1])
        else:
            return toks


def test_engine_repeat_and_two_sessions(llava, monkeypatch):
    """Through the real Engine: a repeated image is encoded once and its prefix is a cache hit; two image requests in flight give their solo tokens."""
    monkeypatch.setenv("KYOJIN_GENERIC_CACHE", "1")
    from kyojin_mlx.serve.engine import Engine, Sampling
    path, be = llava
    v = be.vision
    ia, sa, ra = _ids(v, [_rec(_png(11))], tail=(7, 8, 9, 7, 8, 9, 2))
    ib, sb, rb = _ids(v, [_rec(_png(12))], tail=(9, 8, 2))

    def run(reqs, sessions):
        eng = Engine(be, sessions=sessions, ctx=512, chunk=8, block=8, prefix_bytes=1 << 22)
        eng.start()
        try:
            hs = [eng.submit(i, max_new=6, sampling=Sampling(temperature=0.0), vl=s, real_ids=r) for i, s, r in reqs]
            return [_collect(h) for h in hs], eng
        finally:
            eng.stop()

    (solo_a,), _ = run([(ia, sa, ra)], 1)
    (solo_b,), _ = run([(ib, sb, rb)], 1)
    n0 = v.stats["encodes"]
    (t1, t2), eng = run([(ia, sa, ra), (ib, sb, rb)], 2)
    assert t1 == solo_a and t2 == solo_b
    # the same image again in a later request of the same engine: prefix hit, no second encode
    ia, sa, ra = _ids(v, [_rec(_png(13))])                               # an image never encoded before
    eng = Engine(be, sessions=1, ctx=512, chunk=8, block=8, prefix_bytes=1 << 22)
    eng.start()
    try:
        e0 = v.stats["encodes"]
        first = _collect(eng.submit(ia, max_new=4, sampling=Sampling(temperature=0.0), vl=sa, real_ids=ra))
        e1 = v.stats["encodes"]
        again = _collect(eng.submit(ia, max_new=4, sampling=Sampling(temperature=0.0), vl=sa, real_ids=ra))
        hits = eng.prefix.hits
    finally:
        eng.stop()
    print(f"engine: encodes first={e1 - e0} after repeat={v.stats['encodes'] - e1}, prefix hits={hits}, repeat equals first: {first == again}")
    assert e1 - e0 == 1 and first == again and v.stats["encodes"] == e1 and hits >= 1
