"""DS4.1 vision, HTTP side and bookkeeping: no tower, no GPU. The tower itself is checked by tools/ds41_vision_check.py."""
import base64
import io
import json
import struct

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from kyojin_mlx import ds41_vision as V
from kyojin_mlx.serve import ds41_chat
from kyojin_mlx.serve.chat import Ds41Family, Rendered, token_cuts

IMG = 129264


def png(w=64, h=64, color=(200, 10, 10), mode="RGB", exif_orientation=None, fmt="PNG"):
    im = Image.new(mode, (w, h), color)
    b = io.BytesIO()
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        im.save(b, "JPEG", exif=ex)
    else:
        im.save(b, fmt)
    return b.getvalue()


def rec(raw):
    return {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}


@pytest.fixture
def v(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": IMG, "vision_config": {
        "num_hidden_layers": 32, "hidden_size": 1024, "heads": 16, "intermediate_size": 2816, "patch_size": 14,
        "downsample_ratio": 3, "max_image_tokens": 1024, "min_pixels": 295936}}))
    keys = ["vision.patch_embed.proj.weight", "aligner.w1.weight", "image_start", "image_newline", "image_end"]
    h = json.dumps({k: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    (tmp_path / "global.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"\0\0")
    return V.Vision(tmp_path)


def test_available_needs_the_tower_in_the_pack(v, tmp_path):
    assert v.available()
    (tmp_path / "global.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    assert not V.Vision(tmp_path).available()


def test_span_layout_matches_the_reference(v):
    from mlx_vlm.models.deepseek_v41.processing_deepseek_v41 import image_token_types
    p = v.plan(rec(png()))
    assert p.n_tokens == len(image_token_types(p.n_llm_h, p.n_llm_w))


def test_expand_ids_and_real_ids(v):
    a, b = v.plan(rec(png(color=(1, 2, 3)))), v.plan(rec(png(color=(4, 5, 6))))
    ids, spans = v.expand([5, IMG, 6, IMG, 7], [a, b])
    assert [s.start for s in spans] == [1, 2 + a.n_tokens]
    assert len(ids) == 3 + a.n_tokens + b.n_tokens
    assert ids[0] == 5 and ids[-1] == 7 and ids[1 + a.n_tokens] == 6
    assert all(t >= V.VID_BASE for t in ids[1:1 + a.n_tokens])
    assert max(ids) < 2 ** 31                                         # the hashers read int32
    real = v.real_ids(ids)
    assert real[1:1 + a.n_tokens] == [IMG] * a.n_tokens and max(real) == IMG


def test_two_images_same_grid_never_share_a_prefix(v):
    a, b = v.plan(rec(png(color=(1, 2, 3)))), v.plan(rec(png(color=(4, 5, 6))))
    assert (a.n_llm_h, a.n_llm_w) == (b.n_llm_h, b.n_llm_w)
    assert v.expand([IMG], [a])[0][0] != v.expand([IMG], [b])[0][0]


def test_same_image_same_ids_across_requests(v):
    raw = png()
    assert v.expand([IMG], [v.plan(rec(raw))])[0] == v.expand([IMG], [V.Vision(v.model_dir).plan(rec(raw))])[0]


def test_placeholder_count_mismatch_is_refused(v):
    p = v.plan(rec(png()))
    with pytest.raises(V.VisionError):
        v.expand([IMG, IMG], [p])
    with pytest.raises(V.VisionError):
        v.expand([1], [p])
    assert v.expand([1], [p], whole=False)[0] == [1]


def test_cut_lands_with_an_image_in_history(v):
    p = v.plan(rec(png()))
    enc = {"cut": [5, IMG, 6], "full": [5, IMG, 6, 7, 8]}
    ids, _ = v.expand(enc["full"], [p])
    cuts = token_cuts(["cut"], ids, lambda t: v.expand(enc[t], [p], whole=False)[0])
    assert cuts == [3 + p.n_tokens - 1]


@pytest.mark.parametrize("record", [{"url": "https://example.com/a.png"}, {"url": "/etc/hosts"}, {"url": "file:///etc/hosts"},
                                    {"data": "!!!not base64"}, {"url": "data:image/png,plain"}, {}])
def test_refused_sources(v, record):
    with pytest.raises(V.VisionError):
        v.plan(record)


def test_garbage_bytes_are_a_400(v):
    with pytest.raises(V.VisionError):
        v.plan({"data": base64.b64encode(b"not an image at all").decode()})


def test_truncated_image_is_refused_on_the_http_side(v):
    raw = png(400, 400, fmt="JPEG")
    with pytest.raises(V.VisionError):
        v.plan({"data": base64.b64encode(raw[:len(raw) // 2]).decode()})


def test_exif_rotation_applies(v):
    img = V.decode_image(png(200, 100, exif_orientation=6))           # stored sideways, shown upright
    assert img.size == (100, 200)


def test_transparent_png_falls_on_white(v):
    img = V.decode_image(png(8, 8, color=(0, 0, 0, 0), mode="RGBA"))
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_plan_memo_is_bounded(v):
    for i in range(V.PLAN_LRU + 4):
        v.plan(rec(png(8, 8, color=(i, 0, 0))))
    assert len(v._plans) == V.PLAN_LRU


def test_render_without_image_is_the_plain_render():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert ds41_chat.render_chat_vl(msgs)[0] == ds41_chat.render_chat(msgs)


def test_family_collects_images_in_prompt_order(v):
    fam = Ds41Family(None, v.model_dir)
    a, b = rec(png(color=(1, 1, 1))), rec(png(color=(2, 2, 2)))
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": a}, {"type": "text", "text": "x"},
                                                      {"type": "image_url", "image_url": b}]}]}
    r = fam.render(body)
    assert isinstance(r, Rendered) and r.prompt.count(ds41_chat.IMAGE_PLACEHOLDER) == 2
    assert r.images == [{"url": a["url"]}, {"url": b["url"]}]


def test_family_without_tower_still_refuses_images(tmp_path):
    fam = Ds41Family(None, tmp_path)
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]}
    with pytest.raises(Exception, match="no vision tower"):
        fam.render(body)


def test_real_tokenizer_keeps_the_placeholder_whole():
    import os
    d = os.path.expanduser("~/ds41/model")
    if not os.path.isfile(os.path.join(d, "tokenizer.json")):
        pytest.skip("no model")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d)
    assert tok.encode("a" + ds41_chat.IMAGE_PLACEHOLDER + "b", add_special_tokens=False).count(IMG) == 1


def test_literal_placeholder_in_text_is_not_an_image():
    from kyojin_mlx.serve.ds41_chat import IMAGE_PLACEHOLDER, render_chat_vl
    msgs = [{"role": "user", "content": [{"type": "text", "text": f"file says {IMAGE_PLACEHOLDER} here"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    prompt, records = render_chat_vl(msgs)
    assert len(records) == 1 and prompt.count(IMAGE_PLACEHOLDER) == 1
    prompt, records = render_chat_vl([{"role": "user", "content": f"only text {IMAGE_PLACEHOLDER}"}])
    assert records == [] and IMAGE_PLACEHOLDER not in prompt
