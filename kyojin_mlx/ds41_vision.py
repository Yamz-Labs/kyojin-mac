"""DS4.1 native vision: the tower of the pack, encoded on demand, and the image spans of a prompt.

The tower (`vision.*`, `aligner.*`, `image_*` in global.safetensors) is not part of the served language model. It loads
at the first image that reaches the prefill and the model math is the reference's (mlx_vlm.models.deepseek_v41).

Two threads, two sides:
- HTTP thread (`plan`, `expand`, `real_ids`): PIL and numpy only, never an mx.array. An image that cannot be decoded is a
  400 here, not an engine fault later.
- Engine thread (`prepare`, `overlay`): encodes, keeps the encoded spans in an LRU, overwrites the embeddings of a chunk.

Prompt ids: in `Session.ids` an image span carries virtual ids >= VID_BASE taken from a digest of the image bytes and its
grid. The prefix cache, the disk checkpoints and the shared-prefill match compare those ids, so two images never share
KV and the same image sent again hits the cache without an encode. The model sees `real_ids` (image_token_id everywhere in
the span, which also sets image_mask and the gate's bias_vl)."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

VID_BASE = 1 << 30
_VID_SALT = b"ds41-vision-1"                    # change it when the preprocessing changes: the disk checkpoints hold the old encoding
PLAN_LRU = 16                                   # images whose bytes and grid stay known (history re-sends them on every turn)
ROWS_LRU = 32                                   # encoded spans, about 11 MB each at the largest grid


class VisionError(ValueError):
    """The image cannot be used: the server answers 400."""


@dataclass(frozen=True)
class Plan:
    """One image, decided on the HTTP thread: its bytes and the span it takes in the prompt."""
    key: bytes
    data: bytes
    n_llm_h: int
    n_llm_w: int
    vids: tuple

    @property
    def n_tokens(self) -> int:
        return (self.n_llm_w + 1) * self.n_llm_h + 2                  # START, rows of image tokens each closed by NEWLINE, END


@dataclass
class Span:
    start: int
    plan: Plan
    vision: "Vision" = field(repr=False)

    @property
    def end(self) -> int:
        return self.start + self.plan.n_tokens


def _record_bytes(record: dict) -> bytes:
    """A content part record -> image bytes. Inline data only: the server fetches nothing and reads no local file."""
    data = record.get("data")
    if data is None:
        url = record.get("url")
        if not isinstance(url, str) or not url:
            raise VisionError("image part without data")
        if not url.startswith("data:"):
            raise VisionError("image must be sent inline (a data: URL or base64 data); this server does not fetch URLs or read files")
        header, _, data = url.partition(",")
        if ";base64" not in header:
            raise VisionError("image data: URL must be base64")
    if isinstance(data, bytes):
        return data
    try:
        return base64.b64decode(data)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise VisionError(f"image data is not valid base64: {exc}") from exc


def _heic_to_jpeg(raw: bytes) -> bytes | None:
    """iPhone photos: HEIC bytes -> JPEG through the macOS `sips`. None when it is not HEIC or `sips` cannot do it."""
    if raw[4:8] != b"ftyp" or sys.platform != "darwin":
        return None
    with tempfile.TemporaryDirectory(prefix="ds41img") as d:
        src, dst = os.path.join(d, "in.heic"), os.path.join(d, "out.jpg")
        with open(src, "wb") as f:
            f.write(raw)
        try:
            r = subprocess.run(["/usr/bin/sips", "-s", "format", "jpeg", "-s", "formatOptions", "best", src, "--out", dst],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode or not os.path.isfile(dst):
            return None
        with open(dst, "rb") as f:
            return f.read()


def decode_image(data: bytes):
    """Bytes -> upright RGB PIL image. EXIF orientation applies (a phone photo is stored sideways), alpha falls on white
    (a transparent screenshot with dark text would turn black on black under a plain convert)."""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as src:
        img = ImageOps.exif_transpose(src)
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(bg, rgba).convert("RGB")
        return img.convert("RGB")


class Vision:
    """One per served model. plan/expand: any HTTP thread. prepare/rows: the engine thread only."""

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self.weights = os.path.join(self.model_dir, "global.safetensors")
        self._lock = threading.Lock()
        self._avail = None
        self._cfg = None
        self._plans: OrderedDict = OrderedDict()
        self._rows: OrderedDict = OrderedDict()
        self._tower = None
        self.stats = {"encodes": 0, "row_hits": 0, "encode_s": 0.0}

    def available(self) -> bool:
        """Header check only: the config names a tower and the pack holds it."""
        if self._avail is None:
            self._avail = self._probe()
        return self._avail

    def _probe(self) -> bool:
        try:
            with open(os.path.join(self.model_dir, "config.json")) as f:
                if not (json.load(f).get("vision_config") or {}).get("num_hidden_layers"):
                    return False
            with open(self.weights, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            return all(k in header for k in ("vision.patch_embed.proj.weight", "aligner.w1.weight", "image_start", "image_newline", "image_end"))
        except (OSError, ValueError, struct.error):
            return False

    @property
    def cfg(self):
        if self._cfg is None:
            from mlx_vlm.models.deepseek_v41.config import ModelConfig
            with open(os.path.join(self.model_dir, "config.json")) as f:
                self._cfg = ModelConfig.from_dict(json.load(f))
        return self._cfg

    @property
    def image_token_id(self) -> int:
        return self.cfg.image_token_id

    # ------------------------------------------------------------------------------------------------ HTTP thread
    def plan(self, record: dict) -> Plan:
        raw = _record_bytes(record)
        sha = hashlib.sha256(raw).digest()
        with self._lock:
            p = self._plans.get(sha)
            if p is not None:
                self._plans.move_to_end(sha)
                return p
        p = self._new_plan(raw, sha)
        with self._lock:
            self._plans[sha] = p
            while len(self._plans) > PLAN_LRU:
                self._plans.popitem(last=False)
        return p

    def _new_plan(self, raw: bytes, sha: bytes) -> Plan:
        from PIL import Image, UnidentifiedImageError
        from mlx_vlm.models.deepseek_v41.processing_deepseek_v41 import plan_image_grid
        data = raw
        try:
            try:
                img = decode_image(data)                              # the whole image, once: the engine thread can trust these bytes
            except UnidentifiedImageError:
                data = _heic_to_jpeg(raw)
                if data is None:
                    raise
                img = decode_image(data)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise VisionError(f"image cannot be decoded: {exc}") from exc
        n_llm_h, n_llm_w, _, _ = plan_image_grid(img.width, img.height, self.cfg)
        d = hashlib.blake2b(_VID_SALT + sha + struct.pack("<II", n_llm_h, n_llm_w), digest_size=32).digest()
        vids = tuple(VID_BASE | (v & (VID_BASE - 1)) for v in struct.unpack("<8I", d))
        return Plan(sha, data, n_llm_h, n_llm_w, vids)

    def expand(self, toks: list, plans: list, whole: bool = True) -> tuple[list, list]:
        """Each image token of `toks` (one per placeholder) becomes its span: (ids with virtual ids, spans). whole: the
        text holds every image of `plans` (the prompt). A cut text is a prefix of the prompt and may hold fewer."""
        tid = self.image_token_id
        out, spans, k, at = [], [], 0, 0
        while True:
            try:
                i = toks.index(tid, at)
            except ValueError:
                break
            if k >= len(plans):
                raise VisionError("more image placeholders than images")
            out.extend(toks[at:i])
            p = plans[k]
            spans.append(Span(len(out), p, self))
            out.extend(p.vids[j % len(p.vids)] for j in range(p.n_tokens))
            k, at = k + 1, i + 1
        out.extend(toks[at:])
        if whole and k != len(plans):
            raise VisionError(f"{len(plans)} images but {k} placeholders in the prompt")
        return out, spans

    def real_ids(self, ids: list) -> list:
        tid = self.image_token_id
        return [tid if t >= VID_BASE else t for t in ids]

    # ------------------------------------------------------------------------------------------------ engine thread
    def _load(self):
        if self._tower is None:
            import mlx.core as mx
            from mlx_vlm.models.deepseek_v41.vision import Aligner, ViT
            cfg = self.cfg
            vit, aligner = ViT(cfg), Aligner(cfg)                     # the random init is lazy and replaced before it is ever evaluated
            w = mx.load(self.weights)
            vit.load_weights([(k[len("vision."):], v) for k, v in w.items() if k.startswith("vision.")], strict=True)
            aligner.load_weights([(k[len("aligner."):], v) for k, v in w.items() if k.startswith("aligner.")], strict=True)
            vit.eval(), aligner.eval()
            marks = w["image_start"], w["image_newline"], w["image_end"]
            mx.eval(vit.parameters(), aligner.parameters(), *marks)
            self._tower = vit, aligner, marks
        return self._tower

    def release(self) -> None:
        """Hand the tower's memory back. The weights reload from the page cache at the next new image."""
        if self._tower is not None:
            import mlx.core as mx
            self._tower = None
            mx.clear_cache()

    def _encode(self, plan: Plan):
        """All the rows of the span [n_tokens, hidden]: START, then each grid row + NEWLINE, then END."""
        import mlx.core as mx
        from mlx_vlm.models.deepseek_v4.vision import get_vision_cos_sin
        from mlx_vlm.models.deepseek_v41.processing_deepseek_v41 import load_image
        vit, aligner, (start, newline, end) = self._load()
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(decode_image(plan.data), self.cfg)
        if (n_llm_h, n_llm_w) != (plan.n_llm_h, plan.n_llm_w):
            raise RuntimeError(f"image grid changed between plan and encode: {(plan.n_llm_h, plan.n_llm_w)} -> {(n_llm_h, n_llm_w)}")
        x = vit.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, vit.rope_dim, vit.rope_theta)
        for block in vit.blocks:                                      # one eval per block: the attention matrix of one block at a time
            x = block(x, cos, sin)
            mx.eval(x)
        x = aligner(vit.norm(x), n_vit_h, n_vit_w)
        parts = [start[None]]
        for r in range(n_llm_h):
            parts += [x[r * n_llm_w:(r + 1) * n_llm_w], newline[None]]
        parts.append(end[None])
        rows = mx.concatenate(parts, axis=0).astype(start.dtype)
        mx.eval(rows)
        return rows

    def rows(self, plan: Plan):
        import time
        import mlx.core as mx
        r = self._rows.get(plan.key)
        if r is not None:
            self._rows.move_to_end(plan.key)
            self.stats["row_hits"] += 1
            return r
        t0 = time.perf_counter()
        r = self._encode(plan)
        self._rows[plan.key] = r
        while len(self._rows) > ROWS_LRU:
            self._rows.popitem(last=False)
        mx.clear_cache()
        self.stats["encodes"] += 1
        self.stats["encode_s"] += time.perf_counter() - t0
        return r


def prepare(spans: list, a: int, b: int) -> int:
    """Engine thread, before the chunk's clock starts: the spans that touch prompt positions [a, b) are encoded (returns how
    many were new). The tower leaves memory as soon as the images of this chunk are done."""
    new, used = 0, []
    for s in spans:
        if s.start < b and s.end > a:
            n0 = s.vision.stats["encodes"]
            s.vision.rows(s.plan)
            new += s.vision.stats["encodes"] - n0
            used.append(s.vision)
    for v in used:
        v.release()
    return new


def overlay(h, spans: list, a: int):
    """Write the image rows into the embeddings h [1, L, hidden] of prompt positions [a, a + L). A chunk may cut a span."""
    b = a + h.shape[1]
    for s in spans:
        i, j = max(a, s.start), min(b, s.end)
        if i < j:
            h[:, i - a:j - a] = s.vision.rows(s.plan)[i - s.start:j - s.start].astype(h.dtype)
    return h


def touches(spans: list, a: int, b: int) -> bool:
    return any(s.start < b and s.end > a for s in spans)
