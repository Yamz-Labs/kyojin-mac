"""Generic image input (flag visiongen): any mlx-vlm model whose tower sits in the served model object.

Same contract as ds41_vision.Vision, and it is that class: the HTTP-thread side (plan / expand / real_ids, the plan and row LRUs, the virtual
ids of an image span) and the engine-thread side (prepare / overlay / touches in ds41_vision) are shared. Only three things are model
specific here and they come from the pack, not from code per family:
- preprocessing: the pack's own processor (preprocessor_config.json / processor_config.json through transformers' AutoProcessor, whose class
  mlx-vlm registers). The number of tokens of an image is the number of image tokens that processor puts in the prompt for it;
- the rows: `model.get_input_embeddings(ids, pixel_values, **processor extras)`, i.e. the model's own tower, projector and merge;
- the prefix-cache key: the virtual ids of ds41_vision, salted with the sha of the processor files and the model_type.

What it refuses (reason in `Vision.reason`, the request gets a 400 that says it) instead of approximating:
- a trunk with multimodal rope (it has `get_rope_index`: Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5 / GLM-4V ...). Its positions over an image are
  3-axis and its decode carries a per-sequence delta; the served step takes neither, so overlaying embeddings would shift every position after the image;
- a model whose text embeddings differ from its trunk's `embed_tokens` (a scaled embedding: the overlay would not match the stock path).
The tower stays resident (it is part of the served model object); nothing is loaded or released per image."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import ds41_vision as DV
from .ds41_vision import VID_BASE, Plan, VisionError

_SALT = b"visiongen-1"
_SKIP = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")
ENV = "KYOJIN_VISIONGEN"


def enabled() -> bool:
    return os.environ.get(ENV, "0") != "0"


@dataclass(frozen=True)
class GenPlan(Plan):
    """A Plan whose span length comes from the processor, not from a (rows x cols) grid."""
    n: int = 0

    @property
    def n_tokens(self) -> int:
        return self.n


def _cfg_get(cfg: dict, *keys):
    for scope in (cfg, cfg.get("text_config") or {}):
        for k in keys:
            if isinstance(scope.get(k), int) and not isinstance(scope.get(k), bool):
                return scope[k]
    return None


def unsupported(model) -> str | None:
    """None when the overlay path is exact for this model class, else why not."""
    import mlx.core as mx
    from .multiuser import _trunk
    if not hasattr(model, "get_input_embeddings") or "language_model" not in model:
        return "the model class has no get_input_embeddings / language_model (not an mlx-vlm model with a tower)"
    lm = model.language_model
    if hasattr(lm, "get_rope_index") or hasattr(lm, "_rope_deltas"):
        return ("the trunk uses multimodal rope (get_rope_index): image positions are 3-axis and decode carries a per-sequence delta, "
                "which the served prefill / step do not carry")
    try:
        ids = mx.array([[1, 2, 3]])
        a = model.get_input_embeddings(ids).inputs_embeds
        b = _trunk(lm).embed_tokens(ids)
        mx.eval(a, b)
        if a.shape != b.shape or not bool(mx.array_equal(a, b).item()):
            return "the model's text embeddings differ from the trunk's embed_tokens (a scaled embedding): the overlay would not match the stock path"
    except Exception as exc:                                              # noqa: BLE001
        return f"the text-embedding probe failed: {type(exc).__name__}: {exc}"
    return None


class GenVision(DV.Vision):
    """One per served model. plan / expand: any HTTP thread. rows: the engine thread only."""

    def __init__(self, model_dir, model):
        self.model_dir = str(model_dir)
        self.model = model
        self._lock = threading.Lock()
        self._plock = threading.Lock()                                    # the processor is not assumed thread safe
        self._plans: OrderedDict = OrderedDict()
        self._rows: OrderedDict = OrderedDict()
        self._tower = None
        self._proc = None
        self.stats = {"encodes": 0, "row_hits": 0, "encode_s": 0.0}
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.config = json.load(f)
        self._tok = _cfg_get(self.config, "image_token_id", "image_token_index")
        self.reason = self._probe_reason()
        h = hashlib.sha256(str(self.config.get("model_type")).encode())
        for name in ("preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
            p = os.path.join(self.model_dir, name)
            h.update(name.encode() + (open(p, "rb").read() if os.path.isfile(p) else b"-"))
        self._digest = h.digest()

    def _probe_reason(self) -> str | None:
        if not (self.config.get("vision_config") or {}):
            return "config.json has no vision_config"
        if self._tok is None:
            return "config.json names no image_token_id / image_token_index"
        if not any(os.path.isfile(os.path.join(self.model_dir, n)) for n in ("preprocessor_config.json", "processor_config.json")):
            return "the pack has no preprocessor_config.json / processor_config.json: the image preprocessing is unknown"
        return unsupported(self.model)

    def available(self) -> bool:
        return self.reason is None

    @property
    def image_token_id(self) -> int:
        return self._tok

    # ------------------------------------------------------------------------------------------------ preprocessing (HTTP thread: PIL and numpy)
    def _processor(self):
        if self._proc is None:
            import mlx_vlm.utils                                          # noqa: F401  (registers the mlx-vlm processor classes with transformers)
            from transformers import AutoProcessor
            self._proc = AutoProcessor.from_pretrained(self.model_dir)
        return self._proc

    def _process(self, img) -> tuple[int, dict]:
        """(image tokens the processor puts in the prompt for this image, its array outputs by name)."""
        with self._plock:
            proc = self._processor()
            tok = getattr(proc, "image_token", None) or proc.tokenizer.convert_ids_to_tokens(self._tok)
            out = proc(images=[img], text=[tok], return_tensors="np")
            n = int((np.asarray(out["input_ids"]) == self._tok).sum())
            extra = {k: np.asarray(v) for k, v in out.items() if k not in _SKIP}
        if n < 1 or "pixel_values" not in extra:
            raise VisionError("the model's processor placed no image tokens / produced no pixel_values for this image")
        return n, extra

    def _new_plan(self, raw: bytes, sha: bytes) -> Plan:
        from PIL import Image, UnidentifiedImageError
        if self.reason is not None:
            raise VisionError(f"image input is not available for this model: {self.reason}")
        data = raw
        try:
            try:
                img = DV.decode_image(data)
            except UnidentifiedImageError:
                data = DV._heic_to_jpeg(raw)
                if data is None:
                    raise
                img = DV.decode_image(data)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise VisionError(f"image cannot be decoded: {exc}") from exc
        try:
            n, _ = self._process(img)
        except VisionError:
            raise
        except Exception as exc:                                          # noqa: BLE001  (a processor that rejects the image: 400, not an engine fault)
            raise VisionError(f"the model's processor cannot take this image: {type(exc).__name__}: {exc}") from exc
        d = hashlib.blake2b(_SALT + self._digest + sha + struct.pack("<I", n), digest_size=32).digest()
        vids = tuple(VID_BASE | (v & (VID_BASE - 1)) for v in struct.unpack("<8I", d))
        return GenPlan(sha, data, 0, 0, vids, n)

    # ------------------------------------------------------------------------------------------------ engine thread
    def _encode(self, plan: Plan):
        """The rows of the span [n, hidden]: the model's own tower + projector + merge on `n` image tokens."""
        import mlx.core as mx
        n, extra = self._process(DV.decode_image(plan.data))
        if n != plan.n:
            raise RuntimeError(f"image token count changed between plan and encode: {plan.n} -> {n}")
        arrs = {k: mx.array(v) for k, v in extra.items()}
        pv = arrs.pop("pixel_values")
        ids = mx.array([[self._tok] * n])
        rows = self.model.get_input_embeddings(ids, pv, **arrs).inputs_embeds[0]
        mx.eval(rows)
        if rows.shape[0] != n:
            raise RuntimeError(f"the model merged {rows.shape[0]} rows for {n} image tokens")
        return rows

    def release(self) -> None:
        """Nothing to hand back: the tower is part of the served model."""
