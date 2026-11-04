# SPDX-License-Identifier: Apache-2.0
"""Prefill layer bound: evaluate the layer output and the cache state every `every` layers of a prompt chunk.

A prompt chunk builds one lazy graph over every layer of the model: nothing in it is evaluated until the logits are asked for, so the transients of all N layers
(matmul products, the per-layer score matrices of the indexer, the compressor state of every layer) are alive at the same time, and the peak of a 40-layer model is
the sum over its layers instead of one layer's worth. Evaluating every `every` layers (default 4) caps what the graph holds at `every` layers: the transients of the
layers before the bound are dead before the next layer is built.

Evaluation order changes no arithmetic - every value is computed by the same op on the same inputs - so the logits are bit-identical; the cache buffers are read and
written in the same order, only with their contents materialised in between (which is what `DeepseekV41Cache.state` is documented for: "evaluates this between
prefill chunks to keep a long prompt from becoming one command buffer").

Decode steps (fewer than `KYOJIN_BOUND_MIN_ROWS` rows) reset the phase and are left alone: one row has no transient worth bounding and the extra syncs would only
cost latency.

Decode submit (KYOJIN_BOUND_DEC_EVERY=n, default 2, 0 = off): a decode step (or an MTP verify) submits its graph with mx.async_eval every n layers instead of
once at the end, so the GPU runs the first layers while the host builds the next ones. No sync and no arithmetic change: same logits.

Every layer class, not only `type(layers[0])`: CED (`ced._install_dsv4`) and the drafter taps (`spec.install_taps`) give some layer instances their own subclass
with its own `__call__`, so a bound on the base class alone leaves those layers with no eval group and no YIELD point. `install` wraps every class the layer list
uses, and CED / the taps call `rewrap` for the classes they add (the served order installs this bound first, in `patch_model`). The closure binds `orig` per
class: a loop that reads `cls.__call__` into a variable shared by the wrappers makes every class call the LAST class's `__call__` (a CED decoder layer), and an
encoder layer then runs `_block_call` with the chunk's full `h` but the tail-sliced `image_mask` (`broadcast_shapes (1,n,experts)` against
`(1,tail,experts)`). `_depth` keeps one count per layer call when the wrappers stack (an older one is reached through `orig`): only the outermost bounds.

Decoder half (KYOJIN_BOUND_DEC=1, default 0 = today): CED gives the layers from the midpoint on only the `tail` rows of a chunk, so their `h` reads as a decode
step to the row test, and the midpoint never calls the wrapped block (it runs `_block_mid`), so it is never counted. When on, the layers whose class is not
`type(layers[0])` are bounded too, the midpoint included, and the rows of a call come from the chunk's CED context (`cache[0]._ced.n`) instead of `h`. The
outermost eval and YIELD point are then outside `_block_call` (after CED restored `cache.offset`), where the layer output and the cache state are materialised
exactly as the vendor materialises them between chunks: same values, same bits. 0 leaves those classes as pass-throughs and lets the older inner wrapper (the
base class, today the only one) count with the row test, so the bound count, the phase, the eval points and the counters are exactly today's.
On, it applies only while the engine has a YIELD hook armed for the chunk (another session decodes): solo chunks lost 10 % at 8K to the extra evals.

Switch: KYOJIN_BOUND_LAYERS=0; KYOJIN_BOUND_EVERY (default 4); KYOJIN_BOUND_MIN_ROWS (default 64); KYOJIN_BOUND_DEC_EVERY (default 2); KYOJIN_BOUND_DEC (default 0);
KYOJIN_TTFT_FLOOR=1 (flag ttft_floor) with KYOJIN_TTFT_FLOOR_ROWS (default 256) and KYOJIN_TTFT_FLOOR_BOUND (default 16): a chunk of at most that many rows
bounds every TTFT_FLOOR_BOUND layers instead of EVERY."""
from __future__ import annotations

import os

import mlx.core as mx
from . import fired as FD

ENABLED = os.environ.get("KYOJIN_BOUND_LAYERS", "1") != "0"
EVERY = int(os.environ.get("KYOJIN_BOUND_EVERY", "4"))
MIN_ROWS = int(os.environ.get("KYOJIN_BOUND_MIN_ROWS", "64"))
DEC_EVERY = int(os.environ.get("KYOJIN_BOUND_DEC_EVERY", "2"))
DEC = os.environ.get("KYOJIN_BOUND_DEC", "1") != "0"          # flag bound_dec: also bound the layers whose class is not type(layers[0]) (CED decoder half, tapped layers)
# flag ttft_floor: a short prompt chunk (a tool round, at most TTFT_ROWS rows) takes every TTFT_BOUND_EVERY layers instead of EVERY. The bound exists to
# cap the transients of a chunk whose activations grow with its rows (a 8192-row chunk holds tens of GiB without it); at a few hundred rows the whole
# chunk's transients are small, and each bound is a sync where the host waits for the GPU before it builds the next group of layers. Setting the stride
# higher (or 0 = EVERY, today) changes no arithmetic: only the point where the graph is materialised (see the module docstring).
TTFT_FLOOR = os.environ.get("KYOJIN_TTFT_FLOOR", "0") != "0"
TTFT_ROWS = int(os.environ.get("KYOJIN_TTFT_FLOOR_ROWS", "256"))
TTFT_BOUND_EVERY = int(os.environ.get("KYOJIN_TTFT_FLOOR_BOUND", "16"))

_calls = [0]          # block calls since the last bound (a prompt chunk starts at 0)
_bounds = [0]         # bounds taken, for the tests and the load report
_dcalls = [0]         # decode block calls (any phase: the submit only moves the point where the GPU starts)
_submits = [0]        # decode submits taken
_wrapped = {}         # class -> its __call__ before this install (uninstall puts it back; tests reuse the process-wide vendor classes)
_depth = [0]          # block wrappers on the stack: a class wrapper installed after another one (CED / taps run after `install`) reaches the older wrapper
                      # through `orig`; only the outermost one counts the layer call (otherwise a decoder layer is bounded twice)
YIELD = [None]        # flag mix_slice_ms: a callable the engine installs for one prompt chunk; called right after a bound's mx.eval (layer outputs and cache state are
                      # materialised there, nothing of this chunk is in flight), where the engine may run a decode round of other sessions. Not used by a decode step.


def stats() -> dict:
    return dict(calls=_calls[0], bounds=_bounds[0], submits=_submits[0])


def reset() -> None:
    _calls[0] = 0
    _bounds[0] = 0
    _dcalls[0] = 0
    _submits[0] = 0


def _arrays(v) -> list:
    if isinstance(v, mx.array):
        return [v]
    if isinstance(v, (tuple, list)):
        return [a for x in v for a in _arrays(x)]
    return []


def _state(cache) -> list:
    """Every buffer of the cache, when it can name them (`state` on the mlx-vlm DS4.1 cache and on the mlx-lm caches)."""
    out = []
    for c in (cache if isinstance(cache, (list, tuple)) else [cache]):
        st = getattr(c, "state", None)
        if st is None:
            continue
        try:
            out.extend(_arrays(st))
        except Exception:                                            # a cache whose state is not a plain array list: leave it to the caller
            pass
    return out


def _rows(args, kwargs) -> int:
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, mx.array):
            if v.ndim >= 3:
                return int(v.shape[1])
            if v.ndim:
                return int(v.shape[0])
    return 0


def _cache(args, kwargs):
    return args[-1] if args else kwargs.get("cache")


def _chunk_rows(args, kwargs) -> int:
    """Rows of the chunk a block call belongs to. A CED chunk carries only its `tail` rows in the decoder half (`_block_mid` cut them), which `_rows` would read
    as a decode step; the chunk's own row count is on the cache entry's CED context. Outside a CED chunk this is exactly `_rows`."""
    cache = _cache(args, kwargs)
    entry = cache[0] if isinstance(cache, (list, tuple)) and cache else cache
    ctx = getattr(entry, "_ced", None)
    return int(ctx.n) if ctx is not None else _rows(args, kwargs)


def _layer_classes(layers) -> list:
    """The distinct classes of the layer list, in layer order. CED and the drafter taps replace an instance's class, so layers[1:] can name more classes."""
    seen, out = set(), []
    for layer in layers:
        cls = type(layer)
        if cls not in seen:
            seen.add(cls)
            out.append(cls)
    return out


def _wrap(cls, dec: bool) -> bool:
    """Wrap one layer class. `dec`: a class other than type(layers[0]) (CED decoder half / tapped layer): its work is only there when flag bound_dec is on."""
    if cls.__dict__.get("_kyojin_bound", False):
        return False
    orig = cls.__call__

    def __call__(self, *args, _orig=orig, _dec=dec, **kwargs):      # bind per class: a shared `orig` would make every class run the last class's call
        dec_on = DEC and YIELD[0] is not None                        # the decoder-half bounds exist to serve a yield: a solo chunk (no hook armed) pays nothing
        if _depth[0]:                                                # an outer wrapper owns this layer call (see `_depth`)
            if dec_on or not ENABLED:
                return _orig(self, *args, **kwargs)
            out = _orig(self, *args, **kwargs)                       # bound_dec off: the inner wrapper is today's only one, it counts as it did
        else:
            _depth[0] = 1
            try:
                out = _orig(self, *args, **kwargs)
            finally:
                _depth[0] = 0
            if not ENABLED or (_dec and not dec_on):                    # read per call: KYOJIN_BOUND_LAYERS / KYOJIN_BOUND_DEC switch on a loaded model
                return out
        rows = _chunk_rows(args, kwargs) if dec_on else _rows(args, kwargs)
        if rows < MIN_ROWS:                                          # a decode step (or a tiny chunk): no graph to bound, keep the phase at layer 0
            _calls[0] = 0
            if DEC_EVERY > 0:
                _dcalls[0] += 1
                if _dcalls[0] % DEC_EVERY == 0:
                    arrs = _arrays(out)
                    if arrs:
                        try:
                            mx.async_eval(*arrs)
                            _submits[0] += 1
                        except Exception:                            # a compiled trace: nothing to submit
                            pass
            return out
        _calls[0] += 1
        every = EVERY
        if TTFT_FLOOR and TTFT_BOUND_EVERY > EVERY and rows <= TTFT_ROWS:      # a short chunk: one eval per TTFT_BOUND_EVERY layers, not EVERY
            every = TTFT_BOUND_EVERY
        if _calls[0] % every:
            return out
        _calls[0] = 0
        _bounds[0] += 1
        arrs = _arrays(out)
        arrs.extend(_state(_cache(args, kwargs)))
        if arrs:
            mx.eval(*arrs)
            FD.hit("bound_layers")
            FD.hit("bound_every")
        y = YIELD[0]
        if y is not None:
            y()
            _calls[0] = 0                                            # the decode rounds in between reset the phase; the next bound is `EVERY` layers away
        return out

    cls.__call__ = __call__
    cls._kyojin_bound = True
    _wrapped[cls] = orig
    return True


def install(model) -> int:
    """Bound the layer loop of `model.language_model` (or `model.model`): returns the number of blocks wrapped, 0 when the model has no layer list.
    Idempotent: a second call on a model of the same classes is a no-op."""
    lm = getattr(model, "language_model", None) or getattr(model, "model", None) or model
    layers = getattr(lm, "layers", None)
    if not layers:
        return 0
    base = type(layers[0])
    wrapped = sum(_wrap(cls, cls is not base) for cls in _layer_classes(layers))
    return len(layers) if wrapped else 0


def rewrap(model) -> int:
    """Wrap the layer classes that appeared after `install`: CED (`ced._install_dsv4`) and the drafter taps (`spec.install_taps`) subclass layer instances, and
    the served order arms them after `patch_model` (which installs this bound). No-op on a model `install` never touched. Returns the classes wrapped."""
    lm = getattr(model, "language_model", None) or getattr(model, "model", None) or model
    layers = getattr(lm, "layers", None)
    if not layers or not type(layers[0]).__dict__.get("_kyojin_bound", False):
        return 0
    base = type(layers[0])
    return sum(_wrap(cls, cls is not base) for cls in _layer_classes(layers))


def uninstall() -> int:
    """Put back every class `install` wrapped (the vendor layer class is process-wide: tests that build a second model restore it instead of stacking the
    wrappers; a served process installs once and never calls this). Returns the number of classes restored."""
    n = 0
    for cls, orig in list(_wrapped.items()):
        if cls.__dict__.get("_kyojin_bound", False):
            cls.__call__ = orig
            delattr(cls, "_kyojin_bound")
            n += 1
        del _wrapped[cls]
    _calls[0] = 0
    _depth[0] = 0
    return n
