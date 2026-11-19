#!/usr/bin/env python3
"""opprof: generic per-op profiler for ANY model the Kyojin backends serve (GLM-5.3, DS4.1, Qwen3.x, any mlx-lm class on the generic backend).

One load, then per op (module key shared by all layers of one kind) and per row count R:
  decode (R = 1, 2, 4 ... sessions in one batched step, context --ctx), prefill (--prefill tokens in --chunk chunks):
  ms, bytes read, GB/s and % of the MEASURED bandwidth B, TFLOPS, launches per op / per token, host graph-build time,
  ranked by gap to physics:  gap = self ms - ideal ms,  ideal = max(bytes / B, flops / measured fp16 GEMM peak).

Arms (all in the loaded process, nothing in the engine is changed; class-level hooks are removed at the end):
  bw      pure stream B (sum over a 4 / 8 GiB uint32 array, the same kernel as G0 arm a / BW_MEASURED.md) + a fp16 GEMM peak (measured TFLOPS)
  decode  per R: pipelined step ms, host round trip, host graph-build ms, differential ablation per op (op replaced by constants of its recorded
          shape that still depend on its inputs: the unsynced pipelined step with and without the op = its true cost in the real graph), measured
          expert union, synced cross-check, launches (graph census, per op by an exact node count), host build us per op
  prefill unprofiled tok/s, then a synced per-op profile (excl / incl ms, calls, bytes, TFLOPS), expert union, launches of one chunk

Op keys are derived from module paths: "<kind>:<path inside the layer>" ("lin+moe:mlp.switch_mlp"), "layer.<kind>" (glue of the layer itself),
"top:<path>" (everything outside the layer list). <kind> is built from the classes of the layer's mixer and ffn child through the small tables
below (SHORT, MIXER_RE, FFN_RE); nothing else is model specific.

Library use (a backend already loaded, e.g. inside a server process):  report = opprof.profile(be, cfg, tok, Opts(...)); opprof.write_report(report, out)
CLI:  python tools/opprof.py --model DIR --decode-rows 1,2,4 --ctx 8192 --prefill 8192 --out DIR     (docs/OPPROF.md)
"""
import argparse
import collections
import contextlib
import fcntl
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import mlx.core as mx
import mlx.nn as nn

B_REF = 1159.0                      # GB/s, /path/to/data (M5 Ultra); the in-process arm measures its own
GATE = "/tmp/gpu_gate.lock"
VIEWS = {"Reshape", "Transpose", "Broadcast", "Flatten", "Unflatten", "Squeeze", "ExpandDims", "Split", "Slice", "AsStrided", "StopGradient", "Depends"}
INT_DTYPES = (mx.uint8, mx.uint16, mx.uint32, mx.uint64, mx.int8, mx.int16, mx.int32, mx.int64)

# ---------------------------------------------------------------------------------------------- the small model tables
# class name -> short label of a layer kind (first match wins; unknown classes keep their lower-cased name)
SHORT = [(r"DeltaNet|LinearAttn|LinearAttention|GatedDelta", "gdn"), (r"Mamba|SSM|Mixer", "ssm"), (r"DSA|Indexer|Sparse.?Attn", "dsa"),
         (r"Attention|Attn", "attn"), (r"MoE|Moe|Switch|Expert", "moe"), (r"MLP|Mlp|FeedForward|FFN|Ffn", "mlp")]
MIXER_RE = re.compile(r"attn|attention|mixer|ssm|delta|mamba|conv|rwkv", re.I)       # child of a layer that mixes tokens (reads the KV / state cache)
NORM_RE = re.compile(r"norm|(^|_)ln", re.I)                                              # a norm is never the mixer
FFN_RE = re.compile(r"mlp|moe|ffn|feed_forward|block_sparse|expert", re.I)
ROUTER_LEAF = ("gate", "router")                                                      # last path component of an expert router
EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts", "n_experts")
TOPK_KEYS = ("num_experts_per_tok", "moe_top_k", "top_k", "num_selected_experts", "n_activated_experts")
PACK_FIELDS = ("w", "qs", "meta", "gs", "hi", "tl")                                  # arrays of a device weight pack (foreign_linear.DevPack): what the kernel reads
SUBOPS = [("mx", "gather_mm"), ("mx", "gather_qmm"), ("mx", "quantized_matmul"), ("mx", "argsort"), ("mx.fast", "scaled_dot_product_attention")]


def log(*a):
    print("[opprof]", *a, file=sys.stderr, flush=True)


def med(x):
    x = sorted(x)
    return x[len(x) // 2]


def flat(y):
    if isinstance(y, mx.array):
        return [y]
    if isinstance(y, (tuple, list)):
        return [a for v in y for a in flat(v)]
    if isinstance(y, dict):
        return [a for v in y.values() for a in flat(v)]
    return []


def _spec_of(y):
    if isinstance(y, mx.array):
        return ("a", tuple(y.shape), y.dtype)
    if isinstance(y, (tuple, list)):
        return ("t" if isinstance(y, tuple) else "l", tuple(_spec_of(v) for v in y))
    return ("o", y)


def _rebuild(struct, it):
    if isinstance(struct, mx.array):
        return next(it)
    if isinstance(struct, (tuple, list)):
        out = [_rebuild(v, it) for v in struct]
        return tuple(out) if isinstance(struct, tuple) else out
    return struct


def with_deps(built, deps):
    """the constants of a stubbed module that still depend on its inputs: the upstream work is evaluated, only the module's own compute is gone"""
    outs = flat(built)
    if not outs or not deps:
        return built
    return _rebuild(built, iter(mx.depends(outs, deps)))


# ---------------------------------------------------------------------------------------------- GPU gate
class Gate:
    """flock on the shared GPU gate (RULES_GPU_MERGE.md rule 1) around every timed part. checkpoint() hands the gate on after `hold_max` seconds."""

    def __init__(self, path=GATE, hold_max=540.0, say=log):
        self.path, self.hold_max, self.say, self.fh, self.t0, self.waited = path, hold_max, say, None, 0.0, 0.0

    def acquire(self):
        if self.fh is not None:
            return
        self.fh = open(self.path, "a")
        t = time.time()
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.say(f"waiting for {self.path} ...")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.t0 = time.time()
        self.waited += self.t0 - t

    def release(self):
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None

    def checkpoint(self):
        """True when the gate was released and re-taken (another job may have run in between: re-measure the base)"""
        if self.fh is not None and time.time() - self.t0 > self.hold_max:
            self.release()
            self.acquire()
            return True
        return False

    @contextlib.contextmanager
    def hold(self):
        self.acquire()
        try:
            yield self
        finally:
            self.release()


class NullGate(Gate):
    def __init__(self):
        super().__init__()

    def acquire(self):
        pass

    def release(self):
        pass

    def checkpoint(self):
        return False


# ---------------------------------------------------------------------------------------------- model introspection
def _cfg_text(cfg):
    cfg = cfg or {}
    return cfg.get("text_config") or cfg


def _first(d, keys, default=None):
    for k in keys:
        if isinstance(d.get(k), int):
            return d[k]
    return default


def is_pack(v):
    return (not isinstance(v, (nn.Module, mx.array, list, tuple, dict, str, int, float, bool, type(None))) and hasattr(v, "n") and hasattr(v, "k")
            and any(isinstance(getattr(v, f, None), mx.array) for f in PACK_FIELDS))


def pack_bytes(p):
    return sum(getattr(p, f).nbytes for f in PACK_FIELDS if isinstance(getattr(p, f, None), mx.array))


def own_parts(m):
    """(arrays, packs) held directly by a module (children are counted under their own key)"""
    seen, arrays, packs = set(), [], []
    for src in (dict.items(m) if isinstance(m, dict) else (), vars(m).items()):
        for name, v in list(src):
            if id(v) in seen:
                continue
            seen.add(id(v))
            if isinstance(v, mx.array):
                arrays.append((name, v))
            elif is_pack(v):
                packs.append((name, v))
    return arrays, packs


def shared_packs(m):
    """packs a module reaches through a helper object (foreign_linear.GroupMember -> _Group.dev): one pack, read by the group's first called member"""
    out = []
    for name, v in vars(m).items():
        if not isinstance(v, (nn.Module, mx.array, list, tuple, dict, str, int, float, bool, type(None))) and is_pack(getattr(v, "dev", None)):
            out.append((name, v.dev))
    return out


def module_bytes(m, n_experts):
    """(fixed bytes, expert-stacked bytes, E): the stacked part is read per ACTIVE expert (bytes / E each), the rest every call"""
    arrays, packs = own_parts(m)
    fixed = stacked = 0
    E = None
    for _, a in arrays:
        if n_experts and a.ndim >= 3 and a.shape[0] == n_experts:
            stacked += a.nbytes
            E = n_experts
        else:
            fixed += a.nbytes
    for _, p in packs:
        if getattr(p, "E", None):
            stacked += pack_bytes(p)
            E = int(p.E)
        else:
            fixed += pack_bytes(p)
    return fixed, stacked, E


def module_flops_per_row(m, top_k):
    """matmul flops of one row through the module (experts: x top_k); None when it holds no weight matrix"""
    arrays, packs = own_parts(m)
    tot, found = 0.0, False
    for _, p in packs:
        tot += 2.0 * p.n * p.k * (top_k if getattr(p, "E", None) else 1)
        found = True
    if packs:
        return tot
    for name, a in arrays:
        if name != "weight" or a.ndim < 2 or type(m).__name__.endswith("Embedding"):
            continue
        bits = getattr(m, "bits", None)
        k = a.shape[-1] * 32 // bits if (bits and "scales" in m) else a.shape[-1]
        tot += 2.0 * a.shape[-2] * k * (top_k if a.ndim >= 3 else 1)
        found = True
    return tot if found else None


def short_label(cls):
    for rx, s in SHORT:
        if re.search(rx, cls):
            return s
    return cls.lower()


def layer_kind(layer):
    kids = {n: v for n, v in dict.items(layer) if isinstance(v, nn.Module)} if isinstance(layer, dict) else {}
    mixer = next((v for n, v in kids.items() if MIXER_RE.search(n) and not NORM_RE.search(n)), None)
    ffn = next((v for n, v in kids.items() if FFN_RE.search(n)), None)
    parts = [short_label(type(x).__name__) for x in (mixer, ffn) if x is not None]
    return "+".join(parts) or short_label(type(layer).__name__)


def find_layers(model):
    """(path of the layer list, [layer modules]): the longest list of modules named `layers` anywhere in the model (the trunk, not an MTP head)"""
    best = ("", [])
    for path, m in model.named_modules():
        ls = dict.get(m, "layers") if isinstance(m, dict) else None
        if isinstance(ls, list) and ls and all(isinstance(x, nn.Module) for x in ls) and len(ls) > len(best[1]):
            best = (f"{path}.layers" if path else "layers", ls)
    return best


@dataclass
class Op:
    key: str
    kind: str                       # layer kind label, "top" for modules outside the layer list
    path: str                       # path inside the layer ("" = the layer itself)
    meth: str = "__call__"
    parent: str | None = None
    mods: list = field(default_factory=list)        # [(module, layer index or -1)]
    fixed: float = 0.0              # bytes read per call, summed over the instances
    stacked: float = 0.0            # expert-stacked bytes (all E), summed over the instances
    E: int | None = None
    flops_row: float | None = None  # matmul flops per row, summed over the instances
    gather: bool = False            # embedding table: one row per token
    table_rows: int = 1
    shared: dict = field(default_factory=dict)      # id(pack) -> (bytes, flops per row) of packs shared with other ops (read by the first one called)
    trunk: bool = False             # an ancestor of the layer list (its call is the whole forward): never ablated
    router: bool = False
    mixer: bool = False

    @property
    def layers(self):
        return len({li for _, li in self.mods})

    @property
    def expert(self):
        return self.E is not None


class Registry:
    def __init__(self, model, cfg=None):
        t = _cfg_text(cfg)
        self.n_experts = _first(t, EXPERT_COUNT_KEYS)
        self.top_k = _first(t, TOPK_KEYS, 1 if not self.n_experts else 8)
        self.vocab = _first(t, ("vocab_size",))
        self.ops, self.by_id, self.layers = collections.OrderedDict(), {}, []
        self.pack_refs, self.owner = {}, {}
        lpath, layers = find_layers(model)
        self.lpath = lpath
        sigs = {}
        for li, L in enumerate(layers):
            sig = tuple((p, type(m).__name__) for p, m in L.named_modules())
            if sig not in sigs:
                base = layer_kind(L)
                n = sum(1 for lab in sigs.values() if lab.split("#")[0] == base)
                sigs[sig] = base if n == 0 else f"{base}#{n + 1}"
            self.layers.append(sigs[sig])
        for li, L in enumerate(layers):
            kind = self.layers[li]
            for path, m in L.named_modules():
                key = f"layer.{kind}" if path == "" else f"{kind}:{path}"
                parent = None if path == "" else (f"{kind}:{path.rsplit('.', 1)[0]}" if "." in path else f"layer.{kind}")
                self._add(key, kind, path, m, li, parent)
        under = lpath + "."
        for path, m in model.named_modules():
            if path == "" or path == lpath or path.startswith(under):
                continue
            parent = f"top:{path.rsplit('.', 1)[0]}" if "." in path else None
            self._add(f"top:{path}", "top", path, m, -1, parent)
        for op in list(self.ops.values()):            # tied lm_head: Embedding.as_linear reads the whole table
            m0 = op.mods[0][0]
            if op.kind == "top" and type(m0).__name__.endswith("Embedding") and hasattr(type(m0), "as_linear"):
                self._add(op.key + ".as_linear", "top", op.path + ".as_linear", m0, -1, op.parent, meth="as_linear")
        for op in self.ops.values():
            if op.parent not in self.ops:
                op.parent = None
            op.trunk = op.kind == "top" and (lpath == op.path or lpath.startswith(op.path + "."))
            op.mixer = (op.kind != "top" and bool(op.path) and "." not in op.path and bool(MIXER_RE.search(op.path)) and not NORM_RE.search(op.path))

    def _add(self, key, kind, path, m, li, parent, meth="__call__"):
        op = self.ops.get(key)
        if op is None:
            op = self.ops[key] = Op(key, kind, path, meth, parent)
            op.router = path.rsplit(".", 1)[-1] in ROUTER_LEAF
        op.mods.append((m, li))
        self.by_id[(id(m), meth)] = (key, li)
        fixed, stacked, E = module_bytes(m, self.n_experts)
        fl = module_flops_per_row(m, self.top_k)
        for _, p in shared_packs(m):
            op.shared[id(p)] = (pack_bytes(p), 2.0 * p.n * p.k)
            self.pack_refs.setdefault(id(p), [])
            if key not in self.pack_refs[id(p)]:
                self.pack_refs[id(p)].append(key)
            self.owner.setdefault(id(p), key)
        if meth == "as_linear":                       # the head matmul through the embedding table
            w = dict.get(m, "weight")
            fixed, stacked, E, fl = (w.nbytes if w is not None else 0), 0, None, (2.0 * w.shape[0] * w.shape[1] if w is not None else None)
        op.fixed += fixed
        op.stacked += stacked
        op.E = op.E or E
        if fl is not None:
            op.flops_row = (op.flops_row or 0.0) + fl
        if meth == "__call__" and type(m).__name__.endswith("Embedding") and dict.get(m, "weight") is not None:
            op.gather, op.table_rows = True, dict.get(m, "weight").shape[0]

    def settle(self, first_call):
        """give each shared pack (a fused same-input group) to the member that runs first in a forward: its self time carries the group's bytes and flops"""
        for pid, keys in self.pack_refs.items():
            called = [k for k in keys if k in first_call]
            self.owner[pid] = min(called, key=lambda k: first_call[k]) if called else keys[0]

    def shared_totals(self, op):
        b = f = 0.0
        for pid, (nb, fl) in op.shared.items():
            if self.owner.get(pid) == op.key:
                b += nb
                f += fl
        return b, f

    def classes(self):
        """{(class, method)} to hook"""
        return {(type(m), op.meth) for op in self.ops.values() for m, _ in op.mods}


def expected_union(E, K, R):
    """expected distinct experts of R rows that each pick K of E uniformly (the no-measurement model; opprof measures it from the real routing)"""
    return E * (1.0 - (1.0 - K / E) ** R)


def op_bytes(op, rows, reg, distinct=None):
    """bytes read per call-step by all layer instances of the op, `rows` rows in the call. experts: bytes of the distinct experts touched"""
    if op.gather:
        return op.fixed * rows / max(1, op.table_rows)
    sh = reg.shared_totals(op)[0] if op.shared else 0.0
    if op.expert:
        d = distinct if distinct is not None else expected_union(op.E, min(reg.top_k, op.E), rows)
        return op.fixed + sh + op.stacked / op.E * min(d, op.E)
    return op.fixed + sh


# ---------------------------------------------------------------------------------------------- graph census
_NODE = re.compile(r'(\d+) \[label ?= ?"([^"]*)", shape=rectangle\]')


def graph_census(arrays):
    """dispatch census of the lazy graph ending in `arrays`: {primitive label: count} without the view primitives (each node = one launch)"""
    arrays = [a for a in flat(list(arrays)) if isinstance(a, mx.array)]
    if not arrays:
        return collections.Counter()
    f = io.StringIO()
    mx.export_to_dot(f, *arrays)
    nodes = {m[1]: m[2] for m in _NODE.finditer(f.getvalue())}
    return collections.Counter(v for v in nodes.values() if v not in VIEWS)


# ---------------------------------------------------------------------------------------------- hooks
class Hooks:
    """Class-level hooks; mode None = pass-through. Modes: sync, ablate, route, host, launch (see the module doc). remove() restores every class."""

    def __init__(self, reg, ctx=8192, seed=0):
        self.reg, self.ctx, self.mode = reg, ctx, None
        self.stub, self.subops = set(), False
        self.rng = np.random.default_rng(seed)
        self.time = collections.defaultdict(lambda: [0, 0.0, 0.0])          # key -> calls, incl s, excl s   (sync, host)
        self.calls = collections.Counter()
        self.first = {}                                                     # key -> rank of its first call in a forward (record pass)
        self.spec, self.built, self.variants = {}, {}, {}
        self.route = collections.defaultdict(lambda: [0, 0.0, 0.0])         # key -> calls, sum distinct experts, sum slots
        self.launch = collections.defaultdict(lambda: [0, 0])               # key -> calls, self dispatches
        self.launch_types = collections.defaultdict(collections.Counter)
        self._stack, self._lstack, self._ks, self._active = [], [], [], set()
        self._patched = []

    # -- install / remove
    def install(self):
        for cls, meth in sorted(self.reg.classes(), key=lambda t: (t[0].__name__, t[1])):
            orig = getattr(cls, meth)
            had = meth in cls.__dict__
            setattr(cls, meth, self._wrap(meth, orig))
            self._patched.append((cls, meth, orig, had))
        if self.subops:
            self._wrap_free()
        return self

    def remove(self):
        for cls, meth, orig, had in reversed(self._patched):
            if had:
                setattr(cls, meth, orig)
            else:
                delattr(cls, meth)
        for owner, name, orig in reversed(getattr(self, "_free", [])):
            setattr(owner, name, orig)
        self._patched, self._free = [], []

    def __enter__(self):
        return self.install()

    def __exit__(self, *e):
        self.mode = None
        self.remove()

    def _wrap_free(self):
        self._free = []
        for modname, name in SUBOPS:
            owner = mx.fast if modname == "mx.fast" else mx
            orig = getattr(owner, name, None)
            if orig is None:
                continue
            key = f"#{modname}.{name}"
            hk = self

            def w(*a, _o=orig, _k=key, **k):
                if hk.mode != "sync" or not hk.subops:
                    return _o(*a, **k)
                arrs = [x for x in a if isinstance(x, mx.array)]
                if arrs:
                    mx.eval(arrs)
                fr = [0.0]
                hk._stack.append(fr)
                t0 = time.perf_counter()
                try:
                    y = _o(*a, **k)
                    o = flat(y)
                    if o:
                        mx.eval(o)
                finally:
                    dt = time.perf_counter() - t0
                    hk._stack.pop()
                if hk._stack:
                    hk._stack[-1][0] += dt
                e = hk.time[_k]
                e[0] += 1
                e[1] += dt
                e[2] += dt - fr[0]
                return y
            try:
                setattr(owner, name, w)
                self._free.append((owner, name, orig))
            except Exception as e:      # noqa
                log("subop", key, "not hookable:", e)

    # -- the wrapper
    def _wrap(self, meth, orig):
        hk = self

        def w(self_, *a, **k):
            mode = hk.mode
            if mode is None:
                return orig(self_, *a, **k)
            sk = (id(self_), meth)
            info = hk.reg.by_id.get(sk)
            if info is None or sk in hk._active:
                return orig(self_, *a, **k)
            key, li = info
            if mode == "ablate":
                return hk._ablate(orig, self_, sk, key, a, k)
            hk._active.add(sk)
            try:
                if mode == "record":
                    return hk._record(orig, self_, sk, key, a, k)
                if mode == "sync":
                    return hk._sync(orig, self_, key, a, k)
                if mode == "host":
                    return hk._host(orig, self_, key, a, k)
                if mode == "route":
                    return hk._route(orig, self_, key, a, k)
                if mode == "launch":
                    return hk._launch(orig, self_, key, a, k)
                return orig(self_, *a, **k)
            finally:
                hk._active.discard(sk)
        return w

    # -- modes
    def _timed(self, orig, self_, key, a, k, synced):
        fr = [0.0]
        self._stack.append(fr)
        self._ks.append(key)
        t0 = time.perf_counter()
        try:
            y = orig(self_, *a, **k)
            if synced:
                o = flat(y)
                if o:
                    mx.eval(o)
        finally:
            dt = time.perf_counter() - t0
            self._stack.pop()
            self._ks.pop()
        if self._stack:
            self._stack[-1][0] += dt
        e = self.time[key]
        e[0] += 1
        e[1] += dt
        e[2] += dt - fr[0]
        return y

    def _sync(self, orig, self_, key, a, k):
        arrs = flat(list(a)) + flat(list(k.values()))
        if arrs:
            mx.eval(arrs)
        y = self._timed(orig, self_, key, a, k, True)
        self._note_route(key, a, k)
        return y

    def _host(self, orig, self_, key, a, k):
        return self._timed(orig, self_, key, a, k, False)

    def _idx_arg(self, a, k):
        for x in list(a[1:]) + list(k.values()):
            if isinstance(x, mx.array) and x.dtype in INT_DTYPES:
                return x
        return None

    def _note_route(self, key, a, k):
        op = self.reg.ops[key]
        if not op.expert:
            return
        idx = self._idx_arg(a, k)
        if idx is None:
            return
        v = np.array(idx).reshape(-1, idx.shape[-1] if idx.ndim else 1)
        e = self.route[key]
        e[0] += 1
        e[1] += len(np.unique(v))
        e[2] += v.size

    def _route(self, orig, self_, key, a, k):
        y = orig(self_, *a, **k)
        self._note_route(key, a, k)
        return y

    @staticmethod
    def _sig(sk, a, k):
        """one module instance can be called with different inputs in a step (a shared rope on q and on k): the stub is per (instance, input shape)"""
        for x in a:
            if isinstance(x, mx.array):
                return (sk, tuple(x.shape))
        return (sk, None)

    def _ablate(self, orig, self_, sk, key, a, k):
        sk = self._sig(sk, a, k)
        if key in self.stub and sk in self.spec:
            built = self._const(sk, key)
            deps = flat(list(a)) + flat(list(k.values()))
            return with_deps(built, deps)
        if sk in self.spec:
            return orig(self_, *a, **k)
        self._active.add(sk[0])
        try:
            return self._record(orig, self_, sk, key, a, k)
        finally:
            self._active.discard(sk[0])

    def _record(self, orig, self_, sk, key, a, k):
        y = orig(self_, *a, **k)
        sk = sk if isinstance(sk[0], tuple) else self._sig(sk, a, k)
        if sk not in self.spec:
            self.spec[sk] = _spec_of(y)
        self.calls[key] += 1
        self.first.setdefault(key, len(self.first))
        return y

    def _launch(self, orig, self_, key, a, k):
        arrs = flat(list(a)) + flat(list(k.values()))
        owner = self._lstack[-1] if self._lstack else "<top>"
        if arrs:
            c = graph_census(arrs)
            self._launch_add(owner, c, calls=0)
            mx.eval(arrs)
        self._lstack.append(key)
        try:
            y = orig(self_, *a, **k)
        finally:
            self._lstack.pop()
        o = flat(y)
        if o:
            self._launch_add(key, graph_census(o), calls=1)
            mx.eval(o)
        else:
            self.launch[key][0] += 1
        return y

    def _launch_add(self, key, census, calls):
        e = self.launch[key]
        e[0] += calls
        e[1] += sum(census.values())
        self.launch_types[key].update(census)

    def launch_flush(self, outs):
        """the nodes left after the last module (head slicing, argmax ...) go to <top>"""
        self._launch_add("<top>", graph_census(flat(outs)), calls=0)
        mx.eval(flat(outs))

    # -- constants of a stubbed module
    def _const(self, sk, key):
        op = self.reg.ops[key]
        spec = self.spec[sk]
        if op.router:                                     # a fresh random routing each call, no artificial expert locality; built ahead (no host work in the timed loop)
            v = self.variants.get(sk)
            if v is None:
                v = self.variants[sk] = [[self._build(spec, True) for _ in range(64)], 0]
            v[1] += 1
            return v[0][v[1] % len(v[0])]
        b = self.built.get(sk)
        if b is None:
            b = self.built[sk] = self._build(spec, False)
        return b

    def _build(self, spec, router):
        kind = spec[0]
        if kind == "a":
            _, shp, dt = spec
            if dt in INT_DTYPES:
                last = shp[-1] if shp else 1
                if router and self.reg.n_experts and last <= self.reg.n_experts:
                    n = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
                    v = np.stack([self.rng.permutation(self.reg.n_experts)[:last] for _ in range(n)]).reshape(shp)
                else:
                    v = self.rng.integers(0, max(8, self.ctx), size=shp)
                a = mx.array(v.astype(np.int64)).astype(dt)
            elif dt == mx.bool_:
                a = mx.ones(shp, dtype=dt)
            elif router:
                a = mx.array((self.rng.random(shp) + 0.1).astype(np.float32)).astype(dt)
            else:
                a = mx.full(shp, 0.01, dtype=dt)
            mx.eval(a)
            return a
        if kind in ("t", "l"):
            out = [self._build(s, router) for s in spec[1]]
            return tuple(out) if kind == "t" else out
        return spec[1]

    def reset(self):
        self.time.clear()
        self.route.clear()
        self.launch.clear()
        self.launch_types.clear()
        self.calls.clear()


# ---------------------------------------------------------------------------------------------- measurement helpers
def arm_bw(gibs=(4, 8), reps=13, say=log):
    """pure stream: sum(axis=1) over a uint32 array (MLX's row-reduce kernel, the one BW_MEASURED.md used), best / median GB/s, plus a fp16 GEMM peak"""
    out = {}
    for g in gibs:
        rows = max(1, int(g * 2 ** 30) // 4 // 4096)
        a = mx.ones((rows, 4096), dtype=mx.uint32)
        mx.eval(a)
        ts = []
        for i in range(reps):
            t0 = time.perf_counter()
            mx.eval(a.sum(axis=1))
            dt = time.perf_counter() - t0
            if i >= 3:
                ts.append(a.nbytes / dt / 1e9)
        out[f"sum_axis1_{g}GiB_u32"] = {"best": round(max(ts), 1), "median": round(med(ts), 1)}
        del a
    mx.clear_cache()
    out["B_best"] = max(v["best"] for v in out.values())
    say(f"bandwidth: {out}")
    return out


def arm_peak(ns=(4096, 8192), reps=8, say=log):
    """fp16 GEMM peak (TFLOPS) of the device, best of reps; the compute roof of the prefill gap"""
    best = 0.0
    for n in ns:
        a = mx.random.normal((n, n)).astype(mx.float16)
        b = mx.random.normal((n, n)).astype(mx.float16)
        mx.eval(a, b)
        mx.eval(a @ b)
        for _ in range(reps):
            t0 = time.perf_counter()
            mx.eval(a @ b)
            best = max(best, 2.0 * n ** 3 / (time.perf_counter() - t0) / 1e12)
        del a, b
    mx.clear_cache()
    say(f"fp16 GEMM peak {best:.1f} TFLOPS")
    return {"fp16_gemm_tflops": round(best, 2)}


TEXT = ("The quick survey of the harbour began before dawn. Engineers walked the quay with lanterns, noting which cranes needed oil and which cables had frayed. "
        "In the control room, a single screen listed the arrivals: three grain carriers, a tanker, and a small ferry that had lost a day to the weather. "
        "def schedule(ships, berths):\n    free = sorted(berths, key=lambda b: b.length)\n    return {s: next(b for b in free if b.length >= s.length) for s in ships}\n"
        "If a train leaves at nine and travels sixty miles per hour, how far has it gone by noon? The answer is one hundred eighty miles. ")


def make_ids(tok, n, vocab, text=None, seed=0):
    ids = None
    if tok is not None:
        try:
            r = tok.encode(text or TEXT)
            r = r["input_ids"] if isinstance(r, dict) else r
            ids = [int(i) for i in r]
        except Exception:      # noqa
            ids = None
    if not ids:
        ids = [int(i) for i in np.random.default_rng(seed).integers(3, max(8, (vocab or 1000) - 1), size=256)]
    vmax = (vocab or 10 ** 9) - 1
    ids = [min(i, vmax) for i in ids]
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def prefill(be, ids, chunk, cache=None):
    cache = cache if cache is not None else be.make_cache()
    t0 = time.perf_counter()
    for a in range(0, len(ids), chunk):
        be.prefill_chunk(cache, ids[a:a + chunk])
    mx.eval(*be.state(cache))
    return cache, time.perf_counter() - t0


def cache_read_bytes(c):
    """bytes one decode step reads from a layer's cache: a preallocated KV buffer counts its used rows only (offset), a recurrent state in full"""
    from kyojin_mlx.serve.backends import arrays_of
    off = getattr(c, "offset", None)
    if off is None and getattr(c, "caches", None):
        off = getattr(c.caches[0], "offset", None)
    try:
        off = int(off) if off is not None else None
    except Exception:      # noqa
        off = None
    tot = 0.0
    for a in arrays_of(c):
        n = a.nbytes
        if off and a.ndim >= 3 and a.shape[-2] > off:
            n = n * off / a.shape[-2]
        tot += n
    return tot


# ---------------------------------------------------------------------------------------------- the profiler
@dataclass
class Opts:
    rows: tuple = (1, 2, 4)
    ctx: int = 8192
    prefill: int = 8192
    chunk: int = 2048
    arms: tuple = ("bw", "decode", "prefill")
    blocks: int = 7
    steps: int = 24
    keys: str | None = None             # regex: ablate only these op keys (the rest of the table keeps its non-ablation columns)
    max_depth: int | None = None
    subops: bool = False
    text: str | None = None
    B: float | None = None              # GB/s override (else measured in process, else B_REF)
    peak_tflops: float | None = None
    bw_gibs: tuple = (4, 8)
    prefill_reps: int = 1
    gate: Gate = field(default_factory=NullGate)


class Profiler:
    def __init__(self, be, cfg, tok, opts: Opts, say=log):
        self.be, self.cfg, self.tok, self.o, self.say = be, cfg, tok, opts, say
        model = getattr(be, "model", None)
        self.model = model
        self.reg = Registry(model, cfg)
        if not self.reg.layers:
            raise ValueError("no `layers` list found in the model: opprof needs a layered trunk")
        self.hk = Hooks(self.reg, ctx=opts.ctx)
        self.hk.subops = opts.subops
        self.B = opts.B or B_REF
        self.peak = opts.peak_tflops
        self.res = {"bandwidth": {}, "decode": {}, "prefill": {}}
        self.tcfg = _cfg_text(cfg)
        self.vocab = self.reg.vocab or 1000
        self._cache = None

    # ---- static table
    def static(self):
        reg = self.reg
        rows = []
        for op in reg.ops.values():
            rows.append({"key": op.key, "kind": op.kind, "path": op.path, "layers": op.layers, "parent": op.parent, "class": type(op.mods[0][0]).__name__,
                         "bytes_fixed": op.fixed, "bytes_stacked": op.stacked, "experts": op.E, "flops_row": op.flops_row, "router": op.router, "mixer": op.mixer,
                         "gather": op.gather})
        kinds = collections.Counter(reg.layers)
        return {"layers": len(reg.layers), "layer_kinds": dict(kinds), "n_experts": reg.n_experts, "top_k": reg.top_k, "ops": rows,
                "weight_bytes_total": sum(r["bytes_fixed"] + r["bytes_stacked"] for r in rows if not r["key"].endswith(".as_linear"))}

    # ---- bandwidth / peak
    def bw_arm(self):
        with self.o.gate.hold():
            r = arm_bw(self.o.bw_gibs, say=self.say)
            r.update(arm_peak(say=self.say))
        self.res["bandwidth"] = r
        if self.o.B is None:
            self.B = r["B_best"]
        if self.peak is None:
            self.peak = r["fp16_gemm_tflops"]
        return r

    # ---- decode
    def _ensure_cache(self):
        if self._cache is None:
            ids = make_ids(self.tok, self.o.ctx + 400, self.vocab, self._text())
            self._ids = ids
            self.say(f"prefill {self.o.ctx} tokens for the decode context (chunk {self.o.chunk}) ...")
            with self.o.gate.hold():
                self._cache, dt = prefill(self.be, ids[:self.o.ctx], self.o.chunk)
            self.res["decode_ctx_prefill_s"] = round(dt, 2)
            self.kvbytes = self._kv_by_key()

    def _text(self):
        return open(self.o.text).read() if self.o.text else None

    def _kv_by_key(self):
        """bytes the mixer of each layer reads from its cache per session and step, summed per mixer key"""
        out = collections.Counter()
        cache = self._cache
        if not isinstance(cache, (list, tuple)) or len(cache) != len(self.reg.layers):
            return out
        for li, c in enumerate(cache):
            kind = self.reg.layers[li]
            mk = next((op.key for op in self.reg.ops.values() if op.kind == kind and op.mixer), None)
            if mk:
                out[mk] += cache_read_bytes(c)
        return out

    def _tokens(self, R, n):
        ids = self._ids[self.o.ctx:] + self._ids[:self.o.ctx]
        return [mx.array([ids[(i * R + r) % len(ids)] for r in range(R)], dtype=mx.int32) for i in range(n)]

    def _batch(self, R):
        be = self.be
        return be.batch([be.fork(self._cache) for _ in range(R)])

    def step_ms_set(self, R, stub, toks):
        hk, be, n = self.hk, self.be, self.o.steps
        hk.stub, hk.mode = set(stub), "ablate"
        ts = []
        try:
            for _ in range(self.o.blocks):
                bc = self._batch(R)
                mx.eval(be.step(bc, toks[0]))
                t0 = time.perf_counter()
                prev = None
                for i in range(n):
                    lg = be.step(bc, toks[1 + i])
                    mx.async_eval(lg)
                    if prev is not None:
                        mx.eval(prev)
                    prev = lg
                mx.eval(prev)
                ts.append((time.perf_counter() - t0) / n)
                del bc
        finally:
            hk.mode, hk.stub = None, set()
        return med(ts) * 1e3, (max(ts) - min(ts)) * 1e3

    def _keys_to_ablate(self):
        o, reg = self.o, self.reg
        keys = [k for k, op in reg.ops.items() if self.hk.calls.get(k) and not op.trunk]
        if o.max_depth is not None:
            keys = [k for k in keys if (reg.ops[k].path.count(".") + (1 if reg.ops[k].path else 0)) <= o.max_depth]
        if o.keys:
            keys = [k for k in keys if re.search(o.keys, k)]
        return keys

    def decode_arm(self, R):
        be, hk, reg, o = self.be, self.hk, self.reg, self.o
        self._ensure_cache()
        toks = self._tokens(R, max(o.steps, 12) + 6)
        out = {"rows": R, "ctx": o.ctx}
        # record the output spec of every module on real calls
        hk.spec.clear(); hk.built.clear(); hk.variants.clear(); hk.calls.clear(); hk.first.clear()
        hk.mode, hk.stub = "record", set()
        bc = self._batch(R)
        for i in range(3):
            mx.eval(be.step(bc, toks[i]))
        hk.mode = None
        del bc
        reg.settle(hk.first)
        keys = self._keys_to_ablate()
        called = [k for k in reg.ops if hk.calls.get(k)]
        out["calls_per_step"] = {k: hk.calls[k] / 3 for k in called}
        layer_keys = [k for k in called if k.startswith("layer.")]
        top_keys = [k for k in called if reg.ops[k].kind == "top" and not reg.ops[k].trunk and (reg.ops[k].parent is None or reg.ops[reg.ops[k].parent].trunk)]
        sets = [("BASE", [])] + [(k, [k]) for k in keys] + [("ALL layers", layer_keys), ("ALL layers+top", layer_keys + top_keys)]
        # timed: step times, host round trip, graph build, ablation
        with o.gate.hold() as gate:
            res, bases = {}, []
            for n_, (name, ks) in enumerate(sets):
                if gate.checkpoint():
                    bases.append(self.step_ms_set(R, [], toks)[0])
                ms, spread = self.step_ms_set(R, ks, toks)
                res[name] = {"keys": ks, "step_ms": ms, "spread_ms": spread}
                if n_ % 6 == 0:
                    bases.append(self.step_ms_set(R, [], toks)[0])
            base = med(bases)
            out["step_ms"] = round(base, 3)
            out["base_samples_ms"] = [round(b, 3) for b in bases]
            out["tok_s"] = round(R * 1e3 / base, 2)
            bc = self._batch(R)
            mx.eval(be.step(bc, toks[0]))
            ts = []
            for i in range(16):
                t0 = time.perf_counter()
                lg = be.step(bc, toks[1 + i % (len(toks) - 1)])
                ts.append(time.perf_counter() - t0)
                mx.eval(lg)
            out["graph_build_ms"] = round(med(ts) * 1e3, 3)
            t0 = time.perf_counter()
            for i in range(12):
                lg = be.step(bc, toks[1 + i])
                int(mx.argmax(lg, axis=-1)[0].item())
            out["step_ms_host_roundtrip"] = round((time.perf_counter() - t0) / 12 * 1e3, 3)
            del bc
        for name, r in res.items():
            if name != "BASE":
                r["delta_ms"] = base - r["step_ms"]
        out["ablation"] = {k: v for k, v in res.items()}
        # untimed (correctness-class) passes: routing, synced cross-check, launches, host build per op
        out["route"] = self._route_pass(R, toks)
        out["sync"] = self._sync_pass(R, toks)
        out["launch"] = self._launch_pass(R, toks)
        out["host_us"] = self._host_pass(R, toks)
        out["rows_table"] = self.decode_table(R, out)
        self.res["decode"][str(R)] = out
        return out

    def _route_pass(self, R, toks):
        hk, be = self.hk, self.be
        hk.route.clear()
        bc = self._batch(R)
        hk.mode = "route"
        try:
            for i in range(6):
                mx.eval(be.step(bc, toks[i]))
        finally:
            hk.mode = None
        return {k: {"calls": v[0], "distinct_mean": v[1] / max(1, v[0]), "slots_mean": v[2] / max(1, v[0])} for k, v in hk.route.items()}

    def _sync_pass(self, R, toks):
        hk, be = self.hk, self.be
        bc = self._batch(R)
        mx.eval(be.step(bc, toks[0]))
        hk.time.clear()
        hk.mode = "sync"
        t0 = time.perf_counter()
        try:
            for i in range(8):
                mx.eval(be.step(bc, toks[1 + i]))
        finally:
            hk.mode = None
        wall = (time.perf_counter() - t0) / 8
        return {"step_ms": round(wall * 1e3, 3), "ops": {k: {"calls_step": v[0] / 8, "incl_ms": v[1] / 8 * 1e3, "excl_ms": v[2] / 8 * 1e3} for k, v in hk.time.items()}}

    def _launch_pass(self, R, toks):
        hk, be = self.hk, self.be
        bc = self._batch(R)
        mx.eval(be.step(bc, toks[0]))
        bc2 = self._batch(R)
        mx.eval(be.step(bc2, toks[0]))
        lg = be.step(bc, toks[1])                      # authoritative total: the unsynced lazy graph of one step
        total = graph_census([lg])
        mx.eval(lg)
        hk.launch.clear(); hk.launch_types.clear()
        hk.mode = "launch"
        try:
            lg = be.step(bc2, toks[1])
            hk.launch_flush([lg])
        finally:
            hk.mode = None
        per = {k: {"calls": v[0], "self": v[1]} for k, v in hk.launch.items()}
        return {"total_dispatches": sum(total.values()), "by_type": dict(total.most_common()),
                "attributed": sum(v["self"] for v in per.values()), "per_op": per,
                "per_op_types": {k: dict(c.most_common(6)) for k, c in hk.launch_types.items()}}

    def _host_pass(self, R, toks):
        hk, be = self.hk, self.be
        bc = self._batch(R)
        mx.eval(be.step(bc, toks[0]))
        hk.time.clear()
        n = 8
        hk.mode = "host"
        try:
            for i in range(n):
                lg = be.step(bc, toks[1 + i])
                hk.mode = None
                mx.eval(lg)
                hk.mode = "host"
        finally:
            hk.mode = None
        return {k: {"calls_step": v[0] / n, "incl_us": v[1] / n * 1e6, "excl_us": v[2] / n * 1e6} for k, v in hk.time.items()}

    # ---- decode table
    def decode_table(self, R, d):
        reg, B = self.reg, self.B
        base = d["step_ms"]
        abl, route, launch, host, sync = d["ablation"], d["route"], d["launch"], d["host_us"], d["sync"]["ops"]
        kids = collections.defaultdict(list)
        for k, op in reg.ops.items():
            if op.parent:
                kids[op.parent].append(k)
        rows = []
        for k, op in reg.ops.items():
            if not d["calls_per_step"].get(k):
                continue
            a = abl.get(k)
            delta = a["delta_ms"] if a else None
            sub = [abl[c]["delta_ms"] for c in kids[k] if c in abl]
            self_ms = None if delta is None else delta - sum(sub)
            dist = route.get(k, {}).get("distinct_mean")
            by = op_bytes(op, R, reg, dist) + self.kvbytes.get(k, 0.0) * R
            fr = (op.flops_row or 0.0) + (reg.shared_totals(op)[1] if op.shared else 0.0)
            fl = fr * R if fr else None
            ideal = by / 1e9 / B * 1e3
            r = {"key": k, "layers": op.layers, "calls_step": d["calls_per_step"][k], "delta_ms": delta, "self_ms": self_ms,
                 "noise_ms": a["spread_ms"] if a else None, "bytes": by, "kv_bytes": self.kvbytes.get(k, 0.0) * R, "experts_distinct": dist,
                 "ideal_ms": ideal, "flops": fl, "launch_self": launch["per_op"].get(k, {}).get("self"),
                 "host_us_self": host.get(k, {}).get("excl_us"), "sync_excl_ms": sync.get(k, {}).get("excl_ms")}
            if self_ms is not None:
                r["gap_ms"] = self_ms - ideal
                if self_ms > max(0.02, r["noise_ms"] or 0):
                    r["gbs"] = by / 1e9 / (self_ms / 1e3) if by else None
                    r["pct_B"] = 100 * r["gbs"] / B if r["gbs"] else None
                    r["tflops"] = fl / (self_ms / 1e3) / 1e12 if fl else None
                r["pct_step"] = 100 * self_ms / base
            if r["launch_self"] is not None:
                r["launch_tok"] = r["launch_self"] / R
            rows.append(r)
        top = abl.get("ALL layers+top")
        unattr = None if top is None else base - top["delta_ms"]
        tot_bytes = sum(r["bytes"] for r in rows)
        d["summary"] = {"step_ms": base, "tok_s": d["tok_s"], "bytes": tot_bytes, "ideal_ms": tot_bytes / 1e9 / B * 1e3, "pct_B": 100 * tot_bytes / 1e9 / (base / 1e3) / B,
                        "unattributed_ms": unattr, "launches_step": launch["total_dispatches"], "launches_tok": launch["total_dispatches"] / R,
                        "launches_attributed": launch["attributed"], "graph_build_ms": d["graph_build_ms"], "host_roundtrip_ms": d["step_ms_host_roundtrip"],
                        "host_bound": d["graph_build_ms"] > base}
        return rows

    # ---- prefill
    def prefill_arm(self):
        be, hk, reg, o = self.be, self.hk, self.reg, self.o
        n, chunk = o.prefill, o.chunk
        ids = make_ids(self.tok, n, self.vocab, self._text())
        out = {"tokens": n, "chunk": chunk, "reps": []}
        c0, _ = prefill(be, ids[:min(1024, n)], min(1024, chunk))
        del c0
        mx.clear_cache()
        with o.gate.hold():
            for _ in range(o.prefill_reps):
                c, dt = prefill(be, ids, chunk)
                out["reps"].append({"s": round(dt, 3), "tok_s": round(n / dt, 1)})
                del c
                mx.clear_cache()
            hk.reset()
            hk.mode = "sync"
            try:
                c, dt = prefill(be, ids, chunk)
            finally:
                hk.mode = None
            out["sync_total_s"] = round(dt, 3)
        del c
        out["sync"] = {k: {"calls": v[0], "incl_s": v[1], "excl_s": v[2]} for k, v in hk.time.items()}
        out["route"] = {k: {"calls": v[0], "distinct_mean": v[1] / max(1, v[0]), "slots_mean": v[2] / max(1, v[0])} for k, v in hk.route.items()}
        if o.subops:
            hk.reset()
            hk.mode = "sync"
            try:
                c, dt2 = prefill(be, ids, chunk)
            finally:
                hk.mode = None
            del c
            out["subop_pass_total_s"] = round(dt2, 3)
            out["subops"] = {k: {"calls": v[0], "incl_s": v[1], "excl_s": v[2]} for k, v in hk.time.items() if k.startswith("#")}
        # launches of one chunk
        cache = be.make_cache()
        hk.launch.clear(); hk.launch_types.clear()
        hk.mode = "launch"
        try:
            lg = be.prefill_chunk(cache, ids[:chunk])
            hk.launch_flush([lg])
        finally:
            hk.mode = None
        out["launch"] = {"chunk_tokens": min(chunk, n), "per_op": {k: {"calls": v[0], "self": v[1]} for k, v in hk.launch.items()},
                         "total": sum(v[1] for v in hk.launch.values())}
        del cache
        mx.clear_cache()
        out["rows_table"] = self.prefill_table(out)
        self.res["prefill"] = out
        return out

    def prefill_table(self, d):
        reg, B, peak = self.reg, self.B, self.peak
        n = d["tokens"]
        rows = []
        kids = collections.defaultdict(list)
        for k, op in reg.ops.items():
            if op.parent:
                kids[op.parent].append(k)
        for k, op in reg.ops.items():
            s = d["sync"].get(k)
            if not s or not s["calls"]:
                continue
            calls = s["calls"]
            rt = d["route"].get(k)
            per_call_rows = n / max(1, calls / max(1, op.layers))        # tokens per call (calls = layers x chunks)
            by_call = op_bytes(op, per_call_rows, reg, rt["distinct_mean"] if rt else None)
            bytes_all = by_call / max(1, op.layers) * calls              # weights read once per call by each layer instance
            fr = (op.flops_row or 0.0) + (reg.shared_totals(op)[1] if op.shared else 0.0)
            fl = fr * n if fr else None
            ms = s["excl_s"] * 1e3
            ideal_b = bytes_all / 1e9 / B * 1e3
            ideal_f = (fl / 1e12 / peak * 1e3) if (fl and peak) else 0.0
            ideal = max(ideal_b, ideal_f)
            r = {"key": k, "layers": op.layers, "calls": calls, "self_ms": ms, "incl_ms": s["incl_s"] * 1e3, "bytes": bytes_all, "ideal_ms": ideal,
                 "ideal_bw_ms": ideal_b, "ideal_flops_ms": ideal_f, "gap_ms": ms - ideal, "flops": fl, "experts_distinct": rt["distinct_mean"] if rt else None,
                 "pct_total": 100 * ms / (d["sync_total_s"] * 1e3)}
            if ms > 0.05:
                r["gbs"] = bytes_all / 1e9 / (ms / 1e3)
                r["pct_B"] = 100 * r["gbs"] / B
                if fl:
                    r["tflops"] = fl / (ms / 1e3) / 1e12
                    r["pct_peak"] = 100 * r["tflops"] / peak if peak else None
            la = d["launch"]["per_op"].get(k)
            if la:
                r["launch_chunk"] = la["self"]
                r["launch_tok"] = la["self"] / d["launch"]["chunk_tokens"]
            rows.append(r)
        d["summary"] = {"tok_s_unprofiled": (d["reps"][-1]["tok_s"] if d["reps"] else None), "sync_total_s": d["sync_total_s"],
                        "launches_chunk": d["launch"]["total"], "launches_tok": d["launch"]["total"] / d["launch"]["chunk_tokens"]}
        return rows

    # ---- all
    def run(self):
        o = self.o
        self.res["meta"] = self.meta()
        self.res["static"] = self.static()
        with self.hk:
            for name, fn in (("bw", self.bw_arm), ("decode", self._decode_all), ("prefill", self.prefill_arm)):
                if name not in o.arms:
                    continue
                t0 = time.time()
                try:
                    fn()
                    self.say(f"arm {name} done in {time.time() - t0:.0f} s")
                except Exception:
                    self.res.setdefault("errors", {})[name] = traceback.format_exc()
                    self.say(f"arm {name} FAILED:\n" + traceback.format_exc())
                    self.hk.mode = None
        self.res["meta"]["B_used_gbs"] = self.B
        self.res["meta"]["peak_tflops_used"] = self.peak
        self.res["meta"]["gate_wait_s"] = round(getattr(o.gate, "waited", 0.0), 1)
        return self.res

    def _decode_all(self):
        for R in self.o.rows:
            self.decode_arm(R)
            self.say(f"decode R={R}: {self.res['decode'][str(R)]['summary']}")

    def meta(self):
        try:
            rev = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:      # noqa
            rev = None
        try:
            dev = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in mx.device_info().items()}
        except Exception:      # noqa
            dev = {}
        return {"backend": type(self.be).__name__, "model_type": self.tcfg.get("model_type"), "pack": getattr(self.be, "pack", None), "mlx": mx.__version__,
                "device": dev, "git": rev, "ctx": self.o.ctx, "prefill_tokens": self.o.prefill, "chunk": self.o.chunk, "rows": list(self.o.rows),
                "B_ref_gbs": B_REF, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "layers": len(self.reg.layers), "layer_kinds": dict(collections.Counter(self.reg.layers))}


def profile(be, cfg, tok, opts=None, say=log):
    """Profile a loaded backend; returns the report dict (see write_report)."""
    return Profiler(be, cfg, tok, opts or Opts(), say).run()


# ---------------------------------------------------------------------------------------------- reports
def rank(rows, field_="gap_ms"):
    return sorted([r for r in rows if r.get(field_) is not None], key=lambda r: -r[field_])


def _f(x, n=2, none="-"):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return none
    return f"{x:.{n}f}"


def _mb(x):
    return _f(x / 1e6 if x is not None else None, 1)


def render_md(res, top=25):
    out = []
    P = out.append
    m = res.get("meta", {})
    bw = res.get("bandwidth", {})
    P(f"# opprof report: {m.get('model_type')} ({m.get('backend')}), {m.get('layers')} layers {m.get('layer_kinds')}")
    P(f"\nmlx {m.get('mlx')}, git {m.get('git')}, {m.get('time')}, pack `{m.get('pack')}`. B used = **{_f(m.get('B_used_gbs'), 0)} GB/s** "
      f"(in-process stream best {bw.get('B_best', '-')}, reference {m.get('B_ref_gbs')}), fp16 GEMM peak used = {_f(m.get('peak_tflops_used'), 1)} TFLOPS. "
      f"Timed parts held /tmp/gpu_gate.lock (waited {m.get('gate_wait_s')} s).")
    P("\nGap = self ms - ideal ms; ideal = max(bytes / B, flops / peak). Rows are ranked by gap (largest first). `self` = the op's ablation delta minus the deltas of its child "
      "modules (no double counting; ablations are not strictly additive, `noise` is the block spread: rows below it are marked `~`). bytes = weights actually touched "
      "(experts: measured distinct experts) + KV / state read for the mixer rows. launches = graph nodes without view primitives, counted exactly per op.")
    if bw:
        P("\n## Bandwidth and compute roofs\n")
        P("| cell | best | median |\n|---|---|---|")
        for k, v in bw.items():
            if isinstance(v, dict):
                P(f"| {k} | {v['best']} GB/s | {v['median']} GB/s |")
        P(f"\nfp16 GEMM peak: {bw.get('fp16_gemm_tflops')} TFLOPS.")
    for R, d in sorted(res.get("decode", {}).items(), key=lambda kv: int(kv[0])):
        s = d["summary"]
        P(f"\n## Decode R={R}, context {d['ctx']}\n")
        P(f"step **{_f(s['step_ms'])} ms** pipelined ({_f(s['tok_s'], 1)} tok/s), host round trip {_f(s['host_roundtrip_ms'])} ms, host graph build {_f(s['graph_build_ms'])} ms"
          f"{' (HOST BOUND)' if s['host_bound'] else ''}. Bytes read {_f(s['bytes'] / 1e9, 2)} GB -> ideal {_f(s['ideal_ms'])} ms at B = {_f(s['pct_B'], 1)} % of B achieved. "
          f"Unattributed (not in any module) {_f(s['unattributed_ms'])} ms. Launches {s['launches_step']} per step = {_f(s['launches_tok'], 1)} per token "
          f"({s['launches_attributed']} attributed to ops).")
        P("\n| rank | op | layers | calls/step | self ms | % step | noise | MB | GB/s | % B | ideal ms | gap ms | TFLOPS | launches/step | /token | host us |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rank(d["rows_table"])[:top], 1):
            sm = "~" if (r["noise_ms"] is not None and r["self_ms"] is not None and abs(r["self_ms"]) < r["noise_ms"]) else ""
            P(f"| {i} | {r['key']} | {r['layers']} | {_f(r['calls_step'], 1)} | {sm}{_f(r['self_ms'], 3)} | {_f(r.get('pct_step'), 1)} | {_f(r['noise_ms'], 2)} | {_mb(r['bytes'])} | "
              f"{_f(r.get('gbs'), 0)} | {_f(r.get('pct_B'), 0)} | {_f(r['ideal_ms'], 3)} | {_f(r['gap_ms'], 3)} | {_f(r.get('tflops'), 2)} | {r['launch_self'] if r['launch_self'] is not None else '-'} | "
              f"{_f(r.get('launch_tok'), 1)} | {_f(r['host_us_self'], 0)} |")
        if R == min(res["decode"], key=int):
            P(f"\nDispatches by primitive (R={R}): " + ", ".join(f"{k} {v}" for k, v in list(d["launch"]["by_type"].items())[:12]))
    pf = res.get("prefill")
    if pf and pf.get("rows_table"):
        s = pf["summary"]
        P(f"\n## Prefill {pf['tokens']} tokens, chunk {pf['chunk']}\n")
        P(f"unprofiled {_f(s['tok_s_unprofiled'], 1)} tok/s; synced profile total {s['sync_total_s']} s (per-op ms are exclusive times from eval-bracketed calls). "
          f"Launches {s['launches_chunk']} per chunk = {_f(s['launches_tok'], 3)} per token.")
        P("\n| rank | op | layers | calls | self ms | % total | MB read | GB/s | % B | TFLOPS | % peak | ideal ms | gap ms | launches/token |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rank(pf["rows_table"])[:top], 1):
            P(f"| {i} | {r['key']} | {r['layers']} | {r['calls']} | {_f(r['self_ms'], 1)} | {_f(r['pct_total'], 1)} | {_mb(r['bytes'])} | {_f(r.get('gbs'), 0)} | {_f(r.get('pct_B'), 0)} | "
              f"{_f(r.get('tflops'), 1)} | {_f(r.get('pct_peak'), 0)} | {_f(r['ideal_ms'], 1)} | {_f(r['gap_ms'], 1)} | {_f(r.get('launch_tok'), 3)} |")
    if res.get("errors"):
        P("\n## Errors\n")
        for k, v in res["errors"].items():
            P(f"arm {k}:\n```\n{v}\n```")
    return "\n".join(out) + "\n"


def write_report(res, outdir, top=25):
    os.makedirs(outdir, exist_ok=True)
    jp, mp = os.path.join(outdir, "opprof.json"), os.path.join(outdir, "opprof.md")
    json.dump(res, open(jp, "w"), indent=1, default=str)
    open(mp, "w").write(render_md(res, top))
    return jp, mp


# ---------------------------------------------------------------------------------------------- CLI
def load_tokenizer(path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path)
    except Exception:      # noqa
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="pack directory (anything kyojin_mlx.serve.backends.load_backend serves)")
    ap.add_argument("--decode-rows", default="1,2,4", help="R values: sessions batched in one decode step")
    ap.add_argument("--ctx", type=int, default=8192, help="decode context (tokens)")
    ap.add_argument("--prefill", type=int, default=8192, help="prefill tokens")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk")
    ap.add_argument("--out", default="opprof_out")
    ap.add_argument("--arms", default="bw,decode,prefill")
    ap.add_argument("--blocks", type=int, default=7, help="ablation: blocks per set (median)")
    ap.add_argument("--steps", type=int, default=24, help="ablation: steps per block")
    ap.add_argument("--keys", help="regex: ablate only matching op keys")
    ap.add_argument("--max-depth", type=int, help="ablate modules down to this depth below the layer")
    ap.add_argument("--subops", action="store_true", help="also profile mx.gather_mm / argsort / sdpa ... inside the synced prefill (inflated, split only)")
    ap.add_argument("--text", help="text file for the context / routing (default: a built-in paragraph)")
    ap.add_argument("--B", type=float, help="GB/s: use this as 100 %% (default: measured in process)")
    ap.add_argument("--peak-tflops", type=float)
    ap.add_argument("--bw-gib", default="4,8")
    ap.add_argument("--top", type=int, default=25, help="rows per table in the markdown")
    ap.add_argument("--device", choices=("gpu", "cpu"), help="mlx default device (cpu: tests and tiny models only)")
    ap.add_argument("--no-gate", action="store_true", help="do not take /tmp/gpu_gate.lock (CPU runs / tests ONLY: timings are void on a shared GPU)")
    ap.add_argument("--report-only", help="opprof.json: re-render the markdown and exit")
    return ap.parse_args(argv)


def opts_from(a):
    return Opts(rows=tuple(int(x) for x in a.decode_rows.split(",") if x), ctx=a.ctx, prefill=a.prefill, chunk=a.chunk, arms=tuple(a.arms.split(",")), blocks=a.blocks,
                steps=a.steps, keys=a.keys, max_depth=a.max_depth, subops=a.subops, text=a.text, B=a.B, peak_tflops=a.peak_tflops,
                bw_gibs=tuple(float(x) for x in a.bw_gib.split(",")), gate=NullGate() if a.no_gate else Gate())


def main(argv=None):
    a = parse_args(argv)
    if a.report_only:
        res = json.load(open(a.report_only))
        print(render_md(res, a.top))
        return 0
    if not a.model:
        raise SystemExit("--model is required")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if a.device:
        mx.set_default_device(mx.cpu if a.device == "cpu" else mx.gpu)
    from kyojin_mlx.serve.backends import load_backend
    be, cfg = load_backend(a.model, mtp=False, log=log)
    res = profile(be, cfg, load_tokenizer(a.model), opts_from(a))
    jp, mp = write_report(res, a.out, a.top)
    log("wrote", jp, mp)
    return 1 if res.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
