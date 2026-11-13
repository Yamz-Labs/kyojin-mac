# SPDX-License-Identifier: Apache-2.0
"""Disk KV checkpoints: a prompt prefix that was prefilled once is reloaded from the SSD instead of recomputed (never repay the prompt cache), also
after a restart of the server or when the in-memory prefix snapshots were dropped.

One file per BLOCK of tokens (default 8192) written with `mx.save_safetensors`, keyed by the chained blake2b digest of the int32 token prefix (the
same digest as `serve.engine.PrefixCache`, so both tiers answer the same key). A file holds only what the block added:
  growing arrays   the rows [prev_rows, rows) of every per-layer array that grows with the prompt (DS4.1: compressed KV, index keys, Engram token
                   history; mlx-lm KVCache: keys / values), so a block costs the same at 8K and at 120K
  state at the end of the block   everything fixed-size (DS4.1: sliding-window rings, compressor kv_state / score_state; mlx-lm: ArraysCache and
                   rotating caches through their own `state` / `meta_state`), plus the caller's extras (the MTP drafter's main-KV ring)
A prefix of k blocks is the chain of k files (each names its parent digest); the load concatenates the growing parts and takes the fixed state
from the last file. Reload is bit exact: arrays are written and read in their own dtype (bf16 included); the cross-layer handoffs of the DS4.1
cache (index keys, candidates, compressed KV view, top-k) are not stored, every forward republishes them from the source layers before a consumer
reads them (tests/test_kvdisk.py proves identical next-token logits).

Safety: files are written under a temporary name, fsynced and renamed (a crash leaves only `*.partial.safetensors`, removed at start); a file that does not
read back is deleted and the lookup falls back to a shorter chain; the directory is namespaced by a fingerprint of everything that changes the
cached values (architecture, pack files, prefill chunk, CED signature, MLX version, block size), so a cache built another way is never read.
Disk cap: LRU by last use, a file with a child is never evicted before its child (the chain stays loadable); loading touches the whole chain. Chains named in `pins.json` (and the newest PIN_SHORT parent-less short prefixes, `pins.auto.json`) are never evicted and do not count against the cap."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import uuid

import mlx.core as mx
import numpy as np

VERSION = 1
PINS = "pins.json"
PINS_AUTO = "pins.auto.json"                 # written by the store itself (short roots); pins.json stays the hand-kept list of tools/depth_ab.py
PIN_SHORT = 16                               # short roots (a prefix under one block: a system preamble) kept out of eviction, the newest ones
SUFFIX = ".kv.safetensors"
TMP = ".partial.safetensors"                 # mx.save_safetensors appends .safetensors to any other name


def digest(ids, n: int) -> bytes:
    """blake2b-16 of the first n token ids as int32 bytes (= the chained digest of serve.engine.PrefixCache at n tokens)."""
    return hashlib.blake2b(np.asarray(ids[:n], dtype=np.int32).tobytes(), digest_size=16).digest()


def prefix_digests(ids, lens) -> list:
    """[(n, digest)] for the ascending lengths `lens` (each <= len(ids)), with one incremental blake2b: O(len) for any number of lengths."""
    arr, h, prev, out = np.asarray(ids, dtype=np.int32), hashlib.blake2b(digest_size=16), 0, []
    for n in lens:
        h.update(arr[prev:n].tobytes())
        out.append((n, h.copy().digest()))
        prev = n
    return out


def pack_fingerprint(pack_dir: str) -> str:
    """Config bytes + (name, size, mtime) of every weight file: a re-quantized or re-converted pack never reads old checkpoints."""
    h = hashlib.blake2b(digest_size=12)
    cfg = os.path.join(pack_dir, "config.json")
    if os.path.exists(cfg):
        h.update(open(cfg, "rb").read())
    for f in sorted(os.listdir(pack_dir)):
        if f.endswith(".safetensors") or f.endswith(".json") and f != "config.json":
            st = os.stat(os.path.join(pack_dir, f))
            h.update(f"{f}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()


def ds_cache(c) -> bool:
    """The DS4.1 cache or a subclass of it (fp4pool.PackedDsCache: same state, the pool / index keys stored packed)."""
    from . import archs
    return archs.is_model_cache(c)


# ------------------------------------------------------------------------------------------------------------------------ adapters
class DsAdapter:
    """DeepseekV41Cache (mlx-vlm deepseek_v41; the model cache is a one-entry list)."""
    name = "dsv4"
    FIXED = ("window", "kv_state", "score_state")
    GROW = ("compress", "keys")

    @staticmethod
    def accepts(cache) -> bool:
        return len(cache) == 1 and ds_cache(cache[0])

    def dump(self, cache, prev_rows: dict):
        e = cache[0]
        fixed, grow, rows = {}, {}, {}
        for slot in self.FIXED:
            for l, a in enumerate(getattr(e, slot)):
                if a is not None:
                    fixed[f"{slot}.{l}"] = a
        for slot in self.GROW:
            for l, a in enumerate(getattr(e, slot)):
                if a is not None:
                    k = f"{slot}.{l}"
                    grow[k], rows[k] = a[:, prev_rows.get(k, 0):], a.shape[1]
        if e.engram is not None:
            grow["engram"], rows["engram"] = e.engram[:, prev_rows.get("engram", 0):], e.engram.shape[1]
        return fixed, grow, rows, dict(n_layers=e.n_layers, offset=int(e.offset))

    def restore(self, make_cache, parts):
        e = make_cache()[0]
        meta = parts[-1][1]
        n = meta["n_layers"]
        last_fixed = parts[-1][0]
        grown = {}
        for tensors, _ in parts:
            for k, a in tensors.items():
                if k.startswith("g."):
                    grown.setdefault(k[2:], []).append(a)
        cat = {k: (v[0] if len(v) == 1 else mx.concatenate(v, axis=1)) for k, v in grown.items()}
        empty = mx.zeros((0,))
        state = []
        for slot in ("window", "compress", "keys", "kv_state", "score_state"):
            for l in range(n):
                k = f"{slot}.{l}"
                a = cat.get(k) if slot in self.GROW else last_fixed.get("f." + k)
                state.append(empty if a is None else a)
        state.append(cat.get("engram", empty))
        state.extend([empty] * len(type(e).HANDOFF))
        e.state = state
        e.meta_state = (str(meta["offset"]),)
        return [e]


def _enc(x, out: dict, prefix: str):
    """JSON skeleton of a state pytree (lists / tuples / arrays / None); the arrays go to `out` under `prefix`.<n>."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return {"l": [_enc(v, out, prefix) for v in x], "t": isinstance(x, tuple)}
    if isinstance(x, (bool, int, float, str)):
        return {"s": x}
    if not isinstance(x, mx.array):
        raise TypeError(f"cache state leaf of type {type(x).__name__}")
    name = f"{prefix}.{len(out)}"
    out[name] = x
    return {"a": name}


def _dec(sk, tensors: dict):
    if sk is None:
        return None
    if "l" in sk:
        v = [_dec(c, tensors) for c in sk["l"]]
        return tuple(v) if sk["t"] else v
    if "s" in sk:
        return sk["s"]
    return tensors["f." + sk["a"]]


def _offset_of(e):
    """Token count a cache entry knows: its offset, else the largest one among the members of a CacheList; None when it carries none (a pure state cache)."""
    o = getattr(e, "offset", None)
    if o is None and hasattr(e, "caches"):
        known = [v for v in map(_offset_of, e.caches) if v is not None]
        o = max(known) if known else None
    return None if o is None else int(o)


class GenericAdapter:
    """mlx-lm cache lists: KVCache (keys / values grow along axis 2), anything else through its own state / meta_state (ArraysCache, rotating, ...)."""
    name = "generic"

    @staticmethod
    def accepts(cache) -> bool:
        return len(cache) > 0 and all(hasattr(c, "state") for c in cache)

    def dump(self, cache, prev_rows: dict):
        fixed, grow, rows, metas = {}, {}, {}, []
        for i, e in enumerate(cache):
            if type(e).__name__ == "KVCache" and e.keys is not None:
                n = int(e.offset)                             # keys / values may be longer than offset (spare capacity)
                for nm, a in (("k", e.keys), ("v", e.values)):
                    key = f"kv.{i}.{nm}"
                    grow[key], rows[key] = a[:, :, prev_rows.get(key, 0):n], n
                metas.append(dict(kind="kv"))
            elif type(e).__name__ in ("WinKVCache", "TQKVCache") and e.kq is not None:       # kvwin: pool rows grow, sink / ring overlay is fixed-size state
                n = int(e.offset)
                for nm, a in zip(e.SLOTS[:e.NPOOL], e.state[:e.NPOOL]):
                    key = f"win.{i}.{nm}"
                    grow[key], rows[key] = a[:, :, prev_rows.get(key, 0):n], n
                for nm, a in zip(e.SLOTS[e.NPOOL:], e.state[e.NPOOL:]):
                    if a is not None:
                        fixed[f"win.{i}.{nm}"] = a
                metas.append(dict(kind="win", layout=getattr(e, "CKPT_LAYOUT", 1)))              # row layout of the pool words; the cache tiles K itself on load
            else:
                ms = getattr(e, "meta_state", "")
                metas.append(dict(kind="state", sk=_enc(e.state, fixed, f"st.{i}"), meta=list(ms) if isinstance(ms, (list, tuple)) else ms,
                                  meta_is_seq=isinstance(ms, (list, tuple))))
        offs = [int(getattr(e, "offset", 0)) for e in cache]
        if os.environ.get("KYOJIN_GENERIC_CACHE", "1") != "0":
            known = [o for o in map(_offset_of, cache) if o is not None]
            return fixed, grow, rows, dict(entries=metas, offset=max(known) if known else None, offsets=[o or 0 for o in map(_offset_of, cache)])
        return fixed, grow, rows, dict(entries=metas, offset=max(offs) if offs else 0, offsets=offs)

    def restore(self, make_cache, parts):
        cache = make_cache()
        meta, last_fixed = parts[-1][1], parts[-1][0]
        for i, (e, m) in enumerate(zip(cache, meta["entries"])):
            if m["kind"] == "kv" and type(e).__name__ != "KVCache":    # a plain-KV checkpoint (KYOJIN_TQ_FROM short session) under a TurboQuant default
                from mlx_lm.models.cache import KVCache
                e = cache[i] = KVCache()
            if m["kind"] == "win" and type(e).__name__ == "KVCache":
                raise ValueError("kv checkpoint kinds differ from the cache (win vs plain): refused")
            if any((m["kind"] == "kv" and f"g.kv.{i}.k" not in t) or (m["kind"] == "win" and not any(k.startswith(f"g.win.{i}.") for k in t)) for t, _ in parts):
                raise ValueError("kv checkpoint chain mixes cache kinds: refused")
            if m["kind"] == "kv":
                ks = [t[f"g.kv.{i}.k"] for t, _ in parts]
                vs = [t[f"g.kv.{i}.v"] for t, _ in parts]
                e.keys = ks[0] if len(ks) == 1 else mx.concatenate(ks, axis=2)
                e.values = vs[0] if len(vs) == 1 else mx.concatenate(vs, axis=2)
                e.offset = e.keys.shape[2]
            elif m["kind"] == "win":
                if m.get("layout", 1) != getattr(e, "CKPT_LAYOUT", 1):                                # files from another pool layout: refused, the caller falls back to a shorter prefix
                    raise ValueError(f"kv layout {m.get('layout', 1)} != {getattr(e, 'CKPT_LAYOUT', 1)}")
                cat = lambda nm: (lambda xs: xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=2))([t[f"g.win.{i}.{nm}"] for t, _ in parts])
                e.state = [*(cat(nm) for nm in e.SLOTS[:e.NPOOL]), *(last_fixed.get(f"f.win.{i}.{nm}") for nm in e.SLOTS[e.NPOOL:])]
            else:
                e.state = _dec(m["sk"], last_fixed)
                if m["meta"] not in ("", []):
                    e.meta_state = tuple(m["meta"]) if m["meta_is_seq"] else m["meta"]
            if hasattr(e, "offset") and meta.get("offsets"):
                e.offset = meta["offsets"][i]
        return cache


ADAPTERS = (DsAdapter(), GenericAdapter())


def adapter_for(cache):
    for ad in ADAPTERS:
        if ad.accepts(cache):
            return ad
    raise TypeError(f"no disk checkpoint adapter for {[type(c).__name__ for c in cache][:3]}")


# ------------------------------------------------------------------------------------------------------------------------ store
def _header(path: str) -> dict:
    """The kyojin metadata of a checkpoint file without reading its tensors."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 1 << 26:
            raise ValueError("header too large")
        h = json.loads(f.read(n))
    return json.loads(h["__metadata__"]["kyojin_kv"])


class KvDisk:
    """Checkpoints under `root/<namespace hash>/`. block: tokens per file (multiple of the prefill chunk); cap_bytes: 0 = unlimited."""

    def __init__(self, root: str, namespace: dict, make_cache, block: int = 8192, cap_bytes: int = 0, log=None):
        self.block, self.cap, self.make_cache = int(block), int(cap_bytes), make_cache
        self.log = log or (lambda *_: None)
        self.ns = hashlib.blake2b(json.dumps({**namespace, "block": self.block, "version": VERSION, "mlx": mx.__version__}, sort_keys=True).encode(),
                                  digest_size=8).hexdigest()
        self.dir = os.path.join(root, self.ns)
        os.makedirs(self.dir, exist_ok=True)
        info = os.path.join(self.dir, "namespace.json")
        if not os.path.exists(info):
            open(info, "w").write(json.dumps({**namespace, "block": self.block, "version": VERSION, "mlx": mx.__version__}, indent=1, sort_keys=True))
        self.last_stale, self.refused = False, 0
        self.index: dict = {}                 # (b, digest hex) -> dict(path, size, parent, stamp, rows)
        self.children: dict = {}              # parent key -> number of children in the index
        self.stats = dict(saved=0, bytes_saved=0, save_s=0.0, loaded=0, bytes_loaded=0, load_s=0.0, hits=0, misses=0, skipped=0, evicted=0, bad=0)
        self._scan()

    # -- index
    def _scan(self) -> None:
        for f in os.listdir(self.dir):
            p = os.path.join(self.dir, f)
            if f.endswith(TMP):
                os.unlink(p)
            elif f.endswith(SUFFIX):
                try:
                    self._add(p, _header(p))
                except Exception as e:                                           # noqa: BLE001
                    self.log(f"kvdisk: dropping unreadable {f}: {type(e).__name__}")
                    os.unlink(p)
                    self.stats["bad"] += 1
        # a file whose parent is gone cannot be loaded: remove it (its children go on the next pass)
        changed = True
        while changed:
            changed = False
            for k, e in list(self.index.items()):
                if e["parent"] is not None and e["parent"] not in self.index:
                    self._remove(k)
                    changed = True

    def _add(self, path: str, h: dict) -> None:
        if h.get("version") != VERSION or h.get("block") != self.block:
            raise ValueError("version or block mismatch")
        key = (h["b"], h["digest"])
        parent = (h.get("pb", h["b"] - self.block), h["parent"]) if h["parent"] else None
        self.index[key] = dict(path=path, size=os.path.getsize(path), parent=parent, stamp=os.stat(path).st_mtime, rows=h["rows"],
                               stale=bool(h.get("stale", False)))          # files written before ced_skip_dec=3 have no mark: whole
        if parent is not None:
            self.children[parent] = self.children.get(parent, 0) + 1

    def _remove(self, key) -> None:
        e = self.index.pop(key, None)
        if e is None:
            return
        try:
            os.unlink(e["path"])
        except FileNotFoundError:
            pass
        if e["parent"] is not None:
            self.children[e["parent"]] = self.children.get(e["parent"], 1) - 1

    def bytes(self) -> int:
        return sum(e["size"] for e in self.index.values())

    def _touch(self, key) -> None:
        now = time.time()
        while key is not None and key in self.index:
            e = self.index[key]
            e["stamp"] = now
            try:
                os.utime(e["path"], (now, now))
            except OSError:
                pass
            key = e["parent"]

    def _read_pins(self, name: str) -> list:
        try:
            with open(os.path.join(self.dir, name)) as f:
                return [tuple(k) for k in json.load(f)]
        except (OSError, ValueError, TypeError):
            return []

    def _pin_short(self, key) -> None:
        """A prefix shorter than one block (a system preamble, parent-less) is a few MiB and is asked for by every new session, but it is also the oldest, never-extended leaf
        that LRU removes first once the cap is full of long chains. The newest PIN_SHORT of them go to `pins.auto.json` (a separate file: `pins.json` belongs to the bench tools and is
        never rewritten here); they stay out of the cap like any pin. Older ones fall back to LRU."""
        keys = [k for k in self._read_pins(PINS_AUTO) if k in self.index and k != key] + [key]
        keys = keys[-PIN_SHORT:]
        tmp = os.path.join(self.dir, f".{uuid.uuid4().hex}.pins.tmp")
        with open(tmp, "w") as f:
            json.dump([list(k) for k in keys], f)
        os.replace(tmp, os.path.join(self.dir, PINS_AUTO))

    def _pinned(self) -> set:
        """Keys of `pins.json` (and of `pins.auto.json`) in the namespace directory and every file below them: bench chains that must outlive any traffic (tools/depth_ab.py --build)."""
        keys = self._read_pins(PINS) + self._read_pins(PINS_AUTO)
        out: set = set()
        for key in keys:
            while key is not None and key in self.index and key not in out:
                out.add(key)
                key = self.index[key]["parent"]
        return out

    def _evict(self) -> None:
        if not self.cap or self.bytes() <= self.cap:
            return
        pinned = self._pinned()                                              # outside the cap: traffic keeps its whole budget
        kept = sum(self.index[k]["size"] for k in pinned)
        while self.bytes() - kept > self.cap:
            leaves = [k for k in self.index if not self.children.get(k) and k not in pinned]
            if not leaves:
                break
            victim = min(leaves, key=lambda k: self.index[k]["stamp"])
            self._remove(victim)
            self.stats["evicted"] += 1

    # -- save
    def has(self, ids, b: int) -> bool:
        return (b, digest(ids, b).hex()) in self.index

    def save(self, ids, b: int, cache, extras=None, stale: bool = False, dg: bytes | None = None, pdg: bytes | None = None) -> int:
        """Write the checkpoint of the first b tokens (cache.offset == b: call it at a block boundary or a cut of the prefill; an off-grid b hangs
        off the checkpoint of the block below it, which must be on disk).
        extras: list of mx.array or None (the drafter ring). stale: the cache comes from a chunk that skipped its decoder half (ced_skip_dec=3): its decoder window
        rings and the drafter ring are not the full prefill's, so the file is marked and `lookup` hands it out only where the caller says the rings will be rewritten. Returns the bytes written, 0 when it exists already or its parent is missing."""
        if b <= 0 or b > len(ids):
            raise ValueError(f"checkpoint at {b} tokens: not inside this prompt ({len(ids)} tokens)")
        dg = (digest(ids, b) if dg is None else dg).hex()                # dg / pdg: the caller's incremental digests of ids[:b] and of the parent length (KYOJIN_KV_DIGEST_INC), else the O(b) one-shot hash
        key = (b, dg)
        if key in self.index:
            self._touch(key)
            return 0
        pb = b - self.block if b % self.block == 0 else b // self.block * self.block     # parent: the block below (an off-grid b: the grid point under it)
        pdg = (digest(ids, pb) if pdg is None else pdg).hex() if pb > 0 else None
        pkey = (pb, pdg) if pdg else None
        if pkey is not None and pkey not in self.index:
            self.stats["skipped"] += 1                       # the chain below it is not on disk (evicted, or a prefix hit in memory started above it)
            return 0
        t0 = time.time()
        ad = adapter_for(cache)
        fixed, grow, rows, meta = ad.dump(cache, self.index[pkey]["rows"] if pkey else {})
        if meta["offset"] is None:                           # KYOJIN_GENERIC_CACHE: no member of the cache counts tokens (pure SSM): the state is the state after the caller's b tokens
            meta["offset"] = b
        if meta["offset"] != b:
            raise ValueError(f"cache holds {meta['offset']} tokens, checkpoint asked at {b}")
        tensors = {"f." + k: v for k, v in fixed.items()}
        tensors.update({"g." + k: v for k, v in grow.items()})
        ex = list(extras) if extras else []
        tensors.update({f"x.{i}": a for i, a in enumerate(ex) if a is not None})
        head = dict(version=VERSION, ns=self.ns, b=b, block=self.block, digest=dg, parent=pdg, pb=pb, adapter=ad.name, rows=rows, meta=meta,
                    extras=[a is not None for a in ex], **({"stale": True} if stale else {}))
        tmp = os.path.join(self.dir, f".{uuid.uuid4().hex}{TMP}")
        path = os.path.join(self.dir, f"{b:09d}-{dg}{SUFFIX}")
        mx.save_safetensors(tmp, tensors, {"kyojin_kv": json.dumps(head)})
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)                                     # data on the device before the rename: a power cut never leaves a named file with zero pages
        finally:
            os.close(fd)
        os.replace(tmp, path)
        self._add(path, head)
        self._touch(key)
        size = self.index[key]["size"]
        if pkey is None and b < self.block and self.cap:
            try:
                self._pin_short(key)
            except OSError as e:                                                 # a pin that cannot be written costs only its eviction protection
                self.log(f"kvdisk: pin of {b} not written ({type(e).__name__})")
        self._evict()
        self.stats["saved"] += 1
        self.stats["bytes_saved"] += size
        self.stats["save_s"] += time.time() - t0
        return size

    # -- load
    def lookup(self, ids, below: int | None = None, above: int = 0, ok=None):
        """(b, cache, extras) of the longest checkpoint that is a proper prefix of ids (b < len(ids): one token is left to prefill), or (0, None, None).
        above: only checkpoints longer than this (a caller whose in-memory snapshot already covers that many tokens gains nothing from a shorter
        file, and the load is skipped); below: only checkpoints shorter than this. ok(b) -> bool: whether a checkpoint marked stale at b tokens may be used
        (the caller's remaining prompt starts with a chunk that rewrites the stale part); a stale file it refuses is skipped like a missing one, so the
        answer is the nearest earlier whole checkpoint. `self.last_stale` tells whether the checkpoint returned is marked, `self.refused` how many
        stale ones the call passed over."""
        self.last_stale, self.refused = False, 0
        n = len(ids)
        top = n if below is None else min(n, below)
        off = {k[0] for k in self.index if k[0] % self.block}                    # off-grid checkpoints (cuts)
        lens = sorted({*range(self.block, top, self.block), *(b for b in off if b < top)})
        cands = [(b, dg.hex()) for b, dg in prefix_digests(ids, [b for b in lens if b > above]) if (b, dg.hex()) in self.index]
        for b, dgx in reversed(cands):
            stale = self.index[(b, dgx)].get("stale", False)
            if stale and (ok is None or not ok(b)):
                self.refused += 1
                continue
            try:
                t0 = time.time()
                out = self._load_chain(ids, b)
                self.stats["hits"] += 1
                self.stats["load_s"] += time.time() - t0
                self.last_stale = stale
                return b, out[0], out[1]
            except Exception as e:                                               # noqa: BLE001
                self.log(f"kvdisk: checkpoint at {b} not usable ({type(e).__name__}: {str(e)[:80]}); trying a shorter prefix")
                self.stats["bad"] += 1
        self.stats["misses"] += 1
        return 0, None, None

    def _load_chain(self, ids, b: int):
        keys, k = [], (b, digest(ids, b).hex())
        while k is not None:
            if k not in self.index:
                raise FileNotFoundError(f"chain link {k[0]} missing")
            keys.append(k)
            k = self.index[k]["parent"]
        keys.reverse()
        parts, size = [], 0
        for k in keys:
            e = self.index[k]
            try:
                t = mx.load(e["path"])
                h = _header(e["path"])
            except Exception:
                self._remove(k)                                                  # unreadable: drop it (and with it everything above)
                raise
            if h["digest"] != k[1] or h["ns"] != self.ns:
                self._remove(k)
                raise ValueError("header does not match its name")
            parts.append((t, h["meta"], h))
            size += e["size"]
        ad = next(a for a in ADAPTERS if a.name == parts[-1][2]["adapter"])
        cache = ad.restore(self.make_cache, [(t, m) for t, m, _ in parts])
        mx.eval(*[a for t, _, _ in parts for a in t.values()], *self._arrays(cache))
        h = parts[-1][2]
        extras = None
        if h["extras"]:
            last = parts[-1][0]
            extras = [last[f"x.{i}"] if present else None for i, present in enumerate(h["extras"])]
        self._touch(keys[-1])
        self.stats["loaded"] += 1
        self.stats["bytes_loaded"] += size
        return cache, extras

    @staticmethod
    def _arrays(cache):
        from mlx.utils import tree_flatten
        return [a for c in cache for _, a in tree_flatten(c.state) if isinstance(a, mx.array)]

    def describe(self) -> str:
        s = self.stats
        return (f"kvdisk {self.dir}: {len(self.index)} files {self.bytes() / 2 ** 20:.0f} MiB (cap {self.cap / 2 ** 20:.0f} MiB), saved {s['saved']} "
                f"({s['bytes_saved'] / 2 ** 20:.0f} MiB, {1000 * s['save_s'] / max(1, s['saved']):.0f} ms each), hits {s['hits']} misses {s['misses']} "
                f"loaded {s['bytes_loaded'] / 2 ** 20:.0f} MiB ({1000 * s['load_s'] / max(1, s['hits']):.0f} ms each), evicted {s['evicted']} bad {s['bad']}")
