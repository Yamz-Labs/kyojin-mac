# SPDX-License-Identifier: Apache-2.0
"""Continuous-batching engine for the Mac server.

One thread owns every MLX call. Requests arrive from the HTTP threads through `Engine.submit`; each gets a `Handle` whose queue
carries ("tok", id) per generated token and a final ("done", reason, stats). Per loop iteration:

    1. admit waiting requests while fewer than `sessions` are active (prefilling + decoding)
    2. run ONE prefill chunk of the oldest prefilling session (a long prompt never freezes the other users: between two chunks
       the decoding users get `decode_per_chunk` batched steps)
    3. run one batched decode step for the decoding users; a user that finished leaves and a user whose prefill is done joins
       between steps (the batched cache is rebuilt from the per-user caches only when the membership changes)

Prefix sharing: the prompt is always prefilled on a fixed grid of absolute positions (`chunk`); every `block` tokens (a multiple of
`chunk`) the cache is kept as a snapshot keyed by a hash of the token prefix. A new session forks the longest snapshot that is a prefix
of its prompt and prefills the rest. The grid makes a hit do exactly the same work as a miss, so the result is bit-identical.
Cuts: the chat layer passes `cuts`, token positions that are safe reuse points (end of the system block, end of the conversation before the
generation prompt). A chunk ends at the next cut, and a snapshot (memory and disk) is also kept there, so a preamble shorter than a block, or the tail
past the last block, is not prefilled again; lookup tests the off-grid lengths that are stored, and the prefill returns to the grid after the cut.
Prefill is one session at a time, so a burst of requests with a shared prompt finds the first one's snapshots when its turn comes.

Disk tier (`disk`, a kyojin_mlx.kvdisk.KvDisk): every `disk.block` tokens (8192, a multiple of `chunk`) the cache is also written to the SSD, keyed
by the same hash. A lookup that the memory snapshots cannot answer (restart of the server, snapshots dropped for memory) or answer with fewer
tokens loads the longest checkpoint from disk, bit for bit what the prefill had, and puts it back into the memory tier. A prompt is never
prefilled twice.

A reply that outlasts its share of the context (rq_budget, as the box servers) goes to the back of the queue when other requests wait
for a slot; its cache is kept and it rejoins the batch when a slot is free.

Speculative decode (`mtp_depth` > 0 and a backend with `supports_spec`): every decode tick is one verify pass. Each active session contributes
its pending token and up to D drafts (D adapts to the session's acceptance, 0 allowed), so the pass has sum(D_i + 1) rows; each session keeps the
longest prefix its own drafts got right plus the model's token after it, and its cache rolls back to that point. Every row is bit-identical to the
solo decode step of its position (kyojin_mlx.spec), so the tokens of a greedy session do not depend on D or on the other sessions. A session
that samples (temperature > 0) drafts too: each verify row is sampled from the target distribution (temperature / top-p / top-k / min-p applied) with
its own key, and a draft is kept when it equals that sample. The drafter is greedy, so "accept with probability p(x)" is exactly "draw x from p and
compare": the output distribution is the one of plain sampling (rows after the first rejection are discarded, the token drawn at the rejected row is
the correction, and the bonus row is a draw from p). Greedy sessions are unchanged. Sessions hold their own caches (no batched cache), look-ahead is off.
`KYOJIN_SPEC_FULL` accepts every draft the pass verified (not only the confidence-gated prefix), `KYOJIN_SPEC_RS` draws the drafts from the draft
head's own q and verifies them by rejection sampling with the residual correction (kyojin_mlx.specsam: exact target distribution, acceptance
1 - TV(p, q) instead of p(top1)), `KYOJIN_SPEC_GATE` takes the gate's threshold from the measured P(accept | conf), `KYOJIN_SPEC_LOOKUP` drafts
from a per-session 4-gram index of the committed tokens (kyojin_mlx.pld: no draft-head forward for a copied span, gated on tokens per ms).
`KYOJIN_SPEC_LADDER` picks the width from a measured convex cost ladder (kyojin_mlx.ladder: median round ms per depth, a line fitted from
three depths, argmax expected tokens per priced ms against the measured plain rate, a low-rate probe one row past the best).

Memory (`kv_budget`, bytes): the caches of all sessions that hold one (active, and parked ones that yielded their slot) plus the prefix snapshots
are counted. When the total is over the budget the KV of the oldest parked session is dropped; on resume it is rebuilt: its prompt is prefilled on
the same grid (prefix snapshots are reused) and the tokens it had produced are replayed through the decode path, so the cache holds the bits it
had before and the reply goes on identical. Active sessions are never dropped; a new request is not admitted while the total is over the budget."""
from __future__ import annotations

import bisect
import collections
import json
import math
import os
import hashlib
import itertools
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

import mlx.core as mx

from . import ticktrace as TT
import numpy as np

from .sampling import Sampling, sample_rows
from . import hostloop as HL
from . import pysample as PS
from . import roundlog as RLOG
from .. import dspol as DSPOL
from .. import dspol2 as DSP2
from .. import fired as FD
from .. import specsam as SPECSAM
from .. import hostsync as HS
from .. import gpuspan as GS
from .. import plainsplit as PLS
from .. import fired as FD
from .. import pld as PLD
from .. import ds41_vision
from .. import copyv2 as CV2
from .. import ladder as LADDER
from ..kvdisk import prefix_digests
from .. import applegpu

GPU_KEEPALIVE_S = float(os.environ.get("KYOJIN_GPU_KEEPALIVE_S", "0.5"))     # flags.py gpu_keepalive; 0 = off
GPU_KEEPALIVE_FOR_S = 3600                                                   # stop an hour after the last work: the next request then pays the wake-up once


def mu_auto_on(backend, greedy: bool = True) -> bool:
    """Does one verify pass serve all sessions of this backend (so a round plan is worth tuning from measured rates)?
    A backend that declares `multi_auto` decides (Q35MlxBackend: the fixed depth is unset); one that does not gets it from flag
    mu_tuner (off by default, read per tick) on greedy rounds only: the tuner measures greedy arms. MultiTuner: serve/mu_tuner.py."""
    auto = getattr(backend, "multi_auto", None)
    if auto is not None:
        return bool(auto)
    return greedy and os.environ.get("KYOJIN_MU_TUNER", "0") not in ("0", "")


def ttft_on() -> bool:
    """Flag ttft_stages (read per request): the per-request time to first token is split into its stages. Costs one perf_counter pair per stage
    (~50 ns each, no GPU work, nothing per token), so it is on by default; 0 = the stages are not collected and no dict is made."""
    return os.environ.get("KYOJIN_TTFT_STAGES", "1") not in ("0", "")


RQ_MIN_TOKENS = 2048
GIB = 2 ** 30
PF_SRPT_AGE = 2000.0                                                          # flag pf_srpt: tokens of priority per second of waiting (about one prefill's speed)


def free_unified_bytes() -> int | None:
    """Memory a process can still take on this Mac: free + inactive + speculative + purgeable pages (vm_stat), at most the GPU working-set
    room (recommended working set minus what MLX holds). None when it cannot be read (not macOS)."""
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = sum(int(m.group(1)) for k in ("free", "inactive", "speculative", "purgeable")
                    for m in [re.search(rf"Pages {k}:\s+(\d+)", out)] if m)
    except Exception:
        return None
    room = pages * page
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        room = min(room, int(info["max_recommended_working_set_size"]) - mx.get_active_memory())
    except Exception:
        pass
    return max(room, 0)


KV_HEADROOM = 0.175                                                          # share of the RAM left to macOS, its file cache and the wired overhead (the lane's wired memory is about active + buffer cache + 4 GiB). 31/10 on 256 GiB: at 215 GiB wired the file cache was down to 31 GiB; the 22:20 panic came at 220 GiB resident.


SMALL_RAM_RESERVE_GIB = 6.0                                                  # fixed bytes a machine needs outside the GPU set: macOS + apps (~3.5), this process's host heap (~1), and the 1.5 GiB free the memory guards keep. Binds below ~34 GiB of RAM (0.175 x RAM is less above); 16 GiB Air 03/11: without it the budget filled the recommended working set and left 5 % free.


def kv_budget_for(total: int, recommended: int, active: int, cache_limit: int, margin_gib: float = 1.0, floor_gib: float = 0.25) -> int:
    """KV budget from the fixed terms of the machine: what the GPU may hold, less the RAM headroom, the loaded weights and the MLX buffer cache
    (which fills up in use, so it counts in full). It does not depend on what the page cache happens to hold at load time."""
    return int(max(kv_ceiling(total, recommended, margin_gib) - active - cache_limit, floor_gib * GIB))


def kv_ceiling(total: int, recommended: int, margin_gib: float = 1.0) -> float:
    """Most the GPU set (weights, buffer cache, KV) may reach on this machine: the recommended working set less a margin, the RAM less the share macOS keeps,
    and the RAM less a fixed reserve on a small machine."""
    return min(recommended - margin_gib * GIB, total * (1 - KV_HEADROOM), total - SMALL_RAM_RESERVE_GIB * GIB)


def default_kv_budget(margin_gib: float = 1.0, floor_gib: float = 0.25) -> int | None:
    """Default `kv_budget`, after the weights are loaded and the buffer cache limit is set. Off macOS: free memory minus a margin; None when unknown."""
    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = mx.set_cache_limit(0)                                        # the only way to read the limit; nothing is allocated before it is put back
        mx.set_cache_limit(limit)
        fixed = kv_budget_for(int(info["memory_size"]), int(info["max_recommended_working_set_size"]), mx.get_active_memory(), limit, margin_gib, floor_gib)
    except Exception:
        fixed = None
    if fixed is not None:
        free = free_unified_bytes()                                          # another heavy process (a resident lane) holds RAM the fixed terms do not see
        return fixed if free is None else int(max(min(fixed, free - limit - margin_gib * GIB), floor_gib * GIB))
    try:
        free = free_unified_bytes()
        return None if free is None else int(max(free - margin_gib * GIB, floor_gib * GIB))
    except Exception:
        return None


def rq_budget(ctx: int | None, sessions: int, prompt_tokens: int, max_new: int, page: int = 256, floor: int | None = None) -> int | None:
    """Reply tokens a session may produce before it yields its slot to waiting requests (the box servers' serve_compat.rq_budget with
    no speculative window). None for one session and when the whole reply fits the share."""
    if sessions <= 1 or not ctx:
        return None
    room = ctx // page * page - prompt_tokens
    share = ctx // sessions // page * page - prompt_tokens
    budget = min(max(share, RQ_MIN_TOKENS if floor is None else floor), room)
    if budget < 1 or max_new + 1 <= budget:
        return None
    return budget


MAX_PREFILL_CHUNK = 8192                                      # hard cap of every prefill chunk
K_WINDOW = 5                                                  # KYOJIN_MTP_POLICY=robust: drafted-round cost samples behind the median spec_k
RATE_MARGIN = 0.9                                             # KYOJIN_SPEC_RATE: plain only when the cap's rate is below 0.9 x plain's (served 30/10: plain_ms
                                                              # EMA read 26-35 ms against a true 29.5, and the drops it caused cost 9 % on two texts)
RATE_MIN = 6                                                  # KYOJIN_SPEC_RATE: cap rounds of a fresh window before its measured rate can switch the session to plain
PROBE_CAP = 8                                                 # ... and the longest probe gap / floor, in probe_every units (default policy: 64 / 4096 rounds)
NONDET_FIX = os.environ.get("KYOJIN_NONDET_FIX", "0") not in ("0", "")   # flags.py nondet_fix: the round width may not come from a measured ms
TRIM_RATE = 0.3                                               # KYOJIN_LAZY_TRIM: decay of the per-session EMA of proposals the confidence gate keeps
TRIM_PROBE = 8                                                # ... and every 8th drafted round of a session asks the full plan (keeps the deep positions measured, lifts the censoring)
PAIR_WINDOW = 12                                              # KYOJIN_RATE_PAIR: drafted and plain rounds behind the paired rate (tokens and ms summed over the same rounds)
PAIR_MIN_D = 3                                                # ... samples a drafted / a plain rate needs before the paired rule decides
PAIR_MIN_P = 2
_GREEDY = Sampling()                                          # the argmax row of sample_rows: a spec_rs session samples its rows in specsam.decide
GATE_CANDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)             # flag gate_adapt: confidence thresholds the round's own reward prices
GATE_MARGIN = 1.04                                           # ... a candidate replaces the current one only when its measured tokens per ms beat it by 4 %
SPEC_ROW_MS = 5.5                                            # fallback per-row cost of a verify round (ms) before the engine's own fit exists


def _bound_submits() -> int:
    from .. import bound_layers as BL
    return BL.stats()["submits"]


ORACLE_PATH = os.environ.get("KYOJIN_ORACLE_IDS", "scratch/oracle.json")   # flag oracle_draft: the recorded continuation file


def _oracle_read(path: str | None = None):
    """(prompt tag, ids) of the recorded greedy continuation (flag oracle_draft), or (None, []) when there is none."""
    p = path or os.environ.get("KYOJIN_ORACLE_IDS", ORACLE_PATH)
    try:
        with open(p) as f:
            d = json.load(f)
        return str(d.get("ph") or ""), [int(x) for x in (d.get("ids") or [])]
    except (OSError, ValueError, TypeError):
        return None, []

class Handle:
    """What the HTTP thread holds for one request."""

    def __init__(self, engine: "Engine"):
        self.events = HL.new_queue()                    # ("tok", id) per token, then ("done", reason, stats)
        self.cancelled = threading.Event()
        self._engine = engine

    def cancel(self) -> None:
        self.cancelled.set()
        self._engine.wake()


@dataclass(eq=False)
class Seed:
    """Flag pf_share: the cache of a prefilling session at one of its chunk ends, handed to the sessions that share that prefix and have not started. One object for
    all of them (the bytes are counted once). `snap` is a fork (a view: no byte is copied until a holder writes), `dsnap` the drafter ring snapshot."""
    end: int
    snap: object
    dsnap: object
    nbytes: int
    path: tuple = ()                # the chunk ends that built `snap`


@dataclass(eq=False)
class Session:
    sid: int
    ids: list
    max_new: int
    sampling: Sampling
    stop_ids: frozenset
    handle: Handle
    vl: object = None               # image spans (kyojin_mlx.ds41_vision.Span) of a prompt with images: `ids` holds virtual ids there
    rids: object = None             # the ids the model reads: `ids` itself without an image, image_token_id over the spans otherwise
    cuts: tuple = ()                # sorted token positions where a prefix snapshot is also kept (see the module doc)
    yield_after: int | None = None
    seed: int = 0
    key: object = None
    cache: object = None
    pos: int = 0                    # prompt tokens already in the cache
    cached: int = 0                 # prompt tokens that came from a prefix snapshot
    hasher: object = None
    dg_grid: tuple = (0, None)      # (position, digest of the prompt up to it) at the last disk-grid point this session passed (KYOJIN_KV_DIGEST_INC)
    dg_prev: tuple = (0, None)      # the grid point before it (the parent of a checkpoint taken at the last one)
    n_gen: int = 0
    seg_gen: int = 0                # tokens since the session (re)joined the batch
    last: int = -1                  # last emitted token, not yet fed to the model
    out: list = field(default_factory=list)     # every emitted token (a dropped cache is rebuilt from the prompt and these)
    draft: object = None            # the backend's draft state (speculative decode)
    kv_bytes: int = 0               # size of `cache` when it was last measured
    t_parked: float = 0.0           # when the session yielded its slot
    resume: bool = False            # cache was dropped: prompt prefill, then replay of out[:-1]
    replay_pos: int = 0
    acc_succ: float = 0.0           # decayed counts of accepted drafts / draft trials (speculative decode)
    acc_trial: float = 0.0
    pos_succ: list = field(default_factory=list)    # the same per draft position (position i reached = a trial of it)
    pos_trial: list = field(default_factory=list)
    fl_t0: float = 0.0              # flag pf_floor_catchup: when this decoder's floor window opened (first burst check with a prefill running; 0 = closed)
    fl_n0: int = 0                  # n_gen at that moment
    idle_rounds: int = 0            # rounds at D = 0 since the last probe
    probe_gap: int = 0              # rounds at D = 0 between two probes (0 = the engine's probe_every; doubles while the probes fail)
    probing: bool = False           # the round in flight is a probe
    cap_rounds: int = 0             # rounds below the depth cap since the last round at the cap (measured policy)
    cap_tok: float = 0.0            # tokens per round at the depth cap: the mean of the window, then decayed 0.9 (KYOJIN_SPEC_RATE)
    cap_n: int = 0                  # rounds at the depth cap in the window (a probe opens a fresh one)
    rate_gap: int = 0               # rounds at D = 0 between two cap probes (0 = probe_every; doubles while a probe confirms plain)
    rounds: int = 0
    t_submit: float = field(default_factory=time.perf_counter)
    t_tok: float = 0.0              # when the last token of this reply was accepted
    park_now: bool = False          # flag kv_admit: leave the batch at the next round (the memory budget is over and nothing else can be dropped)
    share_seed: object = None       # flag pf_share: Seed from a session that shares this prompt's prefix (used by _begin_prefill)
    grid: object = None             # flag pf_share: the chunk ends of this prompt on its own grid (memo)
    ids_np: object = None           # flag pf_share: ids as an int32 array (memo)
    path: object = ()               # chunk ends of the prefill that built `cache` (a tuple; None = unknown, e.g. a disk checkpoint): a snapshot is only exact for a prompt whose own grid has this path
    share_t: dict = field(default_factory=dict)     # flag pf_share: other session id -> the chunk end this session hands it (0 = none)
    t_start: float = 0.0
    t_pf_turn: float = 0.0          # flag pf_fair: when this session last got a prefill chunk (0 = never): the queue serves the oldest turn next
    stages: object = None           # flag ttft_stages: the time to first token, split by stage (dict; None = not collected)
    t_first: float = 0.0
    t_done: float = 0.0
    prefilled: bool = False
    lookup_index: object = None             # kyojin_mlx.pld.Index over prompt + committed output (flag spec_lookup)
    lk_ema: float = 0.0                     # accepted lookup drafts per lookup round (decayed; drifts back on MTP rounds)
    mt_ema: float = 0.0                     # accepted MTP drafts per MTP round (decayed)
    lk_streak: bool = False                 # the last lookup round landed every draft
    lk_full_n: int = 0                      # ... and how many consecutive lookup rounds did (flag lookup_confirm)
    lk_adapt: object = None                 # flag spec_lookup_adapt: pld.SuffixBuckets (short-suffix acceptance), created on first use
    lk_suffix: int = 0                      # agreement of the lookup draft planned for this round (read by _spec_update)
    cp_scale: float = 1.0                   # flag copy_v2: confidence scale of the session's copy matches (halved by a spent match)
    skip_hold: int = 0                      # flag spec_skip: rounds left on the plain step before the draft head is tried again
    skip_gap: int = 0                       # flag spec_skip: the hold after the next all-cut round (doubles per consecutive one, 0 after a kept draft)
    kept_ema: float = -1.0                  # flag lazy_trim: decayed proposals the confidence gate keeps per drafted round (-1 = none seen)
    trim_n: int = 0                         # ... and the drafted rounds since the last untrimmed one (every TRIM_PROBE-th round asks the full plan)
    gate_thr: float = 0.5                   # flag gate_adapt: the confidence threshold the next round drafts with
    gate_rate: list = field(default_factory=list)   # ... and the decayed tokens per ms measured for every candidate threshold
    dsp2: object = None                     # flag dspol_v2: dspol2.State (confidence track + the last decision, for the trace)
    anchor: object = None                   # flag kv_tail: the chunk end where the tail anchor snapshot is taken (0 = none, None = not computed)

    def __post_init__(self):
        if self.rids is None:
            self.rids = self.ids

    @property
    def n_prompt(self) -> int:
        return len(self.ids)


LK_KEEP_MIN = 65536        # below this the prompt-lookup insert takes under 7 ms
LK_KEEP = 4                # kept indexes, up to 48 MB each


class PrefixCache:
    """LRU of cache snapshots at multiples of `block` tokens, keyed by (length, hash of the token prefix). `disk` (optional KvDisk): a second tier
    of checkpoints at multiples of `disk.block` tokens that survives a restart and the memory budget."""

    @property
    def disk(self):
        """The disk tier; None while kv_disk_off is set, so a bench that needs no disk tier switches a flag instead of reloading the model."""
        return None if os.environ.get("KYOJIN_KV_DISK_OFF", "0") == "1" else self._disk

    def __init__(self, backend, block: int, budget_bytes: int, disk=None, log=None):
        self.backend, self.block, self.budget = backend, block, budget_bytes
        self._disk, self.log = disk, log or (lambda *a: None)
        self.entries: collections.OrderedDict = collections.OrderedDict()   # key -> (snapshot, nbytes, draft snapshot)
        self.bytes = 0
        self.parts: dict = {}                       # key -> {id: bytes} of the shared segments an entry holds (KYOJIN_KVSEG); `refs` counts the holders: a segment is billed once
        self.refs: dict = {}                        # id -> [holders, bytes]
        self.stale: set = set()                     # keys whose snapshot comes from a chunk that skipped its decoder half (ced_skip_dec=3): see `lookup`
        self.fallbacks = 0                          # lookups that passed over a stale snapshot they could not use (the answer is an earlier whole one, or none)
        self.off = collections.Counter()            # off-grid lengths (cuts) held in `entries` -> count; shrinks with eviction
        self.last_draft = None                      # draft snapshot of the entry the last lookup hit
        self.hits = self.misses = self.stored = 0
        self.sigs: dict = {}                        # key -> the chunk ends that built the snapshot (a tuple)
        self.last_sig = None
        self.too_big = 0                            # snapshots not kept because one alone is over the budget (a long prefix: the next session prefills it again)
        self.off_grid_skipped = 0                   # flag pf_share: snapshots passed over because their length is not on the asking prompt's grid
        self.disk_hits = self.disk_hit_tokens = self.disk_saved = self.disk_errors = 0

    @staticmethod
    def hasher():
        return hashlib.blake2b(digest_size=16)

    def lookup(self, ids: list, ok=None, grid=None):
        """(length, cache) of the longest snapshot (memory, else disk) that is a proper prefix of ids, or (0, None). The cache is the caller's own.
        ok(b) -> bool (ced_skip_dec=3): whether a snapshot marked stale at b tokens may be used, i.e. the prompt after b starts with a chunk that rewrites the
        stale decoder rings; without `ok` a stale snapshot is never returned. A refused stale snapshot makes the answer the nearest earlier whole one (or 0).
        grid(b, sig) -> bool (flag pf_share): sig is the chunk path that built the snapshot (None = unknown). A snapshot is exact for the asking prompt only when that
        path is the prompt's own grid up to b: another partition (a cut the asking prompt does not have) leaves other bits in the cache than its solo prefill."""
        self.last_draft = None
        self.last_sig = None
        self.refused = 0
        b, cache = self._lookup_memory(ids, ok, grid)
        if self.disk is not None:
            hit = self._lookup_disk(ids, above=b + self.block // 4, ok=ok)
            if hit is not None and grid is not None and not grid(hit[0], None):
                hit = None
            if hit is not None:
                self.last_sig = None     # a disk load costs ~0.85 s on the served 190 GiB model (~2300 tokens of prefill): below that gain, prefill is faster
            if hit is not None:
                b, cache = hit
        self.fallbacks += bool(self.refused)
        if b:
            self.hits += 1
        else:
            self.misses += 1
        return b, cache

    def _lookup_memory(self, ids: list, ok=None, grid=None):
        n = len(ids)
        if self.budget <= 0 or not self.entries:
            return 0, None
        lens = sorted({*range(self.block, n, self.block), *(b for b in self.off if b < n)})     # b < n: at least one token is left to prefill
        keys = prefix_digests(ids, lens)
        for key in reversed(keys):
            e = self.entries.get(key)
            if e is not None:
                if grid is not None and not grid(key[0], self.sigs.get(key)):
                    self.off_grid_skipped += 1
                    continue
                if key in self.stale and (ok is None or not ok(key[0])):
                    self.refused += 1
                    continue
                self.entries.move_to_end(key)
                self.last_draft = e[2]
                self.last_sig = self.sigs.get(key)
                return key[0], self.backend.restore(e[0])
        return 0, None

    def _lookup_disk(self, ids: list, above: int, ok=None):
        """The longest checkpoint on disk longer than `above` tokens, loaded and also kept in the memory tier; None when there is none."""
        try:
            b, cache, extras = self.disk.lookup(ids, above=above, ok=ok)
            self.refused += self.disk.refused
        except Exception as e:                                          # noqa: BLE001  (a disk problem costs a prefill, never the request)
            self.disk_errors += 1
            self.log(f"kv-disk: lookup failed ({type(e).__name__}: {str(e)[:100]})")
            return None
        if not b:
            return None
        self.disk_hits += 1
        self.disk_hit_tokens += b
        size = self.backend.nbytes(cache) + sum(a.nbytes for a in extras or [] if a is not None)
        key = prefix_digests(ids, [b])[0]
        if key not in self.entries:
            self._put(key, self.backend.fork(cache), size, extras)
            if self.disk.last_stale and key in self.entries:
                self.stale.add(key)
        self.last_draft = extras
        return b, cache

    def store(self, key, cache, draft=None, stale: bool = False, sig=None) -> None:
        if self.budget <= 0:
            return
        if key in self.entries:
            if stale or key not in self.stale:
                return
            old = self.entries.pop(key)
            if old[2] is not None and self.last_draft is old[2]:
                self.last_draft = None                                 # the entry (and its draft snapshot) is gone: do not pin it
            self._release(key, old[1])                                 # a whole snapshot replaces a stale one of the same prefix
            self.stale.discard(key)
            self.sigs.pop(key, None)
            if key[0] % self.block:
                self.off[key[0]] -= 1
                if self.off[key[0]] <= 0:
                    del self.off[key[0]]
        snap = self.backend.snapshot(cache)
        size = self.backend.nbytes(snap) + (self.backend.draft_nbytes(draft) if draft is not None else 0)
        self._put(key, snap, size, None if draft is None else self.backend.draft_snapshot(draft), sig)
        if stale and key in self.entries:
            self.stale.add(key)

    def _put(self, key, snap, size: int, dsnap, sig=None) -> None:
        if size > self.budget:
            self.too_big += 1
            return
        parts = self.backend.shared_parts(snap)
        own, billed = size - sum(parts.values()), 0                      # KYOJIN_KVSEG: shared segments are billed once, to the first entry that holds them
        if parts:
            self.parts[key] = parts
            for i, nb in parts.items():
                r = self.refs.setdefault(i, [0, nb])
                billed += nb if r[0] == 0 else 0
                r[0] += 1
        self.entries[key] = (snap, own, dsnap)
        if sig is not None:
            self.sigs[key] = sig
        if key[0] % self.block:
            self.off[key[0]] += 1
        self.bytes += own + billed
        self.stored += 1
        self.shrink(self.budget)

    def save_disk(self, ids: list, end: int, cache, draft=None, stale: bool = False, dg=None, pdg=None) -> None:
        """Checkpoint the first `end` tokens to disk; the drafter ring goes along. A failure is logged, never raised. dg / pdg: the digests of ids[:end] and of the parent
        length when the caller has them incrementally (KYOJIN_KV_DIGEST_INC); without them the disk tier hashes the whole prefix, twice, per checkpoint."""
        if os.environ.get("KYOJIN_KV_DISK_RO", "0") == "1":
            return
        try:
            if self.disk.save(ids, end, cache, None if draft is None else self.backend.draft_snapshot(draft), **({"stale": True} if stale else {}),
                              **({"dg": dg} if dg is not None else {}), **({"pdg": pdg} if pdg is not None else {})):
                self.disk_saved += 1
        except Exception as e:                                          # noqa: BLE001
            self.disk_errors += 1
            self.log(f"kv-disk: checkpoint at {end} failed ({type(e).__name__}: {str(e)[:100]})")

    def shrink(self, limit: int) -> bool:
        """Drop least recently used snapshots until at most `limit` bytes are left. True when something was dropped."""
        dropped = False
        while self.bytes > limit and self.entries:
            key, (_, sz, dsnap) = self.entries.popitem(last=False)
            if dsnap is not None and self.last_draft is dsnap:
                self.last_draft = None                                 # the draft snapshot went with its entry: a stale hit must not keep it resident
            n = key[0]
            self.stale.discard(key)
            self.sigs.pop(key, None)
            if n % self.block:
                self.off[n] -= 1
                if self.off[n] <= 0:
                    del self.off[n]
            self._release(key, sz)
            dropped = True
        return dropped

    def _release(self, key, own: int) -> None:
        """Take an entry's bytes off the ledger: its private part, and each shared segment it was the last holder of."""
        self.bytes -= own
        for i in self.parts.pop(key, ()):
            r = self.refs[i]
            r[0] -= 1
            if r[0] == 0:
                self.bytes -= r[1]
                del self.refs[i]


class Engine:
    @property
    def _dspol_on(self) -> bool:
        """Confidence-head width policy (kyojin_mlx.dspol) in solo rounds; read at call time so POST /admin/flags switches it on a loaded server."""
        return os.environ.get(DSPOL.ENV_ENABLE, "1") != "0"

    @property
    def _dspol_v2(self) -> int:
        """Flag dspol_v2 (0 off, 1 plan-time rows, 2 plus the post-draft choice); read at call time. Solo rounds only."""
        return DSP2.enabled()

    @property
    def _spec_full(self) -> bool:
        """Accept every draft the verify pass already paid for, not only the confidence-gated prefix (flag spec_full; read per round)."""
        return os.environ.get("KYOJIN_SPEC_FULL", "0") != "0"

    @property
    def _spec_rs(self) -> bool:
        """Draw the drafts from the draft head's own q and verify them by rejection sampling (flag spec_rs; read per round)."""
        return os.environ.get("KYOJIN_SPEC_RS", "0") != "0"

    @property
    def _spec_couple(self) -> int:
        """Flag spec_couple (percent of the target's Gumbel noise the draft chain adds; 0 = off); read per round."""
        try:
            return max(0, int(os.environ.get("KYOJIN_SPEC_COUPLE", "0") or 0))
        except ValueError:
            return 0

    def _couple_plan(self, members: list, jobs: list, lookups: dict) -> dict:
        """spec_couple: {member index: keys} for every sampled session that drafts this round. depth + 1 fresh keys are drawn BEFORE the draft: key i is
        the one the verify pass uses for row i (the sample after the i-th fed token) and the one the draft chain mixes into proposal i, so a proposal that
        agrees with the target's distribution picks the token the target's noise picks. The keys are independent of everything the round has decided
        before row i, so the verify's sample at row i keeps the plain-sampling distribution whatever the proposals are (a proposal only decides whether the
        next row is reached). Keys drawn and not used (the gate cut the rows, a rejection) are dropped: a key stream has no meaning past its draws."""
        out = {}
        for k, (s, job) in enumerate(zip(members, jobs)):
            if job[3] > 0 and k not in lookups and not s.sampling.greedy:
                out[k] = [self._next_key(s) for _ in range(job[3] + 1)]
        return out

    @property
    def _spec_gate(self) -> bool:
        """Confidence gate threshold from the measured P(accept | conf) (flag spec_gate; read per round)."""
        return os.environ.get("KYOJIN_SPEC_GATE", "0") != "0"

    @property
    def _pf_path(self) -> bool:
        """Flag pf_path (read per call): a prefix snapshot is used only by a prompt whose own chunk grid has the path that built it (see PrefixCache.lookup)."""
        return os.environ.get("KYOJIN_PF_PATH", "0") not in ("0", "")

    @property
    def _kv_admit(self) -> bool:
        """Flag kv_admit (read per call; needs kv_budget): admission by reservation (prompt + reply growth of every active session), cold-start hold, refusal of a request that
        cannot fit alone, and the youngest decoder parked when the estimate was wrong."""
        return self.kv_budget is not None and os.environ.get("KYOJIN_KV_ADMIT", "0") not in ("0", "")

    @property
    def _pf_warm0(self) -> bool:
        """Flag pf_warm0 (read per call): the first prefill chunk's Engram rows are read into the page cache when the request is admitted."""
        return os.environ.get("KYOJIN_PF_WARM0", "0") not in ("0", "")

    @property
    def _pf_share(self) -> bool:
        """Flag pf_share (read per call): shared-prefix sessions start from the prefilling session's cache at a point of their own grid."""
        return os.environ.get("KYOJIN_PF_SHARE", "0") not in ("0", "")

    @property
    def _mix_decode(self) -> int:
        """Flag mix_decode (read per tick): percent of the wall time given to decode rounds while a prefill runs and sessions decode (0 = today: one round per chunk)."""
        v = os.environ.get("KYOJIN_MIX_DECODE", "0")
        return 0 if v in ("0", "") else min(max(int(v), 0), 95)

    @property
    def _pf_fair(self) -> bool:
        """Flag pf_fair (read per chunk): the prefilling sessions take turns by chunk instead of strict FIFO, so a short request behind a long prompt
        reaches its first token in about one chunk of the long one instead of after all of it."""
        return os.environ.get("KYOJIN_PF_FAIR", "1") not in ("0", "")

    @property
    def _pf_floor(self) -> int:
        """Flag pf_floor (read per chunk / per layer bound): tok/s each decoding session should keep while another session prefills (0 = off)."""
        v = os.environ.get("KYOJIN_PF_FLOOR", "15")
        return 0 if v in ("0", "") else max(int(v), 0)

    def _floor_duty(self) -> float:
        """Flag pf_floor: share of the wall (0..1) the decode rounds need to give every decoder `pf_floor` tok/s, from the measured round time and tokens per
        session per round, capped by pf_floor_max percent (the cap wins: below the floor the prefill is not starved further)."""
        f = self._pf_floor
        if not f:
            return 0.0
        cap = min(max(int(os.environ.get("KYOJIN_PF_FLOOR_MAX", "60") or 0), 0), 95) / 100.0
        return min(f * self._rd_s / max(self._tpr, 0.25) * self._floor_k, cap)

    @property
    def _floor_catchup(self) -> float:
        """Flag pf_floor_catchup: most percent (0..95) of the wall a prefill chunk may give the decoders when one of them is behind the floor over its own window
        (0 = off: the duty alone decides, and it only learns from the last chunk)."""
        v = os.environ.get("KYOJIN_PF_FLOOR_CATCHUP", "30")
        return 0.0 if v in ("0", "") else min(max(int(v), 0), 95) / 100.0

    def _floor_deficit(self, now: float) -> float:
        """Flag pf_floor_catchup: seconds of decode rounds the most-behind decoder needs to be back on `pf_floor` tok/s over its window (first look with a prefill
        running until now). The window is per session, so the first decoder, which meets the cold duty estimate and the feedback ramp alone, is not averaged
        with later ones."""
        f = self._pf_floor
        worst = 0.0
        for s in self.members:
            if not s.fl_t0:
                s.fl_t0, s.fl_n0 = now, s.n_gen
                continue
            worst = max(worst, f * (now - s.fl_t0) - (s.n_gen - s.fl_n0))
        return max(worst, 0.0) * self._rd_s / max(self._tpr, 0.25)

    @property
    def _mix_eff(self) -> float:
        """Percent of the wall owed to decoders during prefill: the floor's duty when pf_floor is on, else flag mix_decode."""
        return self._floor_duty() * 100.0 if self._pf_floor else float(self._mix_decode)

    @property
    def _mix_slice_ms(self) -> int:
        """Flag mix_slice_ms (read per chunk): longest prefill stretch (ms) between two decode rounds INSIDE a chunk, at the layer bounds (0 = off)."""
        v = os.environ.get("KYOJIN_MIX_SLICE_MS", "0")
        return 0 if v in ("0", "") else max(int(v), 0)

    @property
    def _lazy_fit(self) -> bool:
        """Lazy draft by the measured trade (flag lazy_fit; read per round): the draft wait the verify build hides against the rows the confidence
        gate would cut out of the verify. Off: the backend's cut-rate threshold (`backends.LAZY_CUT_MAX`), which drops the overlap for good once
        the gate cuts often."""
        return os.environ.get("KYOJIN_LAZY_FIT", "0") not in ("0", "")

    @property
    def _lazy_trim(self) -> bool:
        """Width of the draft ask follows what the confidence gate keeps (flag lazy_trim; read per round): the verify of a lazy round is built at the
        planned depth before the gate can cut it, and a round the gate keeps one row of costs the rows it cut at ~4.9 ms each."""
        return os.environ.get("KYOJIN_LAZY_TRIM", "0") not in ("0", "")

    def _trim_depths(self, members: list, depths: list, lookups: dict) -> list:
        """lazy_trim: session s asks the draft head for at most ceil(kept_ema) + 1 proposals, kept_ema = what the gate kept in its recent drafted
        rounds (the +1 lets the ask grow back: a round that keeps the whole ask raises it). Every TRIM_PROBE-th drafted round asks the full plan so
        the deep positions' acceptance stays measured. Only the number of rows asked changes: the accept rule is the verify's, a draft the gate
        would have cut is never accepted."""
        out = list(depths)
        for i, s in enumerate(members):
            d = depths[i]
            if d <= 0 or i in lookups or s.kept_ema < 0:
                continue
            s.trim_n += 1
            if s.trim_n % TRIM_PROBE == 0:
                continue
            t = max(1, min(d, int(math.ceil(s.kept_ema - 1e-9)) + 1))
            if t < d:
                self.stats["spec_trim_rows"] += d - t
                self.stats["spec_trim_rounds"] += 1
                out[i] = t
        return out

    @property
    def _gpu_span(self) -> bool:
        """Per-round GPU busy / idle from the command buffers' own GPU start and end stamps (probe gpu_span; read per round)."""
        return GS.ON or GS.KERN

    @property
    def _spec_lookup(self) -> bool:
        """Prompt-lookup drafts from the session's own committed tokens (flag spec_lookup; read per round)."""
        return os.environ.get("KYOJIN_SPEC_LOOKUP", "0") not in ("0", "")

    @property
    def _copy_v2(self) -> bool:
        """Copy path on the request's own tokens, route copy > block > plain (flag copy_v2; read per round)."""
        return os.environ.get("KYOJIN_COPY_V2", "0") not in ("0", "")

    @property
    def _lk_long(self) -> bool:
        """Lookup widths from measured solo lookup rounds, widened one tier at a time on a strong match (flag spec_lookup_long; read per round)."""
        return os.environ.get("KYOJIN_SPEC_LOOKUP_LONG", "0") not in ("0", "")

    @property
    def _lk_vote(self) -> int:
        """Candidates a lookup draft is voted from (flag spec_lookup_vote; 0 = the single best occurrence; read per round)."""
        try:
            return max(0, int(os.environ.get("KYOJIN_SPEC_LOOKUP_VOTE", "0") or 0))
        except ValueError:
            return 0

    @property
    def _lk_confirm(self) -> bool:
        """A lookup drafts at the cap on a strong match only after two consecutive rounds landed every draft (flag lookup_confirm; read per round).
        Off: one full round is enough, and the next strong match jumps to the row cap even though one round cannot show that the copy continues."""
        return os.environ.get("KYOJIN_LOOKUP_CONFIRM", "0") not in ("0", "")

    @property
    def _lk_adapt(self) -> bool:
        """Agreements shorter than PLD.MIN_SUFFIX priced from their own measured acceptance (flag spec_lookup_adapt; read per round)."""
        return os.environ.get("KYOJIN_SPEC_LOOKUP_ADAPT", "0") not in ("0", "")

    @property
    def _spec_chain(self) -> int:
        """Draft blocks per round (flag spec_chain; 1 = today; read per round)."""
        try:
            return max(1, int(os.environ.get("KYOJIN_SPEC_CHAIN", "1") or 1))
        except ValueError:
            return 1

    @property
    def _gate_adapt(self) -> bool:
        """Confidence-gate threshold from the round's own measured reward (flag gate_adapt; read per round)."""
        return os.environ.get("KYOJIN_GATE_ADAPT", "0") not in ("0", "")

    @property
    def _oracle(self) -> bool:
        """Perfect-draft probe of a verify round at `oracle_rows` rows (flag oracle_draft; read per round)."""
        return os.environ.get("KYOJIN_ORACLE_DRAFT", "0") not in ("0", "")

    @property
    def _spec_skip(self) -> bool:
        """A round whose drafts were all cut by the confidence gate takes the plain step, and the draft head rests for a growing number of rounds
        (flag spec_skip; read per round)."""
        return os.environ.get("KYOJIN_SPEC_SKIP", "0") not in ("0", "")

    @property
    def _spec_ladder(self) -> bool:
        """Verify width from the measured convex cost ladder (kyojin_mlx.ladder; flag spec_ladder, read per round)."""
        return os.environ.get(LADDER.ENV, LADDER.ENABLED_DEFAULT) not in ("0", "")

    @property
    def _rate_pair(self) -> bool:
        """Depth rule from the PAIRED measured rate (flag rate_pair, read per round): tokens and ms summed over the same last
        drafted / plain rounds, instead of a token mean over the rounds the drafter happened to fill times a separate ms median.
        Both sides are the engine's own measurements, so an engine that gets faster moves both and the ratio decides; the estimate
        cannot be lifted by counting only the rounds that proposed a full block."""
        return os.environ.get("KYOJIN_RATE_PAIR", "0") not in ("0", "")

    def __init__(self, backend, *, sessions: int = 4, ctx: int = 32768, chunk: int = 256, block: int = 512,
                 prefix_bytes: int = 1 << 30, decode_per_chunk: int = 1, lookahead: bool = True, log=None,
                 mtp_depth: int = 0, kv_budget: int | None = None, replay_chunk: int = 16, rq_floor: int | None = None,
                 row_cost: float = 0.4, draft_cost: float = 0.15, probe_every: int = 16, cap_probe: int = 8, kv_refresh: int = 16,
                 depth_override: int | None = None, disk=None, spec_costs: dict | None = None):
        """mtp_depth: most drafts per session and round (0 = no speculative decode; needs backend.supports_spec). kv_budget: bytes (None = no
        limit, see the module doc). replay_chunk: tokens replayed per tick when a dropped cache is rebuilt. rq_floor: smallest reply share before
        a session yields its slot (default RQ_MIN_TOKENS). row_cost / draft_cost: cost of one more verify row / of a draft call in solo steps
        (the depth policy); probe_every: rounds at D = 0 between two probes with D = 1; cap_probe: rounds below the cap between two rounds at
        the cap (measured policy); kv_refresh: ticks between two measurements of the size
        of the active caches (parked caches are measured when they park); depth_override: fixed D for every greedy session (tests, benches);
        disk: a kvdisk.KvDisk (checkpoints on the SSD under the prefix snapshots, see the module doc; its block must be a multiple of chunk).
        spec_costs: {context length: [ms of a solo spec round at depth 0, 1, ...]} measured on this device (tuning.spec_costs, tools/cost103.py):
        a solo session then picks its depth from these costs at its context length (row_cost / draft_cost stay for the shared pass)."""
        assert 1 <= sessions <= backend.max_rows, f"sessions must be in 1..{backend.max_rows}"
        assert chunk >= 1 and block % chunk == 0, "block must be a multiple of chunk"
        assert disk is None or disk.block % chunk == 0, "the disk block must be a multiple of chunk"
        if mtp_depth > 0:
            assert backend.supports_spec, f"backend {backend.kind} has no speculative decode (no draft head loaded)"
            assert sessions <= backend.max_spec_rows, f"sessions must be in 1..{backend.max_spec_rows} with mtp (one row each at least)"
        self.mtp_depth, self.spec = mtp_depth, mtp_depth > 0
        self.tap = None                              # kltap.Tap while an admin KL probe is armed (plain decode, one session)
        self.start_depth, self._lookahead_arg = mtp_depth, lookahead
        self.kv_budget, self.replay_chunk, self.rq_floor = kv_budget, max(1, replay_chunk), rq_floor
        self.row_cost, self.draft_cost, self.probe_every, self.cap_probe = row_cost, draft_cost, probe_every, cap_probe
        GS.install()                                 # probe: catalogue every Metal pipeline at startup (names are captured at creation; dispatch recording is off unless gpu_kern)
        self.bc_bytes = 0
        self.kv_peak = 0
        self.active_peak = 0
        self.kv_refresh = max(1, kv_refresh)
        self.depth_override = depth_override
        self.spec_costs = {int(c): list(v) for c, v in spec_costs.items() if v} if spec_costs else None
        self.bpt = 0.0                              # bytes of cache per token (measured), to size a session before it is admitted
        self.backend, self.sessions, self.ctx = backend, sessions, ctx
        if chunk > MAX_PREFILL_CHUNK:
            raise ValueError(f"prefill chunk {chunk} > {MAX_PREFILL_CHUNK}: 16384-row chunks panicked the Ultra (29/10 00:26)")
        self.big_chunk, self.big_from = int(os.environ.get("KYOJIN_PF_BIG_CHUNK", "0")), int(os.environ.get("KYOJIN_PF_BIG_FROM", "32768"))   # prompts past BIG_FROM tokens prefill in chunks of BIG_CHUNK (a multiple of `chunk` and of `block`; 0 = off): prefix snapshots every BIG_CHUNK there, every `block` before; a re-sent prompt re-prefills at most BIG_CHUNK - 1 tokens
        self.big_chunk = min(self.big_chunk, MAX_PREFILL_CHUNK)
        self.chunk, self.block, self.decode_per_chunk, self.lookahead = chunk, block, decode_per_chunk, lookahead and (not self.spec or hasattr(backend, "spec_ok"))
        self.plain_ms, self.spec_k, self._k_seen, self._spec_obs = None, 1.0, 0, None
        self._plain_ring: list = []                                  # last K_WINDOW warm solo plain rounds (ms): their median is the rate rule's plain cost
        self._k_warm: set = set()                       # measured: ms of a pipelined plain round (solo); drafted-round cost / the formula's (EMAs)
        # KYOJIN_MTP_POLICY=robust: the drafted-round sample leaves out the catch-up feed of the plain rounds' taps (a cost of the switch, not of
        # a steady drafted round), spec_k is the median of the last K_WINDOW samples (one slow round cannot switch drafting off), and the probe
        # gap / floor are bounded so that a wrong D = 0 recovers in a bounded number of rounds. Default off: the constant-cost policy above.
        self.robust = os.environ.get("KYOJIN_MTP_POLICY", "robust").strip().lower() != "constant"      # default robust (29/10: the constant policy stuck at D=0 after one slow probe)
        self._k_samples: list = []
        # One confidence calibration for every session (30/10 trace: a fresh per-session calibrator starts each reply on a 0.7 x 2-trial
        # prior, one early miss took it to 0.45 and the reply sat at D = 0 for 16-121 rounds; the drafter's calibration is the model's).
        self._dspol_cal = DSPOL.Calibrator()
        self._dsp2_cal = DSP2.Calibration()             # flag dspol_v2: P(accept k | kept so far) by bin of -log10(1 - conf), shared like the one above
        self._dsp2_cost = DSP2.RowCost.load()           # flag dspol_v2: the measured width table (scaled by the live plain step per decision)
        self._d_ms: dict = {}                           # solo drafted-round ms per depth (last K_WINDOW warm rounds): the measured cost behind _fit_cost
        self._rd_win: list = []                         # KYOJIN_RATE_PAIR: (tokens, ms) of the last PAIR_WINDOW solo drafted rounds, any proposal length
        self._rp_win: list = []                         # ... and (1, ms) of the last PAIR_WINDOW solo plain rounds: the paired rate's two sides
        self._ladder = LADDER.Ladder()                  # flag spec_ladder: cost ladder of a solo pass (one verify pass = one round, the row cap is shared)
        self._ladders: dict = {}                        # flag spec_ladder: the same ladder per number of sessions n > 1 (a round of n sessions costs another ms per width)
        self._plain_n: dict = {}                        # flag spec_ladder: measured ms of a plain pass of n sessions (EMA); plain_ms is the n = 1 one
        self._lk_costs = PLD.LookupCosts()              # flag spec_lookup_long: measured ms of solo lookup rounds per draft count
        self._ladder_cold = 0                           # flag spec_ladder: rounds at the cap before the first plain measurement exists
        self.plain_ahead = lookahead                                  # plain rounds of the speculative path launch the next step before reading the one in flight
        if kv_budget is not None:
            prefix_bytes = min(prefix_bytes, kv_budget // 4)                 # snapshots are part of the budget; they never take more than a quarter
        self.log = log or (lambda *a: None)
        self.prefix = PrefixCache(backend, block, prefix_bytes, disk=disk, log=self.log)
        self._lk_kept = collections.deque(maxlen=LK_KEEP)
        self.cv = threading.Condition()
        self._last_work = time.monotonic()
        self._inbox: collections.deque = collections.deque()
        self._stop = False
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        # engine-thread state
        self.waiting: collections.deque = collections.deque()
        self.prefilling: list = []
        self.ready: list = []                       # prefill done (or resumed), waiting to join the batch
        self.members: list = []                     # decoding, in batch order
        self.bc = None
        self.cur = None                             # lazy [N] tokens of the last launched step, not yet read
        self.nxt = None                             # look-ahead step launched from cur
        self._tuner = None                          # several sessions share a verify pass: the depth (or the plain step) is picked from measured rates (mu_tuner.py)
        self._round_D = None
        self._in_spec = False                       # the mode of the current batched cache (speculative rounds or plain steps)
        self._since_chunk = 0
        self.stats = collections.Counter()
        self.bad_ids = frozenset()                                   # server: the begin-of-text id when it is not also a stop id
        for _k in ("spec_lookup_rounds", "spec_lookup_drafted", "spec_lookup_accepted", "spec_skip_rounds", "spec_skip_held", "spec_lookup_short_drafted", "spec_lookup_short_accepted",
                   "spec_lookup_match", "spec_lookup_nomatch", "spec_lookup_short", "spec_lookup_refused", "spec_gate_cut", "spec_block_cut", "spec_lazy_rounds", "spec_eager_rounds",
                   "spec_oracle_rounds", "spec_oracle_drafted", "spec_oracle_accepted", "spec_oracle_miss", "spec_oracle_resync",
                   "spec_gate_adapt_rounds", "spec_gate_adapt_up",
                   "mix_slice_rounds", "t_pf_inline_decode", "floor_bursts", "floor_burst_rounds", "share_hits", "share_tokens", "share_seeds", "share_takers", "share_refused_budget", "share_tied", "share_held",
                   "kv_drops", "rebuilds", "kv_parks", "kv_refused", "kv_over_budget", "tok_gap_max_s", "tok_gap_pf_max_s", "mu_recast"):
            self.stats[_k] = 0                                       # always in /v1/stats (path check: a lever that never fires shows as a zero)
        self.max_step_gap = 0.0
        self._mix_debt = 0.0                        # flag mix_decode: seconds of decode the sessions are owed before the next prefill chunk
        self._last_dec_end = time.perf_counter()    # start of the last decode round (flag mix_slice_ms: the slice is counted from here)
        self._in_yield = False
        self._rd_s, self._tpr = 0.05, 1.0           # flag pf_floor: measured seconds per decode round, tokens per session per round (moving averages)
        self._floor_k = 1.0                         # flag pf_floor: feedback factor on the duty (target rate / rate the decoders got in the last chunk)
        self._chunk_t0 = 0.0                        # flag pf_floor: start of the running prefill chunk
        self._inline_s = 0.0                        # flag mix_slice_ms: decode time spent inside the current prefill chunk
        self._dwait_ms = 0.0                        # flag lazy_fit: the draft section's off-CPU time per round (ms), the part the verify build can hide
        self._row_ms = 0.0                          # flag lazy_fit: the GPU wait of a round per verified row (ms)
        self._cut_rows = 0.0                        # flag lazy_fit: rows the confidence gate cuts after the verify, per round

    # ------------------------------------------------------------------------------------------------ public
    def set_depth(self, depth: int) -> None:
        """Admin A/B on a loaded server: 0 = plain decode, up to the depth the server started with. Call when idle (the epoch rebuilds itself)."""
        if not 0 <= depth <= self.start_depth:
            raise ValueError(f"mtp_depth must be in 0..{self.start_depth} (the depth the server started with)")
        self.mtp_depth, self.spec = depth, depth > 0
        self.lookahead = self._lookahead_arg and (not self.spec or hasattr(self.backend, "spec_ok"))

    def start(self) -> None:
        """Run the loop on a new thread. MLX streams are per thread: a lazy array made on one thread cannot be evaluated on another, so the
        thread must build whatever it computes with (the Fake backend of the tests does). For a real model call run() from the thread
        that loaded it."""
        self._thread = threading.Thread(target=self.run, name="kyojin-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        if self._thread:
            self._thread.join(timeout=30)

    def wake(self) -> None:
        with self.cv:
            self.cv.notify_all()

    def submit(self, ids: list, *, max_new: int | None = None, sampling: Sampling | None = None, stop_ids=(), cuts=(), vl=None, real_ids=None,
               stages: dict | None = None) -> Handle:
        """Queue a request. max_new None = as long as the context allows (shared fairly, see rq_budget). `stages` (flag ttft_stages): the client-side
        stages already measured for this request (render_ms, tokenize_ms), filled in further by the engine and returned with the done stats."""
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) >= self.ctx:
            raise ValueError(f"context length exceeded: prompt is {len(ids)} tokens, the server context is {self.ctx}")
        room = self.ctx - len(ids) - 1
        limit = room if max_new is None else min(max_new, room)
        sampling = sampling or Sampling()
        h = Handle(self)
        s = Session(next(self._ids), list(ids), limit, sampling, frozenset(stop_ids), h, cuts=tuple(sorted({int(c) for c in cuts if 0 < int(c) < len(ids)})))
        if vl:
            s.vl, s.rids = vl, list(real_ids)
        s.stages = dict(stages or {}) if (stages is not None or ttft_on()) else None
        s.yield_after = rq_budget(self.ctx, self.sessions, len(ids), limit, floor=self.rq_floor)
        if self._kv_admit and self.bpt > 0 and self._need_bytes(s) > self.kv_budget:
            raise ValueError(f"request needs about {self._need_bytes(s) / GIB:.2f} GiB of KV ({len(ids)} prompt tokens + up to {limit} reply tokens at "
                             f"{self.bpt:.0f} bytes per token), the KV budget is {self.kv_budget / GIB:.2f} GiB: shorten the prompt or max_tokens")
        s.seed = sampling.seed if sampling.seed is not None else int.from_bytes(np.random.bytes(7), "little")
        with self.cv:
            self._inbox.append(s)
            self.cv.notify_all()
        return h

    def status(self) -> dict:
        p = self.prefix
        return {"sessions": self.sessions, "waiting": len(self.waiting) + len(self._inbox), "prefilling": len(self.prefilling),
                "decoding": len(self.members) + len(self.ready), "prefix_entries": len(p.entries), "prefix_bytes": p.bytes,
                "prefix_hits": p.hits, "prefix_misses": p.misses, "peak_gb": round(mx.get_peak_memory() / 2 ** 30, 3),
                "active_gb": round(mx.get_active_memory() / 2 ** 30, 3), "cache_gb": round(mx.get_cache_memory() / 2 ** 30, 3), "mtp_depth": self.mtp_depth,
                "kv_budget_gb": None if self.kv_budget is None else round(self.kv_budget / 2 ** 30, 3),
                "kv_gb": round(self.kv_bytes() / 2 ** 30, 4), "kv_peak_gb": round(self.kv_peak / 2 ** 30, 4),
                "kv_bytes": self.kv_bytes(), "kv_peak_bytes": self.kv_peak, "bound_submits": _bound_submits(), **self._kv_roles(), **self.stats, **({} if self._tuner is None else {"tuner_ms": {n: {a: round(r * 1e3, 1) for a, r in d.items()} for n, d in self._tuner.cost.items()}}),
                **({} if not self._spec_ladder else {"spec_ladder": {**self._ladder.snapshot(), **({"by_n": {str(n): l.snapshot() for n, l in self._ladders.items()}} if self._ladders else {})}}),
                **({} if p.disk is None else {"disk_ns": p.disk.ns, "disk_files": len(p.disk.index), "disk_bytes": p.disk.bytes(), "disk_hits": p.disk_hits,
                                              "disk_hit_tokens": p.disk_hit_tokens, "disk_saved": p.disk_saved, "disk_errors": p.disk_errors})}

    # ------------------------------------------------------------------------------------------------ loop
    def _kv_roles(self) -> dict:
        """Billed bytes by role of the largest live cache (kvledger.summary: metadata walk, no eval), to read the gap above the packed size on a real pack."""
        live = [x for lst in (self.prefilling, self.ready, self.members, self.waiting) for x in list(lst) if x.cache is not None]
        if not live:
            return {}
        from .. import kvledger
        big = max(live, key=lambda x: x.kv_bytes)
        try:
            return {"kv_roles": kvledger.summary(big.cache, max(big.pos, 1))}
        except (RuntimeError, AttributeError, TypeError, KeyError, ValueError):       # the engine thread changes the cache under us: skip this reading
            return {}

    def _has_work(self) -> bool:
        return bool(self._inbox or self.waiting or self.prefilling or self.ready or self.members)

    def run(self) -> None:
        """The engine loop; returns after stop()."""
        while True:
            if msg := applegpu.sync_qos():                  # MLX encodes on this thread; QoS is per thread, so a flags.set() lands here
                self.log(msg)
            if not self._has_work():
                HL.idle()                                   # serve_gc: collect the cycles of the last request now, outside the lock
            with self.cv:
                while not self._stop and not self._has_work():
                    self.cv.wait(timeout=GPU_KEEPALIVE_S or 1.0)
                    if GPU_KEEPALIVE_S and not self._has_work() and time.monotonic() - self._last_work < GPU_KEEPALIVE_FOR_S:
                        FD.hit("gpu_keepalive")                 # idle keepalive commands issued
                        self.backend.gpu_keepalive()    # after ~1.5 s idle the next GPU command stalls ~1.2 s on a 190 GiB model: every Hermes call after a tool paid it
                if self._stop:
                    break
                while self._inbox:
                    self.waiting.append(self._inbox.popleft())
            self._last_work = time.monotonic()
            HL.gc_sync(); HL.switch_sync(); HL.bind(self.stats)      # serve_gc / serve_switch / serve_prof (module flags, read per iteration)
            PS.sync(self.stats)                                     # host_sample: the engine thread's own stack, wall sampled (probe)
            t0 = time.perf_counter()
            HL.mark("loop")
            try:
                self._tick()
            except Exception:                               # a bug must not leave clients hanging
                self.log("engine error:\n" + traceback.format_exc())
                self._fail_all("error")
            self.stats["t_tick"] += time.perf_counter() - t0
        self._fail_all("shutdown")

    def _fail_all(self, reason: str) -> None:
        for s in [*self.waiting, *self.prefilling, *self.ready, *self.members]:
            self._finish(s, reason)
        self.waiting.clear(); self.prefilling.clear(); self.ready.clear(); self.members.clear()
        self.bc = self.cur = self.nxt = None
        with self.cv:
            while self._inbox:
                self._finish(self._inbox.popleft(), reason)

    def _tick(self) -> None:
        self._reap_cancelled()
        self._enforce_kv()
        self._admit()
        HL.mark("tick_pre")
        mix = self._mix_eff
        if self.prefilling and (not self.members or self._since_chunk >= self.decode_per_chunk) and (mix == 0 or self._mix_debt <= 0 or not (self.members or self.ready)):
            self._prefill_step()
            self._since_chunk = 0
        if not self.prefilling:
            for x in self.members:                                  # flag pf_floor_catchup: no prefill, no floor window
                x.fl_t0 = 0.0
        if self.members or self.ready:
            self._decode_phase()
        HL.mark("tick_body")
        self._enforce_kv()
        self._note_kv()
        HL.mark("tick_post")

    def _decode_phase(self) -> None:
        """One decode round of the sessions that are decoding or ready (the second half of a tick; also called from inside a prefill chunk, flag mix_slice_ms)."""
        if not (self.members or self.ready):
            return
        t0 = time.perf_counter()
        PLS.tick_begin()                                            # probe plain_split: the round's own start (plan, then the plain tick)
        tun = self._multi_tuner()                                   # several sessions: the tuner keeps the plan (its arm is the ladder's cap), the ladder prices a solo / unshared pass
        if tun is not None:
            n = len(self.members) + len(self.ready)
            tkey = tuple(id(s) for s in self.members + list(self.ready))
            dmax = min(self.mtp_depth, max(0, getattr(self.backend, "max_spec_rows", 0) // n - 1))
            arm = tun.plan(n, dmax, tkey, self._pooled_acceptance(dmax)) if self._all_greedy() else 0
            self._round_D = arm if arm > 0 else None
            spec_now = arm > 0
        else:
            self._round_D = None
            spec_now = self.spec and self._spec_now()
        if spec_now != self._in_spec:                           # the mode changes with the number of sessions: end the epoch, rebuild in the other mode
            self._end_epoch()
            self._in_spec = spec_now
        fl_m, fl_n = (list(self.members), sum(x.n_gen for x in self.members)) if self._pf_floor else (None, 0)
        r0 = self.stats.get("spec_rows", 0)                  # probe gpu_span: the round's own commit range -> GPU busy / idle (gpuspan)
        if self._gpu_span:
            GS.begin()
        if spec_now:
            self._spec_tick()
        else:
            self._decode_tick()
        if self._gpu_span:
            GS.end(self.stats, (self.stats.get("spec_rows", 0) - r0) or 1)
        if tun is not None and self._all_greedy():
            tun.done(tuple(id(s) for s in self.members + list(self.ready)), sum(s.n_gen for s in self.members))
        self._since_chunk += 1
        self.stats["decode_ticks"] += 1
        t_end = time.perf_counter()
        self.max_step_gap = max(self.max_step_gap, t_end - t0)
        if fl_m and fl_m == self.members:                           # flag pf_floor: same members before / after, so n_gen counts this round's tokens
            tk = sum(x.n_gen for x in self.members) - fl_n
            if tk > 0:
                self._rd_s += 0.2 * ((t_end - t0) - self._rd_s)
                self._tpr += 0.2 * (tk / len(fl_m) - self._tpr)
        self._last_dec_end = t0                                     # the round's START: the slice counts from there (see _pf_yield)
        if self._mix_debt > 0:
            self._mix_debt -= t_end - t0
        PLS.tick_end()                                              # probe plain_split: the round's own wall

    def _multi_tuner(self):
        """The tuner of the round plan while several greedy sessions decode on a backend that verifies them in one pass (`multi_auto`, or flag mu_tuner
        for a backend that does not declare it), else None. Sampled sessions keep the solo policy: the tuner measures greedy rounds only. N = 1 is always
        the solo path."""
        if not (self.spec and mu_auto_on(self.backend, self._all_greedy()) and len(self.members) + len(self.ready) > 1):
            return None
        if self._tuner is None:
            from .mu_tuner import MultiTuner
            self._tuner = getattr(self.backend, "_mu_tuner", None) or MultiTuner()      # the rates per N outlive the engine (and its epochs)
            self.backend._mu_tuner = self._tuner
        return self._tuner

    def _pooled_acceptance(self, dmax: int) -> list:
        """Conditional acceptance of each draft position over the sessions at hand (prior 0.7 worth two trials; a position never drafted inherits flag mu_unseen percent of the one before)."""
        out, prev = [], 0.78
        unseen = int(os.environ.get("KYOJIN_MU_UNSEEN", "100")) / 100            # flag mu_unseen, read per call
        for i in range(max(dmax, 1)):
            tr = su = 0.0
            for s in self.members:
                if i < len(s.pos_trial):
                    tr += s.pos_trial[i]; su += s.pos_succ[i]
            p = (su + 0.7 * 2) / (tr + 2) if tr > 0 else prev * unseen
            out.append(p); prev = p
        return out

    def _all_greedy(self) -> bool:
        return all(s.sampling.greedy for s in self.members) and all(s.sampling.greedy for s in self.ready)

    def _spec_now(self) -> bool:
        """Speculative rounds for the sessions at hand? A backend whose verify pass is not shared between sessions (`spec_ok`) drafts only while it
        pays: with several sessions the batched plain step reads the weights once for all of them."""
        ok = getattr(self.backend, "spec_ok", None)
        return True if ok is None else bool(ok(len(self.members) + len(self.ready)))

    def _end_epoch(self) -> None:
        """Dissolve the batched cache, whatever mode built it (a step in flight is read first)."""
        if self.bc is None:
            return
        leaving: dict = {}
        if self.cur is not None:
            self._consume(self.cur, leaving, False)
            if self.nxt is not None:
                self.cur, self.nxt = self.nxt, None
                self._consume(self.cur, leaving, False)
        self._dissolve(leaving)

    # ------------------------------------------------------------------------------------------------ memory budget
    def kv_bytes(self) -> int:
        """Bytes held by session caches (active and parked), the batched cache and the prefix snapshots, as last measured."""
        n = self.prefix.bytes + self.bc_bytes
        seeds = {}
        for lst in (self.prefilling, self.ready, self.members, self.waiting):
            for s in lst:
                if s.cache is not None:
                    n += s.kv_bytes
                if s.share_seed is not None:
                    seeds[id(s.share_seed)] = s.share_seed.nbytes                 # one Seed serves all its takers: counted once
        n += sum(seeds.values())
        if os.environ.get("KYOJIN_KV_SEG_ONCE", "1") not in ("0", ""):          # flag kv_seg_once: a sealed segment the prefix ledger already bills is not billed again by its live layer
            from .. import fp4pool as FP
            billed = {i: r[1] for i, r in list(self.prefix.refs.items())}
            if billed:
                live = [s.cache for lst in (self.prefilling, self.ready, self.members, self.waiting) for s in list(lst) if s.cache is not None]
                n -= FP.retained_walk(live + ([self.bc] if self.bc is not None else []), set(), billed)["seg_overlap"]
        return n

    def _measure(self, s: Session) -> None:
        s.kv_bytes = self.backend.nbytes(s.cache) + (self.backend.draft_nbytes(s.draft) if s.draft is not None else 0)
        tokens = s.pos if (not s.prefilled or s.resume) else s.n_prompt + max(s.n_gen - 1, 0)       # positions in the cache
        if tokens > 0 and s.kv_bytes > 0:
            self.bpt = s.kv_bytes / tokens if self.bpt == 0 else 0.7 * self.bpt + 0.3 * s.kv_bytes / tokens

    def _note_kv(self) -> None:
        self.kv_peak = max(self.kv_peak, self.kv_bytes())
        if self.kv_budget is not None:                      # live memory at the end of a tick: weights + caches + what is held + the buffers of GPU work still in flight (an upper bound)
            self.active_peak = max(self.active_peak, mx.get_active_memory())

    def _enforce_kv(self) -> None:
        """Over the budget: drop the KV of the oldest parked session (it is rebuilt when it resumes), then prefix snapshots. Active sessions stay."""
        if self.kv_budget is None:
            return
        while self.kv_bytes() > self.kv_budget:
            parked = [s for s in self.waiting if s.cache is not None]
            if parked:
                self._drop(min(parked, key=lambda s: s.t_parked))
            elif not self.prefix.shrink(max(0, self.prefix.bytes - (self.kv_bytes() - self.kv_budget))):
                if self._kv_admit:
                    if any(x.park_now for x in (*self.members, *self.ready)):
                        return                                          # one is already leaving: wait for it before marking another
                    live = list(self.members + self.ready)
                    if len(live) > 1 and self.members:
                        young = max((x for x in live if x in self.members), key=lambda x: x.t_start, default=None)
                        if young is not None:
                            young.park_now = True                       # it leaves at its next round, is parked, and the oldest parked cache is dropped (rebuilt later)
                            self.stats["kv_parks"] += 1
                            return
                if not self.stats["kv_over_budget"]:
                    self.log(f"kv over budget with only active sessions: {self.kv_bytes()} > {self.kv_budget} bytes")
                self.stats["kv_over_budget"] += 1
                return

    def _need_bytes(self, s: Session) -> float:
        """Flag kv_admit: the cache s may grow to before it yields its slot or ends: bytes per token (measured) x (prompt + reply tokens it may produce in one stretch)."""
        reply = s.max_new if s.yield_after is None else min(s.max_new, s.yield_after)
        return self.bpt * (s.n_prompt + max(reply, 0))

    def _reserved(self, skip: Session | None = None) -> float:
        """Flag kv_admit: bytes that are spoken for: the prefix snapshots, the hand-off seeds and, for every active session, the larger of what it holds and what it may grow to.
        Parked caches are not in it: the budget drops them first."""
        n = self.prefix.bytes
        seeds = {}
        for lst in (self.prefilling, self.ready, self.members, self.waiting):
            for x in lst:
                if x.share_seed is not None:
                    seeds[id(x.share_seed)] = x.share_seed.nbytes
        n += sum(seeds.values())
        for x in (*self.prefilling, *self.ready, *self.members):
            if x is not skip:
                n += max(self._need_bytes(x), x.kv_bytes)
        return n

    def _drop(self, s: Session) -> None:
        s.cache, s.draft, s.kv_bytes = None, None, 0
        self.stats["kv_drops"] += 1

    def _kv_room(self, s: Session) -> bool:
        """Session s may start: nothing else runs, or its cache (a new prompt, or the rebuilt cache of a dropped session) fits in the budget."""
        if self._kv_admit:
            if self._active() == 0:
                return True                                                   # nothing else runs: it gets the machine (a request that cannot fit alone is refused in _admit)
            if self.bpt == 0:
                return False                                                  # cold start: the first measurement of bytes per token comes before a second session
            held = s.kv_bytes if (s.prefilled and s.cache is not None) else 0
            return self._reserved(s) + max(self._need_bytes(s) - held, 0) <= self.kv_budget
        if self.kv_budget is None or self._active() == 0 or (s.prefilled and s.cache is not None):
            return True
        tokens = s.n_prompt + (len(s.out) if s.prefilled else 0)
        return self.kv_bytes() + self.bpt * tokens <= self.kv_budget

    # ------------------------------------------------------------------------------------------------ admission
    def _active(self) -> int:
        return len(self.prefilling) + len(self.ready) + len(self.members)

    def _admit(self) -> None:
        if self._kv_admit and self.bpt > 0:
            for x in [x for x in self.waiting if not x.prefilled and self._need_bytes(x) > self.kv_budget]:      # cannot fit even alone (the cold start of submit)
                self.waiting.remove(x)
                self.stats["kv_refused"] += 1
                self.log(f"kv: request {x.sid} needs {self._need_bytes(x):.0f} bytes, budget {self.kv_budget}: refused")
                self._finish(x, "refused")
        while self.waiting and self._active() < self.sessions and self._kv_room(self.waiting[0]):
            s = self.waiting.popleft()
            if s.handle.cancelled.is_set():
                self._finish(s, "cancelled")
                continue
            if s.prefilled and s.cache is not None:          # a session that yielded its slot: its cache is kept
                s.seg_gen = 0
                self.ready.append(s)
            elif s.prefilled:                                # ... and was dropped since: rebuild (prompt, then replay)
                s.resume, s.replay_pos, s.seg_gen = True, 0, 0
                self.prefilling.append(s)
                self.stats["rebuilds"] += 1
            else:
                s.t_start = time.perf_counter()
                self.prefilling.append(s)
                if self._pf_warm0 and not s.resume:
                    wf = getattr(self.backend, "warm_first_chunk", None)
                    if wf is not None and wf(s.rids, 0, self._chunk_end(s, 0, len(s.ids))):
                        self.stats["pf_warm0"] += 1

    def _reap_cancelled(self) -> None:
        for lst in (self.waiting, self.prefilling, self.ready):
            for s in [s for s in lst if s.handle.cancelled.is_set()]:
                lst.remove(s)
                self._finish(s, "cancelled")

    def _finish(self, s: Session, reason: str) -> None:
        s.t_done = time.perf_counter()
        s.cache = s.draft = None
        t_first = s.t_first or s.t_done
        stats = {"prompt_tokens": s.n_prompt, "completion_tokens": s.n_gen, "cached_tokens": s.cached,
                 "queue_s": (s.t_start or s.t_done) - s.t_submit, "ttft_s": t_first - s.t_submit, "total_s": s.t_done - s.t_submit}
        if s.stages is not None:                                     # flag ttft_stages: every stage of the time to first token
            s.stages["queue_ms"] = round(((s.t_start or s.t_done) - s.t_submit) * 1e3, 3)
            s.stages["ttft_ms"] = round((t_first - s.t_submit) * 1e3, 3)
            stats["stages"] = s.stages
            for k, v in s.stages.items():                            # aggregates a served run can read without the client bodies
                if isinstance(v, (int, float)):
                    key = "t_ttft_" + k.removesuffix("_ms")
                    self.stats[key] += v                             # sum over the finished requests
                    self.stats[key + "_max"] = max(self.stats[key + "_max"], v)
        s.handle.events.put(("done", reason, stats))
        self.stats["finished_" + reason] += 1

    # ------------------------------------------------------------------------------------------------ tokens
    def _next_key(self, s: Session):
        if s.sampling.greedy:
            return None
        if s.key is None:                                   # created on the engine thread (every MLX call is made there)
            s.key = mx.random.key(s.seed)
        k = mx.random.split(s.key)
        s.key, sub = k[0], k[1]
        return sub

    def _accept(self, s: Session, tok: int) -> str | None:
        """One sampled token of s. Returns the finish reason when s is done."""
        if s.handle.cancelled.is_set():
            return "cancelled"
        if tok in s.stop_ids:
            return "stop"
        if tok in self.bad_ids:                                      # never legit in decode (a NaN row samples token 0): end the turn, do not feed it back
            self.stats["bad_tok_stop"] += 1
            self.log(f"engine: session {s.sid} sampled token {tok} (begin-of-text) after {s.n_gen} tokens: turn ended")
            return "stop"
        now = time.perf_counter()
        if s.t_tok and s.seg_gen:                           # gap between two tokens of one reply (not across a park / resume)
            g = now - s.t_tok
            if g > self.stats["tok_gap_max_s"]:
                self.stats["tok_gap_max_s"] = g
            if self.prefilling and g > self.stats["tok_gap_pf_max_s"]:
                self.stats["tok_gap_pf_max_s"] = g          # a gap that a prefill was running through
        s.t_tok = now
        s.n_gen += 1
        s.seg_gen += 1
        s.last = tok
        s.out.append(tok)
        s.handle.events.put(("tok", tok))
        if s.n_gen >= s.max_new or s.n_prompt + s.n_gen >= self.ctx:
            return "length"
        return None

    # ------------------------------------------------------------------------------------------------ prompt lookup
    def _lookup_index(self, s: Session):
        """The session's token index (created on first use); KYOJIN_SPEC_LOOKUP_TOKENS bounds it."""
        cap = int(os.environ.get("KYOJIN_SPEC_LOOKUP_TOKENS", str(PLD.TOKENS_DEFAULT)) or PLD.TOKENS_DEFAULT)
        cap = max(PLD.NGRAM + 1, min(cap, 1 << 22))
        s.lookup_index = self._lookup_donor(s, cap) or PLD.Index(cap)
        return s.lookup_index

    def _lookup_donor(self, s: Session, cap: int):
        """Flag lk_keep: the kept index of an earlier prompt this prompt starts with (the same conversation one turn later), taken by this session: the insert of a whole
        long prompt is a sort, ~100 ms per 1M tokens on every turn."""
        if os.environ.get("KYOJIN_LK_KEEP", "1") in ("0", "") or not self._lk_kept:
            return None
        arr = np.asarray(s.rids[:s.n_prompt], dtype=np.int32)
        best = None
        for d in self._lk_kept:
            if d.cap == cap and d.n <= arr.size and (best is None or d.n > best.n) and np.array_equal(d.toks[:d.n], arr[:d.n]):
                best = d
        if best is None or best.n == arr.size:                      # the same prompt again adds nothing, so it stays kept
            return best.fork() if best is not None else None
        self._lk_kept.remove(best)
        return best

    def _lookup_sync(self, s: Session, upto: int):
        """Index the committed tokens up to `upto` (prompt, then output); incremental, one numpy insert per chunk."""
        if not self._spec_lookup:
            return None
        idx = s.lookup_index if s.lookup_index is not None else self._lookup_index(s)
        target = min(int(upto), idx.cap)
        if target <= idx.n:
            return idx
        n0, nprompt = idx.n, s.n_prompt
        if n0 < nprompt:                                            # the prompt (a prefix hit indexes it in one insert, never a rebuild)
            idx.extend(s.rids[n0: min(target, nprompt)])
            n0 = idx.n
            if n0 == nprompt >= LK_KEEP_MIN and os.environ.get("KYOJIN_LK_KEEP", "1") not in ("0", ""):
                self._lk_kept.append(idx.fork())                    # the prompt alone: the session's output is not the next turn's text token for token
        if n0 < target:                                             # later committed output tokens
            idx.extend(s.out[n0 - nprompt: target - nprompt])
        return idx

    def _lookup_cost_ms(self, ctx: int, k: int) -> float | None:
        """Measured cost of a lookup round (k drafts -> k + 1 verify rows, no draft call); None before one is measured."""
        if self._lk_long:
            got = self._lk_costs.estimate(k)                        # measured lookup rounds first (the fit below is from MTP rounds of <= 6 rows)
            if got is not None:
                return max(1e-3, got)
        fit = self._fit_cost()
        if fit is not None:
            return max(1e-3, fit.s_ms + fit.slope_ms * k)
        if self.spec_costs is not None:                             # round_ms(n) includes the draft call; its slope is the per-row cost
            r0, r1 = self.round_ms(ctx, 0), self.round_ms(ctx, 1)
            return max(1e-3, r0 + max(0.0, r1 - r0) * k)
        if self.plain_ms:
            return max(1e-3, self.plain_ms * self.spec_k * (1.0 + self.row_cost * k))
        return None

    def _mtp_cost_ms(self, ctx: int, width: int) -> float | None:
        """Measured cost of the planned MTP round at `width` drafts (the engine's spec_fit tables when present; the cost
        ladder of flag spec_ladder when it has priced that width)."""
        if self._spec_ladder:                                       # the same measured ladder the width was chosen from (of this many sessions)
            n = max(1, len(self.members))
            plain = self.plain_ms if n == 1 else self._plain_n.get(n)
            if width <= 0:
                if plain:
                    return max(1e-3, plain)
            else:
                c = self._ladder_for(n).cost(int(width))
                if c is not None:
                    return max(1e-3, c)
        if self.spec_costs is not None:
            return max(1e-3, self.round_ms(ctx, max(0, int(width))))
        fit = self._fit_cost()
        if fit is not None:
            return max(1e-3, fit.cost_ms(max(0, int(width))))
        if self.plain_ms:
            if width <= 0:
                return max(1e-3, self.plain_ms)
            return max(1e-3, self.plain_ms * self.spec_k * (1.0 + self.row_cost * width + self.draft_cost))
        return None

    def _lookup_depth(self, s: Session, mtp_width: int):
        """(depth, draft) when a lookup should replace this session's MTP drafts this round, else (0, None)."""
        room = min(s.max_new - s.n_gen, self.ctx - s.n_prompt - s.n_gen) - 1
        if room <= 0 or s.lookup_index is None:
            return 0, None
        if s.lookup_index.n != s.n_prompt + max(0, len(s.out) - 1):  # a full index (KYOJIN_SPEC_LOOKUP_TOKENS) stopped at an old tail: it would match tokens that are not the context
            return 0, None
        max_rows = int(getattr(self.backend, "max_spec_rows", 0) or (PLD.MAX_DRAFT + 1))
        if self._copy_v2:
            FD.hit("copy_v2")
            return self._copy_depth(s, mtp_width, room, max_rows)
        vote = self._lk_vote
        if vote:
            FD.hit("spec_lookup_vote")
        m = (s.lookup_index.match_vote(s.last, PLD.strong_cap(max_rows), candidates=max(2, vote)) if vote
             else s.lookup_index.match_best(s.last, PLD.strong_cap(max_rows)))
        if m is None:
            self.stats["spec_lookup_nomatch"] += 1
            return 0, None
        ctx = s.n_prompt + s.n_gen
        s.lk_suffix = m.suffix
        self.stats["spec_lookup_match"] += 1
        if m.suffix < PLD.MIN_SUFFIX:
            self.stats["spec_lookup_short"] += 1                     # a match below the fixed floor (only flag spec_lookup_adapt tries these)
        if self._lk_adapt and PLD.SuffixBuckets.bucket(m.suffix) >= 0:      # a short agreement: priced from its own bucket, never from the long-match EMA
            if s.lk_adapt is None:
                s.lk_adapt = PLD.SuffixBuckets()
            k = PLD.gate(m, room, s.lk_adapt.expect(m.suffix), s.mt_ema if mtp_width > 0 else 0.0, mtp_width, False,
                         lookup_ms=lambda kk: self._lookup_cost_ms(ctx, kk), mtp_ms=self._mtp_cost_ms(ctx, mtp_width), max_rows=max_rows,
                         min_suffix=PLD.NGRAM)
        else:
            if self._lk_long:
                FD.hit("spec_lookup_long")
            streak = s.lk_streak and (not self._lk_confirm or s.lk_full_n >= 2)   # flag lookup_confirm: two full rounds before the cap
            if self._lk_confirm:
                FD.hit("lookup_confirm")
            k = PLD.gate(m, room, s.lk_ema, s.mt_ema if mtp_width > 0 else 0.0, mtp_width, streak,
                         lookup_ms=lambda kk: self._lookup_cost_ms(ctx, kk),
                         mtp_ms=self._mtp_cost_ms(ctx, mtp_width), max_rows=max_rows,
                         tiers=[t for t in PLD.tier_caps(max_rows) if t <= self._lk_costs.unlocked(max_rows)] if self._lk_long else None)
        if k > 0:
            return k, m.draft[:k]
        self.stats["spec_lookup_refused"] += 1                       # a 4-gram match priced below the planned MTP round
        return 0, None

    def _copy_depth(self, s: Session, mtp_width: int, room: int, max_rows: int):
        """(depth, draft) of the copy route (flag copy_v2), else (0, None): the round goes on as block (mtp_width > 0) or plain (0)."""
        ctx = s.n_prompt + s.n_gen
        block = int(getattr(getattr(self.backend, "drafter", None), "block", 5) or 5)
        p = CV2.plan(s.lookup_index, s.last, room, max_rows, block, scale=getattr(s, "cp_scale", 1.0),
                     lookup_ms=lambda kk: self._lookup_cost_ms(ctx, kk), mtp_ms=self._mtp_cost_ms(ctx, mtp_width), mtp_width=mtp_width,
                     mtp_ema=s.mt_ema if mtp_width > 0 else 0.0)
        if p is None:
            self.stats["spec_copy_nomatch"] += 1
            return 0, None
        s.lk_suffix = p.agree
        self.stats["spec_lookup_match"] += 1
        return p.k, p.draft

    def _lookup_plan(self, members: list, mtp_depths: list, clamp: bool = False):
        """(depths, lookups) of one round: the planned MTP depths, where a session whose index match is priced better than its
        MTP round takes a lookup draft instead (flag spec_lookup). The row cap shortens the longest drafts first; `clamp` re-cuts
        the planned depths to the room left (after a step in flight was committed)."""
        depths = [max(0, min(d, min(s.max_new - s.n_gen, self.ctx - s.n_prompt - s.n_gen) - 1)) if clamp else d
                  for s, d in zip(members, mtp_depths)]
        lookups: dict = {}
        if self._spec_lookup:                                        # prompt-lookup drafts: no draft-head forward for these sessions
            for i, s in enumerate(members):
                self._lookup_sync(s, s.n_prompt + max(0, len(s.out) - 1))   # all committed tokens: prompt + output[:-1]
                k, dr = self._lookup_depth(s, depths[i])
                if k > 0:
                    lookups[i] = (k, dr)
                    depths[i] = k
        over = len(members) + sum(depths) - int(getattr(self.backend, "max_spec_rows", 0) or 0)
        while over > 0 and any(depths):                              # too many rows: shorten the longest drafts first
            depths[max(range(len(depths)), key=depths.__getitem__)] -= 1
            over -= 1
        for i in list(lookups):                                      # the row cap may have shortened a lookup draft (or dropped it)
            if depths[i] <= 0:
                del lookups[i]
            else:
                lookups[i] = (depths[i], lookups[i][1][:depths[i]])
        return depths, lookups

    def _oracle_align(self, s: Session, ids: list) -> int | None:
        """Index in `ids` of the token after the session's committed output (flag oracle_draft): len(out) while the reply still follows the recorded
        continuation; after a divergence the latest occurrence of the last 16 / 8 / 4 / 1 committed tokens, so a row-count bit shift does not end the
        probe. None when nothing matches."""
        out = s.out
        n = len(out)
        if n <= len(ids) and list(out) == ids[:n]:
            return n
        for m in (16, 8, 4, 1):
            if n < m:
                continue
            tail = list(out[-m:])
            for j in range(len(ids) - m, n - m - 1, -1):
                if ids[j:j + m] == tail:
                    self.stats["spec_oracle_resync"] += 1
                    return j + m
        return None

    def _oracle_plan(self, members: list):
        """(depths, lookups) of one oracle_draft round: every session drafts the next `oracle_rows` - 1 tokens of the recorded continuation of its own
        prompt, so the round is a served verify pass at that width with perfect drafts and no draft-head call. The verify, the accept loop and the
        emitted tokens are the served ones; the record's prompt tag must match the session's (a stale file drafts nothing)."""
        try:
            rows = max(2, int(os.environ.get("KYOJIN_ORACLE_ROWS", "6") or 6))
        except ValueError:
            rows = 6
        ph, ids = _oracle_read()
        self._orc_round = False
        if not ids:
            self.stats["spec_oracle_miss"] += 1
            return [0] * len(members), {}
        be = self.backend
        cap = int(getattr(be, "max_spec_rows", 0) or 0)
        if rows > cap:                                    # the probe's own width: the served GEMV path up to the row count asked for
            from .. import exl3_linear as EL
            be.max_spec_rows = rows
            EL.ROW_SPLIT_MAX = max(EL.ROW_SPLIT_MAX, rows)
        depths, lookups = [], {}
        for i, s in enumerate(members):
            if ph and ph != self._prompt_tag(s):
                self.stats["spec_oracle_miss"] += 1
                depths.append(0)
                continue
            off = self._oracle_align(s, ids)
            room = min(s.max_new - s.n_gen, self.ctx - s.n_prompt - s.n_gen) - 1
            k = 0 if off is None else max(0, min(rows - 1, room, len(ids) - off))
            depths.append(k)
            if k > 0:
                lookups[i] = (k, list(ids[off:off + k]))
        self._orc_round = bool(lookups)
        return depths, lookups

    # ------------------------------------------------------------------------------------------------ prefill
    def _begin_prefill(self, s: Session) -> None:
        t0 = time.perf_counter()
        hit, snap = self.prefix.lookup(s.ids, self._stale_ok(s), self._on_grid(s) if self._pf_path else None)
        t_lookup = time.perf_counter()
        seed, s.share_seed = s.share_seed, None
        if seed is not None and seed.end > hit and seed.end < len(s.ids):
            hit, snap = seed.end, self.backend.restore(seed.snap)
            self.prefix.last_draft = seed.dsnap
            self.prefix.last_sig = seed.path
            self.stats["share_hits"] += 1
            self.stats["share_tokens"] += seed.end
        elif seed is not None and seed.end == hit:
            self.stats["share_tied"] += 1                             # the parent's own snapshot is at the same length: the prefix entry serves it (same bytes)
        if self.prefix.refused:
            self.stats["pf_skip_snap_fallback"] += 1                  # a snapshot with stale decoder rings was passed over: the earlier whole one (or 0) is used
        self.stats["t_pf_lookup"] += t_lookup - t0
        s.cache = snap if hit else self.backend.make_cache()
        s.pos = hit
        s.path = () if not hit else (seed.path if (seed is not None and hit == seed.end) else self.prefix.last_sig)
        if hit and self._spec_lookup:
            self._lookup_sync(s, hit)                               # the cached prefix is committed: one vectorised insert, no rebuild
        if not s.resume:
            s.cached = hit
        if self.spec:
            snap_draft = self.prefix.last_draft if hit else None
            s.draft = self.backend.draft_restore(snap_draft) if snap_draft is not None else self.backend.new_draft()
        s.hasher = PrefixCache.hasher()
        s.dg_grid = s.dg_prev = (0, None)
        if hit:
            s.hasher.update(np.asarray(s.ids[:hit], dtype=np.int32).tobytes())
            self.stats["prefix_hit_tokens"] += hit
            self._note_grid(s, hit)
        if s.stages is not None:                                     # ttft_stages: lookup (memory, else disk load) and the restore that follows it
            t2 = time.perf_counter()
            s.stages["lookup_ms"] = round((t_lookup - t0) * 1e3, 3)
            s.stages["restore_ms"] = round((t2 - t_lookup) * 1e3, 3)
            s.stages["prefix_tokens"] = hit

    @property
    def _kv_tail(self) -> bool:
        """Flag kv_tail (read per call): a snapshot (memory and disk) at the last chunk end of the prompt that lies before its final cut, see `_tail_anchor`."""
        return os.environ.get("KYOJIN_KV_TAIL", "1") not in ("0", "")

    def _tail_anchor(self, s: Session, L: int) -> int:
        """Flag kv_tail: the chunk end a continuation of this prompt can start from. A continuation (the same conversation one turn later, an agent loop) shares the prompt
        up to its last cut (the end of the conversation before the generation prompt) and then differs: the cut itself is not a cut of the continuation, so
        its snapshot is off the continuation's grid and unusable, and without an anchor the hit stops at the last block (up to a disk block minus one tokens paid
        again). The anchor is the largest chunk end below the last cut (below L without cuts) that is an absolute multiple of `chunk`: the continuation's own
        chunk grid has it (the chunks are the absolute multiples, the cuts it shares are before it), so the chunks that follow are the ones its solo prefill runs and the
        result is bit exact. The step is one chunk, not smaller: a finer start would change the chunk partition, hence the bits. Fixed-size recurrent state
        (window rings, compressor / delta-net state) is saved whole at any position, so it does not bound the step. 0 = none."""
        if s.anchor is None:
            s.anchor = 0
            if self._kv_tail and not s.resume and L > self.chunk:
                last = s.cuts[-1] if s.cuts else L
                s.anchor = max((e for e in self._grid(s) if 0 < e < last and e % self.chunk == 0), default=0)
        return s.anchor

    def _digest_inc(self) -> int:
        """KYOJIN_KV_DIGEST_INC: 0 = the disk tier hashes ids[:b] (and the parent prefix) from scratch at every checkpoint, O(b) host work per chunk at long context;
        1 = the session's running hash gives both digests; 2 = as 1 and each digest is checked against the one-shot hash (a mismatch counts and uses the one-shot hash)."""
        return int(os.environ.get("KYOJIN_KV_DIGEST_INC", "0") or 0) if self.prefix.disk is not None else 0

    def _note_grid(self, s: Session, end: int) -> None:
        """Remember the digest of the prompt up to `end` when it is a disk-grid point (the parent digest of the next checkpoint)."""
        if self._digest_inc() and end % self.prefix.disk.block == 0:
            s.dg_prev, s.dg_grid = s.dg_grid, (end, s.hasher.digest())

    def _disk_digests(self, s: Session, ids: list, end: int) -> dict:
        """dg / pdg keyword arguments for `save_disk`: digests of ids[:end] and of the parent length from the running hash; {} when the mode is off or the parent point was not
        passed by this session (restored from a snapshot above it): the disk tier then hashes."""
        mode = self._digest_inc()
        if not mode:
            return {}
        from .. import kvdisk as KD
        blk = self.prefix.disk.block
        pb = end - blk if end % blk == 0 else end // blk * blk
        dg = s.hasher.digest()
        par = s.dg_prev if end % blk == 0 else s.dg_grid                  # on the grid the latest record is `end` itself, its parent the one before; off it the latest grid point is the parent
        pdg = par[1] if (pb > 0 and par[0] == pb) else None
        if pb > 0 and pdg is None:
            return {}
        if mode == 2:
            ok = dg == KD.digest(ids, end) and (pb <= 0 or pdg == KD.digest(ids, pb))
            self.stats["kv_digest_check" if ok else "kv_digest_mismatch"] += 1
            if not ok:
                return {}
        return {"dg": dg, **({"pdg": pdg} if pb > 0 else {})}

    def _chunk_end(self, s: Session, pos: int, L: int) -> int:
        """End of the prefill chunk that starts at `pos` of a prompt of L tokens."""
        end = min(L, (pos // self.chunk + 1) * self.chunk)         # chunks end on absolute multiples of `chunk`
        if self.big_chunk and pos >= self.big_from and pos % self.big_chunk == 0 and L - pos >= self.big_chunk:
            end = pos + self.big_chunk                              # deep in a long prompt: fewer, larger chunks (MoE / CED prefill is 1.5-2x faster at 8192 rows than at 2048)
        i = bisect.bisect_right(s.cuts, pos)
        if i < len(s.cuts):
            end = min(end, s.cuts[i])                                 # a chunk never runs past the next cut
        return end

    def _grid(self, s: Session) -> list:
        """Chunk ends of s's prompt on its own grid (absolute multiples of `chunk`, the big chunks, its cuts, the end), memoised."""
        if s.grid is None:
            ends, p, L = [], 0, len(s.ids)
            while p < L:
                p = self._chunk_end(s, p, L)
                ends.append(p)
            s.grid = ends
        return s.grid

    def _on_grid(self, s: Session):
        """grid(b, sig) for PrefixCache.lookup: b is a chunk end of s's own grid and the snapshot was built by exactly the grid's chunks up to b (sig None: not known,
        a disk checkpoint: the position alone)."""
        g = self._grid(s)

        def ok(b: int, sig) -> bool:
            i = bisect.bisect_left(g, b)
            if i >= len(g) or g[i] != b:
                return False
            return sig is None or tuple(sig) == tuple(g[:i + 1])
        return ok

    def _share_target(self, s: Session, o: Session) -> int:
        """The chunk end of leader s that session o (not started) can start from bit-identically: the last chunk end of s that is <= the common prefix of the two prompts,
        is also a chunk end of o's own grid, and leaves o at least one token to prefill. 0 = none."""
        t = s.share_t.get(o.sid)
        if t is None:
            if s.ids_np is None:
                s.ids_np = np.asarray(s.ids, dtype=np.int32)
            if o.ids_np is None:
                o.ids_np = np.asarray(o.ids, dtype=np.int32)
            m = min(len(s.ids), len(o.ids))
            diff = np.flatnonzero(s.ids_np[:m] != o.ids_np[:m])
            lcp = int(diff[0]) if diff.size else m
            go = self._grid(o)
            gs = [*s.path, *(e for e in self._grid(s) if e > s.pos)]       # the chunk ends s has run (its real path, whatever snapshot it started from) and will run
            k = 0
            while k < len(gs) and k < len(go) and gs[k] == go[k]:          # same chunk path up to here: the cache of s is the cache o's own prefill would have
                k += 1
            t = max((e for e in gs[:k] if e <= lcp and e < len(o.ids)), default=0)
            s.share_t[o.sid] = t
        return t

    def _share_takers(self, s: Session, end: int) -> list:
        """Sessions that have not started and take the cache of s at `end` (flag pf_share)."""
        if s.resume or not self._pf_share or s.path is None:
            return []
        return [o for o in (*self.prefilling[1:], *self.waiting)
                if o is not s and not o.prefilled and not o.resume and o.cache is None and o.pos == 0
                and (o.share_seed is None or o.share_seed.end < end) and self._share_target(s, o) == end]

    def _stale_ok(self, s: Session):
        """Predicate for `PrefixCache.lookup` (ced_skip_dec=3): a snapshot taken at a skipped boundary `b` has stale decoder window rings and a stale drafter ring; it is exact
        for this session iff the chunk that follows it (this session's grid and cuts) is a CED chunk of more than `tail` rows, which rewrites both whole."""
        rw = getattr(self.backend, "ced_rewrites", None)
        L = len(s.ids)
        return None if rw is None else (lambda b: b < L and bool(rw(self._chunk_end(s, b, L) - b)))

    def _skip_dec(self, s: Session, end: int, L: int) -> bool:
        return bool(self._skip_mode(s, end, L))

    def _skip_mode(self, s: Session, end: int, L: int) -> int:
        """ced_skip_dec: whether the chunk ending at `end` skips its decoder half. Only when the NEXT chunk (same grid) is itself a CED chunk, which rewrites the
        decoder rings and the drafter ring from its own tail: then the final logits and every cache the decode reads are what the full chunk gives, bit for bit.
        Never on the last chunk, never ahead of a short chunk (<= tail rows: stock path, reads the rings) or a resume. Mode 1 also needs that no snapshot is taken at
        `end` (memory block, disk block, cut); mode 2 skips anyway and `_prefill_step` then takes none there (a snapshot with stale decoder rings, restored and followed
        by a short chunk, would give other output). Mode 3 skips too and `_prefill_step` takes the snapshot / checkpoint there marked stale: a lookup uses it only where
        the prompt after it starts with a chunk that rewrites the rings (`_stale_ok`), else it falls back to an earlier whole one. Returns the mode (0 = no skip)."""
        can = getattr(self.backend, "ced_can_skip", None)
        if can is None or end >= L or s.resume:
            return 0
        if self._pf_share and self._share_takers(s, end):
            return 0                                                  # a session takes the cache here: it needs the whole chunk (decoder rings included)
        nxt = self._chunk_end(s, end, L)
        mode = can(s.cache, nxt - end)
        if not mode:
            return 0
        if mode == 1:
            cut = end in s.cuts or (end == self._tail_anchor(s, L) and not s.resume)
            snap = (self.prefix.budget > 0 and (end % self.block == 0 or cut)) or (self.prefix.disk is not None and (end % self.prefix.disk.block == 0 or cut))
            if snap:
                return 0
        return mode

    def _turn_prefill(self) -> None:
        """Flag pf_fair: among the sessions waiting to prefill, run the next chunk of the one whose last chunk started longest ago (one chunk each, round robin;
        a session that has not run yet goes first). The prefill queue was strict FIFO, so a short request that arrived behind a 100K prompt waited for every
        chunk of it (first token = the whole prefill). Each session keeps its own chunk grid and its own cache, so the order changes no arithmetic.
        Flag pf_share: a session whose share target is still ahead of the running session is left in the queue: it would prefill from 0 the very tokens the
        leader is about to hand it, which is longer than the wait.
        Flag pf_srpt: the turn goes to the session with the fewest tokens left. Equal prompts then finish one after the other instead of all at the end: same total
        work, mean first token about 5/8 of round robin at N=4, and fewer session switches."""
        if len(self.prefilling) < 2:
            return
        head = self.prefilling[0]
        cand = [o for o in self.prefilling if o is head or not self._awaiting_seed(head, o)]
        if os.environ.get("KYOJIN_PF_SRPT", "1") not in ("0", "") or self._pf_floor:
            now = time.perf_counter()               # fewest tokens left first; a waiting session gains PF_SRPT_AGE tokens per second, so none starves
            s = min(cand, key=lambda x: (len(x.ids) - x.pos) - PF_SRPT_AGE * (now - x.t_start))
        else:
            s = min(cand, key=lambda x: x.t_pf_turn)
        if s is not head:
            self.prefilling.remove(s)
            self.prefilling.insert(0, s)
        s.t_pf_turn = time.perf_counter()

    def _awaiting_seed(self, head: Session, o: Session) -> bool:
        """Flag pf_share: o has not started and the running session will hand it a cache at a chunk end it has not reached yet (share_held counts
        the chunk turns a session was held that way)."""
        if not self._pf_share or o.cache is not None or o.pos != 0 or o.prefilled or o.resume:
            return False
        t = self._share_target(head, o)
        if t > head.pos:
            self.stats["share_held"] += 1
            return True
        return False

    def _prefill_step(self) -> None:
        if self._pf_fair:
            self._turn_prefill()
        s = self.prefilling[0]
        if s.cache is None:
            self._begin_prefill(s)
        L = len(s.ids)
        if s.pos >= L:                                              # a rebuilt session: the prompt is in, replay what it had produced
            self._replay_step(s)
            return
        end = self._chunk_end(s, s.pos, L)
        skip_mode = self._skip_mode(s, end, L)
        skip = bool(skip_mode)
        stale = skip_mode == 3
        warm = getattr(self.backend, "warm_prompt", None)
        if warm is not None and end < L:
            warm(s.rids, end, min(L, end + max(self.chunk, self.big_chunk or 0)))   # the next chunk's table rows load while this one runs
        vl = {}
        if s.vl:                                                    # new images of this chunk: encoded here, off the chunk's clock
            te = time.perf_counter()
            self.stats["n_vl_encode"] += ds41_vision.prepare(s.vl, s.pos, end)
            self.stats["t_vl_encode"] += time.perf_counter() - te
            vl = {"vl": (s.vl, s.pos)}
        if self._gpu_span:                                          # probe gpu_span: a prefill chunk is bracketed too, keyed by its own rows
            GS.begin()
        if TT.agg():
            TT.start_agg()
        t0 = time.perf_counter()
        self._chunk_t0 = t0
        fl0 = (list(self.members), sum(x.n_gen for x in self.members)) if self._pf_floor else None
        self._inline_s = 0.0
        sliced = (not s.resume) and self._arm_yield()
        try:
            if skip:
                lg = self.backend.prefill_chunk(s.cache, s.rids[s.pos:end], skip_dec=True, **vl)
            elif self.spec:
                lg = self.backend.prefill_chunk(s.cache, s.rids[s.pos:end], draft=s.draft, **vl)
            else:
                lg = self.backend.prefill_chunk(s.cache, s.rids[s.pos:end], **vl)
        finally:
            if sliced:
                self.backend.set_prefill_yield(None)
        t1 = time.perf_counter()
        t0 += self._inline_s                                         # the decode rounds that ran inside the chunk are not prefill time (stats, debt)
        if fl0 and fl0[0] and fl0[0] == self.members and t1 > self._chunk_t0:       # same decoders before / after: their tokens over the chunk's wall
            ach = (sum(x.n_gen for x in self.members) - fl0[1]) / len(fl0[0]) / (t1 - self._chunk_t0)
            if ach > 0:
                self._floor_k = min(max(self._floor_k * min(max(self._pf_floor / ach, 0.85), 1.2), 1.0), 3.0)
        if self._pf_floor and (self.members or self.ready):
            r = self._floor_duty() / (1.0 - self._floor_duty())       # flag pf_floor: the bursts inside the chunk (inline_s) are already paid
            self._mix_debt = (t1 - t0) * r - self._inline_s + min(self._mix_debt, 0.0)
        elif self._mix_decode and (self.members or self.ready):
            m = self._mix_decode
            self._mix_debt = max(self._mix_debt, 0.0) + (t1 - t0) * m / (100 - m)     # the decoders are owed m % of the wall: decode time = (t1 - t0) * m / (100 - m)
        if self._gpu_span:
            GS.end(self.stats, end - s.pos)                          # gpu_*_r<chunk rows>; prefill_chunk evals, so the round is complete
        self.stats["t_pf_chunk"] += t1 - t0; self.stats["n_pf_chunk"] += 1
        if s.stages is not None:                                     # ttft_stages: the first chunk holds the first eval of the prompt, then the whole prefill
            d = (t1 - t0) * 1e3
            s.stages.setdefault("first_chunk_ms", round(d, 3))
            s.stages["prefill_ms"] = round(s.stages.get("prefill_ms", 0.0) + d, 3)
            s.stages["chunks"] = s.stages.get("chunks", 0) + 1
        ah = getattr(self.backend, "engram_ahead", None)
        if ah is not None:                                               # content counters (tools/pfcontent.py): absolute values, the tool takes deltas
            self.stats["pf_eng_rows_total"] = ah.rows_total
            self.stats["pf_eng_rows_uniq"] = ah.rows_uniq
            self.stats["pf_eng_job_ms"] = round(ah.job_seconds * 1e3, 3)
            self.stats["pf_eng_warm_first"] = ah.warmed_first
        if ah is not None and ah.waits:                                  # host time inside each Engram layer of the chunk (row wait + build)
            for k, v in ah.waits.items():
                self.stats[f"t_pf_eng{k}"] += v
            ah.waits.clear()
        s.hasher.update(np.asarray(s.ids[s.pos:end], dtype=np.int32).tobytes())
        self.stats["prefill_tokens" if not s.resume else "rebuild_prefill_tokens"] += end - s.pos
        s.pos = end
        if s.path is not None:
            s.path = (*s.path, end)
        self._note_grid(s, end)
        if self._spec_lookup:
            self._lookup_sync(s, end)                                # incremental 4-gram insert for this chunk
        cut = end in s.cuts
        tail = end == self._tail_anchor(s, L) and not s.resume and end not in s.cuts and end % self.block and end % (self.prefix.disk.block if self.prefix.disk is not None else self.block)
        cut = cut or bool(tail)
        if skip:
            self.stats["pf_skip_dec"] += 1                              # decoder half skipped: the decoder rings are stale; modes 1 and 2 store nothing here, mode 3 stores marked
        if (not skip or stale) and (end % self.block == 0 or cut):
            self.prefix.store((end, s.hasher.digest()), s.cache, s.draft, stale=stale, sig=s.path)
            if stale:
                self.stats["pf_skip_snap"] += 1
        if (not skip or stale) and self.prefix.disk is not None and (end % self.prefix.disk.block == 0 or cut):
            td = time.perf_counter()
            self.prefix.save_disk(s.ids, end, s.cache, s.draft, stale=stale, **self._disk_digests(s, s.ids, end))
            if tail:
                self.stats["kv_tail_saved"] += 1                           # flag kv_tail: the anchor's file (its bytes and ms: disk.stats / status)
                self.stats["t_kv_tail_save"] += time.perf_counter() - td
            if stale:
                self.stats["pf_skip_snap_disk"] += 1
        if self.kv_budget is not None:
            self._measure(s)
        if self._pf_share and not s.resume:
            takers = self._share_takers(s, end)
            if takers:
                self._hand_off(s, end, takers)
        self.stats["t_pf_after"] += time.perf_counter() - t1               # hash, prefix snapshot, disk save, KV measure
        if end < L:
            return
        if s.resume:
            self._replay_step(s)
            return
        self.prefilling.pop(0)
        s.prefilled = True
        s.t_first = time.perf_counter()
        TT.stop()
        if TT.on():
            TT.start()
        tok = int((self.tap(lg) if self.tap is not None else sample_rows(lg, [s.sampling], [self._next_key(s)])).item())
        t_s0 = time.perf_counter()
        self.stats["t_pf_sample0"] += t_s0 - s.t_first
        if s.stages is not None:
            s.stages["sample0_ms"] = round((t_s0 - s.t_first) * 1e3, 3)
        reason = self._accept(s, tok)
        if reason:
            self._finish(s, reason)
            return
        self._measure(s)
        self.ready.append(s)

    def _hand_off(self, s: Session, end: int, takers: list) -> None:
        """Flag pf_share: give `takers` the cache of s at `end` as one shared Seed (a view: bytes are copied only when a holder writes, which the first chunk of
        each does once). Over the memory budget no seed is made and the takers take the normal path (prefix snapshots, else a prefill from the start)."""
        snap = self.backend.fork(s.cache)
        dsnap = self.backend.draft_snapshot(s.draft) if (self.spec and s.draft is not None) else None
        size = self.backend.nbytes(snap) + (self.backend.draft_nbytes(s.draft) if (self.spec and s.draft is not None) else 0)
        if self.kv_budget is not None and self.kv_bytes() + size > self.kv_budget:
            self.stats["share_refused_budget"] += len(takers)
            return
        seed = Seed(end, snap, dsnap, size, tuple(s.path))
        for o in takers:
            o.share_seed = seed
        self.stats["share_seeds"] += 1
        self.stats["share_takers"] += len(takers)

    def _arm_yield(self) -> bool:
        """Flag mix_slice_ms: hand the backend a callback for the layer bounds of this chunk when sessions decode and the backend can run it between layers."""
        ms = self._mix_slice_ms
        if self._pf_floor:
            ms = ms or 1                                            # flag pf_floor: the burst rule in _pf_yield decides, not the slice
        if not ms or not (self.members or self.ready) or not getattr(self.backend, "supports_yield", False):
            return False
        self.backend.set_prefill_yield(self._pf_yield)
        return True

    def _pf_yield(self) -> None:
        """Called by the backend at a layer bound inside a prefill chunk: one decode round when the last one STARTED `mix_slice_ms` ago. Never re-entered.
        Counted from the round's start, not its end: a round's own duration is what the decoder pays anyway, and adding it to the stretch would push the
        next round one round further away whenever a round is slow (the stall that makes a slice arm worse than one round per chunk)."""
        if self._in_yield or not (self.members or self.ready):
            return
        if self._pf_floor:
            self._floor_yield()
            return
        ms = self._mix_slice_ms
        if not ms or (time.perf_counter() - self._last_dec_end) * 1000.0 < ms:
            return
        self._in_yield = True
        t = time.perf_counter()
        try:
            self._decode_phase()
            self.stats["mix_slice_rounds"] += 1
        finally:
            self._in_yield = False
            d = time.perf_counter() - t
            self._inline_s += d
            self.stats["t_pf_inline_decode"] += d

    def _floor_yield(self) -> None:
        """Flag pf_floor, at a layer bound: decode in BURSTS, not one round per slice. The decoders are owed duty/(1-duty) of the prefill time spent in this chunk
        (minus what they already got inside it); once the debt reaches pf_floor_period_ms * that ratio, rounds run back to back until it is paid. A burst pays the
        switch from prefill to decode (bound eval, batch sync) once for many rounds, where mix_slice_ms pays it for every round."""
        d = self._floor_duty()
        if d <= 0.0:
            return
        r = d / (1.0 - d)
        now = time.perf_counter()
        owed = (now - self._chunk_t0 - self._inline_s) * r - self._inline_s + min(self._mix_debt, 0.0)
        cu = self._floor_catchup
        if cu > 0.0:                                                # a decoder behind its floor: the debt is at least its deficit, within cu of the chunk's wall
            lim = (now - self._chunk_t0 - self._inline_s) * cu / (1.0 - cu) - self._inline_s + min(self._mix_debt, 0.0)
            owed = max(owed, min(self._floor_deficit(now), lim))
        per = max(int(os.environ.get("KYOJIN_PF_FLOOR_PERIOD_MS", "1000") or 0), 0) / 1000.0
        if owed < max(per * r, self._rd_s):
            return
        self._in_yield = True
        t_all = time.perf_counter()
        n = 0
        try:
            while (self.members or self.ready) and time.perf_counter() - t_all < owed:
                self._decode_phase()
                n += 1
        finally:
            self._in_yield = False
            dt = time.perf_counter() - t_all
            self._inline_s += dt
            self.stats["t_pf_inline_decode"] += dt
            self.stats["mix_slice_rounds"] += n
            self.stats["floor_bursts"] += 1
            self.stats["floor_burst_rounds"] += n

    def _replay_step(self, s: Session) -> None:
        """Feed the next tokens it had produced to a rebuilt session's cache (teacher forced; the cache covers prompt + out[:-1])."""
        toks = s.out[:-1]
        upto = min(len(toks), s.replay_pos + self.replay_chunk)
        if upto > s.replay_pos:
            s.cache = self.backend.replay(s.cache, toks[s.replay_pos:upto], s.draft if self.spec else None)
            self.stats["replay_tokens"] += upto - s.replay_pos
            s.replay_pos = upto
        if upto < len(toks):
            return
        self.prefilling.pop(0)
        s.resume = False
        self._measure(s)
        self.ready.append(s)

    # ------------------------------------------------------------------------------------------------ decode
    def _launch(self):
        toks = mx.array([s.last for s in self.members], dtype=mx.int32) if self.cur is None else self.cur
        samp = [s.sampling for s in self.members]
        keys = [self._next_key(s) for s in self.members]
        pick = self.tap if self.tap is not None and len(self.members) == 1 else (lambda lg: sample_rows(lg, samp, keys))
        if getattr(self.backend, "gated", None) is not None and not self.spec:
            return self.backend.launch_gated(self.bc, toks, pick)      # submits itself (Engram rows read after the build)
        logits = self.backend.step(self.bc, toks)
        y = pick(logits)
        mx.async_eval(y)
        return y

    def _consume(self, y, leaving: dict, allow_yield: bool) -> None:
        """Read the tokens of one launched step and hand them to their sessions; sessions that end are added to `leaving` (s -> reason)."""
        vals = y.tolist()
        for s, tok in zip(self.members, vals):
            if s in leaving:
                continue
            reason = self._accept(s, int(tok))
            if reason is None and allow_yield and ((s.yield_after is not None and s.seg_gen >= s.yield_after and self.waiting) or s.park_now):
                reason = "yield"
            if reason:
                leaving[s] = reason

    def _recast(self, leaving: dict) -> bool:
        """Flag mu_keep: keep the batched cache and update its membership in place — the members that stay keep their rows (no write-back, no
        re-merge, no re-stack, no re-pack), only the rows of the sessions that leave or join are read or written. False (nothing touched) when
        the epoch cannot be recast: the flag is off, the backend has no `recast`, the cache is not a slot stack, or a pass is open. The caller
        launches the next step (cur / nxt are dropped: both steps in flight were read)."""
        if not isinstance(self.bc, (list, tuple)) or len(self.bc) != 1 or os.environ.get("KYOJIN_MU_KEEP", "1") in ("0", ""):
            return False                                             # None, or the spec path's own _Bc (each session verified on its own)
        recast = getattr(self.backend, "recast", None)
        c = self.bc[0]
        if recast is None or getattr(c, "slots", None) is None or getattr(c, "_counts", None) is not None:
            return False
        drop = [i for i, s in enumerate(self.members) if leaving.get(s) not in (None, "yield")]
        park = [i for i, s in enumerate(self.members) if leaving.get(s) == "yield"]
        add = list(self.ready)
        if not (drop or park or add):
            return False
        try:
            got = recast(self.bc, drop, park, [s.cache for s in add])
        except NotImplementedError:                               # a backend whose batch cache is not a stacked slot cache
            return False
        keep = [s for s in self.members if leaving.get(s) is None]
        for i, s in enumerate(self.members):
            reason = leaving.get(s)
            if reason is None:
                continue
            if reason == "yield":
                s.cache = got[i]
                self._park(s)
            else:
                self._finish(s, reason)
        self.members = keep
        if add:
            self.members.extend(add)
            self.ready.clear()
            for s in add:
                s.cache = None                                        # the batched cache owns the state now
        self.cur = self.nxt = None
        if self.spec and self.members and hasattr(self.backend, "bind_drafts"):
            self.backend.bind_drafts(self.bc, [s.draft for s in self.members])
        if not self.members:
            self.bc = None
            self.bc_bytes = 0
            return True
        self.bc_bytes = sum(s.kv_bytes for s in self.members)
        self.stats["mu_recast"] += 1
        return True

    def _dissolve(self, leaving: dict) -> None:
        """Return the members to their own caches (the batched cache is rebuilt for the next membership)."""
        n = len(self.members)
        if os.environ.get("KYOJIN_FIN_NOSYNC", "1") not in ("0", "") and all(leaving.get(s) not in (None, "yield") for s in self.members):
            caches = [None] * n                                  # flag fin_nosync: every member finishes and drops its cache, nothing reads the rows
        else:
            caches = self.backend.unbatch(self.bc, n)
        keep = []
        for s, c in zip(self.members, caches):
            reason = leaving.get(s)
            if reason is None:
                s.cache = c
                self._measure(s)
                keep.append(s)
            elif reason == "yield":
                s.cache = c
                self._park(s)
            else:
                self._finish(s, reason)
        self.members = keep
        self.bc = self.cur = self.nxt = None
        self.bc_bytes = 0

    def _park(self, s: Session) -> None:
        """The session gives its slot back; its cache stays (until the memory budget asks for it)."""
        self._measure(s)
        s.park_now = False
        s.t_parked = time.perf_counter()
        self.waiting.append(s)
        self.stats["yields"] += 1

    def _decode_tick(self) -> None:
        if self.ready and self.bc is not None:                   # a joiner: end the epoch, read what is in flight first
            leaving: dict = {}
            self._consume(self.cur, leaving, False)
            if self.nxt is not None:
                self.cur, self.nxt = self.nxt, None
                self._consume(self.cur, leaving, False)
            if self._recast(leaving):                            # flag mu_keep: the members that stay keep their rows
                if self.members:
                    self.cur = self._launch()
                return
            self._dissolve(leaving)
        if self.ready:
            self.members.extend(self.ready)
            self.ready.clear()
            self.bc = None
        if not self.members:
            return
        if self.bc is None:
            tr = os.environ.get("KYOJIN_TICK_TRACE", "0") == "1"
            t0 = time.perf_counter()
            self.bc = self.backend.batch([s.cache for s in self.members])
            if self.spec and hasattr(self.backend, "bind_drafts"):
                self.backend.bind_drafts(self.bc, [s.draft for s in self.members])      # plain steps keep feeding the draft states
            t1 = time.perf_counter()
            self.cur = self._launch()
            if tr:
                t2 = time.perf_counter()
                mx.eval(self.cur)
                print(f"tick_trace epoch n={len(self.members)} batch {(t1 - t0) * 1e3:.0f} ms launch {(t2 - t1) * 1e3:.0f} ms eval {(time.perf_counter() - t2) * 1e3:.0f} ms", file=sys.stderr, flush=True)
            self.bc_bytes = sum(s.kv_bytes for s in self.members)
            for s in self.members:
                s.cache = None                                   # the batched cache owns the state now
            self.stats["epochs"] += 1
            return
        if self.kv_budget is not None and self.stats["decode_ticks"] % self.kv_refresh == 0:
            self.bc_bytes = self.backend.nbytes(self.bc)
        # steady state: launch the next step from the lazy tokens of the one in flight, then read the one in flight
        if self.lookahead and self.nxt is None and not self.ready and self._lookahead_ok():
            self.nxt = self._launch()
        leaving = {}
        self._consume(self.cur, leaving, self.nxt is None)
        if leaving:
            if self.nxt is not None:
                self.cur, self.nxt = self.nxt, None
                self._consume(self.cur, leaving, False)
            if self._recast(leaving):                            # flag mu_keep: the members that stay keep their rows
                if self.members:
                    self.cur = self._launch()
                return
            self._dissolve(leaving)
            return
        self.cur, self.nxt = self.nxt, None
        if self.cur is None:
            self.cur = self._launch_from_last()

    def _lookahead_ok(self) -> bool:
        """No session is within one token of its end or of yielding its slot (the step launched ahead would be for nothing)."""
        for s in self.members:
            if s.n_gen + 2 >= s.max_new or s.n_prompt + s.n_gen + 2 >= self.ctx:
                return False
            if (s.yield_after is not None and s.seg_gen + 2 >= s.yield_after) or s.park_now:
                return False
        return True

    def _launch_from_last(self):
        """Launch a step from the last emitted token of each member (no step in flight)."""
        self.cur = None
        return self._launch()

    # ------------------------------------------------------------------------------------------------ speculative decode
    def _prompt_tag(self, s: Session) -> str:
        """First 8 hex of the prompt's token hash (flag round_log): aligns the per-round records of two arms on one prompt."""
        ph = getattr(s, "_rlog_ph", None)
        if ph is None:
            ph = s._rlog_ph = hashlib.md5(np.asarray(s.ids, dtype=np.int32).tobytes()).hexdigest()[:8]
        return ph

    def _plan_depth(self, s: Session) -> int:
        """Drafts to ask for this round: the best D in 0..mtp_depth for the acceptance seen so far (tokens per round / cost of the round, in solo
        steps, the fixed cost of a step shared by the sessions of the pass), never more than the reply can still use. D = 0 is probed with D = 1 every
        `probe_every` rounds so that it can recover. flag nondet_fix: D = cap, and no policy below reads a measured ms (the width moves a row's bits)."""
        room = min(s.max_new - s.n_gen, self.ctx - s.n_prompt - s.n_gen) - 1
        cap = min(self.mtp_depth * self._spec_chain, room)          # flag spec_chain: one plan spans that many draft blocks
        mrows = int(getattr(self.backend, "max_spec_rows", 0) or 0)
        if mrows:
            cap = min(cap, mrows - 1)
        if cap <= 0:
            return 0
        if self.depth_override is not None:
            return min(cap, self.depth_override)
        if self._round_D is not None and len(self.members) > 1:       # the tuner's arm for this round
            return min(cap, self._round_D)
        md = getattr(self.backend, "multi_depth", None)
        if md and len(self.members) > 1:                              # several sessions in one pass: measured fixed depth (the solo cost model does not hold)
            return min(cap, md)
        if NONDET_FIX:                                                # flags.py nondet_fix: no wall time in the plan (the width moves a row's bits)
            FD.hit("nondet_plan")
            return cap
        if self._dspol_v2 and len(self.members) <= 1:                 # flag dspol_v2: one calibrated policy, R = 1 (plain) .. cap + 1 rows
            FD.hit("dspol_v2")
            return self._plan_v2(s, cap)
        if self.spec_costs is None and len(self.members) <= 1 and os.environ.get("KYOJIN_SPEC_RATE", "1") != "0":
            FD.hit("spec_rate")
            return self._plan_rate(s, cap)
        k = 2.0
        p = (s.acc_succ + 0.7 * k) / (s.acc_trial + k)               # per-position acceptance, prior 0.7 worth two trials
        if self.spec_costs is not None and len(self.members) <= 1:
            pol = getattr(s, "dspol", None) if self._dspol_on else None
            best = self._plan_dspol(s, pol, cap) if pol is not None and pol.ready else self._plan_measured(s, p, cap)
            if 0 < best < cap:                                       # the deep positions are only measured when drafted: draft at the cap
                s.cap_rounds += 1                                    # every cap_probe rounds so that their acceptance stays current
                if s.cap_rounds >= self.cap_probe:
                    best = cap
            if best == cap:
                s.cap_rounds = 0
            if best == 0:
                s.idle_rounds += 1
                if s.idle_rounds >= (getattr(s, 'probe_gap', 0) or self.probe_every):
                    s.idle_rounds, s.probing = 0, True
                    return self._probe_depth(cap)
            return best
        # A plain row of a batch of n sessions costs its own row plus 1/n of the fixed cost of the step (the dense weights are read once for all
        # rows): row_cost + (1 - row_cost) / n solo steps, not a whole solo step. So drafts must pay for themselves against a cheaper baseline
        # when sessions share the pass (n = 1: the solo step, as before).
        base = self.row_cost + max(0.0, 1.0 - self.row_cost) / max(1, len(self.members))
        pol = getattr(s, "dspol", None) if (self._dspol_on and len(self.members) <= 1) else None
        if pol is not None and pol.ready:
            best = self._plan_dspol(s, pol, cap)
            # Cap probing for dspol: if policy picks an intermediate depth, occasionally force the cap
            # so deep positions stay measured (same logic as the spec_costs path).
            if 0 < best < cap:
                s.cap_rounds += 1
                if s.cap_rounds >= self.cap_probe:
                    best = cap
            if best == cap:
                s.cap_rounds = 0
            if best:
                return best
        else:
            fit = self._fit_cost() if len(self.members) <= 1 else None
            best, best_g, tok = 0, 1.0 / base, 1.0
            for d in range(1, cap + 1):
                tok += p ** d
                g = tok / (fit.cost_ms(d) / fit.s_ms) if fit is not None else tok / (self.spec_k * (base + self.row_cost * d + self.draft_cost))
                if g > best_g:
                    best, best_g = d, g
            if best:
                return best
        # D = 0 won: probe with D = 1 every probe_every rounds (the probes cost ~2 % of the plain rounds)
        s.idle_rounds += 1
        if s.idle_rounds >= max(getattr(s, 'probe_gap', 0) or self.probe_every, self._probe_floor(base)):
            s.idle_rounds, s.probing = 0, True
            return self._probe_depth(cap)
        return 0

    def _plan_rate(self, s: Session, cap: int) -> int:
        """Solo depth from measured rates: the cap, unless the session's tokens per cap round over the median ms of a cap round (warm samples)
        fall below RATE_MARGIN x one token per median plain round; then D = 0, and every rate_gap rounds a probe: a fresh window of RATE_MIN cap rounds (the gap doubles
        while the probes confirm plain, up to PROBE_CAP x probe_every). Served 30/10 (DS4.1, Ultra): fixed 3 beat the predictor x cost policies on every text (50.9 tok/s vs 45.8 / 48.0,
        plain 33.8; 0.70 acceptance = the lowest seen still drafts 35.0 vs 33.9 plain), the policies lost on cold lock-ins at D = 0 after one miss
        or one slow first sample. A cold engine drafts: plain needs RATE_MIN cap rounds, two warm cap samples and a measured plain round."""
        full = min(self.mtp_depth * self._spec_chain, max(1, int(getattr(self.backend, "max_spec_rows", 0) or 64) - 1))
        if self._rate_pair:
            return self._plan_pair(s, cap)
        if s.cap_n < (2 if s.rate_gap else RATE_MIN):                # a probe's window decides from 2 rounds (probe cost ~1.5 % of the plain rounds)
            return cap
        if not self.plain_ms:                                        # two plain rounds measure it (the first one is not warm), once per engine;
            s.idle_rounds += 1                                       # a backend whose plain rounds never time (no lookahead) keeps the cap
            return 0 if s.idle_rounds <= 2 else cap
        ms = self._d_ms.get(full, ())
        if len(ms) < 2:
            return cap
        pr = sorted(self._plain_ring or [self.plain_ms])
        if s.cap_tok * pr[len(pr) // 2] >= RATE_MARGIN * sorted(ms)[(len(ms) - 1) // 2]:   # upper median of plain, lower of the cap: ties draft
            s.idle_rounds = 0
            if s.cap_n >= RATE_MIN:                                  # a probe's win counts once a full window holds it (two lucky rounds do not
                s.rate_gap = 0                                       # reset the gap: the next round re-decides)
            return cap
        s.idle_rounds += 1
        if s.idle_rounds >= (s.rate_gap or self.probe_every):
            s.idle_rounds, s.cap_n = 0, 0
            s.rate_gap = min(2 * (s.rate_gap or self.probe_every), PROBE_CAP * self.probe_every)
            return cap
        return 0

    # ------------------------------------------------------------------------------------------------ paired rate rule (flag rate_pair)
    @staticmethod
    def _pair_rate(win: list, need: int) -> float | None:
        """Tokens per ms of a window of (tokens, ms) rounds: the SAME rounds on both sides of the sum, so the acceptance and the cost
        belong to the engine state the round was served in. None below `need` samples or with no measured time."""
        if len(win) < need:
            return None
        tok = sum(t for t, ms in win if ms > 0)
        ms = sum(ms for t, ms in win if ms > 0)
        return tok / ms if ms > 0 else None

    def _plan_pair(self, s: Session, cap: int) -> int:
        """The cap, unless the drafted window's measured tok/ms is below RATE_MARGIN x the plain window's. Both rates come from the
        rounds this engine actually served, so an engine that gets faster on either path moves the estimate the round it is measured;
        the decision never rests on a token mean whose rounds are not the rounds whose cost is compared (spec_rate's cap_tok counts only
        the rounds the drafter filled, which on a class with many short proposals reads well above the round's true tokens).
        A plain spell keeps the same probe schedule (one drafted round every rate_gap); the probe's own round joins the drafted window
        and the next decision still uses PAIR_MIN_D rounds, so one unlucky probe no longer re-decides from two rounds."""
        if s.cap_n < RATE_MIN:                                       # a fresh session drafts the cap until its rate is measured
            return cap
        if not self.plain_ms:                                        # the first two plain rounds measure the plain side (and seed its window)
            s.idle_rounds += 1
            return 0 if s.idle_rounds <= 2 else cap
        rd = self._pair_rate(self._rd_win, PAIR_MIN_D)
        rp = self._pair_rate(self._rp_win, PAIR_MIN_P)
        if rd is None or rp is None:                                 # not enough of the window measured yet: the cap keeps drafting
            return cap
        if rd >= RATE_MARGIN * rp:
            s.idle_rounds = 0
            if s.cap_n >= RATE_MIN:
                s.rate_gap = 0
            return cap
        s.idle_rounds += 1
        if s.idle_rounds >= (s.rate_gap or self.probe_every):
            s.idle_rounds = 0
            s.rate_gap = min(2 * (s.rate_gap or self.probe_every), PROBE_CAP * self.probe_every)
            return cap                                               # a probe: this drafted round enters _rd_win, the window decides next
        return 0

    def _learn_k(self) -> None:
        """Cost of a drafted round against the formula (row_cost, draft_cost), from the last drafted solo round and the measured plain round."""
        if self._spec_obs is None or not self.plain_ms:
            return
        ms, d, tok = self._spec_obs
        self._spec_obs = None
        if d not in self._k_warm:                                    # the first drafted round of a depth builds its verify / draft graph (seconds): not a cost sample.
            self._k_warm.add(d)                                      # Without this one cold round pinned spec_k at its cap and the depth policy never drafted again.
            return
        self._d_ms[d] = (self._d_ms.get(d, []) + [ms])[-K_WINDOW:]
        self._rd_win = (self._rd_win + [(tok, ms)])[-PAIR_WINDOW:]   # flag rate_pair: the same round's tokens against its own ms
        k = min(4.0, ms / self.plain_ms / (1.0 + self.row_cost * d + self.draft_cost))
        if self.robust:
            self._k_samples = (self._k_samples + [k])[-K_WINDOW:]
            srt = sorted(self._k_samples)
            k = srt[(len(srt) - 1) // 2]                             # (lower) median: one or two slow rounds out of five never move it
            self.spec_k = min(40.0, max(0.5, k))
        else:
            self.spec_k = min(40.0, max(0.5, k if self._k_seen == 0 else 0.5 * self.spec_k + 0.5 * k))
        self._k_seen += 1
        self.stats['spec_k_x100'], self.stats['plain_ms_x100'] = int(100 * self.spec_k), int(100 * (self.plain_ms or 0))

    def _fit_cost(self):
        """Measured cost of a solo round as DSPOL.CostModel(S, slope, draft): S = the plain EMA, the drafted rounds a line through the median ms
        of each measured depth (one depth: the formula's slope through that point); a depth counts from its second sample. The formula's single scale spec_k charges every row
        row_cost plain steps; on DS4.1 a row costs ~0.2 (fixed 3 = 1.75 plain steps), so one miss (p 0.45) priced D = 1 above plain and the
        policy sat at D = 0 for 16+ rounds (30/10 trace: 10 % below fixed 3). None before a drafted depth is measured or with KYOJIN_SPEC_FIT=0."""
        pts = [(d, sorted(v)[(len(v) - 1) // 2]) for d, v in self._d_ms.items() if len(v) >= 2]   # one sample of a depth may be a stall
        if not self.plain_ms or not pts or os.environ.get("KYOJIN_SPEC_FIT", "1") == "0":
            return None
        FD.hit("spec_fit")
        S = self.plain_ms
        md, mm = sum(d for d, _ in pts) / len(pts), sum(m for _, m in pts) / len(pts)
        if len(pts) >= 2:
            slope = max(1e-3, sum((d - md) * (m - mm) for d, m in pts) / sum((d - md) ** 2 for d, _ in pts))
        else:
            slope = S * self.spec_k * self.row_cost
        return DSPOL.CostModel(S, slope, max(0.0, mm - slope * md - S))

    def _probe_depth(self, cap: int) -> int:
        """Depth of a D = 0 probe: 1, or the cap while the fit has fewer cap samples than D = 1 ones (D = 1 probes alone leave the fit one
        point and the formula's slope; cap probes give it the second depth)."""
        if cap > 1 and os.environ.get("KYOJIN_SPEC_FIT", "1") != "0" and len(self._d_ms.get(cap, ())) < len(self._d_ms.get(1, ())):
            return cap
        return 1

    def _probe_floor(self, base: float) -> int:
        """Rounds between two probes so that the probes cost at most ~2 % of the plain rounds: a probe round costs spec_k x (a D=1 round) plain steps."""
        extra = self.spec_k * (base + self.row_cost + self.draft_cost) - base
        cap = PROBE_CAP * self.probe_every if self.robust else 4096
        return min(cap, int(50 * extra)) if extra > 0 else 0

    # ------------------------------------------------------------------------------------------------ cost ladder (flag spec_ladder)
    def _plan_ladder(self, members: list, cap: int) -> int:
        """One depth for the whole verify pass from the measured cost ladder (kyojin_mlx.ladder): expected tokens over the
        round ms measured at each depth, against the measured plain rate; 0 = the ordinary plain tick. Cold: draft at the
        cap for two rounds, then a few plain ticks so the ladder has a plain rate to compare with."""
        if cap <= 0:
            return 0
        if self.depth_override is not None:
            return min(cap, self.depth_override)
        n = max(1, len(members))
        plain = self.plain_ms if n == 1 else self._plain_n.get(n)    # ms of a plain pass of THESE n sessions (the solo ms does not scale by 1/n: the weights are read once)
        if not plain:                                                # no plain rate yet: measure one (the ladder cannot price without it)
            self._ladder_cold += 1
            if self._ladder_cold <= 8:                               # two one-draft rounds, then plain ticks that time it
                return min(cap, 1) if self._ladder_cold <= 2 else 0
            return min(cap, 2)                                       # a backend whose plain rounds never time: a short draft, never the cap
        self._ladder_cold = 0
        acc = self._pooled_acceptance(cap)                           # per-position acceptance, pooled over the sessions of the pass
        return self._ladder_for(n).plan(cap, acc, plain / n, n)

    def _ladder_for(self, n: int):
        return self._ladder if n <= 1 else self._ladders.setdefault(n, LADDER.Ladder())

    def _round_depths(self, members: list) -> list:
        """Depths of this round when flag spec_ladder owns the plan: the ladder's width for the pass. With several sessions the ladder may
        only NARROW the per-session policy, never widen it: the sessions share one read of the weights, and `_plan_depth` prices a plain row
        of a shared pass at its own row plus 1/n of the fixed step (a round of n sessions costs another ms per width, not n times a solo
        round). The ladder's pass prices are noisy at startup, and widening there is where it drafts rows the shared-step economics never
        paid for."""
        depths = self._ladder_depths(members)
        if len(members) > 1:
            depths = [min(d, self._plan_depth(s)) for d, s in zip(depths, members)]
        return depths

    def _ladder_depths(self, members: list) -> list:
        """Depths of one ladder round: the ladder's width for the pass, clamped by the shared row cap (max_spec_rows / sessions,
        one row per session at least), the draft head's block (it cannot propose more), MAX_ROWS and each session's room."""
        n = max(1, len(members))
        rows = int(getattr(self.backend, "max_spec_rows", 0) or 0)
        cap = min(self.mtp_depth, LADDER.MAX_ROWS, max(0, rows // n - 1)) if rows else min(self.mtp_depth, LADDER.MAX_ROWS)
        blk = int(getattr(getattr(self.backend, "drafter", None), "block", 0) or 0)
        if blk:                                                      # the head proposes at most one block: a wider plan would be cut after the
            cap = min(cap, blk)                                      # verify paid for the rows, and priced as a width it never ran
        rooms = [min(s.max_new - s.n_gen, self.ctx - s.n_prompt - s.n_gen) - 1 for s in members]
        cap = min(cap, min(rooms)) if rooms else cap                # the pass ran at the smallest room: never plan a width the round cannot run
        d = self._plan_ladder(members, cap)
        md = getattr(self.backend, "multi_depth", None) if n > 1 else None
        if md:                                                       # the backend's measured fixed width for a shared pass
            d = min(d, max(0, int(md)))
        out = []
        for s, room in zip(members, rooms):
            out.append(max(0, min(d, room)))
        return out

    def round_ms(self, ctx: int, d: int) -> float:
        """Measured ms of a solo spec round at depth d and `ctx` tokens of context: linear in the context between (and beyond) the measured
        lengths, linear in d past the deepest measured depth."""
        t = self.spec_costs
        cs = sorted(t)
        def at(c):
            v = t[c]
            if d < len(v):
                return v[d]
            return v[-1] + (d - len(v) + 1) * (v[-1] - v[-2] if len(v) > 1 else v[-1])
        if len(cs) == 1:
            return at(cs[0])
        i = 1
        while i < len(cs) - 1 and cs[i] < ctx:
            i += 1
        c0, c1 = cs[i - 1], cs[i]
        a0, a1 = at(c0), at(c1)
        return max(1e-3, a0 + (a1 - a0) * (ctx - c0) / (c1 - c0))

    def _v2_choose(self, probs: list, max_rows: int):
        """flag dspol_v2: the rows R in 1..max_rows maximising E[T](R) / cost(R) (cost = the measured width table scaled by the live plain step)."""
        return DSP2.choose_rows(probs, self._dsp2_cost, self.plain_ms, max_rows)

    def _plan_v2(self, s: Session, cap: int) -> int:
        """flag dspol_v2, plan time: drafts D = R - 1 for the R the calibrated policy picks from the session's per-position confidence track.
        A session that has not drafted yet has no track: it drafts at the cap (the head's confidences are the policy's only input). R = 1 is the
        plain step; every probe_gap rounds of it a D = 1 round re-reads the head's confidences (the block is computed whatever D was verified), and
        a probe that fails doubles the gap, as for the other policies. Mode 2 asks the draft for the whole block and cuts after it (_v2_cut)."""
        st = s.dsp2 if s.dsp2 is not None else DSP2.State()
        s.dsp2 = st
        st.last = None
        if not st.track.ready:
            return cap
        probs = st.track.probs(self._dsp2_cal, cap)
        ch = self._v2_choose(probs, cap + 1)
        st.last = dict(ch.as_dict(), pk=[round(p, 3) for p in probs[:ch.rows - 1]], pa=[round(p, 3) for p in probs])
        self.stats["dsp2_plan_rows"] += ch.rows
        if ch.rows > 1:
            return cap if self._dspol_v2 >= 2 else ch.rows - 1
        s.idle_rounds += 1
        if s.idle_rounds >= (s.probe_gap or self.probe_every):
            s.idle_rounds, s.probing = 0, True
            return 1
        return 0

    def _v2_cut(self, members: list, rows: dict, depths: list, lookups: dict) -> None:
        """flag dspol_v2 mode 2, after the eager draft: this round's own confidences give p_k for the drafted rows, and the verify is built at the
        R that maximises E[T] / cost (rows past it are dropped from `rows`; R = 1 empties the list: the caller runs the plain step)."""
        dc = getattr(self.backend, "draft_confs", None)
        confs = dc() if callable(dc) else None
        if confs is None:
            return
        for i, s in enumerate(members):
            r = rows.get(i)
            if not r or i in lookups or i >= len(confs) or confs[i] is None:
                continue
            st = s.dsp2 if s.dsp2 is not None else DSP2.State()
            s.dsp2 = st
            conf = list(confs[i])
            st.track.update(conf)
            st.fresh = True
            probs = self._dsp2_cal.probs(conf[:len(r)])
            ch = self._v2_choose(probs, len(r) + 1)
            keep = ch.rows - 1
            if keep < len(r):
                self.stats["dsp2_cut_rows"] += len(r) - keep
                rows[i] = r[:keep]
            depths[i] = keep                                         # the verify is built at the chosen width: the width the round's ms, trace and cost fit are filed under
            if not keep:
                s.probing = False                                    # the probe did its job (fresh confidences), the plain step follows
            st.last = dict(ch.as_dict(), pk=[round(p, 3) for p in probs[:keep]], pa=[round(p, 3) for p in probs])

    @staticmethod
    def _v2_trace(s: Session) -> dict:
        """flag dspol_v2 -> round_log keys of the decision behind this round: pk (the p_k of the verified drafts), R (rows chosen), et (E[T] the
        choice expected), cms (its predicted round ms), pa (every block position's p, for a plain round), v2 (the flag's mode). Empty when the flag decided nothing."""
        st = s.dsp2
        if st is None or st.last is None:
            return {}
        out, st.last = dict(st.last, v2=DSP2.enabled()), None
        return out

    def _v2_observe(self, s: Session, conf: list, drafted: int, accepted: int) -> None:
        """flag dspol_v2: a verified round trains the shared calibration (reached positions only) and moves the session's confidence track."""
        st = s.dsp2 if s.dsp2 is not None else DSP2.State()
        s.dsp2 = st
        if not getattr(st, "fresh", False):
            st.track.update(conf)
        st.fresh = False
        self._dsp2_cal.observe(conf, drafted, accepted)

    def _plan_measured(self, s: Session, p: float, cap: int) -> int:
        """Best D in 0..cap for expected tokens per measured ms: E[tokens](D) = 1 + sum_j prod_{i<j} p_i, p_i = the decayed acceptance of draft
        position i given the earlier ones landed, shrunk (two trials' worth) toward the estimate of position i - 1 (position 0: toward the pooled p).
        Conditional acceptance does not fall with depth on streaky text (a draft that lands 3 tokens tends to land the 4th), so a seldom drafted
        position inherits its predecessor's value, not the pessimistic pooled geometric one."""
        ctx = s.n_prompt + s.n_gen
        k = 2.0
        best, best_g = 0, 1.0 / self.round_ms(ctx, 0)
        tok, run, prior = 1.0, 1.0, p
        for d in range(1, cap + 1):
            i = d - 1
            pi = (s.pos_succ[i] + prior * k) / (s.pos_trial[i] + k) if i < len(s.pos_trial) else prior
            prior = pi
            run *= pi
            tok += run
            g = tok / self.round_ms(ctx, d)
            if g > best_g:
                best, best_g = d, g
        return best

    def _plan_dspol(self, s: Session, pol, cap: int) -> int:
        """Width from the confidence head (kyojin_mlx.dspol): argmax expected tokens per ms. The cost is the measured
        round table at this context when the engine has one (the draft call is inside the measured rounds), else the
        operator table (KYOJIN_DSPOL_COSTS / _S / _SLOPE / _DRAFT), else a cost model derived from the measured
        plain_ms and spec_k (which capture the Engram gate's timing difference between plain and drafted rounds)."""
        FD.hit("dspol")
        probs = pol.predict(cap)
        if self.spec_costs is not None:
            ctx = s.n_prompt + s.n_gen
            r0 = self.round_ms(ctx, 0)
            n = max(1, min(cap, 5))
            slope = sum((self.round_ms(ctx, d) - r0) / d for d in range(1, n + 1)) / n
            cost = DSPOL.CostModel(r0, max(1e-3, slope), 0.0)
        elif DSPOL.explicit():
            cost = DSPOL.CostModel.load()
        else:
            # Use measured plain_ms and spec_k to build a cost model in actual milliseconds.
            # Engine cost for d=0: plain_ms. For d>0: plain_ms * spec_k * (1 + row_cost*d + draft_cost).
            # Match with dspol CostModel: cost(d) = S + slope*d + (draft if d>0 else 0).
            #   S = plain_ms
            #   slope = plain_ms * spec_k * row_cost
            #   draft = plain_ms * (spec_k * (1 + draft_cost) - 1)
            fit = self._fit_cost()
            if fit is not None:
                return DSPOL.choose_width(probs, fit, cap)
            s_ms = self.plain_ms if self.plain_ms is not None else DSPOL.DEFAULT_S_MS
            slope_ms = s_ms * self.spec_k * self.row_cost
            draft_ms = s_ms * (self.spec_k * (1.0 + self.draft_cost) - 1.0)
            cost = DSPOL.CostModel(s_ms, max(1e-3, slope_ms), max(0.0, draft_ms))
        return DSPOL.choose_width(probs, cost, cap)

    def _spec_update(self, s: Session, drafted: int, accepted: int, conf=None, planned: int | None = None, lookup: bool = False) -> None:
        if drafted <= 0 and not planned:
            if not lookup:
                s.lk_ema = PLD.drift_step(s.lk_ema)                 # no lookup this round: retry later
            return
        if lookup and self._lk_adapt and PLD.SuffixBuckets.bucket(s.lk_suffix) >= 0 and s.lk_adapt is not None:
            s.lk_adapt.observe(s.lk_suffix, accepted)                # a short-suffix round feeds its own bucket, not the long-match EMA
            self.stats["spec_lookup_drafted"] += drafted
            self.stats["spec_lookup_short_drafted"] += drafted
            self.stats["spec_lookup_short_accepted"] += accepted
            self.stats["spec_drafted"] += drafted
            self.stats["spec_accepted"] += accepted
            return
        if lookup:                                                   # a lookup draft has no head forward: its own acceptance EMA, drift on MTP rounds
            s.lk_ema = PLD.ema_step(getattr(s, "lk_ema", 0.0), accepted)
            s.lk_streak = drafted > 0 and accepted >= drafted
            s.lk_full_n = (s.lk_full_n + 1) if s.lk_streak else 0
            self.stats["spec_drafted"] += drafted
            self.stats["spec_accepted"] += accepted
            self.stats["spec_lookup_drafted"] += drafted
            return
        s.mt_ema = PLD.ema_step(getattr(s, "mt_ema", 0.0), accepted)
        s.lk_ema = PLD.drift_step(getattr(s, "lk_ema", 0.0))
        if s.lk_adapt is not None:
            s.lk_adapt.drift()
        if getattr(s, 'probing', False):                                                # a probe that lands resets the gap, one that fails doubles it (a drafter that
            s.probing = False                                        # never lands after a long prompt stops costing a draft every probe_every rounds)
            top = (PROBE_CAP if self.robust else 64) * self.probe_every
            s.probe_gap = self.probe_every if accepted else min(4 * (getattr(s, 'probe_gap', 0) or self.probe_every), top)
        if (planned if planned is not None else drafted) == self.mtp_depth:              # the round was planned at the cap (the confidence gate may have
            s.cap_n += 1                                                                 # shortened what it proposed: the rate policy must still see it)
            w = max(0.1, 1.0 / s.cap_n)                              # the mean of the window, then a ~10-round decay (sd ~0.25 token at 0.85)
            s.cap_tok += w * (accepted + 1 - s.cap_tok)
        if drafted <= 0:
            return
        s.acc_succ = 0.9 * s.acc_succ + accepted
        s.acc_trial = 0.9 * s.acc_trial + accepted + (1 if accepted < drafted else 0)
        reached = min(drafted, accepted + 1)
        while len(s.pos_trial) < reached:
            s.pos_trial.append(0.0); s.pos_succ.append(0.0)
        for i in range(reached):                                   # only the reached positions decay: one not drafted keeps its last value
            s.pos_trial[i] = 0.9 * s.pos_trial[i] + 1
            s.pos_succ[i] = 0.9 * s.pos_succ[i] + (1 if i < accepted else 0)
        if conf is not None and self._dspol_v2:                     # flag dspol_v2: the calibration of P(accept k | kept so far)
            self._v2_observe(s, list(conf), drafted, accepted)
        if conf is not None and self._dspol_on:                     # the confidence head's per-position values train the width policy
            pol = getattr(s, "dspol", None)
            if pol is None:
                pol = s.dspol = DSPOL.Predictor(self._dspol_cal)
            pol.observe(list(conf), drafted, accepted)
        self.stats["spec_drafted"] += drafted
        self.stats["spec_accepted"] += accepted

    def _plain_launch(self, members: list, toks, p0s: list):
        """Launch one plain step of the speculative batched cache (the draft states take its taps); returns the lazy sampled tokens.
        An armed kltap probe (`POST /admin/tap`) reads the logits of this step for one session, like the plain step of `_launch`."""
        samp = [s.sampling for s in members]
        keys = [self._next_key(s) for s in members]
        pick = self.tap if self.tap is not None and len(members) == 1 else (lambda lg: sample_rows(lg, samp, keys))
        if getattr(self.backend, "gated", None) is not None:
            return self.backend.launch_gated(self.bc, toks, pick, [s.draft for s in members], p0s)
        lg = self.backend.step_feed(self.bc, toks, [s.draft for s in members], p0s)
        y = pick(lg)
        mx.async_eval(y)
        return y

    def _plain_tick(self, members: list, p0s: list, rlog=None) -> None:
        """A round without drafts: the plain step of the batched cache, pipelined like the decode loop (the next step is launched from the lazy
        tokens of the one in flight before those are read). p0s = position of the pending token of each session (the step in flight, if any).
        rlog (flag round_log): (mtp_depths, pre, pin, depths) of the plan that sent this round to the plain step, for the trace."""
        t0 = time.perf_counter()
        c0 = time.thread_time()
        _ps = PLS.begin(self.stats) if PLS.on() else None           # probe plain_split: the host's own marks of this tick
        _out0 = [len(s.out) for s in members] if rlog is not None else None
        _rnd0 = [s.rounds for s in members] if rlog is not None else None
        _last0 = [int(s.last) for s in members] if rlog is not None else None    # the pending token of this round (the round's first row)
        warm = self.cur is not None
        with HS.timed(self.stats, "pl"):                          # the plain step's build, same split as the verify build (host_timed)
            if self.cur is None:
                self.cur = self._plain_launch(members, mx.array([s.last for s in members], dtype=mx.int32), p0s)
            if self.plain_ahead and self.nxt is None and not self.ready and self._lookahead_ok():
                self.nxt = self._plain_launch(members, self.cur, [p + 1 for p in p0s])
        t1 = time.perf_counter()
        c1 = time.thread_time()
        if _ps is not None:
            _ps.mark("launch")
        HL.mark("plain_launch")
        leaving: dict = {}
        self._consume(self.cur, leaving, self.nxt is None)
        if _ps is not None:
            _ps.mark("consume")
        HL.mark("plain_consume")
        for s in members:
            if leaving.get(s) in (None, "yield"):
                s.rounds += 1
        if leaving:
            if self.nxt is not None:
                self.cur, self.nxt = self.nxt, None
                self._consume(self.cur, leaving, False)
            if not self._recast(leaving):                        # flag mu_keep: the members that stay keep their rows
                self._dissolve(leaving)
        else:
            self.cur, self.nxt = self.nxt, None
        if warm and len(members) == 1 and not leaving:
            dt = (time.perf_counter() - t0) * 1e3
            self.plain_ms = dt if self.plain_ms is None else 0.9 * self.plain_ms + 0.1 * dt
            self._plain_ring = (self._plain_ring + [dt])[-K_WINDOW:]
            self._rp_win = (self._rp_win + [(1, dt)])[-PAIR_WINDOW:]     # flag rate_pair: one token against this round's own ms
            self._learn_k()
        elif warm and not leaving and self._spec_ladder:                       # a pass of n sessions: its own plain rate for the ladder of n
            dt = (time.perf_counter() - t0) * 1e3
            n = len(members)
            self._plain_n[n] = dt if n not in self._plain_n else 0.9 * self._plain_n[n] + 0.1 * dt
        st = self.stats
        st["spec_rounds"] += 1
        st["spec_rows"] += len(members)
        st["spec_plain"] += 1
        st["spec_t_verify"] += t1 - t0
        st["spec_t_commit"] += time.perf_counter() - t1
        st["spec_d0"] += len(members)
        r = len(members)                                          # the served cost table: whole round ms and count per row count
        st[f"spec_rt_r{r}"] += (time.perf_counter() - t0) * 1e3
        st[f"spec_rn_r{r}"] += 1
        st[f"spec_rdraft_r{r}"] += (t1 - t0) * 1e3                # launch + build of the plain step
        st[f"spec_rpcpu_r{r}"] += (c1 - c0) * 1e3                 # thread CPU of that launch / build
        st[f"spec_rpost_r{r}"] += (time.perf_counter() - t1) * 1e3   # read + accept + commit
        if rlog is not None:                                      # flag round_log: the plain round the policy chose, with the inputs it chose it on
            mtp_depths, pre, pin, depths_ = rlog
            rms = round((time.perf_counter() - t0) * 1e3, 3)
            for i, s in enumerate(members):
                ct, cn, ir, rg, _pr, asuc, atr, _r0r = pre[i]
                prb, pm, cm, sk, pl = pin[i]
                emitted = [int(x) for x in s.out[_out0[i]:]]
                RLOG.record({"k": "p", "sid": s.sid, "ph": self._prompt_tag(s), "m": len(members), "r": _rnd0[i],
                             "pl": pl, "rows": 1 + depths_[i], "lk": 0,
                             **({"src": CV2.SRC_PLAIN, "dlen": 0} if self._copy_v2 else {}),
                             "dec": "probe" if prb else "plain", "draft": 0, "cut": 0, "acc": 0,
                             "tok": len(emitted), "t": emitted, "ms": rms, "ln": _last0[i],
                             "ct": ct, "cn": cn, "ir": ir, "rg": rg, "pr": int(prb),
                             "pm": round(pm, 2) if pm else 0.0, "cm": round(cm, 2),
                             "asuc": asuc, "atr": atr, "sk": round(sk, 3), **self._v2_trace(s)}, self.stats)
        if _ps is not None:                                       # probe plain_split: close the tick (one row into ps_*_r<rows>)
            _ps.mark("end")
            _ps.close(self.stats, r)

    def _skip_cut(self, members: list, rows: dict, depths: list, p0s: list) -> bool:
        """flag spec_skip, after the eager draft: when the gate cut every proposal there is nothing to verify beyond the pending rows, so the round is the
        plain step of the batched cache (the same logits row per session, one key per session) instead of a one-row-per-session verify + commit + settle.
        Each session that drafted rests for skip_gap rounds (1, 2, 4 .. 16; reset by a kept draft). False = the round goes on to the verify."""
        asked = [i for i, d in enumerate(depths) if d > 0]
        if not asked or any(rows.get(i) for i in asked):
            for i in asked:
                if rows.get(i):
                    members[i].skip_gap = 0                          # a kept draft: the head earned its rounds back
            return False
        for i in asked:
            s = members[i]
            s.skip_gap = min(16, max(1, 2 * s.skip_gap))
            s.skip_hold = s.skip_gap
            s.lk_ema = PLD.drift_step(s.lk_ema)
        self.stats["spec_skip_rounds"] += 1
        self._plain_tick(members, p0s)
        return True

    def _gate_threshold(self, s: Session) -> float | None:
        """Raw confidence threshold for the drafter's gate this round (flag spec_gate): the measured P(accept | conf) that pays for one more
        verified row (its ms over a plain step's ms), mapped back to the confidence the head reports. None: the drafter's fixed threshold."""
        plain = self.plain_ms
        if not plain:
            return None
        if self.spec_costs is not None:
            ctx = s.n_prompt + s.n_gen
            slope = self.round_ms(ctx, 1) - self.round_ms(ctx, 0)
        else:
            fit = self._fit_cost()
            if fit is None:
                return None
            slope = fit.slope_ms
        return SPECSAM.gate_confidence(self._dspol_cal, SPECSAM.break_even(slope, plain))

    def _gate_learn(self, s: Session, conf: list, acc: int, paid_rows: int, plan: int, ms: float) -> None:
        """Flag gate_adapt: price every candidate threshold on the round that just ran and keep the best paying one for the next round.
        A candidate `t` would have proposed the leading run of conf >= t (<= the plan's rows, the block's positions), so its accepted
        drafts are min(acc, keep) and its round is the measured one less `slope` for every row it would not have built. No extra round is
        spent exploring: every candidate is priced from the same round. The current threshold is kept unless another beats it by GATE_MARGIN."""
        keep_max = min(len(conf), max(0, int(plan)))
        fit = self._fit_cost()
        slope = max(0.5, fit.slope_ms) if fit is not None else SPEC_ROW_MS
        if len(s.gate_rate) != len(GATE_CANDS):
            s.gate_rate = [0.0] * len(GATE_CANDS)
        for i, t in enumerate(GATE_CANDS):
            keep = 0
            while keep < keep_max and float(conf[keep]) >= t:
                keep += 1
            tok = min(int(acc), keep) + 1
            ms_t = max(1.0, float(ms) + slope * (keep - int(paid_rows)))
            r = tok / ms_t
            s.gate_rate[i] = r if s.gate_rate[i] <= 0.0 else 0.75 * s.gate_rate[i] + 0.25 * r
        cur = min(range(len(GATE_CANDS)), key=lambda i: abs(GATE_CANDS[i] - float(s.gate_thr)))
        best = max(range(len(GATE_CANDS)), key=lambda i: s.gate_rate[i])
        self.stats["spec_gate_adapt_rounds"] += 1
        if s.gate_rate[best] > GATE_MARGIN * s.gate_rate[cur]:
            if GATE_CANDS[best] != GATE_CANDS[cur]:
                self.stats["spec_gate_adapt_up"] += 1
            s.gate_thr = GATE_CANDS[best]

    def _rs_draws(self, lh, members: list):
        """spec_rs: draw each session's block proposals from the draft head's own q (tempered, top-k filtered) instead of its argmax. Returns
        ({session index: (log q rows, drafts, keys)}, the [live jobs, block] token array the verify pass will use, indexed by lh.slot) or None when a
        session is greedy."""
        if any(s.sampling.greedy for s in members) or getattr(lh, "lg", None) is None:
            return None
        block = int(lh.toks.shape[1])
        rows, q, toks = {}, {}, []
        for k, s in enumerate(members):
            if k not in lh.slot:                                    # no draft this round (depth 0): lh.toks / lh.lg hold the live jobs only, in slot order, so no placeholder row
                continue
            lg = lh.lg[lh.slot[k]]
            lq = SPECSAM.logq(lg, s.sampling.temperature, s.sampling.top_k)
            n = 1 + int(lh.depths[k])
            keys = [self._next_key(s) for _ in range(block + n)]
            toks.append(SPECSAM.draw(lq, keys[:block]))
            q[k] = lq[:n - 1]
            rows[k] = (toks[-1], keys[block:])
        return rows, q, mx.stack(toks)

    def _spec_tick(self) -> None:
        """One verify pass for every active session, on the batched cache of the epoch: pending token + drafts of each session are rows of ONE
        pass through the batched core (the code of a plain decode step); each session keeps its accepted rows, the others are dropped."""
        be = self.backend
        if self.ready and self.bc is not None:                       # a joiner: end the epoch (a plain step may be in flight: read it first)
            leaving: dict = {}
            if self.cur is not None:
                self._consume(self.cur, leaving, False)
            if not self._recast(leaving):                            # flag mu_keep: the epoch goes on, the rows that stay are kept
                self._dissolve(leaving)
        if self.ready:
            self.members.extend(self.ready)
            self.ready.clear()
            self.bc = None
        members = self.members
        if not members:
            return
        if self.bc is None:
            self.bc = (getattr(be, "batch_spec", None) or be.batch)([s.cache for s in members])
            self.bc_bytes = sum(s.kv_bytes for s in members)
            for s in members:
                s.cache = None                                       # the batched cache owns the state now
            self.stats["epochs"] += 1
        elif self.kv_budget is not None and self.stats["spec_rounds"] % self.kv_refresh == 0:
            self.bc_bytes = be.nbytes(self.bc)
        _rl = RLOG.enabled()                                          # flag round_log: the per-round policy trace (off: one getenv per round)
        _pre = ([(round(s.cap_tok, 4), s.cap_n, s.idle_rounds, s.rate_gap, s.probing, round(s.acc_succ, 3), round(s.acc_trial, 3), s.rounds)
                 for s in members] if _rl else None)
        if self._spec_ladder and self._round_D is None:
            depths = self._round_depths(members)
        else:
            depths = [self._plan_depth(s) for s in members]
        if self._spec_skip:                                           # flag spec_skip: a session whose head was all cut rests (lookups still apply below)
            for i, s in enumerate(members):
                if s.skip_hold > 0 and depths[i] > 0:
                    s.skip_hold -= 1
                    depths[i] = 0
                    self.stats["spec_skip_held"] += 1
        mtp_depths = list(depths)
        self._orc_round = False
        if self._oracle:                                              # probe oracle_draft: perfect drafts of a recorded continuation
            depths, lookups = self._oracle_plan(members)
        else:
            depths, lookups = self._lookup_plan(members, mtp_depths)
        _pin = ([(s.probing, self.plain_ms, RLOG.median(self._d_ms.get(self.mtp_depth, ())), self.spec_k, mtp_depths[i])
                 for i, s in enumerate(members)] if _rl else None)
        HL.mark("spec_plan")
        t0 = time.perf_counter()
        c0 = time.thread_time()
        p0s = [s.n_prompt + s.n_gen - 1 for s in members]            # position of the pending token
        if not any(depths) and hasattr(be, "step_feed"):             # nobody drafts: the plain decode step (a one-row-per-session pass, no draft, no commit)
            if self._spec_lookup:
                for s in members:
                    s.lk_ema = PLD.drift_step(s.lk_ema)
            self._plain_tick(members, p0s, rlog=(mtp_depths, _pre, _pin, depths) if _rl else None)
            return
        if self.cur is not None:                                     # a draft round follows a plain step in flight: read it first, then re-fit the plan
            leaving = {}
            self._consume(self.cur, leaving, False)
            self.cur = None
            if leaving:
                if not self._recast(leaving):                        # flag mu_keep: the epoch goes on (cur / nxt dropped)
                    self._dissolve(leaving)
                return
            p0s = [s.n_prompt + s.n_gen - 1 for s in members]
            depths, lookups = self._oracle_plan(members) if self._oracle else self._lookup_plan(members, mtp_depths, clamp=True)   # the token of the step in flight moves the tail: re-plan
            if not any(depths) and hasattr(be, "step_feed"):          # the re-plan found no draft: the plain step, never a one-row verify
                if self._spec_lookup:
                    for s in members:
                        s.lk_ema = PLD.drift_step(s.lk_ema)
                self._plain_tick(members, p0s, rlog=(mtp_depths, _pre, _pin, depths) if _rl else None)
                return
            t0 = time.perf_counter()
            c0 = time.thread_time()
        if lookups and not self._orc_round:
            self.stats["spec_lookup_rounds"] += 1
        cv2 = self._copy_v2
        if self.robust and hasattr(be, "flush_drafts"):               # the taps the plain rounds deferred: fed outside the timed round (same ring bits)
            be.flush_drafts([s.draft for i, s in enumerate(members) if not cv2 or i not in lookups])   # copy_v2: a copy session's ring stays lazy (no head forward in its round)
            t0 = time.perf_counter()
            c0 = time.thread_time()
        drf = getattr(be, "drafter", None)
        if drf is not None:                                           # spec_gate: the gate's threshold from the measured P(accept | conf)
            if self._dspol_v2 and len(members) <= 1:                  # dspol_v2: no threshold gate, every drafted row is verified and counted
                drf.gate = -1.0
            elif self._gate_adapt:                                    # gate_adapt: this session's threshold from its own round rewards
                drf.gate = float(getattr(members[0], "gate_thr", 0.5))
                FD.hit("gate_adapt")
            else:
                drf.gate = self._gate_threshold(members[0]) if self._spec_gate else None
            if drf.gate is not None:
                FD.hit("spec_gate")
        setattr(be, "lazy_budget", (self._dwait_ms, self._cut_rows * self._row_ms) if self._lazy_fit else None)
        if self._lazy_trim and not self._orc_round and not self._spec_full and not self._spec_rs:
            depths = self._trim_depths(members, depths, lookups)
        jobs = [(s.draft, s.last, p0, 0 if i in lookups else depths[i], (s.rids, s.out))
                for i, (s, p0) in enumerate(zip(members, p0s))]
        lazy = getattr(be, "draft_lazy", None)
        if self._dspol_v2 >= 2 and len(members) <= 1:                 # dspol_v2 mode 2: the rows are chosen from this round's confidences, so the draft is read first
            lazy = None
        cpl = {}
        if self._spec_couple and getattr(be, "couples", False) and not self._orc_round and not self._spec_rs and not self._spec_full:
            cpl = self._couple_plan(members, jobs, lookups)           # flag spec_couple: the eager draft, with the verify rows' keys mixed into the chain
            if cpl:
                lazy = None
        want_lg = self._spec_rs and not lookups and all(not s.sampling.greedy for s in members)   # a greedy row keeps the argmax proposal (no q to draw from)
        lh = (lazy(jobs, want_logits=True) if want_lg else lazy(jobs)) if lazy is not None else None
        lh_lazy = lh is not None
        rs = None
        res = None
        if lh is not None and want_lg:                               # spec_rs: the proposals are drawn from the draft head's q, not its argmax
            rs = self._rs_draws(lh, members)
            if rs is not None:
                lh.set_toks(rs[2])
        if lh is None:                                               # eager drafts (no lazy pass, or every job has depth 0)
            if cpl:
                lam = self._spec_couple / 100.0
                drafted = be.draft_many(jobs, couple=[(members[i].sampling.temperature * lam, cpl[i]) if i in cpl else None for i in range(len(members))])
                self.stats["spec_couple_rounds"] += 1
            else:
                drafted = be.draft_many(jobs)
            rows = {i: [int(t) for t in dr[:depths[i]]] for i, dr in enumerate(drafted) if i not in lookups}
            if self._dspol_v2 >= 2 and len(members) <= 1 and rows:
                asked = any(depths)
                self._v2_cut(members, rows, depths, lookups)
                if asked and not lookups and hasattr(be, "step_feed") and not any(rows.get(i) for i in range(len(members))):
                    self.stats["dsp2_plain_after_draft"] += 1
                    self._plain_tick(members, p0s, rlog=(mtp_depths, _pre, _pin, depths) if _rl else None)
                    return
            if self._spec_skip and not lookups and hasattr(be, "step_feed"):
                if self._skip_cut(members, rows, depths, p0s):
                    return
            if lookups:
                lk = {i: dr for i, (k, dr) in lookups.items()}
                lh = be.lookup_drafts(lk, depths, base_rows=rows)            # the lookup rows join the MTP rows in one verify pass
            else:
                toks = [[s.last] + rows.get(i, []) for i, s in enumerate(members)]
                lens = [len(tk) for tk in toks]
        if lh is not None:
            if lookups and not getattr(lh, "is_lookup", False):
                lk = {i: dr for i, (k, dr) in lookups.items()}
                lh = be.lookup_drafts(lk, depths, base=lh)
            if lh_lazy:
                with HS.timed(self.stats, "lazy"):                    # the lazy verify is built under the draft: its host time lands in spec_draft
                    res = be.spec_verify_lazy(self.bc, lh, [s.last for s in members])
                lens = [1 + n for n in lh.counts]
        if not lh_lazy:
            if lh is not None:                                       # lookup rows with an eager backend: build the verify rows now
                if not getattr(lh, "is_lookup", False):
                    lk = {i: dr for i, (k, dr) in lookups.items()}
                    lh = be.lookup_drafts(lk, depths, base=lh)
                toks = [[s.last] + dr for s, dr in zip(members, lh.tolist())]
                lens = [len(tk) for tk in toks]
        t1 = time.perf_counter()
        c1 = time.thread_time()
        HL.mark("spec_draft")
        self.stats["spec_lazy_rounds" if lh_lazy else "spec_eager_rounds"] += 1   # which draft path actually ran (lazy = verify built under the draft)
        if not lh_lazy:                                                       # no lazy verify: the eager pass over the built rows
            with HS.timed(self.stats, "bc"):                                  # host wall vs thread CPU of the build + the blocking calls inside it
                res = be.spec_verify_bc(self.bc, toks)
        t1b = time.perf_counter()                                     # the verify graph is built: what follows is the sampling graph
        c1b = time.thread_time()
        samplings, keys = [], []
        for k, (s, n) in enumerate(zip(members, lens)):
            samp = _GREEDY if rs is not None and k in rs[0] else s.sampling    # a spec_rs row is sampled by specsam.decide (no second filtered draw)
            samplings += [samp] * n
            ck = cpl.get(k)
            keys += ck[:n] if ck is not None and n <= len(ck) else [self._next_key(s) for _ in range(n)]   # one key per verify row (None for a greedy session); spec_couple: the keys the draft chain saw
        va = sample_rows(res.logits, samplings, keys)
        rsl = []                                                     # spec_rs: (out, acc) per session, a graph next to the verify rows
        if rs is not None:
            for k, s in enumerate(members):
                if k not in rs[0]:
                    continue
                n = 1 + int(lh.depths[k])
                r0 = res.starts[k]
                rsl.append(SPECSAM.decide(res.logits[r0:r0 + n], rs[1][k], rs[0][k][0][:n - 1], s.sampling, rs[0][k][1]))
        HL.mark("spec_verify_sample")
        tb = time.perf_counter()                                     # host graph build of the round (drafts + verify) ends here, the GPU wait starts
        extra = [a for pair in rsl for a in pair]
        if lh is not None and HS.enabled():                          # KYOJIN_VERIFY_ASYNC: ONE wait for the verify argmax and the drafts, then plain copies
            mx.eval(va, lh.toks, lh.conf, *extra)
            lh.packed = True
            vals = HS.read(va)
            rsv = [HS.read(a) for a in extra]
        else:
            vals = va.tolist()
            rsv = [a.tolist() for a in extra]
        self.stats["t_rbuild"] += tb - t0; self.stats["t_rwait"] += time.perf_counter() - tb; self.stats["n_round"] += 1
        HL.mark("spec_wait")
        if lh is not None:
            toks = [[s.last] + dr for s, dr in zip(members, lh.tolist())]
        keep = getattr(lh, "keep", None)                             # a confidence gate behind the verified depth: rows past it are never proposals
        confs = None
        if self._dspol_on or self._gate_adapt or self._dspol_v2:
            if lh is not None:
                confs = getattr(lh, "confs", None)                   # per-member full-block confidences (None entries for depth-0 jobs)
            else:
                dc = getattr(be, "draft_confs", None)
                confs = dc() if callable(dc) else None
        t2 = time.perf_counter()
        HL.mark("spec_post")
        leaving: dict = {}
        accepted = []
        built: list = []                                              # rows the verify actually built per session (flag spec_ladder: the width to price)
        full = self._spec_full                                        # accept every draft the pass verified (the gate only trims what the eager path proposes)
        n_rs = 0
        cut = 0                                                       # rows the gate cut this round (flag lazy_fit: what the lazy path would pay for)
        for k, (s, tk) in enumerate(zip(members, toks)):
            r0, drafts = res.starts[k], tk[1:]
            raw_n = len(drafts)                                        # proposals before the confidence gate trims them (flag round_log)
            built.append(raw_n)
            if len(drafts) < depths[k]:                               # the draft head proposed fewer rows than the plan (its block is the cap)
                self.stats["spec_block_cut"] += depths[k] - len(drafts)
            if keep is not None and not full and rs is None:
                if keep[k] < len(drafts):                             # rows the confidence gate cut AFTER the verify paid for them
                    self.stats["spec_gate_cut"] += len(drafts) - keep[k]
                    cut += len(drafts) - keep[k]
                drafts = drafts[:keep[k]]
            elif keep is not None and full and rs is None and keep[k] < len(drafts):
                FD.hit("spec_full", len(drafts) - keep[k])
            if rs is not None and k in rs[0]:
                FD.hit("spec_rs")
                out_l, acc_l = rsv[2 * n_rs], rsv[2 * n_rs + 1]
                n_rs += 1
                a = 0
                while a < len(drafts) and acc_l[a]:
                    a += 1
                tail = int(out_l[a])
            else:
                a = 0
                while a < len(drafts) and vals[r0 + a] == drafts[a]:
                    a += 1
                tail = int(vals[r0 + a])
            accepted.append(a)
            if self._lazy_trim and rs is None and k not in lookups and depths[k] > 0 and not self._orc_round:
                s.kept_ema = float(len(drafts)) if s.kept_ema < 0 else s.kept_ema + TRIM_RATE * (len(drafts) - s.kept_ema)
            conf = confs[k] if confs is not None and k < len(confs) else None
            lk = k in lookups and not self._orc_round
            self._spec_update(s, len(drafts), a, conf=None if (rs is not None and k in rs[0]) or self._spec_chain > 1 else conf,
                              planned=depths[k] if full else None, lookup=lk)
            if self._gate_adapt and conf is not None:
                self._gate_learn(s, conf, a, 1 + raw_n, depths[k], (t2 - t0) * 1e3)
            if self._orc_round:
                self.stats["spec_oracle_drafted"] += len(drafts)
                self.stats["spec_oracle_accepted"] += a
                self.stats["spec_oracle_rounds"] += 1
            if lk:
                self.stats["spec_lookup_accepted"] += a
                if cv2:
                    s.cp_scale = CV2.update_scale(s.cp_scale, len(drafts), a)
                    self.stats["spec_copy_rounds"] += 1
                    self.stats["spec_copy_drafted"] += len(drafts)
                    self.stats["spec_copy_accepted"] += a
            reason = None
            emit = drafts[:a] + [tail]
            n_em = 0
            for t in emit:
                n_em += 1
                reason = self._accept(s, int(t))
                if reason:
                    break
            if reason is None and ((s.yield_after is not None and s.seg_gen >= s.yield_after and self.waiting) or s.park_now):
                reason = "yield"
            if reason is None or reason == "yield":
                s.rounds += 1
            if reason:
                leaving[s] = reason
            if _rl:                                                   # flag round_log: the policy's inputs as the plan saw them, then the round's payoff
                ct, cn, ir, rg, _pr, asuc, atr, r0r = _pre[k]
                prb, pm, cm, sk, pl = _pin[k]
                RLOG.record({"k": "r", "sid": s.sid, "ph": self._prompt_tag(s), "m": len(members), "r": r0r,
                             "pl": pl, "rows": 1 + depths[k], "lk": int(lk),
                             **({"src": CV2.SRC_COPY if lk else CV2.SRC_BLOCK, "dlen": len(drafts)} if cv2 else {}),
                             "dec": "probe" if prb else ("cap" if pl > 0 else "plain"),
                             "draft": raw_n, "cut": raw_n - len(drafts), "acc": a, "tok": n_em,
                             "t": [int(x) for x in emit[:n_em]], "ms": round((time.perf_counter() - t0) * 1e3, 3), "ln": int(tk[0]),
                             "ct": ct, "cn": cn, "ir": ir, "rg": rg, "pr": int(prb),
                             "pm": round(pm, 2) if pm else 0.0, "cm": round(cm, 2),
                             "cf": ([round(float(c), 3) for c in confs[k]] if confs is not None and k < len(confs) and confs[k] is not None else None),
                             "gt": round(float(s.gate_thr), 2) if self._gate_adapt else None,
                             "asuc": asuc, "atr": atr, "sk": round(sk, 3), **self._v2_trace(s)}, self.stats)
        HL.mark("spec_accept")
        if cv2 and lookups and getattr(res, "can_defer", False):    # copy_v2: the taps of a copy session's kept rows wait in its lazy queue (no stage forward in this round)
            res.commit(accepted, [s.draft for s in members], p0s, defer=[i in lookups and not self._orc_round for i in range(len(members))])
        else:
            res.commit(accepted, [s.draft for s in members], p0s)
        be.settle(self.bc, [s.draft for s in members])
        if leaving and not self._recast(leaving):                    # flag mu_keep: the round is committed, the epoch goes on
            self._dissolve(leaving)
        t3 = time.perf_counter()
        HL.mark("spec_commit")
        if self._spec_ladder and self._round_D is None and not lookups and mtp_depths and len(set(mtp_depths)) == 1 and mtp_depths[0] > 0:
            nn_ = len(members)
            plain = self.plain_ms if nn_ == 1 else self._plain_n.get(nn_)
            if plain and len(set(built)) == 1 and built[0] == mtp_depths[0]:   # price the width the verify ran, never a plan the head cut short
                self._ladder_for(nn_).observe(mtp_depths[0], (t3 - t0) * 1e3, sum(a + 1 for a in accepted),   # one round = the whole pass
                                              plain / nn_, plain)
        if self._lk_long and len(members) == 1 and 0 in lookups and rs is None and not self._orc_round:
            self._lk_costs.observe(depths[0], (t3 - t0) * 1e3)       # a solo lookup round: the price of that draft count
        if len(members) == 1 and depths[0] > 0 and self.spec_costs is None and not lookups and not self._orc_round:
            self._spec_obs = ((t3 - t0) * 1e3, depths[0], sum(a + 1 for a in accepted))
            self._learn_k()
        st = self.stats
        st["spec_rounds"] += 1
        rows = sum(len(t) for t in toks)
        st["spec_rows"] += rows
        st["spec_t_draft"] += t1 - t0
        st["spec_t_verify"] += t2 - t1
        st["spec_t_commit"] += t3 - t2
        # flag lazy_fit: the two measured sides of the lazy decision. The draft section's off-CPU time is what the verify build can hide; the rows
        # the gate cuts are what the lazy pass pays for (it builds the verify at the planned depth, before the gate can trim it).
        self._dwait_ms += 0.2 * ((t1 - t0 - (c1 - c0)) * 1e3 - self._dwait_ms)
        self._row_ms += 0.2 * ((t2 - t1) * 1e3 / max(rows, 1) - self._row_ms)
        self._cut_rows += 0.2 * (cut - self._cut_rows)
        if self._lazy_fit:
            st["spec_lfit_save_ms"] = round(self._dwait_ms, 3)
            st["spec_lfit_waste_ms"] = round(self._cut_rows * self._row_ms, 3)
        # served cost table: whole round ms, the three sections and the count, per row count (rows = pending + verified drafts, summed over the
        # sessions of the pass). This is what the depth policy needs and what a lane measurement reads back from /v1/stats.
        st[f"spec_rt_r{rows}"] += (t3 - t0) * 1e3
        st[f"spec_rn_r{rows}"] += 1
        st[f"spec_rdraft_r{rows}"] += (t1 - t0) * 1e3             # drafts (lazy launch) + the row lists
        st[f"spec_rdcpu_r{rows}"] += (c1 - c0) * 1e3              # thread CPU of the draft section (wall - cpu = the draft wait)
        st[f"spec_rvbuild_r{rows}"] += (t1b - t1) * 1e3           # THE VERIFY GRAPH BUILD (host): the pass over rows, before any sampling
        st[f"spec_rvcpu_r{rows}"] += (c1b - c1) * 1e3             # thread CPU of that build: wall - cpu = what the build waited for
        st[f"spec_rsample_r{rows}"] += (tb - t1b) * 1e3           # the sampling graph of the same rows
        st[f"spec_rwait_r{rows}"] += (t2 - t1) * 1e3              # sample graph + the one GPU wait (t2 - t1 = the two above + the wait)
        st[f"spec_rpost_r{rows}"] += (t3 - t2) * 1e3              # accept loop + commit + settle
        if lh_lazy:                                               # draft + build of the lazy rounds: the eager share is (rdraft + rvbuild) minus this
            st[f"spec_rhlazy_r{rows}"] += (t1b - t0) * 1e3
            st[f"spec_rnlazy_r{rows}"] += 1
        for dd in depths:
            st[f"spec_d{dd}"] += 1                                   # rounds per planned depth (the policy's choices)
        HL.mark("spec_stats")


def _timed(name):
    """Host-side wall time of one decode launch / read, kept in Engine.stats (t_<name> seconds, n_<name> calls): launch = graph build + submit,
    consume = wait for the GPU. With the pipelined loop, consume time ~ GPU-bound wait; launch time ~ host cost per token."""
    def deco(fn):
        def run(self, *a, **k):
            t = time.perf_counter()
            try:
                return fn(self, *a, **k)
            finally:
                self.stats["t_" + name] += time.perf_counter() - t
                self.stats["n_" + name] += 1
        run.__name__ = fn.__name__
        return run
    return deco


Engine._launch = _timed("launch")(Engine._launch)
Engine._launch_from_last = _timed("launch")(Engine._launch_from_last)
Engine._consume = _timed("consume")(Engine._consume)
Engine._plain_launch = _timed("plaunch")(Engine._plain_launch)
