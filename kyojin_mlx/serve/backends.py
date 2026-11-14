# SPDX-License-Identifier: Apache-2.0
"""Model adapters for the serving engine. A backend owns the model and speaks in per-user caches and batched caches:

    make_cache() / fork(cache)          a user's cache, an independent copy (copy on write)
    prefill_chunk(cache, ids) -> [1, V] fp32 logits of the last position (evaluated, cache state evaluated)
    batch(caches) -> bc                 one cache for a batched step (row i = caches[i])
    unbatch(bc, n) -> caches            inverse of batch; the users go on from their own cache
    step(bc, tokens [N] int32) -> [N, V] fp32 logits (lazy)

Speculative decode (optional, `supports_spec`): every active session contributes its pending token and up to D drafts to ONE verify pass.
    new_draft() / draft_snapshot(d) / draft_restore(snap)   per-session draft state (the drafter's main-kv ring) and its copy for prefix snapshots
    prefill_chunk(cache, ids, draft=d)  also feeds the draft state with the taps of the chunk
    draft(d, pending, p0, depth, history) -> up to `depth` proposed tokens after `pending` (the token at position p0)
    draft_many([(d, pending, p0, depth, history), ...]) -> one list of proposed tokens per session, one drafter pass for all of them
    draft_lazy(jobs, want_logits=False) -> the same, unevaluated (LazyDrafts); want_logits also carries the drafter's biased block logits
        (`LazyDrafts.lg`, KYOJIN_SPEC_RS): a sampled session draws its proposals from that q instead of the argmax
    spec_verify_bc(bc, [[pending, *drafts], ...]) -> pass over the sessions of the batched cache `bc` (session k = row k of the batch):
        .logits [rows, V] (rows = sum over sessions), .starts, .commit(accepted, drafts_states, p0s) keeps accepted[k] drafts of session k
        in bc (and feeds its draft state with the taps of the kept positions). The rows go through the same batched core as plain decode.
    spec_verify([(cache, [pending, *drafts]), ...]) -> the older chained pass over per-session caches (reference, tools)
    replay(cache, tokens, draft=None) -> cache after consuming `tokens` (teacher forced, same bits as the decode steps that made them)

Every decode op is row invariant (kyojin_mlx.multiuser / kyojin_mlx.rows): row i of a batched step equals the solo step of user i bit for bit,
whatever the number of rows and whoever shares the batch. `prefill_chunk` runs the stock ops, so a prompt must always be cut at the same
absolute positions (the engine's chunk grid) for a forked prefix to give the same bits as a full prefill."""
from __future__ import annotations

import os

import numpy as np

import mlx.core as mx

from .. import ds41_vision
from .. import ds41_vision as DV
from .. import hostsync as HS

MAX_PREFILL_CHUNK = 8192                                                    # hard cap of every prefill chunk (see Engine / server.resolve_chunk)
LAZY_DRAFT = os.environ.get("KYOJIN_LAZY_DRAFT", "1") != "0"      # drafts stay on the GPU while the verify graph is built (VlmBackend.draft_lazy)
LAZY_CUT_MAX = float(os.environ.get("KYOJIN_LAZY_CUT_MAX", "0.25"))
LAZY_CUT_RATE = 0.1
GATED_VERIFY = os.environ.get("KYOJIN_GATED_VERIFY", "0") == "1"   # the lazy verify pass through the Engram commit gate (VlmBackend._verify_gated)
VERIFY_HOLD = 64                                                    # commits held for one verify pass: more than it can make (the rest pass through as credit)
# Path counters of the gated verify (flags.py gated_verify, /v1/stats engage["backends"]): a lever that never fires must show as a number, not as
# an absent effect. rounds = passes built through the gate, skipped = a gated verify was armed but the pass could not use it (1 = no calibrated
# gate yet, 2 = verify replay owns the rows), failed = the self-check threw and the lever turned itself off.
STATS = dict(gated_verify_rounds=0, gated_verify_skipped=0, gated_verify_nogate=0, gated_verify_failed=0)


def arrays_of(obj, _seen=None) -> list:
    """Every mx.array reachable from a cache (lists, tuples, dicts, objects)."""
    seen = _seen if _seen is not None else set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        return [obj]
    out = []
    if hasattr(obj, "raw_arrays"):                              # packed fp4 pool slots (fp4pool): the stored codes / scales, never the unpacked rows
        return obj.raw_arrays()
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out += arrays_of(v, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += arrays_of(v, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for v in vars(obj).values():
            out += arrays_of(v, seen)
    return out


class Backend:
    kind = "base"
    max_rows = 16                                   # most users in one batched step
    supports_spec = False                           # speculative verify passes (see the module doc)
    default_mtp_depth = 3                           # most drafts per round when the server picks the depth itself
    max_spec_rows = 0                               # most rows (sessions + drafts) of one verify pass

    def disk_namespace(self) -> dict | None:
        """Everything that changes the cached values besides the tokens (kvdisk.KvDisk namespace), or None: this backend cannot checkpoint to disk."""
        return None

    def make_cache_for(self, need: int):
        """The cache of a new session that will hold up to `need` tokens (prompt + reply): `make_cache` unless the backend picks the KV format by depth."""
        return self.make_cache()

    def adapt_cache(self, cache, need: int):
        """A restored cache (prefix hit) for a session that will hold up to `need` tokens: unchanged unless the backend picks the KV format by depth."""
        return cache

    def tq_cross_at(self, cache):
        """The depth where this session's cache should turn TurboQuant (KYOJIN_TQ_FROM), or None: nothing to convert."""
        return None

    def to_tq(self, cache):
        """The session's plain KV layers converted to TurboQuant (tq_cross_at reached)."""
        return cache

    def gpu_keepalive(self) -> None:
        """A tiny GPU command while idle, so the first real one does not pay the GPU wake-up (see Engine.run)."""
        mx.eval(mx.array(0.0) + 1.0)

    supports_yield = False                          # flag mix_slice_ms: the prefill forward can call a hook between layer groups (set_prefill_yield)

    def set_prefill_yield(self, cb) -> None:
        """Install (or, with None, remove) a callable the prefill forward calls at its layer bounds. Only for a backend with supports_yield."""
        raise NotImplementedError

    def make_cache(self):
        raise NotImplementedError

    def fork(self, cache):
        raise NotImplementedError

    def restore(self, cache):
        """A cache a session continues from (a prefix-snapshot hit, a shared seed): `fork`. A backend whose cache keeps packed layers overrides this with
        `rows.restore`, which gives a layer restored from a stored snapshot a private code buffer with room for its next appends (flag pf_grow, fp4pool)."""
        return self.fork(cache)

    def prefill_chunk(self, cache, ids: list[int]) -> mx.array:
        raise NotImplementedError

    def batch(self, caches: list):
        raise NotImplementedError

    def unbatch(self, bc, n: int) -> list:
        raise NotImplementedError

    def recast(self, bc, drop: list, park: list, add: list) -> dict:
        """Flag mu_keep: update the membership of a batched cache in place instead of dissolving it. `drop` = rows whose session finished (no
        write-back), `park` = rows whose session leaves the batch and keeps its cache, `add` = the per-session caches of the joiners. Returns
        {row index: cache list} for the parked rows. Only the rows that change are read or written."""
        raise NotImplementedError

    def step(self, bc, tokens: mx.array) -> mx.array:
        raise NotImplementedError

    def state(self, cache) -> list:
        return arrays_of(cache)

    def snapshot(self, cache):
        """A prefix-cache snapshot of `cache`; a backend with derived buffers keeps only the stored KV."""
        return self.fork(cache)

    def nbytes(self, cache) -> int:
        return sum(a.nbytes for a in arrays_of(cache)) + self.seg_bytes(cache)

    def seg_bytes(self, cache) -> int:
        """KYOJIN_KVSEG: the sealed segments a live session (or a hit) keeps resident beside its buffer. `arrays_of` lists only the contiguous buffer, so without this a
        session's bytes would drop below what is resident exactly when snapshots share its segments. O(layers), no walk over the segments."""
        n = 0
        for c in cache if isinstance(cache, (list, tuple)) else [cache]:
            for name in ("compress", "keys"):
                slots = getattr(c, name, None)
                if hasattr(slots, "seg_bytes"):
                    n += slots.seg_bytes()
        return n

    def shared_parts(self, cache) -> dict:
        """{id: bytes} of the parts of a snapshot that other snapshots of the same session share by identity (KYOJIN_KVSEG segments, fp4pool); {} for a plain snapshot.
        `nbytes` counts them whole, the prefix cache's ledger counts each once."""
        out = {}
        for c in cache if isinstance(cache, (list, tuple)) else [cache]:
            for name in ("compress", "keys"):
                slots = getattr(c, name, None)
                if hasattr(slots, "seg_parts"):
                    out.update(slots.seg_parts())
        return out

    # -- draft state (backends without a draft head keep None)
    def new_draft(self):
        return None

    def draft_snapshot(self, d):
        return None

    def draft_restore(self, snap):
        return None

    def draft_nbytes(self, d) -> int:
        return 0

    def draft_arrays(self, d) -> list:
        return []

    couples = False                      # True: draft_many takes `couple` (flag spec_couple: per-job (noise scale, keys) of the verify rows)

    def draft_many(self, jobs: list) -> list:
        """jobs: [(draft state, pending token, position of the pending token, depth, history)] -> the proposed tokens of every job."""
        return [list(self.draft(d, pending, p0, depth, hist)) if depth > 0 else [] for d, pending, p0, depth, hist in jobs]

    def lookup_drafts(self, rows: dict, depths: list, base=None, base_rows: dict | None = None):
        """Drafts from the engine's per-session token index (kyojin_mlx.pld): `rows` = {session index: proposed tokens}.
        Same shape as LazyDrafts for spec_verify_lazy; sessions not in `rows` keep their MTP drafts (the lazy `base` or the
        eager `base_rows`)."""
        return LookupDrafts(rows, depths, base=base, base_rows=base_rows)

    def settle(self, bc, drafts=()) -> None:
        """Evaluate the state a speculative pass left in the batched cache and the draft states (nothing to do for an eager backend)."""

    def replay(self, cache, tokens: list[int], draft=None):
        """The cache after `tokens` were fed to it one decode step at a time (a batch of one: row invariant, so the bits are the ones the
        user's own decode steps left). The logits are not used."""
        bc = self.batch([cache])
        for i in range(0, len(tokens), 8):
            lg = None
            for t in tokens[i:i + 8]:
                lg = self.step(bc, mx.array([t], dtype=mx.int32))
            mx.eval(lg, *self.state(bc))                # bound the lazy graph
        return self.unbatch(bc, 1)[0]

    def warm(self, log=print, n: int = 32, steps: int = 6, chunk: int = 2048) -> float:
        """One short prefill, one 2048-row chunk and six decode steps at load: MLX loads the weights lazily (a layer's tensors are read from disk on its first use), so without this the
        first request pays the 172 GiB read (15-19 s) plus the first kernel compiles (one prefill per row-count band: 2, 4, 32, 64, 256, 512, 2048). A file read-ahead does not help: macOS does not keep these reads in its page cache.
        KYOJIN_WARM_LOAD=0 turns it off."""
        import time
        if chunk > MAX_PREFILL_CHUNK:                                    # 16384-row chunks (CED) panicked the Ultra on 29/10 00:26: never warm or run one
            log(f"[kyojin] warm-up chunk {chunk} capped to {MAX_PREFILL_CHUNK}")
            chunk = MAX_PREFILL_CHUNK
        t0 = time.perf_counter()
        cache = self.make_cache()
        self.prefill_chunk(cache, list(range(1000, 1000 + n)))
        for k in (2, 4, 64, 256, 512):                                 # row-count bands with their own kernel variants: first prompts of these sizes paid +0.07-0.7 s
            self.prefill_chunk(cache, list(range(3000, 3000 + k)))
        lg = self.prefill_chunk(cache, [2000 + i % 100000 for i in range(chunk)])
        bc = self.batch([cache])
        tok = mx.array([int(mx.argmax(lg.reshape(-1)).item())])
        for _ in range(steps):
            tok = mx.argmax(self.step(bc, tok).reshape(1, -1), axis=-1)
            mx.eval(tok)
        dt = time.perf_counter() - t0
        log(f"[kyojin] warm-up forward: {dt:.1f} s")
        return dt


class Q35Backend(Backend):
    """Qwen3.5 / Qwen3-Next family on mlx-lm (multiuser.py): per-user KV caches, stacked GDN states."""
    kind = "q35"

    def __init__(self, model):
        from .. import multiuser as MU
        self.MU = MU
        self.model = model
        self.rowlinears = MU.install(model)
        self.mla = MU.install_mla(model)            # MLA attention: per user under a RowsKV, no shared-prefix pool (PoolKV attends K / V, not the latent)
        from .. import depchain
        depchain.install(MU._trunk(model.language_model if "language_model" in model else model).layers)     # test-only slope probe, off unless KYOJIN_TEST_DEP_CHAIN > 0
        self.max_rows = 16                          # EXL3 GEMV: launches of <= 4 rows, up to 16 rows
        self.pack = None                            # the pack directory (load_q35), for the disk checkpoint namespace

    def disk_namespace(self):
        from .. import kvdisk, kvwin, tq
        if self.pack is None:
            return None
        ns = {"arch": "q35", "pack": kvdisk.pack_fingerprint(self.pack), "kvwin": kvwin.config()}     # kvwin: the quantized pool + overlay is part of the bytes
        if tq.config() is not None:
            ns["tq"] = tq.config()                                                                   # (absent when off: the namespace of existing checkpoints is unchanged)
        return ns

    def _cache_model(self):
        """The module whose `make_cache` (or layer list) defines the caches. The model itself."""
        return self.model

    def make_cache(self):
        from .. import kvwin
        return kvwin.make_prompt_cache(self._cache_model())  # KYOJIN_KVWIN unset = mlx-lm make_prompt_cache, unchanged

    def make_cache_for(self, need):
        from .. import kvwin, tq
        if tq.FROM > 0 and need <= tq.FROM and tq.config() is not None:
            return kvwin.make_prompt_cache(self._cache_model(), plain=True)     # KYOJIN_TQ_FROM: short session, plain fp16 KV (TurboQuant pays off from ~FROM tokens on)
        return self.make_cache()

    def adapt_cache(self, cache, need):
        from .. import tq
        from mlx_lm.models.cache import KVCache
        cfg = tq.config()
        if tq.FROM <= 0 or need <= tq.FROM or cfg is None or not any(type(c) is KVCache for c in cache):
            return cache
        ref = self.make_cache()                     # which layers take TurboQuant (full attention, not MLA)
        return [tq.from_kv(c, cfg) if type(c) is KVCache and type(r) is tq.TQKVCache else c for c, r in zip(cache, ref)]

    def tq_cross_at(self, cache):
        from .. import tq
        from mlx_lm.models.cache import KVCache
        if tq.FROM <= 0 or tq.config() is None or not any(type(c) is KVCache for c in cache):
            return None
        if getattr(self, "_tq_layers", None) is None:
            self._tq_layers = tuple(type(r) is tq.TQKVCache for r in self.make_cache())     # which layers take TurboQuant (full attention, not MLA)
        return tq.FROM if any(t and type(c) is KVCache for c, t in zip(cache, self._tq_layers)) else None

    def to_tq(self, cache):
        from .. import tq
        return self.adapt_cache(cache, tq.FROM + 1)

    def fork(self, cache):
        return self.MU.fork_cache(cache)

    def prefill_chunk(self, cache, ids):
        lg = self.MU.logits_last(self.model, mx.array(ids)[None], cache).astype(mx.float32)
        mx.eval(lg, *self.state(cache))
        return lg

    def batch(self, caches):
        return self.MU.batch_caches(caches, 0 if self.mla else None)

    def unbatch(self, bc, n):
        return self.MU.unbatch_caches(bc, n)

    def step(self, bc, tokens):
        return self.MU.logits_last(self.model, tokens[:, None], bc).astype(mx.float32)


class VlmBackend(Backend):
    """DS4.1 and GLM on the patched mlx-vlm models (rows.py). The model's own batch caches: merge / extract."""
    kind = "vlm"

    @property
    def supports_yield(self) -> bool:
        """Flag mix_slice_ms: the prefill forward bounds its layers (bound_layers), and the engine may hook in right after a bound."""
        from .. import bound_layers
        return bound_layers.ENABLED

    def set_prefill_yield(self, cb) -> None:
        from .. import bound_layers
        ahead = getattr(getattr(self.lm, "engram_hash", None), "_kyojin_ahead", None)
        bound_layers.YIELD[0] = ahead.nested(cb) if cb is not None and ahead is not None else cb

    def __init__(self, model, spec, union: bool = False):
        from .. import headlast, rows as R
        self.R = R
        self.HL = headlast
        self.model = model
        self.spec = spec
        self.report = R.enable(model, spec, union=union)
        from .. import compressor_pf                # prefill chunks pool the compressor groups in one step instead of a per-token loop (bit exact)
        self.report["compressor_pf"] = compressor_pf.install(model)
        self.report["compressor_pf_hits"] = compressor_pf.HITS        # live counters: pool > 0 on a served multi-chunk prompt = the fast path ran
        self.lm = model.language_model
        self.drafter = None
        self.pack = None                            # the pack directory (load_vlm), for the disk checkpoint namespace
        self._last_live: list = []                  # job indices of the last draft_many with depth > 0 (draft_confs alignment)
        self._cut_ema = 0.0                         # share of recent draft rounds the confidence gate cut short (draft_lazy backs off when high)
        self._lazy_skip = 0                         # flag lazy_fit: rounds the measured budget sent to the eager path
        self.lazy_budget = None                     # flag lazy_fit: (ms the overlap hides, ms the cut rows cost), set by the engine each round
        self.gated = None                           # engram_mlx.GatedDecode (KYOJIN_ENGRAM_GATE=1): plain decode steps without the Engram host sync

    def launch_gated(self, bc, toks, sample, drafts=None, p0s=None):
        """One plain decode step through the gated Engram pipeline (see engram_mlx.GatedDecode); `sample(logits)` -> lazy next tokens.
        drafts / p0s: the speculative batched cache (step_feed): the taps of the step go to the draft rings."""
        from .. import dsv4_core as DC
        lm = self.model.language_model
        taps = {}
        def forward(t, ids):
            st = DC.stack_rows(bc[0]) if drafts is not None else None
            if st is not None:
                st.taps = {}
            try:
                lg = lm(t[:, None], cache=bc, engram_hashes=ids).logits[:, -1].astype(mx.float32)
            finally:
                if st is not None:
                    taps.update(st.__dict__.pop("taps", {}))
            dep = [a for c in bc if hasattr(c, "stacked_arrays") for a in c.stacked_arrays()]
            return mx.depends(lg, dep) if dep else lg
        y = self.gated.launch(bc, toks, forward, sample)
        if self.gated.failed:                                           # the commit count never settled (one-buffer MLX settings not in effect): serve on the ungated path from the next step
            print(f"[kyojin] Engram gate disabled: {self.gated.failed}", flush=True)
            self.gated.reset(bc)
            self.gated = None
        if drafts is not None:
            for k, d in enumerate(drafts):
                if d is not None:
                    self._defer_feed(d, {l: tp[k:k + 1, 0] for l, tp in taps.items()}, p0s[k])
        return y

    def gated_reset(self, bc):
        if self.gated is not None:
            self.gated.reset(bc)

    # -- CED prefill (kyojin_mlx.ced): on wherever the layer layout supports it
    def enable_ced(self, disabled: bool = False, tail=None, keep: int = 0, log=print, mid: int | None = None):
        """Arm CED (on by default, `disabled` = --no-ced). Call BEFORE enable_mtp: the drafter taps must wrap the CED blocks, not the reverse.
        keep: positions per layer the drafter taps read (enable_mtp: max(128, drafter window)); the decoder half of a CED chunk has `tail` rows, so
        tail >= keep, else the tap of a decoder layer would not cover the drafter ring."""
        from .. import ced
        mid, tail = ced.spec_defaults(self.spec, mid, tail)
        if keep and tail != "exact":
            win = int(getattr(self.lm.config, "sliding_window", 0) or 0)
            tail = max(win if tail is None else int(tail), keep)
        return ced.auto_enable(self.model, self.spec, disabled=disabled, tail=tail, log=log, mid=mid)

    def disk_namespace(self):
        from .. import ced, kvdisk
        if self.pack is None or not kvdisk.ds_cache(self.make_cache()[0]):      # the adapter is proven on the DS4.1 cache (and its fp4-packed subclass)
            return None
        ns = {"arch": self.spec.name, "pack": kvdisk.pack_fingerprint(self.pack), "ced": ced.signature(self.lm)}
        from .. import latent_tq, tq
        if tq.config() is not None and latent_tq.latent_ok(self.lm):
            ns["tq"] = tq.config()                          # (absent when off: the namespace of existing checkpoints is unchanged)
        from .. import ablit
        if (sig := ablit.signature()) is not None:
            ns["ablit"] = sig                               # decensor hook: other layer outputs (absent when off: namespace unchanged)
        if self.drafter is not None:                # the drafter ring is part of the checkpoint
            ns["draft"] = [list(self.drafter.tap_ids), int(self.drafter.window)]
        return ns

    # -- speculative decode: the draft head of the pack, chained verify passes (kyojin_mlx.spec)
    def enable_mtp(self, pack: str, cfg: dict, conf_threshold: float | None = None, max_rows: int = 24) -> dict:
        """Load the pack's draft head and arm the chained verify (generic through ArchSpec.mtp / ArchSpec.rows). max_rows: most rows of one
        pass. The rows stay row invariant up to the S3 prefill switch of the MoE and are cut in launches of <= 4 rows."""
        from .. import dspark as D, spec as S, exl3_linear as EL
        assert self.spec.mtp is not None, f"{self.spec.name}: no draft head described in its ArchSpec (ArchSpec.mtp)"
        self.drafter = D.build_drafter(self.model, pack, self.spec, cfg, conf_threshold=conf_threshold)
        self.R.enable_draft(self.drafter.lm, self.spec)
        self.S = S
        rep = S.install(self.model, self.spec, tap_layers=set(self.drafter.tap_ids), keep=max(self.spec.mtp.keep, self.drafter.window))
        EL.ROW_SPLIT_MAX = max(EL.ROW_SPLIT_MAX, max_rows)
        self.max_spec_rows = max_rows
        self.supports_spec = True
        rep["replay_warm_calls"] = self.drafter.warm_replay()                  # KYOJIN_REPLAY=1: the draft regions are recorded now, not on a request
        if os.environ.get("KYOJIN_WARM_LOAD", "1") != "0":
            rep["spec_warm_s"] = round(self.warm_spec(), 1)
        return rep

    def warm_spec(self, rows: int = 8, depth: int = 5) -> float:
        """Verify passes of 2..rows rows, their draft feeds and one draft chain at load. Each row count has its own GEMV / MMA kernel variants
        (MR = rows), so without this the first spec rounds of the first request compiled them: 20+ first-seen kernels between tokens 1 and 256,
        0.1-0.6 s stalls (Ultra kernel probe, 30/10). The cache and the draft state are thrown away."""
        import time
        t0 = time.perf_counter()
        cache, d = self.make_cache(), self.new_draft()
        self.prefill_chunk(cache, list(range(1000, 1064)), draft=d)
        p = 64
        for k in range(2, rows + 1):
            cache = self.replay(cache, list(range(4000, 4000 + k)), d)
            p += k
        self.draft(d, 4100, p, depth)
        return time.perf_counter() - t0

    def new_draft(self):
        return self.drafter.new_ctx() if self.drafter is not None else None

    def draft_snapshot(self, d):
        if d is not None:
            self._flush_lazy([d])
        if d is None:
            return None
        if self.R.PREFIX_COMPACT:                                   # flag prefix_compact: the ring is a slice of a (window + chunk) concat; copy it so the parent is not pinned
            from ..fp4pool import own
            rows = [own(a) for a in d.rows]
            mx.eval(*[a for a in rows if a is not None])
            return rows
        return list(d.rows)

    def draft_restore(self, snap):
        d = self.drafter.new_ctx()
        d.rows = list(snap)
        return d

    def draft_nbytes(self, d) -> int:
        return sum(a.nbytes for a in self._ring(d))

    def draft_arrays(self, d) -> list:
        return self._ring(d)

    def draft(self, d, pending: int, p0: int, depth: int, history=None) -> list[int]:
        return self.draft_many([(d, pending, p0, depth, history)])[0]

    couples = True

    def draft_many(self, jobs: list, want_logits: bool = False, couple: list | None = None) -> list:
        """One drafter pass for every job with depth > 0 (rows = sessions x block); a session's proposal does not depend on the others.
        want_logits: also keep the biased block logits (KYOJIN_SPEC_RS). With flag spec_chain > 1 and a deeper plan, `drafter.draft_many_chain`
        runs that many block forwards per session (a pass after the first only when the one before it kept every row it asked for)."""
        live = [j for j in jobs if j[3] > 0]
        self._flush_lazy([j[0] for j in (live if COPY_V2() else jobs)])      # copy_v2: a session that does not draft keeps its lazy taps (no stage forward)
        chain = int(os.environ.get("KYOJIN_SPEC_CHAIN", "1") or 1)
        cpl = [couple[k] for k, j in enumerate(jobs) if j[3] > 0] if couple and chain <= 1 and not want_logits else None   # flag spec_couple: (scale, keys) per live job
        if chain > 1 and not want_logits:
            got = iter(self.drafter.draft_many_chain([j[0] for j in live], [j[1] for j in live], [j[2] for j in live],
                                                     [j[3] for j in live], chain)) if live else iter(())
        else:
            got = iter(self.drafter.draft_many([j[0] for j in live], [j[1] for j in live], [j[2] for j in live], [j[3] for j in live],
                                               want_logits=want_logits, **({"coupling": cpl} if cpl is not None else {}))) if live else iter(())
        out = [next(got) if j[3] > 0 else [] for j in jobs]
        if live:
            self._note_cut(any(len(o) < j[3] for o, j in zip(out, jobs) if j[3] > 0))
        self._last_live = [k for k, j in enumerate(jobs) if j[3] > 0]      # draft_confs aligns last_raw to jobs
        return out

    def draft_confs(self) -> list | None:
        """Per-job full-block confidences of the last draft_many (None entries for depth-0 jobs); None when unavailable.
        The engine's width policy (kyojin_mlx.dspol) trains on these; they never touch the accepted tokens."""
        raw, live = getattr(self.drafter, "last_raw", None), getattr(self, "_last_live", None)
        if not raw or live is None:
            return None
        try:
            conf = [list(c) for c in raw[1]]
            if len(conf) != len(live):
                return None
            out: list = [None] * (max(live) + 1 if live else 0)
            for k, c in zip(live, conf):
                out[k] = c
            return out
        except Exception:
            return None

    def _note_cut(self, cut: bool) -> None:
        self._cut_ema += LAZY_CUT_RATE * (float(cut) - self._cut_ema)

    def draft_lazy(self, jobs: list, want_logits: bool = False):
        """draft_many without the wait: the proposals stay on the GPU (async_eval started) and spec_verify_lazy builds the verify graph at the
        planned depth while the GPU drafts. None (the caller runs draft_many) when the pass cannot be a fixed-shape launch, or when the
        confidence gate cut recent rounds short so often that the extra verify rows would cost more than the overlap saves.
        want_logits: the draft head's biased block logits come along (KYOJIN_SPEC_RS).
        `lazy_budget` (engine, flag lazy_fit): (ms of the draft wait the build hides, ms of the rows the gate cuts) measured over the last rounds;
        when it is set it replaces the cut-rate threshold."""
        live = [i for i, j in enumerate(jobs) if j[3] > 0]
        if not LAZY_DRAFT or not live:
            return None
        if int(os.environ.get("KYOJIN_SPEC_CHAIN", "1") or 1) > 1:     # a chained draft needs the gate's answer before pass 2: eager
            return None
        budget = getattr(self, "lazy_budget", None)
        if budget is not None and not want_logits:
            if budget[0] <= budget[1]:                                # the overlap saves less than the wasted rows cost: eager
                self._lazy_skip += 1
                return None
        elif self._cut_ema > LAZY_CUT_MAX and not want_logits and os.environ.get("KYOJIN_SPEC_FULL", "0") == "0":  # no gate cut under spec_rs / spec_full: every
            return None                                                                                           # verified row counts, so the cut rate says nothing
        self._flush_lazy([jobs[i][0] for i in live] if COPY_V2() else [j[0] for j in jobs])
        got = self.drafter.draft_many_lazy([jobs[i][0] for i in live], [jobs[i][1] for i in live], [jobs[i][2] for i in live],
                                           [jobs[i][3] for i in live], want_logits=want_logits)
        return None if got is None else LazyDrafts(self, jobs, live, *got)

    def spec_verify_lazy(self, bc, lh, pendings):
        """The chained verify of every session at its planned depth (pending token + the full block prefix, still on the GPU)."""
        self.gated_reset(bc)                                            # the gated plain step left the Engram history one position ahead: put it back, else
        xs = []                                                         # the next gated launch restores the pre-verify history and drops the committed rows
        for i, p in enumerate(pendings):
            pend = mx.array([p], dtype=mx.int32)
            xs.append(mx.concatenate([pend, lh.toks[lh.slot[i], :lh.depths[i]].astype(mx.int32)]) if i in lh.slot else pend)
        g = self.gated
        if GATED_VERIFY:
            if g is None or g.stage.cg_k is None:
                STATS["gated_verify_nogate"] += 1                   # the plain steps have not calibrated the gate (yet): nothing held, nothing gained
            elif _verify_replay_on():
                STATS["gated_verify_skipped"] += 1                  # verify replay reads the Engram rows itself
            else:
                return self._verify_gated(bc, xs)
        return BatchedPass(self, self.R.verify(self.model, bc, xs))

    def _verify_gated(self, bc, xs):
        """The lazy verify without its Engram host syncs (KYOJIN_GATED_VERIFY=1, needs the calibrated gate of the plain steps). Ungated, the build
        stops at the first Engram layer until the drafts are done (the rows are read from their hash ids), and with one command buffer per eval the
        whole pass is then encoded with the GPU idle. Here the pass is built with the rows registered, encoded while the drafter still runs (its
        commits held), the rows are read once the hash ids are out, then the commits are released: the bits of the ungated pass.
        Needs the commits of MLX to be issued by the thread that calls (async_)eval (one buffer per eval, Metal): the hash ids are submitted before
        the gate is armed, so the wait for them never waits for a held buffer."""
        from ..engram_mlx import _commit_gate
        st = self.gated.stage
        st.pending = []
        ps = self.R.verify(self.model, bc, xs, stage=st)
        if ps.eh is None or not st.pending:                             # no Engram row in this pass: nothing to hold
            st.pending = []
            return BatchedPass(self, ps)
        cg = _commit_gate()
        mx.async_eval(ps.eh)                                           # committed now, behind the drafts: never one of the held buffers
        cg.cg_hold(VERIFY_HOLD)
        try:
            mx.async_eval(ps.logits, *ps.taps.values())
            st.fill(np.asarray(ps.eh))
        except BaseException:
            st.pending = []
            ps.entry._counts = None                                     # the pass is never committed: drop its pending state, as rows.verify does on a failed build
            ps.entry.pend = {}
            cg.cg_release()
            print("[kyojin] gated verify failed: the next verify passes are ungated", flush=True)
            global GATED_VERIFY
            GATED_VERIFY = False
            STATS["gated_verify_failed"] += 1
            raise
        cg.cg_release()
        st.resync()                                                     # the gate of the plain steps counts from here
        self.stats_gated_verify = getattr(self, "stats_gated_verify", 0) + 1
        STATS["gated_verify_rounds"] += 1
        if self.stats_gated_verify == 1:
            print("[kyojin] gated verify: on (first pass held and released)", flush=True)
        return BatchedPass(self, ps)

    @staticmethod
    def _ring(d) -> list:
        return [] if d is None else [a for a in d.rows if a is not None]

    def _defer_feed(self, d, taps: dict, start: int) -> None:
        """Plain decode step: keep the taps of this position (one row per tap layer) until a draft is planned; only the last `window` rows can matter."""
        q = getattr(d, "lazy", None)
        if q is None:
            q = d.lazy = []
        q.append((start, [taps[l] for l in self.drafter.tap_ids]))
        if len(q) > self.drafter.window:
            q.pop(0)
        mx.async_eval(*q[-1][1])

    def _defer_feed_rows(self, d, taps: dict, start: int) -> None:
        """Verify commit (flag copy_v2): the taps of n kept rows (positions `start` ..) wait in the lazy queue like those of plain steps, so a copy
        round pays no ring feed; the next round that drafts for this session feeds them in one pass (same ring bits as the eager feed: row-wise
        products, the ring keeps the last `window` rows). The queue holds at most `window` ROWS (the feed projects every queued row before it
        keeps the last window), always the newest ones and always contiguous: older rows are cut from the front, the first entry's start moves with them."""
        rows = [taps[l] for l in self.drafter.tap_ids]
        n, w = int(rows[0].shape[0]), int(self.drafter.window)
        if n >= w:
            rows, start = [r[n - w:] for r in rows], start + n - w
            d.lazy = q = []
        else:
            q = getattr(d, "lazy", None)
            if q is None:
                q = d.lazy = []
        q.append((start, rows))
        over = sum(int(e[1][0].shape[0]) for e in q) - w
        while over > 0:
            st0, r0 = q[0]
            m = int(r0[0].shape[0])
            if m <= over:
                q.pop(0)
                over -= m
            else:
                q[0] = (st0 + over, [r[over:] for r in r0])
                over = 0
        mx.async_eval(*q[-1][1])

    def _flush_lazy(self, ds) -> None:
        """Feed the deferred rows of these draft states to their rings (one batched pass; the rows are row-wise products, so the ring bits equal
        the ones of a per-step feed)."""
        items = []
        for d in ds:
            q = getattr(d, "lazy", None) if d is not None else None
            if q:
                items.append((d, [mx.concatenate([r[1][t] for r in q], axis=0) for t in range(len(q[0][1]))], q[0][0]))
                d.lazy = []
        if items:
            self.drafter.append_many(items)

    def flush_drafts(self, ds) -> None:
        """Feed and evaluate the deferred tap rows of these draft states now (the engine's robust MTP policy times the drafted round without them)."""
        self._flush_lazy(ds)
        mx.eval(*[a for d in ds for a in self._ring(d)])     # stays blocking: the engine resets t0 right after, the policy timing must not absorb GPU work

    def _feed(self, d, taps: dict, start: int) -> None:
        """Prompt chunk: the taps of positions start .. (the stock drafter code, any number of rows)."""
        self._flush_lazy([d])
        self.drafter.ctx = d
        self.drafter.append([taps[l] for l in self.drafter.tap_ids], start)

    def _feed_many(self, items: list) -> None:
        """Decode: [(draft state, taps, start)] of several sessions in one row-wise pass (a row's kv does not depend on the other rows)."""
        ids = self.drafter.tap_ids
        self._flush_lazy([it[0] for it in items])
        self.drafter.append_many([(d, [taps[l] for l in ids], start) for d, taps, start in items])

    def spec_verify(self, jobs: list):
        lg, taps, chain = self.S.verify_many(self.model, [c for c, _ in jobs], [t for _, t in jobs])
        return SpecPass(self, lg, taps, chain)

    def spec_verify_bc(self, bc, token_lists: list):
        self.gated_reset(bc)
        return BatchedPass(self, self.R.verify(self.model, bc, token_lists))

    def settle(self, bc, drafts=()) -> None:
        """Evaluate the state a pass left in the batched cache and the draft rings (bounds the lazy graph)."""
        arrs = [*[a for c in bc if hasattr(c, "stacked_arrays") for a in c.stacked_arrays()], *[a for d in drafts for a in self._ring(d)]]
        (mx.async_eval if HS.enabled() else mx.eval)(*arrs)         # async: the next round's graph is built while the GPU commits (one stream: same bits)

    def replay(self, cache, tokens, draft=None):
        """The cache after `tokens` were consumed: groups of up to 8 tokens go through the batched core as rows of one session (each row
        equals the plain decode step of its position, so the bits are the ones the user's own decode steps left)."""
        if not self.supports_spec:
            return super().replay(cache, tokens, draft)
        bc = self.batch([cache])
        p = bc[0].rows[0].offset
        pipe = os.environ.get("KYOJIN_REPLAY_PIPE", "1") not in ("0", "")      # flag replay_pipe: the next group is built while the GPU runs this one
        prev = None
        for i in range(0, len(tokens), 8):
            grp = [int(t) for t in tokens[i:i + 8]]
            ps = self.R.verify(self.model, bc, [grp])
            ps.commit([len(grp) - 1])
            if draft is not None:
                self._feed_many([(draft, {l: t[:, 0] for l, t in ps.taps.items()}, p)])
            p += len(grp)
            arrs = (ps.logits, *bc[0].stacked_arrays(), *self._ring(draft))
            if not pipe:
                mx.eval(*arrs)
                continue
            mx.async_eval(*arrs)                         # same ops on the same stream, only the host wait moves: the bits are the ones of the synchronous loop
            if prev is not None:
                mx.eval(*prev)                           # at most one group in flight behind the one being built (bounds the lazy graph)
            prev = arrs
        if prev is not None:
            mx.eval(*prev)
        return self.unbatch(bc, 1)[0]

    def make_cache(self):
        cache = self.lm.make_cache()
        from .. import fp4pool, latent_tq, tq
        cfg = tq.config()
        if cfg is not None:
            cache = fp4pool.wrap_cache(cache)               # DS4.1: the compressed pool is stored as fp4 codes (lossless; KYOJIN_FP4_POOL=0 / --kv-mode fp16 opt out)                                   # TurboQuant is the default KV cache (--kv-mode fp16 / KYOJIN_TQ=0 opts out)
        if cfg is not None and latent_tq.latent_ok(self.lm):
            latent_tq.wrap_latent_layers(cache, cfg)
        return cache

    def fork(self, cache):
        return self.R.fork(cache)

    def restore(self, cache):
        return self.R.restore(cache)

    def snapshot(self, cache):
        return self.R.snapshot(cache)

    def warm_prompt(self, ids, a: int, b: int) -> None:
        """Hint: prompt positions [a, b) are the next prefill chunk (Engram rows into the page cache while this chunk runs)."""
        ah = getattr(self, "engram_ahead", None)
        if ah is not None:
            ah.warm(ids, a, b)

    def warm_first_chunk(self, ids, a: int, b: int) -> bool:
        """Flag pf_warm0: the first chunk's Engram rows into the page cache at admission (False when the model has no Engram tables)."""
        ah = getattr(self, "engram_ahead", None)
        return bool(ah is not None and ah.warm_first(ids, a, b))

    def adopt_first_chunk(self, sid, ids, a: int, b: int) -> bool:
        """Flag engram_adopt: the first chunk's Engram row jobs submitted at admission (False when the model has no Engram tables)."""
        ah = getattr(self, "engram_ahead", None)
        return bool(ah is not None and ah.adopt_first(sid, ids, a, b))

    def adopt_select(self, sid, a: int, b: int) -> None:
        ah = getattr(self, "engram_ahead", None)
        if ah is not None:
            ah.adopt_select(sid, a, b)

    def adopt_drop(self, sid) -> None:
        ah = getattr(self, "engram_ahead", None)
        if ah is not None:
            ah.adopt_drop(sid)

    def ced_can_skip(self, cache, next_rows: int) -> int:
        """ced_skip_dec: 0 = the chunk before a chunk of `next_rows` rows runs whole, else the flag's mode (1 = only where the engine takes no snapshot at that boundary,
        2 = also where it would, the engine then takes none). Non-zero only when that next chunk is itself a CED chunk: its decoder half restarts at its own tail, so
        it never reads the window rings of the decoder layers (nor, with a drafter, does the ring need this chunk's taps: the next chunk has more than `tail` >= `keep`
        rows and rewrites it whole)."""
        from .. import ced
        return ced.SKIP_DEC if ced.SKIP_DEC and ced.rewrites_decoder(self.lm, cache, next_rows) else 0

    def ced_rewrites(self, rows: int) -> bool:
        """ced_skip_dec=3: whether a prompt chunk of `rows` rows is a CED chunk of more than `tail` rows (rewrites the decoder rings and the drafter ring whole)."""
        from .. import ced
        return ced.rewrites_rows(self.lm, rows)

    def _forward_kw(self, ids, vl):
        """The model's inputs for a prompt chunk. With an image in it: its rows overwrite the token embeddings (the ids stay: image_mask, Engram)."""
        toks = mx.array(ids)[None]
        if vl is None or not ds41_vision.touches(vl[0], vl[1], vl[1] + len(ids)):
            return {"input_ids": toks}
        return {"input_ids": toks, "inputs_embeds": ds41_vision.overlay(self.lm.embed_tokens(toks), vl[0], vl[1])}

    def prefill_chunk(self, cache, ids, draft=None, skip_dec=False, vl=None):
        if skip_dec:                                                  # ced_skip_dec: encoder half + the midpoint's KV and index keys only; no logits, no taps, no draft feed
            from .. import ced
            with self.HL.last_row(), ced.skip_decoder(self.lm):
                self.lm(**self._forward_kw(ids, vl), cache=cache)
            mx.eval(*self.state(cache))
            self._free_handoff(cache)
            return None
        if draft is not None:                                         # the taps of this chunk go to the drafter's ring
            cache[0].taps = {}
            p = cache[0].offset
        with self.HL.last_row():                                      # the engine reads the last row only
            lg = self.lm(**self._forward_kw(ids, vl), cache=cache).logits[:, -1].astype(mx.float32)
        if draft is not None:
            taps = cache[0].taps
            del cache[0].taps
            n = taps[next(iter(taps))].shape[1]                       # the last <= keep positions of the chunk
            self._feed(draft, {l: t[0] for l, t in taps.items()}, p + len(ids) - n)
        mx.eval(lg, *self.state(cache), *self._ring(draft))
        self._free_handoff(cache)
        return lg

    def cut_merge_ok(self, cache, rows: int) -> bool:
        """Flag pf_cut_merge: whether a prompt chunk of `rows` rows on `cache` can run as one pass with the state at a cut captured inside it (kyojin_mlx.cutmerge).
        Mode 1: not when the chunk would take the CED path (its midpoint calls the attention outside the wrapper); mode 2 runs those chunks exact."""
        from .. import archs as _A, ced, cutmerge as CM, latent_tq, tq
        m = CM.mode()
        tq_latent = tq.config() is not None and latent_tq.latent_ok(self.lm)    # KYOJIN_TQ on DS4.1 only packs the compressed pool (fp4pool, covered); latent TQ caches are not
        if not m or not cache or not _A.is_model_cache(cache[0]) or tq_latent or not hasattr(cache[0], "HANDOFF"):
            return False
        if m >= 3:
            st = ced.state(self.lm)
            return st is None or not st.force                  # mode 3 keeps the CED path (decoder half over tail + the rows past the cut); `force` makes n == tail a CED pass: split
        return m >= 2 or not ced.rewrites_rows(self.lm, rows)

    def _tap_keep(self) -> int:
        for l in getattr(self.drafter, "tap_ids", ()):
            return int(getattr(self.lm.layers[l], "_kyojin_tap_keep", 128))
        return 128

    def prefill_chunk_cut(self, cache, ids, k, draft=None):
        """Flag pf_cut_merge: `prefill_chunk` of ids, one pass, that also returns the cache and the drafter ring as the split path holds them after the first k rows:
        (logits, cache at the cut, draft at the cut). The live cache and draft end as after `prefill_chunk(cache, ids)`."""
        import contextlib

        from .. import ced, cutmerge as CM
        CM.install(self.lm)
        ent = cache[0]
        p, n = int(ent.offset), len(ids)
        dsnap = self.draft_snapshot(draft) if draft is not None else None
        ent._cutm = CM.Cut(k, n)
        if draft is not None:
            ent.taps = {}
            keep = self._tap_keep()
            ent.taps_keep = keep + CM.TAIL_MAX
        kw, exact = {}, contextlib.nullcontext()
        st = ced.state(self.lm)
        if CM.mode() >= 3 and st is not None and st.on:
            # CED kept: the decoder half runs the split path's tail before the cut plus the rows after it; a pass with no more rows than that is no CED pass (as the split path's two chunks)
            kw["ced_tail"] = st.tail + (n - k)
            if n > kw["ced_tail"]:
                ent._cutm.dec_n, ent._cutm.dec_k = kw["ced_tail"], st.tail
        elif ced.rewrites_rows(self.lm, n):
            exact = ced.exact(self.lm)
        try:
            with self.HL.last_row(), exact:
                lg = self.lm(**self._forward_kw(ids, None), cache=cache, **kw).logits[:, -1].astype(mx.float32)
        finally:
            cm = ent.__dict__.pop("_cutm", None) if hasattr(ent, "__dict__") else None
            ent.__dict__.pop("taps_keep", None)
        taps = None
        if draft is not None:
            taps = ent.taps
            del ent.taps
        cut = CM.build(cache, cm, p, self.fork)
        cdraft = None
        if taps is not None:
            ms = {l: t.shape[1] for l, t in taps.items()}             # rows per tap layer, ending at p + n (a CED pass: the decoder layers hold fewer)
            mk = min(min(ms.values()), keep)
            self._feed(draft, {l: t[0][-mk:] for l, t in taps.items()}, p + n - mk)
            mc = min(min(ms.values()) - (n - k), keep)                # the taps of the rows before the cut
            cdraft = self.draft_restore(dsnap)
            if mc > 0:
                self._feed(cdraft, {l: t[0][ms[l] - (n - k) - mc:ms[l] - (n - k)] for l, t in taps.items()}, k + p - mc)
        mx.eval(lg, *self.state(cache), *self.state(cut), *self._ring(draft), *self._ring(cdraft))
        self._free_handoff(cache)
        return lg, cut, cdraft

    def _free_handoff(self, cache) -> None:
        """Flag kv_handoff_free (default 0): the cross-layer handoff of the pass is dead once the chunk is evaluated (kvledger.free_handoff)."""
        from .. import kvledger
        if kvledger.enabled():
            kvledger.free_handoff(cache)

    def batch(self, caches):
        return self.R.batch(caches)

    def unbatch(self, bc, n):
        self.gated_reset(bc)
        users = [[] for _ in range(n)]
        for c in bc:
            for i in range(n):
                users[i].append(c.extract(i))
        return users

    def recast(self, bc, drop, park, add):
        """Flag mu_keep: rows `drop` leave without a write-back (their sessions finished), rows `park` leave with their own caches, `add` join.
        The rows of the sessions that stay keep their slot: no member's state is copied or re-packed (dsv4_core.RowsCache.keep_only / adopt)."""
        c = bc[0]
        got = c.release_rows(park) if park else {}
        gone = set(drop) | set(park)
        c.keep_only([i for i in range(len(c.rows)) if i not in gone])
        for cache in add:
            c.adopt(cache[0])
        return {i: [v] for i, v in got.items()}

    def step(self, bc, tokens):
        self.gated_reset(bc)
        return self.R.step(self.model, bc, tokens)

    def step_feed(self, bc, tokens, drafts, p0s):
        """A plain decode step of the speculative batched cache (every session has no draft this round): the logits of `step`, and the taps of the
        step go to the draft rings (p0s[k] = position of the token of session k), as a verify pass of one row per session would leave them."""
        self.gated_reset(bc)
        from .. import dsv4_core as DC
        st = DC.stack_rows(bc[0])
        st.taps = {}
        try:
            lg = self.R.step(self.model, bc, tokens)
        finally:
            taps = st.__dict__.pop("taps", {})
        for k, d in enumerate(drafts):
            if d is not None:
                self._defer_feed(d, {l: t[k:k + 1, 0] for l, t in taps.items()}, p0s[k])
        return lg


def COPY_V2() -> bool:
    """Flag copy_v2 (read per call): copy sessions keep their draft ring lazy."""
    return os.environ.get("KYOJIN_COPY_V2", "0") not in ("0", "")


class LookupDrafts:
    """Prompt-lookup drafts of one MTP round (kyojin_mlx.pld), with the same interface as LazyDrafts so that
    VlmBackend.spec_verify_lazy verifies them in the same pass as the sessions' MTP drafts. `rows[i]` are the proposed
    tokens of session i from its own committed-token index; sessions not in `rows` take `base` (a LazyDrafts) or
    `base_rows` (eager draft_many results). No draft-head forward was paid for a lookup row."""

    is_lookup = True

    def __init__(self, rows: dict, depths: list, base=None, base_rows: dict | None = None):
        self.depths = [int(d) for d in depths]
        self._rows = {int(i): [int(t) for t in r] for i, r in rows.items()}
        self._base, self._base_rows = base, base_rows
        self.width = max(1, max(self.depths) if self.depths else 1)
        self.slot = {i: i for i, d in enumerate(self.depths) if d > 0}
        self.counts = list(self.depths)
        self.keep = None
        self.confs = [None] * len(self.depths)
        self.rs = False
        self.packed = False
        self.toks = self._build_toks()
        self.conf = mx.zeros((len(self.depths), self.width), dtype=mx.float32)

    def _zero(self):
        return mx.zeros((self.width,), dtype=mx.int32)

    def _pad(self, row, d: int):
        if d <= 0:
            return self._zero()
        if d < self.width:
            return mx.concatenate([row[:d], mx.zeros((self.width - d,), dtype=mx.int32)])
        return row[:d]

    def _build_toks(self):
        out = []
        for i, d in enumerate(self.depths):
            if i in self._rows:
                r = np.zeros((self.width,), dtype=np.int32)
                vals = self._rows[i][:d]
                r[:len(vals)] = vals
                out.append(mx.array(r))
            elif self._base is not None and i in self._base.slot:
                out.append(self._pad(self._base.toks[self._base.slot[i]], d))
            elif self._base_rows is not None:
                r = np.zeros((self.width,), dtype=np.int32)
                vals = self._base_rows.get(i, [])[:d]
                r[:len(vals)] = vals
                out.append(mx.array(r))
            else:
                out.append(self._zero())
        return mx.stack(out) if out else mx.zeros((0, self.width), dtype=mx.int32)

    def tolist(self):
        base_list = self._base.tolist() if self._base is not None else None
        base_keep = getattr(self._base, "keep", None)
        out = []
        for i, d in enumerate(self.depths):
            if i in self._rows:
                out.append(list(self._rows[i][:d]))
            elif base_list is not None:
                out.append(list(base_list[i][:d]) if i < len(base_list) else [])
            else:
                out.append(list(self._base_rows.get(i, [])[:d]) if self._base_rows is not None else [])
        self.keep = [d if i in self._rows else (int(base_keep[i]) if base_keep and i < len(base_keep) else d)
                     for i, d in enumerate(self.depths)]
        self.confs = [None] * len(self.depths)
        return out


class LazyDrafts:
    """Proposals of a VlmBackend.draft_lazy call. `toks` [S, block] / `conf` stay on the GPU until `tolist`; `depths[i]` rows of job i are verified
    (its planned depth, 0 = no draft), `slot[i]` its row in `toks`, `keep[i]` (after `tolist`) the proposals the confidence gate keeps: rows past
    it are verified too but never accepted, so the accepted tokens and counts are those of a verify cut at `keep`. `lg` [S, block, V]: the biased
    logits of the drafted rows when draft_lazy was asked for them (KYOJIN_SPEC_RS); `rs` = the proposals were drawn from q (no gate)."""

    def __init__(self, be, jobs, live, toks, conf, lg=None):
        self.be, self.toks, self.conf, self.lg = be, toks, conf, lg
        self.depths = [j[3] for j in jobs]
        self.slot = {i: k for k, i in enumerate(live)}
        self.counts = list(self.depths)
        self.keep = None
        self.rs = False
        self.packed = False                                          # True once toks / conf were evaluated together with the verify argmax

    def set_toks(self, toks) -> None:
        """Replace the proposals by fresh draws from q (spec_rs): the gate no longer applies to them."""
        self.toks, self.rs = toks, True

    def tolist(self):
        rd = HS.read if self.packed else (lambda a: a.tolist())
        toks, conf = rd(self.toks), rd(self.conf)
        live = sorted(self.slot)
        if self.rs:
            self.keep, self.confs = None, [None] * len(self.depths)
            return [[int(t) for t in toks[self.slot[i]][:self.depths[i]]] if i in self.slot else [] for i in range(len(self.depths))]
        kept = self.be.drafter.conf_keep(toks, conf, [self.depths[i] for i in live])
        self.keep = [0] * len(self.depths)
        self.confs = [None] * len(self.depths)                       # per-job full-block confidences (the width policy trains on these)
        for i, kp in zip(live, kept):
            self.keep[i] = kp
            self.confs[i] = list(conf[self.slot[i]])
        self.be._note_cut(any(kp < self.depths[i] for i, kp in zip(live, kept)))
        return [[int(t) for t in toks[self.slot[i]][:self.depths[i]]] if i in self.slot else [] for i in range(len(self.depths))]


def _verify_replay_on() -> bool:
    """verify_replay reads the Engram rows of a replayed pass itself (its own staged slots): the gated verify stays out of its way."""
    from .. import verify_replay as VR
    return VR.ENABLED


class BatchedPass:
    """One verify pass over the sessions of a batched cache (VlmBackend.spec_verify_bc): rows.Pass plus the draft rings."""

    def __init__(self, backend, ps):
        self.backend, self.ps = backend, ps
        self.logits, self.taps, self.starts = ps.logits, ps.taps, ps.starts

    can_defer = True

    def commit(self, accepted: list, drafts=None, p0s=None, defer=None) -> None:
        """Session k keeps its pending token and accepted[k] drafts; drafts[k] (its draft state, or None) takes the taps of the kept
        positions (p0s[k] = position of the pending token). defer[k] (flag copy_v2): the taps wait in the lazy queue instead of being fed now."""
        be = self.backend
        self.ps.commit(accepted)
        if drafts is not None:
            rows = lambda k: {l: t[self.starts[k]:self.starts[k] + accepted[k] + 1, 0] for l, t in self.taps.items()}
            for k, d in enumerate(drafts):
                if d is not None and defer is not None and defer[k]:
                    be._defer_feed_rows(d, rows(k), p0s[k])
            be._feed_many([(d, rows(k), p0s[k]) for k, d in enumerate(drafts) if d is not None and not (defer is not None and defer[k])])


class SpecPass:
    """One verify pass over several sessions (VlmBackend.spec_verify)."""

    def __init__(self, backend, logits, taps, chain):
        self.backend, self.logits, self.taps, self.chain = backend, logits, taps, chain
        self.starts = chain.starts

    def commit(self, k: int, accepted: int, draft=None, p0: int = 0):
        """Cache of session k after `accepted` drafts were kept; the draft ring takes the taps of the kept positions (p0 = position of the
        pending token)."""
        be = self.backend
        cache = be.S.commit_seg(self.chain, k, accepted)
        if draft is not None:
            a0 = self.starts[k]
            be._feed(draft, {l: t[a0:a0 + accepted + 1, 0] for l, t in self.taps.items()}, p0)
        return cache


def load_q35(path: str, mtp: bool | None = None, mtp_head: str | None = None, log=print) -> tuple[Q35Backend, dict]:
    """Qwen3.5 / Qwen3-Next pack: EXL3, or a standard mlx-lm quantized pack (served on MLX's own linears, with MTP speculative decode when the
    pack has an `mtp/` sidecar or `mtp_head` names one; mtp=False turns it off)."""
    from .. import foreign_linear as FL
    if FL.is_foreign(path):                                       # GGUF / NVFP4 / MXFP4 served as stored (foreign_linear); never reached for an EXL3 or MLX pack
        model, cfg = FL.load_foreign(path, log=log, mtp=mtp is not False)
        from .q35mlx import Q35MlxBackend, find_mtp_dir
        heads = FL.take_mtp(model) if mtp is not False else None            # the nextn block of a GGUF: draft head from the file itself
        d = find_mtp_dir(path, mtp_head) if mtp is not False and heads is None else None      # a draft head shipped next to the weights (NVFP4 + MTP repos)
        be = Q35MlxBackend(model) if d is not None or heads is not None else Q35Backend(model)
        be.pack = path
        be.foreign = d is None and heads is None
        if heads is not None:
            log(f"mtp: {be.enable_mtp(None, weights=heads, quant_bits=0)}")
        elif d is not None:
            log(f"mtp: {be.enable_mtp(d)}")
        return be, cfg
    from .. import mlxpack, packcheck
    packcheck.check_pack(path, log=log)                           # header-only (KYOJIN_DESC_STRICT=1 raises); the draft head is checked where it is loaded
    if mlxpack.is_mlx_pack(path) or mlxpack.is_plain_pack(path):          # a plain bf16 / fp16 HF checkpoint is read by mlx-lm as stored (unquantized linears)
        import json
        from .q35mlx import Q35MlxBackend, find_mtp_dir
        model, _ = mlxpack.load(path)
        d = find_mtp_dir(path, mtp_head) if mtp is not False else None
        # Q35MlxBackend is the Qwen3.5 wrapper class (model.language_model, GDN layers, MTP head). A text-only mlx-lm class (Qwen3 dense / MoE, ...) has none of
        # these and serves on the plain Q35Backend, linears as stored (mx.quantized_matmul with the file's group_size / bits / mode).
        be = Q35MlxBackend(model) if "language_model" in model else Q35Backend(model)
        be.pack = path
        if os.environ.get("KYOJIN_WARM_LOAD", "1") != "0":
            be.warm(log)
        if d is not None and isinstance(be, Q35MlxBackend):
            log(f"mtp: {be.enable_mtp(d)}")
        return be, json.load(open(os.path.join(path, "config.json")))
    from ..exl3_linear import load_exl3
    model, cfg = load_exl3(path)
    from .q35mlx import Q35MlxBackend, find_mtp_dir
    d = find_mtp_dir(path, mtp_head) if mtp is not False else None
    be = Q35MlxBackend(model) if d is not None else Q35Backend(model)       # a draft head next to an EXL3 pack: same MTP serving path, EXL3 linears
    be.pack = path
    if d is not None:
        log(f"mtp: {be.enable_mtp(d)}")
    return be, cfg


class GenericBackend(Q35Backend):
    """KYOJIN_BACKEND_FALLBACK: a model_type without a modeldesc row on its stock mlx-lm / mlx-vlm class. Same plumbing as the Q35Backend (per-user caches,
    multiuser rows, kvdisk through the generic adapter); none of the Qwen step regions, no draft head, no kvwin. Correctness path, not a speed path."""
    kind = "generic"
    vision = None                                   # flag visiongen: the model's GenVision (visiongen.py), set by load_generic

    def _cache_model(self):
        """A vlm wrapper without make_cache (mlx-vlm qwen4_exp: the language model has it): the language model defines the caches, so its linear-attention
        layers get their state cache (mlx-lm's default would hand them plain KVCaches), in make_cache and in the short-session plain path alike. Models with
        their own make_cache, or none anywhere: the Q35Backend line, unchanged."""
        lm = self.model.language_model if "language_model" in self.model else None
        if not hasattr(self.model, "make_cache") and lm is not None and hasattr(lm, "make_cache"):
            return lm
        return self.model

    def _embed_kw(self):
        import inspect
        lm = self.model.language_model if "language_model" in self.model else self.model
        t = self.MU._trunk(lm)
        for name in ("inputs_embeds", "input_embeddings"):
            if name in inspect.signature(type(t).__call__).parameters:
                return lm, t, name
        raise ValueError("the trunk takes no input embeddings")

    def prefill_chunk(self, cache, ids, vl=None):
        """Without an image in the chunk: the Q35Backend line, unchanged. With one: the rows of its span overwrite the token embeddings (positions stay
        1-D offsets: GenVision refuses trunks for which that is not the model's math), the trunk runs on them."""
        if vl is None or not DV.touches(vl[0], vl[1], vl[1] + len(ids)):
            return super().prefill_chunk(cache, ids)
        lm, t, name = self._embed_kw()
        toks = mx.array(ids)[None]
        safe = mx.where(toks == self.vision.image_token_id, 0, toks)           # the image token may lie outside the embedding table; the overlay replaces those rows
        h = DV.overlay(t.embed_tokens(safe), vl[0], vl[1])
        out = (t(toks, cache=cache, **{name: h}) if self.MU._kw_cache(type(t)) else t(toks, cache, **{name: h}))[:, -1:]
        lg = self.MU.head(lm, out)[:, -1].astype(mx.float32)
        mx.eval(lg, *self.state(cache))
        return lg

    def disk_namespace(self):
        from .. import kvdisk
        return None if self.pack is None else {"arch": self.arch, "pack": kvdisk.pack_fingerprint(self.pack)}


def _mlx_lm_classes(cfg: dict):
    """The mlx-lm model / args classes of a config, resolved the way mlx_lm.load does (remap table, architecture remap). ModuleNotFoundError when mlx-lm has
    no class for it, so load_generic falls to the mlx-vlm loader."""
    from mlx_lm import utils
    from .. import mlxpack
    mlxpack.remap_text_types()
    try:
        return utils._get_classes(cfg)
    except ValueError as e:
        raise ModuleNotFoundError(str(e)) from e


def _fp8_block_sanitize(cls, cfg: dict):
    """FP8 packs whose quantization_config.weight_block_size is not [128, 128] (mlx-lm's deepseek_v3 / glm_moe_dsa sanitize dequantize with a
    hard-coded 128 block): the class's sanitize first gets the weight_scale_inv pairs dequantized with the pack's own block, then runs as is.
    Returns the original sanitize (to restore) or None when the pack needs nothing."""
    qc = cfg.get("quantization_config") or {}
    bs = qc.get("weight_block_size")
    if qc.get("quant_method") != "fp8" or not bs or list(bs) == [128, 128]:
        return None
    import mlx.core as mx
    bm, bn = int(bs[0]), int(bs[1])

    def deq(w, s):
        w = mx.from_fp8(w, dtype=mx.bfloat16)
        m, n = w.shape
        pm, pn = (-m) % bm, (-n) % bn
        w = mx.pad(w, ((0, pm), (0, pn))).reshape((m + pm) // bm, bm, (n + pn) // bn, bn)
        return (w * s[:, None, :, None]).reshape(m + pm, n + pn)[:m, :n].astype(mx.bfloat16)

    orig = getattr(cls, "sanitize", None)

    def sanitize(self, weights):
        done = {k[:-len("_scale_inv")]: deq(weights[k[:-len("_scale_inv")]], v) for k, v in weights.items() if k.endswith("weight_scale_inv")}
        weights = {k: done.get(k, v) for k, v in weights.items() if not k.endswith("weight_scale_inv")}
        return orig(self, weights) if orig is not None else weights
    cls.sanitize = sanitize
    return orig if orig is not None else False


def load_generic(path: str, log=print) -> GenericBackend:
    import contextlib
    import json
    from pathlib import Path
    cfg = json.load(open(os.path.join(path, "config.json")))
    mt = cfg["model_type"]
    from .. import visiongen
    want_vision = visiongen.enabled() and bool(cfg.get("vision_config"))        # flag visiongen: the mlx-vlm class carries the tower, an mlx-lm class of the same type drops it
    view = contextlib.nullcontext(path)
    if os.environ.get("KYOJIN_MTPGEN", "0") != "0":                          # flag mtpgen: model_mtp.safetensors is the head's, not the trunk's
        from .. import mtpgen
        view = mtpgen.trunk_view(path)
    with view as tpath:
        try:
            if want_vision:
                raise ModuleNotFoundError("visiongen: mlx-vlm class")
            from mlx_lm.utils import load_model
            mcls = _mlx_lm_classes(cfg)[0]                                   # mlx-lm's own resolution (MODEL_REMAPPING: mistral -> llama, minimax_m2 -> minimax, gemma4_unified -> gemma4, falcon_mamba -> mamba ...)
            prev = _fp8_block_sanitize(mcls, cfg)
            try:
                model, _ = load_model(Path(tpath), strict=True)
            except ValueError as e:                                           # the mlx-lm class and the pack disagree on the parameters (config keys the class defaults differently): the mlx-vlm class of the type gets its turn
                from mlx_vlm.utils import load_model as load_vlm_model
                try:
                    model = load_vlm_model(Path(tpath), lazy=False)
                except Exception:
                    raise e from None
                log(f"[kyojin] mlx-lm class refused the pack ({str(e).splitlines()[0][:80]}); serving the mlx-vlm class")
            finally:
                if prev is False:
                    del mcls.sanitize
                elif prev is not None:
                    mcls.sanitize = prev
        except ModuleNotFoundError:
            from mlx_vlm.utils import load_model as load_vlm_model
            model = load_vlm_model(Path(tpath), lazy=False)
        mx.eval(model.parameters())                                          # inside the view: the tensors are read lazily
    be = GenericBackend(model)
    be.pack, be.arch = path, mt
    if want_vision:
        be.vision = visiongen.GenVision(path, model)
        log("visiongen: image input " + ("on" if be.vision.available() else f"refused ({be.vision.reason})"))
    log(f"generic backend for model_type {mt!r}: stock model class, no Kyojin step regions")
    return be


def gate_env_ok(log=print) -> bool:
    """The gated Engram decode needs one command buffer per step (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER, set by kyojin_mlx/__init__ for a DS4.1 pack) and libcommitgate.dylib (built from tools/commitgate/commitgate.mm on first use)."""
    one_buffer = int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0") or 0) >= 100000 and int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0") or 0) >= 1000000
    if not one_buffer and os.environ.get("KYOJIN_GATE_ANYBUF") != "1":
        log("[kyojin] Engram gate off: MLX_MAX_OPS_PER_BUFFER=100000 and MLX_MAX_MB_PER_BUFFER=1000000 were not set before mlx was imported")
        return False
    from .. import engram_mlx
    if engram_mlx.ensure_commitgate() is None:
        log("[kyojin] Engram gate off: libcommitgate.dylib is missing and clang++ could not build it")
        return False
    return True


def load_vlm(path: str, union: bool = False, mtp: bool = False, mtp_rows: int = 24, ced: bool = False, ced_tail=None, ced_mid: int | None = None, log=print) -> tuple[VlmBackend, dict]:
    """DS4.1 / GLM pack: build, patch (EXL3 linears, experts, Engram tables), make the decode row invariant. mtp: also load the draft head
    (tensors `mtp.<stage>.*` of the pack) and arm the speculative verify. ced: CED prefill where the layer layout supports it (the server
    passes it on unless --no-ced; tools that call this directly keep the exact prefill unless they ask);
    ced_tail: rows of the decoder half per chunk (None = the sliding window, "exact" = the receptive field); ced_mid: first decoder layer (slices
    too shallow for the even split)."""
    from .. import packcheck, patch as P
    packcheck.check_pack(path, mtp=mtp, ced=ced, log=log)        # header-only: the pack carries what its descriptor promises (KYOJIN_DESC_STRICT=1 raises)
    model, spec, cfg = P.build_model(path)
    tables = None
    try:
        from .. import engram_rows as ER
        es = ER.EngramSpec.from_config(cfg)
        if es is not None:
            tables = ER.EngramTables(path, es)
            bad = tables.validate()
            assert not bad, "Engram tables do not match the config:\n  " + "\n  ".join(bad)
    except ImportError:
        pass
    P.patch_model(model, path, spec, cfg)
    gate_stage = None
    if tables is not None:
        from .. import engram_mlx
        from .. import gate_wanted
        gate_mode = os.environ.get("KYOJIN_ENGRAM_GATE", "auto")      # 1 = on (also with MTP: the D = 0 rounds of a speculative server are gated plain steps); auto / 0 = off
        if gate_wanted(gate_mode) and gate_env_ok(log):
            gate_stage = engram_mlx.install_staged_gather(model, tables)     # outside GatedDecode the staged embeds read synchronously, like the parallel ones
        else:
            engram_mlx.install_parallel_gather(model, tables)
    be = VlmBackend(model, spec, union=union)
    be.pack = path
    be.engram_ahead = getattr(tables, "ahead", None) if tables is not None else None
    if gate_stage is not None:
        from .. import engram_mlx
        be.gated = engram_mlx.GatedDecode(model, gate_stage)
        log("[kyojin] Engram gated decode ON (KYOJIN_ENGRAM_GATE=1)")
    if ced:                                         # before enable_mtp: the drafter taps wrap the CED blocks
        be.enable_ced(tail=ced_tail, keep=max(spec.mtp.keep if spec.mtp else 0, int(cfg.get("text_config", cfg).get("sliding_window", 0) or 0)) if mtp else 0, log=log, mid=ced_mid)
    if mtp:
        log(f"mtp: {be.enable_mtp(path, cfg, max_rows=mtp_rows)}")
    return be, cfg


def has_mtp_tensors(path: str, prefix: str = "mtp.") -> bool:
    """The pack carries the vendor's draft head (`mtp.*` tensors; Qwen3.5 mlx packs: the `mtp/` sidecar of tools/mtp_sidecar.py)."""
    import glob
    import json
    import struct
    if glob.glob(os.path.join(path, "mtp", "*.safetensors")):
        return True
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        return any(k.startswith(prefix) for k in json.load(open(idx)).get("weight_map", {}))
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if any(k.startswith(prefix) for k in json.loads(fh.read(n))):
                return True
    return False


def wire_memory(log=print) -> int:
    """Keep the MLX buffers resident (an MTLResidencySet sized by mx.set_wired_limit), as mlx_lm does. MLX's default limit 0 leaves every buffer
    out of the set: each command buffer commit re-makes the weights resident and macOS may unwire them between steps. On the Ultra the 172 GiB
    DS4.1 pack then ran 1.2-2.5 tok/s instead of 40-71 (29/10: wired pages swung 113-140 GiB, 13K GPU translation faults/s), fast or slow by
    machine state. KYOJIN_WIRED_MB sets the limit (0 = MLX default), else max_recommended_working_set_size (= iogpu.wired_limit_mb, the
    system-wide cap stays the guard). Returns the limit in bytes."""
    import mlx.core as mx
    env = os.environ.get("KYOJIN_WIRED_MB")
    limit = int(float(env) * 2 ** 20) if env is not None else int(mx.device_info()["max_recommended_working_set_size"])
    if limit > 0:
        mx.set_wired_limit(limit)
        log(f"wired limit {limit / 2 ** 30:.1f} GiB (MLX residency set; KYOJIN_WIRED_MB overrides, 0 = off)")
    return limit


def load_backend(path: str, kind: str | None = None, **kw) -> tuple[Backend, dict]:
    """Backend for a pack directory: the model_type of its config.json picks the family. mtp=None: speculative decode wherever a Qwen3.5 pack carries
    the vendor's draft head (has_mtp_tensors; ModelDesc.mtp_auto: DS4.1 / GLM keep it opt-in), True / False force it. The buffers are wired first (wire_memory)."""
    import json
    wire_memory(kw.get("log", print))
    if os.path.isfile(os.path.join(path, "config.json")):
        cfg = json.load(open(os.path.join(path, "config.json")))
    else:                                                         # a bare GGUF file: the architecture and sizes come from its metadata (foreign_names.gguf_config)
        from ..foreign_gguf import GgufFile, find_gguf
        from ..foreign_names import gguf_config
        cfg = gguf_config(GgufFile(find_gguf(path)))
    mt = cfg.get("model_type", "")
    mtp, mtp_head = kw.pop("mtp", None), kw.pop("mtp_head", None)
    auto = mtp is None
    from .. import modeldesc
    row = modeldesc.backend_desc(mt)                           # the table row of the model (exact model_type, else its backend prefix)
    if row is None and kind is None:
        from .. import foreign_linear as FL
        from ..foreign_names import ARCHS
        if FL.is_foreign(path) == "gguf" and any(a.model_type == mt for a in ARCHS.values()):     # a bare GGUF of a mapped architecture without a table row (qwen2, qwen2_moe, llama, glm4, glm4_moe,
            kind = "q35"                                                                         # gemma3, mistral3): the generic mlx-lm loader + Q35Backend, as qwen3 already is
    prefix = row.mtp_prefix if row is not None else "mtp."
    if kind == "q35" or (kind is None and row is not None and row.backend == "q35"):
        if auto:
            from ..foreign_names import gguf_has_mtp
            mtp = has_mtp_tensors(mtp_head or path, prefix) or gguf_has_mtp(path)
        return load_q35(path, mtp=mtp, mtp_head=mtp_head, log=kw.get("log", print))
    if kind == "vlm" or (row is not None and row.backend == "vlm"):
        from .. import foreign_linear as FL
        if FL.is_foreign(path):
            raise ValueError(f"{path}: a GGUF / NVFP4 / MXFP4 model of family {mt!r} (GLM, DeepSeek V4) is not served as stored yet: only the Qwen3 family is mapped (foreign_linear)")
        if auto:                                            # DS4.1 / GLM: MTP stays opt-in (--mtp-depth N) as before (ModelDesc.mtp_auto False); a row that sets it gets the draft head when the pack carries it
            mtp = bool(row is not None and row.mtp_auto and has_mtp_tensors(mtp_head or path, prefix))
        return load_vlm(path, mtp=bool(mtp), **kw)
    if (row is not None and row.backend == "generic") or os.environ.get("KYOJIN_BACKEND_FALLBACK", "1") != "0":   # generic row or no row: the per-user cache / multiuser plumbing on the stock model class
        be = load_generic(path, log=kw.get("log", print))
        if os.environ.get("KYOJIN_MTPGEN", "0") != "0" and mtp is not False:       # flag mtpgen: the pack's own draft head, any layout mtpgen knows
            from .. import mtpgen
            try:
                mtpgen.enable(be, path, cfg, log=kw.get("log", print))
            except mtpgen.HeadRefused as e:
                kw.get("log", print)(f"[kyojin] mtpgen: no draft head ({e})")
        return be, cfg
    raise ValueError(f"no serving backend for model_type {mt!r}")
