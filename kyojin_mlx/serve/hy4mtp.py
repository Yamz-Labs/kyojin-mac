# SPDX-License-Identifier: Apache-2.0
"""Serve backend of the Hy4 MTP speculation (flag KYOJIN_HY4_MTP, default off; kyojin_mlx/hy4_mtp.py holds the head and the rounds).

Same contract as fnmtp.FnMtpBackend (the engine's existing speculative machinery): supports_spec, new_draft / draft_many (one greedy chain per session),
spec_verify_bc(bc, [[pending, *drafts]]) -> pass with .logits / .starts / .commit(accepted, drafts, p0s) (longest accepted prefix kept, the rest rolled back),
step_feed for rounds without drafts, prefill_chunk(draft=) feeding the head. One session (Hy4Backend's Engram history is per model)."""
from __future__ import annotations

import os

import mlx.core as mx

from .. import hy4_mtp as M
from .backends import Hy4Backend


class _Bc:
    def __init__(self, caches):
        self.caches = list(caches)


class _Pass:
    def __init__(self, be, bc, parts):
        self.be, self.bc, self.parts = be, bc, parts
        self.starts, r = [], 0
        for lg, _, ids in parts:
            self.starts.append(r)
            r += len(ids)
        lgs = [p[0] for p in parts]
        self.logits = lgs[0] if len(lgs) == 1 else mx.concatenate(lgs, axis=0)

    def commit(self, accepted, drafts=None, p0s=None, **_):
        for k, (c, (lg, h, ids)) in enumerate(zip(self.bc.caches, self.parts)):
            st = drafts[k] if drafts is not None else None
            self.be.mtp.commit(c, st, ids, h, int(accepted[k]), exact=self.be.exact)


class Hy4MtpBackend(Hy4Backend):
    kind = "hy4"

    def enable_mtp(self, mtp: M.Mtp, max_rows: int = int(os.environ.get("KYOJIN_SPEC_ROWS", "8"))):
        self.mtp, self.max_spec_rows, self.supports_spec = mtp, max_rows, True
        self.lookup_only, self.exact, self._bound = False, M.exact_commit(), None
        self.default_mtp_depth = M.default_depth()
        self.default_fixed_depth = self.default_mtp_depth if M.fixed_depth() else None      # the engine pins every round to this depth (server.py)
        return {"hy4_mtp_head": "ok", "max_spec_rows": max_rows, "exact_commit": self.exact}

    supports_spec = False
    max_spec_rows = 0

    # -- draft state
    def new_draft(self):
        return self.mtp.new_state() if self.supports_spec else None

    @staticmethod
    def _copy_draft(d):
        """An independent DraftState: the head's KV and the pending trunk states are arrays (values), the token list is copied."""
        if d is None:
            return None
        from mlx_lm.models.cache import KVCache
        n = M.DraftState.__new__(M.DraftState)
        n.dim, n.toks, n.pend = d.dim, list(d.toks), [p[...] for p in d.pend]
        c = KVCache()
        c.keys = None if d.cache.keys is None else d.cache.keys[...]
        c.values = None if d.cache.values is None else d.cache.values[...]
        c.offset = d.cache.offset
        n.cache = c
        return n

    def draft_snapshot(self, d):
        return None if self.one_session else self._copy_draft(d)

    def draft_restore(self, snap):
        return self._copy_draft(snap)

    def draft_arrays(self, d) -> list:
        return [] if d is None else [a for a in (*d.cache.state, *d.pend) if isinstance(a, mx.array)]

    def draft_nbytes(self, d) -> int:
        return sum(a.nbytes for a in self.draft_arrays(d))

    # -- prefill / plain steps keep the head state in step with the trunk
    def prefill_chunk(self, cache, ids, vl=None, draft=None):
        if draft is None or not self.supports_spec or vl is not None:
            return super().prefill_chunk(cache, ids, vl)
        self._claim(cache)
        from .. import headlast
        with headlast.last_row():                                      # the head runs on the last row of a prompt chunk only (the hidden sink keeps every row for the draft head)
            lg, h = self.mtp.forward(cache, [int(t) for t in ids])
        self.mtp.feed(draft, ids, h)
        self.mtp.absorb(draft)
        out = lg[-1:]
        mx.eval(out, *self.state(cache), *self.draft_arrays(draft))
        return out

    def replay(self, cache, tokens, draft=None):
        for i in range(0, len(tokens), 256):
            self.prefill_chunk(cache, [int(t) for t in tokens[i:i + 256]], draft=draft)
        return cache

    def spec_ok(self, n_sessions: int) -> bool:
        return self.supports_spec and n_sessions == 1

    def batch_spec(self, caches):
        for c in caches:
            self._claim(c)
        return _Bc(caches)

    def bind_drafts(self, bc, drafts):
        self._bound = list(drafts)

    def unbatch(self, bc, n):
        return bc.caches if isinstance(bc, _Bc) else super().unbatch(bc, n)

    def step(self, bc, tokens):
        cache = bc.caches[0] if isinstance(bc, _Bc) else bc
        bound = self._bound if self.supports_spec else None
        if not bound or bound[0] is None:
            return Hy4Backend.step(self, cache, tokens)
        return self.step_feed(_Bc([cache]), tokens, bound, None)

    def step_feed(self, bc, tokens, drafts, p0s):
        c = bc.caches[0]
        self._claim(c)
        lg, h = self.mtp.forward(c, tokens.reshape(1).astype(mx.int32))
        if drafts[0] is not None:
            self.mtp.feed(drafts[0], [int(tokens.reshape(-1)[0].item())], h)
        mx.eval(lg, *self.state(c))
        return lg

    # -- drafting and verify
    def draft_many(self, jobs, want_logits: bool = False, couple=None):
        out = []
        for d, pending, p0, depth, hist in jobs:
            r = self.mtp.draft(d, int(pending), int(depth)) if d is not None and p0 == len(d.toks) else None
            out.append([] if r is None else r)
        return out

    def spec_verify_bc(self, bc, token_lists):
        parts = []
        for c, t in zip(bc.caches, token_lists):
            self._claim(c)
            ids = [int(x) for x in t]
            lg, h = self.mtp.forward(c, ids)
            parts.append((lg, h, ids))
        return _Pass(self, bc, parts)


def install(be, gpath: str, log=print):
    """Hy4Backend `be` -> Hy4MtpBackend with the head of the GGUF at `gpath`; refused head: stops with the reason (HY4_ALLOW_SLOW=1: `be` unchanged, plain decode, reason logged)."""
    from .. import foreign_gguf as GF
    from ..foreign_skeleton_rows import ROWS
    g = GF.GgufFile(gpath)
    try:
        gpu = mx.default_device() == mx.gpu
        head = M.build_head(g, be.model, row=ROWS["hy_v4_flash"], dtype=mx.float32, expert_dtype=mx.bfloat16 if gpu else mx.float32, log=log)
    except (M.HeadRefused, Exception) as e:                           # noqa: BLE001
        msg = f"KYOJIN_HY4_MTP is on but the MTP head was refused ({type(e).__name__}: {e})"
        if os.environ.get("HY4_ALLOW_SLOW") != "1":                  # the shipped speed needs the head: stop instead of serving plain without notice
            raise RuntimeError(msg + ". Fix the model file, or set HY4_ALLOW_SLOW=1 to serve plain decode anyway.") from e
        log(f"[kyojin] {msg}; plain decode (HY4_ALLOW_SLOW=1)")
        return be
    finally:
        g.close()
    out = Hy4MtpBackend.__new__(Hy4MtpBackend)
    out.__dict__.update(be.__dict__)
    log(f"[kyojin] hy4 MTP: {out.enable_mtp(M.Mtp(be.model, head))}")
    return out
