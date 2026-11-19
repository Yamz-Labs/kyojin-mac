# SPDX-License-Identifier: Apache-2.0
"""Verified MTP speculative decode for a generic_foreign model that carries a native hyper-connection MTP head (qwen4_exp: Qwen3.8-Flash-Next).

Switch KYOJIN_FOREIGN_MTP=1 (default off); the head comes from the model's own pack (`mtp.*` tensors) or from another pack of the same model
(KYOJIN_FOREIGN_MTP_HEAD=<dir>, e.g. the FP4-Local pack beside a GGUF trunk; hidden size / hc_count / vocab are asserted).

Head forward (reference: mlx_vlm drafters/qwen4_exp_mtp): for the trunk's pre-mixer hyper-connection state h_j [hc * H] at position j and the token t_{j+1}:
    x = fc_embedding(norm_e(embed(t_{j+1})))[..., None, :] + fc_hidden(norm_h(h_j).reshape(hc, H))   -> [hc * H]
    y = decoder layer (attention + MoE, own KV cache, position j);  logits = lm_head(mixer(y)); the next chain step feeds y (pre-mixer) as hidden.
No head (or KYOJIN_FOREIGN_MTP off) and KYOJIN_FOREIGN_LOOKUP_SPEC=1 (default off): `install_lookup` serves the same verify with prompt-lookup (n-gram) drafts from the
engine's per-session index (pld / LookupDrafts), no draft forward: a session with no n-gram draft in a round takes the plain step. Any new model gets speculation on day 0.
Verification: the pending token and its drafts go through the trunk as ONE forward (S = d + 1 rows); acceptance is exact argmax equality, so the text is the
plain greedy text whatever the draft. A partly accepted block rolls back: KV caches trimmed, GDN / conv states restored from the snapshot taken before the pass and the
accepted rows replayed (the head's own KV keeps only pairs of committed tokens)."""
from __future__ import annotations

import json
import os
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from .backends import GenericBackend
from .. import verify_capture as VC

ENABLED = os.environ.get("KYOJIN_FOREIGN_MTP", "0") not in ("0", "")
LOOKUP = os.environ.get("KYOJIN_FOREIGN_LOOKUP_SPEC", "0") not in ("0", "")
HEAD_DIR = os.environ.get("KYOJIN_FOREIGN_MTP_HEAD", "")
STATS = dict(rounds=0, drafted=0, accepted=0, replays=0)


class MtpHead(nn.Module):
    def __init__(self, tc):
        super().__init__()
        from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer, Qwen4ExpGatedResidual, Qwen4ExpRMSNorm
        H, hc = tc.hidden_size, tc.hc_count
        self.hc, self.H = hc, H
        self.pre_fc_norm_embedding = Qwen4ExpRMSNorm(H, eps=tc.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpRMSNorm(hc * H, eps=tc.rms_norm_eps)   # one RMS over all streams, then each stream projected alone
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)
        lc = replace(tc, num_hidden_layers=1, layer_types=["qwen_sparse_attention"], full_attention_interval=1, ple_layer_ids=[])
        self.layers = [Qwen4ExpDecoderLayer(lc, layer_idx=0)]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(lc, use_combine=False)

    def rows(self, embed, toks, hid, cache):
        """toks [R] (the token of each pair), hid [1, R, hc*H] -> (mixed [1, R, H] for the head, pre-mixer [1, R, hc*H] for the next chain step)."""
        from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_attention_mask
        R = toks.shape[0]
        pe = self.fc_embedding(self.pre_fc_norm_embedding(embed(toks[None])))
        ph = self.fc_hidden(self.pre_fc_norm_hidden(hid).reshape(1, R, self.hc, self.H))
        x = (pe[..., None, :] + ph).reshape(1, R, self.hc * self.H)
        pos = (mx.arange(R, dtype=mx.int32) + cache.offset)[None]
        y = self.layers[0](x, toks[None], mask=_create_qwen3_5_attention_mask(x, cache), cache=cache, position_ids=pos)
        return self.hyper_connection_mixer(y), y


def build_head(tc, tensors: dict, switch=None) -> MtpHead:
    """tensors: `mtp.*` name -> array (foreign_linear.read_mtp_head); switch: ForeignSwitchGLU of the routed experts."""
    head = MtpHead(tc)
    w = [(k[len("mtp."):], v) for k, v in tensors.items() if k.startswith("mtp.")]
    mlp = head.layers[0].mlp
    if switch is None and hasattr(mlp, "switch_mlp"):
        raise ValueError("mtp head: the routed-expert pack is missing (no switch_mlp weights): refusing a head with random experts")
    if switch is not None:
        mlp.switch_mlp = switch
    have = {k for k, _ in w}
    want = {k for k, _ in nn.utils.tree_flatten(head.parameters())}
    missing = sorted(k for k in want - have if ".switch_mlp." not in k)
    if missing:
        raise ValueError(f"mtp head: tensors missing for {missing[:6]} (+{max(0, len(missing) - 6)})")
    head.load_weights([(k, v) for k, v in w if k in want], strict=False)
    mx.eval(head.parameters())
    return head


def check_compatible(tc, head_cfg: dict):
    """A head from another pack: the same hidden size / hyper-connection streams / vocabulary as the trunk it will draft for."""
    hc = head_cfg.get("text_config", head_cfg)
    for key in ("hidden_size", "hc_count", "vocab_size"):
        if key in hc and getattr(tc, key, None) is not None and hc[key] != getattr(tc, key):
            raise ValueError(f"mtp head pack {key}={hc[key]} != trunk {getattr(tc, key)}")


def _gguf_tokens(trunk: str):
    """The token strings of a GGUF trunk (tokenizer.ggml.tokens), or None when `trunk` is not a GGUF."""
    from .. import foreign_gguf as GF
    try:
        p = GF.find_gguf(trunk)
    except (GF.GgufError, OSError):
        return None
    if not p or not str(p).endswith(".gguf"):
        return None
    return [str(t) for t in GF.GgufFile(p).kv.get("tokenizer.ggml.tokens", [])] or None


def _json_tokens(f: str) -> list:
    """id-ordered token strings of a tokenizer.json (model vocab + added tokens)."""
    t = json.load(open(f))
    v = dict(t.get("model", {}).get("vocab", {}))
    for a in t.get("added_tokens", []):
        v[a["content"]] = a["id"]
    out = [None] * (max(v.values()) + 1 if v else 0)
    for s, i in v.items():
        out[i] = s
    return out


def check_tokenizer(head_dir: str, trunk_dir: str):
    """A head from a sibling pack drafts in the trunk's token ids: refuse it when the tokenizers differ. Both with a tokenizer.json: same sha256, else the same vocab id by id
    (the loader regenerates a tokenizer.json from a GGUF). A GGUF trunk (no tokenizer.json): the head's tokenizer.json vocab must equal tokenizer.ggml.tokens id by id (padding ids past the vocab ignored)."""
    import hashlib
    fh, ft = os.path.join(head_dir, "tokenizer.json"), os.path.join(trunk_dir, "tokenizer.json")
    if not os.path.isfile(fh):
        raise ValueError(f"mtp head pack {head_dir}: no tokenizer.json: refusing the head")
    if os.path.isfile(ft):
        if hashlib.sha256(open(fh, "rb").read()).hexdigest() == hashlib.sha256(open(ft, "rb").read()).hexdigest():
            return
        g = _json_tokens(ft)          # same ids, other bytes (e.g. the tokenizer.json the loader regenerates from a GGUF): compare the vocab
    else:
        g = _gguf_tokens(trunk_dir)
    if g is None:
        raise ValueError(f"mtp head pack {head_dir}: the trunk ({trunk_dir}) has neither a tokenizer.json nor GGUF tokens: refusing the head")
    h = _json_tokens(fh)
    n = min(len(h), len(g))
    bad = [i for i in range(n) if h[i] is not None and h[i] != g[i]]
    if not h or not g or bad or n < 0.99 * max(len(h), len(g)):
        raise ValueError(f"mtp head pack {head_dir}: tokenizer vocab differs from the trunk's ({len(h)} vs {len(g)} tokens, {len(bad)} ids differ, first {bad[:3]}): refusing the head")


class DraftState:
    """Per session: the head's KV cache (`mv` pairs: pair j = token j + 1 with the trunk hidden of position j) and the trunk hiddens from position `mv` on (pend)."""
    __slots__ = ("mc", "mv", "pend", "base")

    def __init__(self):
        from mlx_vlm.models.qwen4_exp.language import QSAKVCache
        self.mc, self.mv, self.pend, self.base = QSAKVCache(), 0, [], 0      # base: first position the head cache covers (> 0 after a stale reset)


class _Bc:
    def __init__(self, caches):
        self.caches = list(caches)


class _Pass:
    def __init__(self, be, bc, parts, starts):
        self.be, self.bc, self.parts, self.starts = be, bc, parts, starts
        lgs = [p[0] for p in parts]
        self.logits = lgs[0] if len(lgs) == 1 else mx.concatenate(lgs, axis=0)

    def commit(self, accepted, drafts=None, p0s=None, **_):
        be = self.be
        for k, (c, (lg, h, snap, x, sink)) in enumerate(zip(self.bc.caches, self.parts)):
            S, keep = x.shape[0], accepted[k] + 1
            if keep < S:
                be._rollback(c, snap, x, keep, sink)
            if drafts is not None and drafts[k] is not None:
                drafts[k].pend.append(h[:, :keep])
        if be.mtp is None:
            STATS["drafted"] += sum(p[3].shape[0] - 1 for p in self.parts)
        STATS["rounds"] += 1
        STATS["accepted"] += sum(int(a) for a in accepted)


class FnMtpBackend(GenericBackend):
    """GenericBackend + the MTP head: supports_spec, solo-session verify passes (like Q35MlxBackend's spec path)."""
    max_spec_rows = 0

    def enable_mtp(self, head: MtpHead, max_rows: int = int(os.environ.get("KYOJIN_SPEC_ROWS", "16"))):
        self.mtp = head
        self.lm = self.model.language_model if "language_model" in self.model else self.model
        self.trunk = self.MU._trunk(self.lm)
        self.max_spec_rows = max_rows
        self.supports_spec = True
        self._rounds = 0
        self.lookup_only = False
        return {"mtp_head": type(head).__name__, "max_spec_rows": max_rows}

    mtp = None
    lookup_only = False

    def enable_lookup(self, max_rows: int = int(os.environ.get("KYOJIN_SPEC_ROWS", "16"))):
        """Head-free speculation: n-gram drafts from the engine's index, verified by the same one-forward pass; no draft state, no draft forward."""
        self.mtp = None
        self.lm = self.model.language_model if "language_model" in self.model else self.model
        self.trunk = self.MU._trunk(self.lm)
        self.max_spec_rows = max_rows
        self.supports_spec = True
        self.lookup_only = True
        self._rounds = 0
        return {"spec": "lookup", "max_spec_rows": max_rows}

    # -- trunk with the pre-mixer hyper-connection state
    def _fwd(self, toks, cache):
        """toks [B, S] -> (mixed hidden [B, S, H], pre-mixer hyper-connection state [B, S, hc*H])."""
        sink = []
        h = self.trunk(toks, cache=cache, hidden_sink=sink, capture_layer_ids=[])
        return h, sink[0]

    def _lg(self, h):
        return self.MU.head(self.lm, h).astype(mx.float32)

    @staticmethod
    def _pos(cache) -> int:
        for c in cache:
            if hasattr(c, "offset") and not hasattr(c, "cache"):
                return c.offset
        raise ValueError("no attention cache")

    # -- draft state
    def new_draft(self):
        return DraftState() if self.supports_spec and self.mtp is not None else None

    def draft_snapshot(self, d):
        if d is None or not d.pend:
            return None
        return [self.MU.fork_cache([d.mc])[0], d.pend[-1], d.mv, d.base]

    def draft_restore(self, snap):
        d = DraftState()
        d.mc, d.pend, d.mv = self.MU.fork_cache([snap[0]])[0], [snap[1]], snap[2]
        d.base = snap[3] if len(snap) > 3 else 0
        return d

    def draft_arrays(self, d) -> list:
        if d is None:
            return []
        # the head cache state also carries non-array entries (index_block_ratio int, None index keys): arrays only
        return [a for a in (*d.mc.state, *d.pend) if isinstance(a, mx.array)]

    def draft_nbytes(self, d) -> int:
        return sum(a.nbytes for a in self.draft_arrays(d))

    def _pairs(self, d, toks, hid):
        return self.mtp.rows(self.trunk.embed_tokens, toks, hid, d.mc)

    def _absorb(self, d, ids, h, p):
        n = len(ids)
        if p == 0:
            toks, hid = ids[1:], h[:, :-1]
        else:
            toks, hid = ids, mx.concatenate([d.pend[-1], h[:, :-1]], axis=1)
        if len(toks):
            self._pairs(d, mx.array(toks, dtype=mx.int32), hid)
        d.mv, d.pend = p + n - 1 - d.base, [h[:, -1:]]

    def prefill_chunk(self, cache, ids, draft=None, vl=None):
        if draft is None or not self.supports_spec or vl is not None:
            return super().prefill_chunk(cache, ids) if vl is None else super().prefill_chunk(cache, ids, vl)
        p = self._pos(cache)
        h, hid = self._fwd(mx.array(ids, dtype=mx.int32)[None], cache)
        lg = self._lg(h[:, -1:])[:, -1]
        self._absorb(draft, ids, hid, p)
        mx.eval(lg, *self.state(cache), *self.draft_arrays(draft))
        return lg

    def replay(self, cache, tokens, draft=None):
        if draft is None or not self.supports_spec:
            return super().replay(cache, tokens, draft)
        for i in range(0, len(tokens), 256):
            self.prefill_chunk(cache, [int(t) for t in tokens[i:i + 256]], draft=draft)
        return cache

    # -- batching: a speculative round verifies each session on its own cache
    def spec_ok(self, n_sessions: int) -> bool:
        return self.supports_spec and n_sessions == 1

    def batch_spec(self, caches):
        return _Bc(caches)

    def bind_drafts(self, bc, drafts):
        self._bound = list(drafts)

    def unbatch(self, bc, n):
        return bc.caches if isinstance(bc, _Bc) else super().unbatch(bc, n)

    def step(self, bc, tokens):
        if isinstance(bc, _Bc):
            outs = []
            for i, c in enumerate(bc.caches):
                h, _ = self._fwd(tokens[i:i + 1, None], c)
                outs.append(self._lg(h[:, -1:])[:, -1])
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        bound = getattr(self, "_bound", None) if self.supports_spec and self.mtp is not None else None
        if not bound:
            return super().step(bc, tokens)
        h, hid = self._fwd(tokens[:, None], bc)
        for i, d in enumerate(bound):
            if d is not None:
                self._plain_row(d, hid[i:i + 1], len(bound) > 1)
        return self._lg(h[:, -1:])[:, -1]

    @staticmethod
    def _plain_row(d, row, shared):
        """A plain (non-speculative) step's hidden row. Solo: kept for the next draft. Several sessions never draft (spec_ok), so the rows would pile up: the head
        state goes stale instead (fresh head cache from this position, only the last row kept) and the draft rebuilds from the trunk rows of the steps after."""
        if shared:
            pos = d.base + d.mv + sum(x.shape[1] for x in d.pend)
            from mlx_vlm.models.qwen4_exp.language import QSAKVCache
            d.mc, d.mv, d.pend, d.base = QSAKVCache(), 0, [row], pos
        else:
            d.pend.append(row)

    def step_feed(self, bc, tokens, drafts, p0s):
        outs = []
        for i, c in enumerate(bc.caches):
            h, hid = self._fwd(tokens[i:i + 1, None], c)
            if drafts[i] is not None:
                drafts[i].pend.append(hid)
            outs.append(self._lg(h[:, -1:])[:, -1])
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    # -- drafting: one pass over the pairs the head still lacks, then a chain of single-row steps (scratch rows trimmed)
    def _draft_one(self, d, depth, hist):
        hid = d.pend[0] if len(d.pend) == 1 else mx.concatenate(d.pend, axis=1)
        k = hid.shape[1]
        d.mc.trim(d.mc.offset - d.mv)
        o, y = self._pairs(d, mx.array(hist[1][-k:], dtype=mx.int32), hid)
        o, y = o[:, -1:], y[:, -1:]
        d.mv, d.pend = d.mv + k, []
        ds = []
        for j in range(depth):
            t = mx.argmax(self._lg(o)[0, 0])
            ds.append(t)
            if j + 1 < depth:
                o, y = self._pairs(d, t[None], y)
        if depth > 1:
            d.mc.trim(depth - 1)                                       # the chain rows are scratch: the next draft starts from mv
        return mx.stack(ds)

    def _set_head_compile(self, on: bool):
        """KYOJIN_VERIFY_COMPILE_DRAFT: the head layer (and its mlp) wrapped by step_compile, or back to its own classes."""
        from .. import step_compile as SC
        lay = self.mtp.layers[0]
        if on and not hasattr(self, "_head_cls"):
            if not SC.ENABLED:
                return
            self._head_cls = (lay.__class__, lay.mlp.__class__ if hasattr(lay, "mlp") else None)
            SC.install(self.mtp, force=True)
            self._head_cls_on = (lay.__class__, lay.mlp.__class__ if hasattr(lay, "mlp") else None)
        if hasattr(self, "_head_cls"):
            lay.__class__ = (self._head_cls_on if on else self._head_cls)[0]
            if hasattr(lay, "mlp"):
                lay.mlp.__class__ = (self._head_cls_on if on else self._head_cls)[1]
        self._draft_on = on

    def draft_many(self, jobs, want_logits: bool = False):
        if self.mtp is not None and VC.DRAFT != getattr(self, "_draft_on", False):
            self._set_head_compile(VC.DRAFT)
        if self.mtp is None:                                           # lookup-only: the engine's n-gram rows are the only drafts
            return [[] for _ in jobs]
        out = []
        for d, pending, p0, depth, hist in jobs:
            if d is None or depth <= 0 or not d.pend or p0 - d.base - d.mv != sum(x.shape[1] for x in d.pend):
                out.append(None)
                continue
            out.append(self._draft_one(d, depth, hist))
        live = [o for o in out if o is not None]
        if live:
            mx.eval(*live)
        res = [[] if o is None else o.tolist() for o in out]
        STATS["drafted"] += sum(len(r) for r in res)
        return res

    # -- verify
    def spec_verify_bc(self, bc, token_lists):
        parts, starts, r = [], [], 0
        for c, t in zip(bc.caches, token_lists):
            x = mx.array(t, dtype=mx.int32)
            S = x.shape[0]
            snap = self._snap(c) if S > 1 else None
            VC.begin(S) if snap is not None and any(s is not None for s in snap) else None
            try:
                h, hid = self._fwd(x[None], c)
            finally:
                sink = VC.end()
            parts.append((self._lg(h)[0], hid, snap, x, sink))
            starts.append(r)
            r += S
        return _Pass(self, bc, parts, starts)

    @staticmethod
    def _snap(cache):
        return [list(c.cache) if hasattr(c, "cache") else None for c in cache]

    def _rollback(self, cache, snap, x, keep, sink=None):
        S = x.shape[0]
        fixed = any(s is not None for s in snap)
        if fixed and sink is not None and keep >= VC.MINKEEP:                                 # KYOJIN_VERIFY_CAPTURE: the fixed-state layers' state after `keep` rows was recorded by the verify pass
            sel = [sink.select(c, keep) if s is not None else None for c, s in zip(cache, snap)]
            if all(r is not None for r, s in zip(sel, snap) if s is not None):
                for c, s, r in zip(cache, snap, sel):
                    if s is None:
                        c.trim(S - keep)
                    else:
                        c.cache = list(r)
                VC.STATS["captured"] += 1
                return
            VC.STATS["incomplete"] += 1
        for c, s in zip(cache, snap):
            if s is not None:
                c.cache = list(s)
            else:
                c.trim(S if fixed else S - keep)                       # fixed-state layers present: back to before the pass, the replay appends the accepted rows
        if fixed:
            self._fwd(x[None, :keep], cache)                           # the accepted rows again: GDN / conv states and KV of exactly `keep` positions
            STATS["replays"] += 1

    def settle(self, bc, drafts=()):
        self._rounds += 1
        if self._rounds % 16 == 0:
            mx.eval(*[a for c in bc.caches for a in self.state(c)], *[a for d in drafts for a in self.draft_arrays(d)])


def accept_stats() -> dict:
    return dict(STATS)


def install(be, model, path: str, cfg: dict, log=print):
    """The GenericBackend `be` of a qwen4_exp model -> FnMtpBackend with the head read from KYOJIN_FOREIGN_MTP_HEAD (another pack of the model) or from `path` itself.
    No head found / refused: `be` unchanged (plain decode)."""
    import json
    from .. import foreign_linear as FL
    from .backends import has_mtp_tensors
    lm = model.language_model if "language_model" in model else model
    tc = getattr(lm, "args", None)
    src = HEAD_DIR or path
    mt = str(getattr(tc, "model_type", None) or getattr(getattr(model, "args", None), "model_type", ""))
    ok = (mt.startswith("qwen4_exp"), os.path.isdir(src), os.path.isdir(src) and has_mtp_tensors(src))
    if not all(ok):
        log(f"[kyojin] KYOJIN_FOREIGN_MTP: no MTP head (model_type {mt!r} qwen4_exp={ok[0]}, dir={ok[1]}, mtp tensors={ok[2]}, src {src}); plain decode")
        return be
    try:
        check_compatible(tc, json.load(open(os.path.join(src, "config.json"))))
        if os.path.abspath(src) != os.path.abspath(path):
            check_tokenizer(src, path)
        tensors, switch = FL.read_mtp_head(src, log=log)
        head = build_head(tc, tensors, switch)
    except (ValueError, OSError) as e:                               # unreadable config / tokenizer / GGUF too: refuse the head, serve plain
        log(f"[kyojin] KYOJIN_FOREIGN_MTP: head refused ({type(e).__name__}: {e}); plain decode")
        return be
    out = FnMtpBackend.__new__(FnMtpBackend)
    out.__dict__.update(be.__dict__)
    log(f"[kyojin] foreign MTP: {out.enable_mtp(head)} (head from {src})")
    out.foreign = False
    return out


def install_lookup(be, model, log=print):
    """The GenericBackend `be` of a generic_foreign model -> FnMtpBackend in lookup-only mode (KYOJIN_FOREIGN_LOOKUP_SPEC=1, no MTP head). Engine side: flag spec_lookup (read per round)."""
    out = FnMtpBackend.__new__(FnMtpBackend)
    out.__dict__.update(be.__dict__)
    log(f"[kyojin] foreign lookup speculation: {out.enable_lookup()} (no draft head)")
    out.foreign = False
    return out
