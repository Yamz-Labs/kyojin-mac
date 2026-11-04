# SPDX-License-Identifier: Apache-2.0
"""DSpark block drafter (the MTP head of DS4.1) on the patched mlx-vlm model: EXL3 stages, one forward per draft block.

Reference: the checkpoint's inference/model.py (DSparkAttention / DSparkBlock / forward_spec) and the CUDA / ROCm engine
(exllamav3/architecture/deepseek_v4_mtp.py, modules/arch_specific/dspark.py), whose accept rule and draft-length gate this follows.

Draft call, after the trunk consumed p0 positions and sampled the pending token t0 (position p0):
  * block ids [t0, noise x (block - 1)] -> the trunk's embedding -> hc_mult stream copies
  * `stages` blocks (the trunk's DeepseekV41Block: mHC, MoE with `dspark_n_routed_experts` experts, top `dspark_num_experts_per_tok`),
    attention replaced by DSparkAttention: every block query sees, non-causally, the last <= `sliding_window` main-kv rows and the whole
    block in one softmax with the per-head sinks, then the inverse rope on the output and the grouped wo_a / wo_b
  * hc collapse with the deferred pre-mix, the drafter's own norm, the TRUNK's head -> logits of the 5 block positions
  * greedy block sampling with the markov bigram bias: out[i + 1] = argmax(logits[i] + markov_head(markov_embed(out[i])))
  * confidence head per position: sigmoid(proj(cat(pre-norm state, markov embed))) >= threshold; the draft length is the longest
    confident prefix (0 = skip drafting this round)
Main-kv rows: for each committed position, the taps of the trunk (mean over the hc streams of the INPUT of the target layers, concatenated)
go through the fp8 round trip, main_proj + main_norm, then each stage's wkv / kv_norm / rope / fp8 fake quant -> one 512-wide row per stage and position.

Compiled draft (KYOJIN_DSPARK_COMPILE=0 switches it off): once the main-kv ring holds `sliding_window` rows (a fixed shape) the whole forward above, from the block ids to
the tokens and the confidences, is ONE mx.compile region with the block start position as an array input (the rope offset is an array, so a new position is a new
input, not a new trace); before that, and whenever the debug dump is on, the eager code runs. `append` is one region per row count (1 .. block) the same way.
The compiled forward is the eager one with its python overhead (graph build, elementwise ops fused): same values, which tests/test_dspark.py checks.

Everything the drafter produces is only a proposal: the trunk verifies (kyojin_mlx.spec), so output never depends on it."""
from __future__ import annotations

from . import archs as _A
import copy
import importlib
import os

import mlx.core as mx
import mlx.nn as nn

from . import fired as FD, flags as FL, patch as P, replay as RP
from .archs import ArchSpec, mtp_arch


COMPILE = os.environ.get("KYOJIN_DSPARK_COMPILE", "1") != "0"
REFINE = os.environ.get("KYOJIN_DSPARK_REFINE", "0") != "0"      # flags.py dspark_refine: second block forward with the drafted prefix as ids
CAND = int(os.environ.get("KYOJIN_DRAFT_CAND", "0") or 0)       # flags.py draft_cand: markov tail over the K best base-logit tokens per position (0 = the full vocab)
DRAFT_COMPILE = os.environ.get("KYOJIN_DRAFT_COMPILE", "0") not in ("0", "")  # flags.py draft_compile: the serving-path rows draft runs as one mx.compile region (the solo draft already does)
STATS = dict(trace_draft=0, trace_append=0, call_draft=0, call_append=0, eager_draft=0, eager_append=0, refine_pass=0, cand_tail=0,
             rows_compiled=0, rows_compile_fail=0)     # tests/test_dspark.py: flat after the warm-up


def TAIL_CUT() -> bool:
    """Flag draft_tail_cut (KYOJIN_DRAFT_TAIL_CUT, read per call): the draft head's tail (the markov chain, the trunk head and the confidence head) runs
    on the rows the round plans to verify, not on the whole block. A proposal past the planned depth is cut by the confidence gate and never verified, so
    the proposals, the confidences and every emitted token of the rows that are kept are those of the full block."""
    return os.environ.get("KYOJIN_DRAFT_TAIL_CUT", "1") not in ("0", "")


def _conf_pct(pct) -> float | None:
    p = int(pct or 0)
    return p / 100 if p > 0 else None


CONF_OVERRIDE: float | None = _conf_pct(os.environ.get("KYOJIN_DSPARK_CONF_PCT"))    # confidence-gate threshold of flag dspark_conf_pct (start value or POST /admin/flags); None = each drafter keeps its own


def _classes():
    lang = _A.lang()
    return lang


class _Ctx:
    """What the draft attention reads: the block start position and the main-kv rows of every stage."""

    def __init__(self, n_stages: int):
        self.p0 = 0
        self.rows = [None] * n_stages
        self.dbg = None                                   # list: every stage appends its attention output (tools/draft_dump.py)


class _RowsCtx:
    """Rows layout of a draft pass for S sessions: row s * block + i is block position i of session s, at position p0[s] + i; every row is a
    "user" of a decode step ([rows, 1, ...] tensors, the code path of the trunk's batched core), and only the attention looks at the block
    structure: the rows of a session attend over their own ring (`ctxs[s]`) and the block, one SDPA call per session."""

    def __init__(self, ctxs: list, p0s: list | None, block: int, pos: mx.array | None = None):
        self.ctxs, self.block = ctxs, block
        self.pos = pos if pos is not None else mx.array([p + i for p in p0s for i in range(block)], dtype=mx.int32)


def _make_attention_class(lang):
    fq = lang.fake_quant_fp8_ue8m0

    class DSparkAttention(_A.cls("attn")):
        """Compressor-less V4.1 attention with draft-block semantics (see module doc). `cache` is the _Ctx."""

        fq_out = True            # the checkpoint's model.py quantizes the attention output (fp8, blocks of 32) before the bf16 wo_a

        def _rows(self, x: mx.array, rc: _RowsCtx) -> mx.array:
            """x [S * block, 1, H] -> same shape. Row-wise except the per-session SDPA; each row equals the one it has alone (S = 1)."""
            R, L = x.shape[0], rc.block
            D = self.head_dim
            qr = self.q_norm(self.wq_a(fq(x)))
            q = self.rope(self.wq_b(fq(qr)).reshape(R, self.n_heads, 1, D), rc.pos)
            kv = fq(self.rope(self.kv_norm(self.wkv(fq(x)))[:, None], rc.pos).reshape(R, D))
            outs = []
            for s, c in enumerate(rc.ctxs):
                rows = c.rows[self.layer_idx]
                new = kv[s * L:(s + 1) * L][None]                                     # [1, L, D]
                full = (new if rows is None else mx.concatenate([rows, new], axis=1))[:, None].astype(q.dtype)
                qs = mx.contiguous(q[s * L:(s + 1) * L, :, 0].transpose(1, 0, 2)[None])   # [1, heads, L, D], the layout of the stock call
                outs.append(mx.fast.scaled_dot_product_attention(qs, full, full, scale=self.scale, mask=None, sinks=self.attn_sink.astype(q.dtype)))
            out = (outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)).transpose(0, 2, 1, 3).reshape(R, self.n_heads, 1, D)
            out = self.rope(out, rc.pos, inverse=True)
            out = out.reshape(R, self.o_groups, -1, 1, D).transpose(0, 1, 3, 2, 4).flatten(-2)
            out = self.wo_a(fq(out) if self.fq_out else out).transpose(0, 2, 1, 3).flatten(-2)
            return self.wo_b(fq(out))

        def __call__(self, x: mx.array, ctx: _Ctx) -> mx.array:
            if isinstance(ctx, _RowsCtx):
                return self._rows(x, ctx)
            B, L = x.shape[0], x.shape[1]
            p0 = ctx.p0
            qr = self.q_norm(self.wq_a(fq(x)))
            q = self.wq_b(fq(qr)).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            q = self.rope(q, p0)
            kv = self.kv_norm(self.wkv(fq(x)))
            kv = fq(self.rope(kv[:, None], p0).reshape(B, L, self.head_dim))
            rows = ctx.rows[self.layer_idx]
            full = kv if rows is None else mx.concatenate([rows, kv], axis=1)
            full = full[:, None].astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, full, full, scale=self.scale, mask=None, sinks=self.attn_sink.astype(q.dtype))
            out = self.rope(out, p0, inverse=True)
            out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
            out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
            out = self.wo_a(fq(out) if self.fq_out else out)
            out = out.transpose(0, 2, 1, 3).flatten(-2)
            res = self.wo_b(fq(out))
            if ctx.dbg is not None:
                ctx.dbg.append(res)
            return res

    return DSparkAttention


class DraftLM(nn.Module):
    """The parameters of the draft head: `stages` trunk blocks plus the entry (main_proj, main_norm), exit (norm) and heads."""

    def __init__(self, cfg, n_stages: int, n_taps: int, vocab: int, rank: int):
        super().__init__()
        lang = _classes()
        H = cfg.hidden_size
        self.layers = [_A.cls("block")(cfg, i, None) for i in range(n_stages)]
        att = _make_attention_class(lang)
        for blk in self.layers:
            blk.attn.__class__ = att
        self.main_proj = nn.Linear(n_taps * H, H, bias=False)
        self.main_norm = nn.RMSNorm(H, eps=cfg.rms_norm_eps)
        self.norm = nn.RMSNorm(H, eps=cfg.rms_norm_eps)
        self.markov_embed = nn.Embedding(vocab, rank)
        self.markov_head = nn.Linear(rank, vocab, bias=False)
        self.confidence = nn.Linear(H + rank, 1, bias=False)


class DraftRoot(nn.Module):
    """`patch_model` root: `language_model.layers[...]` like the trunk, so the same patcher loads the stages."""

    def __init__(self, lm: DraftLM):
        super().__init__()
        self.language_model = lm

    def sanitize(self, weights: dict) -> dict:
        for k in list(weights):
            if k.endswith(".ffn.gate.weight"):
                weights[k] = weights[k].astype(mx.float32)                    # router scores use fp32 (as the trunk's sanitize)
        return weights


def _cand_weight(lm):
    """The dense markov head weight [vocab, rank], or None (quantized / absent: the full-vocab chain runs)."""
    h = getattr(lm, "markov_head", None)
    w = getattr(h, "weight", None)
    if w is None or hasattr(h, "scales") or getattr(w, "ndim", 0) != 2:
        return None
    return w


class Drafter:
    """DSpark drafter bound to a trunk model. Greedy proposals; `append` feeds it the taps of committed positions."""

    def __init__(self, root: DraftRoot, trunk, cfg: dict, spec: ArchSpec, conf_threshold: float | None = None):
        ms = spec.mtp
        tc = cfg["text_config"]
        self.root, self.lm = root, root.language_model
        self.trunk_lm = trunk.language_model
        self.block = int(tc[ms.keys["block"]])
        self.noise = int(tc[ms.keys["noise"]])
        self.tap_ids = list(tc[ms.keys["taps"]])
        self.window = int(tc["sliding_window"])
        self.hidden = int(tc["hidden_size"])
        self.hc_mult = int(tc["hc_mult"])
        self.n_stages = len(self.lm.layers)
        self.conf_threshold = float(os.environ.get("KYOJIN_DSPARK_CONF", "0.5")) if conf_threshold is None else conf_threshold
        self.gate: float | None = None                                                       # per-round threshold from the engine (flag spec_gate); wins over the fixed one
        self.keep_margin = False                                                             # tests: keep the logits of the proposals in last_lg
        self.ctx = _Ctx(self.n_stages)
        self.lang = _classes()
        self.stats = dict(calls=0, conf_lens=[])
        self.fq_main = True                                 # main_proj is fp8 in the checkpoint: its input goes through the fp8 round trip (blocks of 32)
        self.debug: dict | None = None                    # set to {} to get the intermediate arrays of the next draft call (tools/draft_dump.py)
        self._warm = set()                                # compiled entry points that already ran once eagerly (lazily built kernel parameters evaluate there)
        self._gen = FL.GEN                                # kyojin_mlx.flags generation the compiled draft was built at (retraced when it moves)
        self._cfwd = RP.region(mx.compile(self._forward), "draft")           # KYOJIN_REPLAY=1: recorded once, replayed as an ICB (replay.py); else the compiled function itself
        self._capp = mx.compile(self._append_rows)
        self._rows_region = RP.region(self._rows_fn, "draft_rows")                          # draft_many (the serving path) with full rings
        self._rows_region_q = RP.region(self._rows_fn_q, "draft_rows_q")                    # ... and its block logits (KYOJIN_SPEC_RS)
        self._rows_region_r = RP.region(self._rows_fn_r, "draft_rows_refine")               # ... with the drafted prefix as ids (KYOJIN_DSPARK_REFINE)
        self._rows_region_rq = RP.region(self._rows_fn_rq, "draft_rows_refine_q")
        self._rows_c: dict = {}                           # flag draft_compile: name -> mx.compile'd rows function
        self._rows_c_off: set = set()                     # names whose trace failed: eager for good
        self._rows_c_warm: set = set()                    # (kind, rows) that ran once eagerly (lazily built kernel parameters evaluate there)
        self._rows_c_gen = FL.GEN

    # -- main-kv rows (one _Ctx per session: spec.stream swaps drafter.ctx in and out)
    def new_ctx(self) -> _Ctx:
        return _Ctx(self.n_stages)

    def reset(self) -> None:
        self.ctx.rows = [None] * self.n_stages

    def append(self, taps: list, start: int) -> None:
        """taps: one [S, H] array per target layer (committed positions start .. start + S - 1, in order). Appends one kv row per stage
        and position, keeps the last `window` rows."""
        S = taps[0].shape[0]
        rows = self.ctx.rows
        if COMPILE and self.debug is None and all(r is not None and r.shape[1] == self.window for r in rows):
            if ("append", S) in self._warm:
                STATS["call_append"] += 1
                self.ctx.rows = list(self._capp(mx.array(start, dtype=mx.int32), *taps, *rows))
                return
            self._warm.add(("append", S))
        STATS["eager_append"] += 1
        self.ctx.rows = self._append_rows(start, *taps, *rows)

    def _append_rows(self, start, *args) -> list:
        """The rows of every stage after the committed positions `start ..` (taps = the first n_taps args, the previous rows the rest; None = no rows yet)."""
        fq = self.lang.fake_quant_fp8_ue8m0
        lm = self.lm
        nt = len(self.tap_ids)
        taps, prev = args[:nt], args[nt:]
        if isinstance(start, mx.array):
            STATS["trace_append"] += 1
        S = taps[0].shape[0]
        th = mx.concatenate([t.astype(mx.float16) for t in taps], axis=-1)[None]          # [1, S, n_taps * H]
        main_x = lm.main_norm(lm.main_proj(fq(th) if self.fq_main else th))
        out = []
        for s, blk in enumerate(lm.layers):
            a = blk.attn
            kv = a.kv_norm(a.wkv(fq(main_x)))
            kv = fq(a.rope(kv[:, None], start).reshape(1, S, a.head_dim))
            rows = kv if prev[s] is None else mx.concatenate([prev[s], kv], axis=1)
            out.append(rows[:, -self.window:])
        return out

    # -- one draft block
    def draft(self, pending: int, p0: int, depth: int) -> list:
        """Up to `depth` proposed tokens after `pending` (the token at position p0); [] when the confidence gate says skip."""
        L = self.block
        ids = mx.array([[pending] + [self.noise] * (L - 1)])
        rows = self.ctx.rows
        if COMPILE and self.debug is None and all(r is not None and r.shape[1] == self.window for r in rows):
            if self._gen != FL.GEN:                       # a lever flag switched: rebuild the compiled draft (it keeps the switches it traced)
                self._gen = FL.GEN
                self._cfwd = RP.region(mx.compile(self._forward), "draft")
                self._capp = mx.compile(self._append_rows)
            if "draft" in self._warm:
                STATS["call_draft"] += 1
                toks, conf = self._cfwd(ids, mx.array(p0, dtype=mx.int32), *rows)
            else:
                self._warm.add("draft")
                toks, conf = self._eager(ids, p0)
        else:
            toks, conf = self._eager(ids, p0)
        mx.eval(toks, conf)
        if self.debug is not None:
            self.debug.update(self._dbg)
            mx.eval(*self.debug.values())
        thr = self.gate if self.gate is not None else (self.conf_threshold if CONF_OVERRIDE is None else CONF_OVERRIDE)
        keep = [float(c) >= thr for c in conf.tolist()]
        n = 0
        while n < len(keep) and keep[n]:
            n += 1
        self.stats["calls"] += 1
        self.stats["conf_lens"].append(n)
        return [int(t) for t in toks.tolist()[:min(depth, n)]]

    def _eager(self, ids, p0):
        STATS["eager_draft"] += 1
        self.ctx.p0 = p0
        self.ctx.dbg = [] if self.debug is not None else None
        toks, conf, x, biased, cl = self._core(ids, self.ctx)
        if self.debug is not None:
            self._dbg = dict(logits=mx.stack(biased, axis=1)[0], x=x[0], conf_logit=cl[0], ids=toks, attn=mx.stack([a[0] for a in self.ctx.dbg]))
        return toks, conf

    def _forward(self, ids, p0, *rows):
        """The compiled entry: ids [1, L], p0 scalar int32 array, the main-kv rows of every stage -> (tokens [L], confidences [L])."""
        STATS["trace_draft"] += 1
        ctx = _Ctx(self.n_stages)
        ctx.p0, ctx.rows = p0, list(rows)
        toks, conf, _, _, _ = self._core(ids, ctx)
        return toks, conf

    def _core(self, ids, ctx):
        """Block ids -> (tokens [L], confidences [L], pooled state, biased logits per position, confidence logits)."""
        lang, lm = self.lang, self.lm
        L = self.block
        pending = ids[0, 0:1]
        h = self.trunk_lm.embed_tokens(ids)
        h = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (1, L, self.hc_mult, h.shape[-1])))
        pre = lang.make_identity_pre_mix(1, L, self.hc_mult)
        for blk in lm.layers:
            h, pre = blk(h, pre, None, ctx)
        x = _A.hc_pre(h, pre)
        logits = self.trunk_lm.head(lm.norm(x)).astype(mx.float32)                          # [1, L, V]
        out, embs, biased = [pending], [], []
        for i in range(L):
            e = lm.markov_embed(out[-1])
            embs.append(e)
            lg_i = logits[:, i] + lm.markov_head(e).astype(mx.float32)
            biased.append(lg_i)
            out.append(mx.argmax(lg_i, axis=-1))
        cin = mx.concatenate([x.astype(mx.float32), mx.stack(embs, axis=1).astype(mx.float32)], axis=-1)
        cl = (cin @ lm.confidence.weight.astype(mx.float32).T)[..., 0]                    # [1, L] confidence logits
        conf = mx.sigmoid(cl)[0]                                                              # [L]
        return mx.concatenate(out[1:]), conf, x, biased, cl


    # -- several sessions at once (rows layout, see _RowsCtx)
    def _rows_cap(self) -> int:
        """Sessions per pass: the EXL3 products stay GEMV launches of a decode step (row invariant) up to ROW_SPLIT_MAX rows."""
        from . import exl3_linear as EL
        return max(1, max(EL.ROW_SPLIT_MAX, 4) // self.block)

    def _tail_rows(self, depths) -> int | None:
        """Flag draft_tail_cut: rows of the draft tail (the markov chain, the trunk head and the confidence head) this pass needs, = the deepest depth the
        round plans to verify, since a proposal past it is cut by the confidence gate and never verified. None = the whole block (flag off, or a fixed-shape
        region that traced the block)."""
        if not TAIL_CUT():
            return None
        d = max((int(x) for x in depths), default=0)
        return max(1, min(self.block, d))

    def _forward_rows(self, ctxs: list, pendings: list, p0s: list, nrows: int | None = None):
        """The block of every session through the stages in rows layout: (x [S * block, 1, H] after the collapse, logits [S, block, V] fp32)."""
        S, L = len(ctxs), self.block
        rc = _RowsCtx(ctxs, p0s, L)
        ids = mx.array([[p] + [self.noise] * (L - 1) for p in pendings]).reshape(S * L, 1)
        return self._forward_rows_ids(rc, ids, nrows)

    def _forward_rows_ids(self, rc: _RowsCtx, ids: mx.array, nrows: int | None = None):
        """_forward_rows from the block ids [S * block, 1] and the ready rows context. nrows < block: the trunk head runs on the first nrows rows of
        each session only (a row's logits come from its own state, so the rows the tail skips cannot move the ones it keeps)."""
        lang, lm = self.lang, self.lm
        R = ids.shape[0]
        S, L = R // self.block, self.block
        h = self.trunk_lm.embed_tokens(ids)                                                   # [R, 1, H]
        h = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (R, 1, self.hc_mult, h.shape[-1])))
        pre = lang.make_identity_pre_mix(R, 1, self.hc_mult)
        for blk in lm.layers:
            h, pre = blk(h, pre, None, rc)
        x = _A.hc_pre(h, pre)                                                    # [R, 1, H]
        n = L if nrows is None else max(1, min(L, int(nrows)))                                # flag draft_tail_cut: fewer rows through the trunk head
        if n < L:
            xr = mx.contiguous(x.reshape(S, L, -1)[:, :n]).reshape(S * n, 1, x.shape[-1])
            return x, self.trunk_lm.head(lm.norm(xr)).astype(mx.float32).reshape(S, n, -1)
        return x, self.trunk_lm.head(lm.norm(x)).astype(mx.float32).reshape(S, L, -1)

    def _couple_noise(self, coupling: list, S: int, rows: int):
        """Flag spec_couple: [S, rows, V] fp32 noise added to the draft logits before the chain's argmax, = scale x Gumbel(V; key_i) for a coupled session
        (the exact array `mx.random.categorical(logits, key=key_i)` adds to the target's tempered log-probabilities: categorical is argmax(x + gumbel(key)))
        and 0 for the others and for rows past the keys it was given. None when the pass cannot take it (draft_cand, no coupled session)."""
        if CAND > 0 or not any(coupling) or len(coupling) != S:
            return None
        V = int(self.lm.markov_head.weight.shape[0])
        out = []
        for c in coupling:
            if c is None:
                out.append(mx.zeros((rows, V), dtype=mx.float32))
                continue
            scale, keys = c
            g = [scale * mx.random.gumbel((V,), key=keys[i]) if i < len(keys) else mx.zeros((V,), dtype=mx.float32) for i in range(rows)]
            out.append(mx.stack(g).astype(mx.float32))
        FD.hit("spec_couple")
        return mx.stack(out)

    def _rows_core(self, ctxs: list, pendings: list, p0s: list, keep: bool = False, nrows: int | None = None, noise=None):
        """One draft pass in rows layout: (tokens [S, L], confidences [S, L], logits of every proposal [S, L, V] or None)."""
        S, L = len(ctxs), self.block
        x, logits = self._forward_rows(ctxs, pendings, p0s, nrows)
        out = self._rows_tail(x, logits, mx.array(pendings), S, keep, nrows=nrows, noise=noise)
        if REFINE and not self.keep_margin and noise is None:
            STATS["refine_pass"] += 1                                                 # flag dspark_refine actually ran (path check, /v1/stats engage)
            ids2 = mx.concatenate([mx.array(pendings, dtype=mx.int32)[:, None], out[0][:, :L - 1]], axis=1)
            x2, logits2 = self._forward_rows_ids(_RowsCtx(ctxs, p0s, L), ids2.reshape(S * L, 1), nrows)
            out = self._rows_tail(x2, logits2, ids2[:, 0], S, keep, ids=ids2, nrows=nrows)
        return out

    def _rows_tail(self, x, logits, pend, S: int, keep: bool = False, ids: mx.array | None = None, nrows: int | None = None, noise=None):
        """Markov-biased proposals from the block states: ids = the block's input tokens [S, L] (teacher forced: the bigram bias comes from
        the token each position sees, as in the reference block) or None (the greedy chain from `pend`, the stock block)."""
        lm = self.lm
        L = self.block
        nr = L if logits.shape[1] >= L else int(logits.shape[1])                              # rows the head produced (draft_tail_cut may be fewer than the block)
        out, embs, lgs = [pend], [], []
        W = _cand_weight(lm) if CAND > 0 and not keep and ids is None else None
        if W is not None and CAND < W.shape[0]:
            FD.hit("draft_cand")
        if W is not None and CAND < W.shape[0]:                                               # flag draft_cand: the chain reads K rows of the markov head, not all of them
            STATS["cand_tail"] += 1
            cand = mx.argpartition(-logits, CAND - 1, axis=-1)[..., :CAND]                    # [S, L, K] the K best base logits of every position (one pass)
            base = mx.take_along_axis(logits, cand, axis=-1)
            for i in range(nr):
                e = lm.markov_embed(out[-1])
                embs.append(e)
                bias = (W[cand[:, i]] * e[:, None, :]).sum(-1).astype(mx.float32)             # [S, K] = markov_head(e) at the candidates
                j = mx.argmax(base[:, i] + bias, axis=-1)
                out.append(mx.take_along_axis(cand[:, i], j[:, None], axis=-1)[:, 0])
            return self._tail_conf(x, embs, out, S, nrows=nr)
        from .rowinv import rowmm
        for i in range(nr):
            e = lm.markov_embed(ids[:, i] if ids is not None else out[-1])                    # [S, rank]
            embs.append(e)
            bias = lm.markov_head(e[:, None])[:, 0].astype(mx.float32)                        # row-invariant product
            lg_i = logits[:, i] + bias
            out.append(mx.argmax(lg_i if noise is None else lg_i + noise[:, i], axis=-1))      # spec_couple: the target's own Gumbel noise picks the proposal
            if keep:                                                                          # tests: the logits of every proposal
                lgs.append(lg_i)
        toks, conf, _ = self._tail_conf(x, embs, out, S, nrows=nr)
        return toks, conf, (mx.stack(lgs, axis=1) if lgs else None)

    def _tail_conf(self, x, embs, out, S: int, nrows: int | None = None):
        """(tokens [S, L], confidences [S, L], None) from the chain's tokens and bigram embeddings. nrows < L: the chain ran that many positions and the
        rest of the block is padded with 0 (below every threshold, and past the planned depth: `conf_keep` caps a session at its planned depth)."""
        from .rowinv import rowmm
        L = self.block
        nr = L if nrows is None else max(1, min(L, int(nrows)))
        xr = x.reshape(S, L, -1)[:, :nr] if nr < L else x.reshape(S, L, -1)
        cin = mx.concatenate([mx.contiguous(xr).astype(mx.float32), mx.stack(embs, axis=1).astype(mx.float32)], axis=-1)
        if getattr(self, "_conf_w", None) is None:
            self._conf_w = self.lm.confidence.weight.astype(mx.float32)
        conf = mx.sigmoid(rowmm(cin.reshape(S * nr, 1, -1), self._conf_w)[:, 0, 0]).reshape(S, nr)
        toks = mx.stack(out[1:], axis=1)                                                      # [S, nr]
        if nr < L:
            zt = mx.zeros((S, L - nr), dtype=toks.dtype)
            toks = mx.concatenate([toks, zt], axis=1)
            conf = mx.concatenate([conf, mx.zeros((S, L - nr), dtype=conf.dtype)], axis=1)
        return toks, conf, None

    def _rows_run(self, ids, pend, pos, rings, teacher: bool = False, keep: bool = False, nrows: int | None = None):
        """One draft pass from the block ids [S * L, 1] and the rings (the pure body of the serving-path regions)."""
        ns, L = self.n_stages, self.block
        S = ids.shape[0] // L
        ctxs = []
        for s in range(S):
            c = _Ctx(ns)
            c.rows = list(rings[s * ns:(s + 1) * ns])
            ctxs.append(c)
        x, logits = self._forward_rows_ids(_RowsCtx(ctxs, None, L, pos=pos), ids, nrows)
        return self._rows_tail(x, logits, pend, S, keep=keep, ids=(ids.reshape(S, L) if teacher else None), nrows=nrows)

    def _rows_fn(self, ids, pend, pos, *rings, nrows: int | None = None):
        """draft_many for full rings as a pure function of arrays (fixed shapes): ids [S * L, 1], pending tokens [S], positions [S * L], the rings session after
        session (n_stages arrays each) -> (tokens [S, L], confidences [S, L])."""
        return self._rows_run(ids, pend, pos, rings, nrows=nrows)[:2]

    def _rows_fn_q(self, ids, pend, pos, *rings, nrows: int | None = None):
        """The same pass plus the biased logits [S, L, V] of every position (KYOJIN_SPEC_RS draws and verifies from them)."""
        return self._rows_run(ids, pend, pos, rings, keep=True, nrows=nrows)

    def _rows_fn_r(self, ids, pend, pos, *rings, nrows: int | None = None):
        """The refine pass: ids are the first pass's proposals (the head sees its own draft, not noise) and the bigram bias is teacher forced."""
        return self._rows_run(ids, pend, pos, rings, teacher=True, nrows=nrows)[:2]

    def _rows_fn_rq(self, ids, pend, pos, *rings, nrows: int | None = None):
        return self._rows_run(ids, pend, pos, rings, teacher=True, keep=True, nrows=nrows)

    def draft_many(self, ctxs: list, pendings: list, p0s: list, depths: list, want_logits: bool = False, coupling: list | None = None) -> list:
        """One draft pass for all sessions: up to depths[s] proposed tokens after pendings[s] (the token at position p0s[s]) from the ring
        ctxs[s]. Row-invariant: the proposal of a session is the same whatever the other sessions are (it equals draft_many of that session alone).
        want_logits: also keep the biased block logits of the last pass (self.last_lg, KYOJIN_SPEC_RS).
        coupling (flag spec_couple): per session None or (noise scale, keys): the chain picks argmax(logits + scale x Gumbel(key_i)) at position i (see `_couple_noise`)."""
        cap = self._rows_cap()
        if len(ctxs) > cap:
            return [t for i in range(0, len(ctxs), cap) for t in self.draft_many(ctxs[i:i + cap], pendings[i:i + cap], p0s[i:i + cap], depths[i:i + cap],
                                                                                   want_logits, coupling[i:i + cap] if coupling else None)]
        S, L = len(ctxs), self.block
        lgs = None
        nr = None if want_logits else self._tail_rows(depths)                                 # draft_tail_cut: the rows this pass will verify (None = the block)
        rings = [r for c in ctxs for r in c.rows]
        noise = self._couple_noise(coupling, S, L if nr is None else nr) if coupling and not want_logits else None   # flag spec_couple: eager pass with the target's noise in the chain
        if noise is not None:
            toks, conf, lgs = self._rows_core(ctxs, pendings, p0s, keep=False, nrows=nr, noise=noise)
            mx.eval(toks, conf)
        elif self.keep_margin or any(r is None or r.shape[1] != self.window for r in rings):    # tests (the logits of every proposal) and rings not full yet (prompts shorter than the window): eager, a region needs fixed shapes
            toks, conf, lgs = self._rows_core(ctxs, pendings, p0s, keep=self.keep_margin or want_logits, nrows=nr)
            mx.eval(toks, conf, *([lgs] if lgs is not None else []))
        else:                                                                                  # fixed shapes: one region per number of sessions and ring shapes (replay.py)
            toks, conf, lgs = self._rows_launch(ctxs, pendings, p0s, want_logits, nr)
            mx.eval(toks, conf, *([lgs] if lgs is not None else []))
        toks, conf = toks.tolist(), conf.tolist()
        self.last_lg = lgs
        keep = self.conf_keep(toks, conf, depths)
        return [[int(t) for t in toks[s][:keep[s]]] for s in range(S)]

    def draft_many_chain(self, ctxs: list, pendings: list, p0s: list, depths: list, chain: int = 2) -> list:
        """`chain` block forwards per session (flag spec_chain): pass 2 starts from pass 1's last proposal at position p0 + its length. A pass after the
        first runs only when the pass before it kept every row it asked for, so the confidence gate still decides where a run stops; the proposals are
        the stock block's (rows = sessions x block, one pass each). The block attention still reads the ring the last commit left: the tokens drafted
        between the ring's tail and the pass's position are not in it. The verify decides every emitted token."""
        S, L = len(ctxs), self.block
        todo = [{"ctx": ctxs[s], "pend": int(pendings[s]), "p0": int(p0s[s]), "total": max(0, int(depths[s])),
                 "want": max(0, min(int(depths[s]), L)), "got": []} for s in range(S)]
        for _ in range(max(1, int(chain))):
            live = [t for t in todo if t["want"] > 0]
            if not live:
                break
            got = self.draft_many([t["ctx"] for t in live], [t["pend"] for t in live], [t["p0"] for t in live],
                                  [t["want"] for t in live])
            for t, g in zip(live, got):
                asked = t["want"]
                t["got"].extend(int(x) for x in g)
                if g:
                    t["pend"], t["p0"] = int(g[-1]), t["p0"] + len(g)
                t["want"] = 0 if len(g) < asked else min(L, t["total"] - len(t["got"]))
        return [t["got"] for t in todo]

    def _rows_launch(self, ctxs: list, pendings: list, p0s: list, want_logits: bool = False, nrows: int | None = None):
        """The rows draft region for full rings: (block proposals [S, block], confidences [S, block][, biased logits [S, block, V]]), not evaluated.
        nrows (flag draft_tail_cut): rows of the tail the pass has to produce."""
        S, L = len(ctxs), self.block
        ids = mx.array([[p] + [self.noise] * (L - 1) for p in pendings]).reshape(S * L, 1)
        pos = mx.array([p + i for p in p0s for i in range(L)], dtype=mx.int32)
        pend = mx.array(pendings)
        rings = [r for c in ctxs for r in c.rows]
        out = self._rows_call("q" if want_logits else "", ids, pend, pos, rings, nrows)     # pass 1 is the stock greedy chain, as in the eager _rows_core
        if REFINE:
            FD.hit("dspark_refine")
            ids2 = mx.concatenate([pend[:, None], out[0][:, :L - 1]], axis=1).reshape(S * L, 1)   # the head's own draft as block ids
            out = self._rows_call("rq" if want_logits else "r", ids2, pend, pos, rings, nrows)
        return out if want_logits else (out[0], out[1], None)

    _ROWS_FN = {"": ("_rows_region", "_rows_fn"), "q": ("_rows_region_q", "_rows_fn_q"),
                "r": ("_rows_region_r", "_rows_fn_r"), "rq": ("_rows_region_rq", "_rows_fn_rq")}

    def _rows_call(self, kind: str, ids, pend, pos, rings, nrows: int | None = None):
        """One rows draft pass through the serving-path region of `kind` ("" stock, q = with logits, r = refine, rq = both).
        Flag draft_compile: the pure function `_rows_fn*` runs as ONE mx.compile region (as the solo draft does), so the per-round python graph build
        and the unfused elementwise launches of the block go away. Same values as the eager function (compile fuses, it does not re-round), checked
        bitwise by tests/test_draft_compile.py. The first call of a kind runs eagerly (lazily built kernel parameters evaluate there), a lever flag
        switch retraces (flags.GEN), a trace error leaves that kind eager for good, and replay (KYOJIN_REPLAY) keeps its own region."""
        reg, fn = self._ROWS_FN[kind]
        n = self.block if nrows is None else max(1, min(self.block, int(nrows)))
        if n < self.block:                                                                   # draft_tail_cut: fewer tail rows = its own shapes, so eager (the
            return getattr(self, reg)(ids, pend, pos, *rings, nrows=n)                        # compiled region is one fixed-shape trace per kind)
        if not DRAFT_COMPILE or RP.ENABLED or kind in self._rows_c_off:
            return getattr(self, reg)(ids, pend, pos, *rings)
        if self._rows_c_gen != FL.GEN:                                                       # a lever flag switched: the compiled draft keeps the switches it traced
            self._rows_c_gen = FL.GEN
            self._rows_c.clear()
        warm = (kind, int(ids.shape[0]))                                                      # a new session count has kernel parameters of its own to build
        if warm not in self._rows_c_warm:
            self._rows_c_warm.add(warm)
            return getattr(self, reg)(ids, pend, pos, *rings)
        c = self._rows_c.get(kind)
        if c is None:
            c = self._rows_c[kind] = mx.compile(getattr(self, fn))
        try:
            out = c(ids, pend, pos, *rings)
        except Exception:                                                                   # not traceable (a host read inside a block): eager from here on
            STATS["rows_compile_fail"] += 1
            self._rows_c.pop(kind, None)
            self._rows_c_off.add(kind)
            return getattr(self, reg)(ids, pend, pos, *rings)
        STATS["rows_compiled"] += 1
        FD.hit("draft_compile")
        return tuple(out) if isinstance(out, (list, tuple)) else out

    def draft_many_lazy(self, ctxs: list, pendings: list, p0s: list, depths: list, want_logits: bool = False):
        """draft_many without the wait: the raw block proposals and confidences stay on the GPU (async_eval started), the caller cuts them with
        `conf_keep` once they are read. None when draft_many would not run the fixed-shape region (split into passes, rings not full, debug).
        `depths` is the plan of every session, exactly as draft_many takes it: flag draft_tail_cut runs the tail on the deepest planned depth, so the
        lazy pass needs the same number (it was passed as a local of another function and raised NameError on the first lazy round ever taken).
        want_logits: the biased logits [S, block, V] of the drafted rows come along (KYOJIN_SPEC_RS)."""
        if len(ctxs) > self._rows_cap() or self.keep_margin or any(r is None or r.shape[1] != self.window for c in ctxs for r in c.rows):
            return None
        out = self._rows_launch(ctxs, pendings, p0s, want_logits, None if want_logits else self._tail_rows(depths))
        mx.async_eval(*out)
        return out

    def conf_keep(self, toks: list, conf: list, depths: list) -> list:
        """Proposals kept per session: the leading run above the confidence threshold, at most the planned depth. `self.gate` (flag spec_gate) is a
        per-round threshold from the engine's measured P(accept | conf); the fixed 50 % is the fallback."""
        self.last_raw = (toks, conf)                                                          # the raw block proposals and confidences (tests)
        thr = self.gate if self.gate is not None else (self.conf_threshold if CONF_OVERRIDE is None else CONF_OVERRIDE)
        if self.gate is None and CONF_OVERRIDE is not None:
            FD.hit("dspark_conf_pct")
        keep = []
        for s in range(len(toks)):
            n = 0
            while n < self.block and conf[s][n] >= thr:
                n += 1
            self.stats["calls"] += 1
            self.stats["conf_lens"].append(n)
            keep.append(min(depths[s], n))
        return keep

    def append_many(self, items: list) -> None:
        """items: [(ctx, taps, start)]: for each session the taps (one [S_k, H] array per target layer) of committed positions start ..
        start + S_k - 1. Row-wise products (a row's kv does not depend on the other rows), so the ring bits do not depend on how positions are
        grouped into calls."""
        from .rowinv import rowmm
        cap = max(4, self._rows_cap() * self.block)
        groups, cur, n = [], [], 0
        for it in items:                                                                      # sessions per call: rows <= the GEMV limit
            k = it[1][0].shape[0]
            if cur and n + k > cap:
                groups.append(cur)
                cur, n = [], 0
            cur.append(it)
            n += k
        if cur:
            groups.append(cur)
        fq = self.lang.fake_quant_fp8_ue8m0
        lm = self.lm
        for grp in groups:
            counts = [it[1][0].shape[0] for it in grp]
            taps = [mx.concatenate([it[1][t].astype(mx.float16) for it in grp], axis=0) if len(grp) > 1 else grp[0][1][t].astype(mx.float16)
                    for t in range(len(grp[0][1]))]
            th = mx.concatenate(taps, axis=-1)[:, None]                                       # [rows, 1, n_taps * H]
            pos = mx.array([it[2] + i for it, c in zip(grp, counts) for i in range(c)], dtype=mx.int32)
            main_x = lm.main_norm(lm.main_proj(th))                                           # [rows, 1, H]
            for s, blk in enumerate(lm.layers):
                a = blk.attn
                kv = fq(a.rope(a.kv_norm(a.wkv(fq(main_x)))[:, None], pos).reshape(-1, a.head_dim))
                o = 0
                for (ctx, _, _), c in zip(grp, counts):
                    new = kv[o:o + c][None]
                    o += c
                    rows = new if ctx.rows[s] is None else mx.concatenate([ctx.rows[s], new], axis=1)
                    ctx.rows[s] = rows[:, -self.window:]

    def warm_replay(self) -> int:
        """KYOJIN_REPLAY=1: record the regions of the serving path (draft_many for 1..cap sessions on full rings) now, with throw-away random inputs, so no request pays the
        ~0.3 s recording of a new signature. Returns the number of draft calls made (0 when replay is off)."""
        if not RP.ENABLED or self.debug is not None:
            return 0
        saved = (dict(self.stats), list(self.stats["conf_lens"]), getattr(self, "last_raw", None), getattr(self, "last_lg", None), self.ctx)
        H = self.hidden
        seed = [0]

        def ring():                                                                        # a full ring of its own buffers (inputs that alias would be bound to one another),
            taps = []                                                                      # realistic values: a degenerate input (all-zero rows, NaN fp8 scales) can take paths real data never takes
            for _ in self.tap_ids:
                seed[0] += 1
                taps.append(mx.random.normal((self.window, H), key=mx.random.key(seed[0])).astype(mx.float16))
            r = self._append_rows(0, *taps, *[None] * self.n_stages)
            mx.eval(*r)
            return list(r)

        n = 0
        for S in range(1, self._rows_cap() + 1):
            ctxs = []
            for _ in range(S):
                c = self.new_ctx()
                c.rows = ring()
                ctxs.append(c)
            for _ in range(RP.RECORD_AFTER):
                self.draft_many(ctxs, [1] * S, [self.window] * S, [1] * S)
                n += 1
        self.stats.clear()
        self.stats.update(saved[0])
        self.stats["conf_lens"] = saved[1]
        self.last_raw, self.last_lg, self.ctx = saved[2], saved[3], saved[4]
        return n


def build_drafter(trunk, pack_dir: str, spec: ArchSpec, cfg: dict, over: dict | None = None, conf_threshold: float | None = None) -> Drafter:
    """Load the draft head of `pack_dir` (tensors `mtp.<stage>.*`, stacked by tools/convert_stack.py) for the patched `trunk` model.
    over: text_config overrides of the drafter (tests)."""
    assert spec.mtp is not None, f"{spec.name}: no draft head described in its ArchSpec"
    ms, tc = spec.mtp, cfg["text_config"]
    n = int(tc.get(ms.keys["stages"], 0))
    assert n > 0, f"{pack_dir}: config has no draft stages ({ms.keys['stages']} = 0)"
    dcfg = copy.deepcopy(cfg)
    dtc = dcfg["text_config"]
    dtc.update(num_hidden_layers=n, compress_ratios=[0] * n, kv_source_layer_ids=[], index_source_layer_ids=[], candidate_source_layer_id=-1,
               engram_layer_ids=[], engram_num_embeddings=[], n_routed_experts=int(tc[ms.keys["experts"]]),
               num_experts_per_tok=int(tc[ms.keys["topk"]]))
    dtc.update(over or {})
    mod = _A.package(spec)
    mcfg = mod.ModelConfig.from_dict(dcfg)
    root = DraftRoot(DraftLM(mcfg, n, len(tc[ms.keys["taps"]]), int(tc["vocab_size"]), int(tc[ms.keys["rank"]])))
    P.patch_model(root, pack_dir, mtp_arch(spec), dcfg, prefix=ms.prefix)
    return Drafter(root, trunk, cfg, spec, conf_threshold)
