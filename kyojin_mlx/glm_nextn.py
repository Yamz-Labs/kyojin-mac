"""GLM-5.3 native MTP (nextn, checkpoint layer 45) as a draft head for GlmBackend.

The block is mlx-vlm's Glm5NextMTPDraftModel (mlx-vlm, MIT licence: mlx_vlm/speculative/drafters/glm5_next_mtp): enorm / hnorm / eh_proj, one DSA-attention + MoE
block, shared_head_norm, the target's embedding and lm_head. Its weights come from the pack (foreign_linear.load_foreign(mtp=True): dense linears decoded to fp16,
routed experts kept as stored NVFP4). What differs from mlx-vlm's drafter: the state is per session (GlmDraft), so the engine's epochs (sessions joining, leaving,
restored from the prefix cache) never have to filter a shared batched drafter.

Pair j = (token at position j + 1, target hidden after position j) -> the head's logits are the token at position j + 2. A session's GlmDraft holds the hidden
states of the positions whose pair is not fed yet (`hid`, with the tokens at those positions `htok`); the pair of the last one needs the pending token, which only
the round knows. `draft` feeds the pairs (real positions, kept in the head's cache), then chains the head on its own output for the next tokens under a
speculative transaction that is aborted (those positions are re-fed with the target's real hidden states once the round is verified).
"""
from __future__ import annotations

import json
import os
import re
import time

import mlx.core as mx
import mlx.nn as nn

from . import verify_glue as GV

CHUNK = 512                                       # pairs fed per head forward
MTPFEED = os.environ.get("KYOJIN_GLM_MTPFEED", "0") not in ("0", "")        # flag glm_mtpfeed: prompt feeds keep no projected K / V cache (see _NoProj); read per call
Q_MIN_PARAMS = 1 << 20                            # flag glm_head_q: a linear below this many weights stays fp16 (indexer weights_proj, router: no bytes to win)


def _env_int(name: str) -> int:
    try:
        return max(0, int(os.environ.get(name, "0") or 0))
    except ValueError:
        return 0


def head_mode() -> tuple:
    """The draft-head variant asked for NOW (flags glm_head_q / draftvocab / glm_draft_topk, read per draft call):
    (quant name, '' = fp16 head; draft vocabulary rows K, 0 = the target's full lm_head; routed experts per token, 0 = the checkpoint's)."""
    q = os.environ.get("KYOJIN_GLM_HEAD_Q", "0") or "0"
    q = "" if q in ("0", "") else q
    return q, _env_int("KYOJIN_DRAFT_VOCAB"), _env_int("KYOJIN_GLM_DRAFT_TOPK")


class _Prof:
    """Flag glm_head_prof: per-op wall ms of the draft head, one forced eval per op (the totals are NOT the served step: every op waits for the one before).
    Written to KYOJIN_GLM_HEAD_PROF_OUT (json: op -> [total ms, calls]) every 20 draft calls."""

    def __init__(self):
        self.acc: dict = {}
        self.last = 0.0
        self.calls = 0

    def begin(self):
        mx.synchronize()
        self.last = time.perf_counter()

    def t(self, name, *arrays):
        mx.eval(*arrays)
        now = time.perf_counter()
        a = self.acc.setdefault(name, [0.0, 0])
        a[0] += (now - self.last) * 1e3
        a[1] += 1
        self.last = now

    def dump(self):
        self.calls += 1
        if self.calls % 20 == 0:
            out = os.environ.get("KYOJIN_GLM_HEAD_PROF_OUT") or os.path.expanduser("/path/to/data")
            json.dump({"calls": self.calls, "ops": {k: [round(v[0], 3), v[1]] for k, v in self.acc.items()}}, open(out, "w"), indent=1)


def _bytes(d) -> int:
    from mlx.utils import tree_flatten
    return int(sum(a.nbytes for _, a in tree_flatten(d.parameters())))


class _NoProj:
    """The head's cache list as the stock attention sees it, minus the projected K / V cache (slot 3).

    Prompt feed (P2-MTPFEED): mlx_vlm's `Glm5NextAttention._attend` keeps, up to `_MAX_PROJECTED_PREFILL_TOKENS` = 32768 tokens, a per-head projected K / V cache
    (64 heads x T x 256, grown by `update_and_fetch` = a full copy of up to GBs per 512-row piece) and `GlmDraft.arrays()` makes the engine evaluate it. A prompt
    feed discards the head's output (the head is ONE layer: its caches, the latent, the indexer's key-valid and the pool keys, are written from the layer INPUT, never from
    its attention output), so that cache is pure cost: with slot 3 None the stock code takes the latent path (what it does beyond 32768 anyway), the output graph is
    never evaluated and the three caches that matter come out bit-identical. A later `draft` call with the projected cache empty (size 0 != previous length) also takes
    the latent path, so nothing reads a stale projection."""
    __slots__ = ("cl",)

    def __init__(self, cl):
        self.cl = cl

    def __getitem__(self, i):
        return None if i == 3 else self.cl[i]


class GlmDraft:
    """Per-session draft state: the head's cache plus the target hidden states not yet turned into pairs."""

    def __init__(self):
        self.cache = None
        self.pos = None                           # position of the first pair not yet fed (hid[0]'s position)
        self.hid = None                           # [m, H]: target hidden after positions pos .. pos + m - 1
        self.htok = None                          # [m] int32: the tokens at those positions
        self.variant = None                       # head variant (NextN.mode) whose weights wrote `cache`: a head cache is never read by another variant

    @property
    def m(self) -> int:
        return 0 if self.hid is None else int(self.hid.shape[0])

    def append(self, h, toks, start: int) -> None:
        """h [n, H] hidden after positions start .. start + n - 1, toks [n] their input tokens. A gap (prefix-cache restore, dropped rows) restarts the head's cache."""
        toks = toks.astype(mx.int32)
        if self.pos is not None and start == self.pos + self.m:
            if self.hid is None:
                self.hid, self.htok = h, toks
            else:
                self.hid, self.htok = mx.concatenate([self.hid, h], axis=0), mx.concatenate([self.htok, toks], axis=0)
            return
        self.cache, self.pos, self.hid, self.htok = None, int(start), h, toks

    def arrays(self) -> list:
        out = [a for a in (self.hid, self.htok) if a is not None]
        for c in self.cache or []:
            for sub in c.caches:
                st = getattr(sub, "state", None)
                if st is not None:
                    out += [a for a in (st if isinstance(st, (list, tuple)) else [st]) if isinstance(a, mx.array)]
        return out


def _stored_linears(packs: dict) -> dict:
    """{drafter module path: ForeignLinear} for the head's stored 4-bit Packs (HF prefix -> Pack): the drafter fuses q_a + kv_a (qkv_a_proj) and gate + up (gate_up_proj)
    like mlx-vlm's sanitize does, rows in that order."""
    if not packs:
        return {}
    from . import foreign_linear as FL
    by = {re.sub(r"^.*?layers\.\d+\.", "", k): v for k, v in packs.items()}
    out = {}
    for tail, pk in by.items():
        if tail == "eh_proj":
            out["eh_proj"] = FL.ForeignLinear(pk)
        elif tail in ("self_attn.q_b_proj", "self_attn.o_proj", "mlp.shared_experts.down_proj"):
            out["mtp_block." + tail] = FL.ForeignLinear(pk)
    for fused, (a, b) in (("self_attn.qkv_a_proj", ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa")), ("mlp.shared_experts.gate_up_proj", ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"))):
        if a in by and b in by:
            out["mtp_block." + fused] = FL.make_linear(FL._concat_rows_exact([by[a], by[b]]))
    return out


class NextN:
    """The head + its draft-only variants. Variants are built on first use and cost no load time when their flag stays off:
    glm_head_q  = the head's dense linears (eh_proj, attention, indexer, shared expert) re-quantised 4 / 8 bit from the fp16 copy the head holds
                  (a second module tree sharing the routed experts and every array it does not quantise; the target never sees it);
    draftvocab (KYOJIN_DRAFT_VOCAB = K, kyojin_mlx.draftvocab, shared with the DS4.1 / Qwen drafts): the draft's argmax over the K most frequent tokens' lm_head rows,
                  a row view of the stored fp8 pack (no copy), mapped back to token ids;
    glm_draft_topk  = K: the draft routes to K experts instead of 8 (the head's expert reads are 8 of 288 per step: 0.11 GB, so this is a probe, not a lever).
    The verify pass is untouched, so none of them can change an output: they only move the acceptance and the draft ms."""

    def __init__(self, lm, drafter, cfg=None):
        self.lm, self.d = lm, drafter
        self.cfg = cfg                                      # Glm5NextMTPConfig the head was built from (variants clone it)
        self.embed = lm.model.embed_tokens
        self._vars: dict = {}                               # (quant, topk) -> drafter
        self.pack = None                                    # the model path (draftvocab reads its tokenizer.json)
        self.mode = ("", 0, 0)
        self.info: dict = {}

    # ---- variants
    def _clone(self):
        from mlx.utils import tree_flatten
        from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
        d = Glm5NextMTPDraftModel(self.cfg)
        d.load_weights(list(tree_flatten(self.d.parameters())), strict=False)
        d.mtp_block.mlp.switch_mlp = self.d.mtp_block.mlp.switch_mlp          # the routed experts are shared (one stored copy)
        d.eval()
        return d

    def drafter(self, q: str, topk: int):
        """The head module for this variant (the checkpoint's own for ('', 0))."""
        if not q and not topk:
            return self.d
        key = (q, topk)
        d = self._vars.get(key)
        if d is None:
            d = self._clone()
            if topk:
                d.mtp_block.mlp.gate.top_k = int(topk)
            if q:
                bits, mode, gs = {"4": (4, "affine", 64), "8": (8, "affine", 64), "nvfp4": (4, "nvfp4", 16), "mxfp4": (4, "mxfp4", 32)}[q]
                n0 = _bytes(d)
                nn.quantize(d, group_size=gs, bits=bits, mode=mode,
                            class_predicate=lambda path, m: isinstance(m, nn.Linear) and m.weight.size >= Q_MIN_PARAMS and m.weight.shape[-1] % gs == 0)
                self.info[f"head_q{q}_bytes"] = (n0, _bytes(d))
            mx.eval(d.parameters())
            self._vars[key] = d
        return d

    def vocab_head(self, k: int):
        """draftvocab.Sub over the K most frequent rows of the target's lm_head (a row view of the stored pack), or None (no tokenizer.json next to the pack / unknown head)."""
        from . import draftvocab as DV
        head = getattr(self.lm, "lm_head", None)
        if head is None:
            return None
        V = int(getattr(head, "out_features", 0) or self.embed.weight.shape[0])
        K = DV.size(V) if k else 0
        return DV.sub_head(self, head, K, V, self.pack) if K else None

    def select(self) -> tuple:
        """Read the three flags once per draft call; returns (drafter, vocab head or None)."""
        q, nv, tk = self.mode = head_mode()
        return self.drafter(q, tk), (self.vocab_head(nv) if nv else None)

    @staticmethod
    def build(model, cfg: dict, head: dict, switch, packs: dict | None = None) -> "NextN":
        """head: HF-named fp16 tensors of layer >= num_hidden_layers (load_foreign take_mtp); switch: its ForeignSwitchGLU.
        packs (flag glm_dense4): HF prefix -> stored 4-bit Pack of the head's dense linears; each becomes a ForeignLinear in place of the fp16 nn.Linear."""
        from mlx_vlm.speculative.drafters.glm5_next_mtp.config import Glm5NextMTPConfig
        from mlx_vlm.speculative.drafters.glm5_next_mtp.glm5_next_mtp import Glm5NextMTPDraftModel
        tc = dict(cfg.get("text_config") or cfg)
        d = Glm5NextMTPDraftModel(Glm5NextMTPConfig.from_dict({"text_config": tc}))
        w = {}
        for k, v in head.items():                                 # same renaming as mlx-vlm's Glm5NextMTPSplitter.rename
            k = re.sub(r"^.*?layers\.\d+\.", "", k)
            if k.startswith(("enorm.", "hnorm.", "eh_proj.")):
                w[k] = v
            elif k.startswith("shared_head.norm."):
                w["shared_head_norm." + k[len("shared_head.norm."):]] = v
            else:
                w["mtp_block." + k] = v
        w = d.sanitize(w)
        from mlx.utils import tree_flatten
        swap = _stored_linears(packs or {})
        missing = sorted(k for k, _ in tree_flatten(d.parameters()) if k not in w and ".switch_mlp." not in k and k.rsplit(".", 1)[0] not in swap)
        if missing:
            raise ValueError(f"GLM-5.3 nextn head: no tensor for {len(missing)} parameter(s), e.g. {missing[:4]}")
        d.load_weights(list(w.items()), strict=False)
        if swap:
            from mlx.utils import tree_unflatten
            d.update_modules(tree_unflatten(list(swap.items())))
        d.mtp_block.mlp.switch_mlp = switch
        d.eval()
        return NextN(model.language_model, d, cfg=Glm5NextMTPConfig.from_dict({"text_config": tc}))

    def _logits_argmax(self, h, voc=None):
        lm = self.lm
        if voc is not None:                                                 # draft-only vocabulary: argmax over the ranked rows, mapped back to ids
            return voc.pick(voc(h))
        if hasattr(lm, "lm_head"):
            from mlx_vlm.models.glm5_next.language import linear
            lg = linear(lm.lm_head, h)
        else:
            lg = self.embed.as_linear(h)
        return mx.argmax(lg, axis=-1)

    def _fwd(self, cache, toks, hid, pos0: int | None, last_only: bool, d=None, noproj: bool = False):
        """toks [1, n], hid [1, n, H] -> shared-head-normed hidden [1, n or 1, H]. pos0: true position of the first pair (the embedding at position 0 is zeroed)."""
        d = d or self.d
        e = self.embed(toks)
        if pos0 == 0:
            e = mx.concatenate([mx.zeros_like(e[:, :1]), e[:, 1:]], axis=1)
        hn = d.hnorm(hid)
        if GV.HEAD16 and hn.dtype == mx.float32 and e.dtype in (mx.float16, mx.bfloat16):
            hn = hn.astype(e.dtype)                        # V1-VERIFY: the target's hidden is float32; the head ran fully in float32 (weights cast f16 -> f32 per call, f32 GEMVs): 3.7 ms, ~1.5 ms of it casts / doubled bytes
        h = d.eh_proj(mx.concatenate([d.enorm(e), hn], axis=-1))
        h = d.mtp_block(h, _NoProj(cache[0]) if noproj else cache[0], last_only=last_only)
        return d.shared_head_norm(h)

    def _fwd_prof(self, pr, tag, cache, toks, hid, pos0, last_only, d):
        """_fwd with a forced eval after every op (glm_head_prof)."""
        from mlx_vlm.models.switch_layers import SwitchGLU
        e = self.embed(toks)
        if pos0 == 0:
            e = mx.concatenate([mx.zeros_like(e[:, :1]), e[:, 1:]], axis=1)
        pr.t(f"{tag}.embed", e)
        x = mx.concatenate([d.enorm(e), d.hnorm(hid)], axis=-1)
        pr.t(f"{tag}.norms+concat", x)
        x = d.eh_proj(x)
        pr.t(f"{tag}.eh_proj", x)
        blk = d.mtp_block
        xn = blk.input_layernorm(x)
        att, _ = blk.self_attn(xn, cache=cache[0], last_only=last_only)
        pr.t(f"{tag}.attention(+indexer, kv)", att)
        x = (x[:, -1:] if last_only and x.shape[1] > 1 else x) + att
        m = blk.mlp
        xn = blk.post_attention_layernorm(x)
        idx, w = m.gate(xn)
        pr.t(f"{tag}.router", idx, w)
        sh = m.shared_experts(xn)
        pr.t(f"{tag}.shared_expert", sh)
        r = m.switch_mlp(xn, idx)
        pr.t(f"{tag}.routed_experts", r)
        out = x + SwitchGLU._combine(r, w, sh, None)
        pr.t(f"{tag}.combine", out)
        h = d.shared_head_norm(out)
        pr.t(f"{tag}.head_norm", h)
        return h

    def _draft_prof(self, st: GlmDraft, last: int, p0: int, depth: int, d, voc):
        pr = self.__dict__.setdefault("_prof", _Prof())
        pr.begin()
        toks = mx.concatenate([st.htok[1:], mx.array([int(last)], dtype=mx.int32)])
        n = int(toks.shape[0])
        if st.cache is None:
            st.cache = self.d.make_cache()
        h = self._fwd_prof(pr, f"feed{min(n, 2)}", st.cache, toks[None], st.hid[None], st.pos, True, d)
        st.pos += n
        st.hid = st.htok = None
        out = [self._logits_argmax(h[:, -1:], voc)]
        pr.t("lm_head+argmax" + (" (draft vocab)" if voc is not None else ""), out[0])
        if depth > 1:
            from mlx_vlm.speculative.cache_state import start_speculative_cache
            tx = start_speculative_cache(st.cache, depth - 1)
            try:
                for _ in range(depth - 1):
                    h = self._fwd_prof(pr, "chain1", st.cache, out[-1], h[:, -1:], None, True, d)
                    out.append(self._logits_argmax(h, voc))
                    pr.t("lm_head+argmax", out[-1])
            finally:
                tx.abort()
        pr.dump()
        return mx.concatenate(out, axis=1)[0].astype(mx.int32)

    def _bind(self, st: GlmDraft):
        """(head module, vocab head) of the variant asked for now; a session whose cache was written by another variant starts over (no draft this round)."""
        d, voc = self.select()
        if st.variant is not None and st.variant != self.mode and st.cache is not None:
            st.cache, st.pos, st.hid, st.htok = None, None, None, None
        st.variant = self.mode
        return d, voc

    def feed(self, st: GlmDraft, extra=None, d=None):
        """Turn the pending hidden states whose next token is known into pairs of the head's cache (the next token of the last one is `extra`, when given).
        Returns the head's last output [1, 1, H] (None when nothing was fed). Prefill chunks call it without `extra`: only the last position stays pending."""
        if st.hid is None:
            return None
        if d is None:
            d = self._bind(st)[0]
            if st.hid is None:
                return None
        toks = st.htok[1:] if extra is None else mx.concatenate([st.htok[1:], mx.array([int(extra)], dtype=mx.int32)])
        n = int(toks.shape[0])
        if n == 0:
            return None
        if st.cache is None:
            st.cache = self.d.make_cache()
        h = None
        noproj = MTPFEED and extra is None                      # prompt feed: the output is discarded (prefill_chunk), only the caches are kept
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            h = self._fwd(st.cache, toks[None, a:b], st.hid[None, a:b], st.pos + a, last_only=True, d=d, noproj=noproj)
        st.pos += n
        st.hid, st.htok = (st.hid[n:], st.htok[n:]) if n < st.m else (None, None)
        return h

    def draft(self, st: GlmDraft, last: int, p0: int, depth: int):
        """Lazy [depth] int32 draft tokens following `last` (position p0), or None when the state does not line up."""
        if depth <= 0 or st.hid is None or st.pos + st.m != p0:
            return None
        from mlx_vlm.speculative.cache_state import start_speculative_cache
        d, voc = self._bind(st)
        if st.hid is None:
            return None
        if os.environ.get("KYOJIN_GLM_HEAD_PROF", "0") not in ("0", "") and st.m <= CHUNK:
            return self._draft_prof(st, last, p0, depth, d, voc)
        h = self.feed(st, last, d=d)
        out = [self._logits_argmax(h[:, -1:], voc)]                                           # [1, 1]
        if depth > 1:
            tx = start_speculative_cache(st.cache, depth - 1)
            try:
                for _ in range(depth - 1):
                    h = self._fwd(st.cache, out[-1], h[:, -1:], None, last_only=True, d=d)
                    out.append(self._logits_argmax(h, voc))
            finally:
                tx.abort()
        return mx.concatenate(out, axis=1)[0].astype(mx.int32)
