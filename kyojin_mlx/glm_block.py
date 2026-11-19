# SPDX-License-Identifier: Apache-2.0
"""External GLM-5.3 block drafters on MLX: DFlash 2 (Inco AI, Apache-2.0) and DSpark (Red Hat AI speculator, MIT), as drafters of GlmBackend.

Both are one-forward block drafters of a 5-layer Qwen3-style backbone (hidden 4096, mlp 12288, sliding window 2048):
  * the hidden states of several TARGET layers (concatenated, `fc`, `hidden_norm`) are the context: each draft layer turns them into its own K / V rows
    (the same context rows for every layer, only the projections differ), so the context of a session is a per-layer K / V ring that grows with the
    committed positions (`BlockDraft`, fed by the verify pass exactly like the native nextn head's `GlmDraft`);
  * one block of 8 query rows [pending token, mask x 7] at the positions p0 .. p0 + 7 attends to the last <= 2048 context rows and to the block;
  * DFlash2: input embedding = the TARGET's embedding, the TARGET's lm_head, `block_size - 1` = 7 drafts (row i predicts the token at p0 + i),
    non-causal block, two-tap grouped dynamic convolutions around attention and mlp, then a candidate selector (top-16 of the unary logits per
    position, a bilinear predecessor x successor score, one path chosen sequentially). Reference: z-lab/dflash (MIT) dflash/model.py + model_mlx.py.
  * DSpark: own embedding and own lm_head, 8 drafts (row k predicts the token at p0 + k + 1), causal block, a Markov logit bias W2 @ W1[previous token]
    chained along the block (greedy: tok_k = argmax(logits_k + bias(tok_{k-1}))), and a confidence head per position (sigmoid of a linear on
    [hidden_k, W1[prev]] = P(accept), the draft length is the longest prefix whose cumulative product stays above a threshold).
    Reference: vllm-project/speculators (Apache-2.0) models/dflash + models/dspark.

Which target tensor is "the hidden state of layer i": the checkpoints disagree by one in convention. DFlash2 follows z-lab / HF (`hidden_states[i + 1]` =
the OUTPUT of layer i); DSpark follows speculators / vLLM (id i = the INPUT of layer i = the output of layer i - 1; id num_layers = the final state). A GLM-5.3 layer carries
`hc_mult` hyper-connection streams: the tap is their mean (the same mean the model takes before its final norm). KYOJIN_GLM_TAP_OFF shifts both by a
constant (calibration tool, tools/glm_block_calib.py). Everything a drafter produces is only a proposal: the target verifies, so output never depends on it.

Flags (all off / unset by default): KYOJIN_GLM_BLOCK=dflash2:DIR | dspark:DIR (GlmBackend.enable_block), KYOJIN_GLM_BLOCK_Q=8|4 (quantise the drafter's linears),
KYOJIN_GLM_BLOCK_CAND=N (DSpark: the Markov bias only on the top-N base-logit tokens of each position, 0 = the whole vocabulary),
KYOJIN_GLM_BLOCK_CONF=pct (DSpark: draft-length gate on the cumulative confidence, 0 = off), KYOJIN_GLM_TAP_OFF=n, KYOJIN_GLM_TAP_CLAMP=1 (taps of layers the
loaded trunk does not have fall back to its last layer: plumbing on a layer cut only, acceptance is meaningless)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

WINDOW_DEFAULT = 2048
CHUNK = 1024                                   # context rows turned into K / V rows per matmul


def TAP_OFF() -> int:
    return int(os.environ.get("KYOJIN_GLM_TAP_OFF", "0") or 0)


# ------------------------------------------------------------------------------------------------ target taps

class Taps:
    """Hidden-state taps of the target trunk: the mean over the hyper-connection streams of the OUTPUT of some layers, collected during a forward.
    `install` swaps the class of those layers for a subclass that records into `self.d` while `self.on`; with no install the trunk is untouched."""

    def __init__(self, layers: list, outs: list[int], clamp: bool = False):
        self.layers = layers
        self.n = len(layers)
        self.clamp = clamp
        self.want = list(outs)                                  # requested output layer indices, in the drafter's order (-1 = the embedding: not tapped)
        self.have = [self._fit(j) for j in self.want]
        self.d: dict[int, mx.array] = {}
        self.on = False
        self.installed = False
        self.reduce = "mean"                                    # "raw" keeps the [B, L, hc, H] streams (tools/glm_block_calib.py)

    def _fit(self, j: int) -> int:
        if j < 0:
            raise ValueError("tap of the embedding output (layer id 0 of the input convention) is not supported")
        if j >= self.n:
            if not self.clamp:
                raise ValueError(f"tap layer {j} but the trunk has {self.n} layers (KYOJIN_GLM_TAP_CLAMP=1 maps it to the last one: plumbing only)")
            return self.n - 1
        return j

    def install(self) -> None:
        if self.installed:
            return
        sink = self
        for j in sorted(set(self.have)):
            layer = self.layers[j]
            cls = layer.__class__

            def make(cls=cls, j=j):
                class Tapped(cls):
                    def __call__(self, *a, **k):
                        out = cls.__call__(self, *a, **k)
                        if sink.on:
                            h = out[0] if isinstance(out, tuple) else out
                            sink.d[j] = h if (sink.reduce == "raw" or h.ndim != 4) else h.mean(axis=2)
                        return out
                Tapped.__name__ = cls.__name__
                return Tapped
            layer.__class__ = make()
        self.installed = True

    def begin(self) -> None:
        self.d = {}
        self.on = True

    def end(self) -> mx.array:
        """[B, L, T * H] bf16, the taps in the drafter's order."""
        self.on = False
        d, self.d = self.d, {}
        return mx.concatenate([d[j].astype(mx.bfloat16) for j in self.have], axis=-1)


# ------------------------------------------------------------------------------------------------ weights

def _quant_bits() -> int:
    return int(os.environ.get("KYOJIN_GLM_BLOCK_Q", "0") or 0)


class Lin:
    """y = x @ W.T for a [out, in] weight, bf16 or mx-quantised (group 64)."""

    def __init__(self, w: mx.array, bits: int = 0):
        self.bits = bits if (bits and w.shape[-1] % 64 == 0) else 0
        if self.bits:
            self.q, self.s, self.b = mx.quantize(w, group_size=64, bits=self.bits)
            self.w = None
        else:
            self.w = w

    def __call__(self, x):
        if self.w is not None:
            return x @ self.w.T
        return mx.quantized_matmul(x, self.q, self.s, self.b, transpose=True, group_size=64, bits=self.bits)

    def arrays(self):
        return [self.w] if self.w is not None else [self.q, self.s, self.b]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.arrays())


def _norm(x, w, eps):
    return mx.fast.rms_norm(x, w, eps)


class Cfg:
    def __init__(self, d: dict, kind: str):
        self.kind = kind
        if kind == "dflash2":
            df = d["dflash_config"]
            self.layers, self.heads, self.kv, self.hd = d["num_hidden_layers"], d["num_attention_heads"], d["num_key_value_heads"], d["head_dim"]
            self.hidden, self.eps, self.window = d["hidden_size"], d["rms_norm_eps"], int(d.get("sliding_window") or WINDOW_DEFAULT)
            self.theta = float(d["rope_parameters"]["rope_theta"])
            self.block, self.mask_id = int(df["block_size"]), int(df["mask_token_id"])
            self.tap_ids = list(df["target_layer_ids"])
            self.tap_conv = "out"                                # z-lab / HF: hidden_states[id + 1] = output of layer id
            self.causal = bool(d.get("is_causal") is True)       # false in the checkpoint: the block is bidirectional
            self.conv_k, self.conv_g = int(df["conv_kernel_size"]), int(df["conv_group_size"])
            self.sel_rank, self.sel_k = int(df["selector_rank"]), int(df["selector_top_k"])
            self.vocab = d["vocab_size"]
            self.first_row = 1                                   # row 0 is the pending token, drafts are rows 1 .. block - 1
        else:
            t = d["transformer_layer_config"]
            self.layers, self.heads, self.kv, self.hd = t["num_hidden_layers"], t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
            self.hidden, self.eps, self.window = t["hidden_size"], t["rms_norm_eps"], int(t.get("sliding_window") or WINDOW_DEFAULT)
            self.theta = float(t["rope_parameters"]["rope_theta"])
            self.block, self.mask_id = int(d["block_size"]), int(d["mask_token_id"])
            self.tap_ids = list(d["aux_hidden_state_layer_ids"])
            self.tap_conv = "in"                                 # speculators / vLLM: id i = input of layer i
            self.causal = not bool(d.get("sliding_window_non_causal", False))
            self.markov_rank = int(d["markov_rank"])
            self.conf = bool(d.get("enable_confidence_head")) and bool(d.get("confidence_head_with_markov"))
            self.vocab = t["vocab_size"]
            self.first_row = 0                                   # row k predicts the token at p0 + k + 1
        self.n_drafts = self.block - self.first_row

    def tap_outs(self) -> list[int]:
        """Output-layer index of each tap (after the convention shift and KYOJIN_GLM_TAP_OFF)."""
        sh = 0 if self.tap_conv == "out" else -1
        return [i + sh + TAP_OFF() for i in self.tap_ids]


def _load(path: Path) -> dict:
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


# ------------------------------------------------------------------------------------------------ per-session state

class BlockDraft:
    """One session's drafter state: the per-layer context K / V ring and the target taps of the positions not turned into K / V rows yet (`hid`).
    Same surface as glm_nextn.GlmDraft (append / arrays) so GlmPass.commit and the prefill / step paths feed it unchanged."""

    def __init__(self, window: int, kind: str = ""):
        self.window, self.kind = window, kind
        self.ks: list | None = None                 # per layer [1, Hkv, n, D] (rope'd, k-normed)
        self.vs: list | None = None
        self.rpos = 0                               # absolute position of ring row 0
        self.hid = None                             # [m, T * H] bf16: taps of the positions pos .. pos + m - 1
        self.pos = None
        self.conf = None                            # DSpark: cumulative confidence of the last draft (list of floats, lazy until read)

    @property
    def m(self) -> int:
        return 0 if self.hid is None else int(self.hid.shape[0])

    @property
    def rn(self) -> int:
        return 0 if self.ks is None else int(self.ks[0].shape[2])

    def append(self, h, toks, start: int) -> None:
        """h [n, T*H] taps after positions start .. start + n - 1 (toks, the tokens at those positions, are not used). A gap restarts the ring."""
        if self.pos is not None and start == self.pos + self.m:
            self.hid = h if self.hid is None else mx.concatenate([self.hid, h], axis=0)
        else:
            self.ks = self.vs = None
            self.hid, self.pos = h, int(start)
        if self.m > self.window:                    # older rows are out of every future window: never turn them into K / V
            drop = self.m - self.window
            self.hid, self.pos = self.hid[drop:], self.pos + drop

    def fork(self) -> "BlockDraft":
        """A shallow copy: the arrays are shared (feed / append only ever rebind them), so drafting on the copy leaves this state as it was."""
        o = BlockDraft(self.window, self.kind)
        o.ks, o.vs, o.rpos, o.hid, o.pos = self.ks, self.vs, self.rpos, self.hid, self.pos
        return o

    def arrays(self) -> list:
        out = [a for a in (self.hid,) if a is not None]
        for lst in (self.ks, self.vs):
            if lst:
                out += list(lst)
        return out


# ------------------------------------------------------------------------------------------------ drafters

class _Block:
    """Shared machinery: context ingestion, rope / ring bookkeeping, the Qwen3 layer."""

    def __init__(self, path: Path, cfg: Cfg, w: dict, bits: int):
        self.cfg = cfg
        c = cfg
        L = c.layers
        self.fc = Lin(w["fc.weight"], bits)
        self.hidden_norm = w["hidden_norm.weight"]
        self.norm = w["norm.weight"]
        # context K / V projections of all layers in one matmul: rows [layer0 k | layer0 v | layer1 k ...]
        kv = mx.concatenate([t for i in range(L) for t in (w[f"layers.{i}.self_attn.k_proj.weight"], w[f"layers.{i}.self_attn.v_proj.weight"])], axis=0)
        self.kv_all = Lin(kv, bits)
        self.lay = []
        for i in range(L):
            p = f"layers.{i}."
            d = dict(
                ln1=w[p + "input_layernorm.weight"], ln2=w[p + "post_attention_layernorm.weight"],
                qn=w[p + "self_attn.q_norm.weight"], kn=w[p + "self_attn.k_norm.weight"],
                q=Lin(w[p + "self_attn.q_proj.weight"], bits), k=Lin(w[p + "self_attn.k_proj.weight"], bits), v=Lin(w[p + "self_attn.v_proj.weight"], bits),
                o=Lin(w[p + "self_attn.o_proj.weight"], bits),
                gate=Lin(w[p + "mlp.gate_proj.weight"], bits), up=Lin(w[p + "mlp.up_proj.weight"], bits), down=Lin(w[p + "mlp.down_proj.weight"], bits))
            self.lay.append(d)
        self.kvw = c.kv * c.hd

    # -- context
    def feed(self, st: BlockDraft) -> None:
        """Turn the pending taps into K / V rows of the ring (positions st.pos ..), keep the last `window` rows."""
        if st.hid is None:
            return
        c = self.cfg
        if st.ks is not None and st.rpos + st.rn != st.pos:
            st.ks = st.vs = None
        if st.ks is None:
            st.rpos = st.pos
        n = st.m
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            x = st.hid[a:b]
            ctx = _norm(self.fc(x), self.hidden_norm, c.eps)                       # [m, H]
            kv = self.kv_all(ctx)                                                  # [m, L * 2 * Hkv * D]
            m = b - a
            ks, vs = [], []
            for i, d in enumerate(self.lay):
                base = i * 2 * self.kvw
                k = kv[:, base:base + self.kvw].reshape(m, c.kv, c.hd)
                v = kv[:, base + self.kvw:base + 2 * self.kvw].reshape(m, c.kv, c.hd)
                k = _norm(k, d["kn"], c.eps).transpose(1, 0, 2)[None]              # [1, Hkv, m, D]
                k = mx.fast.rope(k, c.hd, traditional=False, base=c.theta, scale=1.0, offset=st.pos + a)
                ks.append(k)
                vs.append(v.transpose(1, 0, 2)[None])
            if st.ks is None:
                st.ks, st.vs = ks, vs
            else:
                st.ks = [mx.concatenate([o, k], axis=2) for o, k in zip(st.ks, ks)]
                st.vs = [mx.concatenate([o, v], axis=2) for o, v in zip(st.vs, vs)]
            if st.rn > c.window:
                cut = st.rn - c.window
                st.ks = [k[:, :, cut:] for k in st.ks]
                st.vs = [v[:, :, cut:] for v in st.vs]
                st.rpos += cut
        st.pos += n
        st.hid = None

    def _mask(self, st: BlockDraft, p0: int, L: int):
        """[L, rn + L] bool (None when everything is visible): context row j (position rpos + j) is seen by block row i (position p0 + i) when it is inside its window;
        the block rows see each other (non-causal) or the earlier ones (causal)."""
        c = self.cfg
        rn = st.rn
        ctx_all = rn == 0 or (st.rpos > p0 + L - 1 - c.window)
        if ctx_all and not c.causal:
            return None
        q = mx.arange(L)[:, None]
        blk = (mx.arange(L)[None, :] <= q) if c.causal else mx.ones((L, L), dtype=mx.bool_)
        if rn == 0:
            return blk if c.causal else None
        kpos = st.rpos + mx.arange(rn)[None, :]
        ctx = (kpos > (p0 + q - c.window)) if not ctx_all else mx.ones((L, rn), dtype=mx.bool_)
        return mx.concatenate([ctx, blk], axis=1)

    def _attn(self, st: BlockDraft, i: int, d: dict, h, p0: int, mask):
        """h [L, H] (already input-normed) -> attention output [L, H] after o_proj."""
        c = self.cfg
        L = h.shape[0]
        q = _norm(d["q"](h).reshape(L, c.heads, c.hd), d["qn"], c.eps).transpose(1, 0, 2)[None]
        k = _norm(d["k"](h).reshape(L, c.kv, c.hd), d["kn"], c.eps).transpose(1, 0, 2)[None]
        v = d["v"](h).reshape(L, c.kv, c.hd).transpose(1, 0, 2)[None]
        q = mx.fast.rope(q, c.hd, traditional=False, base=c.theta, scale=1.0, offset=p0)
        k = mx.fast.rope(k, c.hd, traditional=False, base=c.theta, scale=1.0, offset=p0)
        if st.rn:
            k = mx.concatenate([st.ks[i], k], axis=2)
            v = mx.concatenate([st.vs[i], v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=c.hd ** -0.5, mask=mask)
        return d["o"](o[0].transpose(1, 0, 2).reshape(L, c.heads * c.hd))

    def nbytes(self) -> int:
        n = self.fc.nbytes + self.kv_all.nbytes
        for d in self.lay:
            n += sum(d[k].nbytes for k in ("q", "k", "v", "o", "gate", "up", "down"))
        return n


def _gdc(h, delta, base, g: int):
    """Grouped dynamic causal convolution over the rows of one block: h [L, D], delta [L, K, groups], base [K, D] -> [L, D]
    (tap t reads row i - t of the SAME block, zero before the block start)."""
    L, D = h.shape
    ng = D // g
    hb = h.reshape(L, ng, g)
    out = None
    for t in range(base.shape[0]):
        sh = hb if t == 0 else mx.concatenate([mx.zeros((t, ng, g), dtype=hb.dtype), hb[:L - t]], axis=0)
        coef = base[t].reshape(1, ng, g).astype(h.dtype) + delta[:, t, :, None].astype(h.dtype)
        term = sh * coef
        out = term if out is None else out + term
    return out.reshape(L, D)


class DFlash2(_Block):
    kind = "dflash2"

    def __init__(self, path: Path, cfg: Cfg, w: dict, bits: int, embed, head):
        super().__init__(path, cfg, w, bits)
        self.embed, self.head = embed, head                   # the TARGET's embedding (ids -> [.., H]) and lm_head (h -> logits)
        for i, d in enumerate(self.lay):
            p = f"layers.{i}."
            d["ac_base"], d["mc_base"] = w[p + "attention_conv.base_kernel"], w[p + "mlp_conv.base_kernel"]
            d["ac_proj"], d["mc_proj"] = Lin(w[p + "attention_conv.kernel_projection.weight"], bits), Lin(w[p + "mlp_conv.kernel_projection.weight"], bits)
        sel = "candidate_selector."
        self.sel_h = Lin(w[sel + "hidden_projection.weight"], bits)
        self.pred_cb, self.succ_cb = w[sel + "predecessor_codebook"], w[sel + "successor_codebook"]
        self.mask_ids = None

    def _conv_prep(self, h, d, which):
        c = self.cfg
        kern = d[which + "_proj"](h).reshape(h.shape[0], 2, c.conv_k, c.hidden // c.conv_g)
        return _gdc(h, kern[:, 0], d[which + "_base"][0], c.conv_g), kern[:, 1]

    def _conv_fin(self, h, delta, d, which):
        return _gdc(h, delta, d[which + "_base"][1], self.cfg.conv_g)

    def block_hidden(self, st: BlockDraft, last, p0: int, L: int | None = None):
        """[L, H] final-normed block hidden (L rows: the pending token and L - 1 masks; default the whole block, a shorter runtime block is what
        z-lab's dflash_generate(block_size=...) runs). `last` = the pending token (int or 0-d array)."""
        c = self.cfg
        L = c.block if L is None else L
        ids = mx.concatenate([mx.array(last, dtype=mx.int32).reshape(1), mx.full((L - 1,), c.mask_id, dtype=mx.int32)])
        x = self.embed(ids)
        x = x.astype(mx.bfloat16)
        mask = self._mask(st, p0, L)
        for i, d in enumerate(self.lay):
            r = x
            h, fin = self._conv_prep(_norm(x, d["ln1"], c.eps), d, "ac")
            a = self._attn(st, i, d, h, p0, mask)
            x = r + self._conv_fin(a, fin, d, "ac")
            r = x
            h, fin = self._conv_prep(_norm(x, d["ln2"], c.eps), d, "mc")
            m = d["down"](nn.silu(d["gate"](h)) * d["up"](h))
            x = r + self._conv_fin(m, fin, d, "mc")
        return _norm(x, self.norm, c.eps)

    def select(self, hid, logits, last, nd: int):
        """The sequential path through the top-k candidates of each draft row. hid [nd, H], logits [nd, V] fp32 -> [nd] int32 (lazy)."""
        c = self.cfg
        V = logits.shape[-1]
        cand = mx.argpartition(logits, V - c.sel_k, axis=-1)[..., V - c.sel_k:]          # [nd, k]
        unary = mx.take_along_axis(logits, cand, axis=-1)
        hp = self.sel_h(hid).astype(mx.float32)                                           # [nd, rank]
        prev = mx.array(last, dtype=mx.int32).reshape(1)
        path = []
        for i in range(nd):
            ctx = self.pred_cb[prev][0].astype(mx.float32) * hp[i]                         # [rank]
            succ = self.succ_cb[cand[i]].astype(mx.float32)                                # [k, rank]
            sc = unary[i] + succ @ ctx
            prev = mx.take(cand[i], mx.argmax(sc)).reshape(1)
            path.append(prev)
        return mx.concatenate(path).astype(mx.int32)

    def draft(self, st: BlockDraft, last, p0: int, depth: int):
        """Lazy [depth] int32 draft tokens after `last` (the token at position p0), or None when the state does not line up."""
        nd = min(depth, self.cfg.n_drafts)
        if nd <= 0 or st.pos is None or st.pos + st.m != p0:
            return None
        self.feed(st)
        hid = self.block_hidden(st, last, p0, nd + 1)
        logits = self.head(hid[1:1 + nd]).astype(mx.float32)
        return self.select(hid[1:1 + nd], logits, last, nd)


class DSpark(_Block):
    kind = "dspark"

    def __init__(self, path: Path, cfg: Cfg, w: dict, bits: int, embed=None, head=None):
        super().__init__(path, cfg, w, bits)
        self.embed_w = w["embed_tokens.weight"]
        self.lm = Lin(w["lm_head.weight"], bits)
        self.w1, self.w2 = w["markov_head.markov_w1.weight"], w["markov_head.markov_w2.weight"]           # [V, r] each
        self.cw = w.get("confidence_head.proj.weight")
        self.cb = w.get("confidence_head.proj.bias")

    @property
    def cand(self) -> int:
        return int(os.environ.get("KYOJIN_GLM_BLOCK_CAND", "0") or 0)

    @property
    def conf_thr(self) -> float | None:
        pct = int(os.environ.get("KYOJIN_GLM_BLOCK_CONF", "0") or 0)
        return pct / 100.0 if pct > 0 else None

    def nbytes(self) -> int:
        return super().nbytes() + self.lm.nbytes + self.w2.nbytes

    def block_hidden(self, st: BlockDraft, last, p0: int, L: int | None = None):
        """[L, H]: the block is causal, so the first L rows of the 8-row block are the same whatever follows them (L = the drafts wanted)."""
        c = self.cfg
        L = c.block if L is None else L
        ids = mx.concatenate([mx.array(last, dtype=mx.int32).reshape(1), mx.full((L - 1,), c.mask_id, dtype=mx.int32)])
        x = self.embed_w[ids]
        mask = self._mask(st, p0, L)
        for i, d in enumerate(self.lay):
            x = x + self._attn(st, i, d, _norm(x, d["ln1"], c.eps), p0, mask)
            h = _norm(x, d["ln2"], c.eps)
            x = x + d["down"](nn.silu(d["gate"](h)) * d["up"](h))
        return _norm(x, self.norm, c.eps)

    def chain(self, hid, last, nd: int):
        """Greedy Markov chain over the block: tok_k = argmax(base_logits_k + W2 @ W1[tok_{k-1}]), tok_{-1} = the pending token.
        Returns (tokens [nd] int32, cumulative confidence [nd] fp32 or None), both lazy."""
        c = self.cfg
        logits = self.lm(hid[:nd]).astype(mx.float32)                                     # [nd, V]
        prev = mx.array(last, dtype=mx.int32).reshape(1)
        toks, conf = [], []
        for k in range(nd):
            e = self.w1[prev][0]                                                           # [r]
            if self.cand:
                cand = mx.argpartition(logits[k], logits.shape[-1] - self.cand)[logits.shape[-1] - self.cand:]
                sc = logits[k][cand] + (self.w2[cand].astype(mx.float32) @ e.astype(mx.float32))
                prev = mx.take(cand, mx.argmax(sc)).reshape(1)
            else:
                sc = logits[k] + (self.w2 @ e).astype(mx.float32)
                prev = mx.argmax(sc).reshape(1).astype(mx.int32)
            toks.append(prev)
            if self.cw is not None:
                feat = mx.concatenate([hid[k], e.astype(hid.dtype)])
                conf.append(mx.sigmoid((self.cw[0].astype(mx.float32) * feat.astype(mx.float32)).sum() + self.cb.astype(mx.float32)[0]))
        cum = mx.cumprod(mx.stack(conf)) if conf else None
        return mx.concatenate(toks).astype(mx.int32), cum

    def draft(self, st: BlockDraft, last, p0: int, depth: int):
        nd = min(depth, self.cfg.n_drafts)
        if nd <= 0 or st.pos is None or st.pos + st.m != p0:
            return None
        self.feed(st)
        hid = self.block_hidden(st, last, p0, nd)
        toks, cum = self.chain(hid, last, nd)
        st.conf = cum
        return toks


# ------------------------------------------------------------------------------------------------ entry

class BlockDrafter:
    """What GlmBackend holds: the drafter, its taps and the session-state factory."""

    def __init__(self, model, taps: Taps, drafter, window: int):
        self.model, self.taps, self.d, self.window = model, taps, drafter, window
        self.kind = drafter.kind
        self.cfg = drafter.cfg

    @property
    def conf_thr(self) -> float | None:
        return getattr(self.d, "conf_thr", None)

    def new_state(self) -> BlockDraft:
        return BlockDraft(self.window, self.kind)

    def draft(self, st: BlockDraft, last: int, p0: int, depth: int):
        return self.d.draft(st, last, p0, depth)

    def trim(self, st: BlockDraft, toks: list) -> list:
        """DSpark confidence gate on an evaluated draft: the longest prefix whose cumulative confidence stays >= the threshold (flag KYOJIN_GLM_BLOCK_CONF)."""
        if self.conf_thr is None or st.conf is None:
            return toks
        cum = [float(x) for x in st.conf.tolist()]
        n = 0
        while n < len(toks) and n < len(cum) and cum[n] >= self.conf_thr:
            n += 1
        return toks[:n]

    def feed(self, st: BlockDraft) -> None:
        self.d.feed(st)


def build(path: str, kind: str | None, lm, log=print) -> BlockDrafter:
    """lm: the target's LanguageModel (mlx-vlm glm5_next) the drafter reads its embedding / lm_head from and whose layers get the taps."""
    p = Path(os.path.expanduser(path))
    d = json.loads((p / "config.json").read_text())
    kind = kind or ("dspark" if "DSpark" in "".join(d.get("architectures", [])) else "dflash2")
    cfg = Cfg(d, kind)
    w = _load(p)
    bits = _quant_bits()
    taps = Taps(lm.model.layers, cfg.tap_outs(), clamp=os.environ.get("KYOJIN_GLM_TAP_CLAMP", "0") != "0")
    if kind == "dflash2":
        from mlx_vlm.models.glm5_next.language import linear
        embed = lm.model.embed_tokens
        head = (lambda h: linear(lm.lm_head, h)) if hasattr(lm, "lm_head") else embed.as_linear
        dr = DFlash2(p, cfg, w, bits, embed, head)
    else:
        dr = DSpark(p, cfg, w, bits)
    mx.eval(*weight_arrays(dr))
    taps.install()
    log(f"[kyojin] GLM block drafter {kind} from {p.name}: block {cfg.block} ({cfg.n_drafts} drafts), taps {cfg.tap_conv} of layers {cfg.tap_ids} -> outputs {taps.have}, "
        f"window {cfg.window}, {dr.nbytes() / 1e9:.2f} GB of drafter weights read per step" + (f", q{bits}" if bits else ""))
    return BlockDrafter(lm, taps, dr, cfg.window)


def weight_arrays(dr) -> list:
    out = []

    def add(v):
        if isinstance(v, mx.array):
            out.append(v)
        elif isinstance(v, Lin):
            out.extend(v.arrays())
    for v in vars(dr).values():
        add(v)
    for d in dr.lay:
        for v in d.values():
            add(v)
    return out
