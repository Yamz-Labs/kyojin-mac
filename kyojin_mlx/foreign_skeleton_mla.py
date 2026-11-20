# SPDX-License-Identifier: Apache-2.0
"""Block N1 of the decoder skeleton: MLA projections, absorbed form, with the N3 kpool selection behind it. Fires on the tensors `attn_kv_a_mqa` +
3-D `attn_k_b` / `attn_v_b` and the keys `attention.{q_lora_rank, kv_lora_rank, key_length_mla, value_length_mla}`; no architecture name appears here.

  qr        = rms(q_a(x))                                   (the q_lora residual; N3's index query reads it)
  q         = q_b(qr) -> [B, nh, L, nope + rope]            q_pe roped at the token positions
  kv        = kv_a_mqa(x) -> latent rms(kv[:kv_lora]) [B, 1, L, kv_lora], k_pe = kv[kv_lora:] roped [B, 1, L, rope]
  scores    = ((q_nope @ k_b^T) . latent + q_pe . k_pe) * scale,  scale = key_length_mla ** -0.5 (nope + rope)
  attention one kv head (the latent is key and value), optional sinks, causal / window mask, optional bool `keep` [B, 1, L, S] (N3)
  out       = (p @ latent) @ v_b^T -> [B, nh, L, v]; then the element output gate sigmoid(gate(x)) and o_proj (the skeleton's)
k_b [nh, kv_lora, nope] and v_b [nh, v, kv_lora] are the GGUF attn_k_b / attn_v_b as stored (numpy order), the layout of mlx-vlm's embed_q /
unembed_out. Rope: llama.cpp NORM = mlx traditional (consecutive pairs), on the trailing `rope` dims only; rope = 0 is the nope-only form.

Cache: one kv head whose key row is [latent | k_pe | kpool value | kpool gate] (value width 0), so any mlx-lm cache holds it; per-row offsets
(int or [B]) follow the rows contract of foreign_skeleton_kpool (right-aligned batch cache).
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

from . import attn_fp16 as AF
from . import fixed_kv as FK
from . import mla_flash_pf as MF
from . import foreign_skeleton_kpool as KP
from .rowinv import absorb_mm


def _amm(x, w):
    """x @ w^T per head (k_b / v_b absorb); flag skinny_absorb: a 2..8-row pass is one batched rowdot launch instead of the fp32 steel GEMM."""
    x = x.astype(w.dtype)                                                  # one dtype (the attn_fp16_parts arms mix fp16 and fp32 operands); no-op in every other path
    r = absorb_mm(x, w)
    return r if r is not None else x @ w.swapaxes(-1, -2)

KEYS = {"attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla"}


def fires(layer_tensors: dict) -> bool:
    return "attn_kv_a_mqa.weight" in layer_tensors and "attn_k_b.weight" in layer_tensors and "attn_v_b.weight" in layer_tensors


# KYOJIN_HY4_ATTN_FUSE=1 (read per call, default off): the scores -> probabilities chain of attend() as two fused elementwise kernels around one logsumexp reduction
# (the stock chain is ~10 full passes over the [B, nh, L, S] fp32 scores: cast, scale, rope add, mask, keep, sink concat, softmax, slice, cast).
def _scores_fn(kind, has_rope, has_keep):
    """sc = s1 * scale (+ s2) with the mask (kind "bool": 0 / -inf, "add": float added) and the keep mask applied, fp32 [B, nh, L, S]."""
    def f(s1, scale, *rest):
        r = list(rest)
        sc = s1.astype(mx.float32) * scale
        if has_rope:
            sc = sc + r.pop(0).astype(mx.float32)
        if kind == "bool":
            sc = sc + mx.where(r.pop(0), 0.0, -mx.inf)
        elif kind == "add":
            sc = sc + r.pop(0).astype(mx.float32)
        if has_keep:
            sc = mx.where(r.pop(0), sc, -mx.inf)
        return sc
    return mx.compile(f, shapeless=True)


@mx.compile
def _probs_fn(sc, lse):
    return mx.exp(sc - lse)


_FUSED: dict = {}


class KpoolProj(nn.Module):
    """N3's projections: index query from the q_lora residual, index key from x, compressor value / gate from the normed index key."""

    def __init__(self, d, q_lora, H, dx, eps):
        super().__init__()
        self.H, self.dx = H, dx
        self.q_b_proj = nn.Linear(q_lora, H * dx, bias=False)
        self.k_proj = nn.Linear(d, dx, bias=False)
        self.k_norm = nn.RMSNorm(dx, eps=eps)
        self.weights_proj = nn.Linear(d, H, bias=False)
        self.value_proj = nn.Linear(dx, dx, bias=False)
        self.gate_proj = nn.Linear(dx, dx, bias=False)

    def rows(self, x, a16=False):
        """-> value, gate [B, L, dx]: the per-position rows the cache keeps (M:5625-5632). a16: fp16 products (flag attn_fp16)."""
        ln = AF.call if a16 else (lambda m, v: m(v))
        ik = self.k_norm(ln(self.k_proj, x))
        return ln(self.value_proj, ik), ln(self.gate_proj, ik)

    def query(self, x, qr, a16=False):
        B, L, _ = x.shape
        ln = AF.call if a16 else (lambda m, v: m(v))
        iq = ln(self.q_b_proj, qr).reshape(B, L, self.H, self.dx).transpose(0, 2, 1, 3)
        w = ln(self.weights_proj, x) * (self.dx * self.H) ** -0.5
        return iq, w


class MlaAttention(nn.Module):
    PATHS = {"swa_slice": 0, "kpool_gather": 0, "kpool_fuse": 0, "kpool_keycache": 0, "kpool_fuse_fixed": 0, "qblocks": 0, "attn_fuse": 0, "attn_flash": 0, "attn_flash_swa": 0, "attn_vec": 0}          # how often each flagged path ran (tests / the c76 cell read it)

    def __init__(self, d, nh, q_lora, kv_lora, nope, rope, v, eps=1e-6, base=10000.0, gate=False, sinks=False, kpool=None):
        """kpool: None, or dict(H, dx, kp, top_k, select_tail, n_rot, base) for the N3 selection."""
        super().__init__()
        self.nh, self.kv, self.nope, self.r, self.hv, self.base = nh, kv_lora, nope, rope, v, base
        self.scale = (nope + rope) ** -0.5
        self.q_a_proj = nn.Linear(d, q_lora, bias=False)
        self.q_a_norm = nn.RMSNorm(q_lora, eps=eps)
        self.q_b_proj = nn.Linear(q_lora, nh * (nope + rope), bias=False)
        self.kv_a_proj = nn.Linear(d, kv_lora + rope, bias=False)
        self.kv_a_norm = nn.RMSNorm(kv_lora, eps=eps)
        self.k_b = mx.zeros((nh, kv_lora, nope))
        self.v_b = mx.zeros((nh, v, kv_lora))
        self.o_proj = nn.Linear(nh * v, d, bias=False)
        self.gate = gate
        if gate:
            self.gate_proj = nn.Linear(d, nh * v, bias=False)
        self.sinks = mx.zeros((nh,)) if sinks else None
        self.kp = None
        if kpool:
            k = dict(kpool)
            self.kpool = KpoolProj(d, q_lora, k["H"], k["dx"], eps)
            self.kpool.ape, self.kpool.pool_norm = mx.zeros((k["kp"], k["dx"])), mx.ones((k["dx"],))
            self.kp = k

    def _rope(self, x, off, dev=False):
        if not self.r:
            return x
        if dev:                                                                        # fixed-state cache: the offset stays a scalar device array (no read)
            return mx.fast.rope(x, self.r, traditional=True, base=self.base, scale=1.0, offset=off)
        off = off if not isinstance(off, mx.array) or off.size > 1 else int(off.reshape(-1)[0].item())
        return mx.fast.rope(x, self.r, traditional=True, base=self.base, scale=1.0, offset=off)

    def project(self, x, off=0, dev=False, a16=False, kv32=False):
        """-> qr, q_nope [B, nh, L, nope], q_pe (roped), latent [B, 1, L, kv_lora], k_pe [B, 1, L, rope] (roped). a16: fp16 products (flag attn_fp16)."""
        B, L, _ = x.shape
        ln = AF.call if a16 else (lambda m, v: m(v))
        qr = self.q_a_norm(ln(self.q_a_proj, x))
        q = ln(self.q_b_proj, qr).reshape(B, L, self.nh, self.nope + self.r).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        lat, k_pe = mx.split((self.kv_a_proj(x) if kv32 else ln(self.kv_a_proj, x)), [self.kv], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.r).transpose(0, 2, 1, 3)
        lat = self.kv_a_norm(lat)[:, None]
        return qr, q_nope, self._rope(q_pe, off, dev), lat, self._rope(k_pe, off, dev)

    def _swa_slice(self, L, S, lat, k_pe, mask):
        """KYOJIN_HY4_SWA_SLICE=1: on a windowed layer keep only the key slots the window mask can still show. Query row l sits at slot S - L + l and sees
        slots (S - L + l - window, S - L + l], so slots < S - L - window + 1 are masked for every row: dropping them (and the same mask columns) leaves
        the softmax over the same nonzero terms. Returns (lat, k_pe, mask)."""
        w = getattr(self, "window", 0)
        if not (w and getattr(self, "swa", False) and isinstance(mask, mx.array)) or os.environ.get("KYOJIN_HY4_SWA_SLICE", "0") != "1":
            return lat, k_pe, mask
        start = max(0, S - L - w + 1)
        if start == 0:
            return lat, k_pe, mask
        MlaAttention.PATHS["swa_slice"] += 1
        return lat[:, :, start:], k_pe[:, :, start:], mask[..., start:]

    def attend_gather(self, q_nope, q_pe, lat, k_pe, slots, ok):
        """KYOJIN_HY4_KPOOL_GATHER=1: the attend() of the keep mask over only the kept keys. slots / ok [B, L, W] from KP.gather_slots (ascending slots,
        the masked path's summation order). Rows are processed in blocks so the gathered [B, rows, W, 576] copy stays small."""
        B, nh, L, _ = q_nope.shape
        qa = _amm(q_nope, self.k_b)                                        # [B, nh, L, kv_lora]
        lat2, kpe2 = lat[:, 0], k_pe[:, 0]
        blk = int(os.environ.get("KYOJIN_HY4_GATHER_ROWS", "128"))
        outs = []
        for a in range(0, L, blk):
            b = min(L, a + blk)
            n, W = b - a, slots.shape[-1]
            ix = slots[:, a:b].reshape(B, n * W, 1)
            lg = mx.take_along_axis(lat2, ix, axis=1).reshape(B, n, W, -1)             # [B, n, W, kv_lora]
            qr_ = qa[:, :, a:b].transpose(0, 2, 1, 3)                                  # [B, n, nh, kv_lora]
            sc = (qr_ @ lg.swapaxes(-1, -2)).astype(mx.float32) * self.scale           # [B, n, nh, W]
            if self.r:
                kg = mx.take_along_axis(kpe2, ix, axis=1).reshape(B, n, W, -1)
                qp = q_pe[:, :, a:b].transpose(0, 2, 1, 3) * self.scale
                sc = sc + (qp @ kg.swapaxes(-1, -2)).astype(mx.float32)
            sc = mx.where(ok[:, a:b, None, :], sc, -mx.inf)
            if self.sinks is not None:
                sc = mx.concatenate([sc, mx.broadcast_to(self.sinks.astype(mx.float32)[None, None, :, None], (B, n, nh, 1))], axis=-1)
            p = mx.softmax(sc, axis=-1, precise=True)[..., :W]
            outs.append((p.astype(lat.dtype) @ lg).transpose(0, 2, 1, 3))
        o = mx.concatenate(outs, axis=2) if len(outs) > 1 else outs[0]                 # [B, nh, L, kv_lora]
        return _amm(o, self.v_b)

    def attend_flash(self, q_nope, q_pe, rows, ids):
        """Absorbed attention over the per-row key lists `ids` [B, L, W] (-1 = none) with the fused flash kernel (mla_flash_pf) -> [B, nh, L, v], or None when its shape / dtype
        contract is not met (the caller then runs the masked chain). rows = the cache buffer rows [1, 1, S, RS] (latent | k_pe | ...), fp16."""
        kb, vb = self._absorbs(rows.dtype)
        qa = _amm(q_nope, kb)                                              # [B, nh, L, kv_lora]
        if rows.dtype != mx.float16 or qa.dtype != mx.float16 or q_pe.dtype != mx.float16 or self.kv != MF.D or self.r != MF.DR or not MF.supported(qa, q_pe, rows, ids, self.sinks):
            return None
        return _amm(MF.attend(qa, q_pe, rows, ids, self.scale, self.sinks), vb)

    def _attend_vec(self, q_nope, q_pe, buf, off, kind, extra, window, ab16):
        """Flag mla_decode_vec: k_b absorb, the ONE-launch attention core over the cells of `kind` (0 window, 1 keep mask, 2 key ids), v_b absorb -> [B, nh, L, v], or None outside the kernel's contract
        (the caller then runs the stock core)."""
        from . import mla_decode_vec as MV
        kb, vb = self._absorbs(mx.float16 if ab16 else mx.float32)
        qa = _amm(q_nope.astype(kb.dtype), kb)
        if not MV.eligible(qa, q_pe, buf, kind, extra, MlaAttention.PATHS):
            return None
        MlaAttention.PATHS["attn_vec"] += 1
        return _amm(MV.attend(qa, q_pe, buf, off, kind, extra, self.sinks, self.scale, window), vb)

    def _absorbs(self, dt):
        """k_b / v_b of the dtype of the cache rows: the fp16 copies (attn_fp16.prepare) for fp16 rows, else the stored ones."""
        if dt == mx.float16 and "_kb16" in self.__dict__:
            return self.__dict__["_kb16"], self.__dict__["_vb16"]
        return self.k_b, self.v_b

    def attend(self, q_nope, q_pe, lat, k_pe, mask=None, keep=None, slice_ok=True, ab16=None):
        """Absorbed attention over one latent head -> [B, nh, L, v] (before the gate / o_proj)."""
        L, S = q_nope.shape[2], lat.shape[2]
        if slice_ok:
            lat, k_pe, mask = self._swa_slice(L, S, lat, k_pe, mask)
        S = lat.shape[2]
        kb, vb = self._absorbs(lat.dtype if ab16 is None else (mx.float16 if ab16 else mx.float32))
        qa = _amm(q_nope.astype(kb.dtype), kb)                             # [B, nh, L, kv_lora]
        if L > 16 and os.environ.get("KYOJIN_HY4_ATTN_FUSE", "0") == "1" and (mask is None or isinstance(mask, (str, mx.array))):    # prefill chunks only: decode / verify rows (<= 16) keep the measured decode path (absorb_mm, step compile)
            return self._attend_fused(qa, q_pe, lat, k_pe, mask, keep, L, S)
        sc = (qa @ lat.swapaxes(-1, -2)).astype(mx.float32) * self.scale
        if self.r:
            sc = sc + ((q_pe * self.scale) @ k_pe.swapaxes(-1, -2)).astype(mx.float32)
        if isinstance(mask, str):
            from mlx_lm.models.base import create_causal_mask
            mask = create_causal_mask(L, S - L) if mask == "causal" else None
        if mask is not None:
            sc = sc + (mx.where(mask, 0.0, -mx.inf) if mask.dtype == mx.bool_ else mask.astype(mx.float32))
        if keep is not None:
            sc = mx.where(keep, sc, -mx.inf)
        if self.sinks is not None:
            B, nh = sc.shape[:2]
            sc = mx.concatenate([sc, mx.broadcast_to(self.sinks.astype(mx.float32)[None, :, None, None], (B, nh, L, 1))], axis=-1)
        p = mx.softmax(sc, axis=-1, precise=True)[..., :S]
        return _amm(p.astype(lat.dtype) @ lat, vb)

    def kpool_sel(self):
        """The KpoolSelect of this layer (pooling weights + selection constants)."""
        kp = self.kp
        return KP.KpoolSelect(self.kpool.ape, self.kpool.pool_norm, self.kpool.k_norm.eps, kp["kp"], kp["top_k"], kp["select_tail"], kp["n_rot"], kp["base"], True)

    def _call_fixed(self, x, cache):
        """KYOJIN_HY4_FIXED_KV: the call on a fixed-state cache (fixed_kv.FixedKVCache): `cache.cache` = [buffer [1, 1, C, D], write offset int32 [1]], the
        row is written with a device-offset slice_update, the masks are built from the offset array, and every shape is static (decode, L <= the compile
        row limit: the whole buffer, or the `window` rows of a windowed layer through a device-offset slice; prefill: the host-known [start, n + L) view).
        Nothing reads the device and no python value changes between steps, so step_compile compiles the whole layer, attention included.
        Slot == position (left-aligned); rows past the offset are masked (visibility / validity), so the softmax runs over the same nonzero terms as the
        concatenating cache's."""
        B, L, _ = x.shape
        assert B == 1, "fixed-state KV cache: one row"
        buf, off = cache.cache[0], cache.cache[1]
        ring = bool(getattr(cache, "ring", False))                                 # KYOJIN_HY4_SWA_RING: windowed layer on a ring buffer (slot = position % rows), the rows read are gathered
        pkc = cache.cache[2] if len(cache.cache) > 2 else None                     # flag KYOJIN_HY4_FIXED_PKCACHE: the pooled keys of a kpool layer
        a16, ab16 = AF.proj16(self), AF.abs16(self)                                # flag attn_fp16 (+ attn_fp16_parts): projections / absorbs in fp16; the KV buffer dtype is the cache's own
        pf16 = L > int(os.environ.get("KYOJIN_STEP_COMPILE_ROWS", "16")) and not AF.ON and AF.pf_on() and "_kb16" in self.__dict__   # flag KYOJIN_HY4_PF_FP16: fp16 attention for prompt chunks only (never decode / verify rows); the buffer keeps its dtype
        kv32 = False
        if pf16:
            a16 = ab16 = True
            kv32 = AF.pf_kv32()                                                         # KYOJIN_HY4_PF_FP16_KV32: the buffer rows stay the stock fp32 products (bit-equal to the stock prefill); only q / scores / P @ V run in fp16
        from . import mla_prelude as MP
        pf = None
        if MP.ON and self.r:                                                            # flag mla_prelude_fuse: kv_a norm + both ropes + q split + row concat / clamp / cast in one launch
            ln = AF.call if a16 else (lambda m, v: m(v))
            qr = self.q_a_norm(ln(self.q_a_proj, x))
            qq, kva = ln(self.q_b_proj, qr), ln(self.kv_a_proj, x)
            if MP.eligible(L, kva, qq, buf.dtype, self.kv, self.r, self.nope, MlaAttention.PATHS):
                MlaAttention.PATHS["prelude_fuse"] = MlaAttention.PATHS.get("prelude_fuse", 0) + 1
                val, gate = self.kpool.rows(x, a16) if self.kp else (None, None)
                pf = MP.prelude(kva, qq, self.kv_a_norm.weight, self.kv_a_norm.eps, self.base, off, self.nh, self.kv, self.r, self.nope, buf.dtype, val, gate)
        if pf is not None:
            k, q_nope, q_pe = pf
            buf = FK.ring_write(buf, k, off, L) if ring else mx.slice_update(buf, k, off, (2,))
        else:
            qr, q_nope, q_pe, lat, k_pe = self.project(x, off.reshape(()), True, a16, kv32)
            row = [lat, k_pe]
            if self.kp:
                val, gate = self.kpool.rows(x, a16 and not kv32)
                row += [val[:, None], gate[:, None]]
            k = mx.concatenate(row, axis=-1) if len(row) > 1 else lat
            k = AF.sat(k).astype(buf.dtype) if (pf16 and not kv32) or buf.dtype == mx.float16 else k.astype(buf.dtype)
            buf = FK.ring_write(buf, k, off, L) if ring else mx.slice_update(buf, k, off, (2,))
        off2 = off + L
        cache.cache[0], cache.cache[1] = buf, off2
        if pkc is not None and self.kp:
            pkc = cache.cache[2] = self.kpool_sel().update_pk(pkc, buf, off, L, self.kv + self.r, self.kp["dx"])
        C, D = buf.shape[2], buf.shape[3]
        win = getattr(self, "window", 0) if getattr(self, "swa", False) else 0
        if L <= int(os.environ.get("KYOJIN_STEP_COMPILE_ROWS", "16")):
            if win and not self.kp:
                Sw = min(win + L - 1, C)                                                # L rows (a verify block) see win - 1 rows before the first one: win + L - 1 slots
                base = mx.maximum(off2 - Sw, 0)                                         # [1]: first row of the window ending at the new row
                kv = mx.take(buf, (base + KP.ar(Sw)) % C, axis=2) if ring else mx.slice(buf, base, (2,), (1, 1, Sw, D))
            else:
                Sw, base, kv = C, mx.zeros((1,), dtype=mx.int32), buf
        else:
            n = cache._n                                                                # eager prefill: the host offset (never traced: L > the compile limit)
            st = max(0, n - win + 1) if win and not self.kp else 0
            Sw, base, kv = n + L - st, st, (mx.take(buf, (st + KP.ar(n + L - st)) % C, axis=2) if ring else buf[:, :, st:n + L])
            if pf16 and kv.dtype != mx.float16:
                kv = AF.sat(kv)                                                         # the visible rows seen as fp16 for this chunk (earlier rows are fp16 values already: lossless)
        kpos = base + KP.ar(Sw)
        qpos = off.reshape(-1) + KP.ar(L)
        vis = kpos[None, :] <= qpos[:, None]
        if win:
            vis = vis & (qpos[:, None] - kpos[None, :] < win)
        mask = vis[None, None]
        lat, k_pe = kv[..., :self.kv], kv[..., self.kv:self.kv + self.r]
        if pf16 and MF.swa_on() and win and not self.kp:                                # KYOJIN_HY4_ATTN_FLASH_SWA=1: windowed layer, key list = the window
            o = self.attend_flash(q_nope, q_pe, kv, MF.window_ids(L, Sw, win))
            if o is not None:
                MlaAttention.PATHS["attn_flash_swa"] += 1
                return self._finish(o.transpose(0, 2, 1, 3), x, B, L, a16)
        from . import mla_decode_vec as MV
        vec = MV.MODE and MV.MIN_ROWS <= L <= MV.MAX_ROWS and not ring and not pf16
        if vec and win and not self.kp:                                                 # flag mla_decode_vec: the windowed layer's attention core in one launch over the window read from the buffer
            o = self._attend_vec(q_nope, q_pe, buf, off, 0, None, win, ab16)
            if o is not None:
                return self._finish(o.transpose(0, 2, 1, 3), x, B, L, a16)
        keep = None
        if self.kp:
            kp, dx, a = self.kp, self.kp["dx"], self.kv + self.r
            sel = self.kpool_sel()
            iq, w = self.kpool.query(x, qr, a16)
            from . import kpool_fuse as KF
            gmin = int(os.environ.get("KYOJIN_HY4_FIXED_GATHER_DEC" if L <= 16 else "KYOJIN_HY4_FIXED_GATHER", "0") or 0)
            if pf16 and MF.on() and not win:                                            # KYOJIN_HY4_ATTN_FLASH=1: the fused flash kernel over the kept key list (fp16 prompt path only)
                v_, g_ = kv[:, 0, :, a:a + dx], kv[:, 0, :, a + dx:a + 2 * dx]
                pools = sel.pools_fixed_top(iq, w, v_, g_, off, pk=pkc) if gmin and Sw >= gmin else sel.pools_fixed(iq, w, v_, g_, off, pk=pkc)       # radix top-k from KYOJIN_HY4_FIXED_GATHER slots on
                slots, ok = KP.gather_slots(pools)
                o = self.attend_flash(q_nope, q_pe, kv, mx.where(ok, slots, -1).astype(mx.int32))
                if o is not None:
                    MlaAttention.PATHS["attn_flash"] += 1
                    return self._finish(o.transpose(0, 2, 1, 3), x, B, L, a16)
                MlaAttention.PATHS["attn_flash_fallback"] = MlaAttention.PATHS.get("attn_flash_fallback", 0) + 1
                o = self.attend_gather(q_nope, q_pe, lat, k_pe, slots, ok).transpose(0, 2, 1, 3)
                return self._finish(o, x, B, L, a16)
            if gmin and Sw >= gmin and not a16:                                         # KYOJIN_HY4_FIXED_GATHER[_DEC]=N: from N key slots on, attend only the keys the indexer kept (top_k + the tail group), not all Sw; _DEC = decode / verify rows
                s = sel.pools_fixed_top(iq, w, kv[:, 0, :, a:a + dx], kv[:, 0, :, a + dx:a + 2 * dx], off, pk=pkc)
                slots, ok = KP.gather_slots(s, sort=not (L <= 16 and os.environ.get("KYOJIN_HY4_GATHER_NOSORT", "0") == "1"))
                MlaAttention.PATHS["fixed_gather"] = MlaAttention.PATHS.get("fixed_gather", 0) + 1
                if vec and MV.MODE >= 2:                                                # flag mla_decode_vec 2: the key ids of the gather path straight into the one-launch core
                    o = self._attend_vec(q_nope, q_pe, buf, off, 2, mx.where(ok, slots, -1).astype(mx.int32), 0, ab16)
                    if o is not None:
                        return self._finish(o.transpose(0, 2, 1, 3), x, B, L, a16)
                o = self.attend_gather(q_nope, q_pe, lat, k_pe, slots, ok).transpose(0, 2, 1, 3)
                return self._finish(o, x, B, L, a16)
            if KF.use(L, Sw):                                                           # flag kpool_fuse: selection + mask in one launch (left-aligned slots); stock selection past KYOJIN_KPOOL_FUSE_MAXS slots
                keep = sel.keep_fused_fixed(iq, w, kv[:, 0, :, a:a + dx], kv[:, 0, :, a + dx:a + 2 * dx], off, pk=pkc)
                MlaAttention.PATHS["kpool_fuse_fixed"] += 1
            else:
                keep = KP.keep_mask(sel.pools_fixed(iq, w, kv[:, 0, :, a:a + dx], kv[:, 0, :, a + dx:a + 2 * dx], off, pk=pkc))
        if vec and MV.MODE >= 2 and self.kp and keep is not None:                       # flag mla_decode_vec 2: the keep mask of the kpool layer (below the gather threshold) into the one-launch core
            o = self._attend_vec(q_nope, q_pe, buf, off, 1, keep, 0, ab16)
            if o is not None:
                return self._finish(o.transpose(0, 2, 1, 3), x, B, L, a16)
        from . import mla_attend_fused as MAF
        MlaAttention.PATHS["attend_seen"] = MlaAttention.PATHS.get("attend_seen", 0) + 1
        if MAF.ON and not ring:                                                         # flag mla_attend_fuse: scores + mask + keep + sinks + softmax + P @ latent in two launches
            kb, vb = self._absorbs(mx.float16 if ab16 else mx.float32)
            qa = _amm(q_nope.astype(kb.dtype), kb)
            if MAF.eligible(qa, q_pe, buf, self.kv, self.r, keep, MlaAttention.PATHS):
                MlaAttention.PATHS["attend_fuse"] = MlaAttention.PATHS.get("attend_fuse", 0) + 1
                o = _amm(MAF.attend(qa, q_pe, buf, off, keep, self.sinks, self.scale, win, self.kv, self.r), vb).transpose(0, 2, 1, 3)
                return self._finish(o, x, B, L, a16)
        if L > 16:                                                                      # prefill chunk on the fixed-state cache: the query blocks (KYOJIN_HY4_ATTN_QBLK[_SWA]) apply as on the concatenating cache; the window slice above already cut the keys, so attend's own slice is a no-op
            o = self.attend_qblocks(q_nope, q_pe, lat, k_pe, mask, keep).transpose(0, 2, 1, 3)
        else:
            o = self.attend(q_nope, q_pe, lat, k_pe, mask, keep, slice_ok=False, ab16=ab16).transpose(0, 2, 1, 3)
        return self._finish(o, x, B, L, a16)

    def _attend_fused(self, qa, q_pe, lat, k_pe, mask, keep, L, S):
        """attend() with the elementwise chain fused (flag above): probabilities = exp(sc - logaddexp(logsumexp(sc), sink)), the sink column never materialised."""
        MlaAttention.PATHS["attn_fuse"] += 1
        if isinstance(mask, str):
            from mlx_lm.models.base import create_causal_mask
            mask = create_causal_mask(L, S - L) if mask == "causal" else None
        s1 = qa @ lat.swapaxes(-1, -2)
        args = [s1, mx.array(self.scale, dtype=mx.float32)]
        if self.r:
            args.append((q_pe * self.scale) @ k_pe.swapaxes(-1, -2))
        kind = None if mask is None else ("bool" if mask.dtype == mx.bool_ else "add")
        if kind:
            args.append(mask)
        if keep is not None:
            args.append(keep)
        key = (kind, bool(self.r), keep is not None)
        if key not in _FUSED:
            _FUSED[key] = _scores_fn(*key)
        sc = _FUSED[key](*args)
        lse = mx.logsumexp(sc, axis=-1, keepdims=True)
        if self.sinks is not None:
            lse = mx.logaddexp(lse, self.sinks.astype(mx.float32)[None, :, None, None])
        p = _probs_fn(sc, lse).astype(lat.dtype)
        return (p @ lat) @ self._absorbs(lat.dtype)[1].swapaxes(-1, -2)

    def attend_qblocks(self, q_nope, q_pe, lat, k_pe, mask=None, keep=None):
        """KYOJIN_HY4_ATTN_QBLK=n (read per call, default 0 = off): a chunk above n query rows attends in blocks of n rows, each block over the keys up to its own last row
        only (causal: the keys after it are masked for every row of the block, so dropping them leaves the same nonzero terms). Memory of the [B, nh, rows, S] scores stays flat in the
        chunk size and the causal triangle inside the chunk is not computed. Masks: "causal" is rebuilt per block, an array mask / keep mask is sliced [rows, keys]."""
        blk = int(os.environ.get("KYOJIN_HY4_ATTN_QBLK", "0") or 0)
        if getattr(self, "swa", False):                                         # windowed layers: their own block size (KYOJIN_HY4_ATTN_QBLK_SWA), falling back to the common one
            blk = int(os.environ.get("KYOJIN_HY4_ATTN_QBLK_SWA", "") or blk)
        L, S = q_nope.shape[2], lat.shape[2]
        if blk <= 0 or L <= blk or not (mask is None or isinstance(mask, (str, mx.array))):
            return self.attend(q_nope, q_pe, lat, k_pe, mask, keep)
        MlaAttention.PATHS["qblocks"] += 1
        outs = []
        for a in range(0, L, blk):
            b = min(L, a + blk)
            sb = S - L + b                                                     # keys 0..sb-1 are the ones the last row of the block can see
            if isinstance(mask, str):
                from mlx_lm.models.base import create_causal_mask
                mb = create_causal_mask(b - a, S - L + a) if mask == "causal" else None
            elif mask is None:
                mb = None
            else:
                mb = mask[..., a:b, :sb]
            kb = None if keep is None else keep[..., a:b, :sb]
            outs.append(self.attend(q_nope[:, :, a:b], q_pe[:, :, a:b], lat[:, :, :sb], k_pe[:, :, :sb], mb, kb))
        return mx.concatenate(outs, axis=2)

    def __call__(self, x, mask=None, cache=None):
        if getattr(cache, "fixed_kv", False):
            return self._call_fixed(x, cache)
        B, L, _ = x.shape
        off = cache.offset if cache is not None else 0
        pf16 = L > 1 and not AF.ON and AF.pf_on() and "_kb16" in self.__dict__   # flag KYOJIN_HY4_PF_FP16: the fp16 path for a prompt chunk only; the cache keeps the decode dtype
        a16 = pf16 or (AF.ON and "_kb16" in self.__dict__)                    # flag attn_fp16: fp16 products and an fp16 KV row (saturated), fp32 accumulation / softmax
        kv32 = pf16 and AF.pf_kv32()                                           # KYOJIN_HY4_PF_FP16_KV32: the cache rows stay the fp32 products
        qr, q_nope, q_pe, lat, k_pe = self.project(x, off, a16=a16, kv32=kv32)
        row = [lat, k_pe]
        if self.kp:
            val, gate = self.kpool.rows(x, a16 and not kv32)
            row += [val[:, None], gate[:, None]]
        k = mx.concatenate(row, axis=-1) if len(row) > 1 else lat
        if a16 and not kv32:
            k = AF.sat(k)
            if pf16:
                k = k.astype(x.dtype)                                          # decode dtype in the cache (the values are fp16 numbers: exact)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        if pf16 and k.dtype != mx.float16:
            k = AF.sat(k)                                                      # the whole cache seen as fp16 for this chunk (earlier rows are fp16 values too: lossless)
        lat, k_pe = k[..., :self.kv], k[..., self.kv:self.kv + self.r]
        if (MF.swa_on() and a16 and L > 1 and B == 1 and getattr(self, "swa", False) and getattr(self, "window", 0) and not self.kp
                and (mask is None or isinstance(mask, (str, mx.array)))):
            o = self.attend_flash(q_nope, q_pe, k, MF.window_ids(L, k.shape[2], self.window))        # KYOJIN_HY4_ATTN_FLASH_SWA=1: windowed layer, key list = the window (the mask is the window by construction)
            if o is not None:
                MlaAttention.PATHS["attn_flash_swa"] += 1
                return self._finish(o.transpose(0, 2, 1, 3), x, B, L)
        keep = None
        if self.kp:
            kp, dx, a = self.kp, self.kp["dx"], self.kv + self.r
            sel = KP.KpoolSelect(self.kpool.ape, self.kpool.pool_norm, self.kpool.k_norm.eps, kp["kp"], kp["top_k"], kp["select_tail"],
                                 kp["n_rot"], kp["base"], True)
            iq, w = self.kpool.query(x, qr, a16)
            pfl = MF.on() and a16 and L > 1 and B == 1 and not getattr(self, "swa", False) and (mask is None or (isinstance(mask, str) and mask == "causal"))
            from . import kpool_fuse as KF
            vg = (k[:, 0, :, a:a + dx], k[:, 0, :, a + dx:a + 2 * dx])
            if KP.KEYCACHE and not pfl and os.environ.get("KYOJIN_HY4_KPOOL_GATHER", "0") != "1":      # flag kpool_keycache: pooled keys kept across tokens
                fused = KF.use(L)
                keep = sel.keep_cached(iq, w, cache, vg[0], vg[1], off, fused)
                if keep is not None:
                    MlaAttention.PATHS["kpool_fuse" if fused else "kpool_keycache"] += 1
                    o = self.attend(q_nope, q_pe, lat, k_pe, mask, keep).transpose(0, 2, 1, 3)
                    return self._finish(o, x, B, L)
            if KF.use(L) and not pfl and os.environ.get("KYOJIN_HY4_KPOOL_GATHER", "0") != "1":      # flag kpool_fuse: decode rows, selection + mask in one launch
                keep = sel.keep_fused(iq, w, vg[0], vg[1], off)
                MlaAttention.PATHS["kpool_fuse"] += 1
                o = self.attend(q_nope, q_pe, lat, k_pe, mask, keep).transpose(0, 2, 1, 3)
                return self._finish(o, x, B, L)
            pools = sel.pools(iq, w, vg[0], vg[1], off)
            if (os.environ.get("KYOJIN_HY4_KPOOL_GATHER", "0") == "1" and not getattr(self, "swa", False)
                    and (mask is None or (isinstance(mask, str) and mask == "causal"))):
                slots, ok = KP.gather_slots(pools)
                MlaAttention.PATHS["kpool_gather"] += 1
                o = self.attend_gather(q_nope, q_pe, lat, k_pe, slots, ok).transpose(0, 2, 1, 3)
                return self._finish(o, x, B, L)
            if MF.on() and a16 and L > 1 and B == 1 and not getattr(self, "swa", False) and (mask is None or (isinstance(mask, str) and mask == "causal")):
                slots, ok = KP.gather_slots(pools)                                  # KYOJIN_HY4_ATTN_FLASH=1: the fused flash kernel over the kept key list (fp16 path only)
                o = self.attend_flash(q_nope, q_pe, k, mx.where(ok, slots, -1).astype(mx.int32))
                if o is not None:
                    MlaAttention.PATHS["attn_flash"] += 1
                    return self._finish(o.transpose(0, 2, 1, 3), x, B, L)
            keep = KP.keep_mask(pools)
        o = self.attend_qblocks(q_nope, q_pe, lat, k_pe, mask, keep).transpose(0, 2, 1, 3)    # [B, L, nh, v]
        return self._finish(o, x, B, L)

    def _finish(self, o, x, B, L, g16=False):
        if self.gate:
            g = AF.call(self.gate_proj, x) if g16 or o.dtype == mx.float16 else self.gate_proj(x)
            o = o * mx.sigmoid(g.reshape(B, L, self.nh, self.hv))
        return self.o_proj(o.reshape(B, L, -1))
