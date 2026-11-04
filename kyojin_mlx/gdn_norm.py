# SPDX-License-Identifier: Apache-2.0
"""Fused GDN output norm: Qwen3NextRMSNormGated (rms_norm(h, w) then silu(gate) * x) as ONE Metal launch.
The checkpoint keeps the norm weight in fp32, so mx.fast.rms_norm emits an fp32 tensor and the stock path costs three
passes over fp32 intermediates (3x slower than the bf16-weight MLX pack at 4K rows). This kernel reproduces the stock
result bit for bit: rms_norm single-row order (4 contiguous elements per lane, sequential x*x, simd_sum, precise rsqrt),
fp32 weight multiply, and MLX's sigmoid (precise exp of |g|, 1/(1+e) mirrored for g >= 0). KYOJIN_GDN_NORM=0 restores
the stock module call."""
import os

import mlx.core as mx

ENABLED = os.environ.get("KYOJIN_GDN_NORM", "1") != "0"
_SRC = """
    uint row = thread_position_in_grid.y;
    uint l = thread_position_in_threadgroup.x;
    const device InT* hp = h + row * 128 + 4 * l;
    const device InT* zp = z + row * 128 + 4 * l;
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) { xs[i] = static_cast<float>(hp[i]); acc += xs[i] * xs[i]; }
    acc = simd_sum(acc);
    float nrm = metal::precise::rsqrt(acc / 128.0f + eps[0]);
    for (int i = 0; i < 4; i++) {
        float o = w[4 * l + i] * (xs[i] * nrm);
        float g = static_cast<float>(zp[i]);
        float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        float s = g * ((g < 0.0f) ? e : 1.0f - e);
        out[row * 128 + 4 * l + i] = static_cast<InT>(s * o);
    }
"""
_kern = None
_eps = {}


def gated_norm(h: mx.array, gate: mx.array, weight: mx.array, eps: float) -> mx.array:
    global _kern
    if _kern is None:
        _kern = mx.fast.metal_kernel(name="kyojin_gdn_norm", input_names=["h", "z", "w", "eps"], output_names=["out"],
                                     source=_SRC, header="#pragma METAL fp contract(off)\n")
    e = _eps.get(eps)
    if e is None:
        e = _eps[eps] = mx.array([eps], dtype=mx.float32)
    return _kern(inputs=[h, gate, weight, e], template=[("InT", h.dtype)], grid=(32, h.size // 128, 1), threadgroup=(32, 1, 1),
                 output_shapes=[h.shape], output_dtypes=[h.dtype])[0]


def install():
    """Route Qwen3NextRMSNormGated through the fused kernel when its shape/dtype contract holds (else the stock call)."""
    from mlx_lm.models import qwen3_next as QN
    cls = QN.Qwen3NextRMSNormGated
    if not ENABLED or getattr(cls, "_kyojin_fused", False):
        return
    stock = cls.__call__

    def call(self, hidden_states, gate=None):
        if (gate is not None and mx.default_device() == mx.gpu and self.weight.dtype == mx.float32 and self.weight.size == 128
                and hidden_states.shape[-1] == 128 and gate.shape == hidden_states.shape
                and hidden_states.dtype in (mx.float16, mx.bfloat16) and gate.dtype == hidden_states.dtype):
            return gated_norm(hidden_states, gate, self.weight, self.eps)
        return stock(self, hidden_states, gate)

    cls.__call__ = call
    cls._kyojin_fused = True
