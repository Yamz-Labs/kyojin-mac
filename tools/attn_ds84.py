"""hd=256 prefill flash attention: MLX's NAX head-dim-split kernel (attention_nax_dsplit) forked as an mx.fast.metal_kernel body, headers inlined from the installed mlx.
Step 1 reproduces stock (same tiles) so that edits (exchange, tiles, loads) can be A/B'd against it. usage: attn_ds84.py [o] [variant...]"""
import os, re, sys, time, struct, functools
import mlx.core as mx
_INC = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
def _read(rel):
    t = open(os.path.join(_INC, rel)).read()
    t = re.sub(r'^\s*#\s*include\s+"mlx/[^"]*"\s*$', "", t, flags=re.M)
    return t.replace("#pragma once", "")

@functools.lru_cache(maxsize=None)
def _parts():
    kern = _read("steel/attn/kernels/steel_attention_nax.h")
    a = kern.index("[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax_dsplit(")
    sig_end = kern.index("{ // clang-format on", a) + len("{ // clang-format on")
    body = kern[sig_end:kern.rindex("}")]
    head = kern[:kern.index("// clang-format off")]                        # helper structs (ops), before the kernels
    head = re.sub(r"constant bool \w+ \[\[function_constant\(\d+\)\]\];", "", head)
    inc = "\n".join(_read(f) for f in ("steel/defines.h", "steel/utils/type_traits.h", "steel/utils/integral_constant.h", "steel/attn/params.h", "steel/attn/nax.h", "steel/attn/transforms.h"))
    return inc, head, body

def header(align_q, align_k, causal=True):
    inc, head, _ = _parts()
    consts = f"constant constexpr bool align_Q = {str(align_q).lower()}; constant constexpr bool align_K = {str(align_k).lower()}; constant constexpr bool has_mask = false; constant constexpr bool do_causal = {str(causal).lower()}; constant constexpr bool has_sinks = false; constant constexpr bool save_lse = false;\n"
    return "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp;\n" + inc + "\nusing namespace mlx::steel;\n" + consts + head + "\n"

PRE = """
  const device AttnParams* params = (const device AttnParams*)params_buf;
  const uint simd_lane_id = thread_index_in_simdgroup;
  const uint simd_group_id = simdgroup_index_in_threadgroup;
  const uint3 tid = threadgroup_position_in_grid;
  const uint3 lid = thread_position_in_threadgroup;
  const device AttnMaskParams* mask_params = nullptr; const device float* mask = nullptr; const device T* sinks = nullptr; device float* LSE = nullptr;
  using MaskType = float; using AccumType = float;
"""

def make_params(B, H, Hkv, qL, kL, D, BQ, BK, scale):
    NQ = (qL + BQ - 1) // BQ; NK = (kL + BK - 1) // BK
    f = struct.pack("<6i", B, H, D, qL, kL, H // Hkv) + struct.pack("<f", scale) + struct.pack("<7i", NQ, NK, qL // BQ, kL // BK, qL - (qL // BQ) * BQ, kL - (kL // BK) * BK, kL - qL)
    st = lambda l: struct.pack("<3q", *l)
    f += st([H * qL * D, qL * D, D]) + st([Hkv * kL * D, kL * D, D]) + st([Hkv * kL * D, kL * D, D]) + st([H * qL * D, qL * D, D])
    return mx.array(list(f), dtype=mx.uint8)

@functools.lru_cache(maxsize=None)
def _kernel(name, align_q, align_k, edits):
    _, _, body = _parts()
    for old, new in edits:
        assert old in body, old
        body = body.replace(old, new)
    return mx.fast.metal_kernel(name=name, input_names=["Q", "K", "V", "params_buf"], output_names=["O"], header=header(align_q, align_k), source=PRE + body)

def attention(q, k, v, scale, BQ=64, BK=32, WM=4, WN=2, edits=(), tag="base"):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    kern = _kernel(f"attn_ds84_{tag}_{BQ}_{BK}_{WM}_{WN}", qL % BQ == 0, kL % BK == 0, tuple(edits))
    p = make_params(B, H, Hkv, qL, kL, D, BQ, BK, scale)
    NQ = (qL + BQ - 1) // BQ
    return kern(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN)],
                grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

def bench(f, it=3, blocks=6):
    mx.eval(f()); best = 1e9
    for _ in range(blocks):
        mx.synchronize(); t0 = time.perf_counter(); ys = [f() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
    return best

if __name__ == "__main__":
    o = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    L = 2048; N = o + L
    q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
    S = mx.fast.scaled_dot_product_attention
    ref = S(q, k, v, scale=0.0625, mask="causal"); y = attention(q, k, v, 0.0625); mx.eval(ref, y)
    print("max|d|", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), "ref max", float(mx.abs(ref.astype(mx.float32)).max()))
    fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
    for n, f in (("mlx", lambda: S(q, k, v, scale=0.0625, mask="causal")), ("fork", lambda: attention(q, k, v, 0.0625)), ("mlx", lambda: S(q, k, v, scale=0.0625, mask="causal")), ("fork", lambda: attention(q, k, v, 0.0625))):
        t = bench(f); print(f"{n}: {t*1e3:.2f} ms {fl/t/1e12:.2f} TF", flush=True)
