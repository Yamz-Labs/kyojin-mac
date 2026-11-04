"""MLX's NAX flash attention template (steel_attention_nax.h, attention_nax, no d-split) instantiated for hd=256 with our own tile shapes via mx.fast.metal_kernel.
Headers are read from the installed mlx package and inlined. The kernel function is turned into a device function and called from the metal_kernel body.
usage (bench): attn_nax84.py [o] [BQ] [BK] [WM] [WN]"""
import os, re, sys, time, struct, functools
import mlx.core as mx
_INC = os.path.join(os.path.dirname(mx.__file__), "include", "mlx", "backend", "metal", "kernels")
def _read(rel):
    t = open(os.path.join(_INC, rel)).read()
    t = re.sub(r'^\s*#\s*include\s+"mlx/[^"]*"\s*$', "", t, flags=re.M)
    return t.replace("#pragma once", "")

@functools.lru_cache(maxsize=None)
def header(do_causal=True):
    kern = _read("steel/attn/kernels/steel_attention_nax.h")
    # keep only the non-dsplit kernel
    kern = kern[:kern.index("template <", kern.index("// through threadgroup memory") - 2000 if "// through threadgroup memory" in kern else len(kern)) ] if False else kern
    a = kern.index("[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(")
    b = kern.index("template <", a)                                           # next template = dsplit
    head, body = kern[:a], kern[a:b]
    head = re.sub(r"constant bool (\w+) \[\[function_constant\(\d+\)\]\];", lambda m: f"constant constexpr bool {m.group(1)} = {'true' if m.group(1) == 'do_causal' and do_causal else 'false'};", head)
    # signature -> plain device function
    sig_end = body.index("{ // clang-format on")
    sig = body[:sig_end]
    sig = sig.replace("[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void", "METAL_FUNC void")
    sig = re.sub(r"\s*const constant AttnMaskParams\* mask_params \[\[buffer\(5\), function_constant\(has_mask\)\]\],", "", sig)
    sig = re.sub(r"\s*const device MaskType\* mask \[\[buffer\(6\), function_constant\(has_mask\)\]\],", "", sig)
    sig = re.sub(r"\s*const device T\* sinks \[\[buffer\(7\), function_constant\(has_sinks\)\]\],", "", sig)
    sig = re.sub(r"\s*device float\* lse \[\[buffer\(8\), function_constant\(save_lse\)\]\],", "", sig)
    sig = re.sub(r"\s*\[\[[^\]]*\]\]", "", sig).replace("const constant AttnParams*", "const device AttnParams*")
    rest = body[sig_end:]
    # has_mask / has_sinks / save_lse are constexpr false: the guarded code must still compile -> declare dummies
    rest = rest.replace("{ // clang-format on", "{", 1)
    pre = "  const device AttnMaskParams* mask_params = nullptr; const device MaskType* mask = nullptr; const device T* sinks = nullptr; device float* lse = nullptr;\n"
    rest = rest.replace("{", "{\n" + pre, 1) if rest.lstrip().startswith("{") else rest
    inc = "\n".join(_read(f) for f in ("steel/defines.h", "steel/utils/type_traits.h", "steel/utils/integral_constant.h", "steel/attn/params.h", "steel/attn/nax.h", "steel/attn/transforms.h"))
    return "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp;\n" + inc + "\nusing namespace mlx::steel;\n" + head + "\n" + sig + rest + "\n"

_SRC = """
  attention_nax<T, BQ, BK, BD, BV, WM, WN, T, float>(Q, K, V, O, (const device AttnParams*)params, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, threadgroup_position_in_grid, thread_position_in_threadgroup);
"""

def make_params(B, H, Hkv, qL, kL, D, qL_off, BQ, BK, scale):
    NQ = (qL + BQ - 1) // BQ; NK = (kL + BK - 1) // BK
    ints = [B, H, D, qL, kL, H // Hkv, 0, NQ, NK, qL // BQ, kL // BK, qL - (qL // BQ) * BQ if qL % BQ else BQ, kL - (kL // BK) * BK if kL % BK else BK, qL_off]
    f = struct.pack("<iiiiii", *ints[:6]) + struct.pack("<f", scale) + struct.pack("<iiiiiii", *ints[7:14])
    f += b"\0" * (-len(f) % 8)
    st = lambda l: struct.pack("<qqq", *l)
    f += st([H * qL * D, qL * D, D]) + st([Hkv * kL * D, kL * D, D]) + st([Hkv * kL * D, kL * D, D]) + st([H * qL * D, qL * D, D])
    return mx.array(list(f), dtype=mx.uint8)

def attention(q, k, v, scale, BQ=32, BK=16, WM=2, WN=1):
    B, H, qL, D = q.shape; Hkv, kL = k.shape[1], k.shape[2]
    kern = mx.fast.metal_kernel(name=f"attn_nax84_{BQ}_{BK}_{WM}_{WN}", input_names=["Q", "K", "V", "params"], output_names=["O"], header=header(), source=_SRC)
    p = make_params(B, H, Hkv, qL, kL, D, kL - qL, BQ, BK, scale)
    NQ = (qL + BQ - 1) // BQ
    return kern(inputs=[q, k, v, p], template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("BV", D), ("WM", WM), ("WN", WN)],
                grid=(NQ * 32 * WM * WN, H, B), threadgroup=(32 * WM * WN, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype])[0]

if __name__ == "__main__":
    o = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    BQ, BK, WM, WN = [int(x) for x in sys.argv[2:6]] if len(sys.argv) > 5 else (32, 16, 2, 1)
    L = 2048; N = o + L
    q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16); k = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 4, N, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")
    y = attention(q, k, v, 0.0625, BQ, BK, WM, WN); mx.eval(ref, y)
    print("max|d|", float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()), "ref max", float(mx.abs(ref.astype(mx.float32)).max()))
    fl = 4 * L * (o + (L + 1) / 2) * 16 * 256
    def bench(f, it=3, blocks=5):
        mx.eval(f()); best = 1e9
        for _ in range(blocks):
            mx.synchronize(); t0 = time.perf_counter(); ys = [f() for _ in range(it)]; mx.eval(ys); best = min(best, (time.perf_counter() - t0) / it)
        return best
    for n, f in (("mlx", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal")), ("fork", lambda: attention(q, k, v, 0.0625, BQ, BK, WM, WN))):
        t = bench(f); print(f"{n}: {t*1e3:.2f} ms {fl/t/1e12:.2f} TF")
