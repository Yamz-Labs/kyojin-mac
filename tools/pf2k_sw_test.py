import mlx.core as mx, numpy as np
from mlx_lm.models.activations import swiglu
rng = np.random.default_rng(3)
g = mx.array((rng.standard_normal(1 << 20) * 3).astype(np.float16)); u = mx.array((rng.standard_normal(1 << 20) * 3).astype(np.float16))
ref = swiglu(g, u)
VAR = {
 "half": "const half g = G[i], u = U[i]; const half e = 1 / (1 + metal::precise::exp(metal::abs(g))); const half sg = (g < 0) ? e : 1 - e; O[i] = (g * sg) * u;",
 "float_all": "const float g = G[i], u = U[i]; const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(g))); const float sg = (g < 0.0f) ? e : 1.0f - e; O[i] = half((g * sg) * u);",
 "f_sig_half_rest": "const float gf = G[i]; const float ef = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf))); const half sg = half((gf < 0.0f) ? ef : 1.0f - ef); const half g = G[i], u = U[i]; O[i] = (g * sg) * u;",
 "f_sig_f_mul_half_u": "const float gf = G[i]; const float ef = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf))); const float sg = (gf < 0.0f) ? ef : 1.0f - ef; const half s2 = half(gf * sg); O[i] = s2 * U[i];",
 "half_sig_f_mul": "const half g = G[i]; const half e = 1 / (1 + metal::precise::exp(metal::abs(g))); const half sg = (g < 0) ? e : 1 - e; const half s2 = half(float(g) * float(sg)); O[i] = half(float(s2) * float(U[i]));",
 "sil_f_then_mul_half": "const float gf = G[i]; const float ef = 1.0f / (1.0f + metal::precise::exp(metal::abs(gf))); const float sg = (gf < 0.0f) ? ef : 1.0f - ef; const half s2 = half(gf * sg); O[i] = half(float(s2) * float(U[i]));",
}
for name, body in VAR.items():
    k = mx.fast.metal_kernel(name="t_" + name, input_names=["G", "U"], output_names=["O"], source="uint i = thread_position_in_grid.x;\n" + body)
    o = k(inputs=[g, u], grid=(1 << 20, 1, 1), threadgroup=(256, 1, 1), output_shapes=[g.shape], output_dtypes=[mx.float16])[0]
    print(name, "nbad", int((o != ref).sum().item()))
