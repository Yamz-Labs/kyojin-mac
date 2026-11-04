import sys, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mlx_lm import load
from mlx_lm.models.gated_delta import normalize_qk, compute_g
from kyojin_mlx import gdn_pf
model, _ = load(sys.argv[1]); lm = model.language_model
layer = next(l for l in lm.model.layers if l.is_linear).linear_attn
S = 256; mx.random.seed(3)
x = (mx.random.normal((1, S, lm.args.hidden_size)) * 0.7).astype(mx.bfloat16)
qkv = layer.in_proj_qkv(x); a = layer.in_proj_a(x); b = layer.in_proj_b(x)
cs = (mx.random.normal((1, 3, layer.conv_dim)) * 0.5).astype(mx.bfloat16)
ci = mx.concatenate([cs, qkv], axis=1); co = nn.silu(layer.conv1d(ci))
q, k, v = [t.reshape(1, S, h, 128) for t, h in zip(mx.split(co, [layer.key_dim, 2 * layer.key_dim], -1), [16, 16, 32])]
q, k = normalize_qk(q, k, inv_scale=128 ** -0.5, eps=1e-6)
g = compute_g(layer.A_log, a, layer.dt_bias); beta = mx.sigmoid(b)
q2, k2, v2, cn, g2, beta2 = gdn_pf.pre(qkv, cs, layer.conv1d.weight, a, b, layer.A_log, layer.dt_bias, 16, 32)
mx.eval(q, k, v, g, beta, q2, k2, v2, cn, g2, beta2, ci)
d = lambda p, r: (float(mx.max(mx.abs(p.astype(mx.float32) - r.astype(mx.float32)))), float(mx.mean((p == r).astype(mx.float32))))
for n, p, r in (("q", q2, q), ("k", k2, k), ("v", v2, v), ("g", g2, g), ("beta", beta2, beta), ("cn", cn, ci[:, -3:])): print(n, p.dtype, r.dtype, "max|d| %.4g bit-equal frac %.4f" % d(p, r))
print(layer.conv1d.weight.shape, layer.dt_bias.dtype, layer.A_log.dtype, layer.norm.weight.dtype)
from mlx_lm.models.gated_delta import gated_delta_update
st0 = (mx.random.normal((1, 32, 128, 128)) * 0.1).astype(mx.float32)
ys, ss = gated_delta_update(q, k, v, layer.in_proj_a(x), b, layer.A_log, layer.dt_bias, st0)
yf, sf = mx.fast.gated_delta_update(q, k, v, g, beta, st0)
yp, sp = mx.fast.gated_delta_update(q2, k2, v2, g2, beta2, st0)
ysp, ssp = gated_delta_update(q2, k2, v2, a, b, layer.A_log, layer.dt_bias, st0)
mx.eval(ys, ss, yf, sf, yp, sp, ysp, ssp)
print("stock-core vs fast-core (same stock inputs): y", d(yf, ys)[0], "|y|", float(mx.max(mx.abs(ys.astype(mx.float32)))), "state", d(sf, ss)[0])
print("fast-core on pre vs fast-core on stock inputs: y", d(yp, yf)[0], "state", d(sp, sf)[0])
print("stock-core on pre inputs vs stock-core stock inputs: y", d(ysp, ys)[0], "state", d(ssp, ss)[0])
z = layer.in_proj_z(x).reshape(1, S, 32, 128)
o_s = layer.norm(ys, z); o_f = gdn_pf.gated_norm(ys, z, layer.norm.weight, layer.norm.eps); mx.eval(o_s, o_f)
print("norm: max|d|", d(o_f, o_s), float(mx.max(mx.abs(o_s.astype(mx.float32)))))
