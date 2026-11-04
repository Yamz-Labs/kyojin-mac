"""Fuse the projections that read the same activation (GDN in_proj_qkv / z / b / a, MLP gate / up, attention q / k / v) into ONE quantized matmul each:
the weights, scales and biases are concatenated along the output dim (identical group size / bits), so the numbers are bit-identical, and the per-layer
dispatch count drops (decode is launch-latency bound between the small kernels). The original modules are replaced by members of a FuseGroup: the first
member called on an activation computes the fused product, the others slice the cached result."""
import mlx.core as mx
import mlx.nn as nn


class FuseGroup:
    def __init__(self, mods):
        m0 = mods[0]
        assert all(isinstance(m, nn.QuantizedLinear) and m.bits == m0.bits and m.group_size == m0.group_size and "bias" not in m and m.biases is not None for m in mods)
        self.sizes = [m.weight.shape[0] for m in mods]
        self.w = mx.concatenate([m.weight for m in mods], axis=0)
        self.s = mx.concatenate([m.scales for m in mods], axis=0)
        self.b = mx.concatenate([m.biases for m in mods], axis=0) if getattr(m0, "biases", None) is not None else None
        self.gs, self.bits = m0.group_size, m0.bits
        self.offs = [0]
        for n in self.sizes:
            self.offs.append(self.offs[-1] + n)
        self.x = None
        self.out = None
        mx.eval(self.w, self.s, self.b)

    def get(self, x, i):
        if self.x is not x:
            self.out = mx.quantized_matmul(x, self.w, self.s, self.b, transpose=True, group_size=self.gs, bits=self.bits)
            self.x = x
        return self.out[..., self.offs[i]:self.offs[i + 1]]


ON = [True]                                                  # A/B switch (the original module stays inside the member)


class Member(nn.Module):
    def __init__(self, grp, i, orig):
        super().__init__()
        self._g, self._i = grp, i
        self.orig = orig

    def __call__(self, x):
        return self._g.get(x, self._i) if ON[0] else self.orig(x)


def _fuse(owner, names):
    grp = FuseGroup([owner[n] for n in names])
    for i, n in enumerate(names):
        owner[n] = Member(grp, i, owner[n])


def fuse_model(model, which=("gdn", "mlp", "attn")):
    lm = model.language_model
    n = 0
    for l in lm.model.layers:
        if "gdn" in which and l.is_linear:
            _fuse(l.linear_attn, ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]); n += 1
        if "attn" in which and not l.is_linear:
            _fuse(l.self_attn, ["q_proj", "k_proj", "v_proj"]); n += 1
        if "mlp" in which:
            _fuse(l.mlp, ["gate_proj", "up_proj"]); n += 1
    mx.clear_cache()
    return n
