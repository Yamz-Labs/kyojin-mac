"""decfuse2: the DS4.1 batched attention core with the kv chain and the q / inverse rope fused (KYOJIN_KV_CHAIN, KYOJIN_ROPE_FUSE).

The new path must be the old path bit for bit: the kv chain kernel (kv_norm + rope tail + fp8 round trip + ring scatter in one launch) and the rope
in the attention kernels reproduce attn.kv_norm / attn.rope / fake_quant_fp8_ue8m0 + the overlay scatter. Small synthetic attention module (dense
linears, the real DeepseekV4RoPE), depth 1 row and verify rows (R = 2 and 4), one and several sessions, a fresh and an existing ring.
"""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm")

from kyojin_mlx import dsv4_core as C

D, W, H, QL, G, OL = 512, 128, 4, 256, 2, 64


class _WoA(nn.Module):
    """MultiLinear stand-in: [n, G, 1, in_g] -> [n, G, 1, OL], one weight per group."""

    def __init__(self, rng):
        super().__init__()
        self.w = mx.array((rng.standard_normal((G, (H // G) * D, OL)) * 0.02).astype(np.float16))

    def __call__(self, x):
        return mx.stack([x[:, g] @ self.w[g] for g in range(G)], axis=1)


class _Attn:
    """The DeepseekV41Attention surface dsv4_core.core reads, with dense linears."""


def make_attn(seed=0):
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    rng = np.random.default_rng(seed)
    a = _Attn()
    a.layer_idx, a.n_heads, a.head_dim, a.window_size = 0, H, D, W
    a.config = SimpleNamespace(qk_rope_head_dim=64, index_topk=8)
    a.compress_ratio = 0                                      # window-only layer: the compressor / indexer do not change with these levers
    a.is_kv_source = a.is_index_source = False
    a.indexer = a.compressor = None
    a.o_groups, a.scale = G, D ** -0.5
    a.attn_sink = mx.zeros((H,), dtype=mx.float32)
    a.rope = DeepseekV4RoPE(64, 10000, None)
    a.wq_a, a.wq_b = nn.Linear(D, QL, bias=False), nn.Linear(QL, H * D, bias=False)
    a.wkv, a.wo_b = nn.Linear(D, D, bias=False), nn.Linear(G * OL, D, bias=False)
    a.q_norm, a.kv_norm = nn.RMSNorm(QL), nn.RMSNorm(D)
    a.wo_a = _WoA(rng)
    for name in ("wq_a", "wq_b", "wkv", "wo_b"):
        mod = getattr(a, name)
        mod.weight = mx.array((rng.standard_normal(mod.weight.shape) * 0.05).astype(np.float16))
    for name in ("q_norm", "kv_norm"):
        mod = getattr(a, name)
        mod.weight = mx.array((rng.standard_normal(mod.weight.shape) * 0.05).astype(np.float16))
    return a


class _Row:
    step = 32


class _St:
    """The stacked-cache surface core() uses (win / pend / hand / commit_layer)."""

    def __init__(self, win, counts):
        self.win, self.pend, self._counts = [win], {}, counts
        self.hand = SimpleNamespace(index_k=None, candidates=None, compress_kv=None, topk_idxs=None)
        self.rows, self.committed = [_Row()], None

    def commit_layer(self, l, last, p_last=None, idx=None, slots=None):
        self.committed = (l, p_last)


def core_run(monkeypatch, attn, x, pos, counts, win, kvchain, ropefuse):
    st = _St(win, counts)
    plan = C.Plan(pos, counts)
    monkeypatch.setattr(C, "stack_rows", lambda cache: st)
    monkeypatch.setattr(C, "plan_of", lambda s: plan)
    monkeypatch.setattr(C, "KV_CHAIN", kvchain)
    monkeypatch.setattr(C, "ROPE_FUSE", ropefuse)
    o = C.core(attn, x, None)
    mx.eval(o)
    ring = st.pend[0].win
    mx.eval(ring)
    return np.array(o), np.array(ring), st.committed


CASES = [                                                   # (n, positions, counts, ring)
    ("depth1-fresh", 1, [10], None, None),
    ("depth1-ring", 1, [130], None, True),
    ("plain-2sessions", 2, [40, 300], None, True),
    ("verify-r2", 2, [40, 41], [2], True),
    ("verify-r4", 4, [40, 41, 42, 43], [4], True),
    ("verify-2sessions", 4, [40, 41, 90, 91], [2, 2], True),
    ("verify-r4-fresh", 4, [0, 1, 2, 3], [4], None),
]


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
@pytest.mark.parametrize("name,n,pos,counts,ring", CASES)
@pytest.mark.parametrize("kvchain,ropefuse", [(True, False), (False, True), (True, True)])
def test_core_fused_paths_bit_equal(monkeypatch, name, n, pos, counts, ring, kvchain, ropefuse):
    rng = np.random.default_rng(7)
    attn = make_attn()
    x = mx.array((rng.standard_normal((n, 1, D)) * 0.1).astype(np.float16))
    win = mx.array((rng.standard_normal((len(counts) if counts else n, W, D)) * 0.1).astype(np.float16)) if ring else None
    ref_o, ref_w, ref_c = core_run(monkeypatch, attn, x, pos, counts, win, False, False)
    got_o, got_w, got_c = core_run(monkeypatch, attn, x, pos, counts, win, kvchain, ropefuse)
    assert np.array_equal(ref_o, got_o), (name, int((ref_o != got_o).sum()))
    assert np.array_equal(ref_w, got_w), (name, int((ref_w != got_w).sum()))
    assert ref_c == got_c


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_attend_qrope_and_inverse_rope_bit_equal():
    """The q rope in the attention kernel's q load and the inverse rope in the merge epilogue equal mx.fast.rope on the fp16 q / o."""
    from mlx_vlm.models.deepseek_v4.language import DeepseekV4RoPE
    rng = np.random.default_rng(11)
    N, Cp, K = 4, 40, 8
    rope = DeepseekV4RoPE(64, 10000, None)
    pos = [5, 130, 300, 127]
    pa = mx.array(pos, dtype=mx.int32)
    q = mx.array((rng.standard_normal((N, H, D)) * 0.1).astype(np.float16))
    win = mx.array((rng.standard_normal((N, W, D)) * 0.1).astype(np.float16))
    pool = mx.array((rng.standard_normal((N, Cp, D)) * 0.1).astype(np.float16))
    idx = mx.array(np.sort(rng.choice(Cp, (N, K), replace=False), axis=1).astype(np.int32))
    sink = mx.array(rng.standard_normal(H).astype(np.float32))
    plan = C.Plan(pos)
    rt, it = plan.attn_rope(rope)
    ref = C.attend(rope(q[:, :, None, :], pa).reshape(N, H, D), win, pool, idx, pa, sink, 0.044, K)
    ref = rope(ref.reshape(N, H, 1, D), pa, inverse=True).reshape(N, H, D)
    got = C.attend(q, win, pool, idx, pa, sink, 0.044, K, rt=rt, it=it)
    assert np.array_equal(np.array(ref), np.array(got)), int((np.array(ref) != np.array(got)).sum())


@pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")
def test_ablit_hook_composes_with_the_fused_paths(monkeypatch):
    """ablit projects the attention output (after wo_b) before hc_expand; the core only changes how that output is computed, so the hook must
    see the same array and keep applying (a real projection, not the identity)."""
    from kyojin_mlx import ablit
    rng = np.random.default_rng(13)
    attn = make_attn()
    x = mx.array((rng.standard_normal((2, 1, D)) * 0.1).astype(np.float16))
    win = mx.array((rng.standard_normal((1, W, D)) * 0.1).astype(np.float16))
    old = core_run(monkeypatch, attn, x, [40, 41], [2], win, False, False)[0]
    new = core_run(monkeypatch, attn, x, [40, 41], [2], win, True, True)[0]
    R = mx.array(rng.standard_normal((1, D)).astype(np.float32))
    R = R / mx.linalg.norm(R, axis=-1, keepdims=True)
    blk = SimpleNamespace(_ablit=SimpleNamespace(wa=0.5, wm=0.0, R=R, mua=None, mum=None))
    a_old = np.array(ablit.attn_out(blk, mx.array(old)))
    a_new = np.array(ablit.attn_out(blk, mx.array(new)))
    assert np.array_equal(a_old, a_new)
    assert not np.array_equal(a_old, old)                      # the hook is armed and does project
    plain = SimpleNamespace(_ablit=None)
    assert np.array_equal(np.array(ablit.attn_out(plain, mx.array(old))), old)
