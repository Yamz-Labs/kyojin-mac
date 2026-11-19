"""K-class GEMV / gather / glu on a stored matrix whose width k is not a multiple of the 512-weight chunk (flag KYOJIN_GEMVK_PARTIAL, default off): CPU proof.
Cause of the old route: DevPack.gemv_ok() was `k % 512 == 0` for every K-class format (gemvk_source reads NC = KIN / 512 whole chunks and 16 units per chunk), so a stored Q5_1 expert of
k = 640 / 704 (or a K type of k = 768) never got the GEMV / gather kernels: it went through the transient decode + matmul. The Metal output is tests/test_types2_metal.py
(gpu_archkit/types2_metal.sh)."""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_pack as FP, foreign_synth as SY

mx = pytest.importorskip("mlx.core")
FM = pytest.importorskip("kyojin_mlx.foreign_metal")
FL = pytest.importorskip("kyojin_mlx.foreign_linear")
FMTS = ("q4_1", "q5_0", "q5_1", "q2_0", "q5_k", "q6_k")
TN = {"q4_1": "Q4_1", "q5_0": "Q5_0", "q5_1": "Q5_1", "q2_0": "Q2_0", "q5_k": "Q5_K", "q6_k": "Q6_K"}


def _base_of(src: str) -> str:
    """Undo the four partial-chunk edits: what comes out is the stock source."""
    guard = "if (c * 512u + j * 8u >= uint(KIN)) { *(threadgroup uint4*)(xs + r * 640u + j * 8u + (j >> 2) * 8u) = uint4(0u); continue; }\n        "
    for new, old in (("NC = (uint(KIN) + 511u) / 512u;", "NC = uint(KIN) / 512u;"), ("u = min(ul, UPR - 1u); cu[j]", "u = ul; cu[j]"),
                     ("u = min((c + 1u) * 16u + ul, UPR - 1u); nu[j]", "u = (c + 1u) * 16u + ul; nu[j]"), (guard, "")):
        assert src.count(new) == 1, new
        src = src.replace(new, old)
    return src


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("gather,glu", [(False, False), (True, False), (True, True), (False, True)])
def test_partial_source_is_the_stock_source_plus_four_edits(fmt, gather, glu):
    stock = FM.gemvk_source(fmt, 2, gather, glu)
    part = FM.gemvk_source(fmt, 2, gather, glu, True)
    assert part != stock and _base_of(part) == stock
    assert FM.gemvk_source(fmt, 2, gather, glu, False) == stock            # the default is the stock text


def emulate(p: FP.Pack, x: np.ndarray, partial: bool) -> np.ndarray:
    """Index math of gemv_source's frame for one output row set: chunk c (512 weights), lane ul 0..15 owns unit u = 16 c + ul (clamped to UPR - 1 in the partial text), the staged x
    chunk is zero past KIN, lane dot = decoded unit (32 weights) . staged x[ul * 32 ...]; sums over chunks and lanes."""
    n, k = p.n, p.k
    UPR = k // 32
    NC = (k + 511) // 512 if partial else k // 512
    W = FP.dequant(p).astype(np.float64)
    out = np.zeros(n)
    xs_all = np.zeros(max(NC * 512, k) + 512)
    xs_all[:k] = x
    for c in range(NC):
        for ul in range(16):
            u = min(c * 16 + ul, UPR - 1) if partial else c * 16 + ul
            xv = xs_all[c * 512 + ul * 32:c * 512 + ul * 32 + 32]
            out += W[:, u * 32:u * 32 + 32] @ xv
    return out


@pytest.mark.parametrize("k", [64, 96, 320, 640, 704, 768, 1024, 1280])
def test_chunk_math_is_the_full_dot(k):
    fmts = [f for f in FMTS if f not in ("q5_k", "q6_k") or k % 256 == 0]
    if k % 64:
        fmts.remove("q2_0")
    for fmt in fmts:
        raw = SY.random_blocks(TN[fmt], 6, k, np.random.default_rng(k))
        raw3 = raw.reshape(6, k // GF.TYPES_BY_NAME[TN[fmt]][1], -1)
        SY.set_scales(TN[fmt], raw3, np.random.default_rng(1).uniform(1e-3, 1e-2, raw3.shape[:2]))
        p = GF.to_pack(TN[fmt], raw, k)
        x = np.random.default_rng(2).standard_normal(k).astype(np.float16).astype(np.float64)
        ref = FP.dequant(p).astype(np.float64) @ x
        np.testing.assert_allclose(emulate(p, x, True), ref, rtol=1e-12, atol=1e-9)
        if k % 512:
            assert not np.allclose(emulate(p, x, False), ref, rtol=1e-6, atol=1e-6) or k < 512      # the stock frame drops the tail (or reads past it)


def test_clamped_lanes_only_meet_zeros():
    """Every lane whose unit is clamped sits at an x position >= KIN, where the staged chunk is zero."""
    for k in (32, 64, 320, 640, 704, 768, 1280):
        UPR, NC = k // 32, (k + 511) // 512
        for c in range(NC):
            for ul in range(16):
                if c * 16 + ul > UPR - 1:
                    assert c * 512 + ul * 32 >= k


@pytest.mark.parametrize("fmt,k,n,flag,ok", [("q5_1", 640, 128, "1", True), ("q5_1", 704, 128, "1", True), ("q5_1", 640, 128, "0", False), ("q5_1", 1024, 128, "0", True),
                                              ("q2_0", 704, 128, "1", True), ("q6_k", 768, 128, "1", True), ("q6_k", 768, 128, "0", False), ("q5_1", 640, 8, "1", False),
                                              ("q4_0", 640, 128, "1", False), ("fp8", 640, 128, "1", True), ("q4_k", 768, 128, "1", False)])
def test_gemv_ok_routing(monkeypatch, fmt, k, n, flag, ok):
    monkeypatch.setenv("KYOJIN_GEMVK_PARTIAL", flag)
    d = SimpleNamespace(k=k, n=n, fmt=fmt)
    assert FL.DevPack.gemv_ok(d) is ok or FL.DevPack.gemv_ok(d) == ok


def test_part_selector():
    assert FL.DevPack._part(SimpleNamespace(k=640, fmt="q5_1")) and not FL.DevPack._part(SimpleNamespace(k=1024, fmt="q5_1")) and not FL.DevPack._part(SimpleNamespace(k=640, fmt="q4_k"))


def test_kernel_cache_key_and_name_differ():
    FL._gemvk_kern.cache_clear()
    a, b = FL._gemvk_kern("q5_1", 2), FL._gemvk_kern("q5_1", 2, partial=True)
    assert a is not b


def _metal_ok():
    return shutil.which("xcrun") is not None and subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode == 0


def wrap(fmt, body, gather, glu, k, n=128, r=1, nbr=2):
    """The body inside a kernel function with the names MLX's metal_kernel binds (built-ins as arguments, template ints as constants)."""
    ins = FM.gemvk_inputs(fmt, gather, glu)
    ty = {"x": "half", "x2": "half"}
    args = "".join(f"device const {ty.get(nm, 'uint')}* {nm} [[buffer({i})]], " for i, nm in enumerate(ins))
    return ("#include <metal_stdlib>\nusing namespace metal;\n" + FM.hdrk(fmt) + (FM.glu_helper(None) if glu else "") +
            f"constant constexpr int R = {r}, KIN = {k}, N = {n}, NBR = {nbr};\n"
            f"kernel void kk({args}device half* y [[buffer({len(ins)})]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], "
            "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], "
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n" + body + "\n}\n")


@pytest.mark.skipif(not _metal_ok(), reason="no Metal compiler")
@pytest.mark.parametrize("fmt", ("q5_1", "q2_0", "q6_k"))
@pytest.mark.parametrize("gather,glu", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("partial,k", [(False, 1024), (True, 640), (True, 704), (True, 1024)])
def test_generated_kernel_compiles_offline(tmp_path, fmt, gather, glu, partial, k):
    """The Metal compiler runs on the CPU: the whole generated kernel (stock and partial text) compiles (no GPU, no run)."""
    if fmt == "q6_k" and k % 256:
        pytest.skip("K types need k % 256")
    src = wrap(fmt, FM.gemvk_source(fmt, 2, gather, glu, partial), gather, glu, k)
    f = tmp_path / "k.metal"
    f.write_text(src)
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", str(f), "-o", str(tmp_path / "k.air")], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-800:]


@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("k", (640, 768, 1024, 1280))
def test_fmma_x_permutation_is_complete_off_the_512_chunk(fmt, k):
    """The few-row MMA (k % 128 == 0 only, no 512 chunk) reads x through fmma_xperm_index: it must be a permutation of 0..k-1 at k = 640 / 768 / 1280 too,
    so a partial 512 chunk cannot drop or repeat an input column."""
    idx = FM.fmma_xperm_index(fmt, k)
    assert idx.shape == (k,) and np.array_equal(np.sort(idx), np.arange(k))


@pytest.mark.parametrize("fmt", ("q4_1", "q5_0", "q5_1", "q2_0", "q5_k", "q6_k"))
@pytest.mark.parametrize("k,R,mma", [(640, 3, False), (640, 4, True), (704, 4, False), (768, 8, True), (1280, 16, True), (640, 17, False)])
def test_fmma_ok_route_at_off_chunk_widths(fmt, k, R, mma):
    """apply() order for a stored K-class matrix: few-row MMA when R >= 4 and k % 128 == 0, else the GEMV (partial-chunk text when k % 512 != 0)."""
    d = SimpleNamespace(k=k, n=128, fmt=fmt, E=0)
    assert bool(FL.DevPack.fmma_ok(d, R)) == mma
