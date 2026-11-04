"""dec_keys_codes (keyscodes.py, dsa_decode.scores_rows_codes): CPU checks of everything around the Metal kernel, which needs a GPU and is checked on the Mac by
tools/cdecdeep_mac.py (bit equality against scores_rows(nax=False) over the unpacked keys, and the first-use self check in keyscodes._selfcheck).

Covered here: the flag (default 0, bound to keyscodes.ENABLED), flag 0 = the core's hook is never entered, the gate (what reaches the kernel), the slot addressing of a mu_keep
stack with non-identity slots (Plan.sess is the buffer slot, Plan.rank the session: the bug class of 2d40b8f), and a numpy transliteration of the kernel's tile loop (nibble
order, ue8m0 scale, LUT, mask skip, lens cut, slot base) against the dense reference."""
import os
import re

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kyojin_mlx import dsa_decode as DSA                                       # noqa: E402
from kyojin_mlx import dsv4_core as C                                          # noqa: E402
from kyojin_mlx import flags as FL                                             # noqa: E402
from kyojin_mlx import fp4pack                                                 # noqa: E402
from kyojin_mlx import keyscodes as KC                                         # noqa: E402
from kyojin_mlx import stackpack as SP                                         # noqa: E402
from kyojin_mlx.fp4pool import PackedView                                      # noqa: E402

NH, DI = 32, 128


def keys_world(slots=3, cap=96, seed=5):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((slots, cap, DI)).astype(np.float32)
    codes, scales = fp4pack.pack(mx.array(k), 32, False)
    dense = fp4pack.unpack(codes, scales, 32, False, mx.float16)
    mx.eval(codes, scales, dense)
    return codes, scales, dense


def stack_of(codes, scales):
    return SP.PackedStack(codes, scales, mx.float16, 32, False)


def test_flag_is_registered_default_off_and_bound():
    f = FL.FLAGS["dec_keys_codes"]
    assert f.env == "KYOJIN_DEC_KEYS_CODES" and f.default == "0" and f.kind == "bool" and f.runtime
    assert ("keyscodes", "ENABLED") == tuple(f.attrs[0][:2])
    assert KC.ENABLED == (os.environ.get("KYOJIN_DEC_KEYS_CODES", "0") not in ("0", ""))


def test_core_hook_is_behind_the_flag():
    """Flag 0 must not run a single new statement in the score branch: the hook is `if KC.ENABLED and ...` and the old test is `scores is None and <old condition>`."""
    src = open(C.__file__).read()
    assert "if KC.ENABLED and ISCORE_MMA and rt and t_max >= ISCORE_MMA_MIN:" in src
    assert "if scores is None and ISCORE_MMA and rt and t_max >= ISCORE_MMA_MIN and getattr(CD.dense_of(hand.index_k)" in src
    assert src.index("if KC.ENABLED and ISCORE_MMA") < src.index("if scores is None and ISCORE_MMA")


def test_scores_returns_none_when_the_flag_is_off(monkeypatch):
    codes, scales, _ = keys_world(slots=1)
    called = []
    monkeypatch.setattr(KC, "ENABLED", False)
    monkeypatch.setattr(DSA, "scores_rows_codes", lambda *a, **k: called.append(1))
    q = mx.zeros((1, NH, DI), dtype=mx.float16)
    assert KC.scores(q, mx.zeros((1, NH), dtype=mx.float16), PackedView(codes, scales, 96, mx.float16, None, 32, False),
                     mx.array([96], dtype=mx.int32), 96, None, mx.array([0], dtype=mx.uint32)) is None
    assert not called


def test_gate():
    codes, scales, dense = keys_world(slots=1)
    q = mx.zeros((2, NH, DI), dtype=mx.float16)
    view = PackedView(codes, scales, 96, mx.float16, None, 32, False)
    assert KC.applies(view, q, 96) and KC.applies(stack_of(codes, scales), q, 96)
    assert not KC.applies(dense, q, 96)                                                        # a dense layer has no codes
    assert not KC.applies(None, q, 96)
    assert not KC.applies(view, q, 97)                                                         # wider than the capacity
    assert not KC.applies(view, q.astype(mx.float32), 96)
    assert not KC.applies(view, mx.zeros((2, 16, DI), dtype=mx.float16), 96)                   # 16 heads: not the row kernel's geometry
    assert not KC.applies(PackedView(codes, scales, 96, mx.float16, None, 16, True), q, 96)    # the pool codec (block 16, e4m3) is not the key codec
    assert KC.applies(view, q.astype(mx.bfloat16), 96) is False                                # the view's dtype is fp16: a bf16 query needs bf16 keys


def test_dispatch_addresses_the_slot_not_the_session(monkeypatch):
    """mu_keep: sessions 0 / 1 live in slots 1 / 0 of a 3-slot stack. The kernel gets Plan.sess_arr (slots), the cut `lens` and the codes of the whole stack."""
    codes, scales, _ = keys_world(slots=3)
    plan = C.Plan([40, 41, 70], counts=[2, 1], slots=[1, 0], nslots=3)
    assert plan.sess == [1, 1, 0] and plan.rank == [0, 0, 1] and plan.NS == 3
    seen = {}

    def rec(q, w, c, s, lens, t_max, cand, sess):
        seen.update(codes=c, scales=s, sess=np.array(sess).tolist(), t_max=t_max, cand=cand)
        return mx.zeros((q.shape[0], t_max), dtype=q.dtype)

    monkeypatch.setattr(KC, "ENABLED", True)
    monkeypatch.setattr(KC, "_state", {"ok": True})
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(DSA, "scores_rows_codes", rec)
    q = mx.zeros((3, NH, DI), dtype=mx.float16)
    out = KC.scores(q, mx.zeros((3, NH), dtype=mx.float16), stack_of(codes, scales), mx.array([41, 42, 71], dtype=mx.int32), 96, None, plan.sess_arr)
    assert out is not None and seen["sess"] == [1, 1, 0] and seen["codes"].shape == (3, 96, 64) and seen["t_max"] == 96
    assert KC.HITS["codes"] >= 1


def test_selfcheck_failure_keeps_the_old_path(monkeypatch):
    if mx.default_device() == mx.Device(mx.gpu):
        pytest.skip("a real GPU runs the kernel: the self check passes")
    codes, scales, _ = keys_world(slots=1)
    monkeypatch.setattr(KC, "ENABLED", True)
    monkeypatch.setattr(KC, "_state", {"ok": None})
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    q = mx.zeros((1, NH, DI), dtype=mx.float16)
    # no GPU: the kernel cannot run, the self check turns the lever off and scores() hands the call back to the core
    got = KC.scores(q, mx.zeros((1, NH), dtype=mx.float16), PackedView(codes, scales, 96, mx.float16, None, 32, False),
                    mx.array([96], dtype=mx.int32), 96, None, mx.array([0], dtype=mx.uint32))
    assert got is None and KC._state["ok"] is False and KC.HITS["selfcheck_fail"] >= 1


# ------------------------------------------------------------------------------------------------------ the kernel's loop, transliterated
LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def emulate(q, w, codes, scales, lens, T, cand, sess):
    """The tile loop of _SCORE_ROWS_P in numpy: 8-key tiles, a masked tile skipped (-inf), the tile unpacked byte by byte, scores through the fp16 rounding points."""
    N = q.shape[0]
    Ck = codes.shape[1]
    out = np.full((N, T), np.nan, dtype=np.float32)
    for u in range(N):
        nn = min(int(lens[u]), T)
        base = int(sess[u])
        for kt in range(0, nn, 8):
            if cand is not None and not cand[u, kt]:
                out[u, kt:min(kt + 8, T)] = -np.inf
                continue
            tile = np.zeros((8, DI), dtype=np.float16)
            for r in range(8):
                kk = kt + r
                if kk >= nn:
                    continue
                for bi in range(64):
                    b = int(codes[base, kk, bi])
                    sv = np.float32(2.0) ** (np.float32(scales[base, kk, bi >> 4]) - np.float32(127.0))
                    for j, c in enumerate((b & 15, b >> 4)):
                        m = np.float32(LUT[c & 7])
                        tile[r, 2 * bi + j] = np.float16((-m if c & 8 else m) * sv)
            dots = tile.astype(np.float32) @ q[u].astype(np.float32).T                      # [8, 32]
            c = np.maximum(dots.astype(np.float16), np.float16(0))
            s = (c.astype(np.float16) * w[u].astype(np.float16)).astype(np.float16).astype(np.float32).sum(axis=1)
            for r in range(8):
                if kt + r < nn:
                    out[u, kt + r] = np.float16(s[r])
    return out


def test_kernel_loop_emulation_slots_mask_and_lens():
    codes, scales, dense = keys_world(slots=3, cap=96)
    rng = np.random.default_rng(1)
    q = rng.standard_normal((3, NH, DI)).astype(np.float16)
    w = np.abs(rng.standard_normal((3, NH))).astype(np.float16)
    lens = np.array([93, 41, 70], dtype=np.int32)
    sess = np.array([1, 1, 0], dtype=np.uint32)                                               # Plan.sess of slots [1, 0]
    blocks = rng.random((3, 96 // 8)) < 0.5
    cand = np.repeat(blocks, 8, axis=1)
    cn, sn, dn = np.array(codes), np.array(scales), np.array(dense).astype(np.float16)
    for cm in (None, cand):
        got = emulate(q, w, cn, sn, lens, 96, cm, sess)
        for u in range(3):
            n = int(lens[u])
            k = dn[int(sess[u]), :n].astype(np.float32)
            ref = np.maximum((k @ q[u].astype(np.float32).T).astype(np.float16), np.float16(0)).astype(np.float16)
            ref = (ref * w[u]).astype(np.float16).astype(np.float32).sum(axis=1)
            ref = ref.astype(np.float16).astype(np.float32)
            for t in range(n):
                if cm is not None and not cm[u, t - t % 8]:
                    assert got[u, t] == -np.inf
                else:
                    assert abs(got[u, t] - ref[t]) <= 2e-2 * max(1.0, abs(ref[t])), (u, t, got[u, t], ref[t])


def test_kernel_unpack_formula_is_fp4pack_unpack():
    """The kernel's per-byte formula (nibble order, sign, LUT, 2^(s - 127)) rebuilds fp4pack.unpack bit for bit, and the LUT in the Metal source is fp4pack's."""
    codes, scales, dense = keys_world(slots=2, cap=40)
    cn, sn, dn = np.array(codes), np.array(scales), np.array(dense)
    got = np.zeros_like(dn)
    for s in range(2):
        for kk in range(40):
            for bi in range(64):
                b = int(cn[s, kk, bi])
                sv = np.float32(2.0) ** (np.float32(sn[s, kk, bi >> 4]) - np.float32(127.0))
                for j, c in enumerate((b & 15, b >> 4)):
                    m = np.float32(LUT[c & 7])
                    got[s, kk, 2 * bi + j] = np.float16((-m if c & 8 else m) * sv)
    assert np.array_equal(got, dn)
    assert np.array_equal(np.array(fp4pack._LUT), np.array(LUT, dtype=np.float32))
    m = re.search(r"FP4_LUT\[8\] = \{([^}]*)\}", DSA._HEADER_P)
    assert [float(x.strip().rstrip("f")) for x in m.group(1).split(",")] == LUT
    src = DSA._SCORE_ROWS_P
    assert "exp2(float(sca[row * 4 + (bi >> 4)]) - 127.0f)" in src and "kbase = ulong(sess[u]) * Ck" in src and "(c0 & 8u) ? -m0 : m0" in src


def test_wrapper_checks_its_inputs():
    codes, scales, _ = keys_world(slots=1)
    q = mx.zeros((1, NH, DI), dtype=mx.float16)
    w = mx.zeros((1, NH), dtype=mx.float16)
    lens = mx.array([8], dtype=mx.int32)
    with pytest.raises(AssertionError):
        DSA.scores_rows_codes(q, w, codes, scales, lens, 200)                                  # wider than the capacity
    with pytest.raises(AssertionError):
        DSA.scores_rows_codes(q.astype(mx.float32), w, codes, scales, lens, 8)
