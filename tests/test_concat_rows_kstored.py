# SPDX-License-Identifier: Apache-2.0
"""sanitize-fused linears (GLM-5.3 shared gate+up, KDA q/k/v and f_a / b / g_a, MLA qkv_a) of GGUF types: same type -> stored bytes concatenated (exact);
mixed types -> a Split of stored members (no decoded copy); a mixed group that cannot be split raises."""
import numpy as np
import pytest

from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_pack as FP


def _raw(rng, n, k):
    nb = k // 256
    r = rng.integers(0, 256, (n, nb * 210), dtype=np.uint8)
    r.reshape(n, nb, 210)[:, :, 208:210] = np.array([0x00, 0x2C], np.uint8)       # fp16 d = 0.0625: finite
    return r


def _q8(rng, n, k):
    return FP.Pack("q8_0", n, k, rng.integers(0, 256, (n, k), dtype=np.uint8), (rng.standard_normal((n, k // 32)) * 0.01).astype(np.float16))


def test_same_k_type_concat_is_exact():
    rng = np.random.default_rng(0)
    a, b = GF.to_pack("Q6_K", _raw(rng, 8, 512), 512), GF.to_pack("Q6_K", _raw(rng, 4, 512), 512)
    p = FL._concat_rows_exact([a, b])
    assert (p.fmt, p.n, p.k) == ("q6_k", 12, 512)
    assert np.array_equal(FP.dequant(p), np.concatenate([FP.dequant(a), FP.dequant(b)], 0))


def test_same_q8_0_concat_is_stored_and_exact():          # GLM-5.3 KDA fbg_a: f_a / beta / g_a all Q8_0 (34 / 34 in UD-IQ3_XXS)
    rng = np.random.default_rng(2)
    a, b, c = _q8(rng, 8, 512), _q8(rng, 4, 512), _q8(rng, 4, 512)
    p = FL._concat_rows_exact([a, b, c])
    assert (p.fmt, p.n, p.k) == ("q8_0", 16, 512)
    assert np.array_equal(FP.dequant(p), np.concatenate([FP.dequant(x) for x in (a, b, c)], 0))


def test_mixed_types_stay_stored_as_a_split():           # GLM-5.3 MLA qkv_a: attn_q_a Q6_K + attn_kv_a_mqa Q8_0
    rng = np.random.default_rng(1)
    a, b = GF.to_pack("Q6_K", _raw(rng, 8, 512), 512), _q8(rng, 4, 512)
    p = FL._concat_rows_exact([a, b])
    assert isinstance(p, FL.Split) and (p.fmt, p.n, p.k) == ("split", 12, 512)
    assert p.packs[0] is a and p.packs[1] is b                                         # the stored members themselves: no fp16 matrix anywhere
    with pytest.raises(TypeError):
        FL.ForeignLinear(p)


def test_mixed_group_that_cannot_be_split_raises():
    rng = np.random.default_rng(3)
    a, b = GF.to_pack("Q6_K", _raw(rng, 8, 512), 512), _q8(rng, 4, 256)
    with pytest.raises(ValueError, match="cannot be split"):
        FL._concat_rows_exact([a, b])
