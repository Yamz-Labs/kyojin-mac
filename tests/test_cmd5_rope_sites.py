# SPDX-License-Identifier: Apache-2.0
"""cmd5b: the rope pairs that rope_fma does NOT cover are not under `fp contract(off)`, so they already compile to the same fma form as the stock mx.fast.rope
(cmd5a_REPORT listed them as the same defect class; reading them again shows they are not). Only the kernels built with dsv4_core._CONTRACT_OFF have the two-rounding
pair, and every one of those takes its rope text through `rope_src` (kv chain, q load, inverse rope, dec_qkv)."""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kyojin_mlx")


def src(name):
    return open(os.path.join(ROOT, name + ".py")).read()


def test_no_pragma_off_in_the_uncovered_rope_files():
    for f in ("fuse_attn", "dsa_attend", "ds_pf_idx"):
        assert "contract(off)" not in src(f) and "_CONTRACT_OFF" not in src(f), f


def test_pf_glue_rope_kernels_use_the_default_contraction():
    s = src("pf_glue")
    assert re.search(r'^_HEADER = ""', s, re.M)
    assert 'header=_HEADER)' in s                                   # the generic builder of q_rope / o_rope_inv / fq_t
    off = [m.start() for m in re.finditer(r"contract\(off\)", s)]
    # the pragma appears only for expand / combine / prenorm (mHC arithmetic, no rope pair)
    for i in off:
        line = s[:i].count("\n")
        ctx = "\n".join(s.splitlines()[max(0, line - 2):line + 2])
        assert any(k in ctx for k in ("_EXPAND", "_COMBINE", "_PRENORM", "expand", "mhc")), ctx


def test_decrows_serves_the_contracted_variant():
    s = src("decrows")
    assert re.search(r"^VARIANT = \(True, True\)", s, re.M)         # (fast trig, fp contract on): bit-equal to mx.fast.rope + rms_norm (tests/test_decrows.py)


def test_every_contract_off_rope_kernel_of_the_ds41_core_goes_through_rope_src():
    s = src("dsv4_core")
    for kernel in ("_KV_CHAIN", "_ATTN_PART_Q", "_ATTN_MERGE_Q"):
        assert re.search(rf"rope_src\({kernel}\)|rope_src\(_ATTN_PART_Q\)|rope_src\(_ATTN_MERGE_Q\)", s), kernel
    assert "rope_src(SOURCE)" in src("stagek2")
