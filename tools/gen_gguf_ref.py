#!/usr/bin/env python3
"""Regenerate tests/data/gguf_ggml_ref.npz: raw blocks + the float32 values ggml's OWN dequantize function returns for them.
usage: gen_gguf_ref.py <llama.cpp checkout at the pinned commit> <work dir>      (gcc + g++ only; no cmake, no GPU, no model file)
Builds ggml.c, ggml-quants.c, ggml-threading.cpp straight with gcc; three backend symbols that ggml.c references are stubbed (never called)."""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import foreign_gguf as GF  # noqa: E402

PIN = "29c720eaf72fcf8e4e61a56a88b994d92668261d"
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"F32", "F16", "BF16", "I8", "I16", "I32", "I64", "F64", "MXFP4", "NVFP4", "Q8_1"}


def build(llama, work):
    src, inc = os.path.join(llama, "ggml/src"), os.path.join(llama, "ggml/include")
    os.makedirs(os.path.join(work, "inc"), exist_ok=True)
    open(os.path.join(work, "inc/ggml-version.h"), "w").write('#define GGML_VERSION "0"\n#define GGML_COMMIT "pinned"\n')
    I = ["-I" + src, "-I" + inc, "-I" + os.path.join(work, "inc")]
    objs = []
    for f, cc in (("ggml.c", ["gcc", "-std=gnu11", "-D_GNU_SOURCE"]), ("ggml-quants.c", ["gcc", "-std=gnu11"]), ("ggml-threading.cpp", ["g++", "-std=gnu++17"])):
        o = os.path.join(work, f + ".o")
        subprocess.run(cc + ["-O2"] + I + ["-c", os.path.join(src, f), "-o", o], check=True)
        objs.append(o)
    stub = os.path.join(work, "stubs.c")
    open(stub, "w").write("void ggml_backend_tensor_memset(){} void ggml_backend_tensor_set(){} int ggml_backend_buffer_get_usage(){return 0;}\n")
    for t in ("dump", "quant"):
        subprocess.run(["gcc", "-O2"] + I + [os.path.join(HERE, "ggml_ref", t + ".c"), stub] + objs + ["-o", os.path.join(work, t), "-lm", "-lpthread", "-lstdc++"], check=True)


def fix_scales(nm, raw):
    """random bytes with a finite fp16 scale (clear the exponent MSB of the scale's high byte)"""
    off = {"Q2_K": [80, 82], "Q3_K": [108], "TQ1_0": [52], "TQ2_0": [64], "Q6_K": [208], "IQ1_M": [55], "Q8_K": [3]}.get(nm, [0, 2] if nm in ("Q4_1", "Q5_1", "Q4_K", "Q5_K") else [0])
    for o in off:
        raw[:, o + (0 if nm in ("Q8_K", "IQ1_M") else 1)] &= 0x3F
    return raw


def main(llama, work):
    os.makedirs(work, exist_ok=True)
    build(llama, work)
    rng = np.random.default_rng(20260101)
    out = {}
    dump = lambda tid, raw, n: np.frombuffer(subprocess.run([os.path.join(work, "dump"), str(tid), str(n)], input=raw.tobytes(), capture_output=True, check=True).stdout, "<f4")
    for tid, (nm, be, bb) in sorted(GF.TYPES.items()):
        if nm in SKIP:
            continue
        r = subprocess.run([os.path.join(work, "quant"), str(tid), "2", "512", "7", os.path.join(work, nm)], capture_output=True)
        parts, refs = [], []
        if r.returncode == 0:                                        # blocks written by ggml's own quantizer (Gaussian weights, random importance matrix)
            parts.append(np.fromfile(os.path.join(work, nm + ".raw"), np.uint8).reshape(-1, bb))
            refs.append(np.fromfile(os.path.join(work, nm + ".f32"), "<f4"))
        rb = fix_scales(nm, rng.integers(0, 256, (4, bb), dtype=np.uint8))       # random payload: every index / sign / scale pattern
        parts.append(rb)
        refs.append(dump(tid, rb, 4 * be))
        out[nm + "_raw"], out[nm + "_ref"] = np.concatenate(parts), np.concatenate(refs)
    # Q8_1 has no to_float in ggml: its payload is d, s, 32 x int8, decoded as d * q = ggml's Q8_0 function on (d, q)
    rb = rng.integers(0, 256, (4, 36), dtype=np.uint8)
    rb[:, 1] &= 0x3F
    out["Q8_1_raw"] = rb
    out["Q8_1_ref"] = dump(GF.NAME_TO_ID["Q8_0"], np.concatenate([rb[:, 0:2], rb[:, 4:36]], 1), 128)
    out["pin"] = np.frombuffer(PIN.encode(), np.uint8)
    np.savez_compressed(os.path.join(HERE, "..", "tests", "data", "gguf_ggml_ref.npz"), **out)


main(sys.argv[1], sys.argv[2])
