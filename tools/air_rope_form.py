# SPDX-License-Identifier: Apache-2.0
"""Which fma form does the SHIPPED mx.fast.rope use? Read it out of mlx.metallib (the AIR of the `rope*` kernels), no GPU needed.

The fused DS4.1 kernels (kvchain, ropefuse) compute the rope pair under `fp contract(off)`; MLX's own kernels are compiled with clang's default
contraction, so its AIR carries `llvm.fmuladd` where the fused kernels have a separate multiply and add. This prints, for every `rope*` kernel of the
library, the dataflow of each fmuladd with the roles xa (first element of the pair), xb (second), cos, sin, and the form it amounts to:

  form 1  r1 = fma(xa, cos, -(sin * xb))   r2 = fma(xa, sin, cos * xb)      (== KYOJIN_ROPE_FMA=1, `x1 * c - x2 * s` with the first product fused)
  form 2  r1 = fma(-xb, sin, xa * cos)     r2 = fma(xb, cos, xa * sin)      (== KYOJIN_ROPE_FMA=2)

  python tools/air_rope_form.py [--metallib PATH] [--llvm-dis llvm-dis] [--match rope] [--show]

PATH defaults to mlx/lib/mlx.metallib or mlx.metallib inside the installed mlx package. The AIR is LLVM bitcode with a trailing symbol table that newer
llvm-dis refuses, so each module is cut to its exact extent (block walk) first. Needs an llvm-dis (brew install llvm; apt install llvm); memory < 600 MB
(the library is ~150 MB and is read once).
The result says what the compiler REQUESTED (fmuladd = "may fuse"); that the GPU backend fuses it is shown by tools/kvchain_bits.py / rope_contract_probe.py on a Mac.
"""
from __future__ import annotations

import argparse
import bisect
import os
import re
import shutil
import subprocess
import sys

MAGIC = b"BC\xc0\xde"


class _Bits:
    def __init__(self, b: bytes):
        self.b, self.p = b, 0

    def read(self, n: int) -> int:
        v = 0
        for i in range(n):
            byte, bit = divmod(self.p + i, 8)
            v |= ((self.b[byte] >> bit) & 1) << i
        self.p += n
        return v

    def vbr(self, n: int) -> int:
        v, shift, hi = 0, 0, 1 << (n - 1)
        while True:
            c = self.read(n)
            v |= (c & (hi - 1)) << shift
            if not c & hi:
                return v
            shift += n - 1

    def align(self):
        self.p = (self.p + 31) // 32 * 32


def bitcode_extent(raw: bytes) -> int:
    """Byte length of the bitcode stream at raw[0:]: magic + top-level blocks up to and including the STRTAB block (id 23)."""
    r = _Bits(raw)
    r.p = 32
    while r.p // 8 + 8 <= len(raw):
        if r.read(2) != 1:                                  # top level holds only ENTER_SUBBLOCK
            break
        bid = r.vbr(8)
        r.vbr(4)
        r.align()
        words = r.read(32)
        r.p += words * 32
        if bid == 23:
            return r.p // 8
    return r.p // 8


def modules(data: bytes, match: bytes):
    """(offset, length) of every bitcode module of the library whose symbol text names a `match` source (e.g. `_GLOBAL__sub_I_rope.metal`)."""
    pos = [m.start() for m in re.finditer(re.escape(MAGIC), data)]
    seen = {}
    for m in re.finditer(rb"_GLOBAL__sub_I_" + re.escape(match) + rb"\.metal", data):
        k = bisect.bisect_right(pos, m.start()) - 1
        seen[pos[k]] = None
    for off in sorted(seen):
        yield off, bitcode_extent(data[off:off + 1 << 20])


def disassemble(raw: bytes, llvm_dis: str) -> str:
    r = subprocess.run([llvm_dis, "-", "-o", "-"], input=raw, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace")[:300])
    return r.stdout.decode()


_DEF = re.compile(r"^\s*(%[\w.]+) = (?:tail )?(\w+) ?(?:float|half|i\d+|<[^>]*>|ptr)? ?(.*)$")


def roles(ll: str) -> list[str]:
    """Role-string of every llvm.fmuladd.f32 of the function bodies in `ll`: cos, sin, and the two elements of the pair as xa (defined first = x1, the
    element read at the first index) and xb, whatever the load type (half, bfloat, float)."""
    out = []
    for body in re.split(r"^define ", ll, flags=re.M)[1:]:
        body = body.split("\n}\n", 1)[0]
        name = {}
        defline = {}

        def res(x):
            return name.get(x, "v" + x[1:])                  # an SSA value no rule named: an element of the pair (or an index)

        for ln in body.split("\n"):
            m = re.match(r"\s*(%[\w.]+) = (.*)", ln)
            if not m:
                continue
            v, rest = m.groups()
            defline.setdefault("v" + v[1:], len(defline))
            if "@air.fast_cos.f32" in rest:
                name[v] = "cos"
            elif "@air.fast_sin.f32" in rest:
                name[v] = "sin"
            elif rest.startswith("fneg float"):
                a = res(rest.split()[-1])
                name[v] = a[1:] if a.startswith("-") else "-" + a
            elif rest.startswith("fmul float"):
                a, b = (res(s.strip()) for s in rest[len("fmul float"):].split(","))
                neg = a.startswith("-") != b.startswith("-")
                a, b = sorted((a.lstrip("-"), b.lstrip("-")))
                name[v] = ("-" if neg else "") + f"({a}*{b})"
            elif "@llvm.fmuladd.f32" in rest:
                a, b, c = (res(x) for x in re.findall(r"float (%[\w.]+)", rest))
                name[v] = f"fma({a}, {b}, {c})"
                out.append(name[v])
        toks = sorted({t for f in out for t in re.findall(r"v[\w.]+", f)}, key=lambda t: defline.get(t, -1))
        for t, r in zip(toks[:2], ("xa", "xb")):                # the pair's elements, named in the order the kernel defines (loads) them: x1 then x2
            out = [re.sub(rf"\b{re.escape(t)}\b", r, f) for f in out]
    return out


def form_of(fmas: list[str]) -> str:
    """'form1' / 'form2' / 'other' for the forward rotation among the fmuladd role strings (the inverse-direction branch of a kernel is listed too)."""
    s = set(fmas)
    f1 = {"fma(xa, cos, -(sin*xb))", "fma(xa, sin, (cos*xb))"}
    f2 = {"fma(-xb, sin, (cos*xa))", "fma(xb, cos, (sin*xa))"}
    if f1 <= s:
        return "form1"
    if f2 <= s:
        return "form2"
    return "other"


def find_metallib() -> str:
    for mod in ("mlx", "mlx_metal"):
        try:
            m = __import__(mod)
            base = os.path.dirname(m.__file__) if getattr(m, "__file__", None) else list(m.__path__)[0]
        except Exception:
            continue
        for rel in ("lib/mlx.metallib", "mlx.metallib", "../mlx_metal/lib/mlx.metallib"):
            p = os.path.normpath(os.path.join(base, rel))
            if os.path.exists(p):
                return p
    raise SystemExit("mlx.metallib not found: pass --metallib")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metallib")
    ap.add_argument("--llvm-dis", default=None)
    ap.add_argument("--match", default="rope")
    ap.add_argument("--show", action="store_true", help="print every fmuladd role string")
    a = ap.parse_args()
    dis = a.llvm_dis or next((c for c in ("llvm-dis", "llvm-dis-18", "llvm-dis-17", "llvm-dis-16", "llvm-dis-15") if shutil.which(c)), None)
    if not dis:
        print("no llvm-dis on PATH (brew install llvm / apt install llvm)")
        return 2
    path = a.metallib or find_metallib()
    data = open(path, "rb").read()
    print(f"{path}: {len(data)} bytes")
    worst = 0
    for off, n in modules(data, a.match.encode()):
        ll = disassemble(data[off:off + n], dis)
        name = re.search(r"define weak_odr void @(\w+)", ll)
        fm = roles(ll)
        form = form_of(fm)
        print(f"{name.group(1) if name else off:34s} fmuladd={len(fm)}  {form}")
        if a.show:
            for f in fm:
                print("    ", f)
        worst += form != "form1"
    print("every rope kernel is form 1 (KYOJIN_ROPE_FMA=1)" if not worst else f"{worst} kernel(s) not form 1")
    return 0 if not worst else 1


if __name__ == "__main__":
    sys.exit(main())
