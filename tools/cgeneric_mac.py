#!/usr/bin/env python3
"""cgeneric on the Mac: what the CPU suite could not run (it needs Metal), bounded under 3.5 GiB, prints no timings.

  python tools/cgeneric_mac.py --pack ~/ds41/model          # stage config: header-only, never reads a weight
  python tools/cgeneric_mac.py --stage metal                # stage metal: tiny synthetic packs through the SERVED loader and backend
  python tools/cgeneric_mac.py --pack PACK --stage all      # both

stage config  (pack given)  reads config.json, the safetensors index (or headers) and the tokenizer JSON; runs `packcheck.check` with KYOJIN_DESC_STRICT=1
                            semantics (any finding = FAIL) and prints the resolved descriptor fields: row, geometry, which fused paths cover it, the CED
                            midpoint / tail the server would use, the stop strings, the draft-head keys. For DS4.1 expect: mid 20, tail 128 (see the
                            ledger: never lowered). The 172 GiB pack is never loaded.
stage metal                 writes the synthetic packs of kyojin_mlx/synth.py (a few MB), loads them through `serve.backends.load_vlm` (the VlmBackend of
                            the lane, rows + fused-path installs on Metal, hc_mult=2 so the DS4.1-only fused paths fall back), and compares with the slow
                            CPU reference (one forward over the sequence on the CPU stream):
                              prefill_chunk (chunked)  vs reference      batch/step decode vs reference      fork bitwise      KvDisk bitwise (the backend's
                              disk namespace)      CED (exact tail) vs exact prefill, the shape that supports CED
                            Tolerances are for Metal kernels vs CPU: --tol (default 5e-2 absolute on fp32 logits) and exact argmax at every step.
                            Fork and KvDisk must be bitwise. Qwen-style shapes need an EXL3 / mlx-quantized pack and are not run here.

Memory: mx.set_memory_limit(--cap-gib) plus a check of MLX peak and process RSS after every step; above the cap the script aborts with exit 3.
--cap-gib defaults to 3.0 and may not exceed 3.4 (the bound is 3.5 GiB)."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import resource
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("KYOJIN_WARM_LOAD", "0")

GIB = 2 ** 30
FAILS: list = []


TMP: list = []
atexit.register(lambda: [shutil.rmtree(p, ignore_errors=True) for p in TMP])          # the synthetic packs and checkpoints are scratch


def scratch(prefix: str) -> str:
    p = tempfile.mkdtemp(prefix=prefix)
    TMP.append(p)
    return p


def rss_gib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / GIB if sys.platform == "darwin" else r * 1024 / GIB          # macOS: bytes, Linux: KiB


class Cap:
    def __init__(self, gib: float):
        import mlx.core as mx
        self.mx, self.gib = mx, gib
        mx.set_memory_limit(int(gib * GIB))

    def check(self, where: str) -> None:
        peak = self.mx.get_peak_memory() / GIB
        rss = rss_gib()
        if peak > self.gib or rss > self.gib:
            print(f"ABORT {where}: MLX peak {peak:.2f} GiB / RSS {rss:.2f} GiB above the cap {self.gib} GiB", flush=True)
            sys.exit(3)


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}".rstrip(), flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------------------------ stage config
def stage_config(pack: str) -> None:
    from kyojin_mlx import archs, ced as C, modeldesc, packcheck
    cfg = json.load(open(os.path.join(pack, "config.json")))
    row, g = modeldesc.describe(cfg)
    print(modeldesc.report(cfg), flush=True)
    names = packcheck.tensor_names(pack)
    findings = packcheck.check(pack, cfg, mtp=any(n.startswith("mtp.") for n in names), ced=True, names=names)
    for f in findings:
        print(f"      finding {f}", flush=True)
    report("packcheck (strict: any finding fails)", not findings, f"{len(findings)} finding(s)")
    if row is None or row.arch is None:
        return
    spec = archs.ARCHS[row.arch]
    t = modeldesc.text_config(cfg)
    print(f"      stop_tokens {row.stop_tokens}; mtp_auto {row.mtp_auto}; chunk {row.chunk}; thinking_key {row.thinking_key}", flush=True)
    if spec.mtp:
        print(f"      draft head: prefix {spec.mtp.prefix!r} keep {spec.mtp.keep} keys " + ", ".join(f"{r}={k}:{t.get(k)}" for r, k in spec.mtp.keys.items()), flush=True)
    if spec.ced:
        import types
        c = types.SimpleNamespace(**{**t, "num_hidden_layers": g.layers})
        mid, tail = C.spec_defaults(spec, None, None)
        why = C.explain(c, mid)
        if why is None:
            lay = C.layout(c, mid)
            print(f"      CED resolved: midpoint layer {lay.mid} of {lay.n_layers}, tail {lay.window if tail is None else tail} rows per chunk "
                  f"(CedSpec.mid={spec.ced.mid}, tail={spec.ced.tail}; DS4.1 lane: 20 / 128)", flush=True)
        else:
            print(f"      CED not supported by this layout: {why}", flush=True)
    if spec.vendor:
        print(f"      vendor {spec.vendor}", flush=True)


# ------------------------------------------------------------------------------------------------------------------ stage metal
def stage_metal(cap: Cap, tol: float) -> None:
    import mlx.core as mx
    if not mx.metal.is_available():
        report("metal stage", False, "no Metal device: run this on the Mac")
        return
    from kyojin_mlx import ced as C, conformance as K, kvdisk, synth
    from kyojin_mlx.serve import backends as B
    for name in ("ds_moe_ced", "ds_moe_plain"):
        sh = synth.SHAPES[name]
        d = scratch(f"kyojin_mac_{name}_")
        vocab = int(sh.text["vocab_size"])
        mx.set_default_device(mx.cpu)                                          # the slow reference runs on the CPU stream
        ref_model, ref_lm, ref_mk = synth.write_pack(sh, d)
        ids = K.seq(37, vocab)
        ref_prefill = K.reference_last(ref_lm, ref_mk, ids)
        toks, refs = list(ids), []
        tok = int(mx.argmax(ref_prefill).item())
        for _ in range(5):
            toks.append(tok)
            r = K.reference_last(ref_lm, ref_mk, toks)
            refs.append((tok, r))
            tok = int(mx.argmax(r).item())
        exact_ced = K.reference_last(ref_lm, ref_mk, K.seq(100, vocab, 4)) if sh.ced else None
        mx.eval(ref_prefill, [r for _, r in refs])
        mx.set_default_device(mx.gpu)
        cap.check(f"{name} reference")

        be, _ = B.load_vlm(d, ced=sh.ced, log=lambda *a: None)
        cap.check(f"{name} load_vlm")
        report(f"{name}: load_vlm builds VlmBackend", be.kind == "vlm", f"ced_on={bool(C.state(be.model) and C.state(be.model).on)}")
        cache = be.make_cache()
        lg = None
        for s in range(0, len(ids), 8):
            lg = be.prefill_chunk(cache, ids[s:s + 8])
        d1 = K.maxdiff(lg.reshape(-1), ref_prefill)
        report(f"{name}: prefill_chunk vs CPU reference", d1 <= tol and int(mx.argmax(lg).item()) == int(mx.argmax(ref_prefill).item()), f"max abs diff {d1:.3e}")
        snap = be.fork(cache)
        bc = be.batch([cache])
        worst, argmax_ok, tok = 0.0, True, int(mx.argmax(lg).item())
        gpu_steps = []
        for i in range(5):
            out = be.step(bc, mx.array([tok], dtype=mx.int32)).reshape(-1)
            mx.eval(out)
            gpu_steps.append(out)
            worst = max(worst, K.maxdiff(out, refs[i][1]))
            argmax_ok &= int(mx.argmax(out).item()) == int(mx.argmax(refs[i][1]).item())
            tok = int(mx.argmax(out).item())
        report(f"{name}: batch/step decode vs CPU reference", worst <= tol and argmax_ok, f"max abs diff {worst:.3e}, argmax equal at every step: {argmax_ok}")
        bc2 = be.batch([snap])
        same = True
        for i in range(5):
            out = be.step(bc2, mx.array([refs[i][0]], dtype=mx.int32)).reshape(-1)
            mx.eval(out)
            same &= K.bitwise(out, gpu_steps[i])
        report(f"{name}: fork continues bitwise", same)
        ns = be.disk_namespace()
        if ns is None:
            report(f"{name}: KvDisk", False, "backend has no disk namespace")
        else:
            kv = kvdisk.KvDisk(scratch("kyojin_mac_kv_"), ns, be.make_cache, block=16)
            c2 = be.make_cache()
            be.prefill_chunk(c2, ids[:16])
            kv.save(ids, 16, c2)
            b, got, _ = kv.lookup(ids)
            lg_a = lg_b = None
            if got is not None:
                for s in range(b, len(ids), 8):
                    lg_b = be.prefill_chunk(got, ids[s:s + 8])
                for s in range(16, len(ids), 8):
                    lg_a = be.prefill_chunk(c2, ids[s:s + 8])
            report(f"{name}: KvDisk restore bitwise", got is not None and b == 16 and K.bitwise(lg_a, lg_b), f"hit at {b}")
        if sh.ced:
            st = C.state(be.model)
            long_ids = K.seq(100, vocab, 4)
            C.enable(be.model, be.spec, tail="exact")
            c3 = be.make_cache()
            cut = be.prefill_chunk(c3, long_ids).reshape(-1)
            with _NoCed(st):
                c4 = be.make_cache()
                ex = be.prefill_chunk(c4, long_ids).reshape(-1)
            d3 = K.maxdiff(cut, ex)
            report(f"{name}: CED (exact tail) vs exact prefill on Metal", d3 <= tol and st.stats.get("rows_cut", 0) > 0, f"max abs diff {d3:.3e}, rows cut {st.stats.get('rows_cut', 0)}")
            report(f"{name}: exact prefill vs CPU reference (same 100 tokens)", K.maxdiff(ex, exact_ced) <= tol, f"max abs diff {K.maxdiff(ex, exact_ced):.3e}")
        cap.check(f"{name} done")


class _NoCed:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        self.prev, self.st.on = self.st.on, False

    def __exit__(self, *e):
        self.st.on = self.prev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", help="pack directory: config + index/headers + tokenizer JSON are read, no weight")
    ap.add_argument("--stage", choices=("config", "metal", "all"), default=None)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--cap-gib", type=float, default=3.0)
    a = ap.parse_args()
    if a.cap_gib > 3.4:
        ap.error("--cap-gib must stay under 3.5 GiB (max 3.4)")
    stage = a.stage or ("config" if a.pack else "metal")
    cap = Cap(a.cap_gib)
    if stage in ("config", "all"):
        if not a.pack:
            ap.error("--pack is required for the config stage")
        stage_config(a.pack)
        cap.check("config stage")
    if stage in ("metal", "all"):
        stage_metal(cap, a.tol)
    print(f"RESULT {'FAIL' if FAILS else 'PASS'}  failures: {FAILS}  (peak MLX {cap.mx.get_peak_memory() / GIB:.2f} GiB, RSS {rss_gib():.2f} GiB, cap {a.cap_gib})", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
