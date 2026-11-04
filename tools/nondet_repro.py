# SPDX-License-Identifier: Apache-2.0
"""Nondeterminism reproducer: the same forward pass of the synthetic DS4.1 fixture (tests/decstage_fixture.py), once per FRESH process.

    python tools/nondet_repro.py --n 4 [--rows 1] [--depth 8192] [--steps 64] [--dump DIR] [--flags K=V,...]
    python tools/nondet_repro.py --child ...            # one run, the form the parent spawns

Parent: spawns `--n` child interpreters of this same file, each prints a digest per tapped array (`logits`, the taps of the last verify pass,
one line per decode step), then prints every key whose digest is not identical over all processes, the first differing decode step and the
first differing op. A key that repeats on all processes is bit-identical; `DIFFER: none` is the clean answer.

Child: builds the fixture with the served attention shape (FX.REAL), prefills `--depth` tokens, runs `RW.verify` over `--rows` rows, then
`--steps` plain decode steps. Each child stays under 2 GiB and a few seconds (2 layers, vocab 128, hidden 512).

Token ids stay INSIDE the fixture vocabulary (vocab_size = 128): an id >= 128 makes mx `take` read rows past the embedding table, i.e. whatever
memory sits there, which is the host process's buffer and not this engine's arithmetic (that alone makes the digest of every array downstream
move between processes, and it is where the "logits move between runs" of tests/test_dec_attn.py::test_core_runs_with_iscore_mma comes from: it
prefills list(range(5, 305)) against a 128-row table).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
VLM = os.environ.get("VLM_PKG", os.path.expanduser("~/kyojin-mlx/vlm_pkg"))


def TOKS(n: int, start: int = 0, oov: bool = False) -> list:
    """`n` token ids of the fixture vocabulary (128). oov = the old test inputs, ids past the table (`take` then reads other buffers)."""
    return [5 + start + i for i in range(n)] if oov else [5 + ((start + i) % 120) for i in range(n)]


POISON_SIZES = (1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22)


def poison(kind: str) -> None:
    """Fill and free a pool of buffers of the sizes this engine allocates: an uninitialised read of a kernel output then sees the pattern instead of
    whatever the process happened to leave there. 'a' = 0xAA bytes, 'n' = fp32 NaN."""
    import mlx.core as mx
    a = None
    for sz in POISON_SIZES:
        if kind == "n":
            a = mx.full((sz,), float("nan"), mx.float32)
        else:
            a = mx.full((sz,), 0xAA, mx.uint8)
        mx.eval(a)
    del a


def digest(a) -> str:
    import numpy as np
    import mlx.core as mx
    mx.eval(a)
    return hashlib.sha1(np.array(a).tobytes()).hexdigest()[:16]


def child(args) -> None:
    import numpy as np
    import mlx.core as mx

    import decstage_fixture as FX
    from kyojin_mlx import archs
    from kyojin_mlx import dsv4_core as C
    from kyojin_mlx import idxfuse as IXF
    from kyojin_mlx import rows as RW

    taps = {}

    def tap(name, fn, first_only=False):
        def wrap(*a, **k):
            r = fn(*a, **k)
            outs = r if isinstance(r, (tuple, list)) else (r,)
            for i, o in enumerate(outs[:1] if first_only else outs):
                if isinstance(o, mx.array):
                    taps[f"{name}{i}"] = o
            return r
        return wrap

    for nm in ("index_scores", "index_topk_rt", "candidate_mask_rt", "attend"):
        setattr(C, nm, tap(nm, getattr(C, nm)))
    IXF.fused = tap("fused", IXF.fused, first_only=True)     # the mask / scratch of the fused kernel are defined on t < lens only
    IXF.scores_fast = tap("scores_fast", IXF.scores_fast)

    over = {}
    if args.consumer:                                # a second index layer that READS the source layer's block mask
        over = dict(num_hidden_layers=3, compress_ratios=[0, 1, 1], kv_source_layer_ids=[1, 2], index_source_layer_ids=[1, 2],
                    candidate_source_layer_id=1)
    m, cfg = FX.build(**{**FX.REAL, **over})
    RW.enable(m, archs.arch_of({"model_type": "deepseek_v41"}))
    pc = m.make_cache()
    if args.poison:
        poison(args.poison)
    mx.eval(m(mx.array([TOKS(args.depth, oov=args.oov)]), cache=pc).logits)
    cache = RW.batch([pc])
    if args.poison:
        poison(args.poison)
    p = RW.verify(m, cache, [TOKS(args.rows, 400, oov=args.oov)])
    lg = p.logits
    names = list(taps)
    print("logits", digest(lg), len(names), flush=True)
    for n in names:
        print("tap", n, digest(taps[n]), flush=True)
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        np.save(os.path.join(args.dump, f"logits_{os.getpid()}.npy"), np.array(lg))
        for n in names:
            np.save(os.path.join(args.dump, f"tap_{n}_{os.getpid()}.npy"), np.array(taps[n]))
    if args.steps:                                   # the served shape: a decode loop after the prefill
        ids = mx.argmax(p.logits, axis=-1).astype(mx.int32)[-1:]     # one row per session after the commit
        p.commit([args.rows - 1])
        out = []
        for i in range(args.steps):
            taps.clear()
            if args.poison:
                poison(args.poison)
            lg = RW.step(m, cache, ids)
            mx.eval(lg)
            ids = mx.argmax(lg, -1).astype(mx.int32)
            out.append(int(ids[0]))
            print("step", i, "logits", digest(lg), out[-1], flush=True)
            for t in taps:
                print("step", i, t, digest(taps[t]), flush=True)
        print("tokens", "".join(str(t) for t in out), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--depth", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--flags", default="")
    ap.add_argument("--consumer", action="store_true", help="3 layers: a consumer index layer reads the source block mask")
    ap.add_argument("--poison", default="", help="'a' or 'n': fill freed buffers with a pattern before every pass")
    ap.add_argument("--oov", action="store_true", help="token ids past the fixture vocabulary (the old tests' inputs: an out-of-range read, not the engine)")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(args)
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([HERE, os.path.join(HERE, "tests"), VLM])
    for kv in filter(None, args.flags.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--rows", str(args.rows), "--depth", str(args.depth),
           "--steps", str(args.steps)] + (["--dump", args.dump] if args.dump else []) + \
          (["--consumer"] if args.consumer else []) + (["--poison", args.poison] if args.poison else []) + \
          (["--oov"] if args.oov else [])
    per = []
    for i in range(args.n):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=240)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-4000:], sep="\n")
            return 1
        d = {"steps": [], "otaps": {}}
        for line in r.stdout.splitlines():
            f = line.split()
            if not f:
                continue
            if f[0] == "logits":
                d["logits"] = f[1]
            elif f[0] == "tap":
                d[f[1]] = f[2]
            elif f[0] == "step" and f[2] == "logits":
                d["steps"].append(f[3])
            elif f[0] == "step":
                d["otaps"].setdefault((int(f[1]), f[2]), f[3])
            elif f[0] == "tokens":
                d["tokens"] = f[1]
        per.append(d)
        print(f"proc{i} " + " ".join(f"{k}={v}" for k, v in d.items() if k not in ("steps", "otaps")), flush=True)
    bad = [k for k in sorted(per[0]) if k not in ("steps", "otaps") and len({p.get(k) for p in per}) > 1]
    if any(len({p["steps"][i] for p in per}) > 1 for i in range(min(len(p["steps"]) for p in per))):
        i = next(i for i in range(min(len(p["steps"]) for p in per)) if len({p["steps"][i] for p in per}) > 1)
        print(f"first differing decode step: {i}")
        bad.append(f"step{i}")
    ot = sorted({k for p in per for k in p["otaps"]})
    odiff = [k for k in ot if len({p["otaps"].get(k) for p in per}) > 1]
    if odiff:
        print("first differing decode op:", odiff[0])
        for k in odiff[:8]:
            print(f"  {k}: " + " ".join(str(p["otaps"].get(k)) for p in per))
    print("DIFFER:", ",".join(bad) if bad else "none")
    for k in bad:
        if k in per[0]:
            print(f"  {k}: " + " ".join(str(p.get(k, "?")) for p in per))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
