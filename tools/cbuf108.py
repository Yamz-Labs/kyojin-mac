#!/usr/bin/env python3
"""One command buffer per decode step vs the MLX default buffer split (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER), whole decode steps, warm.
The env is read at MLX init, so every variant runs in its own worker process; workers alternate (A B A B ...) inside this one job.
A worker times --steps consecutive greedy decode steps (step -> eval -> next token), ms per step min / median over --reps chains.
Packs: a DS4.1 slice (R.step of the serving path, rows.enable + spec.install) or a Qwen3.5 EXL3 pack (qspec.plain_generate, slope of n and 4n tokens).
usage: cbuf108.py PACK [--steps 16] [--reps 5] [--rounds 2] [--len 135] [--variants default,one]"""
import argparse, hashlib, json, os, subprocess, sys, time

VARIANTS = {"default": {}, "one": {"MLX_MAX_OPS_PER_BUFFER": "100000", "MLX_MAX_MB_PER_BUFFER": "1000000"},
            "mb4k": {"MLX_MAX_MB_PER_BUFFER": "4096"}, "idx0": {"KYOJIN_DEC_IDX_FUSE": "0"}, **{f"ops{n}": {"MLX_MAX_OPS_PER_BUFFER": str(n)} for n in (5, 10, 20, 40, 80, 200)}}


def worker(pack, steps, reps, plen):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..")); sys.path.insert(0, here); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg"))
    import numpy as np
    import mlx.core as mx
    rng = np.random.default_rng(1)
    toks = []
    if json.load(open(os.path.join(pack, "config.json"))).get("model_type") == "qwen3_5":
        from kyojin_mlx import exl3_linear as EL, qspec as QS
        model, _ = EL.load_exl3(pack); QS.uninstall_capture()
        prompt = rng.integers(16, 4000, plen).tolist()
        def gen(n):
            t0 = time.perf_counter(); QS.plain_generate(model, prompt, n); return time.perf_counter() - t0
        def chain():                                  # slope over n and 4n tokens: the prompt prefill cancels out
            return (gen(4 * steps) - gen(steps)) / (3 * steps) * 1e3
    else:
        from kyojin_mlx import patch as P, rows as R, spec as S
        model, spec, cfg = P.build_model(pack)
        P.patch_model(model, pack, spec, cfg); R.enable(model, spec); S.install(model, spec)
        tc = cfg["text_config"]
        cache0, lg, _, _ = S.prefill(model, rng.integers(16, tc["vocab_size"] - 16, plen).tolist())
        first = mx.array([int(mx.argmax(lg[0]).item())])
        def chain():
            c, tok = R.fork(cache0), first
            t0 = time.perf_counter()
            for _ in range(steps):
                lg = R.step(model, c, tok); tok = mx.argmax(lg, -1).reshape(1).astype(mx.int32); mx.eval(tok, lg); toks.append(hashlib.sha1(bytes(memoryview(lg))).hexdigest())
            return (time.perf_counter() - t0) / steps * 1e3
    chain()
    ts = [chain() for _ in range(reps)]
    print(json.dumps({"t": ts, "tok": toks[-steps:]}), flush=True)


def main():
    if "--worker" in sys.argv:
        _, _, pack, steps, reps, plen = sys.argv
        worker(pack, int(steps), int(reps), int(plen)); return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack"); ap.add_argument("--steps", type=int, default=16); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--len", type=int, default=135); ap.add_argument("--variants", default="default,one")
    a = ap.parse_args()
    vs = a.variants.split(",")
    T, K = {v: [] for v in vs}, {}
    for _ in range(a.rounds):
        for v in vs:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", a.pack, str(a.steps), str(a.reps), str(a.len)],
                               env={**os.environ, **VARIANTS[v]}, capture_output=True, text=True)
            if r.returncode:
                print(f"{v}: worker rc {r.returncode}\n{r.stderr[-1500:]}", flush=True); sys.exit(2)
            o = json.loads(r.stdout.strip().splitlines()[-1])
            T[v] += o["t"]; K.setdefault(v, o["tok"])
    import statistics
    b0 = min(T[vs[0]])
    print(f"{os.path.basename(a.pack.rstrip('/'))} decode ms/step: " + ", ".join(
        f"{v} {min(T[v]):.3f} (med {statistics.median(T[v]):.3f}, {(min(T[v]) / b0 - 1) * 100:+.1f}%)" for v in vs) + (f"; logits bitwise equal to {vs[0]}: " + ", ".join(f"{v} {K[v] == K[vs[0]]}" for v in vs[1:]) if K.get(vs[0]) else ""), flush=True)


if __name__ == "__main__":
    main()
