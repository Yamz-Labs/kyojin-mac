"""tg128.py PACK ENGINE(kyojin|mlxlm) [USERS=1]: greedy decode of 128 tokens after a 512-token prompt (pp not timed), pipelined (next step queued before the sync). Prints tok/s (aggregate over users)."""
import os, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in ("..", ".")]
import mlx.core as mx
PACK, ENG = sys.argv[1], sys.argv[2]
U = int(sys.argv[3]) if len(sys.argv) > 3 else 1
N = 128
ids = [(i * 7919) % 20000 + 1000 for i in range(512)]
if ENG == "kyojin":
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(PACK, mtp=False, log=lambda *a: None)
    base = be.make_cache(); lg = be.prefill_chunk(base, ids)
    best = 0
    for rep in range(3):
        bc = be.batch([be.fork(base) for _ in range(U)]) if U > 1 else be.fork(base)
        first = mx.argmax(lg[0, -1] if lg.ndim == 3 else lg.reshape(-1, lg.shape[-1])[-1])
        tok = mx.broadcast_to(first, (U,)).astype(mx.int32)
        stepf = (lambda t: be.step(bc, t)) if U > 1 else (lambda t: be.step(bc, t))
        def one(t):
            l = stepf(t)
            return mx.argmax(l.reshape(U, -1), axis=-1).astype(mx.int32)
        nxt = one(tok); mx.async_eval(nxt)
        t0 = time.perf_counter()
        for i in range(N - 1):
            cur = nxt; nxt = one(cur); mx.async_eval(nxt)
            mx.eval(cur)
        mx.eval(nxt)
        dt = time.perf_counter() - t0
        best = max(best, (N - 1) * U / dt)
    print(f"{ENG} users {U} tg128 {best:.2f} tok/s (aggregate)")
else:
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    model, tok = load(PACK)
    best = 0
    for rep in range(3):
        last = None
        for r in stream_generate(model, tok, prompt=ids, max_tokens=N):
            last = r
        best = max(best, last.generation_tps)
    print(f"{ENG} users 1 tg128 {best:.2f} tok/s")
