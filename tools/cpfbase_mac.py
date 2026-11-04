"""cpfbase Mac check: pf_win_slice and pf_pool_lazy, flag 0 against 1, bounded under 3.5 GiB (no pack, no served model).

Part 1, logits: the tiny random DS4.1 (tests/mini_ds41.py, fp16 and fp32) on the default device (Metal on the Mac), a packed-pool cache, unequal prefill chunks over several
window wraps. Logits of the flag-0 run and the flag-1 run are compared BYTE for byte, plus the dense cache. Any difference: exit 1.
Part 2, op time and retained bytes, synthetic tensors at the real per-layer dims (window 128 x 512, pool rows x 512, chunk 8192 rows):
  ring   : 40 layers of (history read + ring write) per chunk, stock Gather / Scatter against slices. ms per chunk, peak bytes.
  pool   : L pool layers per chunk, each: a matmul that produces 8192 rows, fp4 grid rounding, append_rows, then a consumer reads the dense pool. Stock = checked append (host
           read per layer); lazy = verdict evaluated with the state at the chunk end. ms per chunk, active-memory growth, a last arm that never evaluates the state
           (what a pending verdict would keep alive if the engine did not: it does, serve/backends.py prefill_chunk).
These are OP times in isolation. They are NOT a served gain (pf_moe_rowmap and epi won their microbenches and lost in situ); the served A/B stays with the lane (pf_all).
Hold the lane lock when running on the Ultra GPU. usage: PYTHONPATH=. python tools/cpfbase_mac.py [--depth 131072] [--layers 4] [--chunks 6] [--cpu]"""
import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
LIMIT = int(3.5 * 2 ** 30)

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=131072, help="pool rows already in each pool layer")
ap.add_argument("--layers", type=int, default=4, help="pool (KV-source) layers in the pool bench")
ap.add_argument("--chunks", type=int, default=6)
ap.add_argument("--rows", type=int, default=8192)
ap.add_argument("--cpu", action="store_true", help="run on the CPU (smoke test)")
a = ap.parse_args()
if a.cpu:
    mx.set_default_device(mx.cpu)
mx.set_memory_limit(LIMIT)

from kyojin_mlx import depthflat as DF, fp4pack, fp4pool, pf_ring as R  # noqa: E402

DF.ENABLED = True
D_POOL, WIN, D_WIN = 512, 128, 512
fail = []


PEAK = [0.0]


def peak_gib():
    """Running maximum over the whole run (never reset: pool construction counts)."""
    PEAK[0] = max(PEAK[0], mx.get_peak_memory() / 2 ** 30)
    return PEAK[0]


def bits(x):
    return x.view({2: mx.uint16, 4: mx.uint32}[x.dtype.size]) if x.dtype.size in (2, 4) else x


def same(x, y):
    """Bit for bit, on the device (no host copy of a pool)."""
    return x.shape == y.shape and x.dtype == y.dtype and bool(mx.array_equal(bits(x), bits(y)).item())


def logits_part():
    import mini_ds41 as M
    from kyojin_mlx import attn_prefill as AP, fused_group as FG
    import mlx_vlm.models.deepseek_v41.language as L
    for dt in (mx.float16, mx.float32):
        lm, cfg = M.make(dtype=dt)
        cls = type(lm.layers[2].attn)
        orig = cls._window_part
        patched = FG.share_window_math()                            # False when this mlx-vlm is not the audited source: then the vendor window runs and nothing is tested
        ids = M.tokens(300, seed=5)
        res = {}
        try:
            for on in (False, True):
                R.ENABLED, fp4pool.LAZY_POOL = on, on
                for d in (R.HITS, fp4pool.HITS):
                    for k in d:
                        d[k] = 0
                cache = fp4pool.wrap_cache(lm.make_cache())
                outs, pos = [], 0
                for T in (70, 1, 33, 64, 5, 90, 37):
                    outs.append(lm(mx.array([ids[pos:pos + T]]), cache=cache).logits)
                    pos += T
                mx.eval(outs)
                wins = [mx.array(w) for w in cache[0].window if w is not None]
                comp = [mx.array(x) for x in cache[0].compress if x is not None]
                # the served window path, called directly: attn_prefill.window_part
                att, c2, pos, parts = lm.layers[2].attn, L.DeepseekV41Cache(8), 0, []
                for T in (40, 1, 17, 16, 15, 3, 70):
                    x = mx.random.normal((1, T, att.wq_a.weight.shape[-1]), key=mx.random.key(pos)).astype(dt)
                    parts.append(AP.window_part(att, x, pos, c2, need_mask=True)[0])
                    pos += T
                parts.append(c2.window[att.layer_idx])
                mx.eval(parts)
                res[on] = (outs, wins, comp, parts, dict(R.HITS), dict(fp4pool.HITS))
        finally:
            cls._window_part = orig
            R.ENABLED = fp4pool.LAZY_POOL = False
        o0, w0, c0, p0 = res[False][:4]
        o1, w1, c1, p1, rh, ph = res[True]
        ok_l = len(o0) == len(o1) and all(same(a, b) for a, b in zip(o0, o1))
        ok_w = len(w0) == len(w1) and all(same(a, b) for a, b in zip(w0, w1)) and all(same(a, b) for a, b in zip(p0, p1))
        ok_c = len(c0) == len(c1) and all(same(a, b) for a, b in zip(c0, c1))
        engaged = patched and rh["read"] > 0 and rh["write"] > 0 and ph["lazy"] > 0
        print(f"logits {dt}: chunks={len(o0)} logits_equal={ok_l} window_equal={ok_w} pool_equal={ok_c}  window patched={patched} ring hits={rh} pool hits={ph}")
        if not engaged:
            print("NOT ENGAGED: a flag-1 arm that does not take the new paths proves nothing")
            fail.append(f"not engaged {dt}")
        if not (ok_l and ok_w and ok_c):
            fail.append(f"logits/cache {dt}")


def ring_part():
    rng = np.random.default_rng(0)
    T, nl = a.rows, 40
    bufs = [mx.array(rng.standard_normal((1, WIN, D_WIN)).astype(np.float16)) for _ in range(nl)]
    kv = mx.array(rng.standard_normal((1, T, D_WIN)).astype(np.float16))
    mx.eval(bufs, kv)
    start = 5 * T + 37

    def chunk(on):
        outs = []
        for b in bufs:
            base, end = start - WIN + 1, start + T
            if on:
                h = R.read(b, 1, base, start, WIN)
                nb = R.write(b, 1, kv[:, -WIN:], end, WIN)
            else:
                h = mx.take(b[:1], mx.arange(base, start) % WIN, axis=1)
                nb = mx.array(b)
                nb[:1, mx.arange(end - WIN, end) % WIN] = kv[:, -WIN:]
            outs += [mx.concatenate([h, kv], axis=1), nb]
        return outs

    a0, a1 = chunk(False), chunk(True)
    mx.eval(a0, a1)
    eqb = all(bool(mx.array_equal(x, y).item()) for x, y in zip(a0, a1))
    print(f"ring: 40 layers x (read+write), outputs equal={eqb}")
    if not eqb:
        fail.append("ring op output")
    for on in (False, True, False, True):
        mx.reset_peak_memory()
        ts = []
        for _ in range(a.chunks):
            t0 = time.perf_counter()
            mx.eval(chunk(on))
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"ring  pf_win_slice={int(on)}: ms/chunk median {np.median(ts):.2f} min {min(ts):.2f}  peak {peak_gib():.2f} GiB")


def pool_part():
    rng = np.random.default_rng(1)
    d = a.depth
    est = a.layers * d * D_POOL * 2 * 1.25 / 2 ** 30                 # mirrors, per arm; codes and the reference pools come on top (see below)
    if 3 * est > 2.0:
        sys.exit(f"pool bench would hold ~{3 * est:.2f} GiB (mirrors of the arm under test, one reference copy, codes): lower --depth or --layers")
    wproj = mx.array(rng.standard_normal((D_POOL, D_POOL)).astype(np.float16) * 0.05)
    xin = mx.array(rng.standard_normal((a.rows, D_POOL)).astype(np.float16))
    mx.eval(wproj, xin)

    def grid_rows(n, seed):
        x = mx.array(np.random.default_rng(seed).standard_normal((1, n, D_POOL)).astype(np.float32)).astype(mx.float16)
        c, s = fp4pack.pack(x, 16, True)
        return fp4pack.unpack(c, s, 16, True, mx.float16)

    def build(lazy):
        fp4pool.LAZY_POOL = lazy
        slots = fp4pool.PoolSlots([None] * a.layers)
        for l in range(a.layers):
            for a0 in range(0, d, 32768):
                slots.append_rows(l, grid_rows(min(32768, d - a0), 100 * l + a0 // 32768))
            mx.eval(slots.raw(l).codes, slots[l])                    # builds the mirror
        return slots

    def chunk(slots, state_eval):
        h = xin
        reads = []
        for l in range(a.layers):
            h = (h @ wproj) * 0.5 + xin                           # work the chunk does before this layer's append
            c, s = fp4pack.pack(h[None], 16, True)
            rows = fp4pack.unpack(c, s, 16, True, mx.float16)    # the vendor fake_quant_fp4: on the grid by construction
            slots.append_rows(l, rows)
            reads.append(slots[l][:, -64:].sum())                 # a consumer of the dense pool
        mx.eval(reads)
        if state_eval:                                            # backends.prefill_chunk: mx.eval(lg, *self.state(cache)), arrays_of -> raw_arrays (carries the verdicts)
            mx.eval(*slots.raw_arrays())

    ref = None
    # arms: (lazy, state_eval). The last arm never evaluates the state: it shows what a pending verdict keeps alive if the engine did not (it does, backends.py prefill_chunk).
    for on, state_eval in ((False, True), (True, True), (False, True), (True, True), (True, False)):
        mx.clear_cache()
        slots = build(on)
        for k in fp4pool.HITS:
            fp4pool.HITS[k] = 0
        base_mem = mx.get_active_memory()
        ts = []
        for k in range(a.chunks):
            t0 = time.perf_counter()
            chunk(slots, state_eval)
            ts.append((time.perf_counter() - t0) * 1e3)
        grown = (mx.get_active_memory() - base_mem) / 2 ** 20
        peak_gib()
        cur = [mx.array(slots[l]) for l in range(a.layers)]           # on device
        mx.eval(cur)
        if ref is None:
            ref = cur
            eqs = True
        else:
            eqs = all(same(x, y) for x, y in zip(ref, cur))
            if not eqs:
                fail.append("pool dense cache differs")
        print(f"pool  pf_pool_lazy={int(on)} state_eval={int(state_eval)}: depth {d} x {a.layers} layers, ms/chunk median {np.median(ts):.2f} min {min(ts):.2f}  "
              f"active memory +{grown:.1f} MiB over {a.chunks} chunks  lazy appends {fp4pool.HITS['lazy']} settles {fp4pool.HITS['settle']} dense {fp4pool.HITS['settle_dense']}  dense cache equal={eqs}")
        del slots, cur
    fp4pool.LAZY_POOL = False


print(f"device {mx.default_device()}  limit {LIMIT / 2 ** 30:.1f} GiB")
logits_part()
ring_part()
pool_part()
print(f"peak overall (running max, never reset) {peak_gib():.2f} GiB (limit 3.5; the MLX allocator limit above also stops the run)")
if peak_gib() > 3.5:
    fail.append("memory bound")
print("RESULT", "FAIL: " + ", ".join(fail) if fail else "logits equal, dense caches equal (op times above are isolated ops, not a served gain)")
sys.exit(1 if fail else 0)
