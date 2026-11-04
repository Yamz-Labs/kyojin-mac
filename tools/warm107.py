import os, sys, time
sys.path.insert(0, os.path.expanduser("~/kyojin-mlx"))
import mlx.core as mx
from kyojin_mlx import tq
from kyojin_mlx.sdpa_dec import sdpa_decode
from kyojin_mlx import sdpa_dec as SD
Hq, Hk, D = 16, 4, 256
mk = lambda n: mx.random.normal((1, Hk, n, D)).astype(mx.float16)
t00 = time.time(); slow = 0
for L in (1500, 3000, 5000, 9000, 17000, 33000, 66000, 131000):
    c = tq.TQKVCache(6, 6, 4, 256); c.update_and_fetch(mk(L), mk(L)); mx.eval(c.state)
    k, v = mk(L), mk(L); mx.eval(k, v)
    for T in (1, 2, 3, 4, 5):
        q = mx.random.normal((1, Hq, T, D)).astype(mx.float16)
        for f in (lambda: c.attend(q, 0.0625), lambda: sdpa_decode(q, k, v, 0.0625)):
            t0 = time.perf_counter()
            try:
                mx.eval(f())
            except Exception as e:
                print("WARN", L, T, repr(e)[:120]); continue
            dt = time.perf_counter() - t0; slow += dt > 0.05
    del c, k, v; mx.clear_cache()
print("WARM done s %.0f compiles>50ms %d" % (time.time() - t00, slow), flush=True)
