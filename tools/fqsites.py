#!/usr/bin/env python3
"""Where the DS4.1 fp8 activation round trip (fakequant fp8_roundtrip kernel) still launches in one served prefill chunk: count, shape and caller per site.
usage: flock /tmp/ultra_heavy.lock python tools/fqsites.py [rows 8192]"""
import collections, glob, os, sys, time, traceback
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_vlm.models.deepseek_v41 import fakequant as FQ  # noqa: E402
from kyojin_mlx.serve import backends as B  # noqa: E402

PF = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
SITES, ON = collections.Counter(), [False]
k0 = FQ._fp8_roundtrip_kernel


def k(*a, **kw):
    if ON[0]:
        fr = [f for f in traceback.extract_stack()[:-1] if "fakequant" not in f.filename][-2:]
        site = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(fr))
        SITES[(site, tuple(kw["inputs"][0].shape), str(kw["inputs"][0].dtype))] += 1
    return k0(*a, **kw)


FQ._fp8_roundtrip_kernel = k
B.wire_memory()
be, cfg = B.load_vlm(os.path.expanduser("~/ds41/model"), mtp=True, ced=True)
from kyojin_mlx.serve.server import cache_limit_mb_for  # noqa: E402
mx.set_cache_limit(cache_limit_mb_for(mx.get_active_memory()) * 2 ** 20)
from tokenizers import Tokenizer  # noqa: E402
txt = "".join(open(f, errors="replace").read() for f in sorted(glob.glob(os.path.expanduser("~/kmlx-wave1/kyojin_mlx/*.py"))))
ids = Tokenizer.from_file(os.path.expanduser("~/ds41/model/tokenizer.json")).encode(txt).ids[:PF]
c, d = be.make_cache(), be.new_draft()
be.prefill_chunk(c, ids, draft=d); mx.synchronize()
ON[0] = True
c, d = be.make_cache(), be.new_draft()
t = time.perf_counter(); be.prefill_chunk(c, ids, draft=d); mx.synchronize()
print(f"FQSITES prefill {len(ids)} rows {time.perf_counter() - t:.3f} s, {sum(SITES.values())} fq calls (graph nodes; folded ones never launch)", flush=True)
for (site, shp, dt), n in sorted(SITES.items(), key=lambda kv: -kv[1] * np.prod(kv[0][1])):
    print(f"  {n:4d} x {str(shp):22s} {dt:14s} {n * np.prod(shp) * 4 / 2 ** 30:6.2f} GiB rw  {site}", flush=True)
