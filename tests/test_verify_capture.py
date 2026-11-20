"""GPU: KYOJIN_VERIFY_CAPTURE. Real qwen4_exp layers (128-wide GDN heads, bf16, fused kernel + step compile): a verify pass of S = 4 rows then a partial accept
(keep = 1..3): the cache state selected from the recorded per-row states vs the snapshot + replay forward, and the next plain step's logits. Prints max |diff|
(the recurrence and conv are the kernel's own, so a diff can only come from the lower layers' row count)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
import step_compile_micro as M
M.cfg.linear_key_head_dim = M.cfg.linear_value_head_dim = 128
import test_fn_mtp as T
from kyojin_mlx.serve import fnmtp as F
from kyojin_mlx import foreign_gdn_fast as GF, step_compile as SC, verify_capture as VC, flags as FL


def setup(compile_):
    lm = T.tiny()
    lm.update(tree_map(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a, lm.parameters())); mx.eval(lm.parameters()); lm.eval()
    GF.ACTIVE = True
    GF.install(lm)
    cls = next(type(m) for _, m in lm.named_modules() if hasattr(m, "in_proj_qkv"))
    if not getattr(cls.__call__, "_cast", False):
        orig = cls.__call__
        def call(self, inputs, mask=None, cache=None): return orig(self, inputs.astype(mx.bfloat16), mask, cache)
        call._cast = True; call._kyojin_gdn_fast = True
        cls.__call__ = call
    if compile_:
        SC.ENABLED = SC.ACTIVE = True; SC.PREP = SC.ATTN = False
        SC.install(lm, force=True)
    head = F.MtpHead(M.cfg)
    head.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.08).astype(a.dtype), head.parameters()))
    be = F.FnMtpBackend(T.Holder(lm)); be.enable_mtp(head)
    return be


def arrays(cache):
    out = []
    for c in cache:
        out += [np.array(a.astype(mx.float32)) for a in (c.cache if hasattr(c, "cache") else [c.keys[..., :c.offset, :]]) if a is not None]
    return out


def case(compile_):
    be = setup(compile_)
    prompt = [5, 9, 77, 120, 33, 8, 5, 9, 77, 120, 4, 6, 7, 2, 90, 33]
    worst = {}
    for keep in (1, 2, 3):
        res = {}
        for cap in (False, True):
            VC.ACTIVE = cap; FL.GEN += 1
            VC.STATS.update(captured=0, incomplete=0)
            cache = be.make_cache(); d = be.new_draft()
            be.prefill_chunk(cache, prompt, draft=d)
            bc = F._Bc([cache])
            for _ in range(3):                                   # first sights warm, then the compiled program: the verify must hold in both
                x = [11, 12, 13, 14]
                r = be.spec_verify_bc(bc, [x]); r.commit([keep - 1], [d], [len(prompt)])
            st = arrays(cache)
            lg = be.step(bc, mx.array([21], dtype=mx.int32))
            res[cap] = (st, np.array(lg.astype(mx.float32)), dict(VC.STATS))
        assert res[True][2]["incomplete"] == 0, res[True][2]
        assert (res[True][2]["captured"] > 0) == (keep >= VC.MINKEEP), (keep, res[True][2])
        assert res[False][2]["captured"] == 0
        dst = max(float(np.abs(a - b).max()) for a, b in zip(res[False][0], res[True][0]))
        dlg = float(np.abs(res[False][1] - res[True][1]).max())
        worst[keep] = (dst, dlg, int(np.argmax(res[False][1])) == int(np.argmax(res[True][1])))
    print("compile" if compile_ else "eager", "keep -> (max state diff, max logit diff, argmax equal):", worst, flush=True)
    return worst


def test_capture_equals_replay():
    mx.set_default_device(mx.gpu)
    for c in (False, True):
        w = case(c)
        assert all(v[2] for v in w.values()), w
        assert all(v[0] == 0.0 and v[1] == 0.0 for v in w.values()), w        # captured state == replayed state, bit for bit, every keep


if __name__ == "__main__":
    test_capture_equals_replay()


def test_select_does_not_keep_the_capture_buffer():
    """The selected state owns its memory: after the sink and the d+1-state buffer go away, only one state stays alive (a view kept all d+1)."""
    import gc
    import mlx.core as mx
    from kyojin_mlx import verify_capture as VC

    class C:
        cache = [None]

    gc.collect(); mx.clear_cache(); base = mx.get_active_memory()
    ss = mx.random.normal((3, 32, 128, 128)).astype(mx.float32); mx.eval(ss)   # 3 states, 6 MiB
    c = C(); c.cache = [ss[0:1]]
    sink = VC.Sink(3); sink.put(c, 0, [ss[k:k + 1] for k in range(3)])
    sel = sink.select(c, 2); mx.eval(sel[0])
    c.cache = [sel[0]]; del ss, sink, sel; gc.collect()
    assert (mx.get_active_memory() - base) <= 2.5 * 2 ** 20
