"""Per-group profile of one DS4.1 decode step on a real-weight slice (generic: groups are found by module role, nothing names a layer).
usage: prof_groups.py PACK [--len 135] [--reps 15] [--json F] [--count] [--cpu]
  default   synced: every group evaluates its inputs on entry and its output on exit, so the time of a group is its own (children excluded);
            the min over --reps steps is reported, minus the per-eval floor measured with a one-kernel group (--floor).
  --cpu     no sync: python graph build time per group (exclusive), then the encode + commit time of mx.async_eval and the rest.
  --count   launches (graph nodes that dispatch GPU work) per group, from the graph of each group's own output.
The step itself is the one of tools/spec_gen.py (rows.enable: row invariant ops), so the numbers are those of the exactness-checked path."""
import argparse, collections, json, os, re, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("M10_LIGHT"):
    import light

import numpy as np
import mlx.core as mx

from kyojin_mlx import patch as P, rows as R

VIEWS = {"Reshape", "Transpose", "Broadcast", "Squeeze", "ExpandDims", "Flatten", "Slice", "AsStrided", "View", "Unflatten", "StopGradient", "Depends"}


def arrays(o, out=None):
    out = [] if out is None else out
    if isinstance(o, mx.array):
        out.append(o)
    elif isinstance(o, (list, tuple)):
        for v in o:
            arrays(v, out)
    elif isinstance(o, dict):
        for v in o.values():
            arrays(v, out)
    return out


def count_nodes(outs) -> collections.Counter:
    path = "/tmp/_m10_graph.dot"
    mx.export_to_dot(path, *outs)
    labels = re.findall(r'label ?= ?"([^"]*)"', open(path).read())
    return collections.Counter(l for l in labels if l not in VIEWS)


def tree_bytes(o, seen=None, depth=0):
    """Bytes of every distinct mx.array reachable from o (modules, lists, dicts, plain objects), cache excluded."""
    seen = set() if seen is None else seen
    if isinstance(o, mx.array):
        if id(o) in seen:
            return 0
        seen.add(id(o)); return o.nbytes
    if depth > 7 or id(o) in seen or isinstance(o, (str, bytes, int, float, bool, type(None))) or callable(o) and not hasattr(o, "__dict__"):
        return 0
    seen.add(id(o))
    if getattr(o, "grouped", None) is not None and hasattr(o, "groups"):   # grouped wo_a keeps its weights twice (per-group modules for prefill, stacked for decode): decode reads one copy
        return tree_bytes(o.grouped, seen, depth + 1)
    if hasattr(o, "_orig"):                                   # a member of a fused group: its own weights, not the group's
        return tree_bytes(o._orig, seen, depth + 1)
    n = 0
    if isinstance(o, (list, tuple, set)):
        return sum(tree_bytes(v, seen, depth + 1) for v in o)
    if isinstance(o, dict):
        n += sum(tree_bytes(v, seen, depth + 1) for k, v in o.items() if k not in ("cache", "_cache"))
    if hasattr(o, "__dict__"):
        n += sum(tree_bytes(v, seen, depth + 1) for k, v in vars(o).items() if k not in ("cache", "_cache"))
    return n


def group_bytes(name, mod):
    """Weight bytes one call of this group reads from memory (decode, one row). Containers count 0 (their children are groups)."""
    if name.endswith(".glue") or name in ("attn.rope", "engram.gather", "embed", "attn.core", "attn.window", "engram.hash"):
        return 0                                              # containers / KV and table reads (not weights)
    if name == "moe.routed":                                  # top-k of the experts
        return tree_bytes(mod) * mod._m10_k / mod._m10_e
    if name == "mhc.mix":                                     # one of the two fn tables of the block per call
        return getattr(mod, "hc_attn_fn").nbytes
    return tree_bytes(mod)


class Prof:
    def __init__(self, mode):
        self.mode = mode                      # "sync" | "cpu" | "count"
        self.stack, self.mark = [], None
        self.reset()

    def reset(self):
        self.t, self.n = collections.defaultdict(float), collections.defaultdict(int)
        self.launch = collections.defaultdict(int)
        self.kern = collections.defaultdict(collections.Counter)
        self.cur = -1                         # index of the layer being built (-1: outside the layers)
        self.bk = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))   # layer -> group -> primitive -> count
        self.bl = collections.defaultdict(lambda: collections.defaultdict(int))     # layer -> group -> launches
        self.bt = collections.defaultdict(lambda: collections.defaultdict(float))   # layer -> group -> seconds
        self.bn = collections.defaultdict(lambda: collections.defaultdict(int))     # layer -> group -> calls
        self.by = collections.defaultdict(float)                                     # group -> weight bytes read this step
        self.bb = collections.defaultdict(lambda: collections.defaultdict(float))   # layer -> group -> weight bytes

    def _charge(self):
        now = time.perf_counter()
        g = self.stack[-1] if self.stack else "<top>"
        self.t[g] += now - self.mark
        self.bt[self.cur][g] += now - self.mark
        return now

    def enter(self, name, a, k):
        if self.mode == "sync":
            xs = arrays(a) + arrays(k)
            if xs:
                mx.eval(*xs)
        elif self.mode == "count" and self.stack:
            pass
        self._charge()
        self.stack.append(name)
        self.mark = time.perf_counter()

    def exit(self, out):
        name = self.stack[-1]
        outs = arrays(out)
        if self.mode == "count" and outs:
            c = count_nodes(outs)
            self.launch[name] += sum(c.values())
            self.bl[self.cur][name] += sum(c.values())
            self.bk[self.cur][name].update(c)
            self.kern[name].update(c)
        if self.mode == "sync" and outs:
            mx.eval(*outs)
        elif self.mode == "count" and outs:
            mx.eval(*outs)                    # children of the next group start from evaluated inputs
        self._charge()
        self.stack.pop()
        self.n[name] += 1
        self.bn[self.cur][name] += 1
        self.mark = time.perf_counter()


def wrap_fn(prof, name, fn):
    def w(*a, **k):
        prof.enter(name, a, k)
        o = fn(*a, **k)
        prof.exit(o)
        return o
    return w


def override(prof, mod, method, name):
    cls = type(mod)
    orig = getattr(cls, method)

    def wrapper(self, *a, **k):
        gb = group_bytes(name, self)
        prof.by[name] += gb; prof.bb[prof.cur][name] += gb
        prof.enter(name, a, k)
        o = orig(self, *a, **k)
        prof.exit(o)
        return o

    mod.__class__ = type(cls.__name__, (cls,), {method: wrapper, "__module__": cls.__module__})


def tag_layer(prof, layer, li):
    cls = type(layer)
    orig = cls.__call__

    def wrapper(self, *a, **k):
        prof.cur = li
        o = orig(self, *a, **k)
        prof.cur = -1
        return o

    layer.__class__ = type(cls.__name__, (cls,), {"__call__": wrapper, "__module__": cls.__module__})


def instrument(model, prof):
    import mlx_vlm.models.deepseek_v41.language as L
    import mlx_vlm.models.deepseek_v41.engram as E
    lm = model.language_model
    seen = 0

    def call(mod, name):
        nonlocal seen
        if mod is not None:
            override(prof, mod, "__call__", name); seen += 1

    def meth(mod, method, name):
        nonlocal seen
        if mod is not None and hasattr(mod, method):
            override(prof, mod, method, name); seen += 1

    call(lm.embed_tokens, "embed"); call(lm.norm, "norm.final"); call(lm.head, "head")
    for li, layer in enumerate(lm.layers):
        tag_layer(prof, layer, li)
        meth(layer, "hc_mixes", "mhc.mix")
        call(layer.attn_norm, "norm.attn"); call(layer.ffn_norm, "norm.ffn")
        at = layer.attn
        call(at, "attn.glue")
        for nm in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            call(getattr(at, nm), f"attn.{nm}")
        call(at.q_norm, "attn.norm"); call(at.kv_norm, "attn.norm"); call(at.rope, "attn.rope")
        meth(at, "_window_part", "attn.window"); meth(at, "_sparse_attention", "attn.core")
        if at.compressor is not None:
            c = at.compressor
            call(c, "comp.glue"); call(c.wkv, "comp.wkv"); call(getattr(c, "wgate", None), "comp.wgate"); call(c.norm, "comp.norm")
        if at.indexer is not None:
            ix = at.indexer
            call(ix, "idx.glue"); call(ix.wq_b, "idx.wq_b"); call(ix.weights_proj, "idx.weights_proj")
            if ix.owns_k:
                call(ix.wk, "idx.wk"); call(ix.k_norm, "idx.k_norm")
        f = layer.ffn
        f.switch_mlp._m10_e, f.switch_mlp._m10_k = f.gate.num_experts, f.gate.top_k
        call(f, "ffn.glue"); call(f.gate, "moe.router"); call(f.switch_mlp, "moe.routed")
        sh = f.shared_experts
        call(sh, "moe.shared.glue")
        for nm in ("gate_proj", "up_proj", "down_proj"):
            call(getattr(sh, nm), "moe.shared.lin")
        if layer.engram is not None:
            en = layer.engram
            call(en, "engram.glue"); call(en.embed, "engram.gather"); call(en.wkv, "engram.wkv")
    if lm.engram_hash is not None:
        call(lm.engram_hash, "engram.hash")
    for gname, name in (("hc_expand", "mhc.expand"), ("fake_quant_fp8_ue8m0", "fq8"), ("fake_quant_fp4_e4m3", "fq4"), ("fake_quant_fp4_ue8m0", "fq4"),
                        ("_index_scores", "idx.scores"), ("_apply_index_rotary", "idx.rotary"), ("_apply_rope_at_positions", "attn.pool_rope"),
                        ("select_candidate_blocks", "idx.candidates"), ("_swiglu", "swiglu")):
        if hasattr(L, gname):
            setattr(L, gname, wrap_fn(prof, name, getattr(L, gname))); seen += 1
    for gname, name in (("_engram_gate_score", "engram.gate"), ("_engram_apply_gate", "engram.apply")):
        if hasattr(E, gname):
            setattr(E, gname, wrap_fn(prof, name, getattr(E, gname)))
    # the static compiled hc_pre
    Blk = L.DeepseekV41Block
    Blk.hc_pre = staticmethod(wrap_fn(prof, "mhc.pre", Blk.hc_pre))
    return seen


def eval_caches(bc):
    arrs = []
    for c in bc:
        for v in vars(c).values():
            arrs += arrays(v)
    if arrs:
        mx.eval(*arrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("--len", type=int, default=135); ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--json"); ap.add_argument("--count", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bylayer", action="store_true"); ap.add_argument("--kinds", type=int, default=-2, help="print the primitive histogram of this layer (count mode)")
    ap.add_argument("--s5", action="store_true", help="install fused_group (S5) first")
    ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, spec, cfg = P.build_model(a.pack)
    P.patch_model(model, a.pack, spec, cfg)
    R.enable(model, spec)
    if a.s5:
        from kyojin_mlx import fused_group as FG
        print("S5:", {k: v for k, v in FG.install(model).items() if k != "names"})
    lm = model.language_model
    vocab = cfg["text_config"]["vocab_size"]
    rng = np.random.default_rng(a.seed)
    ids = rng.integers(16, vocab - 16, a.len).tolist()
    cache = lm.make_cache()
    for i in range(0, len(ids), a.chunk):
        lg = lm(mx.array(ids[i:i + a.chunk])[None], cache=cache).logits[:, -1].astype(mx.float32)
        mx.eval(lg)
    eval_caches(cache)
    tok = mx.argmax(lg, axis=-1)
    mode = "count" if a.count else "cpu" if a.cpu else "sync"
    prof = Prof(mode)
    instrument(model, prof)

    # floor of one synced group: a one-kernel group in the same harness
    x0 = mx.ones((1, 5120), mx.float16); mx.eval(x0)
    fl = []
    if mode == "sync":
        prof.reset(); prof.mark = time.perf_counter()
        for _ in range(200):
            prof.enter("floor", (), {}); y = x0 * 1.0001; prof.exit(y)
            x0 = y
        floor = prof.t["floor"] / prof.n["floor"]
        print(f"eval floor per group: {1e6 * floor:.1f} us")
    else:
        floor = 0.0

    def step(cur):
        return R.step(model, cache, cur)

    runs = []
    for rep in range(a.reps + 3):
        prof.reset()
        prof.stack = []
        prof.mark = time.perf_counter()
        t0 = time.perf_counter()
        lg = step(tok)
        t_build = time.perf_counter() - t0
        t1 = time.perf_counter()
        if mode == "cpu":
            mx.async_eval(lg)
            t_enc = time.perf_counter() - t1
            mx.eval(lg); eval_caches(cache)
            t_total = time.perf_counter() - t0
        else:
            prof.enter("<end>", (), {}); mx.eval(lg); eval_caches(cache); prof.exit(None)
            t_enc = 0.0
            t_total = time.perf_counter() - t0
        tok = mx.argmax(lg, axis=-1)
        if rep >= 3:
            runs.append(dict(t=dict(prof.t), n=dict(prof.n), launch=dict(prof.launch), by=dict(prof.by), bb={l: dict(v) for l, v in prof.bb.items()}, kern={k: dict(v) for k, v in prof.kern.items()},
                             build=t_build, enc=t_enc, total=t_total,
                             bl={l: dict(v) for l, v in prof.bl.items()}, bk={l: {g: dict(c) for g, c in v.items()} for l, v in prof.bk.items()}, bt={l: dict(v) for l, v in prof.bt.items()}, bn={l: dict(v) for l, v in prof.bn.items()}))
        if mode == "count" and rep >= 3:
            break
    names = sorted({k for r in runs for k in r["t"]})
    agg = {}
    for nm in names:
        ts = [r["t"].get(nm, 0.0) for r in runs]
        n = max(r["n"].get(nm, 0) for r in runs)
        agg[nm] = dict(ms=1e3 * min(ts), ms_med=1e3 * statistics.median(ts), n=n, ms_net=1e3 * max(0.0, min(ts) - floor * n), launches=runs[0]["launch"].get(nm, 0), mb=runs[0]["by"].get(nm, 0.0) / 1e6)
    tot = 1e3 * min(r["total"] for r in runs)
    print(f"mode {mode}: step {tot:.2f} ms (min of {len(runs)}), build {1e3 * min(r['build'] for r in runs):.2f} ms" + (f", async_eval return {1e3 * min(r['enc'] for r in runs):.2f} ms" if mode == 'cpu' else ""))
    key = "ms" if mode != "sync" else "ms_net"
    for nm, d in sorted(agg.items(), key=lambda kv: -kv[1][key]):
        print(f"{nm:20s} calls {d['n']:3d}  {d['ms']:7.3f} ms" + (f"  net {d['ms_net']:7.3f}" if mode == "sync" else "") + (f"  launches {d['launches']:4d}" if mode == "count" else ""))
    print(f"sum of groups {sum(d[key] for d in agg.values()):.2f} ms; peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    if a.bylayer:
        r = runs[0] if mode == "count" else min(runs, key=lambda r: r["total"])
        src = r["bl"] if mode == "count" else r["bt"]
        for l in sorted(src, key=int):
            row = sorted(src[l].items(), key=lambda kv: -kv[1])
            tot_l = sum(v for _, v in row)
            print(f"layer {l}: " + (f"{int(tot_l)} launches" if mode == "count" else f"{1e3 * tot_l:.3f} ms") + "  " +
                  "  ".join(f"{g}={int(v) if mode == 'count' else round(1e3 * v, 3)}" for g, v in row[:16]))
    if a.kinds > -2 and mode == "count":
        for g, c in sorted(runs[0]["bk"][a.kinds].items(), key=lambda kv: -sum(kv[1].values())):
            print(f"  L{a.kinds} {g:16s} {sum(c.values()):3d}: " + " ".join(f"{k}x{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
    if a.json:
        json.dump(dict(pack=a.pack, mode=mode, len=a.len, floor_us=1e6 * floor, step_ms=tot, groups=agg, runs=runs[-1:] if mode == "count" else [dict(bt=min(runs, key=lambda r: r["total"])["bt"], bn=min(runs, key=lambda r: r["total"])["bn"])]), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
