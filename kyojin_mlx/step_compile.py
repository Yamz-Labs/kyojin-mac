"""KYOJIN_STEP_COMPILE: the decode step as compiled programs, generic over every model that loads through the kit (generic_foreign backend).

Why: a decode step is host-bound (the host builds ~2000 lazy nodes in Python and MLX encodes ~2000 launches; the GPU waits). mx.compile of a whole decoder layer
replaces that layer's Python graph build by one cached call and lets MLX fuse the elementwise chains (fewer launches).

How, with no model name anywhere (structure only):
 * a decoder layer is a module of a `layers` list. Its call is compiled as a pure function of (array arguments, the layer cache's arrays); the new cache arrays are
   returned and written back (`cache.cache[i] = v`), so the cache update is an output, never a hidden side effect.
 * whole-layer compile needs a cache whose whole state is a list of fixed-shape arrays (ArraysCache family: linear attention / SSM / conv state) and no
   left_padding / lengths. A layer with a KV cache (python offset, growing slices) keeps its own call but its cache-free `mlp` (MoE / dense block, one array in,
   one out) is compiled alone.
 * weights are captured as constants (not compile inputs): module-side lazily cached derived arrays (decode conv weights, tables) would otherwise leak tracers.
 * per (layer, shape key) the FIRST call runs the stock path (it fills every lazy per-module cache with concrete arrays), the second traces + compiles, the rest replay.
   A trace that raises (host sync, .item(), a python decision on data) marks that key bad: stock path from then on. Rows above MAX_ROWS take the stock path.
 * data-dependent parts (routed expert ids, top-k, mapped row ids) stay array ops inside the program; python decisions keyed on shapes are re-taken per shape key.
Default off. Read at load (install); `ACTIVE` toggles the installed wrappers at runtime (A/B cells)."""
import os
import ast
import copy
import inspect
import textwrap
import mlx.core as mx

from . import flags as FL
from . import verify_capture as VC

ENABLED = os.environ.get("KYOJIN_STEP_COMPILE", "0") not in ("0", "")
ACTIVE = ENABLED
MAX_ROWS = int(os.environ.get("KYOJIN_STEP_COMPILE_ROWS", "16"))
MLP = os.environ.get("KYOJIN_STEP_COMPILE_MLP", "1") != "0"        # also compile the cache-free `mlp` of layers that keep a KV cache
PREP = os.environ.get("KYOJIN_STEP_COMPILE_PREP", "0") not in ("0", "")    # layers with a host-prep module (mapped row lookup): the host part runs before the program, rows go in as an input
SPLIT = os.environ.get("KYOJIN_STEP_COMPILE_SPLIT", "0") not in ("0", "")  # KV layers: program before the attention call + program after it, the attention call itself eager
ATTN = os.environ.get("KYOJIN_STEP_COMPILE_ATTN", "0") not in ("0", "")    # KV layers: program 1 = the part before the attention call (returns the attention inputs + every value the rest needs), eager attention, program 2 = the rest only (no recompute)
MAX_KEYS = 64
LEAK_ENV = "KYOJIN_STEP_COMPILE_LEAK"                              # escape hatch: 1 restores the old per-cache-object program key (keeps every request's KV alive)
_INLINE = [0]                                                       # > 0 while a split program is traced: the wrapped mlp runs inline (it becomes part of that program)
STATS = {"compiled": 0, "eager_warm": 0, "bad": 0, "stock": 0, "bad_reasons": []}
MAX_REASONS = 32


def _bad(reason: str):
    """One shape key that cannot be compiled (the layer keeps running the stock path for THAT key, other keys may compile)."""
    STATS["bad"] += 1
    if len(STATS["bad_reasons"]) < MAX_REASONS:
        STATS["bad_reasons"].append(reason)


def _gen_sync(st: dict, box: list):
    """A runtime flag was switched (flags.set bumps GEN): drop every compiled program of this wrapper so the next calls retrace and read the new value."""
    if box[0] != FL.GEN:
        box[0] = FL.GEN
        st.clear()


def _rows(x) -> int:
    n = 1
    for d in x.shape[:-1]:
        n *= d
    return n


def _fixed_cache(c) -> bool:
    """Cache whose entire state is `.cache`: a list of arrays / None, nothing else that changes."""
    lst = getattr(c, "cache", None)
    return (isinstance(lst, list) and all(a is None or isinstance(a, mx.array) for a in lst)
            and getattr(c, "left_padding", None) is None and getattr(c, "lengths", None) is None
            and not getattr(c, "is_speculating", False) and (getattr(c, "fixed_kv", False) or not hasattr(c, "offset")))


def _tree(v) -> bool:
    """A program input: an array, or a tuple / list holding arrays (e.g. the Engram rows): they must enter the program as inputs, never as captured constants."""
    return isinstance(v, mx.array) or (isinstance(v, (list, tuple)) and any(_tree(i) for i in v))


def _sk(v):
    if isinstance(v, mx.array):
        return (v.shape, v.dtype)
    if isinstance(v, (int, float, bool, str, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return (type(v).__name__,) + tuple(_sk(i) for i in v)
    return (type(v).__name__, id(v))


def _leak() -> bool:
    """A fixed-state layer cache is one object per request. Its identity must not enter the program key and the program template must hold no array: else every request
    is a new key (retrace) and the program closure keeps that request's whole KV buffers alive until MAX_KEYS. Default: no leak. KYOJIN_STEP_COMPILE_LEAK=1 restores the old key."""
    return os.environ.get(LEAK_ENV, "0") not in ("0", "")


def _bare(cache):
    """Program template of a fixed cache: the same object without its arrays (`_shell` copies it and sets the arrays of each call)."""
    t = copy.copy(cache)
    t.cache = [None] * len(cache.cache)
    return t


def _key(args: dict, cache):
    k = []
    for n, v in args.items():
        k.append((n, _sk(v)))
    if cache is not None:
        k.append(tuple(None if a is None else (a.shape, a.dtype) for a in cache.cache))
    return tuple(k)


class _Split(Exception):
    """Raised by the attention stub while tracing the pre-cache program: carries the attention call's arguments (+ the state of every `around` frame it crossed)."""

    def __init__(self, a, k):
        super().__init__("attention reached")
        self.a, self.k = a, k
        self.state = []


class _Tape:
    """Trace-time channel of a split program. mode 'pre': the attention stub raises _Split. mode 'post': the stub returns `val`, `around` frames that
    enclosed it take their state from `frames` (program inputs) instead of the values recomputed in the trace (those become dead and are pruned)."""

    def __init__(self, mode, val=None, frames=None):
        self.mode, self.val, self.frames, self.hits = mode, val, list(frames or []), 0

    def take(self):
        return self.frames.pop(0)


TAPE = None


def around(state, x, sub):
    """Hook for a residual wrapper that calls a closure (hyper-connection stream: collapse -> sub -> expand). `state` = the tuple of values the wrapper needs after
    `sub` returns (arrays / None / statics); returns (sub(x), state). Outside a split trace it is a plain call. In a pre trace a crossing `_Split` leaves with the
    state attached; in a post trace the state of the frame that enclosed the attention call is the one carried over from the pre program (no recompute)."""
    t = TAPE
    if t is None:
        return sub(x), state
    if t.mode == "pre":
        try:
            return sub(x), state
        except _Split as e:
            e.state.append(tuple(state))
            raise
    n0 = t.hits
    y = sub(x)
    return y, (t.take() if t.hits > n0 and t.frames else state)


class _Unsupported(Exception):
    pass


def _attn_name(layer, attn):
    for k, v in layer.items():
        if v is attn:
            return k
    raise _Unsupported("attention child name")


def _is_call(n, name):
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "self" and n.func.attr == name


def _has_call(node, name):
    return any(_is_call(n, name) for n in ast.walk(node))


def _ast_split(stock, layer, name):
    """Source-level split of the layer call at the one `self.<name>(...)` call: returns (pre_src, post_src, params, needed).
    pre: statements before the call, then returns (call args, call kwargs, {needed locals}); post: needed locals restored, the call replaced by `__attn_out`, then the rest.
    `if` statements on the path whose test only reads `self` (a static layer property) are resolved now and inlined."""
    fn = ast.parse(textwrap.dedent(inspect.getsource(stock))).body[0]
    params = [a.arg for a in fn.args.args][1:]
    if fn.args.vararg or fn.args.kwonlyargs or fn.args.kwarg:
        raise _Unsupported("signature")
    env = {"self": layer}

    def static_names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} <= set(env) | set(dir(__builtins__) if not isinstance(__builtins__, dict) else __builtins__)

    def flatten(stmts):
        """The straight path through the statements for THIS layer's static config: `if` tests that only read `self` / locals bound from `self` are resolved now (taken
        branch inlined, the rest of that block after a `return` dropped); simple assignments of such locals are evaluated to resolve later tests."""
        out = []
        for s in stmts:
            if isinstance(s, ast.If) and static_names(s.test):
                taken = flatten(s.body if eval(compile(ast.Expression(s.test), "<t>", "eval"), {}, dict(env)) else s.orelse)
                out += taken
                if taken and isinstance(taken[-1], ast.Return):
                    return out
                continue
            if isinstance(s, ast.Assign) and static_names(s.value) and not any(isinstance(n, ast.Call) for n in ast.walk(s.value)):
                try:
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(s)], type_ignores=[])), "<s>", "exec"), {}, env)
                except Exception:
                    pass
            out.append(s)
            if isinstance(s, ast.Return):
                return out
        return out
    body = flatten(list(fn.body))
    if sum(1 for s in body for n in ast.walk(s) if _is_call(n, name)) != 1:
        raise _Unsupported("call count on the static path: " + str(sum(1 for s in body for n in ast.walk(s) if _is_call(n, name))))

    def find(stmts):
        for i, s in enumerate(stmts):
            if not _has_call(s, name):
                continue
            if isinstance(s, ast.If):
                if {n.id for n in ast.walk(s.test) if isinstance(n, ast.Name)} - {"self"}:
                    raise _Unsupported("data-dependent if on the path")
                taken = s.body if eval(compile(ast.Expression(s.test), "<t>", "eval"), {}, {"self": layer}) else s.orelse
                pre, post = find(taken)
                return stmts[:i] + pre, post + stmts[i + 1:]
            if not isinstance(s, (ast.Assign, ast.Expr)):
                raise _Unsupported("call inside " + type(s).__name__)
            call = next(n for n in ast.walk(s) if _is_call(n, name))
            if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
                raise _Unsupported("star args")
            cap = [ast.Assign(targets=[ast.Name("__ca", ast.Store())], value=ast.List(elts=list(call.args), ctx=ast.Load()), lineno=0),
                   ast.Assign(targets=[ast.Name("__ck", ast.Store())], value=ast.Dict(keys=[ast.Constant(k.arg) for k in call.keywords], values=[k.value for k in call.keywords]), lineno=0)]

            class Rep(ast.NodeTransformer):
                def visit_Call(self, n):
                    return ast.copy_location(ast.Name("__attn_out", ast.Load()), n) if n is call else self.generic_visit(n)
            return stmts[:i] + cap, [Rep().visit(copy.deepcopy(s) if False else s)] + stmts[i + 1:]
        raise _Unsupported("call not found")

    pre, post = find(body)
    stored = set(params) | {n.id for s in pre for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for s in post for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    needed = sorted((stored & loaded) - {"self", "cache"})
    ret = ast.Return(ast.Tuple(elts=[ast.Name("__ca", ast.Load()), ast.Name("__ck", ast.Load()),
                                     ast.Dict(keys=[ast.Constant(n) for n in needed], values=[ast.Name(n, ast.Load()) for n in needed])], ctx=ast.Load()))
    args = ast.arguments(posonlyargs=[], args=[ast.arg("self")] + [ast.arg(p) for p in params], kwonlyargs=[], kw_defaults=[], defaults=copy.deepcopy(fn.args.defaults))
    f1 = ast.FunctionDef(name="__pre", args=args, body=pre + [ret], decorator_list=[], lineno=0)
    pro = [ast.Assign(targets=[ast.Name(n, ast.Store())], value=ast.Subscript(ast.Name("__L", ast.Load()), ast.Constant(n), ast.Load()), lineno=0) for n in needed]
    args2 = ast.arguments(posonlyargs=[], args=[ast.arg("self"), ast.arg("__L"), ast.arg("__attn_out")] + ([ast.arg("cache")] if "cache" in loaded else []), kwonlyargs=[], kw_defaults=[], defaults=[])
    f2 = ast.FunctionDef(name="__post", args=args2, body=pro + post, decorator_list=[], lineno=0)
    mod = ast.fix_missing_locations(ast.Module(body=[f1, f2], type_ignores=[]))
    ns = dict(stock.__globals__)
    exec(compile(mod, "<step_compile_attn>", "exec"), ns)
    return ns["__pre"], ns["__post"], params, needed, ("cache" in loaded)


def _hosts(layer):
    """Submodules that declare the host-prep hook (duck typing): `step_host_prep(args) -> [mx.array]` run eagerly before the program, `step_host_feed(arrs)` inside it."""
    return [m for _, m in layer.named_modules() if callable(getattr(m, "step_host_prep", None)) and callable(getattr(m, "step_host_feed", None))]


def _attn_child(layer, hosts):
    """The one direct child module that takes the layer cache and is not a host-prep module: the attention whose cache update keeps python offsets."""
    cands = [v for v in layer.values() if hasattr(v, "named_modules") and not any(v is h for h in hosts) and "cache" in inspect.signature(type(v).__call__).parameters]
    return cands[0] if len(cands) == 1 else None


def _wrap_layer(layer):
    cls = layer.__class__
    stock = cls.__call__
    params = [p for p in inspect.signature(stock).parameters][1:]
    has_cache = "cache" in params
    st = {}                                                         # key -> "warm" | compiled fn | "bad"
    gen = [FL.GEN]
    hosts = _hosts(layer)
    attn = _attn_child(layer, hosts) if has_cache and not hosts else None
    attn_a = _attn_child(layer, hosts) if has_cache else None
    spl = []                                                        # [(pre, post, params, needed, post_cache)] | [None]
    abox = {}
    stub = None
    if attn is not None:
        acls = attn.__class__
        mode = {}

        def stub_call(self, *a, **k):
            t = mode["t"]
            if t.mode == "pre":
                raise _Split(a, k)
            t.hits += 1
            return t.val
        stub = type("Stub" + acls.__name__, (acls,), {"__call__": stub_call})

    def build(names, statics, cache_tpl, counts, capS=None):
        box = []                                                    # slots of the captured arrays the program returns (set while tracing)

        def fn(arrs, cache_arrs, extra):
            kw = dict(statics)
            kw.update(zip(names, arrs))
            shell = None
            if cache_tpl is not None:
                shell = _shell(cache_tpl, cache_arrs)
                kw["cache"] = shell
            o = 0
            for m, n in zip(hosts, counts):                         # host-prep modules take their rows from the program inputs
                m.step_host_feed(extra[o:o + n])
                o += n
            prev = VC.SINK
            sub = VC.SINK = VC.Sink(capS) if capS is not None and shell is not None else None
            try:
                out = stock(layer, **kw)
            finally:
                VC.SINK = prev
                for m in hosts[:len(counts)]:
                    m.step_host_feed([None])
            caps = []
            if sub is not None and id(shell) in sub.items:           # verify capture: the per-row states leave the program as outputs
                got = sub.items[id(shell)][1]
                box[:] = sorted(got)
                caps = [got[i] for i in box]
            return out, (list(shell.cache) if shell is not None else []), caps
        return mx.compile(fn), box

    ptn, PH, ptdone, tested = [], {}, [False], set()                               # passthrough args: arrays the layer only forwards to the attention (the growing mask): not program inputs, not in the shape key

    def tape_run(kw, tape):
        global TAPE
        mode.clear(); mode["t"] = tape
        attn.__class__ = stub
        TAPE = tape
        _INLINE[0] += 1
        try:
            return stock(layer, **kw)
        finally:
            _INLINE[0] -= 1
            TAPE = None
            attn.__class__ = acls
            mode.clear()

    def discover(args, ast_mode):
        """One eager run up to the attention call (lazy graph, nothing evaluated): which array arguments does the attention receive untouched and the rest of the layer never reads."""
        ptdone[0] = True
        tested.update(n for n, v in args.items() if isinstance(v, mx.array))
        try:
            keep = []
            if ast_mode:
                ca, ck, L = spl[0][0](layer, **args)
                used, keep = list(ca) + list(ck.values()), list(L.values())
            else:
                try:
                    tape_run(dict(args), _Tape("pre"))
                    return
                except _Split as sp:
                    used = list(sp.a) + list(sp.k.values())
                    keep = [v for fr in sp.state for v in fr]
            pt = [n for n, v in args.items() if n != "cache" and isinstance(v, mx.array) and any(v is u for u in used) and not any(v is k for k in keep)]
        except Exception:
            pt = []
        ptn[:] = sorted(set(ptn) | set(pt))
        PH.update({n: mx.zeros((0,)) for n in pt})

    def split_args(args):
        names = [n for n, v in args.items() if _tree(v) and n != "cache" and n not in ptn]
        statics = {n: v for n, v in args.items() if n not in names and n != "cache" and n not in ptn}
        statics.update(PH)
        return names, statics

    def item_kind(v, cache):
        if v is cache:
            return "cache", None
        for n, p in PH.items():
            if v is p:
                return "pt", n
        return ("arr", None) if isinstance(v, mx.array) else ("static", v)

    def build_split(names, statics, cache, meta):
        """KV layer, any call structure (several call sites, the attention inside a closure handed to an `around` wrapper such as the hyper-connection stream):
        program 1 = the stock layer code run up to the attention call (the stub raises), returns the attention inputs + the state of the closures it crossed;
        the attention (cache update, python offsets, selection) runs eagerly; program 2 = the stock layer code run again with the attention stub returning the
        eager result and the crossed closures' state fed from program 1 (what was recomputed in the trace is dead and pruned): the rest of the layer."""
        def run(arrs, tape):
            kw = dict(statics)
            kw.update(zip(names, arrs))
            kw["cache"] = cache
            return tape_run(kw, tape)

        def pre(arrs):
            try:
                run(arrs, _Tape("pre"))
            except _Split as sp:
                items = [("a", i, v) for i, v in enumerate(sp.a)] + [("k", n, v) for n, v in sp.k.items()]
                meta["items"] = [(t, n) + item_kind(v, cache) for t, n, v in items]
                meta["state"] = [[("arr", None) if isinstance(v, mx.array) else ("static", v) for v in fr] for fr in sp.state]
                return [v for t, n, v in items if item_kind(v, cache)[0] == "arr"] + [v for fr in sp.state for v in fr if isinstance(v, mx.array)]
            raise RuntimeError("attention call not reached")

        def post(arrs, branch, st_arrs):
            it = iter(st_arrs)
            frames = [tuple(next(it) if k == "arr" else v for k, v in fr) for fr in meta["state"]]
            return run(arrs, _Tape("post", branch[0], frames))
        return mx.compile(pre), mx.compile(post)

    def split_call(self, args, a, kw, key):
        s = st.get(key)
        if s is None:
            if len(st) > MAX_KEYS:
                return stock(self, *a, **kw)
            st[key] = "warm"
            STATS["eager_warm"] += 1
            if any(isinstance(v, mx.array) and n not in tested for n, v in args.items()):
                discover(args, False)
            return stock(self, *a, **kw)
        if s == "bad":
            return stock(self, *a, **kw)
        names, statics = split_args(args)
        cache = args["cache"]
        if s == "warm":
            try:
                meta = {}
                pre, post = build_split(names, statics, cache, meta)
                outs = pre([args[n] for n in names])
                st[key] = s = (pre, post, meta, names)
            except Exception as e:
                st[key] = "bad"
                _bad(f"{cls.__name__}[split-pre]: {(str(e).splitlines() or [repr(e)])[0][:100]}")
                return stock(self, *a, **kw)
        else:
            pre, post, meta, names = s
            outs = pre([args[n] for n in names])
        STATS["attn_pre"] = STATS.get("attn_pre", 0) + 1
        ai, ak, it = [], {}, iter(outs)
        for t, n, kind, val in meta["items"]:
            v = cache if kind == "cache" else (args[val] if kind == "pt" else (next(it) if kind == "arr" else val))
            if t == "a":
                ai.append(v)
            else:
                ak[n] = v
        st_arrs = list(it)
        branch = attn(*ai, **ak)                                    # real attention: cache update + python offsets stay eager
        try:
            out = post([args[n] for n in names], [branch], st_arrs)
        except Exception as e:
            st[key] = "bad"                                         # the attention already updated the cache: finish with the uncompiled post part, never rerun the layer
            _bad(f"{cls.__name__}[split-post]: {(str(e).splitlines() or [repr(e)])[0][:100]}")
            it2 = iter(st_arrs)
            frames = [tuple(next(it2) if k == "arr" else v for k, v in fr) for fr in meta["state"]]
            kw2 = dict(statics)
            kw2.update({n: args[n] for n in names})
            kw2.update({n: args[n] for n in ptn})
            kw2["cache"] = cache
            out = tape_run(kw2, _Tape("post", branch, frames))
        STATS["attn_post"] = STATS.get("attn_post", 0) + 1
        STATS["split"] = STATS.get("split", 0) + 1
        return out

    def build_attn(names, statics, cache, counts, meta):
        pre_f, post_f, params, needed, pcache = spl[0]

        def pre(arrs, extra):
            kw = dict(statics)
            kw.update(zip(names, arrs))
            kw["cache"] = cache
            o = 0
            for m, n in zip(hosts, counts):
                m.step_host_feed(extra[o:o + n])
                o += n
            _INLINE[0] += 1
            try:
                ca, ck, L = pre_f(layer, **kw)
            finally:
                _INLINE[0] -= 1
                for m in hosts[:len(counts)]:
                    m.step_host_feed([None])
            flat = [("a", i, v) for i, v in enumerate(ca)] + [("k", n, v) for n, v in ck.items()] + [("L", n, v) for n, v in L.items()]
            meta["items"] = [(t, n) + item_kind(v, cache) for t, n, v in flat]
            return [v for _, _, v in flat if item_kind(v, cache)[0] == "arr"]

        def post(arrs, branch):
            it = iter(arrs)
            L = {}
            for t, n, kind, val in meta["items"]:
                if t != "L":
                    continue
                L[n] = abox["cache"] if kind == "cache" else (next(it) if kind == "arr" else val)   # (a passthrough never reaches L: discover keeps it a program input then)
            _INLINE[0] += 1
            try:
                return post_f(layer, L, branch[0], *([abox["cache"]] if pcache else []))
            finally:
                _INLINE[0] -= 1
        return mx.compile(pre), mx.compile(post), post

    def attn_call(self, args, a, kw, key):
        s = st.get(key)
        if s is None:
            if len(st) > MAX_KEYS:
                return stock(self, *a, **kw)
            st[key] = "warm"
            STATS["eager_warm"] += 1
            if not hosts and any(isinstance(v, mx.array) and n not in tested for n, v in args.items()):
                discover(args, True)
            return stock(self, *a, **kw)
        if s == "bad":
            return stock(self, *a, **kw)
        names, statics = split_args(args)
        cache = args["cache"]
        abox["cache"] = cache
        extra, counts = [], []
        try:
            if hosts:
                for m in hosts:
                    r = m.step_host_prep(args)
                    extra += r
                    counts.append(len(r))
            if s == "warm":
                meta = {}
                pre, post, praw = build_attn(names, statics, cache, counts, meta)
                outs = pre([args[n] for n in names], extra)
                st[key] = s = (pre, post, meta, names, praw)
            else:
                pre, post, meta, names, praw = s
                outs = pre([args[n] for n in names], extra)
        except Exception as e:
            st[key] = "bad"
            _bad(f"{cls.__name__}[attn-pre]: {(str(e).splitlines() or [repr(e)])[0][:100]}")
            return stock(self, *a, **kw)
        ai, ak, it = [], {}, iter(outs)
        for t, n, kind, val in meta["items"]:
            if t == "L":
                continue
            v = cache if kind == "cache" else (args[val] if kind == "pt" else (next(it) if kind == "arr" else val))
            if t == "a":
                ai.append(v)
            else:
                ak[n] = v
        rest = list(it)
        branch = attn_a(*ai, **ak)                                  # real attention: cache update + python offsets stay eager
        try:
            out = post(rest, [branch])
        except Exception as e:
            st[key] = "bad"                                         # the attention already updated the cache: finish this step with the uncompiled post part, never rerun the layer
            _bad(f"{cls.__name__}[attn-post]: {(str(e).splitlines() or [repr(e)])[0][:100]}")
            out = praw(rest, [branch])
        STATS["attn"] = STATS.get("attn", 0) + 1
        STATS["attn_pre"] = STATS.get("attn_pre", 0) + 1
        STATS["attn_post"] = STATS.get("attn_post", 0) + 1
        return out

    def tape_ok():
        return attn is not None and spl and spl[0] is None

    def attn_ready():
        if not spl:
            try:
                if attn_a is None or (hosts and not PREP):
                    raise _Unsupported("no attention child / hosts without PREP")
                spl.append(_ast_split(stock, layer, _attn_name(layer, attn_a)))
            except Exception as e:
                spl.append(None)
                why = f"{cls.__name__}[attn-split]: {(str(e).splitlines() or [repr(e)])[0][:100]}"
                if attn is not None:                                # the source split refused (closure / unresolved branch): the trace split below takes the layer
                    STATS.setdefault("ast_refused", []).append(why)
                else:
                    _bad(why)
        return spl[0] is not None

    def call(self, *a, **kw):
        if not ACTIVE:
            return stock(self, *a, **kw)
        _gen_sync(st, gen)
        args = dict(zip(params, a))
        args.update(kw)
        x = args.get(params[0])
        cache = args.get("cache") if has_cache else None
        if not isinstance(x, mx.array) or _rows(x) > MAX_ROWS:
            STATS["stock"] += 1
            return stock(self, *a, **kw)
        if cache is not None and not _fixed_cache(cache):
            if ATTN and attn_ready():
                return attn_call(self, args, a, kw, _key({n: v for n, v in args.items() if n != "cache" and n not in ptn}, None) + (id(cache.__class__),))
            if (SPLIT or (ATTN and tape_ok())) and attn is not None and cache is not None:
                return split_call(self, args, a, kw, _key({n: v for n, v in args.items() if n not in ptn}, None) + (id(cache.__class__),))
            STATS["stock"] += 1
            return stock(self, *a, **kw)
        capS = VC.SINK.S if VC.SINK is not None and cache is not None else None
        leak = _leak()
        key = _key(args if leak else {n: v for n, v in args.items() if n != "cache"}, cache) + (capS,)
        s = st.get(key)
        if s is None and len(st) > MAX_KEYS:                        # shapes keep changing (growing mask): not worth a program per step
            STATS["stock"] += 1
            return stock(self, *a, **kw)
        if s is None:                                               # first sight of this shape: stock path fills the lazy caches with concrete arrays
            st[key] = "warm"
            STATS["eager_warm"] += 1
            return stock(self, *a, **kw)
        if s == "bad":
            return stock(self, *a, **kw)
        names = [n for n, v in args.items() if _tree(v) and n != "cache"]
        statics = {n: v for n, v in args.items() if n not in names and n != "cache"}
        carr = list(cache.cache) if cache is not None else []
        extra, counts = [], []
        if hosts and PREP:
            try:
                for m in hosts:
                    r = m.step_host_prep(args)
                    extra += r
                    counts.append(len(r))
            except Exception as e:
                st[key] = "bad"
                _bad(f"{cls.__name__}[prep]: {(str(e).splitlines() or [repr(e)])[0][:100]}")
                return stock(self, *a, **kw)
        if s == "warm":
            try:
                fn, box = build(names, statics, cache if leak or cache is None else _bare(cache), counts, capS)
                out, new, caps = fn([args[n] for n in names], carr, extra)
            except Exception as e:                                  # a host sync / data-dependent python in the layer: this shape stays stock
                st[key] = "bad"
                _bad(f"{cls.__name__}: {(str(e).splitlines() or [repr(e)])[0][:100]}")
                return stock(self, *a, **kw)
            st[key] = (fn, names, statics, box)
        else:
            fn, names, statics, box = s
            out, new, caps = fn([args[n] for n in names], carr, extra)
        STATS["compiled"] += 1
        if cache is not None:
            cache.cache[:] = new
            if capS is not None and VC.SINK is not None:
                for i, v in zip(box, caps):
                    VC.SINK.put(cache, i, v)
        return out

    layer.__class__ = type("Stc" + cls.__name__, (cls,), {"__call__": call, "_step_compiled": True, "_step_programs": st})


def _shell(cache, arrs):
    import copy
    sh = copy.copy(cache)
    sh.cache = list(arrs)
    return sh


def _wrap_mlp(layer):
    m = layer.mlp
    cls = m.__class__
    stock = cls.__call__
    comp = {}
    gen = [FL.GEN]

    def call(self, x, *rest, **kw):
        if not ACTIVE or rest or kw or _INLINE[0] or not isinstance(x, mx.array) or _rows(x) > MAX_ROWS:
            return stock(self, x, *rest, **kw)
        _gen_sync(comp, gen)
        k = (x.shape, x.dtype)
        s = comp.get(k)
        if s is None:
            comp[k] = "warm"
            return stock(self, x)
        if s == "bad":
            return stock(self, x)
        if s == "warm":
            try:
                fn = mx.compile(lambda y: stock(m, y))
                out = fn(x)
            except Exception as e:
                comp[k] = "bad"
                _bad(f"{cls.__name__}: {(str(e).splitlines() or [repr(e)])[0][:100]}")
                return stock(self, x)
            comp[k] = fn
            STATS["compiled"] += 1
            return out
        STATS["compiled"] += 1
        return s(x)

    m.__class__ = type("Stm" + cls.__name__, (cls,), {"__call__": call, "_step_compiled": True})


def _layers(model):
    for _, m in model.named_modules():
        ls = getattr(m, "layers", None) if not hasattr(m, "_step_compiled") else None
        if isinstance(ls, list) and ls and all(hasattr(l, "named_modules") for l in ls) and any(("cache" in inspect.signature(type(l).__call__).parameters) for l in ls):
            return ls
    return []


def install(model, log=print, force=False) -> int:
    global ACTIVE
    if not (ENABLED or force):
        return 0
    ACTIVE = True
    try:
        from .foreign_mapped_rows import install_step_hooks
        install_step_hooks(model)                                   # declares step_host_prep / step_host_feed on the modules that own a mapped row table
    except ImportError:
        pass
    layers = _layers(model)
    nl = nm = 0
    for l in layers:
        if getattr(l.__class__, "_step_compiled", False):
            continue
        _wrap_layer(l)
        nl += 1
        if MLP and hasattr(l, "mlp") and not getattr(l.mlp.__class__, "_step_compiled", False):
            _wrap_mlp(l)
            nm += 1
    if nl:
        log(f"[foreign] step_compile: {nl} decoder layers (fixed-state caches whole, KV layers mlp only: {nm} mlp wrappers), rows <= {MAX_ROWS}")
    return nl
