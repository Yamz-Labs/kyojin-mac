"""round93.py CTX [REPS] [TQ] (ROUND93_AB=nax: fp16 KV both, label fp16 = q4nax routing on, tq = off): per-op table of one served MTP round (depth 4, the engine's lazy path) at CTX tokens, fp16 KV vs TurboQuant KV (TQ, default
6:6:4:256) in ONE process, interleaved reps (the fanless Air drifts between runs), ms best + median.
  round      draft_lazy + spec_verify_lazy + argmax + commit + settle, as Engine._spec_round (the rollback of a round runs inside the next evaluation)
  draft      the 4 MTP draft steps alone (synced)        verify    the 5-row trunk pass alone (synced)       commit   trim + GDN re-run (synced)
  attn5      the full-attention layers' self_attn on 5 rows (projections + pool write + ring + attention), trimmed back
  upd5       their cache.update_and_fetch alone (rotation, pool write, ring)           core5/1   attention core alone (5 rows / 1 row; TQ also with ROWS=0, the per-row-chunk kernel)
  upd5_old  TQ update with the verify chunk written through the rows-view patch (before 932460b)
  vs_*      the verify pass layer by layer (synced per layer): full-attention layers, GDN layers, embed + head
(timing only: verify-like ops leave the GDN states advanced, the attention caches are trimmed back)"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
TQ = sys.argv[3] if len(sys.argv) > 3 else "6:6:4:256"
os.environ.pop("KYOJIN_TQ", None)
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
CTX = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
DEPTH, ACC = 4, (1, 2, 2, 1, 3, 0)                         # accepted drafts per round (~46 % at depth 4, as served)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
pids = ids[300000:300000 + CTX]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
L = be.lm.model.layers
S = {}
AB = os.environ.get("ROUND93_AB") == "nax"                 # A/B the few-row routing instead of the KV format: fp16 KV both, q4nax on ("fp16") / off ("tq")
from kyojin_mlx import q4mma as QM
NAX0 = QM.nax_on()
for name, env in (("fp16", None), ("tq", None if AB else TQ)):
    if env: os.environ["KYOJIN_TQ"] = env
    else: os.environ.pop("KYOJIN_TQ", None)
    cache = be.make_cache(); d = be.new_draft()
    for i in range(0, len(pids), 2048): lg = be.prefill_chunk(cache, pids[i:i + 2048], draft=d)
    t0 = int(mx.argmax(lg[0]).item())                       # the first emitted token (the engine samples it from the prefill logits)
    S[name] = dict(cache=cache, d=d, bc=be.batch_spec([cache]), att=[c for l, c in zip(L, cache) if not l.is_linear], last=t0, out=[t0], n=0)
    mx.eval(be.state(cache)); mx.clear_cache()
os.environ.pop("KYOJIN_TQ", None)
time.sleep(float(os.environ.get("ROUND93_IDLE", "45")))   # the Air throttles right after a 32K prefill: let it cool before timing
attl = [l for l in L if not l.is_linear]
_e = be.lm.model.embed_tokens(mx.array([0])); H, DT = _e.shape[-1], _e.dtype
x5 = (mx.random.normal((1, 5, H)) * 0.5).astype(DT)

def round_(s):
    """One engine spec round; returns the emitted count."""
    jobs = [(s["d"], s["last"], CTX + len(s["out"]) - 1, DEPTH, (pids, s["out"]))]
    lh = be.draft_lazy(jobs)
    res = be.spec_verify_lazy(s["bc"], lh, [s["last"]])
    vals = mx.argmax(res.logits, axis=-1).tolist()
    drafts = lh.tolist()[0]
    a = min(ACC[s["n"] % len(ACC)], len(drafts)); s["n"] += 1
    res.commit([a], [s["d"]], [CTX + len(s["out"]) - 1])
    be.settle(s["bc"], [s["d"]])
    assert len(drafts) == DEPTH, drafts
    s["out"] += drafts[:a] + [vals[a]]; s["last"] = s["out"][-1]

def draft(s):
    jobs = [(s["d"], s["last"], CTX + len(s["out"]) - 1, DEPTH, (pids, s["out"]))]
    snap = (s["d"].mc.offset, s["d"].mv, list(s["d"].pend))
    lh = be.draft_lazy(jobs); mx.eval(*lh.arrays.values())
    s["d"].mc.trim(s["d"].mc.offset - snap[0]); s["d"].mv, s["d"].pend = snap[1], snap[2]

def draft_s(s):
    """The draft with the MTP layer's KV cut to 256 keys (timing only: its rows past 256 get overwritten): the MTP attention share of the draft."""
    mc = s["d"].mc; n = mc.offset; cut = n - 256
    mc.offset = 256; s["d"].mv -= cut
    jobs = [(s["d"], s["last"], CTX + len(s["out"]) - 1 - cut, DEPTH, (pids, s["out"]))]
    snap = (s["d"].mv, list(s["d"].pend))
    t0 = time.perf_counter(); lh = be.draft_lazy(jobs); mx.eval(*lh.arrays.values()); dt = (time.perf_counter() - t0) * 1e3
    assert lh.counts[0] == DEPTH
    mc.offset = n; s["d"].mv, s["d"].pend = snap[0] + cut, snap[1]
    return {"draft_s256": dt}

def verify(s):
    p = be._verify(s["bc"], [mx.array([s["last"]] + [1000 + i for i in range(DEPTH)], dtype=mx.int32)]); mx.eval(p.logits)
    for c in s["att"]: c.trim(DEPTH + 1)
    return p

def commit(s):
    p = be._verify(s["bc"], [mx.array([s["last"]] + [1000 + i for i in range(DEPTH)], dtype=mx.int32)]); mx.eval(p.logits, *be.state(s["cache"]))
    t0 = time.perf_counter()
    p.commit([1], None, None); mx.eval(*be.state(s["cache"]))
    dt = (time.perf_counter() - t0) * 1e3
    for c in s["att"]: c.trim(2)
    return dt

def attn5(s):
    outs = []
    for l, c in zip(attl, s["att"]):
        outs.append(l.self_attn(x5, mask=create_attention_mask(x5, c), cache=c))
    mx.eval(outs)
    for c in s["att"]: c.trim(5)

def upd5(s):
    outs = []
    for l, c in zip(attl, s["att"]):
        k = v = mx.zeros((1, HK, 5, HD), dtype=DT) + 0.01
        outs += list(c.update_and_fetch(k, v))
    mx.eval([o if isinstance(o, mx.array) else o.cache.kq[0] for o in outs], *be.state(s["cache"]))
    for c in s["att"]: c.trim(5)

from mlx_lm.models.base import create_ssm_mask
def vsplit(s):
    """The verify pass layer by layer, synced after each layer: ms in the full-attention layers, the GDN layers, embed + head."""
    m = be.lm.model; c = s["cache"]; QS = be.QS; QS.CAP = []
    t = {"vs_attn": 0.0, "vs_gdn": 0.0, "vs_head": 0.0}
    t0 = time.perf_counter()
    h = m.embed_tokens(mx.array([[s["last"]] + [1000 + i for i in range(DEPTH)]], dtype=mx.int32))
    fa = create_attention_mask(h, c[m.fa_idx]); ssm = create_ssm_mask(h, c[m.ssm_idx]); mx.eval(h)
    t["vs_head"] += time.perf_counter() - t0
    for l, cc in zip(m.layers, c):
        t0 = time.perf_counter(); h = l(h, mask=ssm if l.is_linear else fa, cache=cc); mx.eval(h)
        t["vs_gdn" if l.is_linear else "vs_attn"] += time.perf_counter() - t0
    t0 = time.perf_counter(); mx.eval(be._head(m.norm(h))[0]); t["vs_head"] += time.perf_counter() - t0
    QS.CAP = None
    for cc in s["att"]: cc.trim(DEPTH + 1)
    return {k: v * 1e3 for k, v in t.items()}

from kyojin_mlx import tq as TQM
from mlx_lm.models import qwen3_next as QN                  # the model's SDPA (multiuser.install: verify rows GQA-folded into one blocked pass)
def core(s, name, n=5):
    """Attention core alone, 5 query rows, on caches holding the 5 rows (fp16: the model's SDPA over the KVCache keys, TQ: TQKVCache.attend,
    also with the per-row-chunk kernel ROWS=0). The rows are trimmed back."""
    kv = []
    for c in s["att"]:
        z = mx.zeros((1, HK, n, HD), dtype=DT) + 0.01
        kv.append(c.update_and_fetch(z, z))
    q = (mx.random.normal((1, HQ, n, HD)) * 0.3).astype(DT)
    mx.eval(q, kv if name == "fp16" else [], be.state(s["cache"]))
    out = {}
    for tag, rows in (((f"core{n}", None),) if name == "fp16" or n == 1 else ((f"core{n}", TQM.ROWS), (f"core{n}_rows0", 0))):
        if rows is not None: old = TQM.ROWS; TQM.ROWS = rows
        f = (lambda: [QN.scaled_dot_product_attention(q, k, v, cache=c, scale=HD ** -0.5, mask="causal" if n > 1 else None) for (k, v), c in zip(kv, s["att"])]) if name == "fp16" else \
            (lambda: [c.attend(q, HD ** -0.5) for c in s["att"]])
        mx.eval(f())
        t0 = time.perf_counter(); mx.eval(f()); out[tag] = (time.perf_counter() - t0) * 1e3
        if rows is not None: TQM.ROWS = old
    for c in s["att"]: c.trim(n)
    return out

a0 = attl[0].self_attn
HK = a0.num_key_value_heads; HQ = a0.num_attention_heads; HD = a0.head_dim

import inspect
_new_wp = TQM.TQKVCache._write_pool
_src = inspect.getsource(_new_wp).replace("elif n <= 64:", "elif False:")
if "elif False:" in _src:
    _ns = dict(vars(TQM)); exec("if 1:\n" + _src, _ns); _old_wp = _ns["_write_pool"]
else:
    _old_wp = None                                          # TQ write path without the rows-view branch: no upd5_old
res = {}
def rec(k, v): res.setdefault(k, []).append(v)
def t_round(s, name):
    mx.eval(be.state(s["cache"]), be.draft_arrays(s["d"]))
    t0 = time.perf_counter()
    for _ in range(4): round_(s)
    mx.eval(be.state(s["cache"]), be.draft_arrays(s["d"]))
    return {"round": (time.perf_counter() - t0) * 1e3 / 4}
def t_op(f, tag):
    def g(s, name):
        t0 = time.perf_counter(); f(s); return {tag: (time.perf_counter() - t0) * 1e3}
    return g
def t_old_upd(s, name):
    if name != "tq" or _old_wp is None: return {}
    TQM.TQKVCache._write_pool = _old_wp; upd5(s)
    t0 = time.perf_counter(); upd5(s); dt = (time.perf_counter() - t0) * 1e3
    TQM.TQKVCache._write_pool = _new_wp; upd5(s)
    return {"upd5_old": dt}
OPS = [t_round, t_op(draft, "draft"), t_op(verify, "verify"), t_op(attn5, "attn5"), t_op(upd5, "upd5"), t_old_upd, lambda s, n: {"commit": commit(s)},
       lambda s, n: vsplit(s), lambda s, n: core(s, n), lambda s, n: core(s, n, 1), lambda s, n: draft_s(s)]
for s in S.values():
    for _ in range(3): round_(s)                             # steady state
k = 0
for r in range(REPS + 1):
    for op in OPS:                                           # interleaved per op, the mode order alternates (the fanless Air heats within seconds)
        k += 1
        for name in (("fp16", "tq") if k % 2 else ("tq", "fp16")):
            if AB: QM._NAX_ON[:] = [NAX0 and name == "fp16"]
            for tag, v in op(S[name], name if not AB else "fp16").items():
                if r: rec(f"{name}:{tag}", v)
print("MTPKV", S["fp16"]["d"].mc.keys.dtype, S["fp16"]["d"].mc.keys.shape)
print("ROUND93", CTX, TQ, json.dumps({k: [round(min(v), 2), round(statistics.median(v), 2)] for k, v in res.items()}), flush=True)
