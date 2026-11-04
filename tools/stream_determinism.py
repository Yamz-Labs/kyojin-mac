"""Stream determinism: the same verify pass (same state, same tokens) run repeatedly must record the SAME command stream (kernel names + binding counts), whoever else holds the state buffers."""
import ctypes, os, sys, hashlib
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "100000"); os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1000000")
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "replay", "build", "librp.dylib")); lib.rp_install(1)
import numpy as np, mlx.core as mx
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import dsv4_core as DC
be, cfg = load_backend(sys.argv[1], mtp=True, ced=False, log=lambda *x: None)
V = cfg.get("text_config", cfg)["vocab_size"]; rng = np.random.default_rng(7); base = rng.integers(16, V - 16, 61).tolist()
c = be.make_cache(); d = be.new_draft(); be.prefill_chunk(c, [16] + (base * 4)[:65], draft=d); bc = be.batch([c])
toks = [[int(t) for t in rng.integers(16, V - 16, 4)]]
st = DC.stack_rows(bc[0])
def one(keep_ref):
    hold = [a_[...] for l in (st.cmp, st.keys, st.win, st.kvs, st.scs) for a_ in l if a_ is not None] if keep_ref else None
    mx.synchronize(); mx.clear_cache(); lib.rp_rec_begin(80); lib.rp_rec_snap_born(80, 1)
    ps = be.spec_verify_bc(bc, toks); mx.eval(ps.logits, *[a_ for pd in st.pend.values() for a_ in vars(pd).values() if isinstance(a_, mx.array)])
    mx.synchronize(); n = lib.rp_rec_end(); lib.rp_dump(80, b"/tmp/dt.txt")
    names = [ln.split()[1] for ln in open("/tmp/dt.txt")]
    st._counts = None; st.pend = {}; st.taps = {}
    return n, hashlib.sha1("\n".join(names).encode()).hexdigest()[:10], np.array(ps.logits)
res = []
for i in range(8): res.append(one(i % 2 == 1))
print("command counts / stream hash (alternating: no extra reference / extra references on every state array):")
for i, r in enumerate(res): print(" ", i, "extra-ref" if i % 2 else "plain    ", r[0], r[1])
print("stream identical in all 8 runs:", len({(r[0], r[1]) for r in res}) == 1, "| logits identical:", all(np.array_equal(res[0][2], r[2]) for r in res))

# consecutive positions (what the replay fit does): probe k at offset p0 + k, fresh Plan, no commit
row = st.rows[0]; p0 = row.offset
def probe(k):
    row.offset = p0 + k; st._plan = None
    mx.synchronize(); mx.clear_cache(); lib.rp_rec_begin(81); lib.rp_rec_snap_born(81, 1)
    ps = be.spec_verify_bc(bc, toks); pend = [a_ for pd in st.pend.values() for a_ in vars(pd).values() if isinstance(a_, mx.array)]
    taps = list(ps.taps.values()); mx.eval(ps.logits, *pend, *taps)
    mx.synchronize(); n = lib.rp_rec_end(); lib.rp_dump(81, b"/tmp/dt2.txt")
    names = [ln.split()[1] for ln in open("/tmp/dt2.txt")]
    st._counts = None; st.pend = {}; row.offset = p0
    return n, names
mx.eval(be.spec_verify_bc(bc, toks).logits); st._counts = None; st.pend = {}; st.taps = {}          # warm
pr = [probe(k) for k in range(12)]
print("consecutive-position probes, command counts:", [n for n, _ in pr])
ref = pr[0][1]
for k, (n, names) in enumerate(pr[1:], 1):
    if names != ref:
        d = next((i for i, (a_, b_) in enumerate(zip(ref, names)) if a_ != b_), min(len(ref), len(names)))
        print(f"  probe {k} differs from probe 0 at command {d}: {ref[d] if d < len(ref) else None} vs {names[d] if d < len(names) else None}")
print("all probes share one stream:", all(nm == ref for _, nm in pr))
