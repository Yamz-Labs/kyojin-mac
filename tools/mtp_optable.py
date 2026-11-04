"""Per-op time table for one MTP spec cycle on a Qwen3.5 mlx-lm quantized pack (Air). usage: mtp_optable.py PACK BF16 [--rows 1,2,3,4,5,6,8] [--depths 2,3,4]"""
import argparse, time, statistics, os
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("bf16")
ap.add_argument("--rows", default="1,2,3,4,5,6,8"); ap.add_argument("--depths", default="2,3,4"); ap.add_argument("--part", default="abc"); ap.add_argument("--q4mma", action="store_true"); ap.add_argument("--warm", type=int, default=0, help="seconds of GPU warm-up first")
a = ap.parse_args()
from kyojin_mlx import qspec as QS, mlxpack
from mlx_lm.models.gated_delta import gated_delta_update
model, tok = mlxpack.load(a.pack); lm = model.language_model
if not hasattr(lm, "lm_head") or lm.args.tie_word_embeddings:
    class _T(nn.Module):
        def __init__(s, e): super().__init__(); s.__dict__["_e"] = e
        def __call__(s, h): return s._e.as_linear(h)
    lm.__dict__["lm_head"] = _T(lm.model.embed_tokens)
rows = [int(x) for x in a.rows.split(",")]
if a.q4mma:
    from kyojin_mlx import q4mma as QM
    print("q4mma linears", QM.install(model), "MIN_ROWS", QM.MIN_ROWS, flush=True)
big = mx.random.randint(0, 2**31 - 1, (1 << 27,)).astype(mx.uint32); mx.eval(big)
bw = 0.0
for _ in range(8):
    t0 = time.perf_counter(); mx.eval(mx.sum(big)); bw = max(bw, big.nbytes / (time.perf_counter() - t0) / 1e9)
del big
print(f"read bandwidth (streaming reduce, best of 8): {bw:.0f} GB/s", flush=True)
if a.warm:
    z = mx.random.normal((2048, 2048)).astype(mx.float16); t0 = time.perf_counter()
    while time.perf_counter() - t0 < a.warm: mx.eval([z @ z for _ in range(8)])
def nbytes(m):
    m = getattr(m, "_e", m)
    return m.weight.nbytes + m.scales.nbytes + m.biases.nbytes
def bench(fn, n=10, warm=2):
    """fn() -> list of lazy arrays (n independent/chained graphs queued, one eval); returns ms per graph, median of 5."""
    ts = []
    for r in range(warm + 5):
        t0 = time.perf_counter(); outs = [fn() for _ in range(n)]; mx.eval(outs); dt = time.perf_counter() - t0
        if r >= warm: ts.append(dt / n * 1e3)
    return statistics.median(ts)
H = lm.args.hidden_size
if "a" in a.part:
    print("== A. trunk + linears (ms per call, all layers) ==", flush=True)
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    p = tok.apply_chat_template([{"role": "user", "content": "Explain how a hash map handles collisions."}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    p = p["input_ids"] if not isinstance(p, list) else p
    mx.eval(lm.model(mx.array(p)[None], cache))
    L = lm.model.layers
    groups = {"gdn.in_qkv": [l.linear_attn.in_proj_qkv for l in L if l.is_linear], "gdn.in_z": [l.linear_attn.in_proj_z for l in L if l.is_linear],
              "gdn.in_b": [l.linear_attn.in_proj_b for l in L if l.is_linear], "gdn.in_a": [l.linear_attn.in_proj_a for l in L if l.is_linear],
              "gdn.out": [l.linear_attn.out_proj for l in L if l.is_linear],
              "att.q": [l.self_attn.q_proj for l in L if not l.is_linear], "att.k": [l.self_attn.k_proj for l in L if not l.is_linear],
              "att.v": [l.self_attn.v_proj for l in L if not l.is_linear], "att.o": [l.self_attn.o_proj for l in L if not l.is_linear],
              "mlp.gate": [l.mlp.gate_proj for l in L], "mlp.up": [l.mlp.up_proj for l in L], "mlp.down": [l.mlp.down_proj for l in L],
              "lm_head": [lm.lm_head]}
    def K_of(m):
        if hasattr(m, "weight") and m.weight.dtype == mx.uint32: return m.weight.shape[1] * 32 // getattr(m, "bits", 4)
        return m.weight.shape[1]
    res = {}; roofs = {}
    tuned = dict(QM._TUNE) if a.q4mma else {}
    def modes():
        return [("stock", 99)] + ([("q4mma-tuned", QM._DEFAULT_MIN), ("q4mma-flat5", QM._DEFAULT_MIN)] if a.q4mma else [])
    def timed(fn, **kw):
        out = {}
        for rep in range(3):                                     # interleaved: stock, q4mma, stock, q4mma ... one hot job
            for name, mr in modes():
                QM.MIN_ROWS = mr if a.q4mma else 5
                QM._TUNE.clear()
                if name != "q4mma-flat5": QM._TUNE.update(tuned)
                out.setdefault(name, []).append(bench(fn, **kw))
        return {k: statistics.median(v) for k, v in out.items()}
    for S in rows:
        line = {}
        for g, mods in groups.items():
            K = K_of(mods[0]) if g != "lm_head" else H
            x0 = mx.random.normal((1, S, K)).astype(lm.model.embed_tokens.scales.dtype)
            def indep():                                         # all layers' calls queued independently (weights differ per layer, nothing cached): bandwidth/ALU cost of the op itself
                return [m(x0) for m in mods]
            line[g] = timed(indep, n=2 if g != "lm_head" else 1)
            roofs[g] = sum(nbytes(m) for m in mods) / (bw * 1e9) * 1e3
        res[S] = line
        for name, _ in modes():
            tot = sum(v[name] for k, v in line.items() if k != "lm_head")
            print(f"S={S} [{name}] " + " ".join(f"{k}={v[name]:.2f}({roofs[k] / v[name] * 100:.0f}%)" for k, v in line.items()) + f" | trunk linears {tot:.2f} ms (roof {sum(v for k, v in roofs.items() if k != 'lm_head'):.2f}), head {line['lm_head'][name]:.2f} (roof {roofs['lm_head']:.2f})", flush=True)
    for S in rows:
        x = mx.array([[p[-1]] * S])
        def untrim():                                                   # keep the KV length fixed (the graph is already built when the offset moves back)
            for l, c in zip(L, cache):
                if not l.is_linear: c.trim(S)
        def fwd():
            o = lm.model(x, cache); untrim(); return o
        def full():
            o = lm.lm_head(lm.model(x, cache)); untrim(); return o
        t_tr = timed(fwd, n=4); t_all = timed(full, n=4)
        for name, _ in modes():
            lin = sum(v[name] for k, v in res[S].items() if k != "lm_head")
            print(f"S={S} [{name}]: trunk {t_tr[name]:.2f} ms, trunk+head {t_all[name]:.2f}, trunk linears {lin:.2f} ({lin/t_tr[name]*100:.0f}%), remainder outside linears {t_tr[name]-lin:.2f} ms", flush=True)
    QM.MIN_ROWS = QM._DEFAULT_MIN if a.q4mma else 5
    QM._TUNE.update(tuned)
    # GDN kernel + SDPA
    ga = L[0].linear_attn if L[0].is_linear else None
    nk, nv, dk, dv = ga.num_k_heads, ga.num_v_heads, ga.head_k_dim, ga.head_v_dim
    for S in rows:
        q = mx.random.normal((1, S, nk, dk)).astype(mx.float16); k = q; v = mx.random.normal((1, S, nv, dv)).astype(mx.float16)
        aa = mx.random.normal((1, S, nv)).astype(mx.float16); bb = aa
        st0 = mx.zeros((1, nv, dv, dk), dtype=mx.float32)
        def g():
            st = st0; o = None
            for _ in range(24):
                o, st = gated_delta_update(q, k, v, aa, bb, ga.A_log, ga.dt_bias, st, None, use_kernel=True)
            return o
        tg = bench(g, n=3)
        att = [l.self_attn for l in L if not l.is_linear][0]
        ctx = cache[3].offset
        qq = mx.random.normal((1, 16, S, 256)).astype(mx.float16)
        kk = mx.random.normal((1, 4, ctx + S, 256)).astype(mx.float16)
        def sd():
            o = None
            for _ in range(8):
                o = mx.fast.scaled_dot_product_attention(qq + (o[..., :1, :1] * 0 if o is not None else 0), kk, kk, scale=256 ** -0.5, mask="causal" if S > 1 else None)
            return o
        print(f"S={S}: GDN kernel x24 {tg:.2f} ms, SDPA x8 (ctx {ctx}) {bench(sd, n=3):.2f} ms", flush=True)
if "b" in a.part or "c" in a.part:
    print("== B. cycle ==", flush=True)
    mtp = QS.load_mtp(model, a.bf16, quant_bits=4)
    if a.q4mma: QM.install_mtp(mtp)
    dh = QS.load_draft_head(a.bf16, 32768)
    p = tok.apply_chat_template([{"role": "user", "content": "Explain how a hash map handles collisions and compare chaining with open addressing."}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    p = p["input_ids"] if not isinstance(p, list) else p
    from mlx_lm.models.gated_delta import gated_delta_update as gdu
    for D in [int(x) for x in a.depths.split(",")]:
        sp = QS.QSpec(model, mtp, depth=D, dhead=dh); sp.prefill(p)
        for _ in range(3): sp.step()
        T = {"draft": [], "verify": [], "rollback": [], "whole": []}; toks = 0
        for r in range(12):
            # whole (overlapped) step, as served
            sp_state = None
            t0 = time.perf_counter(); new = sp.step(); mx.eval([c.state for c in sp.cache if hasattr(c, "state")][:0]); T["whole"].append((time.perf_counter() - t0) * 1e3); toks += len(new)
        # piecewise with barriers on the same engine state (read-only: draft and verify of the current state, rollback emulated by full recompute)
        nev = 0
        for r in range(8):
            t0 = time.perf_counter(); darr = sp._draft_lazy(D); mx.eval(darr); t1 = time.perf_counter()
            QS.CAP = []
            x = mx.concatenate([mx.array([sp.t0]), darr])[None]
            lg, h = sp.trunk(x); am = mx.argmax(lg[0], -1); mx.eval(am); t2 = time.perf_counter()
            cap, QS.CAP = QS.CAP, None
            keep = 1
            for l, c in zip(sp.lm.model.layers, sp.cache):
                if not l.is_linear: c.trim(D + 1 - keep)
            outs = []
            for rec in cap:
                c = sp.cache[sp.gdn_index[id(rec[0])]]
                QS.rollback_gdn(rec[0], c, keep, rec); outs.append(c[1])
            mx.eval(outs); t3 = time.perf_counter()
            sp.mcache.trim(0) if False else None
            sp.n += keep; sp.pend = ([int(am[0])], h[:, :keep]); sp.t0 = int(am[0])
            T["draft"].append((t1 - t0) * 1e3); T["verify"].append((t2 - t1) * 1e3); T["rollback"].append((t3 - t2) * 1e3)
        med = {k: statistics.median(v) for k, v in T.items()}
        print(f"D={D}: whole cycle {med['whole']:.1f} ms (tokens/cycle {toks/12:.2f}, {toks/12/med['whole']*1e3:.1f} tok/s) | barriered pieces: draft {med['draft']:.1f} verify(trunk {D+1} rows + full head) {med['verify']:.1f} rollback(GDN recompute 1 step) {med['rollback']:.1f} sum {med['draft']+med['verify']+med['rollback']:.1f}", flush=True)
