"""Hy4 MTP speculative decode (CPU only, tiny synthetic Hy4 GGUF with a next-n block): (a) the draft head against an independent float64 numpy
transcription of llama.cpp graph_mtp, (b) greedy ids with speculation ON == OFF, (c) rollback of a forced rejection: the next step equals the plain run."""
import importlib

import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

E = importlib.import_module("test_skeleton_hy4_e2e")
from kyojin_mlx import foreign_gguf as GF  # noqa: E402
from kyojin_mlx import foreign_skeleton as SK  # noqa: E402
from kyojin_mlx import foreign_skeleton_rows as R  # noqa: E402
from kyojin_mlx import foreign_synth as SY  # noqa: E402
from kyojin_mlx import engram_gguf as EG  # noqa: E402
from kyojin_mlx import hy4_mtp as M  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

A, D, V = E.A, E.D, E.V


def _head_tensors(rng, t):
    f = lambda *s, k=0.3: (rng.standard_normal(s) * k).astype(np.float32)     # noqa: E731
    p0, p1, q = "blk.0.", "blk.1.", "blk.2."
    out = {}
    for k, v in t.items():                                                      # the head block: attention + indexer shaped like layer 0, MoE like layer 1
        src = None
        if k.startswith(p0) and not any(s in k for s in ("hc_", "ffn_gate.", "ffn_up.", "ffn_down.")):
            src = q + k[len(p0):]
        elif k.startswith(p1) and any(s in k for s in ("ffn_gate_inp", "exp_probs_b", "ffn_routed", "_exps", "_shexp")):
            src = q + k[len(p1):]
        elif k.startswith(p1) and k.endswith(("ffn_norm.weight", "post_ffw_norm.weight")):
            src = q + k[len(p1):]
        if src:
            out[src] = (v + rng.standard_normal(v.shape).astype(np.float32) * 0.05 * np.abs(v).mean()).astype(np.float32)
    out[q + "attn_norm.weight"] = t[p0 + "attn_norm.weight"].copy()
    out[q + "nextn.eh_proj.weight"] = f(D, 2 * D)
    for r in ("enorm", "hnorm", "shared_head_norm"):
        out[q + f"nextn.{r}.weight"] = (1 + rng.standard_normal(D) * 0.1).astype(np.float32)
    return out


def _kv_mtp():
    kv = E._kv()
    p = A + "."
    kv[p + "block_count"], kv[p + "nextn_predict_layers"] = 3, 1
    kv[p + "attention.head_count"] = E.NH + [E.NH[0]]
    kv[p + "attention.sliding_window_pattern"] = [False, True, False]
    kv[p + "swiglu_clamp_exp"] = [0.0, E.CLAMP, E.CLAMP]
    return kv                                                                   # indexer.types stays at the 2 trunk entries: the head always owns an indexer


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    rng = np.random.default_rng(11)
    t = E._model_tensors(rng)
    t.update(_head_tensors(rng, t))
    table = rng.integers(0, 256, (sum(E.PRIMES), E.HD)).astype(np.uint8)
    table[(table & 127) == 127] = 0x38
    path = str(tmp_path_factory.mktemp("hy4m") / "hy4_mtp.gguf")
    out = [(EG.TABLE, (E.HD, table.shape[0]), "F8_E4M3", table)] + [(n, a.shape[::-1], "F32", a.astype(np.float32)) for n, a in t.items()]
    SY.write_gguf(path, _kv_mtp(), out)
    model, rep = SK.load_cpu(path, row=R.ROWS[A])
    g = GF.GgufFile(path)
    head = M.build_head(g, model, log=lambda *_: None)
    return t, path, model, head, M.Mtp(model, head)


def _ref_head(t, toks, h):
    t = {k: v.astype(np.float64) for k, v in t.items()}
    q = "blk.2."
    ta = {("blk.0." + k[len(q):]): v for k, v in t.items() if k.startswith(q)}
    tm = {("blk.1." + k[len(q):]): v for k, v in t.items() if k.startswith(q)}
    pos = np.arange(len(toks), dtype=np.float64)
    e, hh = E._rms(t["token_embd.weight"][toks], t[q + "nextn.enorm.weight"]), E._rms(h, t[q + "nextn.hnorm.weight"])
    x = np.concatenate([e, hh], -1) @ t[q + "nextn.eh_proj.weight"].T
    a, _ = E._attention(ta, 0, E._rms(x, t[q + "attn_norm.weight"]), pos)
    x = x + a
    f = E._moe(tm, E._rms(x, t[q + "ffn_norm.weight"]))
    x = x + E._rms(f, t[q + "post_ffw_norm.weight"])
    y = E._rms(x, t[q + "nextn.shared_head_norm.weight"])
    return y, y @ t["output.weight"].T


IDS = [5, 17, 3, 29, 8, 8, 12, 1, 44 % V, 21 % V, 7, 0, 13]


def test_head_equals_numpy_reference(built):
    t, _, model, head, mtp = built
    ids = [i % V for i in IDS]
    lg, h = mtp.forward([KVCache() for _ in model.layers], ids)
    h = np.array(h[0]).astype(np.float64)
    toks = np.array(ids[1:])
    hin = h[:-1]                                                                # pair j: token j+1 with the trunk state of j
    y_ref, lg_ref = _ref_head(t, toks, hin)
    c = KVCache()
    y = np.array(head(model, mx.array(toks, dtype=mx.int32)[None], mx.array(hin.astype(np.float32))[None], c)[0]).astype(np.float64)
    assert np.abs(y - y_ref).max() / np.abs(y_ref).max() < 1e-5
    got = np.array(M.head_logits(model, mx.array(y.astype(np.float32))[None])[0]).astype(np.float64)
    assert np.abs(got - lg_ref).max() / np.abs(lg_ref).max() < 1e-5
    # the same pairs through the head cache in two pieces equal one piece
    c2 = KVCache()
    a = head(model, mx.array(toks[:5], dtype=mx.int32)[None], mx.array(hin[:5].astype(np.float32))[None], c2)
    b = head(model, mx.array(toks[5:], dtype=mx.int32)[None], mx.array(hin[5:].astype(np.float32))[None], c2)
    both = np.concatenate([np.array(a[0]), np.array(b[0])])
    assert np.abs(both - y).max() / np.abs(y).max() < 1e-5
    # the greedy draft chain of the Mtp class equals the numpy chain's first token
    st = mtp.new_state()
    mtp.feed(st, ids, mx.array(h.astype(np.float32))[None]), mtp.absorb(st)


def _fresh(model):
    model.engram_source = SK.engram_source_of(GF.GgufFile(_PATH[0]))
    return [KVCache() for _ in model.layers]


_PATH = [None]


def _prompts():
    rng = np.random.default_rng(3)
    return [list(rng.integers(0, V, 9)), list(rng.integers(0, V, 14))]


@pytest.mark.parametrize("pi", [0, 1])
def test_greedy_ids_on_equal_off(built, pi):
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = _prompts()[pi]
    off = M.generate(mtp, _fresh(model), None, ids, 64, 3, spec=False)
    stats = []
    st = mtp.new_state()
    on = M.generate(mtp, _fresh(model), st, ids, 64, 3, spec=True, stats=stats)
    assert on == off
    assert stats and sum(d for d, _ in stats) > 0                                # rounds really drafted and verified


def _rejected_next(mtp, model, ids, pending, bad, nxt, exact):
    c = _fresh(model)
    mtp.forward(c, ids)
    lgv, h = mtp.verify(c, pending, bad)
    m, got_next = mtp.accept(lgv, bad)
    assert m == 0 and got_next == nxt                                            # the forced rejection
    mtp.commit(c, None, [pending] + bad, h, m, exact=exact)
    assert c[0].offset == len(ids) + 1 and model.engram_source.history_len() == len(ids) + 1
    got, _ = mtp.forward(c, [nxt])
    return np.array(got), c


def _plain_next(mtp, model, ids):
    c = _fresh(model)
    lg, _ = mtp.forward(c, ids)
    pending = int(mx.argmax(lg[-1]).item())
    ref_l, _ = mtp.forward(c, [pending])
    nxt = int(mx.argmax(ref_l[0]).item())
    ref_next, _ = mtp.forward(c, [nxt])
    return pending, nxt, np.array(ref_next)


def test_rollback_exact_mode_bit_equal_to_plain(built):
    """Force a rejection (drafts that cannot be the greedy token). KYOJIN_HY4_MTP_EXACT: the next one-row step equals the plain run's logits BIT FOR BIT."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = [i % V for i in IDS]
    pending, nxt, ref = _plain_next(mtp, model, ids)
    got, _ = _rejected_next(mtp, model, ids, pending, [(nxt + 1) % V, (nxt + 2) % V], nxt, True)
    assert float(np.abs(got - ref).max()) == 0.0


def test_rollback_fast_mode(built):
    """Default commit (trim only): stale draft rows beyond the offset never reach the next step (bit-equal whatever the rejected drafts were); against the
    plain one-row run the kept pending row came from a 3-row matmul, so only f32 row-count noise is allowed."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = [i % V for i in IDS]
    pending, nxt, ref = _plain_next(mtp, model, ids)
    a, ca = _rejected_next(mtp, model, ids, pending, [(nxt + 1) % V, (nxt + 2) % V], nxt, False)
    b, _ = _rejected_next(mtp, model, ids, pending, [(nxt + 3) % V, (nxt + 4) % V], nxt, False)
    assert float(np.abs(a - b).max()) == 0.0
    assert float(np.abs(a - ref).max()) < 1e-5 * float(np.abs(ref).max())
    assert ca[0].offset == len(ids) + 2


def test_exact_mode_ids_on_equal_off(built):
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = _prompts()[0]
    off = M.generate(mtp, _fresh(model), None, ids, 40, 3, spec=False)
    on = M.generate(mtp, _fresh(model), mtp.new_state(), ids, 40, 3, spec=True, exact=True)
    assert on == off


@pytest.mark.parametrize("exact", [False, True])
def test_mixed_accept_reject_rounds_ids_equal(built, exact):
    """The random tiny head accepts almost nothing, so an oracle drafter (the plain continuation with a wrong token planted every third round) drives rounds
    that accept 0..3 drafts: ids ON == OFF over 64 tokens, and the head state stays in step with the trunk."""
    _, path, model, head, mtp = built
    _PATH[0] = path
    ids = _prompts()[1]
    off = M.generate(mtp, _fresh(model), None, ids, 64, 3, spec=False)
    st, rnd, real = mtp.new_state(), [0], mtp.draft

    def oracle(state, pending, depth):
        assert real(state, pending, depth) is not None                           # the real head runs (state lines up) ...
        n = len(state.toks) - len(ids) + 1                                        # ... but its proposals are replaced by the oracle's
        ds = off[n:n + depth]
        rnd[0] += 1
        if rnd[0] % 3 == 0 and ds:
            ds = ds[:1] + [(ds[1] + 1) % V if len(ds) > 1 else 0] + ds[2:]
        return list(ds) or None
    mtp.draft = oracle
    stats = []
    try:
        on = M.generate(mtp, _fresh(model), st, ids, 64, 3, spec=True, stats=stats, exact=exact)
    finally:
        del mtp.draft
    assert on == off
    acc = {m for _, m in stats}
    assert {0, 1, 3} & acc and max(acc) >= 2, stats
    assert st.cache.offset <= len(st.toks)
