"""CPU: tiny random qwen4_exp model + random MTP head: engine-protocol spec decode (depth 1 / 2 / 3) == plain greedy tokens over 64 steps."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_vlm.models.qwen4_exp import language as L
import step_compile_micro as M
from kyojin_mlx.serve import fnmtp as F


def tiny():
    mx.random.seed(3)
    lm = L.LanguageModel(M.cfg)
    lm.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.08).astype(a.dtype) if a.dtype in (mx.float32, mx.float16, mx.bfloat16) else a, lm.parameters()))
    for l in lm.model.layers:
        if "ple" in l:
            old = l.ple.ple_embedding.ngram_embedding
            raw = np.random.RandomState(7).standard_normal((old.shard_offsets[-1], M.W)).astype(np.float16).view(np.uint8)
            from kyojin_mlx import foreign_mapped_rows as MR
            l.ple.ple_embedding.ngram_embedding = MR.MappedRows(raw, "F16", M.W)
    mx.eval(lm.parameters())
    return lm


class Holder(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm

    def make_cache(self):
        return self.language_model.make_cache()


def run(depth, steps=64, oracle=None):
    lm = tiny()
    head = F.MtpHead(M.cfg)
    head.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.08).astype(a.dtype), head.parameters()))
    be = F.FnMtpBackend(Holder(lm))
    be.enable_mtp(head)
    prompt = [5, 9, 77, 120, 33, 8, 5, 9, 77, 120]
    d = be.new_draft()
    cache = be.make_cache()
    lg = be.prefill_chunk(cache, prompt, draft=d)
    out, hist = [], None
    pending = int(mx.argmax(lg[0]).item()); emitted = [pending]
    if depth == 0:
        bc = be.batch_spec([cache]) if False else F._Bc([cache])
        while len(out) < steps:
            out.append(pending)
            lg = be.step(bc, mx.array([pending], dtype=mx.int32)); pending = int(mx.argmax(lg[0]).item())
        return out[:steps]
    bc = F._Bc([cache]); p0 = len(prompt)
    while len(out) < steps:
        drafts = be.draft_many([(d, pending, p0, depth, (prompt, emitted))])[0]
        if oracle is not None:                                          # exact drafts, every third round the 2nd one wrong: full accepts and partial rollbacks
            n = len(out)
            drafts = [int(t) for t in oracle[n + 1:n + 1 + depth]]
            if (n // 2) % 3 == 2 and len(drafts) > 1:
                drafts[1] = (drafts[1] + 1) % 1000
        res = be.spec_verify_bc(bc, [[pending] + drafts])
        pred = [int(t) for t in mx.argmax(res.logits, axis=-1).tolist()]
        acc = 0
        while acc < len(drafts) and drafts[acc] == pred[acc]:
            acc += 1
        res.commit([acc], [d], [p0])
        out += [pending] + drafts[:acc]
        p0 += acc + 1
        pending = pred[acc]; emitted += drafts[:acc] + [pending]
    return out[:steps]



import pytest as _pytest


@_pytest.fixture(autouse=True)
def _restore_default_device():
    """These tests run on the CPU (mx.set_default_device): put the previous device back, or later Metal tests in the same session see the CPU."""
    dev = mx.default_device()
    yield
    mx.set_default_device(dev)


def test_pend_bounded_with_shared_plain_steps():
    """Two sessions decoding plain together (spec_ok False): pend / draft bytes stay bounded; back solo, spec text == plain greedy."""
    mx.set_default_device(mx.cpu)
    lm = tiny()
    head = F.MtpHead(M.cfg)
    head.update(tree_map(lambda a: (mx.random.normal(a.shape) * 0.08).astype(a.dtype), head.parameters()))
    be = F.FnMtpBackend(Holder(lm)); be.enable_mtp(head)
    prompts = [[5, 9, 77, 120, 33], [8, 5, 9, 77, 3, 1]]
    ds = [be.new_draft() for _ in prompts]
    cs = [be.make_cache() for _ in prompts]
    pend = [int(mx.argmax(be.prefill_chunk(c, p, draft=d)[0]).item()) for c, p, d in zip(cs, prompts, ds)]
    seq = [[t] for t in pend]
    bc = be.batch(cs)
    be.bind_drafts(bc, ds)
    sizes = []
    for _ in range(64):
        lg = be.step(bc, mx.array(pend, dtype=mx.int32))
        pend = [int(t) for t in mx.argmax(lg, axis=-1).tolist()]
        for q, t in zip(seq, pend):
            q.append(t)
        sizes.append((max(len(d.pend) for d in ds), max(be.draft_nbytes(d) for d in ds)))
    assert max(s[0] for s in sizes) <= 1, sizes[-1]
    assert sizes[-1][1] <= sizes[8][1] * 2 + 1, (sizes[8], sizes[-1])
    # back to a solo session: spec rounds resume, tokens == plain greedy of the same model
    c, d, pr = be.unbatch(bc, 2)[0], ds[0], prompts[0]
    be.bind_drafts(None, [])
    sb = F._Bc([c]); toks = list(seq[0]); p0 = len(pr) + len(toks) - 1; pending = toks[-1]; drafted = 0
    for _ in range(12):
        dr = be.draft_many([(d, pending, p0, 2, (pr, toks))])[0]
        drafted += len(dr)
        res = be.spec_verify_bc(sb, [[pending] + dr])
        pred = [int(t) for t in mx.argmax(res.logits, axis=-1).tolist()]
        acc = 0
        while acc < len(dr) and dr[acc] == pred[acc]:
            acc += 1
        res.commit([acc], [d], [p0])
        p0 += acc + 1; toks += dr[:acc] + [pred[acc]]; pending = pred[acc]
    assert drafted > 0
    ref = be.make_cache(); r = be.prefill_chunk(ref, pr)
    rb = F._Bc([ref]); t = int(mx.argmax(r[0]).item()); want = [t]
    while len(want) < len(toks):
        t = int(mx.argmax(be.step(rb, mx.array([t], dtype=mx.int32))[0]).item()); want.append(t)
    assert toks == want[:len(toks)]


def test_head_without_experts_refused():
    import pytest
    head_cfg = M.cfg
    with pytest.raises(ValueError, match="expert"):
        F.build_head(head_cfg, {}, None)


def test_tokenizer_mismatch_refused(tmp_path):
    import pytest
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "tokenizer.json").write_text("{\"v\": 1}"); (b / "tokenizer.json").write_text("{\"v\": 2}")
    with pytest.raises(ValueError, match="tokenizer"):
        F.check_tokenizer(str(a), str(b))
    (b / "tokenizer.json").write_text("{\"v\": 1}")
    F.check_tokenizer(str(a), str(b))



def test_tokenizer_gguf_trunk(tmp_path, monkeypatch):
    """GGUF trunk (no tokenizer.json): compared id by id against tokenizer.ggml.tokens; the real FP4 head vs the real UD GGUF when both are on disk."""
    import json as _j, os as _o, pytest
    h = tmp_path / "h"; t = tmp_path / "t"; h.mkdir(); t.mkdir()
    (h / "tokenizer.json").write_text(_j.dumps({"model": {"vocab": {"a": 0, "b": 1, "c": 2}}, "added_tokens": [{"content": "<e>", "id": 3}]}))
    monkeypatch.setattr(F, "_gguf_tokens", lambda d: ["a", "b", "c", "<e>"])
    F.check_tokenizer(str(h), str(t))
    monkeypatch.setattr(F, "_gguf_tokens", lambda d: ["a", "x", "c", "<e>"])
    with pytest.raises(ValueError, match="tokenizer vocab differs"):
        F.check_tokenizer(str(h), str(t))
    monkeypatch.setattr(F, "_gguf_tokens", lambda d: None)
    with pytest.raises(ValueError, match="neither"):
        F.check_tokenizer(str(h), str(t))
    monkeypatch.undo()
    fp4, gg = "/path/to/models/qwen38_flash_next_fp4local", "/path/to/models/qwen38_flash_next_ud_q4kxl_gguf"
    if _o.path.isdir(fp4) and _o.path.isdir(gg):
        F.check_tokenizer(fp4, gg)          # the served pairing must be accepted


def test_tokenizer_same_vocab_other_bytes(tmp_path):
    """The trunk's tokenizer.json regenerated from a GGUF has other bytes but the same ids: accepted; one id changed: refused."""
    import json as _j, os as _o, pytest
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    va = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}, "added_tokens": [{"content": "<e>", "id": 2}]}
    vb = {"version": "1.0", "added_tokens": [{"content": "<e>", "id": 2, "special": True}], "model": {"vocab": {"b": 1, "a": 0}, "merges": []}}
    (a / "tokenizer.json").write_text(_j.dumps(va)); (b / "tokenizer.json").write_text(_j.dumps(vb))
    F.check_tokenizer(str(a), str(b))
    vb["model"]["vocab"] = {"a": 0, "x": 1}; (b / "tokenizer.json").write_text(_j.dumps(vb))
    with pytest.raises(ValueError, match="tokenizer vocab differs"):
        F.check_tokenizer(str(a), str(b))
    fp4 = "/path/to/models/qwen38_flash_next_fp4local"
    cache = _o.path.expanduser("~/.cache/kyojin_mlx/gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004-003be222c5ed09c0")
    if _o.path.isdir(fp4) and _o.path.isfile(_o.path.join(cache, "tokenizer.json")):
        F.check_tokenizer(fp4, cache)        # the served pairing (trunk = the loader's GGUF cache dir) must be accepted

if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    ref = run(0)
    ok = True
    for dd in (1, 2, 3):
        r = run(dd)
        print("depth", dd, "equal", r == ref, F.accept_stats())
        ok &= r == ref
        r2 = run(dd, oracle=ref + [0] * 8)
        print("depth", dd, "oracle equal", r2 == ref, F.accept_stats())
        ok &= r2 == ref
    sys.exit(0 if ok else 1)
