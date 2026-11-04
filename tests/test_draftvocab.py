"""KYOJIN_DRAFT_VOCAB: the K-row draft head gives the same logits as the rows S of the full head, its argmax maps back to token ids, for every head type."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from kyojin_mlx import draftvocab as DV  # noqa: E402

N, H, K = 512, 256, 128
S = np.sort(np.random.default_rng(7).choice(N, K, replace=False)).astype(np.int32)      # a scattered row set, not a prefix


def _x(seed=1):
    return mx.array(np.random.default_rng(seed).standard_normal((1, 3, H)).astype(np.float16))


def _check(head, sub, tol=0.0):
    x = _x()
    full, part = head(x) if not isinstance(head, nn.QuantizedEmbedding) else head.as_linear(x), sub(x)
    assert part.shape[-1] == K
    ref = full[..., mx.array(S)]
    d = float(mx.abs(ref.astype(mx.float32) - part.astype(mx.float32)).max().item())
    assert d <= tol
    sub = DV.Sub(sub, S)
    # sub-head argmax mapped back to ids == full-head argmax restricted to S
    assert (sub.pick(part) == mx.array(S)[mx.argmax(ref, -1)]).all().item()


def test_size_rounds_and_turns_off(monkeypatch):
    monkeypatch.setenv("KYOJIN_DRAFT_VOCAB", "300")
    assert DV.size(1000) == 256
    assert DV.size(200) == 0                      # not smaller than the full head
    monkeypatch.setenv("KYOJIN_DRAFT_VOCAB", "0")
    assert DV.size(1000) == 0
    monkeypatch.setenv("KYOJIN_DRAFT_VOCAB", "x")
    assert DV.size(1000) == 0


def test_dense_linear_and_embedding():
    mx.random.seed(0)
    lin = nn.Linear(H, N, bias=False)
    _check(lin, DV._build(lin, S))
    emb = nn.Embedding(N, H)
    _check(emb.as_linear if False else (lambda x: x @ emb.weight.T), DV._build(emb, S))


@pytest.mark.parametrize("bits", (4, 8))
def test_quantized_linear_and_tied_embedding(bits):
    mx.random.seed(1)
    ql = nn.QuantizedLinear(H, N, bias=False, group_size=64, bits=bits)
    _check(ql, DV._build(ql, S))
    qe = nn.QuantizedEmbedding(N, H, group_size=64, bits=bits)
    _check(qe, DV._build(qe, S))


def test_unknown_head_keeps_full():
    assert DV._build(object(), S) is None


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
@pytest.mark.parametrize("t", ("Q4_K", "Q6_K", "Q8_0", "Q4_0"))
def test_foreign_linear_rows(t):
    from kyojin_mlx import foreign_gguf as GF, foreign_linear as FL, foreign_synth as SY
    k, n = 1024, 512
    p = GF.to_pack(t, SY.random_blocks(t, n, k, np.random.default_rng(3)), k)
    lin = FL.ForeignLinear(p)
    sub = DV._build(lin, S)
    x = mx.array(np.random.default_rng(4).standard_normal((1, 1, k)).astype(np.float16))
    full, part = lin(x), sub(x)
    assert part.shape[-1] == K
    d = float(mx.abs(full[..., mx.array(S)].astype(mx.float32) - part.astype(mx.float32)).max().item())
    assert d <= 1e-2 * float(mx.abs(full).max().item())
    assert int(mx.array(S)[mx.argmax(full[..., mx.array(S)], -1)].item()) == int(DV.Sub(sub, S).pick(part).item())


def test_choose_keeps_base_and_special_ids():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models
    vocab = 1024
    tok = Tokenizer(models.WordLevel({f"w{i}": i for i in range(vocab - 3)}, unk_token="w0"))
    tok.add_special_tokens(["<|im_end|>", "<think>", "</think>"])
    cnt = np.zeros(vocab + 5, dtype=np.int64)
    cnt[600:900] = np.arange(300) + 1                      # the frequent ids sit high
    s = DV.choose(cnt, 512, tok)
    assert len(s) == 512 and (np.diff(s) > 0).all()
    assert set(range(256)) <= set(s.tolist())
    assert set(tok.get_added_tokens_decoder()) <= set(s.tolist())
    assert 899 in s and 600 + 100 in s                      # the top counts made it
    assert int(cnt[s].sum()) >= int(np.sort(cnt)[-200:].sum())
