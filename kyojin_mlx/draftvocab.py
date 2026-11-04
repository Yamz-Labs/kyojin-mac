"""Flag draftvocab (KYOJIN_DRAFT_VOCAB=K): the draft steps take their argmax over K head rows chosen by token frequency, then map the winner back to its token id.
The verify keeps the full head, so the output text is the verify's; only the acceptance can change. K = 0: off.

The row set S (|S| = K) is the K most frequent ids of a local text sample (code + English + French notes) tokenized with the model's own tokenizer. Ids < 256
and every added / special token are always in S. Counts are built once and cached on disk (key: tokenizer hash, then K).

`sub_head(owner, head, K, tok_dir)` finds the head the draft would run and returns a `Sub`: h -> logits[.., K] over the rows of S plus `pick(logits)`, the argmax
mapped back to ids. Foreign packs go through DevPack.rows (every gemv / mma path runs on it unchanged), MLX quantized and dense heads through weight[S]."""
import glob
import hashlib
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROUND = 128                                  # K is cut down to a multiple of this (the gemv kernels want N % 16 / % 64)
SAMPLE_BYTES = 20 << 20                      # text cap of the counting sample
BASE_IDS = 256                               # ids below this are always in S


def size(vocab: int) -> int:
    """The draft vocabulary: the flag value cut to ROUND, 0 when off or when it would not be smaller than the full head."""
    try:
        k = int(os.environ.get("KYOJIN_DRAFT_VOCAB", "0") or 0)
    except ValueError:
        return 0
    k -= k % ROUND
    return k if 0 < k < vocab else 0


# ---- frequency ranking
def sample_files(roots=None) -> list[str]:
    """*.md and *.py under this repo, sorted (read only). KYOJIN_DRAFT_SAMPLE overrides the roots (os.pathsep list)."""
    home = os.path.expanduser("~")
    env = os.environ.get("KYOJIN_DRAFT_SAMPLE")
    roots = roots or ([(r, ("md", "py")) for r in env.split(os.pathsep)] if env else
                      [(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ("md", "py"))])
    out = []
    for root, exts in roots:
        for e in exts:
            out += glob.glob(os.path.join(root, "**", "*." + e), recursive=True)
    return sorted(set(f for f in out if os.path.isfile(f) and "/.git/" not in f and "/.venv/" not in f))


def _load_tok(tok_dir):
    from tokenizers import Tokenizer
    f = os.path.join(tok_dir, "tokenizer.json") if os.path.isdir(tok_dir) else tok_dir
    return Tokenizer.from_file(f), open(f, "rb").read()


def count_ids(tok, files, vocab, budget=SAMPLE_BYTES) -> np.ndarray:
    """Token id counts [vocab] over the text of `files`, stopping once `budget` bytes were read (files that cannot be read as text are skipped)."""
    cnt = np.zeros(vocab, dtype=np.int64)
    used, batch = 0, []
    def flush():
        for e in tok.encode_batch(batch, add_special_tokens=False):
            ids = np.asarray(e.ids, dtype=np.int64)
            cnt[:] += np.bincount(ids[ids < vocab], minlength=vocab)
        batch.clear()
    for f in files:
        try:
            t = open(f, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        if used + len(t) > budget:
            break
        used += len(t)
        batch.append(t)
        if len(batch) >= 64:
            flush()
    flush()
    return cnt


def counting_files(files=None) -> list[str]:
    """The files that feed the counts: every 10th file is kept out (held-out coverage tests)."""
    files = sample_files() if files is None else files
    return [f for i, f in enumerate(files) if i % 10 != 9]


def choose(cnt: np.ndarray, K: int, tok) -> np.ndarray:
    """The K row ids (sorted): ids < 256 and all added / special tokens first, then the most frequent ids (ties: lower id)."""
    vocab = len(cnt)
    must = set(range(min(BASE_IDS, vocab))) | {i for i in tok.get_added_tokens_decoder() if i < vocab}
    assert len(must) <= K, "K is smaller than the always-included ids"
    score = cnt.astype(np.float64)
    score[list(must)] = np.inf
    order = np.lexsort((np.arange(vocab), -score))
    return np.sort(order[:K]).astype(np.int32)


def cache_dir() -> str:
    from .serve.autoconfig import user_cache_dir
    return os.path.join(os.path.dirname(user_cache_dir()), "draftfreq")


def ranking(tok_dir: str, vocab: int, K: int) -> np.ndarray | None:
    """S for this tokenizer and K, from the disk cache or built once (None: no tokenizer.json at tok_dir, the draft keeps the full head)."""
    if os.path.isfile(tok_dir) and tok_dir.endswith(".gguf"):       # a GGUF file: its tokenizer.json lives in the serve dir built from the metadata
        from . import foreign_names as FN
        tok_dir = FN.gguf_serve_dir(tok_dir)
    try:
        tok, raw = _load_tok(tok_dir)
    except Exception:
        return None
    key = hashlib.sha1(raw).hexdigest()[:16]
    d = cache_dir()
    fs, fc = os.path.join(d, f"{key}_{vocab}_{K}.npy"), os.path.join(d, f"{key}_{vocab}_counts.npy")
    if os.path.exists(fs):
        return np.load(fs)
    os.makedirs(d, exist_ok=True)
    cnt = np.load(fc) if os.path.exists(fc) else count_ids(tok, counting_files(), vocab)
    np.save(fc, cnt)
    S = choose(cnt, K, tok)
    np.save(fs, S)
    return S


class Sub:
    """The draft head over the rows `ids`: __call__ gives logits[.., K], pick() the argmax mapped back to token ids."""

    def __init__(self, fn, ids):
        self.fn, self.ids = fn, mx.array(ids, dtype=mx.int32)

    def __call__(self, h):
        return self.fn(h)

    def pick(self, logits):
        return self.ids[mx.argmax(logits, -1)]


class _QRows:
    """Quantized weight rows `idx`: MLX affine / mx formats, bias optional."""

    def __init__(self, m, idx):
        cut = lambda a: None if a is None else mx.take(a, idx, axis=0)
        self.w, self.s, self.b = cut(m["weight"]), cut(m["scales"]), cut(m.get("biases"))
        self.bias = cut(m["bias"]) if "bias" in m else None
        self.gs, self.bits, self.mode = m.group_size, m.bits, getattr(m, "mode", "affine")
        mx.eval([a for a in (self.w, self.s, self.b, self.bias) if a is not None])

    def __call__(self, h):
        y = mx.quantized_matmul(h, self.w, scales=self.s, biases=self.b, transpose=True, group_size=self.gs, bits=self.bits, mode=self.mode)
        return y if self.bias is None else y + self.bias


class _DenseRows:
    def __init__(self, m, idx):
        self.w = mx.take(m["weight"], idx, axis=0)
        self.bias = mx.take(m["bias"], idx, axis=0) if "bias" in m else None
        mx.eval([a for a in (self.w, self.bias) if a is not None])

    def __call__(self, h):
        y = h @ self.w.T
        return y if self.bias is None else y + self.bias


def _build(head, ids):
    """The callable over rows `ids` (int32 array) of `head`; None for an unknown head type."""
    from . import foreign_linear as FL
    from . import q4mma
    idx = mx.array(ids, dtype=mx.int32)
    if isinstance(head, q4mma.Q4Head):
        return _build(head._e, ids)
    if isinstance(head, FL.ForeignLinear):
        sub = FL.ForeignLinear.__new__(FL.ForeignLinear)
        nn.Module.__init__(sub)
        sub.dev, sub._in_perm, sub.in_features, sub.out_features, sub.fmt = head.dev.rows(idx), head._in_perm, head.in_features, len(ids), head.fmt
        sub.out_dtype, sub._bias = head.out_dtype, None if head._bias is None else mx.take(head._bias, idx, axis=0)
        return sub
    if isinstance(head, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        return _QRows(head, idx)
    if isinstance(head, (nn.Linear, nn.Embedding)):
        return _DenseRows(head, idx)
    return None                                # an unknown head type: the draft keeps the full head


def head_of(lm, own=None):
    """The head module the draft logits come from: the draft layer's own head, else the trunk's (same pick as multiuser.head)."""
    if own is not None:
        return own
    from . import multiuser as MU
    hd = lm.__dict__.get("lm_head") if MU.HEAD_ROUTE else None
    if hd is not None:
        return hd
    tie = getattr(getattr(lm, "args", None) or lm.config, "tie_word_embeddings", None)
    if tie is None:
        tie = "lm_head" not in lm
    if tie:
        t = MU._trunk(lm)
        return t.embed_tokens if "embed_tokens" in t else t.embeddings
    return lm.lm_head


def sub_head(owner, head, K, vocab, tok_dir):
    """The frequency-ranked K-row Sub for `head`, built once and kept in owner.__dict__; None when the head type is unknown or no tokenizer is found."""
    cache = owner.__dict__.setdefault("_draft_subheads", {})
    key = (id(head), K)
    if key not in cache:
        S = ranking(tok_dir, vocab, K) if tok_dir else None
        fn = None if S is None else _build(head, S)
        cache[key] = None if fn is None else Sub(fn, S)
        print(f"draftvocab: K={K} " + ("frequency-ranked sub-head active" if cache[key] is not None else f"off (no tokenizer.json at {tok_dir} or unknown head type)"), flush=True)
    return cache[key]
