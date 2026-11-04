"""flat92_dwin.py CTX N_GEN W[,W...]: does the MTP draft need the whole context? Teacher-forced draft agreement with the MTP layer's attention
limited to the last W keys (strided view, no copy; W = 0 = full) vs the full draft.

A real CTX-token chat prompt (depth89 document + question) is prefilled with the draft state, the trunk then decodes N_GEN greedy tokens (one
row per step, so the draft state absorbs the hiddens as in serving). Then, over those N_GEN positions in one causal pass of the MTP layer:
depth-1 acceptance (draft argmax == trunk's greedy token) and agreement with the full-context draft, per W."""
import os, sys, json, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", s) for s in ("tests", "")]
import mlx.core as mx
import test_q35mlx as T
from kyojin_mlx.serve.backends import load_backend
from mlx_lm.models import qwen3_next
from mlx_lm.models.cache import KVCache
from transformers import AutoTokenizer
CTX, NG = int(sys.argv[1]), int(sys.argv[2]); WS = [tuple(int(y) for y in x.split(":")) if ":" in x else int(x) for x in sys.argv[3].split(",")]   # W, W:S (sink), -D (drop the first D)
tok = AutoTokenizer.from_pretrained(T.PACK)
ids = json.load(open(os.path.expanduser("~/rivals/corpus_ids.json"))); ids = ids["ids"] if isinstance(ids, dict) else ids
doc = tok.decode(ids[300000:300000 + CTX - 120])
QS = ["Explain how a hash map handles collisions and compare chaining with open addressing.", "Write a Python function that merges overlapping intervals, then explain its complexity."]
QS = QS[:int(os.environ.get("DWIN_NQ", len(QS)))]
be, _ = load_backend(T.PACK, mtp_head=T.HEAD, log=lambda *a: None)
head = be.dhead if be.dhead is not None else be._head
WIN = [0]
_sdpa = qwen3_next.scaled_dot_product_attention


def win_sdpa(q, k, v, cache=None, scale=None, mask=None, sinks=None):
    W, S = WIN[0] if isinstance(WIN[0], tuple) else (WIN[0], 0)    # (W, S): sink of the first S pairs + the last W; W < 0: drop the first -W pairs only
    n = W + q.shape[2]
    if W < 0:
        k, v = k[..., -W:, :], v[..., -W:, :]
    elif W and k.shape[2] > n + S:
        k, v = (mx.concatenate([k[..., :S, :], k[..., -n:, :]], axis=2), mx.concatenate([v[..., :S, :], v[..., -n:, :]], axis=2)) if S else (k[..., -n:, :], v[..., -n:, :])
    return _sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)


qwen3_next.scaled_dot_product_attention = win_sdpa
res = {}
for qi, q in enumerate(QS):
    p = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n---\nIgnore the text above. " + q}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    p = p["input_ids"] if not isinstance(p, list) else p
    cache = be.make_cache(); d = be.new_draft(); t0 = time.perf_counter()
    for i in range(0, len(p), 2048):
        lg = be.prefill_chunk(cache, p[i:i + 2048], draft=d)
    t_pf = time.perf_counter() - t0
    gen, hs = [int(mx.argmax(lg[0]))], []
    for _ in range(NG):                              # greedy trunk decode, hiddens kept for the teacher-forced draft pass
        h = be.lm.model(mx.array([[gen[-1]]], dtype=mx.int32), cache)[:, -1:]
        gen.append(int(mx.argmax(be._head(h)[0, -1]))); hs.append(h)
    # draft state: pairs up to the prompt end are absorbed; pending hidden = d.pend[-1] (last prompt position)
    hid = mx.concatenate([d.pend[-1]] + hs[:-1], axis=1)      # hidden of position j goes with token j + 1
    toks = mx.array(gen[:NG], dtype=mx.int32)                  # token after each hidden
    target = gen[1:NG + 1]
    n0 = d.mc.offset
    out = {}
    for W in [0] + WS:
        WIN[0] = W
        y = be._mtp_rows(d, toks, hid); a = mx.argmax(head(y)[0], -1); mx.eval(a); d.mc.trim(d.mc.offset - n0)
        out[W] = a.tolist()
    WIN[0] = 0
    full = out[0]
    for W, a in out.items():
        acc = sum(x == y for x, y in zip(a, target)); agr = sum(x == y for x, y in zip(a, full))
        r = res.setdefault(W, [0, 0, 0]); r[0] += acc; r[1] += agr; r[2] += len(a)
    print("Q", qi, "prompt", len(p), "prefill_s", round(t_pf, 1), {W: [sum(x == y for x, y in zip(a, target)), sum(x == y for x, y in zip(a, full))] for W, a in out.items()}, flush=True)
    del cache, d
print("DWIN92", os.path.basename(T.PACK), CTX, "n", NG, json.dumps({str(W): {"accept_d1": round(r[0] / r[2], 3), "agree_full": round(r[1] / r[2], 3)} for W, r in res.items()}), flush=True)
