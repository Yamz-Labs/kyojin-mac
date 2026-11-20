"""CPU: tiny random qwen4_exp model, NO MTP head, lookup-only FnMtpBackend: n-gram drafts through the fn-mtp verify == plain greedy tokens over 64 steps
(repetitive prompt: drafts accepted; random prompt: mostly plain / rejected). Counters: rounds, plain rounds, drafted, accepted."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mlx.core as mx
from test_fn_mtp import tiny, Holder
from kyojin_mlx.serve import fnmtp as F


def ngram(hist, depth, n=2):
    """Latest earlier occurrence of the last n tokens -> the tokens that followed it."""
    if len(hist) <= n:
        return []
    tail = hist[-n:]
    for j in range(len(hist) - n - 1, -1, -1):
        if hist[j:j + n] == tail:
            return hist[j + n:j + n + depth]
    return []


def run(prompt, lookup, steps=64, depth=4, seed=()):
    be = F.FnMtpBackend(Holder(tiny()))
    be.enable_lookup()
    assert be.supports_spec and be.lookup_only and be.new_draft() is None and be.draft_many([(None, 0, 0, 3, None)]) == [[]]
    cache = be.make_cache()
    lg = be.prefill_chunk(cache, prompt, draft=None)
    pending = int(mx.argmax(lg[0]).item())
    bc = F._Bc([cache]); out, hist = [], list(prompt)
    st = dict(rounds=0, plain=0, drafted=0, accepted=0)
    while len(out) < steps:
        drafts = ngram(list(seed) + hist + [pending], depth) if lookup else []
        if drafts and len(drafts) > 1 and st['rounds'] % 4 == 3:
            drafts[1] = (drafts[1] + 1) % 1000                          # a partial accept: rollback + replay
        if not drafts:                                                  # engine: a session with no lookup draft takes the plain step
            out.append(pending); hist.append(pending); st["plain"] += 1; st["rounds"] += 1
            pending = int(mx.argmax(be.step(bc, mx.array([pending], dtype=mx.int32))[0]).item())
            continue
        res = be.spec_verify_bc(bc, [[pending] + drafts])
        pred = [int(t) for t in mx.argmax(res.logits, axis=-1).tolist()]
        acc = 0
        while acc < len(drafts) and drafts[acc] == pred[acc]:
            acc += 1
        res.commit([acc], [None], [0])
        st["rounds"] += 1; st["drafted"] += len(drafts); st["accepted"] += acc
        out += [pending] + drafts[:acc]; hist += [pending] + drafts[:acc]
        pending = pred[acc]
    return out[:steps], st


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    ok = True
    for name, pr in (("repetitive", [5, 9, 77, 120, 33, 8] * 4), ("random", [3, 141, 59, 26, 53, 58, 97, 93, 238, 46, 26, 43])):
        ref, _ = run(pr, False)
        seed = list(pr) + ref if name == "repetitive" else ()           # repetitive: the index holds the text the model is about to reproduce (an edit that repeats its input)
        r, st = run(pr, True, seed=seed)
        print(name, "equal", r == ref, st, F.accept_stats())
        assert F.accept_stats()["drafted"] >= st["drafted"]
        ok &= r == ref
        if name == "repetitive":
            ok &= st["drafted"] > 0 and st["accepted"] > 0
    sys.exit(0 if ok else 1)
