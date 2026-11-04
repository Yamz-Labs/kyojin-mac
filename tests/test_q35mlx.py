"""Qwen3.5 mlx-lm pack behind the engine with MTP speculative decode: the draft state follows prefill, spec rounds, rollback, prefix snapshots and
plain batched steps; greedy output agrees with plain decode; the verify rows stay within rounding of the plain rows (KL / top-1 gate).
Needs a Qwen3.5 mlx pack with an mtp/ sidecar: KYOJIN_Q35MLX=<dir> (e.g. ~/rivals/models/Qwen3.5-4B-mlx4)."""
import os
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
PACK = os.environ.get("KYOJIN_Q35MLX", os.path.expanduser("~/rivals/models/Qwen3.5-4B-mlx4"))
HEAD = os.environ.get("KYOJIN_Q35_HEAD")                  # draft head directory for a pack without its own mtp/ sidecar (EXL3 packs)
pytestmark = [pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal"),
              pytest.mark.skipif(not os.path.isdir(HEAD or os.path.join(PACK, "mtp")), reason="needs a Qwen3.5 mlx pack with an mtp/ sidecar")]


def prompt(text, reps=1):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PACK)
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    ids = ids["input_ids"] if not isinstance(ids, list) else ids
    return ids * reps


def run(spec, waves, n, prefix=False):
    """Load the pack and run the engine loop on ONE thread (MLX streams are per thread). waves: list of lists of prompts; a wave is submitted when
    the previous one is done. Returns [[tokens of each request]], engine stats."""
    from kyojin_mlx.serve.backends import load_backend
    from kyojin_mlx.serve.engine import Engine
    from kyojin_mlx.serve.sampling import Sampling
    box = {}
    ready = threading.Event()

    def worker():
        be, _ = load_backend(PACK, mtp=None if spec else False, mtp_head=HEAD, log=lambda *a: None)
        box["supports"] = be.supports_spec
        eng = Engine(be, sessions=4, ctx=2048, chunk=64, block=64, prefix_bytes=(256 << 20) if prefix else 0, mtp_depth=be.default_mtp_depth if spec else 0, log=print,
                     row_cost=getattr(be, "row_cost", 0.4), draft_cost=getattr(be, "draft_cost", 0.15))
        box["eng"] = eng
        ready.set()
        eng.run()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert ready.wait(120)
    eng, res = box["eng"], []
    for wave in waves:
        hs = [eng.submit(ids, max_new=n, sampling=Sampling()) for ids in wave]
        outs = []
        for h in hs:
            out = []
            while True:
                ev = h.events.get(timeout=120)
                if ev[0] == "tok":
                    out.append(ev[1])
                elif ev[0] == "done":
                    assert ev[1] == "length", ev
                    break
            outs.append(out)
        res.append(outs)
    stats = {**dict(eng.stats), **eng.status()}
    eng.stop()
    t.join(30)
    assert box["supports"] == spec
    return res, stats


A = "Count from one to twenty in words, then list three primes."
B = "Explain in three sentences how a hash map handles collisions."


def test_spec_matches_plain_and_accepts_drafts():
    ids = prompt(A, 3)                                               # > one prefill chunk: the draft state follows the chunk boundaries
    (plain,), _ = run(False, [[ids]], 48)
    (spec,), st = run(True, [[ids]], 48)
    assert len(spec[0]) == len(plain[0]) == 48
    assert spec[0][:16] == plain[0][:16]
    assert st["spec_accepted"] > 0 and st["spec_rounds"] < 48


def test_prefix_snapshot_restores_the_draft_state():
    ids = prompt(A, 4)
    (r1, r2), st = run(True, [[ids], [ids]], 40, prefix=True)        # the second request restores the first one's snapshots (prefix cache + draft ring)
    assert st["prefix_hits"] > 0
    assert r2[0][:24] == r1[0][:24]
    assert st["spec_accepted"] > 0


def test_several_sessions_use_plain_steps_and_keep_the_drafts_alive():
    a, b = prompt(A), prompt(B)
    (solo_a,), _ = run(True, [[a]], 64)
    (two,), st = run(True, [[a, b]], 64)                             # b is shorter to finish? both 64 tokens: a leaves, then b is solo again
    assert [len(x) for x in two] == [64, 64]
    assert two[0][:16] == solo_a[0][:16]
    assert st["spec_rounds"] > 0                                     # the plain batched steps fed the hiddens, the last session still drafts afterwards


def test_verify_rows_stay_within_rounding_of_the_plain_rows():
    """Teacher forced: the logits of a 4 row (MLX qmm) and a 5 row (q4mma) verify block against the 1 row plain steps."""
    from kyojin_mlx.serve.backends import load_backend
    be, _ = load_backend(PACK, mtp_head=HEAD, log=lambda *a: None)
    ids = prompt(A, 2)
    n_ctx = len(ids)
    (nat,), _ = run(False, [[ids]], 80)
    text = nat[0]                                                    # 80 natural greedy tokens, teacher forced after the context
    ca, cb4, cb5 = be.make_cache(), be.make_cache(), be.make_cache()
    for c in (ca, cb4, cb5):
        for i in range(0, n_ctx, 64):
            be.prefill_chunk(c, ids[i:i + 64])
    ref, bc = [], be.batch([ca])
    for t in text:
        ref.append(be.step(bc, mx.array([t], dtype=mx.int32))[0])
    ref = mx.stack(ref)

    def rows(cache, R):
        out = []
        for i in range(0, len(text), R):
            blk = text[i:i + R]
            h = be.lm.model(mx.array(blk, dtype=mx.int32)[None], cache)
            out.append(be._head(h)[0].astype(mx.float32))
        return mx.concatenate(out, axis=0)

    for R, cache in ((4, cb4), (5, cb5)):
        got = rows(cache, R)
        lp, lq = ref - mx.logsumexp(ref, axis=-1, keepdims=True), got - mx.logsumexp(got, axis=-1, keepdims=True)
        kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1)
        mx.eval(kl)
        top1 = float(np.mean(np.array(mx.argmax(ref, -1)) == np.array(mx.argmax(got, -1))))
        print(f"KLSTAT R={R} mean {float(kl.mean()):.2e} p99 {float(mx.sort(kl)[int(0.99 * kl.size)]):.2e} max {float(kl.max()):.2e} top1 {top1:.3f}", flush=True)
        assert float(kl.mean()) <= 2e-3 and float(kl.max()) <= 2e-2, (R, float(kl.mean()), float(kl.max()))
        assert top1 >= 0.99, (R, top1)
