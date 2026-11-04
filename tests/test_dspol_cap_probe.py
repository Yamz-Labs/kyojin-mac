# SPDX-License-Identifier: Apache-2.0
"""Test that dspol cap probing explores deep positions even when confidence head is initially uncalibrated."""
import time
import threading
import random

import mlx.core as mx
import pytest

from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling
from kyojin_mlx import dspol as DSPOL
from test_serve_engine import V, nxt, onehot, collect


class UncalibratedConfidenceFake:
    """Fake backend where confidence head is initially uncalibrated (low confidence)
    but actual acceptance is high. This simulates a new model where the confidence
    head needs calibration data.
    
    Without cap probing, dspol would only probe D=1 and never learn deep positions.
    With cap probing, it should eventually try D=cap and learn.
    """
    kind = "uncalibrated"
    supports_spec = True
    max_spec_rows = 8
    max_rows = 8

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.gated = object()
        self.depth_asked = []
        self.accept_probs = (0.95, 0.9, 0.85, 0.8, 0.75)  # High actual acceptance
        self.confidence_phase = 0  # 0: uncalibrated (low conf), 1: calibrated (high conf)
        self.calibration_round = 20  # Switch to calibrated after this many drafted rounds

    def make_cache(self):
        return {"h": []}

    def snapshot(self, cache):

        return self.fork(cache)


    def fork(self, cache):
        return {"h": list(cache["h"])}

    def nbytes(self, cache):
        return sum(8 * len(c["h"]) for c in cache) if isinstance(cache, list) else 8 * len(cache["h"])

    def batch(self, caches):
        return list(caches)

    def unbatch(self, bc, n):
        return list(bc)

    def prefill_chunk(self, cache, ids, draft=None):
        cache["h"].extend(ids)
        return onehot([nxt(cache["h"])])

    def new_draft(self):
        return {"fed": 0}

    def draft_snapshot(self, d):
        return dict(d)

    def draft_restore(self, snap):
        return dict(snap)

    def draft_nbytes(self, d):
        return 8

    def draft_arrays(self, d):
        return []

    def launch_gated(self, bc, toks, sample, drafts=None, p0s=None):
        lg = self.step(bc, toks)
        y = sample(lg)
        mx.async_eval(y)
        return y

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt(c["h"]) for c in bc])

    def step_feed(self, bc, tokens, drafts, p0s):
        return self.step(bc, tokens)

    def draft_many(self, jobs):
        out, confs = [], []
        for d, pending, p0, depth, history in jobs:
            self.depth_asked.append(depth)
            ids, oo = history
            assert oo[-1] == pending and p0 == len(ids) + len(oo) - 1
            h, dr, cf = list(ids) + list(oo), [], []
            for i in range(depth):
                t = nxt(h)
                accept = self.rng.random() < self.accept_probs[min(i, len(self.accept_probs) - 1)]
                # Phase 0: uncalibrated - low confidence even when accepted
                # Phase 1: calibrated - high confidence when accepted
                if self.confidence_phase == 0:
                    c = self.rng.uniform(0.1, 0.4) if accept else self.rng.uniform(0.0, 0.2)
                else:
                    c = self.rng.uniform(0.8, 1.0) if accept else self.rng.uniform(0.0, 0.3)
                dr.append(t if accept else (t + 1 + self.rng.randrange(V - 1)) % V)
                cf.append(c)
                h = h + [dr[-1]]
            out.append(dr)
            confs.append(cf)
            # Switch to calibrated phase after enough drafted rounds
            if len(self.depth_asked) >= self.calibration_round:
                self.confidence_phase = 1
        self._confs = confs
        return out

    def draft_confs(self):
        return self._confs

    def spec_verify_bc(self, bc, token_lists):
        rows, starts = [], []
        for cache, toks in zip(bc, token_lists):
            starts.append(len(rows))
            rows += [nxt(cache["h"] + toks[:j + 1]) for j in range(len(toks))]
        return SpecPass(onehot(rows), starts, bc, token_lists)

    def settle(self, bc, drafts):
        pass

    def flush_drafts(self, drafts):
        pass


class SpecPass:
    def __init__(self, logits, starts, bc, token_lists):
        self.logits, self.starts, self.bc, self.token_lists = logits, starts, bc, token_lists

    def commit(self, accepted, drafts=None, p0s=None):
        for k, (cache, toks) in enumerate(zip(self.bc, self.token_lists)):
            if drafts is not None and drafts[k] is not None:
                drafts[k]["fed"] += accepted[k] + 1
            cache["h"].extend(toks[:accepted[k] + 1])


def run_uncalibrated(reqs, backend, mtp_depth=5, **kw):
    eng = Engine(backend, sessions=1, mtp_depth=mtp_depth, ctx=4096, chunk=4, block=8, prefix_bytes=0, 
                 probe_every=8, cap_probe=4, **kw)
    eng.start()
    out = {}

    def worker(name):
        ids, sub = reqs[name]
        out[name] = collect(eng.submit(ids, **sub))

    ths = [threading.Thread(target=worker, args=(n,)) for n in reqs]
    [t.start() for t in ths]
    [t.join() for t in ths]
    st = eng.status()
    eng.stop()
    return out, st, backend


def test_dspol_cap_probing_explores_deep_positions(monkeypatch):
    """With an initially uncalibrated confidence head, the policy must use cap probing
    to explore deep positions. Without cap probing, it would only probe D=1 and
    never learn that deeper positions have high acceptance.
    """
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    
    be = UncalibratedConfidenceFake(seed=42)
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=80))}
    got, st, be = run_uncalibrated(reqs, be)
    
    # Check output matches plain greedy
    want_be = UncalibratedConfidenceFake(seed=42)
    want = run_uncalibrated(reqs, want_be, mtp_depth=0)[0]
    assert got["code"][0] == want["code"][0], "Output must match plain greedy"
    
    # The policy should have used deep depths (3, 4, 5) after calibration
    # Even before calibration, cap probing should have tried deep depths
    print(f"Depths asked: {be.depth_asked[:30]}...")
    print(f"Stats: d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)} d4={st.get('spec_d4',0)} d5={st.get('spec_d5',0)}")
    
    # With cap probing, we should see depths 4 and 5 being used
    assert st.get("spec_d4", 0) + st.get("spec_d5", 0) > 0, "Cap probing should reach max depth"
    
    # The majority of rounds should be at deep depths (after calibration)
    deep_rounds = sum(st.get(f"spec_d{d}", 0) for d in [3, 4, 5])
    total_spec_rounds = st.get("spec_rounds", 1)
    assert deep_rounds > total_spec_rounds * 0.3, "Should use deep depths for high-acceptance code"


if __name__ == "__main__":
    import os
    os.environ[DSPOL.ENV_ENABLE] = "1"
    be = UncalibratedConfidenceFake(seed=42)
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=80))}
    got, st, be = run_uncalibrated(reqs, be)
    print(f"Depths asked: {be.depth_asked}")
    print(f"Stats: d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)} d4={st.get('spec_d4',0)} d5={st.get('spec_d5',0)}")