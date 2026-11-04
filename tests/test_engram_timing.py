# SPDX-License-Identifier: Apache-2.0
"""Test that reproduces the Engram gate timing mismatch bug and verifies the fix.

Scenario: Engram gate active -> plain steps use launch_gated (measured plain_ms=22ms).
Drafted rounds use spec_verify_bc. The old dspol cost model used hardcoded S=1.0 units
ignoring measured plain_ms and spec_k. The fix uses measured timings.
"""
import time
import threading
import random

import mlx.core as mx
import pytest

from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling
from kyojin_mlx import dspol as DSPOL
from test_serve_engine import V, nxt, onehot, collect


class EngramGateTimingFake:
    """Simulates Engram gate timing characteristics:
    - Plain steps (D=0) go through launch_gated: 22ms
    - Drafted rounds (D>0) go through draft+verify: actual time based on depth
    - spec_k learns the ratio between drafted and predicted time
    - High acceptance for code (0.9)
    """
    kind = "engram_timing"
    supports_spec = True
    max_spec_rows = 8
    max_rows = 8

    def gpu_keepalive(self) -> None:
        pass

    def __init__(self, plain_ms=22.0, draft_ms_per_depth=None, accept_probs=None, seed=0):
        self.rng = random.Random(seed)
        self.plain_ms = plain_ms
        self.draft_ms_per_depth = draft_ms_per_depth or {1: 34.8, 2: 44.3, 3: 53.8, 4: 63.3, 5: 72.8}
        self.accept_probs = accept_probs or (0.9, 0.9, 0.9, 0.9, 0.9)
        self.gated = object()  # Triggers launch_gated for plain steps
        self.gated_launches = 0
        self._plain_start = 0
        self._draft_start = 0
        self.last_depth = 0

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
        """Plain step with Engram gate timing."""
        self.gated_launches += 1
        self._plain_start = time.perf_counter()
        lg = self.step(bc, toks)
        y = sample(lg)
        mx.async_eval(y)
        return y

    def step(self, bc, tokens):
        time.sleep(self.plain_ms / 1000.0)                               # the plain step cost the policies measure (plain_ms)
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt(c["h"]) for c in bc])

    def step_feed(self, bc, tokens, drafts, p0s):
        self._plain_start = time.perf_counter()
        return self.step(bc, tokens)

    def draft_many(self, jobs):
        out, confs = [], []
        for d, pending, p0, depth, history in jobs:
            self.last_depth = depth
            self._draft_start = time.perf_counter()
            ids, oo = history
            assert oo[-1] == pending and p0 == len(ids) + len(oo) - 1
            h, dr, cf = list(ids) + list(oo), [], []
            for i in range(depth):
                t = nxt(h)
                accept = self.rng.random() < self.accept_probs[min(i, len(self.accept_probs) - 1)]
                c = self.rng.uniform(0.8, 1.0) if accept else self.rng.uniform(0.0, 0.3)
                dr.append(t if accept else (t + 1 + self.rng.randrange(V - 1)) % V)
                cf.append(c)
                h = h + [dr[-1]]
            out.append(dr)
            confs.append(cf)
        self._confs = confs
        return out

    def draft_confs(self):
        return self._confs

    def spec_verify_bc(self, bc, token_lists):
        """Drafted round verify with simulated timing."""
        target_ms = self.draft_ms_per_depth.get(self.last_depth, 50.0)
        elapsed = (time.perf_counter() - self._draft_start) * 1000
        if elapsed < target_ms:
            time.sleep((target_ms - elapsed) / 1000.0)

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


def run_engram(reqs, backend, mtp_depth=5, probe_every=16, cap_probe=4, **kw):
    eng = Engine(backend, sessions=1, mtp_depth=mtp_depth, ctx=4096, chunk=4, block=8, prefix_bytes=0,
                 probe_every=probe_every, cap_probe=cap_probe, **kw)
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


def test_engram_gate_timing_mismatch_fixed(monkeypatch):
    """With Engram gate, plain_ms=22ms (gated), drafted rounds at depth 3 ~54ms.
    Old dspol cost model used S=1.0, ignoring measured plain_ms.
    Fixed cost model uses plain_ms and spec_k, correctly picking depth 3 for code.
    """
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                          # the dspol cost model, not the solo rate rule
    
    # Timings matching mac_results.log: plain 22ms, depth 3 ~54ms (63 tok/s)
    be = EngramGateTimingFake(plain_ms=22.0, 
                              draft_ms_per_depth={1: 34.8, 2: 44.3, 3: 53.8, 4: 63.3, 5: 72.8},
                              accept_probs=(0.9, 0.9, 0.9, 0.9, 0.9))
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_engram(reqs, be)
    
    # Verify output matches plain greedy
    want_be = EngramGateTimingFake(plain_ms=22.0, accept_probs=(0.9,)*5)
    want = run_engram(reqs, want_be, mtp_depth=0)[0]
    assert got["code"][0] == want["code"][0], "Output must match plain greedy"
    
    # Policy should use depth 3 (optimal for 0.9 acceptance with these timings)
    print(f"Stats: d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)} d4={st.get('spec_d4',0)} d5={st.get('spec_d5',0)}")
    print(f"plain_ms={st.get('plain_ms_x100',0)/100:.1f} spec_k={st.get('spec_k_x100',0)/100:.2f}")
    print(f"spec_accepted={st.get('spec_accepted',0)} spec_drafted={st.get('spec_drafted',0)}")
    
    # Should NOT stay at D=0 (plain)
    assert st.get("spec_d0", 0) < st.get("spec_rounds", 1) * 0.3, "Policy should not stay at D=0"
    # Should use depth 3 significantly
    assert st.get("spec_d3", 0) > st.get("spec_rounds", 1) * 0.3, "Policy should prefer depth 3 for code"
    
    # spec_k should be close to 1.0 (timings match prediction)
    spec_k = st.get('spec_k_x100', 100) / 100.0
    assert 0.8 < spec_k < 1.3, f"spec_k should be near 1.0, got {spec_k}"


def test_engram_gate_with_slower_drafted_rounds(monkeypatch):
    """If drafted rounds are much slower than predicted (spec_k >> 1), 
    the cost model should adapt and prefer plain (D=0).
    """
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    monkeypatch.setenv("KYOJIN_SPEC_RATE", "0")                          # the dspol cost model, not the solo rate rule
    
    # Drafted rounds much slower than predicted
    be = EngramGateTimingFake(plain_ms=22.0,
                              draft_ms_per_depth={1: 45.0, 2: 58.0, 3: 70.0, 4: 82.0, 5: 94.0},
                              accept_probs=(0.5, 0.5, 0.5, 0.5, 0.5))  # Lower acceptance -> more plain rounds
    reqs = {"prose": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_engram(reqs, be, probe_every=4)  # More frequent probes
    
    spec_k = st.get('spec_k_x100', 100) / 100.0
    plain_ms = st.get('plain_ms_x100', 0) / 100.0
    print(f"spec_k={spec_k:.2f}, plain_ms={plain_ms:.1f}, d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)}")
    
    # With very slow drafted rounds, policy should prefer plain (D=0); the drafted rounds run slower than predicted
    assert spec_k > 1.0, f"spec_k should be above 1 for slow drafted rounds, got {spec_k}"
    # Policy should mostly use D=0
    assert st.get("spec_d0", 0) > st.get("spec_rounds", 1) * 0.8, "Policy should prefer plain when drafted rounds are very slow"


if __name__ == "__main__":
    import os
    os.environ[DSPOL.ENV_ENABLE] = "1"
    be = EngramGateTimingFake(plain_ms=22.0, 
                              draft_ms_per_depth={1: 34.8, 2: 44.3, 3: 53.8, 4: 63.3, 5: 72.8},
                              accept_probs=(0.9, 0.9, 0.9, 0.9, 0.9))
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_engram(reqs, be)
    print(f"Stats: d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)} d4={st.get('spec_d4',0)} d5={st.get('spec_d5',0)}")
    print(f"plain_ms={st.get('plain_ms_x100',0)/100:.1f} spec_k={st.get('spec_k_x100',0)/100:.2f}")