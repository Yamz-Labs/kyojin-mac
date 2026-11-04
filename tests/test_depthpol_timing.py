# SPDX-License-Identifier: Apache-2.0
"""Test depth policy with injected timings simulating Engram gate:
plain steps ~22ms, drafted rounds at depth 3 ~54ms (3.44 tokens -> 63 tok/s).
The policy should pick depth 3 for code-like acceptance (0.9), not plain.
"""
import time
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kyojin_mlx.serve.engine import Engine
from kyojin_mlx.serve.sampling import Sampling
from kyojin_mlx import dspol as DSPOL
from test_serve_engine import V, nxt, onehot, collect


class TimedFakeSpec:
    """Fake backend with controllable timings to simulate Engram gate:
    - Plain steps (D=0) take `plain_ms` milliseconds
    - Drafted rounds take `draft_ms[d]` milliseconds for depth d
    - Acceptance rate per position controlled by `accept_probs`
    """
    kind = "timedfakespec"
    supports_spec = True
    max_spec_rows = 8
    max_rows = 8

    def __init__(self, plain_ms=22.0, draft_ms=None, accept_probs=None, seed=0):
        import random
        self.rng = random.Random(seed)
        self.plain_ms = plain_ms
        self.draft_ms = draft_ms or {1: 30.0, 2: 42.0, 3: 54.0, 4: 66.0, 5: 78.0}
        self.accept_probs = accept_probs or (0.9, 0.85, 0.8, 0.75, 0.7)
        self.gated = object()  # triggers launch_gated path for plain steps
        self.gated_launches = 0
        self.draft_calls = 0
        self.verify_calls = 0
        self.last_depth = 0
        self._plain_start = 0
        self._draft_start = 0

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
        """Simulate gated plain step timing."""
        self.gated_launches += 1
        self._plain_start = time.perf_counter()
        time.sleep(self.plain_ms / 1000.0)                           # the plain step's time (untimed, a plain round cost ~0 ms and plain always won)
        # Run the actual step
        lg = self.step(bc, toks)
        y = sample(lg)
        mx.async_eval(y)
        return y

    def step(self, bc, tokens):
        for c, t in zip(bc, tokens.tolist()):
            c["h"].append(int(t))
        return onehot([nxt(c["h"]) for c in bc])

    def step_feed(self, bc, tokens, drafts, p0s):
        """Plain step with draft feeding (for speculative path)."""
        self._plain_start = time.perf_counter()
        time.sleep(self.plain_ms / 1000.0)
        return self.step(bc, tokens)

    def draft_many(self, jobs):
        """Generate drafts with simulated timing and acceptance."""
        self.draft_calls += 1
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
                # High confidence when accepted, low when rejected
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
        """Simulate verify timing based on depth."""
        self.verify_calls += 1
        # Wait to simulate the drafted round time
        target_ms = self.draft_ms.get(self.last_depth, 50.0)
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


def run_timed(reqs, backend, mtp_depth=5, **kw):
    """Run requests and return results, engine status, backend."""
    eng = Engine(backend, sessions=1, mtp_depth=mtp_depth, ctx=4096, chunk=4, block=8, prefix_bytes=0, **kw)
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


def test_depth_policy_picks_mtp_when_it_wins(monkeypatch):
    """With Engram-like timings (plain 22ms, depth3 54ms) and code-like acceptance (0.9),
    the policy should pick depth 3, not plain (depth 0)."""
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    # Plain 22ms -> 45 tok/s. Depth 3: 54ms for ~3.44 tokens -> 63 tok/s.
    be = TimedFakeSpec(plain_ms=22.0, draft_ms={1: 30.0, 2: 42.0, 3: 54.0, 4: 66.0},
                       accept_probs=(0.9, 0.9, 0.9, 0.9, 0.9))
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_timed(reqs, be, probe_every=16, cap_probe=4)

    # Check tokens match plain greedy
    want = run_timed(reqs, TimedFakeSpec(plain_ms=22.0, accept_probs=(0.9,)*5), mtp_depth=0)[0]
    assert got["code"][0] == want["code"][0], "Output must match plain greedy"

    # The policy should have used depth > 0 for most rounds
    depths_used = [d for d in range(6) if st.get(f"spec_d{d}", 0) > 0]
    print(f"Depths used: {depths_used}")
    print(f"Stats: spec_d0={st.get('spec_d0',0)}, spec_d1={st.get('spec_d1',0)}, spec_d2={st.get('spec_d2',0)}, spec_d3={st.get('spec_d3',0)}, spec_d4={st.get('spec_d4',0)}, spec_d5={st.get('spec_d5',0)}")
    print(f"spec_accepted={st.get('spec_accepted',0)}, spec_drafted={st.get('spec_drafted',0)}")

    # With 0.9 acceptance, depth 3 should be optimal. The policy should not stay at D=0.
    assert st.get("spec_d0", 0) < st.get("spec_rounds", 1) * 0.5, "Policy should not stay at D=0 most of the time"
    assert any(st.get(f"spec_d{d}", 0) > 0 for d in [2, 3, 4, 5]), "Policy should use deeper depths for code"


def test_depth_policy_with_dspol_cost_model(monkeypatch):
    """Test that dspol cost model uses correct units when spec_costs is None."""
    monkeypatch.setenv(DSPOL.ENV_ENABLE, "1")
    # Use explicit dspol costs matching the mac_results.log defaults
    monkeypatch.setenv(DSPOL.ENV_COSTS, '{"S_ms": 22.0, "slope_ms": 9.5, "draft_ms": 3.3}')

    be = TimedFakeSpec(plain_ms=22.0, draft_ms={1: 34.8, 2: 44.3, 3: 53.8, 4: 63.3},
                       accept_probs=(0.9, 0.9, 0.9, 0.9, 0.9))
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_timed(reqs, be, probe_every=16, cap_probe=4)

    print(f"Stats with explicit costs: spec_d0={st.get('spec_d0',0)}, spec_d3={st.get('spec_d3',0)}")
    assert any(st.get(f"spec_d{d}", 0) > 0 for d in [2, 3, 4, 5]), "Policy should use deeper depths with correct cost model"


if __name__ == "__main__":
    import sys
    import os
    os.environ[DSPOL.ENV_ENABLE] = "1"
    # Run without pytest to see output
    be = TimedFakeSpec(plain_ms=22.0, draft_ms={1: 30.0, 2: 42.0, 3: 54.0, 4: 66.0},
                       accept_probs=(0.9, 0.9, 0.9, 0.9, 0.9))
    reqs = {"code": ([1, 2, 3, 4, 5], dict(max_new=100))}
    got, st, be = run_timed(reqs, be, probe_every=16, cap_probe=4)
    print(f"Depths: d0={st.get('spec_d0',0)} d1={st.get('spec_d1',0)} d2={st.get('spec_d2',0)} d3={st.get('spec_d3',0)} d4={st.get('spec_d4',0)} d5={st.get('spec_d5',0)}")
    print(f"Accepted: {st.get('spec_accepted',0)}/{st.get('spec_drafted',0)}")