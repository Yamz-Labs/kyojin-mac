"""GLM-5.3 speculative round cost model and the per-round draft planner (flags glm_plan, glm_chain).

Measured on the full model (M5 Ultra, served, glmnextn_ab 15/11, 1 session): a verify pass of R rows costs V(R) ms of GPU wait (V(1) = the plain step) and a round adds the
draft section and the commit. The row is expensive on this model (every row of a pass adds its own routed experts to the union and runs the recurrent chunk path):
about 12 ms per row from 2 to 8 rows, 17 beyond, against 31 ms for the whole plain step. So a draft row pays only if it is accepted with probability above
(row ms) / (ms per token of the round it joins) ~ 0.5: depth 1 at 90 % acceptance clears it with room, a second row (second-position acceptance ~0.7-0.9) does not
except on the most predictable text, and 2+ sessions share a pass whose rows already cost more than a plain step per session.

plan_depths picks, per session, the number of MTP drafts d_i in 0..cap that maximises the pass's expected tokens per ms: sum_i T_i(d_i) / C(N, rows, drafts), with
T_i(d) = 1 + sum_{j<=d} p_i1 ... p_ij from the session's own measured per-position acceptance. Exhaustive for few sessions, hill climbing beyond.
"""
from __future__ import annotations

import itertools
import json
import os

# GPU wait of one verify pass by rows (ms), full model, measured; plain step at 1 row
VERIFY_MS = ((1, 31.1), (2, 38.5), (3, 52.6), (4, 62.2), (8, 110.8), (16, 247.1))
POST_MS = ((1, 0.1), (2, 2.0), (3, 2.8), (4, 3.7), (8, 5.4), (16, 9.0))      # commit + hidden hand-off + sampling readback
DRAFT_FIXED_MS, DRAFT_FIRST_MS, DRAFT_STEP_MS = 1.0, 3.0, 2.9                # draft section = fixed + sessions * (first head step + 2.9 per chained step)
PLAIN_BATCH = 0.92                                                           # a plain step of N sessions vs the N-row verify pass (N4: 57 ms vs 62 ms)
PRIOR = (0.85, 0.75, 0.70, 0.65)                                             # per-position acceptance before any round was seen
PRIOR_N = 2.0
PROBE_EVERY = 24                                                             # a session planned at 0 drafts tries 1 every this many rounds (its estimate must move)


def _interp(tab, x: float) -> float:
    if x <= tab[0][0]:
        return tab[0][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = tab[-2], tab[-1]
    return y1 + (y1 - y0) / (x1 - x0) * (x - x1)


class Costs:
    def __init__(self, verify=VERIFY_MS, post=POST_MS):
        self.verify_tab, self.post_tab = tuple(verify), tuple(post)

    @staticmethod
    def from_env() -> "Costs":
        """KYOJIN_GLM_COSTS = path of a json {"verify": [[rows, ms], ...], "post": [...]} (a re-measured table)."""
        p = os.environ.get("KYOJIN_GLM_COSTS", "")
        if p:
            d = json.load(open(os.path.expanduser(p)))
            return Costs(d.get("verify", VERIFY_MS), d.get("post", POST_MS))
        return Costs()

    def verify(self, rows: int) -> float:
        return _interp(self.verify_tab, rows)

    def post(self, rows: int) -> float:
        return _interp(self.post_tab, rows)

    def plain(self, n: int) -> float:
        """One plain step of n sessions."""
        return self.verify(1) if n <= 1 else PLAIN_BATCH * self.verify(n)

    def draft(self, depths: list) -> float:
        """Draft section of a round: the head runs once per session that drafts, one more step per extra depth."""
        act = [d for d in depths if d > 0]
        if not act:
            return 0.0
        return DRAFT_FIXED_MS + sum(DRAFT_FIRST_MS + DRAFT_STEP_MS * (d - 1) for d in act)

    def round_ms(self, depths: list, lookup_rows: int = 0) -> float:
        """One pass of len(depths) sessions, session i with depths[i] drafts (lookup_rows extra rows from n-gram drafts: no draft cost)."""
        n = len(depths)
        rows = n + sum(depths) + lookup_rows
        if rows == n:
            return self.plain(n)
        return self.verify(rows) + self.post(rows) + self.draft(depths)

    def lookup_round_ms(self, k: int, n: int = 1) -> float:
        """A round where one session drafts k tokens from the n-gram index (n sessions in the pass)."""
        return self.round_ms([0] * n, lookup_rows=k)

    def mtp_round_ms(self, width: int, n: int = 1) -> float:
        return self.round_ms([width] + [0] * (n - 1))


class Acc:
    """Per-position conditional acceptance of the MTP head from the engine's decayed counters (Session.pos_succ / pos_trial)."""

    @staticmethod
    def p(pos_succ, pos_trial, j: int) -> float:
        pr = PRIOR[min(j, len(PRIOR) - 1)]
        if j < len(pos_trial):
            return (pos_succ[j] + pr * PRIOR_N) / (pos_trial[j] + PRIOR_N)
        return pr


def tokens(ps: list, d: int) -> float:
    """Expected tokens of a session that drafts d tokens: 1 + p1 + p1 p2 + ..."""
    t, c = 1.0, 1.0
    for j in range(d):
        c *= ps[j]
        t += c
    return t


def plan_depths(ps_list: list, cap: int, costs: Costs, max_rows: int = 0, margin: float = 0.02) -> list:
    """ps_list[i] = per-position acceptance of session i (list of >= cap floats). Returns the drafts per session (0..cap) that maximise tokens per ms of the pass;
    a plan must beat the all-plain pass by `margin` to draft at all."""
    n = len(ps_list)
    cap = max(0, cap)
    if n == 0 or cap == 0:
        return [0] * n

    def rate(ds):
        if max_rows and n + sum(ds) > max_rows:
            return -1.0
        return sum(tokens(ps, d) for ps, d in zip(ps_list, ds)) / costs.round_ms(list(ds))

    base = rate([0] * n)
    if n <= 4:
        best, best_r = [0] * n, base
        for ds in itertools.product(range(cap + 1), repeat=n):
            r = rate(list(ds))
            if r > best_r * (1 + 1e-9):
                best, best_r = list(ds), r
    else:                                            # uniform start, then one row at a time
        best = max(([d] * n for d in range(cap + 1)), key=rate)
        best_r = rate(best)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for dd in (-1, 1):
                    c = list(best)
                    c[i] += dd
                    if 0 <= c[i] <= cap and rate(c) > best_r * (1 + 1e-9):
                        best, best_r, improved = c, rate(c), True
    return best if best_r > base * (1 + margin) else [0] * n


def chain_extra(p1: float, rho: float, depths: list, i: int, costs: Costs, max_ext: int, max_rows: int = 0) -> int:
    """Session i drafted by the head AND the n-gram index proposes a continuation that starts with the head's token (two independent drafters agree).
    How many of the index's following tokens to append as free draft rows: rows are added while the expected extra tokens per extra ms beat the pass's rate.
    p1 = P(the head's token is accepted); rho = P(an index token is accepted given the agreement, the previous one accepted)."""
    ds = list(depths)
    tok0 = 0.0

    def rate(e):
        # expected tokens of the pass: the session's own T plus the extension (reached only when the head's token passes)
        ext, c = 0.0, p1
        for _ in range(e):
            c *= rho
            ext += c
        return (tok0 + ext) / costs.round_ms(ds, lookup_rows=e)
    best, best_r = 0, None
    # a round with e extension rows yields (n + the head's accepted token p1 + ext(e)) tokens in total over the pass; the other sessions' tokens are the same in every arm
    tok0 = float(len(ds)) + p1
    for e in range(0, max_ext + 1):
        if max_rows and len(ds) + sum(ds) + e > max_rows:
            break
        r = rate(e)
        if best_r is None or r > best_r * (1 + 1e-9):
            best, best_r = e, r
    return best
