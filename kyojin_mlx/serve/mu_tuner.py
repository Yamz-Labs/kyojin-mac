# SPDX-License-Identifier: Apache-2.0
"""Self-tuning round plan for several sessions that share one batched verify pass.

The arms are 0 = the batched plain step (no drafts) and D = 1..dmax drafts per session in a speculative round. The best one depends on the model, the context, the
chip and the number of sessions N, so nothing is fixed. Two things are measured while serving:
  - the cost of a round per (N, arm): the median wall time of the ticks of a window of W ticks, smoothed; a cost is stable (hardware), so each arm is measured in
    turn once and the neighbourhood of the best again every `hold` ticks;
  - the acceptance of each draft position, pooled over the sessions at hand (they move with the text: the replies loop, the drafts land more often later).
The plan is the arm with the most tokens per second: N (1 + sum_j prod_{i<=j} p_i) / cost, p_i = conditional acceptance of position i (Engine._spec_update).
"""
from __future__ import annotations

import os
import time


class MultiTuner:
    def __init__(self, window: int = 4, skip: int = 2, hold: int = 400, start: int = 2):
        self.window, self.skip, self.hold, self.start = window, skip, hold, start
        self.cost: dict[int, dict[int, float]] = {}         # N -> arm -> seconds per tick (smoothed median of windows)
        self.arm: int | None = None
        self.n = 0
        self.dmax = 4
        self.key = None
        self.ticks = 0
        self.last = None
        self.obs: list = []
        self.held = 0
        self.windows = 0
        self.probs: list = []
        self.first: dict[int, int] = {}                     # N -> the arm measured first: its window ran on cold pipelines, it is measured once more

    def expect(self, d: int) -> float:
        """Expected tokens of a session in a round of depth d."""
        tok, run = 1.0, 1.0
        for j in range(d):
            run *= self.probs[j] if j < len(self.probs) else (self.probs[-1] if self.probs else 0.7)
            tok += run
        return tok

    def rate(self, n: int, a: int) -> float:
        return n * self.expect(a) / self.cost[n][a]

    def best(self, n: int, dmax: int) -> int | None:
        c = self.cost.get(n, {})
        ok = [a for a in c if a <= dmax]
        return max(ok, key=lambda a: self.rate(n, a)) if ok else None

    def _choose(self) -> int:
        n, dmax = self.n, self.dmax
        c = self.cost.setdefault(n, {})
        unknown = [a for a in range(dmax + 1) if a not in c]
        if unknown:                                           # measure each arm once, the likely ones first
            return min(unknown, key=lambda a: abs(a - self.start))
        if os.environ.get("KYOJIN_MU_REMEASURE", "1") not in ("0", "") and n in self.first and self.first[n] <= dmax:      # the table is full: replace the cold price
            a = self.first.pop(n)
            c.pop(a, None)
            return a
        best = self.best(n, dmax)
        self.held += 1
        if self.held >= self.hold:                            # the costs drift with the context: measure a neighbour of the best again
            self.held = 0
            nb = [a for a in (best + 1, best - 1) if 0 <= a <= dmax]
            return nb[self.windows % len(nb)]
        return best

    def plan(self, n: int, dmax: int, key, probs: list) -> int:
        """Arm for the tick that starts now (n sessions, member set `key`, probs = pooled conditional acceptance per draft position)."""
        self.dmax, self.probs = dmax, probs
        if n != self.n or key != self.key:
            self.n, self.key, self.ticks, self.last, self.obs = n, key, 0, None, []
            self.arm = None
        if self.arm is None or self.arm > dmax:
            self.arm = self._choose()
        elif self.arm != self.best(n, dmax) and self.obs == [] and self.ticks == 0 and len(self.cost.get(n, {})) > dmax:
            self.arm = self._choose()
        return self.arm

    def done(self, key, tokens: int) -> None:
        now = time.perf_counter()
        if key != self.key:
            self.key, self.ticks, self.last, self.obs = key, 0, now, []
            return
        prev, self.last = self.last, now
        self.ticks += 1
        if prev is None or self.ticks <= self.skip:
            return
        self.obs.append(now - prev)
        if len(self.obs) >= self.window:
            med = sorted(self.obs)[len(self.obs) // 2]
            c = self.cost.setdefault(self.n, {})
            if not c:
                self.first[self.n] = self.arm
            c[self.arm] = med if self.arm not in c else 0.5 * c[self.arm] + 0.5 * med
            self.windows += 1
            self.ticks, self.obs, self.last = 0, [], None
            self.arm = self._choose()
