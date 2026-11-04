"""Offline width-policy simulation (no GPU, no model): predicted tok/s of fixed depth 3 vs the dspol adaptive
width policy for prose-like and code-like acceptance, with the cost model cost(R) = S + slope*(R-1) + draft.
Acceptance priors (see kyojin_mlx/dspol.py): prose p ~ 0.68 and code p ~ 0.97 per position, fitted from the served
fixed-d3 numbers (prose 45, code 71 tok/s, mac_results.log); bimodal sensitivity from Sushi (26% of blocks accept
0 of 5 drafts, 24% accept all 5, rivals_REPORT.md:103). usage: dspol_sim.py [--costs JSON-or-file] [--slope X]"""
import argparse
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from kyojin_mlx import dspol as DSPOL

ap = argparse.ArgumentParser()
ap.add_argument("--costs", default="")
ap.add_argument("--s", type=float, default=None)
ap.add_argument("--slope", type=float, default=None)
a = ap.parse_args()
import os
if a.costs:
    os.environ[DSPOL.ENV_COSTS] = a.costs
if a.s is not None:
    os.environ[DSPOL.ENV_S] = str(a.s)
if a.slope is not None:
    os.environ[DSPOL.ENV_SLOPE] = str(a.slope)
cost = DSPOL.CostModel.load()
print(f"cost model: S={cost.s_ms:.1f} ms slope={cost.slope_ms:.1f} ms draft={cost.draft_ms:.1f} ms "
      f"(cost(R) = S + slope*(R-1) + draft, R = verify rows)", flush=True)


def show(name, probs, cap=5, fixed=3):
    ef = DSPOL.expected_tokens(probs, fixed) / (cost.cost_ms(fixed) / 1e3)
    d = DSPOL.choose_width(probs, cost, cap)
    ep = DSPOL.expected_tokens(probs, d) / (cost.cost_ms(d) / 1e3)
    print(f"{name:34s} E[T|d3]={DSPOL.expected_tokens(probs, fixed):.2f} fixed-d3 {ef:5.1f} tok/s"
          f"  -> policy D={d} E[T]={DSPOL.expected_tokens(probs, d):.2f} {ep:5.1f} tok/s"
          f" ({(ep / ef - 1) * 100:+.1f} %)", flush=True)


print("--- geometric per-position acceptance, fitted from served fixed-d3 (prose 45, code 71 tok/s) ---", flush=True)
show("prose p=0.68 (fixed-d3 44.8 ~= 45)", [0.68] * 5)
show("code p=0.97 (fixed-d3 71.0 ~= 71)", [0.97] * 5)
print("--- bimodal (Sushi: 26% accept 0/5, 24% accept 5/5, rest geometric p=0.6): policy sees the mode ---", flush=True)
for name, probs in (("miss-mode rounds (26%)", [0.02] * 5),
                    ("hit-mode rounds (24%)", [0.99] * 5),
                    ("middle rounds (50%, p=0.6)", [0.6] * 5)):
    show(name, probs)
mix = 0.26 * DSPOL.expected_tokens([0.02] * 5, DSPOL.choose_width([0.02] * 5, cost, 5)) / (cost.cost_ms(DSPOL.choose_width([0.02] * 5, cost, 5)) / 1e3) \
    + 0.24 * DSPOL.expected_tokens([0.99] * 5, DSPOL.choose_width([0.99] * 5, cost, 5)) / (cost.cost_ms(DSPOL.choose_width([0.99] * 5, cost, 5)) / 1e3) \
    + 0.50 * DSPOL.expected_tokens([0.6] * 5, DSPOL.choose_width([0.6] * 5, cost, 5)) / (cost.cost_ms(DSPOL.choose_width([0.6] * 5, cost, 5)) / 1e3)
fix = 0.26 * DSPOL.expected_tokens([0.02] * 5, 3) + 0.24 * DSPOL.expected_tokens([0.99] * 5, 3) + 0.50 * DSPOL.expected_tokens([0.6] * 5, 3)
fix = fix / (cost.cost_ms(3) / 1e3)
print(f"bimodal mixture: fixed-d3 {fix:.1f} tok/s vs mode-aware policy {mix:.1f} tok/s ({(mix / fix - 1) * 100:+.1f} %)", flush=True)
