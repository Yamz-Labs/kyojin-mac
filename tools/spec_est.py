"""Speed estimate of speculative decode on the Ultra from the Air row costs and the draft cost.
usage: spec_est.py [--cost cost_a.json cost_b.json ...] [--solo 40,55,70] [--draft 0.23]
Round = one draft call + one chained verify of D + 1 rows; it emits 1 + a + a^2 + ... + a^D tokens on average (a = per-position acceptance, flat across the block
like the DSpark head: see PR-dspark.md, 0.95-1.0 on easy text, 0.73 measured on the box in September, 0.59 prose / 0.75 code reported for oMLX's MTP).
Row cost r(n) = time of an n-row verify / time of a solo step, two models:
  air    the medians measured on the Air (tools/spec_cost.py): work-proportional, what ALU-bound decode pays
  bytes  bandwidth-bound Ultra: dense weights are read once for all rows, routed experts once per DISTINCT expert (random routing, 384 experts, top 6);
         per token 4.8 GiB dense + 2.5 GiB routed (ULTRA.md section 2)
Draft cost d is a fraction of a solo step, counted in bytes from the real mtp.0-2 headers (ULTRA.md section 11): the 3 stages' non-routed weights 0.34 GiB
+ their routed experts (14.3 distinct of 128 for 5 rows x top 3, 2.115 GiB per stage in the file) 0.71 GiB + the trunk head over 5 rows 0.62 GiB = 1.67 GiB
of a 7.3 GiB step = 0.23. The Air gives the same: 3 stages on 5 rows = 2.8 slice steps = 0.14 of a 40-layer step, plus the head (0.08).
Speed-up = tokens per round / (r(D + 1) + d)."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--cost", nargs="*", default=[]); ap.add_argument("--solo", default="40,55,70"); ap.add_argument("--draft", type=float, default=0.23)
a = ap.parse_args()
air = [1.0, 1.43, 1.78, 2.28, 3.0]
if a.cost:
    cols = []
    for f in a.cost:
        ms = json.load(open(f))["ms"]
        cols.append([ms[f"verify{n}"] / ms["verify1"] for n in range(1, 6)])
    air = [sum(c[i] for c in cols) / len(cols) for i in range(5)]
E, K, TOP = 384, 384, 6
byt = [(4.8 + 2.5 * (E * (1 - (1 - TOP / E) ** n) / TOP)) / 7.3 for n in range(1, 6)]
print("row cost r(n), n = 1..5 rows")
print("  air  ", " ".join(f"{x:5.2f}" for x in air)); print("  bytes", " ".join(f"{x:5.2f}" for x in byt))
for name, r in (("air", air), ("bytes", byt)):
    print(f"\nmodel {name}: speed-up over solo decode, draft cost {a.draft:.2f} step")
    print("  accept/pos   D=1    D=2    D=3    D=4   best")
    for al in (0.5, 0.6, 0.735, 0.85, 0.95):
        sp = [sum(al ** i for i in range(D + 1)) / (r[D] + a.draft) for D in (1, 2, 3, 4)]
        b = max(range(4), key=lambda i: sp[i])
        print(f"  {al:9.3f} " + " ".join(f"{s:6.2f}" for s in sp) + f"   D={b + 1} x{sp[b]:.2f}")
print("\nUltra tok/s with the best D (solo estimate in ULTRA.md: 40-70):")
for name, r in (("air", air), ("bytes", byt)):
    for al in (0.6, 0.735, 0.85):
        g = max(sum(al ** i for i in range(D + 1)) / (r[D] + a.draft) for D in (1, 2, 3, 4))
        print(f"  {name:5s} a={al}: " + "  ".join(f"{s} -> {s * g:.0f}" for s in [int(x) for x in a.solo.split(",")]))
