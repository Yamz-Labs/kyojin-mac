"""Ultra estimate (solo 70 tok/s) of served tok/s with grouped routed experts. Same bandwidth model as tools/spec_est.py ('bytes': dense 4.8 GiB read once per step,
routed 2.5 GiB per 6-expert row, a routed expert read once per DISTINCT expert), draft call = 0.15 solo step per session. Routing: users independent over 384 experts;
inside a user each next row copies a fraction f of the previous row's experts (verify chain). eta = share of the ideal saving (pairs - distinct) the kernel realizes
(layer bench ov384, Air: 0.6-0.9). eta = 0 is today's per-pair path."""
import itertools
E, TOP = 384, 6
def distinct_rows(N, m, f):
    per_user = 1 + (m - 1) * (1 - f)                       # distinct 6-expert row equivalents of one user over m consecutive rows
    tot = N * per_user
    return E / TOP * (1 - (1 - TOP / E) ** (tot * TOP / TOP * 1.0)) if False else tot
def r(N, m, f, eta):
    n = N * m
    d = distinct_rows(N, m, f)
    # users overlap each other by chance: expected collisions among N*d distinct-ish experts out of 384 (small, ignored for N*m*6 << 384*...) -> include
    pairs = n * TOP
    dist_exp = d * TOP
    dist_exp = E * (1 - (1 - dist_exp / E) ** 1)             # no extra collision term: keep the model simple
    eff = pairs - eta * (pairs - dist_exp)
    return (4.8 + 2.5 * eff / TOP) / 7.3
def agg(N, a, f, eta, solo=70.0):
    best = (0, 0)
    for D in range(0, 5):
        tok = sum(a ** i for i in range(D + 1))
        t = r(N, D + 1, f, eta) + (0.15 * N if D else 0)
        best = max(best, (N * tok / t * solo, D))
    return best
print("aggregate tok/s (per session) with the best D, solo 70; columns eta 0 (today) / 0.7 grouped")
for a in (0.735, 0.85):
    for f in (0.0, 0.3, 0.5, 0.7):
        row = []
        for N in (1, 2, 4):
            x0, D0 = agg(N, a, f, 0.0); x1, D1 = agg(N, a, f, 0.7)
            row.append(f"N={N}: {x0:5.0f} (D{D0}) -> {x1:5.0f} (D{D1}) [{x1 / N:4.0f}/sess]")
        print(f"a={a} overlap f={f}: " + " | ".join(row))
print("plain batched decode (D=0) is 70 / 104 / 138 aggregate in this model:", [round(N * 70 / r(N, 1, 0, 0.0)) for N in (1, 2, 4)])
