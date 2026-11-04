"""cmudec: discrete simulation of the N-user prefill/decode scheduler on the numbers MEASURED on the M5 Ultra (DS4.1, served lane, N=4, 33K prompts).
Nothing here is a timing of the engine: it is the scheduler rules (fair, pf_srpt, pf_floor) run on a cost model, no GPU.

Measured inputs: solo prefill 2433 tok/s; fair aggregate prefill 2424; solo decode 53 tok/s; N=4 decode 80-84 tok/s aggregate; decode round ms by rows 4:55 6:63 8:82 10:104 12:120.
Fitted from the measured fair wall (61.4 s): tokens per user T. Fitted from the measured pf_srpt wall (69.7 s): OVH, the extra cost of a decode round run INSIDE a
prefill chunk (layer-bound eval + prefill/decode switch). OVH is the one free parameter and the pf_floor prediction rests on the claim that a burst pays it once."""
from dataclasses import dataclass

L, N, CHUNK = 33000, 4, 8192
P_SOLO, P_FAIR = 2433.0, 2424.0
G = 1.15                                  # tokens per user per round: 4 users, 55 ms, ~83 tok/s aggregate
SLICE_S = 0.250                           # mix_slice_ms on the lane
RA, RB = 0.0106, 0.0111                   # round seconds = RA + RB * users: solo 21.7 ms (53 tok/s * 1.15), 4 users 55 ms


def round_s(k):
    return RA + RB * k


@dataclass
class Policy:
    name: str
    order: str = "fair"                   # fair | srpt
    slice_s: float = 0.0                  # >0: one round inside the chunk every slice_s of wall (mix_slice_ms)
    floor: float = 0.0                    # pf_floor tok/s per decoder
    cap: float = 0.60                     # pf_floor_max
    period_s: float = 1.0                 # pf_floor_period_ms


def simulate(pol: Policy, T: float, ovh: float, n: int = N, plen: int = L):
    chunks = [[min(CHUNK, plen - i) for i in range(0, plen, CHUNK)] for _ in range(n)]
    left, idx, last_run = [plen] * n, [0] * n, [-1.0] * n
    P = P_FAIR if pol.order == "fair" else P_SOLO          # fair switches cache every chunk: measured 2424 vs 2433
    t = 0.0
    ttft, gen = [None] * n, [0.0] * n
    track = {}                                             # user -> [tokens, seconds] while another user prefills
    state = dict(debt=0.0, k=1.0)                         # k: the engine's feedback factor (_floor_k)

    def decoders():
        return [u for u in range(n) if ttft[u] is not None and gen[u] < T]

    def prefilling():
        return any(x is None for x in ttft)

    def duty():
        d = decoders()
        return min(pol.cap, pol.floor * round_s(len(d)) / G * state["k"]) if d else 0.0

    def do_round(extra=0.0):
        nonlocal t
        d = decoders()
        if not d:
            return 0.0
        dt = round_s(len(d)) + extra
        t += dt
        for u in d:
            g0 = gen[u]
            gen[u] = min(T, g0 + G)
            if prefilling():
                tr = track.setdefault(u, [0.0, 0.0])
                tr[0] += gen[u] - g0
                tr[1] += dt
        return dt

    while prefilling():
        cand = [u for u in range(n) if ttft[u] is None]
        u = min(cand, key=lambda x: last_run[x]) if pol.order == "fair" else min(cand, key=lambda x: left[x])
        last_run[u] = t
        sz = chunks[u][idx[u]]
        need = sz / P
        done, inline, last_round = 0.0, 0.0, t
        t_chunk0, gen0, d0 = t, list(gen), decoders()
        while done < need - 1e-9:
            d = decoders()
            step = need - done
            if d and pol.floor:
                dy = duty()
                r = dy / (1 - dy)
                owed = done * r - inline + min(state["debt"], 0.0)
                step = min(step, max((max(pol.period_s * r, round_s(len(d))) - owed) / r, 1e-6))
            elif d and pol.slice_s:
                step = min(step, max(SLICE_S - (t - last_round), 1e-6))
            t += step
            done += step
            for v in d:
                track.setdefault(v, [0.0, 0.0])[1] += step
            if done >= need - 1e-9:
                break
            d = decoders()
            if d and pol.floor:
                dy = duty()
                r = dy / (1 - dy)
                owed = done * r - inline + min(state["debt"], 0.0)
                if owed >= max(pol.period_s * r, round_s(len(d))) - 1e-9:
                    spent, first = 0.0, True               # one burst: the switch is paid once
                    while decoders() and spent < owed:
                        spent += do_round(ovh if first else 0.0)
                        first = False
                    inline += spent
            elif d and pol.slice_s and t - last_round >= SLICE_S - 1e-6:
                last_round = t
                inline += do_round(ovh)                    # every slice pays the switch
        if pol.floor and d0 and (t - t_chunk0) > 0:         # feedback: the floor is a rate, so the duty is scaled by target / achieved (as the engine does)
            ach = sum(gen[v] - gen0[v] for v in d0) / len(d0) / (t - t_chunk0)
            if ach > 0:
                state["k"] = min(max(state["k"] * min(max(pol.floor / ach, 0.85), 1.2), 1.0), 3.0)
        left[u] -= sz
        idx[u] += 1
        if left[u] <= 0:
            ttft[u] = t
        if pol.floor and decoders():
            dy = duty()
            state["debt"] = need * dy / (1 - dy) - inline + min(state["debt"], 0.0)
            while decoders() and state["debt"] > 0 and prefilling():
                state["debt"] -= do_round()
        elif decoders() and prefilling():
            do_round()                                     # decode_per_chunk = 1
    t_pf_end = t
    while decoders():
        do_round()
    rates = [tr[0] / tr[1] for _, tr in sorted(track.items()) if tr[1] > 0]
    return dict(ttft=[round(x, 1) for x in ttft], mean_ttft=sum(ttft) / n, wall=t, agg_prefill=n * plen / t_pf_end,
                rates=[round(x, 1) for x in rates], floor_min=min(rates) if rates else 0.0)


def fit(fair_wall=61.4, srpt_wall=69.7):
    lo, hi = 20.0, 400.0
    for _ in range(50):
        T = (lo + hi) / 2
        lo, hi = (lo, T) if simulate(Policy("f"), T, 0.0)["wall"] > fair_wall else (T, hi)
    T = (lo + hi) / 2
    lo, hi = 0.0, 2.0
    for _ in range(50):
        o = (lo + hi) / 2
        lo, hi = (o, hi) if simulate(Policy("s", "srpt", slice_s=SLICE_S), T, o)["wall"] < srpt_wall else (lo, o)
    return T, (lo + hi) / 2


POLICIES = [Policy("off", "fair"), Policy("pf_srpt", "srpt", slice_s=SLICE_S), Policy("pf_floor=15", "srpt", floor=15),
            Policy("pf_floor=15 max35", "srpt", floor=15, cap=0.35), Policy("pf_floor=8", "srpt", floor=8)]

if __name__ == "__main__":
    T, o = fit()
    print("SIMULATION (no GPU). fitted T=%.0f tokens/user, OVH=%.0f ms per in-chunk round" % (T, o * 1e3))
    for p in POLICIES:
        r = simulate(p, T, o)
        print("%-18s ttft=%s mean=%.1f wall=%.1f aggpf=%.0f floor_min=%.1f rates=%s" % (p.name, r["ttft"], r["mean_ttft"], r["wall"], r["agg_prefill"], r["floor_min"], r["rates"]))
