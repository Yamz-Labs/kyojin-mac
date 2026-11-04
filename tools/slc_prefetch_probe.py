#!/usr/bin/env python3
# Premise check for the "prefetch the next layer's dense weights from inside the ALU-bound routed kernels" idea (cdecaudit report, lever 2).
# Needs Metal. Memory: one eviction buffer of --evict-mib (default 1536 MiB) + the probe buffer (<= 256 MiB) + a small ALU buffer = under 2 GiB.
# NOT run where it was written (no Metal): the kernels below are untested, read the numbers only after the self-check line says OK.
#
# usage: python3 -I tools/slc_prefetch_probe.py [--mib 32,64,128] [--reps 9] [--evict-mib 1536] [--json out.json]
#
# Question: the routed-expert GEMV decode runs at ~527 GB/s of a ~1030 GB/s roof (ALU bound), the dense GEMVs after it at ~636 GB/s. If threadgroups of
# the ALU-bound kernel also touched the NEXT dense weights (read, discard) and the system level cache kept them, the dense kernel would read them
# faster. Three facts decide whether that can work at all, and this script measures each in isolation (no model):
#   A. retention:   time to read P cold (after streaming the eviction buffer) vs warm (just read): the SLC speed-up for a P of that size
#   B. overlap:     an ALU-bound kernel (dependent FMA chain) alone vs the same kernel with extra threads that read P: does the touch ride free
#   C. end to end:  evict; ALU kernel (+touch / no touch); read P. total time with touch vs without = what the idea would collect per layer
# Reading: gain_us = T(no touch) - T(touch) in C. Positive and above the noise = worth building into the MoE kernels; <= 0 = SLC does not hold P
# (or the touch is not free): drop the idea. P of 32-128 MiB is the size of one layer's dense weights (80 MB at DS4.1).
import argparse, json, statistics, sys, time

import mlx.core as mx

SRC_MIX = r"""
    const device float4* P = (const device float4*)p4;
    const uint t = thread_position_in_grid.x;
    if (t < NA) {
        float a = x[t];
        for (uint i = 0; i < IT; ++i) a = fma(a, 1.0000001f, 1e-7f);
        out[t] = a;
    } else if (TOUCH) {
        const uint j = t - NA;
        float4 s = float4(0.0f);
        for (uint c = 0; c < CH; ++c) s += P[ulong(c) * NT + j];
        if (s.x == -1234.5f && s.y == s.z) out[0] = s.w;          // never true: keeps the loads alive
    }
"""
SRC_READ = r"""
    const device float4* P = (const device float4*)p4;
    const uint j = thread_position_in_grid.x;
    float4 s = float4(0.0f);
    for (uint c = 0; c < CH; ++c) s += P[ulong(c) * NT + j];
    if (s.x == -1234.5f && s.y == s.z) out[0] = s.w + dep[0];
    else if (j == 0) out[0] = dep[0];
"""
_k = {}


def kern(name, src, ins):
    if name not in _k:
        _k[name] = mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"], source=src)
    return _k[name]


def read(p4, nt, ch, dep):
    return kern("slc_read", SRC_READ, ["p4", "dep"])(inputs=[p4, dep], template=[("NT", nt), ("CH", ch)], grid=(nt, 1, 1), threadgroup=(256, 1, 1),
                                                     output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0]


def mix(x, p4, na, nt, ch, it, touch):
    return kern("slc_mix", SRC_MIX, ["x", "p4"])(inputs=[x, p4], template=[("NA", na), ("NT", nt), ("CH", ch), ("IT", it), ("TOUCH", touch)],
                                                 grid=(na + (nt if touch else 0), 1, 1), threadgroup=(256, 1, 1), output_shapes=[(na,)],
                                                 output_dtypes=[mx.float32])[0]


def timed(fn, reps):
    ts = []
    for _ in range(reps + 2):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", default="32,64,128"); ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--evict-mib", type=int, default=1536); ap.add_argument("--json")
    a = ap.parse_args()
    if not mx.metal.is_available():
        sys.exit("no Metal")
    nt = 1 << 16
    ev_ch = a.evict_mib * 1024 * 1024 // 16 // nt
    ev = mx.random.normal((ev_ch * nt, 4)); mx.eval(ev)
    dep = mx.zeros((1,)); x = mx.random.normal((1 << 16,)); mx.eval(dep, x)
    res = {}
    for mib in [int(v) for v in a.mib.split(",")]:
        ch = mib * 1024 * 1024 // 16 // nt
        p4 = mx.random.normal((ch * nt, 4)); mx.eval(p4)
        flat_ev = ev.reshape(-1, 4)

        def evict():
            return read(flat_ev, nt, ev_ch, dep)

        # self-check: the touch kernel must not change the ALU result
        r0 = mix(x, p4, x.shape[0], nt, ch, 2000, False); r1 = mix(x, p4, x.shape[0], nt, ch, 2000, True); mx.eval(r0, r1)
        same = bool(mx.array_equal(r0, r1).item())
        # B: ALU alone vs with touch (IT chosen so the ALU kernel runs ~ the time P takes at 500 GB/s)
        it = 2000
        t_alu = timed(lambda: mix(x, p4, x.shape[0], nt, ch, it, False), a.reps)
        t_alu_t = timed(lambda: mix(x, p4, x.shape[0], nt, ch, it, True), a.reps)
        # A: cold vs warm read of P
        def cold():
            mx.eval(evict()); return read(p4, nt, ch, dep)
        def warm():
            mx.eval(read(p4, nt, ch, dep)); return read(p4, nt, ch, dep)
        # (the cold / warm times include the first read for warm; subtract the plain read)
        t_read = timed(lambda: read(p4, nt, ch, dep), a.reps)
        tc, tw = [], []
        for _ in range(a.reps + 2):
            mx.eval(evict()); t0 = time.perf_counter(); mx.eval(read(p4, nt, ch, dep)); tc.append((time.perf_counter() - t0) * 1e6)
            mx.eval(read(p4, nt, ch, dep)); t0 = time.perf_counter(); mx.eval(read(p4, nt, ch, dep)); tw.append((time.perf_counter() - t0) * 1e6)
        t_cold, t_warm = statistics.median(tc[2:]), statistics.median(tw[2:])
        # C: evict; ALU (+touch); read P -> total of the last two
        def c_arm(touch):
            ts = []
            for _ in range(a.reps + 2):
                mx.eval(evict())
                t0 = time.perf_counter(); y = mix(x, p4, x.shape[0], nt, ch, it, touch); mx.eval(read(p4, nt, ch, y[:1])); ts.append((time.perf_counter() - t0) * 1e6)
            return statistics.median(ts[2:])
        c_no, c_yes = c_arm(False), c_arm(True)
        c_no2, c_yes2 = c_arm(False), c_arm(True)                        # second pair: order / thermal check
        gain = (statistics.mean([c_no, c_no2]) - statistics.mean([c_yes, c_yes2]))
        gbps = lambda us: mib * 1.048576e6 / us / 1e3
        res[mib] = dict(self_check_same_alu_result=same, alu_us=t_alu, alu_touch_us=t_alu_t, read_us=t_read, cold_read_us=t_cold, warm_read_us=t_warm,
                        cold_GBps=gbps(t_cold), warm_GBps=gbps(t_warm), C_no_touch_us=[c_no, c_no2], C_touch_us=[c_yes, c_yes2], gain_us=gain)
        print(f"P={mib} MiB self-check {'OK' if same else 'FAIL'} | A cold {t_cold:.0f} us ({gbps(t_cold):.0f} GB/s) warm {t_warm:.0f} us ({gbps(t_warm):.0f} GB/s) | "
              f"B alu {t_alu:.0f} us, alu+touch {t_alu_t:.0f} us | C no touch {c_no:.0f}/{c_no2:.0f} us, touch {c_yes:.0f}/{c_yes2:.0f} us, gain {gain:+.0f} us per layer "
              f"(x40 layers = {gain * 40 / 1000:+.2f} ms per token if it carried over)")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
