"""Ultra MTP estimate with grouped routed experts in the chained verify (ULTRA.md s.13 / s.14 method, tools/mtp_estimate.py, 'split' GPU rows).
A round = max(CPU rows, GPU rows) + draft call (CPU min + sync); tokens per round = sum_{i<n} a^i; best D per acceptance a.
GPU rows(n) = gpu_ms * W_n(f) + L_n * gap; W_n = work multipliers of the verify of n rows (s.11: 1.00 1.36 1.81 2.27 3.08).
Grouped experts: the extra rows of the chain read each routed expert once; with a fraction f of every row's experts repeated from the previous row's, and a share eta of
that saving realized by the kernel (Air layer bench 0.6-0.9), the work of row n drops by r * eta * f * (n - 1) solo steps, r = 2.5 / 7.3 = routed bytes per row
over the bytes of a solo step (tools/ultra_est_grouped.py bandwidth model). The grouped path costs one more launch per MoE layer (the plan), 43 per verify of n >= 2 rows;
its CPU encode measured equal to the per-row path (Air, verify_gpu cpu column).
usage: mtp_ultra_est.py [--overhead-ms 0.45] [--eta 0.7] [--solo-cpu 11.8]"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--solo-cpu", type=float, default=11.8); ap.add_argument("--solo-launches", type=float, default=1770)
ap.add_argument("--gpu-ms", type=float, default=9.1); ap.add_argument("--overhead-ms", type=float, default=0.45)
ap.add_argument("--eta", type=float, default=0.7); ap.add_argument("--sync-ms", type=float, default=1.5); ap.add_argument("--draft-cpu", type=float, default=2.29)
ap.add_argument("--per-row-launches", type=float, default=600.0)
ap.add_argument("--cpu-ratio", default="0.99,1.18,1.61,1.77,2.62", help="CPU of a verify of 1..5 rows in solo steps (tools/spec_cpu.py, min)")
ap.add_argument("--dmax", type=int, default=4, help="deepest draft tried (owner allows 10)")
ap.add_argument("--w-slope", type=float, default=0.0, help="GPU work per extra verify row beyond n = 5, in solo steps (0 = flat: the NAX / MMA row paths tie decode chunks up to ~36 rows, macmoepf)")
ap.add_argument("--cpu-slope", type=float, default=0.4, help="CPU encode per extra verify row beyond n = 5, in solo steps")
ap.add_argument("--vb-gain", type=float, default=0.0, help="share of the routed work of a verify of n >= 2 rows saved by the per-bitrate passes (KYOJIN_VB_SPLIT_K; Air ds_l10 grouped GEMV 0.015-0.03)")
ap.add_argument("--af-gain", type=float, default=0.0, help="share of the routed work of every row saved by the affine fold (KYOJIN_MOE_AFFOLD, opt-in, KL-gated)")
a = ap.parse_args()
W = [1.00, 1.36, 1.81, 2.27, 3.08]
CPU_RATIO = [float(x) for x in a.cpu_ratio.split(",")]     # default: s.14 (spec_cpu, ds2m, loaded Air)
for _n in range(6, 12):
    W.append(W[4] + a.w_slope * (_n - 5)); CPU_RATIO.append(CPU_RATIO[4] + a.cpu_slope * (_n - 5))
R_ROUTED = 2.5 / 7.3
MOE_LAYERS = 43


def best(gap, f, acc, grouped):
    solo = max(a.solo_cpu, a.gpu_ms * (1 - R_ROUTED * a.af_gain) + a.overhead_ms + a.solo_launches * gap * 1e-3)
    top = (0, 1e3 / solo, solo)
    for D in range(1, a.dmax + 1):
        n = D + 1
        w = W[n - 1] - (R_ROUTED * a.eta * f * (n - 1) if grouped else 0.0)
        w -= R_ROUTED * (W[n - 1] - (R_ROUTED * a.eta * f * (n - 1) if grouped else 0.0)) * (a.af_gain + (a.vb_gain if n >= 2 else 0.0))
        launches = a.solo_launches + (n - 1) * a.per_row_launches + (MOE_LAYERS if grouped and n >= 2 else 0)
        gpu = a.gpu_ms * w + a.overhead_ms + launches * gap * 1e-3
        cpu = CPU_RATIO[n - 1] * a.solo_cpu
        rnd = max(cpu, gpu) + a.draft_cpu + a.sync_ms
        tps = 1e3 * sum(acc ** i for i in range(n)) / rnd
        if tps > top[1]:
            top = (D, tps, rnd)
    return top


for gap in (1.7, 3.0):
    print(f"== gap {gap} us, eta {a.eta}, overhead {a.overhead_ms} ms, solo CPU {a.solo_cpu} ms: MTP tok/s at a = 0.6 / 0.735 / 0.85 (best D)")
    print(f"  per-row experts (feat/linear1 state)         : " + " / ".join(f"{best(gap, 0, ac, False)[1]:5.1f} (D{best(gap, 0, ac, False)[0]})" for ac in (0.6, 0.735, 0.85)))
    for f in (0.0, 0.3, 0.5):
        print(f"  grouped, overlap f = {f:.1f}                      : " + " / ".join(f"{best(gap, f, ac, True)[1]:5.1f} (D{best(gap, f, ac, True)[0]})" for ac in (0.6, 0.735, 0.85)))
