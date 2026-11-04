"""Ultra MTP estimate of ULTRA.md s.13 / s.14: a round = max(CPU rows, GPU rows) + draft call; tokens per round = sum_{i<=D} a^i; best D per acceptance a.
usage: mtp_estimate.py [--solo-cpu 11.8] [--solo-launches 2302] [--gpu-ms 9.1] [--gap-us 1.7,3.0]
CPU rows  = ratio_n x solo CPU (tools/spec_cpu.py on ds2m, min).
GPU rows  = 'old': GPU_ROWS_n x (gpu-ms + solo-launches x gap)                (s.13: the s.11 multipliers scale the whole GPU side)
            'split': gpu-ms x W_n + L_n x gap, W_n = the s.11 multipliers as WORK multipliers of the roof, L_n = launches of a verify of n rows of the 40-layer mix
            (solo launches + (n-1) x per-row launches: 600 now, 640 before, composed over the 40-layer mix from verify_census layer-kind slices: reuse 2, full 69-72, reindex 31-34, local ~30, candidate 12, Engram ~55) -- launch cuts of the verify path show here.
draft     = draft call CPU min + SYNC_MS (the draft tokens go back to python before the verify is built), as in s.13."""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--solo-cpu", type=float, default=11.8); ap.add_argument("--solo-launches", type=float, default=2302)
ap.add_argument("--gpu-ms", type=float, default=9.1); ap.add_argument("--gap-us", default="1.7,3.0"); ap.add_argument("--sync-ms", type=float, default=1.5)
a = ap.parse_args()
W = [1.00, 1.36, 1.81, 2.27, 3.08]
CASES = {                                                         # name: (verify/solo CPU ratios n = 1..5, draft CPU min ms, extra launches per verify row beyond the first)
    "s.13 (macair12)": ([1.39, 1.50, 2.00, 2.44, 3.44], 4.0, 640.0),
    "s.14 base rerun": ([1.38, 1.67, 2.17, 2.43, 3.49], 2.97, 640.0),
    "s.14 now": ([0.99, 1.18, 1.61, 1.77, 2.62], 2.29, 600.0),
    "now, CPU rows = solo (ceiling of CPU work)": ([1.0] * 5, 0.0, 600.0),
}


def best(ratio, dcpu, per_row, gap, model, acc):
    solo_gpu = a.gpu_ms + a.solo_launches * gap * 1e-3
    solo = max(a.solo_cpu, solo_gpu)
    top = (0, 1e3 / solo, solo)
    for D in (1, 2, 3, 4):
        n = D + 1
        cpu = ratio[n - 1] * a.solo_cpu
        if model == "old":
            gpu = W[n - 1] * solo_gpu
        else:
            gpu = a.gpu_ms * W[n - 1] + (a.solo_launches + (n - 1) * per_row) * gap * 1e-3
        rnd = max(cpu, gpu) + dcpu + a.sync_ms
        tps = 1e3 * sum(acc ** i for i in range(n)) / rnd
        if tps > top[1]:
            top = (D, tps, rnd)
    return top


for gap in (float(g) for g in a.gap_us.split(",")):
    print(f"== gap {gap} us (solo step {max(a.solo_cpu, a.gpu_ms + a.solo_launches * gap * 1e-3):.1f} ms)")
    for name, (ratio, dcpu, per_row) in CASES.items():
        for model in ("old", "split"):
            r = [best(ratio, dcpu, per_row, gap, model, acc) for acc in (0.6, 0.735, 0.85)]
            print(f"  {name:44s} GPU rows {model:5s}: " + "  ".join(f"a={acc}: D={b[0]} {b[1]:5.1f} tok/s" for acc, b in zip((0.6, 0.735, 0.85), r)))
    print("  rows (ms) at D=2, n=3: " + "; ".join(
        f"{name}: CPU {ratio[2] * a.solo_cpu:.1f} / GPU old {W[2] * (a.gpu_ms + a.solo_launches * gap * 1e-3):.1f} split {a.gpu_ms * W[2] + (a.solo_launches + 2 * per_row) * gap * 1e-3:.1f}"
        for name, (ratio, dcpu, per_row) in list(CASES.items())[:3]))
