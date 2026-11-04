"""Why the Engram disk path holds command buffers instead of polling a flag in a kernel: a GPU kernel that polls a word the CPU writes after the kernel started
does NOT see the write when the GPU has read that cache line before. Chain of kernels (step i is committed while step i - 1 runs, the host writes the flag of
step i after step i - 1 has finished, as the pipelined decode would): with ONE reused flag line nearly every step times out; with a fresh line per step none does,
and a reused line only works after the GPU went idle. Measured on the M5 Air (volatile, relaxed atomic and read-modify-write polls alike; coherent(device)
compiles and changes nothing; thread_scope_system does not compile).
usage: python tools/gate_probe.py [--steps 80] [--spins 1048576]   (bounded spin: ~0.1 s per timed-out kernel; keep it well under the GPU watchdog)"""
import argparse
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=80)
ap.add_argument("--spins", type=int, default=1 << 20)
a = ap.parse_args()
SRC = """
    uint tid = thread_position_in_threadgroup.x; uint want = tgt[0]; uint go = 0; uint k = 0;
    for (k = 0; k < SPINS; ++k) { go = ((volatile const device uint*)flag)[LINE * 32]; if (go == want) break; }
    if (tid == 0) { dbg[0] = k; dbg[1] = go; }
    threadgroup_barrier(mem_flags::mem_device);
    out[tid] = prev[tid] + 1;
"""
kern = mx.fast.metal_kernel(name="gate_probe", input_names=["flag", "tgt", "prev"], output_names=["out", "dbg"], source=SRC)
flag = mx.array(np.zeros(32 * (a.steps + 1), dtype=np.uint32)); mx.eval(flag)
fnp = np.frombuffer(memoryview(flag), dtype=np.uint32)
for name, fresh in (("one reused flag line", False), ("a fresh line per step", True)):
    prev, pend, timeouts = mx.zeros((256,), dtype=mx.uint32), None, 0
    mx.eval(prev)
    for i in range(1, a.steps + 1):
        want, line = (1000 if fresh else 5000) + i, (i if fresh else 0)
        o = kern(inputs=[flag, mx.array(np.full(16, want, dtype=np.uint32)), prev], template=[("SPINS", a.spins), ("LINE", line)], grid=(256, 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(256,), (4,)], output_dtypes=[mx.uint32, mx.uint32])
        mx.async_eval(*o)
        if pend is not None:
            mx.eval(pend[1]); timeouts += int(np.array(pend[1])[0]) >= a.spins - 1          # step i - 1 finished (its poll ended)
        fnp[line * 32] = want                                                               # the host writes the release of step i after step i - 1 is done
        pend, prev = o, o[0]
    mx.eval(pend[1]); timeouts += int(np.array(pend[1])[0]) >= a.spins - 1
    print(f"{name:24s} {timeouts:3d} of {a.steps} gates never saw the CPU write (timed out)")
