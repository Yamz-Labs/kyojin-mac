"""Calibrate lane-op throughput on the GPU: ns per op per lane, Gops/s. 8 independent chains x 512 iterations."""
import time, mlx.core as mx
OPS = {
 "iadd":   "a{i} = a{i} + c{i};",
 "iand_shr": "a{i} = (a{i} & 0x00ff00ffu) + (a{i} >> 8);",   # 3 ops
 "imul":   "a{i} = a{i} * 0x83DCD12Du;",
 "bfe":    "a{i} = extract_bits(a{i}, 3u, 16u) + c{i};",      # bfe + add
 "mul24":  "a{i} = mul24(a{i}, 0xCD12Du) + c{i};",
 "mulhi":  "a{i} = mulhi(a{i}, 0x83DCD12Du) + c{i};",
 "ffma":   "f{i} = fma(f{i}, 1.0001f, fc{i});",
 "hfma":   "h{i} = fma(h{i}, half(1.001h), hc{i});",
 "cvt_i2f": "f{i} = float(a{i}) + f{i}; a{i} += 1u;",           # cvt + fadd + iadd
 "cvt_h2f": "f{i} = float(h{i}) * f{i}; ",
 "popc":   "a{i} = popcount(a{i}) + c{i};",
 "rot":    "a{i} = rotate(a{i}, 8u) + c{i};",
}
body = r"""
    uint tid = thread_position_in_grid.x;
    uint c0=tid|1u,c1=tid+3u,c2=tid+5u,c3=tid+7u,c4=tid+9u,c5=tid+11u,c6=tid+13u,c7=tid+15u;
    uint a0=c0,a1=c1,a2=c2,a3=c3,a4=c4,a5=c5,a6=c6,a7=c7;
    float f0=1.f,f1=1.1f,f2=1.2f,f3=1.3f,f4=1.4f,f5=1.5f,f6=1.6f,f7=1.7f;
    float fc0=1e-3f*tid,fc1=2e-3f,fc2=3e-3f,fc3=4e-3f,fc4=5e-3f,fc5=6e-3f,fc6=7e-3f,fc7=8e-3f;
    half h0=1.h,h1=1.1h,h2=1.2h,h3=1.3h,h4=1.4h,h5=1.5h,h6=1.6h,h7=1.7h;
    half hc0=half(1e-3f*(tid&7)),hc1=2e-3h,hc2=3e-3h,hc3=4e-3h,hc4=5e-3h,hc5=6e-3h,hc6=7e-3h,hc7=8e-3h;
    for (uint it = 0; it < 512; ++it) {
      BODY
    }
    out[tid] = float(a0^a1^a2^a3^a4^a5^a6^a7) + f0+f1+f2+f3+f4+f5+f6+f7 + float(h0+h1+h2+h3+h4+h5+h6+h7);
"""
N = 1 << 20
for name, op in OPS.items():
    b = "\n      ".join(op.format(i=i) for i in range(8))
    k = mx.fast.metal_kernel(name="cal_" + name, input_names=["dummy"], output_names=["out"], source=body.replace("BODY", b))
    f = lambda: k(inputs=[mx.zeros(1)], grid=(N, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
    for _ in range(3): mx.eval(f())
    mx.synchronize(); t0 = time.perf_counter(); o = [f() for _ in range(10)]; mx.eval(o); mx.synchronize()
    dt = (time.perf_counter() - t0) / 10
    nops = N * 512 * 8
    print(f"{name:10s} {dt*1e3:7.2f} ms  {nops/dt/1e9:7.1f} G(src stmts)/s", flush=True)
