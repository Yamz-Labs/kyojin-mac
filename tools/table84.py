"""jsonl of prof84 -> per-prompt-length table. usage: table84.py JSONL H I [T,T,..]   (H hidden, I intermediate; 32 layers = 24 GDN + 8 attn, 16 q heads, 4 kv heads, hd 256, GDN 16/32 heads x 128)
Per prompt T: sum over the chunks o < T of plain ms; class ms scaled by plain/inst, interpolated linearly in o when the run used a stride."""
import sys, json, numpy as np
P = 13.7e12; BW = 139.3e9
f, H, I = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); Ts = [int(t) for t in (sys.argv[4] if len(sys.argv) > 4 else "4096,16384,32768").split(",")]
GROUP = {"norm_in": "norm", "norm_gdn": "norm", "norm_qk": "norm", "gdn_conv": "gdn_front", "glue_gdn": "gdn_front", "glue_mlp": "mlp_glue", "rope": "attn_misc", "kv_write": "attn_misc", "glue_attn": "attn_misc"}
rs = [json.loads(l) for l in open(f)]
for r in rs:
    c = {}
    for k, v in r["cls_ms"].items(): c[GROUP.get(k, k)] = c.get(GROUP.get(k, k), 0) + v
    r["cls_ms"] = c; L = rs[0]["chunk"]; offs = np.array([r["o"] for r in rs])
classes = sorted({k for r in rs for k in r["cls_ms"]}, key=lambda k: -np.mean([r["cls_ms"].get(k, 0) for r in rs]))
def interp(vals, o): return np.interp(o, offs, vals)
def roof_ms(cls, o):                                           # per chunk of L rows at offset o, ms
    pg = H * 8192 + H * 4096 + 2 * H * 32 + 4096 * H; pa = H * 8192 + 2 * H * 1024 + 4096 * H; pm = 3 * H * I     # params per layer
    if cls == "gemm": return 2 * L * (24 * (pg + pm) + 8 * (pa + pm)) / P * 1e3
    if cls == "sdpa": return 8 * 4 * L * (o + (L + 1) / 2) * 16 * 256 / P * 1e3
    if cls == "gdn_core": return 24 * L * 5.8e6 / P * 1e3
    if cls == "gdn_front": return 24 * (L * 8192 * 2 * 2 + L * 8192 * 2) / BW * 1e3        # fused: read qkv, write q k v (conv + silu + qk norm + g / beta in one pass)
    if cls == "norm": return (64 * L * H * 2 * 2 + 24 * 3 * L * 4096 * 2 + 16 * L * 1024 * 2 * 2) / BW * 1e3
    if cls == "mlp_glue": return 32 * 3 * L * I * 2 / BW * 1e3                              # silu(gate) * up: 2 reads + 1 write
    if cls == "attn_misc": return 8 * (L * 4096 * 2 * 2 + L * 1024 * 2 * 2 * 3) / BW * 1e3   # rope q k, new k v write
    if cls == "outside": return 64 * 3 * L * H * 2 / BW * 1e3                               # residual adds (head = last row only)
    return 0.0
for T in Ts:
    os_ = np.arange(0, T, L); n = len(os_)
    plain = sum(interp([r["plain_ms"] for r in rs], o) for o in os_)
    print(f"\nT={T} ({n} chunks of {L}): {plain:.0f} ms = {T/plain*1e3:.0f} tok/s")
    print(f"{'class':10s} {'ms/chunk avg':>12s} {'share %':>8s} {'roof ms/chunk':>14s} {'% of roof':>10s} {'gap ms/chunk':>13s}")
    tot_gap = 0
    for c in classes:
        ms = np.mean([interp([r["cls_ms"].get(c, 0) * r["plain_ms"] / r["inst_ms"] for r in rs], o) for o in os_])
        rf = np.mean([roof_ms(c, o) for o in os_]); sh = ms * n / plain * 100
        print(f"{c:10s} {ms:12.1f} {sh:8.1f} {rf:14.1f} {(rf/ms*100 if rf else float('nan')):10.0f} {ms-rf:13.1f}")
