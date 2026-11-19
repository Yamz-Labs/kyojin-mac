"""Hy4-flash footprint plan on the Mac line (numbers in GiB) for 128K..1M tokens. Pure arithmetic, no MLX.
Model facts (config.json): 42 layers = 12 DSA (kpool) + 30 sliding-window (window 512); MLA row = latent 512 + rope 64 = 576; kpool layers also keep the raw pooled-key
value + gate rows (2 x 128) per token and the pooled keys (128 per 4 tokens). Measured anchors (fixed-state KV, fp32 rows): 123163 tokens -> kv 12.97 GiB (c131).
NOTE: c131 measured 222.78 GiB peak at 123K tokens on the OLD path (pre-flash gather attend, chunk 2048, 16 GiB prefix snapshot, kv_peak = 2 x kv): that gap to this plan (~116) is attention / scorer temporaries of the old path plus snapshot, NOT shown to be gone at 128K+ yet: the c135 cells (prefix cache off, flash path) record the real peak per arm; treat every total below as an estimate until they land.
BASE = resident weights + runtime at short context: served peak 101 GiB at 8K with cache limit 16 GiB -> ~85 GiB (estimate: re-measure with the 1m cells)."""
import sys
G = 2 ** 30
DSA, SWA, LAT, DX, KP, WIN = 12, 30, 576, 128, 4, 512
BASE = float(sys.argv[1]) if len(sys.argv) > 1 else 85.0
CACHE = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0           # MLX buffer cache limit (HY4_CACHE_MB)
CHUNK = 4096
J = 128
rows = []
for n in (131072, 262144, 524288, 1048576):
    cap = -(-n // 1024) * 1024
    def kv(b_lat, b_raw, ring, drop_raw):
        dsa = DSA * cap * (LAT * b_lat + (0 if drop_raw else 2 * DX * b_raw))
        pk = DSA * (cap // KP) * DX * 4
        swa = SWA * (WIN + CHUNK + 16 if ring else cap) * LAT * b_lat
        return (dsa + pk + swa) / G
    pools = cap // KP
    act = 2 * 16 * 256 * pools * 4 / G                                # pools_fixed_top block 256: [16, 256, J] fp32 + the weighted copy
    grow = DSA and (cap * (LAT * 4 + 2 * DX * 4)) / G                 # one layer's buffer is copied on growth: old + new alive (one layer at a time)
    for name, args in (("now (fp32, SWA full rows)", (4, 4, False, False)), ("SWA ring, fp32", (4, 4, True, False)),
                       ("SWA ring, fp32, no raw rows", (4, 4, True, True)), ("SWA ring, bf16 latent, no raw rows", (2, 4, True, True))):
        k = kv(*args)
        rows.append((n, name, k, BASE + CACHE + k + act + grow))
print(f"BASE {BASE} GiB (weights + runtime), MLX cache {CACHE} GiB, prefill chunk {CHUNK}")
print(f"{'ctx':>8} {'variant':36} {'KV GiB':>8} {'total GiB':>10}   (total = base + cache + KV + scorer temporaries + one-layer growth copy)")
for n, name, k, t in rows:
    print(f"{n // 1024:>7}K {name:36} {k:8.1f} {t:10.1f}  {'OVER 236' if t > 236 else ('<=125 (lane beside)' if t <= 125 else 'fits 236, > 125')}")
