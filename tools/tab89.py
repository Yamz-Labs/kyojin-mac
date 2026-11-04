"""summarise curve83_ops jsonl records into the per-op table: measured vs roof (BW 139.3 GB/s). usage: tab89.py JSONL [N]"""
import sys, json
BW = 139.3e9
for l in open(sys.argv[1]):
    r = json.loads(l)
    if len(sys.argv) > 2 and str(r["N"]) != sys.argv[2]: continue
    g = {k: v["p25"] for k, v in r["stats"].items()}
    if "no_head" not in g: continue
    w = r["wbytes"]; kv = r["kv_bytes"]; gs = r["gdn_state_bytes"]
    rows = [("sdpa+kv", g["full"] - g["no_sdpa_kv"], kv / BW), ("attn proj/norm/rope", g["no_sdpa_kv"] - g["no_attnmod"], w["attn_lin"] / BW),
            ("GDN recurrent core", g["full"] - g["no_gdn_core"], 2 * gs / BW), ("GDN rest (proj/conv/norm)", g["full"] - g["no_gdnmod"] - (g["full"] - g["no_gdn_core"]), w["gdn_all"] / BW),
            ("MLP", g["full"] - g["no_mlp"], w["mlp"] / BW), ("lm_head", g["full"] - g["no_head"], w["head"] / BW), ("floor(host+embed)", g["floor"], 0)]
    roof = sum(x[2] for x in rows) * 1e3
    print("N=%d full %.2f ms (%.1f tok/s) async %.2f host_enc %.2f roof_total %.2f ms (%.0f%%)" % (r["N"], g["full"], 1e3 / g["full"], r["async_ms"], r["host_encode_ms"], roof, 100 * roof / g["full"]))
    for n, m, ro in rows: print("  %-28s %6.2f ms  roof %5.2f  over %5.2f" % (n, m, ro * 1e3, m - ro * 1e3))
