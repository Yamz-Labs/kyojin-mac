"""Re-quantise ONLY the lm_head of an existing EXL3 pack at another K (head-only conversion; blocks / embedding are copied unchanged).
   python tools/rehead.py SRC_BF16 PACK OUT_PACK --cal cal.safetensors --head-bits 4
The Hessian is the one the converter uses: final-norm hidden states of the calibration rows through the (quantised) pack, H = X^T X."""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_mlx as CM
from kyojin_mlx.exl3_linear import load_exl3

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--cal", required=True)
    ap.add_argument("--head-bits", type=int, default=4); ap.add_argument("--bs", type=int, default=1); ap.add_argument("--head-chunk", type=int, default=16384)
    a = ap.parse_args()
    mx.set_memory_limit(10 * 2**30); mx.set_cache_limit(2**30)
    ids = np.array(mx.load(a.cal)["input_ids"]); R, C = ids.shape
    hp = f"{a.out}/hess.npy.npz"
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(f"{a.pack}/config.json")); tcfg = cfg.get("text_config", cfg)
    nl = tcfg["num_hidden_layers"]
    if os.path.exists(hp):
        d = np.load(hp, allow_pickle=True); Hh = mx.array(d["H"]); cnt = int(d["n"])
    else:
        model, _ = load_exl3(a.pack)
        lm = model.language_model
        Hh = None; cnt = 0
        for s in range(0, R, a.bs):
            h = lm.model(mx.array(ids[s:s + a.bs].astype(np.int32))).reshape(-1, tcfg["hidden_size"]).astype(mx.float32)
            g = mx.matmul(h.T, h); Hh = g if Hh is None else Hh + g; mx.eval(Hh); cnt += h.shape[0]
            mx.clear_cache(); print("hess", s, f"peakGB {CM.mem_gb():.2f}", flush=True)
        np.savez(hp, H=np.array(Hh), n=cnt)
        del model, lm; mx.clear_cache()
    hd = CM.finalize_H(Hh, cnt, nl)
    print("finalize done", f"peakGB {CM.mem_gb():.2f}", flush=True)
    W = {}
    idx = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"]
    hk = "lm_head.weight" if "lm_head.weight" in idx else "model.language_model.embed_tokens.weight"
    head_w = mx.load(f"{a.src}/{idx[hk]}")[hk]
    r = CM.quantize_big(head_w, hd, a.head_bits, nl, f"{a.out}/head_scratch.npy", a.head_chunk, 1)
    print("head K", a.head_bits, "proxy", r["proxy"], "g", r["g"], flush=True)
    t = mx.load(f"{a.pack}/model.safetensors")
    for k in ("trellis", "suh", "svh"): t[f"lm_head.{k}"] = mx.array(r[k])
    mx.save_safetensors(f"{a.out}/model.safetensors", t, metadata={"format": "pt"})
    for f in os.listdir(a.pack):
        if f not in ("model.safetensors", "tuning") and os.path.isfile(f"{a.pack}/{f}"): shutil.copy(f"{a.pack}/{f}", f"{a.out}/{f}")
    for fn in ("config.json", "quantization_config.json"):
        d = json.load(open(f"{a.out}/{fn}")); q = d.get("quantization_config", d); q["head_bits"] = a.head_bits
        if fn == "quantization_config.json":
            ts = d["tensor_storage"]["lm_head"]["stored_tensors"]; ts["lm_head.trellis"]["shape"] = list(r["trellis"].shape)
            ts["lm_head.trellis"]["n_bytes"] = int(np.prod(r["trellis"].shape)) * 2
        json.dump(d, open(f"{a.out}/{fn}", "w"), indent=2)
    os.remove(f"{a.out}/head_scratch.npy") if os.path.exists(f"{a.out}/head_scratch.npy") else None
main()
