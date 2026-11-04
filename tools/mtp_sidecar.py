"""Write the MTP sidecar of a Qwen3.5 mlx-lm pack: the `mtp.*` tensors of the bf16 checkpoint, into <pack>/mtp/mtp.safetensors.
usage: mtp_sidecar.py BF16_DIR PACK_DIR"""
import glob, os, sys
import mlx.core as mx
src, pack = sys.argv[1], sys.argv[2]
w = {}
for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    for k, v in mx.load(f).items():
        if k.startswith("mtp."):
            w[k] = v
assert w, f"no mtp.* tensors in {src}"
out = os.path.join(pack, "mtp"); os.makedirs(out, exist_ok=True)
mx.save_safetensors(os.path.join(out, "mtp.safetensors"), w)
print(f"{len(w)} tensors, {sum(v.nbytes for v in w.values()) / 2**20:.0f} MiB -> {out}")
