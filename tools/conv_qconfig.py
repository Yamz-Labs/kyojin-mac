"""Write quantization_config.json (per-module tensor map the exllamav3 loaders read) for an EXL3 pack directory, from the safetensors headers + config.json.
usage: python tools/conv_qconfig.py PACK_DIR"""
import glob, json, struct, sys
MUL1 = 0x83DCD12D
D = {"F16": "torch.float16", "BF16": "torch.bfloat16", "F32": "torch.float32", "I16": "torch.int16", "I32": "torch.int32"}
SZ = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4}
d = sys.argv[1]
storage = {}
for p in sorted(glob.glob(f"{d}/*.safetensors")):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    for k, v in sorted(h.items()):
        mod = k.rsplit(".", 1)[0]
        if mod.endswith((".linear_attn", ".conv1d")):
            continue                                  # GatedDeltaNet state parameters are not listed by the converter
        e = storage.setdefault(mod, {"stored_tensors": {}})
        nel = 1
        for s in v["shape"]: nel *= s
        e["stored_tensors"][k] = {"shape": v["shape"], "n_bytes": nel * SZ[v["dtype"]], "dtype": D[v["dtype"]]}
        if k.endswith(".trellis"):
            e["quant_format"] = "exl3"; e["bits_per_weight"] = v["shape"][-1] // 16; e["mul1_multiplier"] = MUL1
qc = json.load(open(f"{d}/config.json"))["quantization_config"]; qc["tensor_storage"] = storage
json.dump(qc, open(f"{d}/quantization_config.json", "w"), indent=4)
print(len(storage), "modules")
