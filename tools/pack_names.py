"""Dump {tensor name: [dtype, shape]} of an EXL3 pack from the safetensors headers, with routed experts and wo_a slices collapsed to
the stacked names convert_stack.py writes (<p>.experts.<proj>.trellis, <p>.slices.trellis). usage: pack_names.py PACK_DIR OUT.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import convert_stack as CS
src = CS.Src(sys.argv[1]); out = {}
for k, v in src.t.items():
    m = CS.EXP.match(k) or CS.SLC.match(k)
    if m and CS.EXP.match(k):
        if m[4] == "trellis": out[f"{m[1]}.experts.{m[3]}.trellis"] = ["I16", v[4]]
    elif m:
        if m[3] == "trellis": out[f"{m[1]}.slices.trellis"] = ["I16", v[4]]
    else:
        out[k] = [v[3], v[4]]
json.dump(out, open(sys.argv[2], "w"))
print(len(src.t), "->", len(out), "names")
