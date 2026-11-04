"""Which EXL3 pack linears have a matching module in mlx-vlm's model tree? Builds the model from the pack config (lazy weights, nothing
evaluated) and maps every quantized linear of a names file (tools/pack_names.py) onto it.
usage: PYTHONPATH=<mlx_vlm dir>:. python tools/map_vlm.py glm5_next|deepseek_v41 CONFIG.json NAMES.json"""
import collections, importlib, json, re, sys
import mlx.nn as nn
arch, cfgp, namesp = sys.argv[1:4]
cfg, names = json.load(open(cfgp)), json.load(open(namesp))
mod = importlib.import_module(f"mlx_vlm.models.{arch}")
model = mod.Model(mod.ModelConfig.from_dict(cfg))
mods = {k: v for k, v in model.named_modules()}
kinds = collections.Counter(type(v).__name__ for k, v in mods.items())
print(arch, "modules", len(mods), {k: v for k, v in kinds.items() if k in ("Linear", "MultiLinear", "SwitchGLU", "DeepseekV41SwitchGLU", "Embedding", "ParallelHead")})
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
from kyojin_mlx.archs import FUSED, target as _target
target = lambda p: _target(arch, p)

res = collections.defaultdict(list)
for n, (dt, shp) in names.items():
    if not n.endswith(".trellis"): continue
    p = n[:-len(".trellis")]
    if p.endswith(".slices"): p = p[:-len(".slices")] + ".wo_a.slices"; p = p.replace(".wo_a.wo_a", ".wo_a")
    q = target(p)
    if q is None: res["dropped by mlx-vlm sanitize (mtp.*)"].append(p); continue
    leaf = q.rsplit(".", 1)[-1]; base = q.rsplit(".", 1)[0]
    if q in mods: res[f"direct: {type(mods[q]).__name__}"].append(p)
    elif f"{base}.{FUSED.get(leaf, '-')}" in mods: res[f"fused in mlx-vlm -> {FUSED[leaf]} (split into two Exl3Linear)"].append(p)
    else: res["NO MODULE"].append(p)
for k, v in sorted(res.items()):
    pat = collections.Counter(re.sub(r"\d+", "N", x) for x in v)
    print(f"{len(v):6d}  {k}")
    for x, c in pat.most_common(6): print(f"         {c:6d}  {x}")
