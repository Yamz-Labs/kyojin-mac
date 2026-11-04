import sys
from kyojin_mlx.serve.backends import load_backend
from kyojin_mlx import tuning as TN, layer_pf as LP, mlp_pf as MP
be, _ = load_backend(sys.argv[1])
print(TN.INFO["status"], TN.TABLE.get("prefill"), "layer max", LP.max_rows(), "mlp on", MP.enabled(), "split max", LP.split_max_rows())
