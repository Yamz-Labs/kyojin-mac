"""fp4run114.py [--dense] TOOL.py ARGS...: run an existing tool (spec_check, kvdisk_check, ...) with the DS4.1 model's make_cache returning the fp4-packed cache
(kyojin_mlx.fp4pool), or with --dense the stock cache, so the two runs can be compared line by line."""
import os, runpy, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.expanduser("~/kyojin-mlx/vlm_pkg")); sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
dense = sys.argv[1] == "--dense"
if dense:
    sys.argv.pop(1)
tool = sys.argv[1]; sys.argv = sys.argv[1:]
if not dense:
    from kyojin_mlx import fp4pool
    from mlx_vlm.models.deepseek_v41 import language as L
    cls = [c for c in vars(L).values() if isinstance(c, type) and "make_cache" in vars(c)]
    for c in cls:
        old = c.make_cache
        c.make_cache = (lambda self, old=old: fp4pool.wrap_cache(old(self)))
    print("fp4run114: packed cache installed on", [c.__name__ for c in cls], flush=True)
runpy.run_path(tool, run_name="__main__")
