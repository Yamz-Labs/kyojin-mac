#!/bin/zsh
# macair26 layer gate: PACK (ds2 = DS4.1 dims, glm2 = GLM dims) patched with the prefill kernels OFF (today: KYOJIN_ATTN_PF=0 KYOJIN_DENSE_FUSED_MAX=128) and ON
# (KYOJIN_ATTN_PF=1 + $2 env), both vs the fp32 numpy-reconstructed reference built on the router ids of the ON run. usage: dense_gate.sh ds2|glm2 [--no-fq]
set -e
cd ~/kyojin-mlx/.worktrees/pf3
export PYTHONPATH=$PWD:$HOME/kyojin-mlx/vlm_pkg:$PWD/tools
PY=$HOME/kyojin-mlx/.venv/bin/python
pack=$1; shift
out=$HOME/kyojin-mlx/results_mac/macair30; mkdir -p $out
tag=${pack}${1:+_nofq}${GATE_TAG:+_$GATE_TAG}
ntok=${GATE_NTOK:-129}
toks=$($PY -c "import numpy as np;print(','.join(map(str,np.random.default_rng(7).integers(1,3000,$ntok))))")
KYOJIN_PF_MIN_ROWS=1 KYOJIN_ATTN_PF=0 KYOJIN_PF_GLUE=0 $PY tools/layer_gate.py run $HOME/kyojin-mlx/models/$pack $out/g_${tag}_off.npz --tokens $toks "$@" 2>&1 | tail -2
KYOJIN_PF_MIN_ROWS=1 KYOJIN_ATTN_PF=1 $PY tools/layer_gate.py run $HOME/kyojin-mlx/models/$pack $out/g_${tag}_on.npz --tokens $toks "$@" 2>&1 | tail -2
[ -n "$GATE_NOREF" ] || $PY tools/layer_gate.py run $HOME/kyojin-mlx/models/$pack $out/g_${tag}_ref.npz --mode ref --ids $out/g_${tag}_on.npz --tokens $toks "$@" 2>&1 | tail -2
if [ -z "$GATE_NOREF" ]; then for v in off on; do echo "=== $tag $v vs fp32 ref"; $PY tools/layer_gate.py compare $out/g_${tag}_$v.npz $out/g_${tag}_ref.npz --json $out/cmp_${tag}_${v}_vs_ref.json; done; fi
echo "=== $tag on vs off"; $PY tools/layer_gate.py compare $out/g_${tag}_on.npz $out/g_${tag}_off.npz --json $out/cmp_${tag}_on_vs_off.json
