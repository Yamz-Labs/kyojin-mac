#!/bin/bash
# S6 layer gate: for PACK (ds2 = DS4.1 dims, glm2 = GLM dims): patched with S3 (decode_pf + gather_mm) and with S6 (fused), both forced through the
# MoE prefill path, vs the fp32 numpy-reconstructed reference built on the router ids of the S6 run. usage: s6_gate.sh ds2|glm2 [--no-fq]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PYTHONPATH=$PWD:${KYOJIN_VLM_PKG:-$PWD/vlm_pkg}:$PWD/tools
PY=${PYTHON:-python}
M=${KYOJIN_MODELS:-$PWD/models}
pack=$1; shift
out=${S6_GATE_OUT:-$PWD/results_mac/s6_gate}; mkdir -p $out
tag=${pack}${1:+_nofq}
toks=$($PY -c "import numpy as np;print(','.join(map(str,np.random.default_rng(7).integers(1,3000,129))))")
KYOJIN_PF_MIN_ROWS=1 KYOJIN_PF_FUSED=0 $PY tools/layer_gate.py run $M/$pack $out/g_${tag}_s3.npz --tokens $toks "$@" 2>&1 | tail -2
KYOJIN_PF_MIN_ROWS=1 KYOJIN_PF_FUSED=1 $PY tools/layer_gate.py run $M/$pack $out/g_${tag}_s6.npz --tokens $toks "$@" 2>&1 | tail -2
$PY tools/layer_gate.py run $M/$pack $out/g_${tag}_ref.npz --mode ref --ids $out/g_${tag}_s6.npz --tokens $toks "$@" 2>&1 | tail -2
for v in s3 s6; do echo "=== $tag $v vs fp32 ref"; $PY tools/layer_gate.py compare $out/g_${tag}_$v.npz $out/g_${tag}_ref.npz --json $out/cmp_${tag}_${v}_vs_ref.json; done
echo "=== $tag s6 vs s3"; $PY tools/layer_gate.py compare $out/g_${tag}_s6.npz $out/g_${tag}_s3.npz --json $out/cmp_${tag}_s6_vs_s3.json
