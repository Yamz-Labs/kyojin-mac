#!/bin/bash
# pair.sh TAG JOBS...  job = model:ctx:cfgA,cfgB ; one depth89 pass per job through macrun (long, yields to short jobs), one at a time
TAG=$1; shift
cd ~/kyojin-mlx
for j in "$@"; do
  IFS=: read M C CF <<< "$j"; K=$(echo "$CF" | tr ',+=' '__-')
  echo "=== $TAG $M $C $CF $(date +%s)"
  D89_PACK=$HOME/rivals/models/Qwen3.5-$M-mlx4 DEPTH89_PAUSE=45 D89_TEXTS=/tmp/w107/txt_${TAG}_${M}_${C}_$K.json MACRUN_LONG=1 \
    ~/kyojin-mlx/macguard/macrun ~/kyojin-mlx/.venv/bin/python tools/depth89.py $C 1 $CF > /tmp/w107/p_${TAG}_${M}_${C}_$K.log 2>&1
  echo "rc $? $(date +%s)"; grep -A6 SUMMARY /tmp/w107/p_${TAG}_${M}_${C}_$K.log
done
