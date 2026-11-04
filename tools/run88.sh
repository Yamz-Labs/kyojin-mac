#!/bin/bash
# run86.sh WT KIND N CFG [BLOCKS] [STEPS] [VARIANTS] -> one macrun job (<3 min) on worktree WT (path on the Mac), out jsonl in ~/kyojin-mlx/lc88_out
I=$HOME/kmlx-maclc88
PACK=${PACK:-$HOME/rivals/models/Qwen3.5-4B-$([ "$2" = exl3 ] && echo exl3-4.0 || echo mlx4)}
mkdir -p ~/kyojin-mlx/lc88_out
TAG=$(basename $1)
PYTHONPATH=$1:$I/tools ~/kyojin-mlx/macguard/macrun perl -e "alarm ${ALARM:-175}; exec @ARGV" ~/kyojin-mlx/.venv/bin/python $I/tools/${OPS:-curve83_ops.py} $2 $PACK $3 $4 ${5:-9} ${6:-8} ~/kyojin-mlx/lc88_out/ops_$TAG.jsonl ${7:-full} 2>&1 | grep -v Warning | cut -c1-300
