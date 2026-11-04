#!/bin/zsh
# usage: rows.sh SIZE(4B|9B) T C REPS ; ours+mlx-lm interleaved in-process, then BaseRT bench
cd ~/kmlx-macpf4; export PYTHONPATH=$PWD:$HOME/kyojin-mlx/vlm_pkg
S=$1; T=$2; C=$3; R=$4
~/kyojin-mlx/macguard/macrun ~/kyojin-mlx/.venv/bin/python tools/pf3_var.py ~/rivals/models/Qwen3.5-$S-exl3-eq ~/rivals/models/Qwen3.5-$S-mlx4 $T $C $R full 2>&1 | tail -3
if [ -n "$BASERT" ] && [ "$S" = 4B ]; then ~/kyojin-mlx/macguard/macrun ~/rivals/baseRT-bin/v0.4.1/x/basert-bench ~/rivals/baseRT-bin/v0.4.1/models/q35-4b-q4.base -p $T -n 0 -r $R 2>&1 | tail -4; fi
