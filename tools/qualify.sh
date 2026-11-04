#!/usr/bin/env bash
# qualify --backend mlx in ONE command (see tools/qualify/DESIGN.md). Resumable: rerun it after any stop.
#   tools/qualify.sh --pack PACK_DIR [--ref BF16_DIR] [--dry-run] [--stage a,b] [--redo a] [--scale air|ultra]
# Output: tools/../qualify_out/<pack>/mlx/{REPORT.md,qualify.json,<stage>.json,qualify.log}. Every GPU job holds ~/kyojin-mlx/.gpu.lock (one heavy job at a time).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
for v in "${KYOJIN_VENV:-}" "$REPO/.venv" "$REPO/../../.venv" "$REPO/../.venv"; do [ -n "$v" ] && [ -f "$v/bin/activate" ] && . "$v/bin/activate" && break; done
export PYTHONPATH="$REPO:${KYOJIN_VLM_PKG:-$REPO/vlm_pkg}:$REPO/tools${PYTHONPATH:+:$PYTHONPATH}"
exec python "$REPO/tools/qualify/qualify.py" --repo "$REPO" "$@"
