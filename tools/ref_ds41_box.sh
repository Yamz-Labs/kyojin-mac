#!/usr/bin/env bash
# Box side of the day-1 reference: 3 fixed prompts x 64 greedy tokens on the DS4.1 pack (engine of ~/kyojin-work/ds41golive, serve_ds41.sh settings,
# runtime ablit OFF so the plain pack is compared). Only through the gpu queue:
#   ~/kyojin-work/bin/gpurun --who macair3 --need 100 bash tools/ref_ds41_box.sh OUT_DIR
set -euo pipefail
OUT=${1:?out dir}; HERE=$(cd "$(dirname "$0")" && pwd); ROOT=${DS41_ROOT:-$HOME/kyojin-work/ds41golive}
pgrep -x Ryujinx >/dev/null && { echo "ref_ds41_box: a game runs (Ryujinx), not now" >&2; exit 90; }
pf=$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -c 'amdgpu.*page fault' || true)
[ "$pf" -eq 0 ] || { echo "ref_ds41_box: REFUSED, $pf amdgpu page fault(s) this boot" >&2; exit 97; }
export DS41_PACK=${DS41_PACK:-$HOME/models/ds41-1017-yamz} DS41_ENGRAM_DIR=${DS41_ENGRAM_DIR:-$HOME/models/ds41-1017/engram_fp8}
export EXL3_DSV41_ENGRAM_DIR=$DS41_ENGRAM_DIR DS41_VALIDATE=0 DS41_VALIDATED=1 EXL3_ABLIT_RUNTIME=off
export DS41_WARM=${DS41_WARM:-$HOME/kyojin-work/ds41tierws/warm_counts.json}
set +eu; source "$ROOT/tools/ds41/ds41_env.sh" || exit $?; set -eu
exec python "$HERE/ref_ds41_box.py" "$DS41_PACK" "$OUT" --engram-dir "$DS41_ENGRAM_DIR" --chunk "$DS41_CHUNK"
