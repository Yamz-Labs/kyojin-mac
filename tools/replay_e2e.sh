#!/bin/bash
# tools/replay_e2e.sh PACK [extra args]: the real-drafter end-to-end digest with replay off and on (two processes, each under the GPU lock), then the comparison
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
for v in "$REPO/.venv" "$REPO/../../.venv" "$REPO/../.venv"; do [ -f "$v/bin/activate" ] && . "$v/bin/activate" && break; done
export PYTHONPATH="$REPO:${KYOJIN_VLM_PKG:-$REPO/vlm_pkg}:$REPO/tools"
LOCK="${KYOJIN_GPU_LOCK:-$HOME/kyojin-mlx/.gpu.lock}"; LK=(); command -v lockf > /dev/null && LK=(lockf -k "$LOCK")
PACK=${1:?pack}; shift; T=$(mktemp -d)
KYOJIN_REPLAY=0 ${LK[@]+"${LK[@]}"} python tools/replay_e2e.py "$PACK" $T/off.json "$@" || exit 1
KYOJIN_REPLAY=1 ${LK[@]+"${LK[@]}"} python tools/replay_e2e.py "$PACK" $T/on.json "$@" || exit 1
python - $T/off.json $T/on.json <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = all(a[k][f] == b[k][f] for k in ("solo", "rows") for f in a[k] if f != "seconds")
print("replay off vs on: tokens, proposals, confidences, rings", "IDENTICAL" if ok else "DIFFERENT", "| solo decode seconds off %.2f on %.2f" % (a["solo"]["seconds"], b["solo"]["seconds"]))
sys.exit(0 if ok else 1)
PY
