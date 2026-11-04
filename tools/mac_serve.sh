#!/usr/bin/env bash
# Multi-session OpenAI-compatible server on a Mac (continuous batching, prefix sharing, chunked prefill).
#   tools/mac_serve.sh MODEL_DIR [--sessions 4] [--port 8080] [--ctx 32768] [--prefix-cache-mb 1024] [--thinking on] ...
#   tools/mac_serve.sh --help          all options
#   --mtp-depth D    speculative decode inside the batch (DS4.1 packs with mtp.* tensors): every active session adds its pending token and up to D
#                    drafts to one verify pass, the depth adapts to each session's acceptance (0 allowed). 0 (default) = plain batched decode.
#   --kv-budget GiB  memory for session caches (active + parked) and prefix snapshots; over it the oldest parked session's KV is dropped and rebuilt
#                    when it resumes. Default: free unified memory after the load minus 1 GiB.
# MODEL_DIR: an EXL3 Qwen3.5 pack (tested on the Air with Qwen3.5-2B 2.10 bpw), or a Kyojin DS4.1 / GLM-5.3 pack (config.json model_type
# deepseek_v41 / glm5_next; needs mlx-vlm 0.7.10 in ./vlm_pkg; pip install mlx-vlm==0.7.10 or point KYOJIN_VLM_PKG at a copy). Listens on 127.0.0.1 unless --host is given.
# Keeps the Mac awake while it runs (caffeinate -dimsu watches this process). Stop it with Ctrl-C or kill <pid>.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in "$HERE/.venv/bin/python" "$HERE/../../.venv/bin/python" "$HERE/../.venv/bin/python"; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
[ -n "$PY" ] || PY=python3
VLM=""
for d in "${KYOJIN_VLM_PKG:-}" "$HERE/vlm_pkg" "$HERE/../../vlm_pkg"; do
  [ -n "$d" ] && [ -d "$d/mlx_vlm" ] && VLM="$d" && break
done
export PYTHONPATH="$HERE${VLM:+:$VLM}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
[ "${KYOJIN_MALLOC_EFFICIENT:-0}" = 1 ] && export MallocSpaceEfficient=1   # GGUF load: freed host pack buffers stay counted in the footprint (26 -> 19 GB on Qwen3.8-27B Q4_K_M)
if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w $$ &
fi
exec "$PY" -m kyojin_mlx.serve.server "$@"
