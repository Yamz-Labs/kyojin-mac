#!/bin/bash
# day2 lever gate for the lead, on the Ultra with the real DS4.1 pack. An arm = every lever OFF + that one lever ON.
#   1. served, ONE resident server per depth (tools/day2_served.py): the levers switch through POST /admin/flags, no reload between arms.
#      Greedy md5 of 4 x 128-token replies (prose + code, 2 reps) per arm vs the all-off arm, at depth 0 and 3 (a closing off arm checks that
#      switching back restores the bits); at depth 3 also one tok/s A/B all-off vs all-on (golive prompts, 4 prose + 4 code, 256 tokens),
#      then a second depth-3 server started with the Engram gate ON: rows "engram" (levers off) and "on+engram" (the real all-on).
#   2. Engram gate (not switchable at runtime: MLX buffer env read at mlx import): fresh processes tools/served_ds41.py DEPTH 128 PACK, gate 0 vs 1.
#   3. logits, one fresh process per arm: tools/dump_logits.py with a long random prompt (prefill levers run, one prefill call = PROMPT rows <= 8192)
#      -> bitwise --cmp vs all-off; when not bitwise (always for the moefastdec class, fp8-like fast decode) --kl: mean KL(off || arm) <= KL_MAX.
# Every real-pack run holds /tmp/ultra_heavy.lock (one heavy job at a time on the box).
# usage: tools/day2_gate.sh [--dry-run] [--only arm1,arm2] [--no-md5] [--no-logits] [--no-served] [--no-engram]
#   env: MODEL (default ~/ds41/model)  OUT (/tmp/day2_gate)  PROMPT (4096)  STEPS (64)  KL_MAX (0.02)  PORT (8391)
#        PY (~/venv-mlx/bin/python)  VLM (TREE/vlm_pkg)  ON_EXTRA (JSON flags added to the served all-on arm, e.g. '{"pf_split_gu": 1}')
#        ALL_ON_EXTRA (same for the offline logits all-on arm, as VAR=VAL, e.g. "KYOJIN_PF_SPLIT_GU=1")
# Output: OUT/summary.tsv (arm, check, result) and one log per run in OUT/.
set -u
TREE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$HOME/ds41/model}"; OUT="${OUT:-/tmp/day2_gate}"; PROMPT="${PROMPT:-4096}"; STEPS="${STEPS:-64}"
KL_MAX="${KL_MAX:-0.02}"; PORT="${PORT:-8391}"; PY="${PY:-$HOME/venv-mlx/bin/python}"; VLM="${VLM:-$TREE/vlm_pkg}"; LOCK=/tmp/ultra_heavy.lock
DRY=0; ONLY=""; DO_MD5=1; DO_LOG=1; DO_SRV=1; DO_ENG=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;; --only) ONLY="$2"; shift ;; --no-md5) DO_MD5=0 ;; --no-logits) DO_LOG=0 ;; --no-served) DO_SRV=0 ;; --no-engram) DO_ENG=0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
[ "$PROMPT" -le 8192 ] || { echo "PROMPT must be <= 8192 rows" >&2; exit 2; }

# Every lever explicitly OFF (default-ON ones included: Engram gate, DSPOL, bound layers, indexer query blocks).
ALL_OFF="KYOJIN_ENGRAM_GATE=0 KYOJIN_GLUE_FUSE=0 KYOJIN_GLUE_FUSE_NORM=0 KYOJIN_IDX_FAST=0 KYOJIN_MOE_FASTDEC=0 KYOJIN_MOEROOF=0 KYOJIN_DENSE_FAST=0 KYOJIN_MOEFUSE=0
 KYOJIN_PF_GLU=0 KYOJIN_PF_SPLIT_GU=0 KYOJIN_DSPOL=0 KYOJIN_BOUND_LAYERS=0 KYOJIN_IDX_QBLOCK=0 KYOJIN_PFROOF2=0"
# All-on = every lever a branch kept; opt-ins measured slower (PF_SPLIT_GU, GLUE_FUSE_NORM) only via ALL_ON_EXTRA.
ALL_ON="KYOJIN_ENGRAM_GATE=1 KYOJIN_GLUE_FUSE=1 KYOJIN_IDX_FAST=1 KYOJIN_MOE_FASTDEC=1 KYOJIN_DENSE_FAST=1 KYOJIN_MOEFUSE=1 KYOJIN_PF_GLU=1
 KYOJIN_DSPOL=1 KYOJIN_BOUND_LAYERS=1 KYOJIN_IDX_QBLOCK=512 ${ALL_ON_EXTRA:-}"
ARMS="engram glue gluenorm idxfast moefastdec moeroof moeroof1 moeroofnopf moeroof+fastdec densefast moefuse pfglu pfsplit dspol bound qblock pfroof2 on"
arm_env() {   # lever arm -> its ON switches (applied after ALL_OFF, so they win)
  case "$1" in
    off) echo "" ;; engram) echo "KYOJIN_ENGRAM_GATE=1" ;; glue) echo "KYOJIN_GLUE_FUSE=1" ;;
    gluenorm) echo "KYOJIN_GLUE_FUSE=1 KYOJIN_GLUE_FUSE_NORM=1" ;; idxfast) echo "KYOJIN_IDX_FAST=1" ;; moefastdec) echo "KYOJIN_MOE_FASTDEC=1" ;;
    moeroof) echo "KYOJIN_MOEROOF=1" ;; moeroof1) echo "KYOJIN_MOEROOF=1 KYOJIN_MOEROOF_DEC=1" ;; moeroofnopf) echo "KYOJIN_MOEROOF=1 KYOJIN_MOEROOF_PF=0" ;;
    moeroof+fastdec) echo "KYOJIN_MOEROOF=1 KYOJIN_MOE_FASTDEC=1" ;;
    densefast) echo "KYOJIN_DENSE_FAST=1" ;; moefuse) echo "KYOJIN_MOEFUSE=1" ;; pfglu) echo "KYOJIN_PF_GLU=1" ;; pfsplit) echo "KYOJIN_PF_SPLIT_GU=1" ;;
    dspol) echo "KYOJIN_DSPOL=1" ;; bound) echo "KYOJIN_BOUND_LAYERS=1" ;; qblock) echo "KYOJIN_IDX_QBLOCK=512" ;; pfroof2) echo "KYOJIN_PFROOF2=1 KYOJIN_IDX_QBLOCK=512" ;; on) echo $ALL_ON ;;
    *) echo "BAD" ;;
  esac
}
always_kl() { case "$1" in moefastdec|moeroof+fastdec|on) return 0 ;; *) return 1 ;; esac; }   # not bit-exact by design (fast trellis decode)
KNOWN="$ARMS"; if [ -n "$ONLY" ]; then ARMS="$(echo "$ONLY" | tr ',' ' ')"; fi
for a in $ARMS; do [ "$(arm_env "$a")" != BAD ] || { echo "unknown arm $a (known: off $KNOWN)" >&2; exit 2; }; done

mkdir -p "$OUT"; SUM="$OUT/summary.tsv"; [ "$DRY" = 1 ] || : > "$SUM"
note() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$SUM"; }
# run NAME ARM CMD... : the command under the heavy lock, in the tree, with ALL_OFF + the arm switches; log in OUT/NAME.log
run() {
  local name="$1" arm="$2"; shift 2
  local envs; envs="$(echo $ALL_OFF) $(arm_env "$arm")"
  if [ "$DRY" = 1 ]; then echo "[dry] flock $LOCK env PYTHONPATH=.:$VLM $envs $* > $OUT/$name.log"; return 0; fi
  ( cd "$TREE" && flock "$LOCK" env PYTHONPATH=".:$VLM" KYOJIN_VLM_PKG="$VLM" $envs "$@" ) > "$OUT/$name.log" 2>&1
}
md5s() { grep "tokens md5" "$OUT/$1.log" 2>/dev/null | awk '{print $3}' | tr '\n' ' '; }

[ "$DRY" = 1 ] || [ -d "$MODEL" ] || { echo "no pack at $MODEL" >&2; exit 2; }
echo "tree $TREE  model $MODEL  out $OUT  arms: off $ARMS"

# resident-server arms: the runtime ones (every arm but engram; the all-on arm is the served A/B)
SRV_ARMS=""; for a in $ARMS; do case "$a" in engram|on) ;; *) SRV_ARMS="$SRV_ARMS,$a" ;; esac; done; SRV_ARMS="${SRV_ARMS#,}"; SRV_EXTRA="${ON_EXTRA:-}"; [ -n "$SRV_EXTRA" ] || SRV_EXTRA="{}"
for d in 0 3; do
  want_md5=0; want_srv=0
  [ "$DO_MD5" = 1 ] && [ -n "$SRV_ARMS" ] && want_md5=1
  [ "$d" = 3 ] && [ "$DO_SRV" = 1 ] && want_srv=1
  [ $want_md5 = 1 ] || [ $want_srv = 1 ] || continue
  opts=""; [ $want_md5 = 1 ] || opts="--no-md5"; [ $want_srv = 1 ] || opts="$opts --no-served"
  run "served_d$d" off "$PY" tools/day2_served.py "$TREE" "$MODEL" "$PORT" --depth "$d" --arms "${SRV_ARMS:-glue}" --vlm "$VLM" \
    --on-extra "$SRV_EXTRA" --log "$OUT/server_d$d.log" $opts
  [ "$DRY" = 1 ] && continue
  grep -qE '^(MD5|SERVED) ' "$OUT/served_d$d.log" || { note served "d$d" "ERROR (see $OUT/served_d$d.log)"; continue; }
  grep '^MD5 ' "$OUT/served_d$d.log" | while read -r _ a r rest; do note "$a" "md5 d$d resident" "$r $rest"; done
  grep '^SERVED ' "$OUT/served_d$d.log" | while read -r _ a rest; do note "$a" "served d$d" "$rest"; done
done
# the Engram gate is fixed at server start: one more depth-3 server with it ON gives "engram" (levers off) and "on+engram" (the real all-on)
if [ "$DO_SRV" = 1 ]; then
  run served_eng_d3 engram "$PY" tools/day2_served.py "$TREE" "$MODEL" "$PORT" --depth 3 --no-md5 --vlm "$VLM" \
    --on-extra "$SRV_EXTRA" --log "$OUT/server_eng_d3.log"
  if [ "$DRY" != 1 ]; then
    grep -qE '^SERVED ' "$OUT/served_eng_d3.log" || note served "eng d3" "ERROR (see $OUT/served_eng_d3.log)"
    grep '^SERVED ' "$OUT/served_eng_d3.log" | while read -r _ a rest; do
      case "$a" in off) a=engram ;; on) a=on+engram ;; esac; note "$a" "served d3" "$rest"; done
  fi
fi

case " $ARMS " in *" engram "*) has_eng=1 ;; *) has_eng=0 ;; esac
if [ "$DO_ENG" = 1 ] && [ "$DO_MD5" = 1 ] && [ $has_eng = 1 ]; then
  for d in 0 3; do
    run "md5_off_d$d" off "$PY" tools/served_ds41.py "$d" 128 "$MODEL"
    run "md5_engram_d$d" engram "$PY" tools/served_ds41.py "$d" 128 "$MODEL"
    [ "$DRY" = 1 ] && continue
    A="$(md5s md5_off_d$d)"; B="$(md5s md5_engram_d$d)"
    if [ "$(echo $A | wc -w)" -ne 4 ] || [ "$(echo $B | wc -w)" -ne 4 ]; then note engram "md5 d$d" "ERROR (see $OUT/md5_*_d$d.log)"
    elif [ "$A" = "$B" ]; then note engram "md5 d$d" "EQUAL"; else note engram "md5 d$d" "DIFF off=[$A] arm=[$B]"; fi
  done
fi

if [ "$DO_LOG" = 1 ]; then
  run logits_off off "$PY" tools/dump_logits.py "$MODEL" "$OUT/logits_off.npz" --prompt "$PROMPT" --steps "$STEPS"
  for a in $ARMS; do
    run "logits_$a" "$a" "$PY" tools/dump_logits.py "$MODEL" "$OUT/logits_$a.npz" --prompt "$PROMPT" --steps "$STEPS"
    [ "$DRY" = 1 ] && { always_kl "$a" && echo "[dry] $PY tools/dump_logits.py --kl off $a $KL_MAX"; continue; }
    [ -f "$OUT/logits_$a.npz" ] && [ -f "$OUT/logits_off.npz" ] || { note "$a" logits "ERROR (see $OUT/logits_$a.log)"; continue; }
    C="$("$PY" "$TREE/tools/dump_logits.py" --cmp "$OUT/logits_off.npz" "$OUT/logits_$a.npz")"; rc=$?
    note "$a" "logits bitwise" "$C"
    if [ $rc -ne 0 ] || always_kl "$a"; then
      K="$("$PY" "$TREE/tools/dump_logits.py" --kl "$OUT/logits_off.npz" "$OUT/logits_$a.npz" "$KL_MAX")"; note "$a" "logits KL" "$K"
    fi
  done
fi

[ "$DRY" = 1 ] || echo "summary: $SUM"
