#!/bin/zsh
# Served gate of the cloud/mtppolicy levers (KYOJIN_MTP_POLICY=robust, KYOJIN_GATED_VERIFY=1), one server at a time, through tools/golive_bench.py:
#   - 4 greedy replies of 128 tokens (GOLIVE_MD5): their md5 with a lever ON must equal the lever-OFF run at the same depth (D auto vs D=0 reported);
#   - decode tok/s prose/code (4 + 4 Hermes-style requests) for D auto (--depth 3, the engine's depth policy) and D=0, levers OFF and ON;
#   - a KYOJIN_ENGRAM_POISON=1 run with both levers ON: a verify pass committed before its Engram rows are in turns into NaN, so the md5 moves.
# usage: tools/mtppolicy_gate.sh [PACK] [PORT]   (default ~/ds41/model 8391; GATE_DRY=1 prints the runs only)
# Needs the box quiet: each run waits until no other Kyojin server is left (180 s max) and starts with an empty kv-disk. Prefill chunk 2048 (golive_bench), ctx 8192.
set -e
tree=${0:A:h:h}
pack=${1:-$HOME/ds41/model}
port=${2:-8391}
PY=${PY:-$HOME/venv-mlx/bin/python}
out=${GATE_OUT:-$tree/results_mac/mtppolicy}
mkdir -p $out
export KYOJIN_VLM_PKG=${KYOJIN_VLM_PKG:-$tree/vlm_pkg}
export GOLIVE_MD5=1
quiet() {                                                     # one heavy job at a time: wait for the previous server to be gone (golive only sends SIGTERM)
  local t=0
  while pgrep -f "[k]yojin_mlx.serve.server" >/dev/null; do
    t=$((t + 2)); [ $t -gt 180 ] && { echo "a Kyojin server is still running after 180 s: one heavy job at a time"; exit 2; }
    sleep 2
  done
}
# name depth env...
runs=(
  "d0_off 0"
  "dauto_off 3"
  "dauto_robust 3 KYOJIN_MTP_POLICY=robust"
  "dauto_gatedv 3 KYOJIN_GATED_VERIFY=1"
  "dauto_both 3 KYOJIN_MTP_POLICY=robust KYOJIN_GATED_VERIFY=1"
  "d0_both 0 KYOJIN_MTP_POLICY=robust KYOJIN_GATED_VERIFY=1"
  "dauto_both_poison 3 KYOJIN_MTP_POLICY=robust KYOJIN_GATED_VERIFY=1 KYOJIN_ENGRAM_POISON=1"
)
for r in $runs; do
  parts=(${=r}); name=$parts[1]; depth=$parts[2]; envs=(${parts[3,-1]})
  echo "=== $name depth=$depth ${envs[*]}"
  if [ -n "$GATE_DRY" ]; then
    echo "env ${envs[*]} $PY $tree/tools/golive_bench.py $tree $pack $port --depth $depth --n 256 > $out/$name.log"; continue
  fi
  quiet
  rm -rf /tmp/golive_kvdisk_$port                              # every run starts cold: no prefix restored from an earlier config
  env $envs $PY $tree/tools/golive_bench.py $tree $pack $port --depth $depth --n 256 > $out/$name.log 2>&1 || echo "  run failed (see $out/$name.log)"
  cp /tmp/golive_$port.log $out/$name.server.log 2>/dev/null || true
  grep "MEDIAN\|MD5ALL\|spec stats" $out/$name.log || true
done
[ -n "$GATE_DRY" ] && exit 0
quiet
md5() { grep MD5ALL $out/$1.log | awk '{print $2}'; }
fail=0
same() {                                                      # same NAME REF [info]: md5 of NAME equals REF's (info: reported, not a lever failure)
  local a=$(md5 $1) b=$(md5 $2)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "MD5 OK   $1 = $2"
  elif [ -n "$3" ]; then echo "MD5 DIFF $1 vs $2 (spec vs plain, not caused by the levers)"
  else echo "MD5 FAIL $1 vs $2 ($a vs $b)"; fail=1; fi
}
for n in dauto_robust dauto_gatedv dauto_both dauto_both_poison; do same $n dauto_off; done   # levers ON vs OFF at the same depth
same d0_both d0_off
same dauto_off d0_off info
for n in dauto_gatedv dauto_both dauto_both_poison; do                                        # the gated verify must have engaged, never failed
  if ! grep -q "gated verify: on" $out/$n.server.log 2>/dev/null; then echo "GATED FAIL $n: never engaged"; fail=1; fi
  if grep -q "gated verify failed" $out/$n.server.log 2>/dev/null; then echo "GATED FAIL $n: a fill failed"; fail=1; fi
done
echo "=== tok/s (MEDIAN prose / code)"
for r in $runs; do name=${${=r}[1]}; echo "$name: $(grep MEDIAN $out/$name.log | awk '{print $2, $3}' | tr '\n' ' ')"; done
exit $fail
