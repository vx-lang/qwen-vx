#!/bin/bash
# The benchmark of the Qwen3.6 app against llama.cpp on an M4, as docs/benchmarking_policy.md of
# the Vx repository asks. Run it under `caffeinate -dims`, on a Mac with no other work. It writes
# logs/, log.txt and env.txt into OUT. Then `parse.py OUT` writes runs.csv and summary.md.
#
#   1. Flash attention of llama.cpp: on and off at the 2,048-token prompt. The later stages use
#      the setting with the faster median.
#   2. llama-bench: pp512, pp2048 and pp8192, and tg128 after 0 to 32,768 tokens, 5 repetitions.
#   3. Table A: 256 generated tokens after 64 to 32,768 prompt tokens.
#   4. Table P: the prompts of 512 to 8,192 tokens, PROMPT_RUNS times in one process. The first
#      run includes the first launch of each kernel. The runs after it show the time in a session.
#   5. Table T: the draft lengths, on the 12 tuning prompts, once each. The app tries each length of
#      DRAFTS, and llama-speculative-simple with the MTP heads each of MTP_DRAFTS. Table C uses the
#      length with the fastest geometric mean. llama-speculative-simple with n-gram drafts
#      (ngram-mod and ngram-cache) keeps the settings of llama.cpp: ngram-cache always drafts 8
#      tokens. llama-lookup is not in the benchmark: on this model its text differs from that
#      of llama-completion, and its log has errors from llama_decode.
#   6. Table C: the 12 held-out prompts, 512 generated tokens. The programs are the app, the app
#      with --draft, llama-completion, and llama-speculative-simple with ngram-cache, ngram-mod
#      and the MTP heads.
#
# Each case runs ROUNDS times. Round 2 runs the programs in the reverse order. Before each run, the
# script waits until no other process uses more than 20% of a CPU and the GPU is idle, and then
# waits COOL seconds more. A run takes about 1 to 2 minutes, so all the stages take about 10 hours.
# Build llama.cpp from its commit b1b770d with llama-timing.patch. Without the patch,
# llama-speculative-simple stops the clock of the prompt before the GPU ends the prompt, and counts
# the rest of the prompt in the time of the generated tokens.
# STAGES selects the stages: fa, bench, A, P, T and C. Without the stage fa, give FA=on or FA=off.
# Without the stage T, give DRAFT and MTP_N. A short test can also set CONTEXTS,
# P_CONTEXTS, PROMPTS, A_TOKENS, TOKENS and the lists of draft lengths. The report uses the defaults.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/out}
ROUNDS=${ROUNDS:-3}
COOL=${COOL:-30}
STAGES=${STAGES:-fa bench A P T C}
DRAFTS=${DRAFTS:-3 5 7 9 11 15}
MTP_DRAFTS=${MTP_DRAFTS:-1 2 3 4 6}
PROMPT_RUNS=${PROMPT_RUNS:-3}
CONTEXTS=${CONTEXTS:-64 512 2048 8192 16384 32768}
P_CONTEXTS=${P_CONTEXTS:-512 2048 8192}
PROMPTS=${PROMPTS:-edit1 complete1 write1 explain1 edit2 complete2 write2 explain2 edit3 complete3 write3 explain3}
A_TOKENS=${A_TOKENS:-256}
TOKENS=${TOKENS:-512}
VXC=${VXC:?the release vxc}
FLEET=${FLEET:?the fleet/ folder of that vxc}
QV=${QV:-$(cd "$HERE/../../.." && pwd)}
VXW=${VXW:?the weight file of the app}
VXT=${VXT:?the tokenizer file of the app}
GGUF=${GGUF:?the GGUF file}
MTP=${MTP:?the GGUF file of the MTP heads}
LB=${LB:?the bin/ folder of llama.cpp}
LLAMA_SRC=${LLAMA_SRC:-$LB/../..}
P=$HERE/prompts
want() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$OUT/logs"
cd "$OUT" || exit 1

gpu_busy() { # the Device Utilization % of the GPU
  ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2
}
quiet() {
  while ps -Ao %cpu,comm -r | awk 'NR>1 && $2 !~ /claude|Terminal|WindowServer|kernel_task/ && $1 > 20 {found=1} END {exit !found}' \
    || [ "$(gpu_busy)" -gt 5 ]; do
    sleep 10
  done
  sleep "$COOL"
}
busy() { ps -Ao %cpu,comm -r | awk 'NR>1 && $2 !~ /claude|Terminal/ {printf "%s %s", $1, $2; exit}' | sed 's|.*/||'; }

# Each program prints its whole command line first, so that the log has it.
show() {
  echo "command: $*"
  "$@"
}
vx() { # PROMPT STEPS [options]
  local f=$1 n=$2
  shift 2
  show "$VXC" -O2 "$QV/src/main.vx" --package qwen="$QV/src" --host default --machine "$FLEET/m4-uma.vx" -- \
    "$VXW" --tokenizer "$VXT" --prompt-file "$f" --steps "$n" --ignore-eos "$@"
}
llama() { # TOOL PROMPT N CTX [options]
  local tool=$1 f=$2 n=$3 c=$4
  shift 4
  show "$LB/$tool" -m "$GGUF" -f "$f" -n "$n" -c "$c" --temp 0 -ngl 99 -fa "$FA" --no-escape --ignore-eos "$@"
}
completion() { llama llama-completion "$@" -no-cnv; }
# llama-speculative-simple refuses a prompt longer than the batch (-b, 2,048 by default). The
# longest prompt has 2,251 tokens.
spec() { llama llama-speculative-simple "$@" -b 4096; }

run() { # NAME COMMAND...: one run, its output in logs/NAME.txt, a line in log.txt
  local name=$1
  shift
  quiet
  local b g t0
  b=$(busy)
  g=$(gpu_busy)
  t0=$(date +%Y-%m-%dT%H:%M:%S)
  "$@" > "logs/$name.txt" 2>&1
  local rc=$?
  echo "$name start=$t0 end=$(date +%Y-%m-%dT%H:%M:%S) rc=$rc busy=[$b] gpu=$g%" | tee -a log.txt
}

{
  echo "date: $(date), stages: $STAGES"
  echo "machine: $(sysctl -n hw.model), $(sysctl -n machdep.cpu.brand_string), $(($(sysctl -n hw.memsize) / 1073741824)) GB"
  system_profiler SPDisplaysDataType 2> /dev/null | grep -E "Chipset Model|Total Number of Cores|Metal Support"
  echo "macOS: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "power: $(pmset -g batt | head -1)"
  echo "vxc: $("$VXC" --version 2>&1)"
  echo "qwen-vx: $(git -C "$QV" log -1 --format='%H %s'), changed files: $(git -C "$QV" status --short -- src | wc -l | tr -d ' ')"
  echo "llama.cpp: $("$LB/llama-completion" --version 2>&1 | tr '\n' ' ')"
  echo "llama.cpp source: $(git -C "$LLAMA_SRC" log -1 --format=%H), changes: $(git -C "$LLAMA_SRC" diff --stat | tail -1)"
  git -C "$LLAMA_SRC" diff > llama.cpp.diff
  echo "GGUF: $GGUF $(stat -f %z "$GGUF") bytes"
  echo "MTP GGUF: $MTP $(stat -f %z "$MTP") bytes"
  echo "app weights: $VXW $(stat -f %z "$VXW") bytes"
  echo "app tokenizer: $VXT $(stat -f %z "$VXT") bytes"
  echo "rounds: $ROUNDS, cool: $COOL s, prompt runs: $PROMPT_RUNS"
  echo "contexts: $CONTEXTS, prompt contexts: $P_CONTEXTS, generated tokens: $A_TOKENS and $TOKENS"
  echo "prompts: $PROMPTS"
  echo "draft lengths tried: app $DRAFTS, MTP $MTP_DRAFTS"
} >> env.txt

# The order of the programs in round r: as given in odd rounds, reversed in even rounds.
order() { # R NAME...
  local r=$1
  shift
  if [ $((r % 2)) = 1 ]; then echo "$@"; else echo "$@" | tr ' ' '\n' | tail -r | tr '\n' ' '; fi
}

# 1. Flash attention of llama.cpp
if want fa; then
  for r in $(seq 1 "$ROUNDS"); do
    for fa in $(order "$r" on off); do
      FA=$fa run "fa-$fa.ctx2048.r$r" completion "$P/context/llama/ctx2048.txt" 256 4096
    done
  done
  ms() { grep 'common_perf_print: *eval time' "$1" | sed -E 's/.*\( *([0-9.]+) ms per token.*/\1/'; }
  med() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
  on=$(for f in logs/fa-on.ctx2048.r*.txt; do ms "$f"; done | med)
  off=$(for f in logs/fa-off.ctx2048.r*.txt; do ms "$f"; done | med)
  FA=$(awk -v a="$on" -v b="$off" 'BEGIN {print (a <= b) ? "on" : "off"}')
  echo "flash attention: on $on ms, off $off ms (medians at 2,048 tokens); chosen: $FA" | tee -a log.txt env.txt
fi
FA=${FA:?give FA=on or FA=off, or run the stage fa}
want fa || echo "flash attention: given by FA; chosen: $FA" | tee -a log.txt env.txt

# 2. llama-bench
want bench && run llama-bench-pp show "$LB/llama-bench" -m "$GGUF" -p 512,2048,8192 -n 0 -r 5 -ngl 99 -fa "$FA" -o md
want bench && run llama-bench-tg show "$LB/llama-bench" -m "$GGUF" -p 0 -n 128 -d 0,2048,8192,16384,32768 -r 5 -ngl 99 -fa "$FA" -o md

# 3. Table A
want A && for r in $(seq 1 "$ROUNDS"); do
  for c in $CONTEXTS; do
    for tool in $(order "$r" vx llama); do
      if [ "$tool" = vx ]; then
        run "A.vx.ctx$c.r$r" vx "$P/context/vx/ctx$c.txt" "$A_TOKENS"
      else
        run "A.llama.ctx$c.r$r" completion "$P/context/llama/ctx$c.txt" "$A_TOKENS" $((c + 512))
      fi
    done
  done
done

# 4. Table P
want P && for r in $(seq 1 "$ROUNDS"); do
  for c in $P_CONTEXTS; do
    for tool in $(order "$r" vx llama); do
      if [ "$tool" = vx ]; then
        run "P.vx.ctx$c.r$r" vx "$P/context/vx/ctx$c.txt" 1 --prompt-runs "$PROMPT_RUNS"
      else
        run "P.llama.ctx$c.r$r" completion "$P/context/llama/ctx$c.txt" 1 $((c + 512))
      fi
    done
  done
done

# 5. Table T
if want T; then
  cases="vx completion ngram cache"
  for d in $DRAFTS; do cases="$cases vxdraft$d"; done
  for d in $MTP_DRAFTS; do cases="$cases mtp$d"; done
  r=0
  for n in $PROMPTS; do
    r=$((r + 1))
    for kind in $(order "$r" $cases); do
      f=$P/twelve/llama/$n.txt
      case $kind in
        vx) run "T.$n.r1.vx" vx "$P/twelve/vx/$n.txt" "$TOKENS" ;;
        vxdraft*) run "T.$n.r1.$kind" vx "$P/twelve/vx/$n.txt" "$TOKENS" --draft "${kind#vxdraft}" ;;
        completion) run "T.$n.r1.completion" completion "$f" "$TOKENS" 8192 ;;
        ngram) run "T.$n.r1.ngram" spec "$f" "$TOKENS" 8192 --spec-type ngram-mod ;;
        mtp*) run "T.$n.r1.$kind" spec "$f" "$TOKENS" 8192 --spec-type draft-mtp -md "$MTP" -ngld 99 --spec-draft-n-max "${kind#mtp}" ;;
        cache) run "T.$n.r1.cache" spec "$f" "$TOKENS" 8192 --spec-type ngram-cache ;;
      esac
    done
  done
  choice=$(python3 "$HERE/parse.py" "$OUT" --choose) || exit 1
  eval "$choice"
  echo "draft lengths chosen in table T: $choice" | tee -a log.txt env.txt
else
  echo "draft lengths given: DRAFT=${DRAFT:?} MTP_N=${MTP_N:?}" | tee -a log.txt env.txt
fi

# 6. Table C
want C && for r in $(seq 1 "$ROUNDS"); do
  for n in $PROMPTS; do
    f=$P/heldout/llama/$n.txt
    for kind in $(order "$r" vx vxdraft completion cache ngram mtp); do
      case $kind in
        vx) run "C.$n.r$r.vx" vx "$P/heldout/vx/$n.txt" "$TOKENS" ;;
        vxdraft) run "C.$n.r$r.vxdraft$DRAFT" vx "$P/heldout/vx/$n.txt" "$TOKENS" --draft "$DRAFT" ;;
        completion) run "C.$n.r$r.completion" completion "$f" "$TOKENS" 8192 ;;
        cache) run "C.$n.r$r.cache" spec "$f" "$TOKENS" 8192 --spec-type ngram-cache ;;
        ngram) run "C.$n.r$r.ngram" spec "$f" "$TOKENS" 8192 --spec-type ngram-mod ;;
        mtp) run "C.$n.r$r.mtp$MTP_N" spec "$f" "$TOKENS" 8192 --spec-type draft-mtp -md "$MTP" -ngld 99 --spec-draft-n-max "$MTP_N" ;;
      esac
    done
  done
done
echo "done $(date +%Y-%m-%dT%H:%M:%S)" | tee -a log.txt
