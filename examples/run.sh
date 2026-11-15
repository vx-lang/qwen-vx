#!/bin/bash
# Asks the model for each program in prompts/, runs the Vx program to write it, and checks it.
#
#   examples/run.sh WEIGHTS TOKENIZER_DIR [TASK...]
#
# WEIGHTS is a weight file of convert.py, such as the Q4_0 file of Qwen3-Coder-30B-A3B-Instruct.
# TOKENIZER_DIR is the Hugging Face folder of the same model. With SKILLS=1, the Rust tasks get
# RUST-SKILLS.md as the system message. The replies and the programs go in examples/out, or OUT.
# vxc and the machine files come from VX_HOME (default ~/.vx), and python3 needs `tokenizers`.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -eu
[ $# -ge 2 ] || { sed -n '4p' "$0"; exit 2; }
weights=$1 tokenizer=$2
shift 2
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
out=${OUT:-$here/out}
VX_HOME=${VX_HOME:-$HOME/.vx}
command -v vxc > /dev/null || PATH="$VX_HOME/bin:$PATH"
[ $# -gt 0 ] || set -- $(cd "$here/prompts" && ls *.txt | sed 's/\.txt$//')
suffix=""
[ -z "${SKILLS:-}" ] || suffix=_skills
for task in "$@"; do
  skills=""
  [ -n "${SKILLS:-}" ] && [[ $task == rust_* ]] && skills=$here/RUST-SKILLS.md
  [ -z "${SKILLS:-}" ] || [ -n "$skills" ] || continue
  ids=$(python3 "$here/harness.py" encode "$tokenizer" "$here/prompts/$task.txt" $skills)
  mkdir -p "$out/$task$suffix"
  vxc src/main.vx --package qwen=src --host default \
      --machine "$VX_HOME/current/fleet/m4-uma.vx" -- "$weights" --tokens "$ids" --steps 700 \
      2> "$out/$task$suffix/vxc.log" > "$out/$task$suffix/vx.txt"
  printf '%-24s ' "$task$suffix"
  python3 "$here/harness.py" decode "$tokenizer" "$out/$task$suffix/vx.txt" "$out/$task$suffix"
done
"$here/check.sh" "$out" "$suffix"
