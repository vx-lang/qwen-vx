#!/bin/bash
# Compiles and runs each program the model wrote, and compares its output with the known answer.
#
#   check.sh OUT_DIR [SUFFIX]
#
# The program of task NAME is in OUT_DIR/NAME$SUFFIX/code.*. A task with no folder is skipped.
# It needs rustc, clang++, python3 and `timeout` (in coreutils on a Mac).
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

out=$1
suffix=${2:-}
pass=0; total=0
rust="rustc -O --edition 2021 -o prog code.rs"
check() {  # name, build command, run command, stdin, expected
  local name=$1 build=$2 run=$3 input=$4 expected=$5 d=$out/$1$suffix
  [ -d "$d" ] || return
  total=$((total + 1))
  if ! ls "$d"/code.* > /dev/null 2>&1; then
    echo "FAIL $name$suffix: no code block"; return
  fi
  if ! (cd "$d" && eval "$build") > "$d/build.log" 2>&1; then
    echo "FAIL $name$suffix: does not compile"
    grep -E "^error" "$d/build.log" | head -3 | sed 's/^/    /'; return
  fi
  local got
  got=$(cd "$d" && printf '%s' "$input" | eval "timeout 60 $run" 2>&1)
  if [ "$got" == "$expected" ]; then
    echo "PASS $name$suffix"; pass=$((pass + 1))
  else
    echo "FAIL $name$suffix: wrong output"
    echo "    expected: $(echo "$expected" | tr '\n' '|')"
    echo "    got:      $(echo "$got" | tr '\n' '|')"
  fi
}
check rust_wordfreq "$rust" "./prog" "The cat and the dog, and THE bird; a cat." $'the 3\nand 2\ncat 2'
check rust_bst "$rust" "./prog" "" $'20 30 40 50 60 70 80\ntrue\nfalse'
check rust_csv "$rust" "./prog" $'name,age,city\nAna,34,Lisbon\nBo,27,Oslo\nbad line\nCy,34,Cairo\nDi,x,Rome\n' \
  $'Ana (34) from Lisbon\nCy (34) from Cairo\nBo (27) from Oslo'
check rust_lexer "$rust" "./prog" "" \
  $'Ident("let")\nIdent("total")\nOp(\'=\')\nIdent("price")\nOp(\'*\')\nNumber(3)\nOp(\'+\')\nIdent("tax_rate")\nOp(\';\')'
check cpp_lru "clang++ -std=c++17 -O2 -Wall -Werror -o prog code.cpp" "./prog" "" $'one\nmiss\nthree'
check cpp_primes "clang++ -std=c++17 -O2 -Wall -Werror -pthread -o prog code.cpp" "./prog" "" "78498"
check python_nqueens "python3 -m py_compile code.py" "python3 code.py" "" "1 0 0 2 10 4 40 92 352 724"
echo "$pass of $total passed"
