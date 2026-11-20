#!/bin/bash
# Runs the RUN lines of the tests, as the fixture runner of Vx runs them, from the root of this
# repository. A python3 with the packages of requirements-test.txt must be on PATH. FileCheck of
# LLVM on PATH comes first, else the `filecheck` of those packages.
#
#   tests/run.sh              # the Python tests of the scripts, and every test here
#   tests/run.sh FILE...      # only these tests
#
# vxc and the machine files come from the Vx toolchain in VX_HOME/current, where VX_HOME is the
# folder of the Vx installer (`curl -fsSL https://vxlang.org/install.sh | VX_HOME=DIR sh`). Without
# VX_HOME, a vxc that is already on PATH comes first, with the machine files of its toolchain, and
# then the toolchain in ~/.vx/current. VX_FLEET names another folder of machine files, such as the
# fleet/ folder of a Vx checkout.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -u
cd "$(dirname "$0")/.." || exit 2
if [ -n "${VX_HOME:-}" ]; then
  toolchain=$VX_HOME/current
  # VX_HOME may also name a toolchain folder itself, as run.sh once took it.
  [ -d "$toolchain" ] || toolchain=$VX_HOME
  PATH="$toolchain/bin:$PATH"
elif command -v vxc > /dev/null; then
  toolchain=$(cd "$(dirname "$(readlink -f "$(command -v vxc)")")/.." && pwd)
else
  toolchain=$HOME/.vx/current
  PATH="$toolchain/bin:$PATH"
fi
fleet=$toolchain/fleet
# A vxc that a Vx checkout built has no fleet/ beside it.
[ -d "$fleet" ] || fleet=$HOME/.vx/current/fleet
export VX_FLEET=${VX_FLEET:-$fleet}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# The RUN lines call FileCheck. The Python port has the same options, under another name.
if ! command -v FileCheck > /dev/null && command -v filecheck > /dev/null; then
  mkdir "$tmp/bin"
  ln -s "$(command -v filecheck)" "$tmp/bin/FileCheck"
  PATH="$tmp/bin:$PATH"
fi
for tool in vxc FileCheck python3; do
  command -v "$tool" > /dev/null || { echo "run.sh: $tool is not on PATH"; exit 2; }
done
[ -f "$VX_FLEET/m4-uma.vx" ] || { echo "run.sh: no $VX_FLEET/m4-uma.vx. Set VX_HOME or VX_FLEET"; exit 2; }
failed=0
if [ $# -eq 0 ]; then
  set -- tests/*.vx
  if output=$(python3 -m unittest discover -s scripts -p 'test_*.py' 2>&1); then
    echo "PASS the Python tests of scripts/"
  else
    echo "FAIL the Python tests of scripts/"
    echo "$output" | tail -20 | sed 's/^/    /'
    failed=1
  fi
fi
export VX_NO_AST_FALLBACK=1
for test in "$@"; do
  n=0
  while IFS= read -r line; do
    n=$((n + 1))
    cmd=${line//%s/$test}
    cmd=${cmd//%S/$(dirname "$test")}
    cmd=${cmd//%t/$tmp/$(basename "$test" .vx)}
    if bash -o pipefail -c "$cmd" > "$tmp/output" 2>&1; then
      echo "PASS $test, RUN line $n"
    else
      echo "FAIL $test, RUN line $n: $line"
      tail -20 "$tmp/output" | sed 's/^/    /'
      failed=1
    fi
  done < <(sed -n 's|^// RUN: ||p' "$test")
done
exit $failed
