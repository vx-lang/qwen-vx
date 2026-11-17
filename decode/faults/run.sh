#!/bin/bash
# run.sh OUT COMMAND... : runs COMMAND with sampler.dylib. Every process that COMMAND starts
# writes a sample each 200 ms to OUT/<pid>.tsv.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$1
shift
[ -f "$here/sampler.dylib" ] || clang -O2 -dynamiclib "$here/sampler.c" -o "$here/sampler.dylib"
mkdir -p "$out"
FAULTS_DIR=$(cd "$out" && pwd) DYLD_INSERT_LIBRARIES="$here/sampler.dylib" "$@"
