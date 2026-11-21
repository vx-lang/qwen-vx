#!/usr/bin/env python3
"""Bytes of weights that one generated token reads, in the app's weight file and in a GGUF file.

    python3 bytes_per_token.py APP.vxw MODEL.gguf QWEN_VX/scripts LLAMA/gguf-py

A token reads one row of the token embedding, the matrices of the 8 experts it routes to (of 256),
and every other tensor whole.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import re
import sys
from collections import defaultdict

vxw, gguf_path, qwen_scripts, gguf_py = sys.argv[1:5]
sys.path.insert(0, qwen_scripts)
sys.path.insert(0, gguf_py)
import qwen_weights  # noqa: E402
from gguf import GGUFReader  # noqa: E402

TOP, EXPERTS = 8, 256


def share(name, vocab):
    if re.search(r"(^|\.)(embed|token_embd)", name):
        return 1 / vocab
    if re.search(r"(experts|_exps)", name):
        return TOP / EXPERTS
    return 1.0


config, tensors = qwen_weights.read_weight_file(vxw)
vocab = config["vocab_size"]
print("config:", {k: config[k] for k in sorted(config) if not isinstance(config[k], (list, dict))})
sums = defaultdict(float)
for name, (_, array) in tensors.items():
    kind = re.sub(r"^layers\.\d+\.", "", name)
    sums[kind] += array.nbytes * share(name, vocab)
print(f"app: {sum(sums.values()) / 1e9:.3f} GB per token, file {sum(a.nbytes for _, a in tensors.values()) / 1e9:.3f} GB")
for k, v in sorted(sums.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {k}: {v / 1e6:.1f} MB")

sums = defaultdict(float)
ts = GGUFReader(gguf_path).tensors
for t in ts:
    kind = re.sub(r"^blk\.\d+\.", "", t.name)
    sums[kind] += t.n_bytes * share(t.name, vocab)
print(f"gguf: {sum(sums.values()) / 1e9:.3f} GB per token, file {sum(t.n_bytes for t in ts) / 1e9:.3f} GB")
for k, v in sorted(sums.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {k}: {v / 1e6:.1f} MB")
