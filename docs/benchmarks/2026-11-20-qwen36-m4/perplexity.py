#!/usr/bin/env python3
"""Perplexity of the two weight sets of the Qwen3.6 benchmark, with one program: scripts/reference.py.

    python3 perplexity.py APP.vxw MODEL.gguf QWEN_VX/scripts LLAMA/gguf-py HF_MODEL_DIR TEXT... [--max-tokens N]

The app's weight file and the GGUF hold the same Q4_0 matrices. They differ in the token
embedding (bf16 against Q4_0) and the output matrix (6-bit codes against Q6_K). The llama.cpp set
is the app's file with these two tensors taken from the GGUF, as float32. For each text, the script
prints the perplexity of each set, the mean KL divergence of the llama.cpp set from the app set,
and how often the two sets give the same most likely next token.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import argparse
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("vxw")
p.add_argument("gguf")
p.add_argument("qwen_scripts")
p.add_argument("gguf_py")
p.add_argument("hf_dir")
p.add_argument("texts", nargs="+")
p.add_argument("--max-tokens", type=int, default=512)
args = p.parse_args()
sys.path.insert(0, args.qwen_scripts)
sys.path.insert(0, args.gguf_py)
import reference  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

app = reference.Model(args.vxw)
llama = reference.Model(args.vxw)
theirs = {t.name: t for t in GGUFReader(args.gguf).tensors}
for ours, name in (("embed", "token_embd.weight"), ("lm_head", "output.weight")):
    t = theirs[name]
    values = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
    want = app.tensors[ours][1].shape[0], app.config["hidden_size"]
    values = values.reshape(want)
    llama.tensors[ours] = ("f32", values)
    print(f"{ours}: {t.tensor_type.name} from the GGUF, {values.shape}", flush=True)

tokenizer = Tokenizer.from_file(f"{args.hf_dir}/tokenizer.json")


def log_softmax(logits):
    logits = logits.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


for path in args.texts:
    tokens = tokenizer.encode(open(path).read()).ids[: args.max_tokens]
    out = {}
    for label, model in (("app", app), ("llama.cpp", llama)):
        start = time.time()
        out[label] = log_softmax(reference.forward(model, tokens, reference.Cache(model.config)))
        print(f"{path}: {label} forward of {len(tokens)} tokens in {time.time() - start:.0f} s", flush=True)
    next_tokens = np.asarray(tokens[1:])
    rows = np.arange(len(tokens) - 1)
    a, b = out["app"][:-1], out["llama.cpp"][:-1]
    ppl = {k: float(np.exp(-v[:-1][rows, next_tokens].mean())) for k, v in out.items()}
    kl = float((np.exp(a) * (a - b)).sum(axis=-1).mean())
    top1 = float((a.argmax(axis=-1) == b.argmax(axis=-1)).mean())
    print(f"{path}: tokens={len(tokens)} perplexity app={ppl['app']:.4f} llama.cpp={ppl['llama.cpp']:.4f} "
          f"kl={kl:.6f} same_top1={100 * top1:.2f}%", flush=True)
