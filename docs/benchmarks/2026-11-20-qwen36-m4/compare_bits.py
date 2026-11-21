#!/usr/bin/env python3
"""Check that every Q4_0 matrix of the app's weight file holds the bits of the GGUF file, in all layers.

    python3 compare_bits.py APP.vxw MODEL.gguf QWEN_VX/scripts LLAMA/gguf-py

llama.cpp splits the joined input matrices, and orders the value heads of the DeltaNet output
matrix in another way. The check allows these orders and nothing else.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import sys

import numpy as np

vxw, gguf_path, qwen_scripts, gguf_py = sys.argv[1:5]
sys.path.insert(0, qwen_scripts)
sys.path.insert(0, gguf_py)
import qwen_weights  # noqa: E402
from gguf import GGUFReader  # noqa: E402

config, ours = qwen_weights.read_weight_file(vxw)
theirs = {t.name: t for t in GGUFReader(gguf_path).tensors}
hidden, inter = config["hidden_size"], config["moe_intermediate_size"]


def our_rows(name):
    """Codes [rows, blocks, 32] and scale bits [rows, blocks] of one of our Q4_0 matrices."""
    codes = qwen_weights.unpack_u4(np.asarray(ours[name][1]))
    scales = np.asarray(ours[name + ".scale"][1]).view(np.uint16)
    return codes.reshape(len(codes), -1, 32), scales.reshape(len(codes), -1)


def their_rows(name, cols):
    t = theirs[name]
    assert t.tensor_type.name == "Q4_0", f"{name} is {t.tensor_type.name}"
    raw = np.asarray(t.data).reshape(-1, cols // 32, 18)
    scales = raw[..., :2].copy().view(np.uint16)[..., 0]
    qs = raw[..., 2:]
    return np.concatenate([qs & 15, qs >> 4], axis=-1), scales


def same(a, b):
    return a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def same_rows(a, parts):
    """Whether the rows of the parts, together, are the rows of a in some order."""
    def keys(x):
        flat = np.concatenate([x[0].reshape(len(x[0]), -1).astype(np.uint16), x[1]], axis=1)
        return {row.tobytes() for row in flat}, len(flat)
    mine, n = keys(a)
    found = set()
    total = 0
    for p in parts:
        k, m = keys(p)
        found |= k
        total += m
    return n == total == len(mine) and found == mine


checked, failed = 0, []
for layer in range(config["num_layers"]):
    L, B = f"layers.{layer}.", f"blk.{layer}."
    results = {}
    codes, scales = our_rows(L + "experts_gate_up")
    experts = len(codes) // (2 * inter)
    codes = codes.reshape(experts, 2, inter, -1, 32)
    scales = scales.reshape(experts, 2, inter, -1)
    for i, part in enumerate(("gate", "up")):
        c, s = their_rows(f"{B}ffn_{part}_exps.weight", hidden)
        results[f"experts {part}"] = same((codes[:, i].reshape(c.shape), scales[:, i].reshape(s.shape)), (c, s))
    results["experts down"] = same(our_rows(L + "experts_down"), their_rows(B + "ffn_down_exps.weight", inter))
    sh = our_rows(L + "shared_gate_up")
    half = len(sh[0]) // 2
    results["shared gate"] = same((sh[0][:half], sh[1][:half]), their_rows(B + "ffn_gate_shexp.weight", hidden))
    results["shared up"] = same((sh[0][half:], sh[1][half:]), their_rows(B + "ffn_up_shexp.weight", hidden))
    down = our_rows(L + "shared_down")
    results["shared down"] = same(down, their_rows(B + "ffn_down_shexp.weight", down[0].shape[1] * 32))
    if qwen_weights.has_full_attention(config, layer):
        results["qkv"] = same_rows(our_rows(L + "qkv"), [their_rows(f"{B}attn_{p}.weight", hidden) for p in "qkv"])
        out = our_rows(L + "attn_out")
        results["attn out"] = same(out, their_rows(B + "attn_output.weight", out[0].shape[1] * 32))
    else:
        parts = [their_rows(B + n + ".weight", hidden) for n in ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha")]
        results["linear in"] = same_rows(our_rows(L + "linear_in"), parts)
        c, s = our_rows(L + "linear_out")
        tc, ts = their_rows(B + "ssm_out.weight", c.shape[1] * 32)
        heads = 32
        per = c.shape[1] // heads
        # Our value head 2k is their head k, and our head 2k + 1 is their head 16 + k.
        order = [h // 2 if h % 2 == 0 else heads // 2 + h // 2 for h in range(heads)]
        tc = tc.reshape(len(tc), heads, per, 32)[:, order].reshape(c.shape)
        ts = ts.reshape(len(ts), heads, per)[:, order].reshape(s.shape)
        results["linear out"] = same((c, s), (tc, ts))
    for k, v in results.items():
        checked += 1
        if not v:
            failed.append(f"layer {layer} {k}")
print(f"{checked} matrices checked in {config['num_layers']} layers, {len(failed)} differ: {failed}")
