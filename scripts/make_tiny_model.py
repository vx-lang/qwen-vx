#!/usr/bin/env python3
"""Write a tiny Qwen model with random weights, for tests that cannot download a real model.

    python3 scripts/make_tiny_model.py tiny.vxw
    python3 scripts/make_tiny_model.py --fp8 tiny-fp8.vxw
    python3 scripts/make_tiny_model.py --moe --q4 tiny-moe-q4.vxw
    python3 scripts/make_tiny_model.py --moe --hugging-face DIR tiny-moe.vxw

The model has the layers of Qwen2.5-Coder at a small size: hidden size 128, 2 layers, 2 query
heads that share 1 key-value head of 64 values, and a vocabulary of 256 tokens. Its weights come
from a xorshift generator with a fixed seed, so the file has the same bytes on every machine and
with every NumPy version.

With --moe, the model has the layers of Qwen3-Coder-30B-A3B: hidden size 256, 2 layers, 4 query
heads and 2 key-value heads of 64 values, and 8 experts of which each token uses 2. Its weights
come from splitmix64, which NumPy computes for all of them at once. They have Hugging Face names
and go through `convert.py`. --hugging-face also writes them as a Hugging Face folder, for
`check_against_mlx.py`.

With --qwen35, the model has the layers of Qwen3.6-35B-A3B, and its head sizes, so that it runs the
same kernels: hidden size 256, 4 layers of which the last has full attention and the others a Gated
DeltaNet, 4 query heads and 2 key-value heads of 256, 2 key heads and 4 value heads of 128 in a
DeltaNet, and 8 experts of which each token uses 2, and a shared expert. Its weights come from
splitmix64 too, and --hugging-face works as for --moe. With --mtp, it also has the MTP layer of
Qwen3.6-35B-A3B, whose weights come after all the others, so that the other weights do not change.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

import convert
import qwen_weights

CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_layers": 2,
    "num_q_heads": 2,
    "num_kv_heads": 1,
    "head_dim": 64,
    "vocab_size": 256,
    "max_positions": 64,
    "tied_embeddings": 1,
    "qkv_bias": 1,
    "eos_token_id": 255,
    "weight_format": 0,
    "num_experts": 0,
    "experts_per_token": 0,
    "moe_intermediate_size": 0,
    "qk_norm": 0,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
}

# The config.json of the tiny MoE model.
MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 256,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "eos_token_id": 255,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "torch_dtype": "bfloat16",
}
QWEN35_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "norm_topk_prob": True,
    "vocab_size": 256,
    "max_position_embeddings": 64,
    "eos_token_id": 255,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "partial_rotary_factor": 0.25,
    # The sections of the multimodal RoPE add up to half of the 64 rotary values.
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                        "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    "dtype": "bfloat16",
}
# The config.json of the tiny Qwen3.5 MoE model, which keeps the text model in text_config.
QWEN35_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": QWEN35_TEXT_CONFIG,
}
# The same, with the MTP layer.
QWEN35_MTP_CONFIG = dict(QWEN35_CONFIG, text_config=dict(QWEN35_TEXT_CONFIG, mtp_num_hidden_layers=1,
                                                         mtp_use_dedicated_embeddings=False))
SEED = 0x2545F4914F6CDD1D


class Xorshift:
    def __init__(self, seed):
        self.state = seed

    def uniform(self, count, low, high):
        """`count` float32 values, spread evenly between `low` and `high`."""
        values = np.empty(count, dtype=np.float64)
        mask = (1 << 64) - 1
        for i in range(count):
            x = self.state
            x ^= (x << 13) & mask
            x ^= x >> 7
            x ^= (x << 17) & mask
            self.state = x
            values[i] = (x >> 11) / float(1 << 53)
        return (low + (high - low) * values).astype(np.float32)


def tiny_model(config=CONFIG):
    rng = Xorshift(SEED)
    shapes = qwen_weights.tensor_shapes(config)
    tensors = {}
    for name in qwen_weights.tensor_names(config):
        shape = shapes[name]
        count = int(np.prod(shape))
        if name.endswith("norm"):
            values = rng.uniform(count, 0.9, 1.1).reshape(shape)
        else:
            # About 1 / sqrt(hidden size), so the values in each layer keep a similar size.
            values = rng.uniform(count, -0.15, 0.15).reshape(shape)
        if len(shape) == 2:
            values = values * block_sizes(name, shape)
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values))
    return tensors


# Each kind of matrix, and each block of 128 x 128 in it, gets another size. Then the fp8 scales
# of the blocks differ, and a kernel that uses the scales of the wrong block or matrix gives other
# logits. With one size everywhere, all scales are almost equal and such a mistake is not seen.
KIND_SIZES = {"embed": 1.0, "qkv": 1.0, "attn_out": 0.7, "gate_up": 1.2, "down": 0.5}
BLOCK_SIZES = (1.0, 0.3, 1.8, 0.6)


def block_sizes(name, shape):
    """The factor of each weight of the matrix `name`, by its kind and its block of 128 x 128."""
    rows, cols = shape
    row_block = np.arange(rows)[:, None] // 128
    col_block = np.arange(cols)[None, :] // 128
    block = np.asarray(BLOCK_SIZES, dtype=np.float32)[(row_block * 3 + col_block) % len(BLOCK_SIZES)]
    return np.float32(KIND_SIZES[name.rsplit(".", 1)[-1]]) * block


def splitmix64_uniform(start, count, low, high):
    """Values number `start` to `start + count - 1` of the stream, spread evenly between `low`
    and `high`. Value i is splitmix64 of the seed plus i + 1 times its constant."""
    i = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(SEED) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (low + (high - low) * ((z >> np.uint64(11)).astype(np.float64) / float(1 << 53))).astype(np.float32)


# The size of each expert. With one size for all, a kernel that uses the scales of the wrong expert
# gives almost the same logits. The router is larger, so that its choice has a clear margin.
EXPERT_SIZES = (1.0, 0.4, 1.6, 0.7)
ROUTER_SIZE = 4.0


def tiny_moe_model():
    """The tensors of the tiny MoE model, as bf16 bits with their Hugging Face names."""
    c = MOE_CONFIG
    hidden, head_dim, inter = c["hidden_size"], c["head_dim"], c["moe_intermediate_size"]
    shapes = {"model.embed_tokens.weight": (c["vocab_size"], hidden)}
    for layer in range(c["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        shapes[p + "input_layernorm.weight"] = (hidden,)
        shapes[p + "self_attn.q_proj.weight"] = (c["num_attention_heads"] * head_dim, hidden)
        shapes[p + "self_attn.k_proj.weight"] = (c["num_key_value_heads"] * head_dim, hidden)
        shapes[p + "self_attn.v_proj.weight"] = (c["num_key_value_heads"] * head_dim, hidden)
        shapes[p + "self_attn.o_proj.weight"] = (hidden, c["num_attention_heads"] * head_dim)
        shapes[p + "self_attn.q_norm.weight"] = (head_dim,)
        shapes[p + "self_attn.k_norm.weight"] = (head_dim,)
        shapes[p + "post_attention_layernorm.weight"] = (hidden,)
        shapes[p + "mlp.gate.weight"] = (c["num_experts"], hidden)
        for e in range(c["num_experts"]):
            shapes[f"{p}mlp.experts.{e}.gate_proj.weight"] = (inter, hidden)
            shapes[f"{p}mlp.experts.{e}.up_proj.weight"] = (inter, hidden)
            shapes[f"{p}mlp.experts.{e}.down_proj.weight"] = (hidden, inter)
    shapes["model.norm.weight"] = (hidden,)
    shapes["lm_head.weight"] = (c["vocab_size"], hidden)
    tensors, start = {}, 0
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        if name.endswith("norm.weight"):
            values = splitmix64_uniform(start, count, 0.9, 1.1)
        else:
            values = splitmix64_uniform(start, count, -0.1, 0.1)
            if ".experts." in name:
                values *= np.float32(EXPERT_SIZES[int(name.split(".")[5]) % len(EXPERT_SIZES)])
            elif name.endswith("mlp.gate.weight"):
                values *= np.float32(ROUTER_SIZE)
        start += count
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values).reshape(shape))
    return tensors


def tiny_qwen35_model(mtp=False):
    """The tensors of the tiny Qwen3.5 MoE model, as bf16 bits with their Hugging Face names. `mtp`
    adds the MTP layer after all the other tensors."""
    c = QWEN35_TEXT_CONFIG
    hidden, head_dim, inter = c["hidden_size"], c["head_dim"], c["moe_intermediate_size"]
    heads, kv_heads, experts = c["num_attention_heads"], c["num_key_value_heads"], c["num_experts"]
    k_total = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    v_heads = c["linear_num_value_heads"]
    v_total = v_heads * c["linear_value_head_dim"]
    shared = c["shared_expert_intermediate_size"]
    src = "model.language_model."
    shapes = {src + "embed_tokens.weight": (c["vocab_size"], hidden)}
    layers = [(f"{src}layers.{layer}.", kind) for layer, kind in enumerate(c["layer_types"])]
    for p, kind in layers + ([("mtp.layers.0.", "full_attention")] if mtp else []):
        if p.startswith("mtp."):
            # The MTP layer comes after lm_head.
            shapes[src + "norm.weight"] = (hidden,)
            shapes["lm_head.weight"] = (c["vocab_size"], hidden)
            shapes["mtp.pre_fc_norm_embedding.weight"] = (hidden,)
            shapes["mtp.pre_fc_norm_hidden.weight"] = (hidden,)
            shapes["mtp.fc.weight"] = (hidden, 2 * hidden)
        shapes[p + "input_layernorm.weight"] = (hidden,)
        if kind == "full_attention":
            shapes[p + "self_attn.q_proj.weight"] = (heads * head_dim * 2, hidden)
            shapes[p + "self_attn.k_proj.weight"] = (kv_heads * head_dim, hidden)
            shapes[p + "self_attn.v_proj.weight"] = (kv_heads * head_dim, hidden)
            shapes[p + "self_attn.o_proj.weight"] = (hidden, heads * head_dim)
            shapes[p + "self_attn.q_norm.weight"] = (head_dim,)
            shapes[p + "self_attn.k_norm.weight"] = (head_dim,)
        else:
            shapes[p + "linear_attn.in_proj_qkv.weight"] = (2 * k_total + v_total, hidden)
            shapes[p + "linear_attn.in_proj_z.weight"] = (v_total, hidden)
            shapes[p + "linear_attn.in_proj_b.weight"] = (v_heads, hidden)
            shapes[p + "linear_attn.in_proj_a.weight"] = (v_heads, hidden)
            shapes[p + "linear_attn.conv1d.weight"] = (2 * k_total + v_total, 1, c["linear_conv_kernel_dim"])
            shapes[p + "linear_attn.A_log"] = (v_heads,)
            shapes[p + "linear_attn.dt_bias"] = (v_heads,)
            shapes[p + "linear_attn.norm.weight"] = (c["linear_value_head_dim"],)
            shapes[p + "linear_attn.out_proj.weight"] = (hidden, v_total)
        shapes[p + "post_attention_layernorm.weight"] = (hidden,)
        shapes[p + "mlp.gate.weight"] = (experts, hidden)
        shapes[p + "mlp.experts.gate_up_proj"] = (experts, 2 * inter, hidden)
        shapes[p + "mlp.experts.down_proj"] = (experts, hidden, inter)
        shapes[p + "mlp.shared_expert.gate_proj.weight"] = (shared, hidden)
        shapes[p + "mlp.shared_expert.up_proj.weight"] = (shared, hidden)
        shapes[p + "mlp.shared_expert.down_proj.weight"] = (hidden, shared)
        shapes[p + "mlp.shared_expert_gate.weight"] = (1, hidden)
    shapes[src + "norm.weight"] = (hidden,)
    shapes["lm_head.weight"] = (c["vocab_size"], hidden)
    if mtp:
        shapes["mtp.norm.weight"] = (hidden,)
    # The ranges of the weights whose kind needs another one than (-0.1, 0.1).
    ranges = {"conv1d.weight": (-0.5, 0.5), "A_log": (-1.0, 1.0), "dt_bias": (-1.0, 1.0), "linear_attn.norm.weight": (0.9, 1.1)}
    tensors, start = {}, 0
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        low, high = next((r for suffix, r in ranges.items() if name.endswith(suffix)), (-0.1, 0.1))
        values = splitmix64_uniform(start, count, low, high).reshape(shape)
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            # The model multiplies by 1 + w. MLX adds the 1 in bf16, so keep 1 + w exact there:
            # a multiple of 1/128.
            values = np.round(values * 128) / np.float32(128)
        elif name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
            values *= np.asarray([EXPERT_SIZES[e % len(EXPERT_SIZES)] for e in range(experts)], dtype=np.float32)[:, None, None]
        elif name.endswith("mlp.gate.weight"):
            values *= np.float32(ROUTER_SIZE)
        start += count
        tensors[name] = ("bf16", qwen_weights.f32_to_bf16(values.astype(np.float32)))
    return tensors


def write_hugging_face_folder(directory, config, tensors):
    """Write config.json and model.safetensors, with bf16 tensors given as uint16 bits."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as out:
        json.dump(config, out, indent=2)
    header, offset = {}, 0
    for name, (dtype, array) in tensors.items():
        assert dtype == "bf16", f"{name}: {dtype}"
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(os.path.join(directory, "model.safetensors"), "wb") as out:
        out.write(struct.pack("<Q", len(text)) + text)
        for _, array in tensors.values():
            out.write(np.ascontiguousarray(array).data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", help="the weight file to write")
    weight_format = parser.add_mutually_exclusive_group()
    weight_format.add_argument("--fp8", action="store_true", help="store the weight matrices as f8e4m3 codes")
    weight_format.add_argument("--q4", action="store_true", help="store the weight matrices as Q4_0 codes")
    parser.add_argument("--q6-lm-head", action="store_true", help="with --q4, store lm_head as 6-bit codes, not f8e4m3")
    parser.add_argument("--moe", action="store_true", help="write the tiny mixture-of-experts model")
    parser.add_argument("--qwen35", action="store_true", help="write the tiny model with the layers of Qwen3.6-35B-A3B")
    parser.add_argument("--mtp", action="store_true", help="with --qwen35, add the MTP layer of Qwen3.6-35B-A3B")
    parser.add_argument("--hugging-face", metavar="DIR", help="also write the MoE or Qwen3.5 model as a Hugging Face folder")
    parser.add_argument("--max-positions", type=int, help="the positions of the model, 64 if not given; the weights do not change")
    parser.add_argument("--vocab-size", type=int, help="the tokens of the dense model, 256 if not given, for a bigger tokenizer; other weights come out")
    args = parser.parse_args()
    assert not (args.moe and args.qwen35), "give --moe or --qwen35, not both"
    assert not ((args.moe or args.qwen35) and args.vocab_size), "--vocab-size is for the dense model"
    assert not args.q6_lm_head or args.q4, "--q6-lm-head needs --q4"
    assert not args.mtp or args.qwen35, "--mtp needs --qwen35"
    if args.moe or args.qwen35:
        if args.moe:
            hf_config, hf = MOE_CONFIG, tiny_moe_model()
        else:
            hf_config, hf = (QWEN35_MTP_CONFIG if args.mtp else QWEN35_CONFIG), tiny_qwen35_model(args.mtp)
        if args.hugging_face:
            write_hugging_face_folder(args.hugging_face, hf_config, hf)
        config, tensors = convert.convert_tensors(qwen_weights.config_from_json(hf_config), hf)
    else:
        assert not args.hugging_face, "--hugging-face needs --moe or --qwen35"
        config = CONFIG if not args.vocab_size else dict(CONFIG, vocab_size=args.vocab_size)
        config, tensors = config, tiny_model(config)
    if args.max_positions:
        config = dict(config, max_positions=args.max_positions)
    if args.fp8 or args.q4:
        weight_format = "fp8" if args.fp8 else ("q4_0_q6" if args.q6_lm_head else "q4_0")
        config, tensors = qwen_weights.quantize(config, tensors, weight_format)
    qwen_weights.write_weight_file(args.output, config, tensors)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
