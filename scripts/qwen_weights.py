"""Read Qwen weights from safetensors files, and read and write the weight file of the Qwen app.

The weight file holds 8-byte little-endian words, so a Vx program can read it with plain `i64`
and `f64` loads and needs no JSON parser:

    word 0       magic, the bytes b"VXQWEN01"
    word 1       format version, 1 or 2
    word 2       number of tensors
    word 3       alignment of each tensor, in bytes
    word 4       byte offset of the tensor table
    words 5-20   version 1: the first 16 integers of CONFIG_INTS, in that order
    words 21-22  version 1: rope_theta and rms_norm_eps, as f64
    words 5-36   version 2: the integers of CONFIG_INTS, in that order, then zeros
    words 37-38  version 2: rope_theta and rms_norm_eps, as f64

A model of `arch` 0 (Qwen2, Qwen3 MoE) gets version 1, and a model of `arch` 1 (Qwen3.5 MoE, which
Qwen3.6 also uses) gets version 2.

The tensor table has one entry of 16 words (128 bytes) per tensor: the name in words 0-7 (ASCII,
padded with zero bytes), the dtype code, the rank, four dimensions (unused ones are 0), the byte
offset of the data and its size in bytes. The tensors come in the order of `tensor_names`.

`weight_format` says how the file stores the weight matrices of MATRICES. Each other tensor is
bf16 in every format, except the f32 tensors of arch 1 below.

    0, bf16    each weight matrix is bf16.
    1, fp8     each weight matrix is f8e4m3 codes, one byte for each weight. The tensor after it,
               `<name>.scale`, has one f32 scale for each block of 128 x 128 weights. A weight is
               the value of its code times the scale of its block.
    2, q4_0    each weight matrix except lm_head is u4 codes, two in each byte, the first weight
               in the low 4 bits. Its `<name>.scale` has one f16 scale for each 32 weights of a
               row. A weight is (code - 8) times its scale, as in the Q4_0 of llama.cpp. lm_head
               is f8e4m3, as in an fp8 file.
    3, q4_0_q6 as q4_0, but lm_head has 6-bit codes, as `quantize_q6_0` makes them: its low 4 bits
               in `lm_head`, u4 as in q4_0, its high 2 bits in `lm_head.high`, u2, four in each
               byte, the first weight in the low 2 bits, and one f16 scale for each 32 weights of a
               row in `lm_head.scale`. A weight is (code - 32) times its scale. That is 6.5 bits for
               each weight, as the Q6_K of llama.cpp has.

A file in fp8 or q4_0 always has `lm_head`, also for a model whose output projection is its
embedding table.

A mixture-of-experts (MoE) model has num_experts > 0. Each of its layers has a `router` with one
row for each expert, and the matrices `experts_gate_up` and `experts_down` in place of `gate_up`
and `down`. These hold expert e after expert e - 1. An expert has 2 * moe_intermediate_size rows
in `experts_gate_up`, its gate rows and then its up rows, and hidden_size rows in `experts_down`.

In arch 1, layer l has full attention when (l + 1) % full_attention_interval is 0, and a Gated
DeltaNet (a linear attention) otherwise. Each RMSNorm of the model multiplies by 1 + w, and the file
holds 1 + w as f32, so that a program multiplies by the weight as for arch 0. A layer with full
attention has:

    qkv          the rows of q, k and v, then the rows of the gate of each query head. The
                 attention output of head h is multiplied by sigmoid of gate head h.
    q_norm, k_norm, attn_out
                 as in arch 0. RoPE turns only the first rotary_dim values of each head.

A Gated DeltaNet layer has, with K = linear_k_heads * linear_k_dim and V = linear_v_heads *
linear_v_dim:

    linear_in    rows [q K | k K | v V | z V | b linear_v_heads | a linear_v_heads], one matrix.
    conv         f32 [2K + V, conv_width], the causal convolution of q, k and v. Column
                 conv_width - 1 multiplies the current token, and column 0 the oldest.
    neg_A        f32 [linear_v_heads], -exp(A_log).
    dt_bias      f32 [linear_v_heads].
    linear_norm  f32 [linear_v_dim], the RMSNorm of each head of the output. It multiplies by w,
                 not by 1 + w.
    linear_out   [hidden_size, V].

Each layer of arch 1 also has a shared expert. Its `router` has one more row, the gate of the
shared expert, and `shared_gate_up` and `shared_down` hold the shared expert as one expert of
`experts_gate_up` and `experts_down`, with shared_expert_intermediate_size in place of
moe_intermediate_size.

A model of arch 1 can also have mtp_layers = 1: the multi-token prediction (MTP) layer of the model,
which predicts the token after the next one. Its tensors come after all the other tensors, so the
other tensors of a file with them have the same order and bytes as in a file without them, and the
same offsets unless the longer tensor table moves the data to the next 16 KiB page. The
MTP layer uses the `embed` and `lm_head` of the model. At position p + 1, its input is token p + 1
and the hidden state h of position p, after final_norm (or, for a draft after a draft, the h of
the MTP layer itself). It has:

    mtp.embed_norm   f32 [hidden_size], the RMSNorm (1 + w) of the embedding of the token.
    mtp.hidden_norm  f32 [hidden_size], the RMSNorm (1 + w) of h.
    mtp.fc           [hidden_size, 2 * hidden_size], a weight matrix. Its input is the normed
                     embedding, then the normed h.
    mtp.layers.0.*   the tensors of a layer with full attention, as below. The layer has its own
                     KV cache.
    mtp.final_norm   f32 [hidden_size], the RMSNorm (1 + w) after the layer. Its output is the h of
                     the MTP layer, and lm_head of it gives the logits of position p + 2.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import concurrent.futures
import json
import os
import struct
from pathlib import Path

import numpy as np

MAGIC = b"VXQWEN01"
TABLE_OFFSET = 4096
ENTRY_BYTES = 128
# A Metal buffer over mapped file memory must start on a page, and an Apple Silicon page is 16 KiB.
ALIGNMENT = 16384

CONFIG_INTS = [
    "hidden_size",
    "intermediate_size",
    "num_layers",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "vocab_size",
    "max_positions",
    "tied_embeddings",
    "qkv_bias",
    "eos_token_id",
    "weight_format",
    "num_experts",
    "experts_per_token",
    "moe_intermediate_size",
    "qk_norm",
    # Version 2 only. A version 1 file reads them as V1_DEFAULTS.
    "arch",
    "full_attention_interval",
    "linear_k_heads",
    "linear_v_heads",
    "linear_k_dim",
    "linear_v_dim",
    "conv_width",
    "rotary_dim",
    "shared_expert_intermediate_size",
    "mtp_layers",
]
# The integers that the header of each version has room for.
CONFIG_INT_SLOTS = {1: 16, 2: 32}
assert len(CONFIG_INTS) <= CONFIG_INT_SLOTS[2]
V1_DEFAULTS = dict.fromkeys(CONFIG_INTS[CONFIG_INT_SLOTS[1] :], 0)
ARCHS = {"qwen": 0, "qwen3_5": 1}
WEIGHT_FORMATS = {"bf16": 0, "fp8": 1, "q4_0": 2, "q4_0_q6": 3}

DTYPE_CODES = {"f32": 0, "bf16": 1, "f8e4m3": 2, "f16": 3, "u4": 4, "u2": 5}
DTYPE_NAMES = {code: name for name, code in DTYPE_CODES.items()}
DTYPE_BITS = {"f32": 32, "bf16": 16, "f8e4m3": 8, "f16": 16, "u4": 4, "u2": 2}
# The NumPy array of a u4 tensor holds two codes in each uint8, so its last axis is half as long,
# and that of a u2 tensor four, so its last axis is a quarter as long.
STORAGE = {"f32": np.float32, "bf16": np.uint16, "f8e4m3": np.uint8, "f16": np.float16, "u4": np.uint8, "u2": np.uint8}

# The weight matrices that an fp8 or q4_0 file quantizes. The kinds of a layer end the name.
MATRICES = ("qkv", "attn_out", "gate_up", "down", "experts_gate_up", "experts_down", "lm_head",
            "linear_in", "linear_out", "shared_gate_up", "shared_down", "fc")
BLOCK = 128
Q4_GROUP = 32
# A 6-bit matrix has one f16 scale for each Q6_GROUP weights of a row.
Q6_GROUP = 32


def bf16_to_f32(bits):
    """Widen bf16 values, given as uint16 bit patterns, to float32. The result is exact."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    """Round float32 values to bf16 bit patterns, to nearest with ties to even."""
    values = np.asarray(values, dtype=np.float32)
    assert np.all(np.isfinite(values)), "only finite values round to bf16 here"
    bits = values.view(np.uint32)
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    return ((bits + rounding) >> 16).astype(np.uint16)


def is_matrix(name):
    """Whether an fp8 or q4_0 file quantizes the tensor `name`, and has a `<name>.scale`."""
    return name.rsplit(".", 1)[-1] in MATRICES


def is_six_bit(config, name):
    """Whether the matrix `name` has 6-bit codes in a file with this config: its low 4 bits in
    `name`, its high 2 bits in `<name>.high`, and its scales in `<name>.scale`."""
    return config["weight_format"] == WEIGHT_FORMATS["q4_0_q6"] and name == "lm_head"


def matrix_dtype(config, name):
    """The dtype of the weight matrix `name` in a file with this config. A 6-bit matrix is u4, the
    dtype of its low 4 bits."""
    weight_format = config["weight_format"]
    if weight_format == WEIGHT_FORMATS["q4_0_q6"]:
        return "u4"
    if weight_format == WEIGHT_FORMATS["q4_0"] and name != "lm_head":
        return "u4"
    return "f8e4m3" if weight_format else "bf16"


def file_version(config):
    """The format version of the weight file of a model: 1 for arch 0, and 2 for arch 1."""
    return 2 if config.get("arch", 0) else 1


def has_full_attention(config, layer):
    """Whether layer `layer` has full attention, and not a Gated DeltaNet."""
    return not config.get("arch", 0) or (layer + 1) % config["full_attention_interval"] == 0


def layer_names(config, layer, prefix=None):
    """The names of the tensors of one layer of an arch 1 model, in file order. `prefix` names the
    MTP layer, which has full attention."""
    full = prefix is not None or has_full_attention(config, layer)
    prefix = prefix or f"layers.{layer}."
    if full:
        kinds = ["attn_norm", "qkv", "q_norm", "k_norm", "attn_out"]
    else:
        kinds = ["attn_norm", "linear_in", "conv", "neg_A", "dt_bias", "linear_norm", "linear_out"]
    kinds += ["mlp_norm", "router", "experts_gate_up", "experts_down"]
    if config["shared_expert_intermediate_size"]:
        kinds += ["shared_gate_up", "shared_down"]
    return [prefix + kind for kind in kinds]


def tensor_names(config):
    """The names of the tensors in the weight file, in file order."""
    assert config["weight_format"] in WEIGHT_FORMATS.values(), f"weight format {config['weight_format']}"
    assert not (config["weight_format"] and config["tied_embeddings"]), "a quantized file has its own lm_head"
    names = ["embed"]
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        if config.get("arch", 0):
            names += layer_names(config, layer)
            continue
        names += [prefix + "attn_norm", prefix + "qkv"]
        if config["qkv_bias"]:
            names.append(prefix + "qkv_bias")
        if config["qk_norm"]:
            names += [prefix + "q_norm", prefix + "k_norm"]
        names += [prefix + "attn_out", prefix + "mlp_norm"]
        if config["num_experts"]:
            names += [prefix + "router", prefix + "experts_gate_up", prefix + "experts_down"]
        else:
            names += [prefix + "gate_up", prefix + "down"]
    names.append("final_norm")
    if not config["tied_embeddings"]:
        names.append("lm_head")
    names += mtp_names(config)
    if config["weight_format"]:
        def parts(name):
            if is_six_bit(config, name):
                return [name, name + ".high", name + ".scale"]
            return [name, name + ".scale"] if is_matrix(name) else [name]

        names = [n for name in names for n in parts(name)]
    return names


def mtp_names(config):
    """The names of the tensors of the MTP layer, in file order: none when mtp_layers is 0."""
    if not config.get("mtp_layers", 0):
        return []
    assert config.get("arch", 0) and config["mtp_layers"] == 1, "only a model of arch 1 can have one MTP layer"
    return ["mtp.embed_norm", "mtp.hidden_norm", "mtp.fc"] + layer_names(config, 0, "mtp.layers.0.") + ["mtp.final_norm"]


def tensor_shapes(config):
    """The shape of each tensor of the weight file, by name."""
    hidden = config["hidden_size"]
    head_dim = config["head_dim"]
    q_rows = config["num_q_heads"] * head_dim
    qkv_rows = q_rows + 2 * config["num_kv_heads"] * head_dim
    inter = config["intermediate_size"]
    experts, moe_inter = config["num_experts"], config["moe_intermediate_size"]
    shapes = {"embed": (config["vocab_size"], hidden), "final_norm": (hidden,)}
    if not config["tied_embeddings"]:
        shapes["lm_head"] = (config["vocab_size"], hidden)
    for layer in range(config["num_layers"]):
        prefix = f"layers.{layer}."
        if config.get("arch", 0):
            shapes.update(arch1_layer_shapes(config, layer))
            continue
        shapes[prefix + "attn_norm"] = (hidden,)
        shapes[prefix + "qkv"] = (qkv_rows, hidden)
        if config["qkv_bias"]:
            shapes[prefix + "qkv_bias"] = (qkv_rows,)
        if config["qk_norm"]:
            shapes[prefix + "q_norm"] = (head_dim,)
            shapes[prefix + "k_norm"] = (head_dim,)
        shapes[prefix + "attn_out"] = (hidden, q_rows)
        shapes[prefix + "mlp_norm"] = (hidden,)
        if experts:
            shapes[prefix + "router"] = (experts, hidden)
            shapes[prefix + "experts_gate_up"] = (experts * 2 * moe_inter, hidden)
            shapes[prefix + "experts_down"] = (experts * hidden, moe_inter)
        else:
            shapes[prefix + "gate_up"] = (2 * inter, hidden)
            shapes[prefix + "down"] = (hidden, inter)
    if config.get("mtp_layers", 0):
        shapes.update({"mtp.embed_norm": (hidden,), "mtp.hidden_norm": (hidden,), "mtp.fc": (hidden, 2 * hidden),
                       "mtp.final_norm": (hidden,)})
        shapes.update(arch1_layer_shapes(config, 0, "mtp.layers.0."))
    if config["weight_format"]:
        for name in [n for n in shapes if is_matrix(n)]:
            rows, cols = shapes[name]
            if is_six_bit(config, name):
                assert cols % Q6_GROUP == 0, f"{name}: {cols} columns are not groups of {Q6_GROUP}"
                shapes[name + ".high"] = (rows, cols)
                shapes[name + ".scale"] = (rows, cols // Q6_GROUP)
            elif matrix_dtype(config, name) == "u4":
                assert cols % Q4_GROUP == 0, f"{name}: {cols} columns are not groups of {Q4_GROUP}"
                shapes[name + ".scale"] = (rows, cols // Q4_GROUP)
            else:
                shapes[name + ".scale"] = (-(-rows // BLOCK), -(-cols // BLOCK))
    return shapes


def arch1_layer_shapes(config, layer, prefix=None):
    """The shape of each tensor of one layer of an arch 1 model, by name. `prefix` names the MTP
    layer, which has full attention."""
    full = prefix is not None or has_full_attention(config, layer)
    prefix = prefix or f"layers.{layer}."
    hidden, head_dim, q_heads = config["hidden_size"], config["head_dim"], config["num_q_heads"]
    k_total = config["linear_k_heads"] * config["linear_k_dim"]
    v_heads = config["linear_v_heads"]
    v_total = v_heads * config["linear_v_dim"]
    shared = config["shared_expert_intermediate_size"]
    shapes = {
        "attn_norm": (hidden,),
        "mlp_norm": (hidden,),
        "router": (config["num_experts"] + (1 if shared else 0), hidden),
        "experts_gate_up": (config["num_experts"] * 2 * config["moe_intermediate_size"], hidden),
        "experts_down": (config["num_experts"] * hidden, config["moe_intermediate_size"]),
    }
    if shared:
        shapes.update(shared_gate_up=(2 * shared, hidden), shared_down=(hidden, shared))
    if full:
        shapes.update(
            qkv=(2 * q_heads * head_dim + 2 * config["num_kv_heads"] * head_dim, hidden),
            q_norm=(head_dim,),
            k_norm=(head_dim,),
            attn_out=(hidden, q_heads * head_dim),
        )
    else:
        shapes.update(
            linear_in=(2 * k_total + 2 * v_total + 2 * v_heads, hidden),
            conv=(2 * k_total + v_total, config["conv_width"]),
            neg_A=(v_heads,),
            dt_bias=(v_heads,),
            linear_norm=(config["linear_v_dim"],),
            linear_out=(hidden, v_total),
        )
    return {prefix + kind: shape for kind, shape in shapes.items()}


def storage_shape(dtype, shape):
    """The shape of the NumPy array that holds a tensor of this dtype and shape."""
    per_byte = {"u4": 2, "u2": 4}.get(dtype, 1)
    assert shape[-1] % per_byte == 0, f"{dtype} rows of {shape[-1]} codes do not fill whole bytes"
    return (*shape[:-1], shape[-1] // per_byte)


def resolve(entry):
    """A (dtype name, array) pair of a tensors dict, whose value can also be a function that
    returns the pair."""
    return entry() if callable(entry) else entry


def write_weight_file(path, config, tensors):
    """Write `tensors`, a dict from name to (dtype name, numpy array), in file order. A value can
    also be a function that returns the pair. The file then gets each tensor when it is written,
    so that only one is in memory at a time."""
    names = tensor_names(config)
    shapes = tensor_shapes(config)
    assert sorted(names) == sorted(tensors), "the tensors do not match the names of the config"
    # The data starts at the first aligned offset after the tensor table.
    data_start = -(-(TABLE_OFFSET + ENTRY_BYTES * len(names)) // ALIGNMENT) * ALIGNMENT
    header = bytearray(data_start)
    version = file_version(config)
    fields = CONFIG_INTS[: CONFIG_INT_SLOTS[version]]
    if version == 1:
        assert all(config.get(field, 0) == 0 for field in V1_DEFAULTS), "a version 1 file has no room for these fields"
    words = [version, len(names), ALIGNMENT, TABLE_OFFSET]
    words += [int(config[field]) for field in fields]
    words += [0] * (CONFIG_INT_SLOTS[version] - len(fields))
    header[0:8] = MAGIC
    struct.pack_into(f"<{len(words)}q", header, 8, *words)
    struct.pack_into("<2d", header, 8 * (5 + CONFIG_INT_SLOTS[version]), config["rope_theta"], config["rms_norm_eps"])

    offset = data_start
    with open(path, "wb") as out:
        for index, name in enumerate(names):
            dtype, array = resolve(tensors[name])
            shape = shapes[name]
            assert tuple(array.shape) == storage_shape(dtype, shape), f"{name}: shape {array.shape} for {shape}"
            assert array.dtype == STORAGE[dtype], f"{name}: storage {array.dtype} for {dtype}"
            nbytes = int(np.prod(shape)) * DTYPE_BITS[dtype] // 8
            dims = list(shape) + [0] * (4 - len(shape))
            entry = name.encode("ascii").ljust(64, b"\0")
            assert len(entry) == 64, f"{name}: the name is longer than 64 bytes"
            entry += struct.pack("<8q", DTYPE_CODES[dtype], len(shape), *dims, offset, nbytes)
            header[TABLE_OFFSET + index * ENTRY_BYTES : TABLE_OFFSET + (index + 1) * ENTRY_BYTES] = entry
            out.seek(offset)
            out.write(np.ascontiguousarray(array).data)
            offset += -(-nbytes // ALIGNMENT) * ALIGNMENT
        out.truncate(offset)
        out.seek(0)
        out.write(header)


def read_weight_file(path):
    """Return the config and a dict from name to a float32 view of each tensor.

    The views are read from a memory map, and widened from bf16 when they are used.
    """
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert bytes(data[0:8]) == MAGIC, f"{path}: not a weight file of the Qwen app"
    (version,) = struct.unpack_from("<q", data, 8)
    assert version in CONFIG_INT_SLOTS, f"{path}: format version {version}, expected 1 or 2"
    slots = CONFIG_INT_SLOTS[version]
    words = struct.unpack_from(f"<{4 + slots}q", data, 8)
    count, alignment, table = words[1:4]
    config = dict(V1_DEFAULTS)
    config.update(zip(CONFIG_INTS[:slots], words[4:]))
    config["rope_theta"], config["rms_norm_eps"] = struct.unpack_from("<2d", data, 8 * (5 + slots))
    tensors = {}
    for index in range(count):
        start = table + index * ENTRY_BYTES
        name = bytes(data[start : start + 64]).rstrip(b"\0").decode("ascii")
        dtype, rank, *rest = struct.unpack_from("<8q", data, start + 64)
        dims, offset, nbytes = rest[:4], rest[4], rest[5]
        assert offset % alignment == 0, f"{name}: offset {offset} is not aligned"
        dtype = DTYPE_NAMES[dtype]
        storage = STORAGE[dtype]
        raw = np.frombuffer(data, dtype=storage, count=nbytes // np.dtype(storage).itemsize, offset=offset)
        tensors[name] = (dtype, raw.reshape(storage_shape(dtype, dims[:rank])))
    assert list(tensors) == tensor_names(config), f"{path}: the tensors are not in file order"
    return config, tensors


def as_f32(entry):
    """The float32 values of a bf16 or f32 (dtype name, array) entry of `read_weight_file`."""
    dtype, array = entry
    assert dtype in ("bf16", "f32"), f"use widen() for {dtype}"
    return bf16_to_f32(array) if dtype == "bf16" else np.asarray(array, dtype=np.float32)


def _e4m3_values():
    """The value of each of the 256 f8e4m3 codes: OCP e4m3, bias 7, no infinity, NaN at 0x7F."""
    codes = np.arange(256)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    values = np.where(exponent == 0, mantissa / 8 * 2.0**-6, (1 + mantissa / 8) * 2.0 ** (exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


E4M3 = _e4m3_values()
E4M3_MAX = 448.0


def f32_to_e4m3(values):
    """Round float32 values to f8e4m3 codes, to nearest with ties to even. Values above 448 in
    size become 448."""
    values = np.asarray(values, dtype=np.float32)
    assert np.all(np.isfinite(values)), "only finite values round to f8e4m3 here"
    positive = E4M3[:0x7F]  # codes 0 to 126, in increasing order
    size = np.minimum(np.abs(values), E4M3_MAX)
    high = np.minimum(np.searchsorted(positive, size), 0x7E)
    low = np.maximum(high - 1, 0)
    up, down = positive[high] - size, size - positive[low]
    code = np.where((up < down) | ((up == down) & (high % 2 == 0)), high, low).astype(np.uint8)
    return code | np.where(np.signbit(values), 0x80, 0).astype(np.uint8)


def quantize_fp8(matrix):
    """f8e4m3 codes and f32 block scales for a float32 matrix. The scale of a block of 128 x 128
    is the largest size in it divided by 448, so that block's largest weight becomes 448."""
    rows, cols = matrix.shape
    scales = np.ones((-(-rows // BLOCK), -(-cols // BLOCK)), dtype=np.float32)
    codes = np.empty((rows, cols), dtype=np.uint8)
    for i in range(scales.shape[0]):
        band = np.asarray(matrix[i * BLOCK : (i + 1) * BLOCK], dtype=np.float32)
        for j in range(scales.shape[1]):
            block = band[:, j * BLOCK : (j + 1) * BLOCK]
            largest = np.float32(np.abs(block).max())
            if largest > 0:
                scales[i, j] = largest / np.float32(E4M3_MAX)
            codes[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = f32_to_e4m3(block / scales[i, j])
    return codes, scales


def quantize_q4_0(matrix, importance=None):
    """u4 codes, two in each byte, and f16 scales for a float32 matrix, in the Q4_0 of llama.cpp.

    Without `importance`, the codes round as `quantize_row_q4_0_ref` of llama.cpp rounds them. In
    each group of 32 weights of a row, the weight of the largest size gets code 0, so the scale is
    that weight divided by -8.

    `importance` comes from importance.py. It has one row of column weights for each block of
    rows, so its shape is [blocks, columns]. Each group then gets the codes and the scale that
    `quantize_row_q4_0_impl` of llama.cpp chooses with an importance matrix. See `_q4_0_search`."""
    rows, cols = matrix.shape
    assert cols % Q4_GROUP == 0, f"{cols} columns are not groups of {Q4_GROUP}"
    if importance is not None:
        assert importance.shape[1] == cols and rows % len(importance) == 0, f"importance {importance.shape} for {matrix.shape}"
    codes = np.empty((rows, cols // 2), dtype=np.uint8)
    scales = np.empty((rows, cols // Q4_GROUP), dtype=np.float16)
    # About 16M weights at a time, so that a large matrix needs little memory. The search does
    # about 20 times more work, so it takes 1M at a time on each of several threads.
    step = max(1, (1 << (24 if importance is None else 20)) // cols)

    def rows_from(start):
        chunk = np.asarray(matrix[start : start + step], dtype=np.float32)
        assert np.all(np.isfinite(chunk)), "only finite values round to Q4_0 here"
        groups = chunk.reshape(len(chunk), -1, Q4_GROUP)
        if importance is None:
            q, d = _q4_0_ref(groups)
        else:
            block = importance[np.arange(start, start + len(chunk)) // (rows // len(importance))]
            q, d = _q4_0_search(chunk, groups, block.reshape(groups.shape))
        q = q.reshape(len(chunk), cols)
        codes[start : start + step] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[start : start + step] = d.reshape(len(chunk), -1).astype(np.float16)

    starts = range(0, rows, step)
    if importance is None:
        for start in starts:
            rows_from(start)
    else:
        # NumPy lets other threads run while it computes on large arrays.
        with concurrent.futures.ThreadPoolExecutor(max(1, (os.cpu_count() or 1) - 2)) as pool:
            list(pool.map(rows_from, starts))
    return codes, scales


def _q4_0_ref(groups):
    """The codes and float32 scales of `quantize_row_q4_0_ref`, for groups of shape [rows, n, 32]."""
    largest = np.take_along_axis(groups, np.abs(groups).argmax(axis=-1)[..., None], axis=-1)
    # As in llama.cpp, a group of zeros keeps +0.0 here, also when its weights are -0.0.
    largest = np.where(largest == 0, np.float32(0), largest)
    d = largest / np.float32(-8)
    inverse = np.divide(np.float32(1), d, out=np.zeros_like(d), where=d != 0)
    # All float32, as in llama.cpp. The sum is at least 0.5, so the floor is its cast to int.
    return np.minimum(np.float32(15), np.floor(groups * inverse + np.float32(8.5))).astype(np.uint8), d


# The sizes in steps that the search tries for the largest weight of a group: 8 first, then 7.1 to
# 8.9 in tenths, as `make_qx_quants` of llama.cpp tries them, in float32.
Q4_TRIES = [np.float32(8)] + [np.float32(8) + np.float32(0.1) * np.float32(i) for i in range(-9, 10) if i != 0]


def _q4_0_search(rows, groups, importance):
    """The codes and float32 scales of `quantize_row_q4_0_impl` of llama.cpp, for groups of shape
    [rows, n, 32], and the importance of each of their weights.

    The error of weight x counts importance * sqrt(sigma2 + x * x) times, where sigma2 is the mean
    square of its row. For each size of `Q4_TRIES`, the largest weight of the group sets the step,
    each weight rounds to a code, and the codes get the scale of the smallest weighted square error.
    The group keeps the try of the smallest error."""
    sigma2 = np.mean(rows * rows, axis=-1)[:, None, None]
    weight = importance * np.sqrt(sigma2 + groups * groups)
    largest = np.take_along_axis(groups, np.abs(groups).argmax(axis=-1)[..., None], axis=-1)[..., 0]
    zero = largest == 0
    largest = np.where(zero, np.float32(1), largest)
    best = np.zeros(largest.shape, dtype=np.float32)
    scale = np.zeros(largest.shape, dtype=np.float32)
    codes = np.zeros(groups.shape, dtype=np.float32)
    for index, size in enumerate(Q4_TRIES):
        q = np.clip(np.rint(groups * (-size / largest)[..., None]), -8, 7)
        sumlx = (weight * groups * q).sum(axis=-1)
        suml2 = (weight * q * q).sum(axis=-1)
        d = np.divide(sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 > 0)
        # The error is the sum of weight * x * x minus d * sumlx, so a larger d * sumlx is better.
        better = ((suml2 > 0) & (sumlx * sumlx > best * suml2)) | (index == 0)
        best = np.where(better, d * sumlx, best)
        scale = np.where(better, d, scale)
        codes = np.where(better[..., None], q, codes)
    codes = np.where(zero[..., None], 0, codes)
    return (codes + 8).astype(np.uint8), np.where(zero, np.float32(0), scale)


def quantize_q6_0(matrix):
    """6-bit codes and f16 scales for a float32 matrix: the low 4 bits of the codes as u4, two in
    each byte, their high 2 bits as u2, four in each byte, and one scale for each 32 weights of a
    row. A weight is (code - 32) times its scale, and the code is 0 to 63. Each group of 32 weights
    gets the codes and the scale that `make_qx_quants(32, 32, x, L, 1, NULL)` of llama.cpp chooses,
    as its Q6_K does for groups of 16. See `_q6_0_search`."""
    rows, cols = matrix.shape
    assert cols % Q6_GROUP == 0, f"{cols} columns are not groups of {Q6_GROUP}"
    low = np.empty((rows, cols // 2), dtype=np.uint8)
    high = np.empty((rows, cols // 4), dtype=np.uint8)
    scales = np.empty((rows, cols // Q6_GROUP), dtype=np.float16)
    # About 1M weights at a time on each of several threads, as the Q4_0 search.
    step = max(1, (1 << 20) // cols)

    def rows_from(start):
        chunk = np.asarray(matrix[start : start + step], dtype=np.float32)
        assert np.all(np.isfinite(chunk)), "only finite values round to 6 bits here"
        q, d = _q6_0_search(chunk.reshape(len(chunk), -1, Q6_GROUP))
        q = q.reshape(len(chunk), cols)
        low[start : start + step] = (q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)
        hi = q >> 4
        high[start : start + step] = hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)
        scales[start : start + step] = d.reshape(len(chunk), -1).astype(np.float16)

    with concurrent.futures.ThreadPoolExecutor(max(1, (os.cpu_count() or 1) - 2)) as pool:
        list(pool.map(rows_from, range(0, rows, step)))
    return low, high, scales


# The sizes in steps that the 6-bit search tries for the largest weight of a group: 32 first, then
# 31.1 to 32.9 in tenths, as `make_qx_quants` of llama.cpp tries them, in float32.
Q6_TRIES = [np.float32(32)] + [np.float32(32) + np.float32(0.1) * np.float32(i) for i in range(-9, 10) if i != 0]


def _q6_0_search(groups):
    """The codes (0 to 63) and float32 scales of `make_qx_quants(32, 32, x, L, 1, NULL)` of
    llama.cpp, for groups of shape [rows, n, 32]. The error of weight x counts x * x times. For each
    size of `Q6_TRIES`, the largest weight of the group sets the step, each weight rounds to a code
    from -32 to 31, and the codes get the scale of the smallest weighted square error. The group
    keeps the try of the smallest error."""
    weight = groups * groups
    largest = np.take_along_axis(groups, np.abs(groups).argmax(axis=-1)[..., None], axis=-1)[..., 0]
    zero = largest == 0
    largest = np.where(zero, np.float32(1), largest)
    best = np.zeros(largest.shape, dtype=np.float32)
    scale = np.zeros(largest.shape, dtype=np.float32)
    codes = np.zeros(groups.shape, dtype=np.float32)
    for index, size in enumerate(Q6_TRIES):
        q = np.clip(np.rint(groups * (-size / largest)[..., None]), -32, 31)
        sumlx = (weight * groups * q).sum(axis=-1)
        suml2 = (weight * q * q).sum(axis=-1)
        d = np.divide(sumlx, suml2, out=np.zeros_like(sumlx), where=suml2 > 0)
        better = ((suml2 > 0) & (sumlx * sumlx > best * suml2)) | (index == 0)
        best = np.where(better, d * sumlx, best)
        scale = np.where(better, d, scale)
        codes = np.where(better[..., None], q, codes)
    codes = np.where(zero[..., None], 0, codes)
    return (codes + 32).astype(np.uint8), np.where(zero, np.float32(0), scale)


def unpack_u2(packed):
    """The codes of a u2 array, one in each uint8: bits 0-1 of a byte, then 2-3, 4-5 and 6-7."""
    return np.stack([(packed >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(*packed.shape[:-1], -1)


def unpack_u4(packed):
    """The codes of a u4 array, one in each uint8: the low 4 bits of a byte, then the high 4."""
    return np.stack([packed & 15, packed >> 4], axis=-1).reshape(*packed.shape[:-1], -1)


def widen(tensors, name, rows=None):
    """The float32 values of the tensor `name` of `read_weight_file`, also of a quantized matrix.
    `rows`, a (start, stop) pair, widens only those rows of a matrix."""
    dtype, array = tensors[name]
    start, stop = rows if rows is not None else (0, len(array))
    array = array[start:stop]
    if dtype in ("bf16", "f32"):
        return as_f32((dtype, array))
    scales = tensors[name + ".scale"][1]
    if dtype == "u4" and name + ".high" in tensors:
        group_scales = np.repeat(scales[start:stop].astype(np.float32), Q6_GROUP, axis=1)
        codes = unpack_u4(array) | (unpack_u2(tensors[name + ".high"][1][start:stop]) << 4)
        return (codes.astype(np.float32) - 32) * group_scales
    if dtype == "u4":
        group_scales = np.repeat(scales[start:stop].astype(np.float32), Q4_GROUP, axis=1)
        return (unpack_u4(array).astype(np.float32) - 8) * group_scales
    assert dtype == "f8e4m3", f"{name}: cannot widen {dtype}"
    blocks = np.repeat(scales[np.arange(start, stop) // BLOCK], BLOCK, axis=1)[:, : array.shape[1]]
    return E4M3[array] * blocks


def _lazy_parts(compute, count):
    """`count` functions that return the parts of `compute()`. It runs once for all of them, and its
    result is freed when each function has returned its part."""
    result, taken = [], set()

    def part(index):
        def get():
            if not result:
                result.append(compute())
            value = result[0][index]
            taken.add(index)
            if len(taken) == count:
                result.clear()
                taken.clear()
            return value

        return get

    return tuple(part(i) for i in range(count))


def _lazy_pair(compute):
    """Two functions that return the two parts of `compute()`, as `_lazy_parts`."""
    return _lazy_parts(compute, 2)


def quantize(config, tensors, weight_format, importance=None):
    """The config and tensors of the fp8, q4_0 or q4_0_q6 file for a bf16 model. A model with tied
    embeddings gets a copy of its embedding table as `lm_head`. Each quantized matrix and its
    scales are functions, which quantize it when `write_weight_file` gets to it. `importance`, a
    dict from matrix name to the array of importance.py, guides the rounding of each Q4_0 matrix."""
    assert config["weight_format"] == WEIGHT_FORMATS["bf16"], "the model is quantized already"
    tensors = dict(tensors)
    if config["tied_embeddings"]:
        tensors["lm_head"] = tensors["embed"]
    config = dict(config, weight_format=WEIGHT_FORMATS[weight_format], tied_embeddings=0)
    for name in [n for n in tensors if is_matrix(n)]:
        dtype = matrix_dtype(config, name)
        if is_six_bit(config, name):
            parts = _lazy_parts(lambda entry=tensors[name]: quantize_q6_0(as_f32(resolve(entry))), 3)
            tensors[name] = lambda part=parts[0]: ("u4", part())
            tensors[name + ".high"] = lambda part=parts[1]: ("u2", part())
            tensors[name + ".scale"] = lambda part=parts[2]: ("f16", part())
            continue
        if dtype == "u4":
            # An importance file need not have the MTP layer, which importance.py does not run.
            assert importance is None or name in importance or name.startswith("mtp."), f"the importance file has no {name}"
            weights = None if importance is None else importance.get(name)
            quantizer, scale_dtype = (lambda matrix, weights=weights: quantize_q4_0(matrix, weights)), "f16"
        else:
            quantizer, scale_dtype = quantize_fp8, "f32"
        codes, scales = _lazy_pair(lambda entry=tensors[name], quantizer=quantizer: quantizer(as_f32(resolve(entry))))
        tensors[name] = lambda codes=codes, dtype=dtype: (dtype, codes())
        tensors[name + ".scale"] = lambda scales=scales, scale_dtype=scale_dtype: (scale_dtype, scales())
    return config, tensors


def read_safetensors(model_dir):
    """Return a dict from Hugging Face tensor name to (dtype name, numpy array) for a model folder.

    Handles one `model.safetensors` file and a set of shards listed in
    `model.safetensors.index.json`. The arrays are memory maps of the files.
    """
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    storage = {"BF16": ("bf16", np.uint16), "F32": ("f32", np.float32)}
    tensors = {}
    for file in files:
        data = np.memmap(model_dir / file, dtype=np.uint8, mode="r")
        (header_bytes,) = struct.unpack_from("<Q", data, 0)
        header = json.loads(bytes(data[8 : 8 + header_bytes]))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            assert info["dtype"] in storage, f"{name}: dtype {info['dtype']} is not supported"
            dtype, numpy_type = storage[info["dtype"]]
            begin, end = info["data_offsets"]
            raw = np.frombuffer(data, dtype=numpy_type, count=(end - begin) // np.dtype(numpy_type).itemsize,
                                offset=8 + header_bytes + begin)
            tensors[name] = (dtype, raw.reshape(info["shape"]))
    return tensors


def config_from_hugging_face(model_dir):
    """The config of the weight file, from the `config.json` of a model folder."""
    return config_from_json(json.loads((Path(model_dir) / "config.json").read_text()))


def config_from_json(hf):
    """The config of the weight file, from the `config.json` of a Qwen2, a Qwen3 MoE or a Qwen3.5
    MoE model."""
    if hf["model_type"] == "qwen3_5_moe":
        return config_from_qwen3_5(hf)
    assert hf["model_type"] in ("qwen2", "qwen3_moe"), f"model_type {hf['model_type']} is not supported yet"
    moe = hf["model_type"] == "qwen3_moe"
    if moe:
        assert hf["decoder_sparse_step"] == 1 and not hf["mlp_only_layers"], "each layer must have experts"
        assert hf["norm_topk_prob"], "the weights of the chosen experts must be scaled to sum to 1"
        assert not hf.get("attention_bias", False), "Q, K and V of a Qwen3 MoE model have no bias"
    assert hf.get("rope_scaling") is None, "RoPE scaling is not supported yet"
    eos = hf.get("eos_token_id")
    return {
        "hidden_size": hf["hidden_size"],
        "intermediate_size": hf["intermediate_size"],
        "num_layers": hf["num_hidden_layers"],
        "num_q_heads": hf["num_attention_heads"],
        "num_kv_heads": hf["num_key_value_heads"],
        "head_dim": hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
        "vocab_size": hf["vocab_size"],
        "max_positions": hf["max_position_embeddings"],
        "tied_embeddings": int(hf.get("tie_word_embeddings", False)),
        "qkv_bias": int(not moe),
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "weight_format": WEIGHT_FORMATS["bf16"],
        "num_experts": hf["num_experts"] if moe else 0,
        "experts_per_token": hf["num_experts_per_tok"] if moe else 0,
        "moe_intermediate_size": hf["moe_intermediate_size"] if moe else 0,
        "qk_norm": int(moe),
        "rope_theta": float(hf["rope_theta"]),
        "rms_norm_eps": float(hf["rms_norm_eps"]),
    }


def config_from_qwen3_5(hf):
    """The config of the weight file, from the `config.json` of a Qwen3.5 MoE model, such as
    Qwen3.6-35B-A3B. The file keeps only the text model and its MTP layer: no vision tower."""
    text = hf["text_config"]
    assert text["model_type"] == "qwen3_5_moe_text", f"text model_type {text['model_type']}"
    interval = text["full_attention_interval"]
    kinds = ["full_attention" if (l + 1) % interval == 0 else "linear_attention" for l in range(text["num_hidden_layers"])]
    assert text.get("layer_types", kinds) == kinds, "the layers must have full attention at every interval"
    assert text["attn_output_gate"], "the attention output must have a gate"
    assert not text["attention_bias"], "Q, K and V of a Qwen3.5 model have no bias"
    assert text["hidden_act"] == "silu", f"activation {text['hidden_act']}"
    assert text["linear_num_value_heads"] % text["linear_num_key_heads"] == 0, "v heads must share k heads evenly"
    rope = text["rope_parameters"]
    assert rope.get("rope_type", "default") == "default", f"RoPE type {rope.get('rope_type')}"
    head_dim = text["head_dim"]
    eos = text.get("eos_token_id")
    return {
        "hidden_size": text["hidden_size"],
        "intermediate_size": 0,
        "num_layers": text["num_hidden_layers"],
        "num_q_heads": text["num_attention_heads"],
        "num_kv_heads": text["num_key_value_heads"],
        "head_dim": head_dim,
        "vocab_size": text["vocab_size"],
        "max_positions": text["max_position_embeddings"],
        "tied_embeddings": int(hf.get("tie_word_embeddings", text.get("tie_word_embeddings", False))),
        "qkv_bias": 0,
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "weight_format": WEIGHT_FORMATS["bf16"],
        "num_experts": text["num_experts"],
        "experts_per_token": text["num_experts_per_tok"],
        "moe_intermediate_size": text["moe_intermediate_size"],
        "qk_norm": 1,
        "arch": ARCHS["qwen3_5"],
        "full_attention_interval": interval,
        "linear_k_heads": text["linear_num_key_heads"],
        "linear_v_heads": text["linear_num_value_heads"],
        "linear_k_dim": text["linear_key_head_dim"],
        "linear_v_dim": text["linear_value_head_dim"],
        "conv_width": text["linear_conv_kernel_dim"],
        # The text model gives the same position to the three sections of the multimodal RoPE, so
        # it is plain RoPE on the first rotary_dim values of each head.
        "rotary_dim": int(head_dim * rope.get("partial_rotary_factor", 1.0)),
        "shared_expert_intermediate_size": text["shared_expert_intermediate_size"],
        "mtp_layers": text.get("mtp_num_hidden_layers", 0),
        "rope_theta": float(rope["rope_theta"]),
        "rms_norm_eps": float(text["rms_norm_eps"]),
    }
