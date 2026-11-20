#!/usr/bin/env python3
"""Compute the logits and the greedy tokens of a Qwen2, Qwen3 MoE or Qwen3.5 MoE model with NumPy,
in float32.

    python3 scripts/reference.py qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
    python3 scripts/reference.py qwen-0.5b.vxw --tokenizer DIR --prompt "def f(n):"
    python3 scripts/reference.py qwen-0.5b.vxw --tensors
    python3 scripts/reference.py qwen-0.5b.vxw --tokenizer DIR --perplexity FILE
    python3 scripts/reference.py tiny-qwen35.vxw --tokens 1,2,3 --dump-layer 0 --dump-dir DIR

--dump-layer writes the stages of one Gated DeltaNet layer for each position, as NumPy files in
--dump-dir: `pos<P>_<stage>.npy`, for the stages in DUMP_STAGES.

It reads the weight file that `convert.py` writes, so the Vx program and this script use the same
bytes. Tests of the Vx program compare with the lines it prints:

    step=0 token=715 logit=16.3279 margin=0.1044;

Step 0 is the token after the prompt. Each later step feeds the previous token back in.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import os
import sys

import numpy as np

import qwen_weights

# Below this size, the widened float32 copies of all the weights stay in memory between tokens.
CACHE_LIMIT_BYTES = 4 << 30

# The stages of a Gated DeltaNet layer that --dump-layer writes, for one position each:
#   input     the layer's input after attn_norm [hidden]
#   conv_out  q, k and v after the convolution and SiLU [2K + V]
#   q, k      after the L2 norm of each head, q also times 1 / sqrt(linear_k_dim) [v heads, k dim]
#   beta, g   [v heads]
#   state     the state after the update [v heads, k dim, v dim]
#   o         the output of each head before its RMSNorm [v heads, v dim]
#   output    the layer's output, after linear_out [hidden]
DUMP_STAGES = ("input", "conv_out", "q", "k", "beta", "g", "state", "o", "output")


class Model:
    def __init__(self, path):
        self.config, self.tensors = qwen_weights.read_weight_file(path)
        f32_bytes = sum(array.size * (8 if dtype == "u4" else 4) for dtype, array in self.tensors.values())
        self.cache = {} if f32_bytes <= CACHE_LIMIT_BYTES else None
        # A function that gets each input of `product`, as importance.py uses it, or None.
        self.see_input = None
        # The layer and folder of --dump-layer, or None.
        self.dump_layer, self.dump_dir = None, None

    def weight(self, name, rows=None):
        """The float32 values of a tensor, or of the rows (start, stop) of a matrix."""
        if self.cache is None:
            return qwen_weights.widen(self.tensors, name, rows)
        if name not in self.cache:
            self.cache[name] = qwen_weights.widen(self.tensors, name)
        return self.cache[name] if rows is None else self.cache[name][rows[0] : rows[1]]


def product(model, x, name, rows=None):
    """x times the transpose of the matrix `name`, or of its rows (start, stop)."""
    if model.see_input is not None:
        model.see_input(name, rows, x)
    return x @ model.weight(name, rows).T


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope(x, positions, theta):
    """Rotate each head of x, shaped [tokens, heads, head_dim], by its position.

    The first half of a head pairs with the second half, as in Hugging Face's `rotate_half`.
    """
    half = x.shape[-1] // 2
    inv_freq = 1.0 / theta ** (np.arange(half, dtype=np.float64) * 2 / x.shape[-1])
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    first, second = x[..., :half], x[..., half:]
    return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def partial_rope(x, positions, theta, rotary_dim):
    """`rope` on the first rotary_dim values of each head of x. The other values stay."""
    if rotary_dim == x.shape[-1]:
        return rope(x, positions, theta)
    return np.concatenate([rope(x[..., :rotary_dim], positions, theta), x[..., rotary_dim:]], axis=-1)


def mixture_of_experts(model, prefix, h):
    """The sum of the experts that the router chooses for each token of h, each times its weight.

    The router keeps the experts_per_token largest of the softmax over all experts, and scales
    their weights to sum to 1. A model with a shared expert also adds it for every token, times
    sigmoid of the last row of the router.
    """
    c = model.config
    hidden, inter = c["hidden_size"], c["moe_intermediate_size"]
    logits = h @ model.weight(prefix + "router").T
    scores = logits[:, : c["num_experts"]]
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores = scores / scores.sum(axis=-1, keepdims=True)
    chosen = np.argsort(-scores, axis=-1, kind="stable")[:, : c["experts_per_token"]]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = np.zeros_like(h)
    for expert in np.unique(chosen):
        tokens, slots = np.nonzero(chosen == expert)
        gate_up = product(model, h[tokens], prefix + "experts_gate_up", (expert * 2 * inter, (expert + 1) * 2 * inter))
        act = silu(gate_up[:, :inter]) * gate_up[:, inter:]
        down = product(model, act, prefix + "experts_down", (expert * hidden, (expert + 1) * hidden))
        out[tokens] += weights[tokens, slots][:, None] * down
    shared = c.get("shared_expert_intermediate_size", 0)
    if shared:
        gate_up = product(model, h, prefix + "shared_gate_up")
        act = silu(gate_up[:, :shared]) * gate_up[:, shared:]
        out += sigmoid(logits[:, c["num_experts"] :]) * product(model, act, prefix + "shared_down")
    return out


class Cache:
    """The keys and values of every position so far, per layer, rounded to f16 as in the Vx program.
    A Gated DeltaNet layer has instead the last conv_width - 1 inputs of its convolution and its
    state, in f32."""

    def __init__(self, config):
        layers = config["num_layers"]
        self.keys = [None] * layers
        self.values = [None] * layers
        self.conv = [None] * layers
        self.state = [None] * layers
        self.length = 0
        for layer in range(layers):
            if not qwen_weights.has_full_attention(config, layer):
                k_total = config["linear_k_heads"] * config["linear_k_dim"]
                channels = 2 * k_total + config["linear_v_heads"] * config["linear_v_dim"]
                self.conv[layer] = np.zeros((config["conv_width"] - 1, channels), dtype=np.float32)
                shape = (config["linear_v_heads"], config["linear_k_dim"], config["linear_v_dim"])
                self.state[layer] = np.zeros(shape, dtype=np.float32)


def attend(cache, layer, q, k, v, positions):
    """Add k and v to the cache of `layer`, and return the attention of q over all positions so
    far, shaped [tokens, query heads * head_dim]. q is [tokens, query heads, head_dim], and k
    and v are [tokens, key-value heads, head_dim]."""
    count, qh, hd = q.shape
    # The KV cache of the Vx program holds f16 values, so round them as it does.
    k = k.astype(np.float16).astype(np.float32)
    v = v.astype(np.float16).astype(np.float32)
    if cache.keys[layer] is None:
        cache.keys[layer], cache.values[layer] = k, v
    else:
        cache.keys[layer] = np.concatenate([cache.keys[layer], k])
        cache.values[layer] = np.concatenate([cache.values[layer], v])
    keys, values = cache.keys[layer], cache.values[layer]
    # Query head i uses key-value head i // group: grouped-query attention.
    group = qh // k.shape[1]
    scores = np.einsum("tgd,sgd->gts", q, np.repeat(keys, group, axis=1)) / np.float32(np.sqrt(hd))
    visible = np.arange(keys.shape[0])[None, :] <= positions[:, None]
    scores = np.where(visible[None, :, :], scores, -np.inf)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return np.einsum("gts,sgd->tgd", scores, np.repeat(values, group, axis=1)).reshape(count, qh * hd)


def gated_attention(model, prefix, h, positions, cache, layer):
    """The full attention of an arch 1 layer: q and k get an RMSNorm per head and RoPE on their
    first rotary_dim values, and each head's output is multiplied by sigmoid of its gate."""
    c = model.config
    hd, qh, kvh, eps = c["head_dim"], c["num_q_heads"], c["num_kv_heads"], c["rms_norm_eps"]
    count = len(h)
    qkv = product(model, h, prefix + "qkv")
    q = qkv[:, : qh * hd].reshape(count, qh, hd)
    k = qkv[:, qh * hd : (qh + kvh) * hd].reshape(count, kvh, hd)
    v = qkv[:, (qh + kvh) * hd : (qh + 2 * kvh) * hd].reshape(count, kvh, hd)
    gate = qkv[:, (qh + 2 * kvh) * hd :]
    q = partial_rope(rms_norm(q, model.weight(prefix + "q_norm"), eps), positions, c["rope_theta"], c["rotary_dim"])
    k = partial_rope(rms_norm(k, model.weight(prefix + "k_norm"), eps), positions, c["rope_theta"], c["rotary_dim"])
    attended = attend(cache, layer, q, k, v, positions) * sigmoid(gate)
    return product(model, attended, prefix + "attn_out")


def delta_net(model, prefix, h, cache, layer):
    """The Gated DeltaNet of an arch 1 layer, one position after the other.

    Each value head j keeps a state S of [k dim, v dim], and reads key head j // (v heads / k heads).
    For each position, S decays by exp(g), then takes the part of v that S does not already give
    for k, times beta: S += k (beta (v - k S)). The output of the head is q S.
    """
    c = model.config
    kh, vh, kd, vd = c["linear_k_heads"], c["linear_v_heads"], c["linear_k_dim"], c["linear_v_dim"]
    k_total, v_total, width = kh * kd, vh * vd, c["conv_width"]
    count = len(h)
    mixed = product(model, h, prefix + "linear_in")
    z = mixed[:, 2 * k_total + v_total : 2 * k_total + 2 * v_total].reshape(count, vh, vd)
    b = mixed[:, 2 * k_total + 2 * v_total : 2 * k_total + 2 * v_total + vh]
    a = mixed[:, 2 * k_total + 2 * v_total + vh :]
    # The causal convolution of each channel over this position and the width - 1 before it.
    conv = model.weight(prefix + "conv")
    padded = np.concatenate([cache.conv[layer], mixed[:, : 2 * k_total + v_total]])
    cache.conv[layer] = padded[count:]
    conv_out = silu(np.stack([(padded[t : t + width] * conv.T).sum(axis=0) for t in range(count)]))
    q = conv_out[:, :k_total].reshape(count, kh, kd)
    k = conv_out[:, k_total : 2 * k_total].reshape(count, kh, kd)
    v = conv_out[:, 2 * k_total :].reshape(count, vh, vd)
    q = q / np.sqrt((q * q).sum(axis=-1, keepdims=True) + np.float32(1e-6)) / np.float32(np.sqrt(kd))
    k = k / np.sqrt((k * k).sum(axis=-1, keepdims=True) + np.float32(1e-6))
    q, k = np.repeat(q, vh // kh, axis=1), np.repeat(k, vh // kh, axis=1)
    beta = sigmoid(b)
    g = model.weight(prefix + "neg_A") * np.logaddexp(np.float32(0), a + model.weight(prefix + "dt_bias"))
    state = cache.state[layer]
    out = np.empty((count, vh, vd), dtype=np.float32)
    for t in range(count):
        state = state * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum("hk,hkv->hv", k[t], state)) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hk,hkv->hv", q[t], state)
        if model.dump_layer == layer:
            stages = dict(conv_out=conv_out[t], q=q[t], k=k[t], beta=beta[t], g=g[t], state=state, o=out[t])
            dump(model, cache.length + t, stages)
    cache.state[layer] = state
    normed = rms_norm(out, model.weight(prefix + "linear_norm"), c["rms_norm_eps"]) * silu(z)
    return product(model, normed.reshape(count, v_total), prefix + "linear_out")


def dump(model, position, stages):
    for stage, value in stages.items():
        np.save(os.path.join(model.dump_dir, f"pos{position}_{stage}.npy"), np.asarray(value, dtype=np.float32))


def forward_qwen3_5(model, tokens, cache):
    """`forward` for an arch 1 model."""
    c = model.config
    eps = c["rms_norm_eps"]
    count = len(tokens)
    positions = np.arange(cache.length, cache.length + count)
    # Widen only the rows of the tokens: the whole table of a real model is 2 GB in f32.
    x = np.stack([qwen_weights.widen(model.tensors, "embed", (t, t + 1))[0] for t in tokens])
    for layer in range(c["num_layers"]):
        p = f"layers.{layer}."
        h = rms_norm(x, model.weight(p + "attn_norm"), eps)
        if qwen_weights.has_full_attention(c, layer):
            out = gated_attention(model, p, h, positions, cache, layer)
        else:
            out = delta_net(model, p, h, cache, layer)
            if model.dump_layer == layer:
                for t in range(count):
                    dump(model, cache.length + t, dict(input=h[t], output=out[t]))
        x = x + out
        x = x + mixture_of_experts(model, p, rms_norm(x, model.weight(p + "mlp_norm"), eps))
    cache.length += count
    x = rms_norm(x, model.weight("final_norm"), eps)
    return product(model, x, "embed" if c["tied_embeddings"] else "lm_head").astype(np.float32)


def forward(model, tokens, cache):
    """Run `tokens` after the `cache.length` tokens already in the cache. Return their logits."""
    c = model.config
    if c.get("arch", 0):
        return forward_qwen3_5(model, tokens, cache)
    hd, qh, kvh = c["head_dim"], c["num_q_heads"], c["num_kv_heads"]
    eps = c["rms_norm_eps"]
    count = len(tokens)
    positions = np.arange(cache.length, cache.length + count)
    x = model.weight("embed")[np.asarray(tokens)]
    for layer in range(c["num_layers"]):
        p = f"layers.{layer}."
        h = rms_norm(x, model.weight(p + "attn_norm"), eps)
        qkv = product(model, h, p + "qkv")
        if c["qkv_bias"]:
            qkv = qkv + model.weight(p + "qkv_bias")
        q = qkv[:, : qh * hd].reshape(count, qh, hd)
        k = qkv[:, qh * hd : (qh + kvh) * hd].reshape(count, kvh, hd)
        v = qkv[:, (qh + kvh) * hd :].reshape(count, kvh, hd)
        if c["qk_norm"]:
            # One RMSNorm over each head, with one weight vector for all the heads of q or of k.
            q = rms_norm(q, model.weight(p + "q_norm"), eps)
            k = rms_norm(k, model.weight(p + "k_norm"), eps)
        q = rope(q, positions, c["rope_theta"])
        k = rope(k, positions, c["rope_theta"])
        x = x + product(model, attend(cache, layer, q, k, v, positions), p + "attn_out")
        h = rms_norm(x, model.weight(p + "mlp_norm"), eps)
        if c["num_experts"]:
            x = x + mixture_of_experts(model, p, h)
        else:
            gate_up = product(model, h, p + "gate_up")
            gate, up = gate_up[:, : c["intermediate_size"]], gate_up[:, c["intermediate_size"] :]
            x = x + product(model, silu(gate) * up, p + "down")
    cache.length += count
    x = rms_norm(x, model.weight("final_norm"), eps)
    return product(model, x, "embed" if c["tied_embeddings"] else "lm_head").astype(np.float32)


def generate(model, prompt, steps):
    """Greedy decoding. Return a list of (token, logits) for each step."""
    cache = Cache(model.config)
    logits = forward(model, prompt, cache)[-1]
    result = []
    for _ in range(steps):
        token = int(np.argmax(logits))
        result.append((token, logits))
        logits = forward(model, [token], cache)[-1]
    return result


def perplexity(model, tokens):
    """exp of the mean negative log-likelihood of each token after the first, given the ones
    before it."""
    logits = forward(model, tokens, Cache(model.config)).astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    chosen = log_probs[np.arange(len(tokens) - 1), np.asarray(tokens[1:])]
    return float(np.exp(-chosen.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a weight file from convert.py or make_tiny_model.py")
    parser.add_argument("--tokens", help="the prompt as token ids, separated by commas")
    parser.add_argument("--prompt", help="the prompt as text; needs --tokenizer")
    parser.add_argument("--tokenizer", help="a Hugging Face model folder that has tokenizer.json")
    parser.add_argument("--steps", type=int, default=8, help="the number of tokens to generate")
    parser.add_argument("--tensors", action="store_true", help="print the first codes and the sum of the codes of each tensor")
    parser.add_argument("--perplexity", help="a text file whose perplexity to print; needs --tokenizer")
    parser.add_argument("--max-tokens", type=int, default=512, help="the number of tokens of --perplexity to use")
    parser.add_argument("--dump-layer", type=int, help="a Gated DeltaNet layer whose stages to write; needs --dump-dir")
    parser.add_argument("--dump-dir", help="the folder for the files of --dump-layer")
    args = parser.parse_args()
    model = Model(args.weights)
    if args.dump_layer is not None:
        assert args.dump_dir, "--dump-layer needs --dump-dir"
        assert not qwen_weights.has_full_attention(model.config, args.dump_layer), "--dump-layer needs a Gated DeltaNet layer"
        os.makedirs(args.dump_dir, exist_ok=True)
        model.dump_layer, model.dump_dir = args.dump_layer, args.dump_dir

    if args.tensors:
        # Bits, not decimals, so that a Vx program can print the same line exactly. A code is the
        # bits of one value as an unsigned integer: 16 for bf16 and f16, 8 for f8e4m3 and 32 for
        # f32. For u4, it is a byte of two codes. The sum adds the codes of the whole tensor.
        c = model.config
        print(f"config hidden={c['hidden_size']} layers={c['num_layers']} vocab={c['vocab_size']};")
        for name, (dtype, array) in model.tensors.items():
            codes = array.reshape(-1).view(f"u{array.dtype.itemsize}")
            first = ",".join(str(int(v)) for v in codes[:4])
            print(f"{name} count={codes.size} codes={first} sum={int(codes.sum(dtype=np.uint64))};")
        return 0

    if args.perplexity is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
        with open(args.perplexity) as text:
            tokens = tokenizer.encode(text.read()).ids[: args.max_tokens]
        print(f"perplexity={perplexity(model, tokens):.4f} tokens={len(tokens)};")
        return 0

    if args.prompt is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
        prompt = tokenizer.encode(args.prompt).ids
    else:
        assert args.tokens, "give the prompt with --tokens or --prompt"
        prompt = [int(t) for t in args.tokens.split(",")]
    print("prompt=" + ",".join(map(str, prompt)) + ";")
    for step, (token, logits) in enumerate(generate(model, prompt, args.steps)):
        # The margin is how far the chosen logit is above the next one. A small margin means that
        # a small rounding difference in another program can choose another token.
        margin = logits[token] - np.partition(logits, -2)[-2]
        print(f"step={step} token={token} logit={logits[token]:.4f} margin={margin:.4f};")
    return 0


if __name__ == "__main__":
    sys.exit(main())
