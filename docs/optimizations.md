# A journal of the optimizations

This journal records each change that made the Qwen app faster or smaller, with the measurement
that led to it and the result. It also records the changes that did not help, so that nobody
tries them again without a new reason. The details of each entry are in its pull request or issue.

All the numbers are from an M4 Mac with 32 GB of unified memory and a 10-core GPU, unless an entry
says otherwise. The main model is Qwen3-Coder-30B-A3B in Q4_0 (17.8 GB), a mixture of experts
that uses 8 of its 128 experts in each layer for each token.

## The state now

| | Time | Notes |
| --- | --- | --- |
| A generated token, short context | 23.4 ms (about 43 tokens a second) | The floor at the speed of memory is 18.4 ms |
| A generated token, 2000 tokens of context | about 30 ms | Measured with the f32 KV cache, before #31 halved its bytes |
| A prompt token, 2000-token prompt | 8.1 ms (7.0 ms without the first compile) | 7.2 ms with #30, which waits for a Vx release |
| The start of a session | about 60 s | 51 s of it is the load of the weights (#27) |
| The KV cache | 98 KB a token | f16 since #31; it was 196 KB |

llama.cpp 0.7.0 took 24.3 ms for a generated token on the same Mac, in `llama-bench -n 128`
([#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596324)). The two were not
measured side by side since, so compare them again before a claim.

## How to measure

- **A profile by subtraction.** Run the same tokens with one part of a layer left out, and
  subtract. The tokens are wrong, but the time is right. `decode/profile_decode.vx` does this for
  a generated token. The error of each part is about ±0.5 ms.
- **The limit of memory.** `decode/bandwidth.m` reads 1 GiB with a Metal kernel. The M4 reads at
  104 GB/s, not the 120 GB/s of its spec. Divide the bytes of a part by its time, and compare with
  104 GB/s.
- **The same output.** Compare the tokens after a prompt with `scripts/compare.py`, against the old
  program or against `scripts/reference.py`. With greedy decoding, the tokens must be equal, and
  the logits close.
- **A quiet Mac.** The 17.8 GB of weights leave little free memory on a Mac with 32 GB. When other
  programs push it into swap, a token takes 2 to 4 times as long, and the times of a comparison
  mean nothing. Close other large programs, run one model at a time, and alternate the old and the
  new program, because each run after the first can be slower.

## The journal

### 1. Q4_0 weights, so that the MoE model fits

**Problem.** Qwen3-Coder-30B-A3B is 61 GB in bf16 and 31 GB in fp8. Neither fits in 32 GB.

**Change.** `convert.py --q4` stores each matrix except `lm_head` in the Q4_0 format of
llama.cpp: a 4-bit code for each weight and an f16 scale for each 32 weights. `lm_head` stays fp8,
because its errors go straight into the logits. Of four 4-bit formats, Q4_0 was the closest to
bf16 on the 7B model ([#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596230)).

**Result.** 17.8 GB. A token reads about 1.9 GB.

### 2. An importance matrix for the rounding (#2)

**Problem.** The Q4_0 file was 1.30% above bf16 in perplexity, and the file of llama.cpp 0.76%
(#8).

**Change.** `importance.py` measures how much each input of each matrix matters over a sample
text. `convert.py --q4 --importance` then picks the scale of each group of 32 weights with the
smallest weighted error, as llama.cpp does.

**Result.** The perplexity on `MetalImage.cpp` went from 3.7290 to 3.6881 (bf16: 3.6813). The
importance matrix removed 86%, 46% and 23% of the gap to bf16 on the three test files. This
changes the quality, not the speed.

### 3. Cheaper kernel launches (Vx)

**Problem.** A token of the MoE model runs about 477 kernels, and each launch cost much time on
the CPU. With the Metal runtime of Vx main at that time, a token took 119 ms.

**Change.** vx-lang/Vx#3152, vx-lang/Vx#3165 and vx-lang/Vx#3168 made a launch cheaper in the
Metal runtime of Vx.

**Result.** 119 ms to 28.0 ms a token, 4.3 times as fast. Now 530 launches of a kernel that does
almost nothing take 1.1 ms, so the launches are a small part of a token.

### 4. The prompt in chunks (#16)

**Problem.** The program ran the prompt one token at a time, and each token read all the weights
that it used: 46.7 ms a prompt token on the 30B model.

**Change.** `src/prefill.vx` runs the prompt in chunks, so that a chunk reads each weight once for
all its tokens. A product decodes a tile of weights into shared memory, and multiplies it with
the inputs of 16 tokens with `@`. For the experts, the CPU sorts the tokens of a chunk by expert,
so that a tile of 16 tokens reads the weights of one expert.

**Result.** 16.7 ms a prompt token at 1000 tokens, 2.8 times as fast. The 0.5B model went from
11.1 ms to 1.26 ms, 8.8 times as fast. The same tokens after the prompt.

### 5. Attention of a generated token with positions split among blocks (#23)

**Problem.** The `nn_attention` of Vx v0.0.4 has one GPU block for each query head, and each block
reads all the positions of the KV cache. A token took 57.3 ms at 2000 positions, against 26.7 ms
at 5.

**Change.** `src/attention.vx` is a copy of the `nn_attention` of Vx main (vx-lang/Vx#3216),
which splits the positions among blocks and then merges the parts.

**Result.**

| Positions | Before | After |
| --- | --- | --- |
| 5 | 26.7 ms | 25.4 ms |
| 1000 | 40.7 ms | 27.5 ms |
| 2000 | 57.3 ms | 30.8 ms |

### 6. A session and a cache of the prompt (#25)

**Problem.** Each request ran the whole prompt again. An agent sends the same system prompt and
the conversation so far with each turn.

**Change.** `src/main.vx --session` keeps the model and the KV cache between requests. It runs only
the tokens after the longest prefix that the cache already holds, as the llama.cpp server does.
`save` and `load` write the KV cache of a prompt to a file, so that a system prompt runs once,
ahead of time.

**Result.** For a system prompt of 2465 tokens:

| Turn | Tokens that ran | Prompt time |
| --- | --- | --- |
| First, no cache | 2465 | 50.0 s |
| Second, same session | 29 | 2.3 s |
| New session after `load` | 20 | 2.5 s |

### 7. Attention of the prompt as matmuls of tiles, and chunks of 512 (#28)

**Problem.** A profile by subtraction of a 2000-token prompt showed that attention took half the
time, and that its time grew with each chunk: 0.3 s in the first chunk, 2.9 s in the seventh.
Each block did one token, so each chunk read the whole KV cache once for each token, in each
layer.

**Change.**
- Attention in three passes: the scores as a matmul of the queries and the keys, a softmax, and a
  matmul of the weights and the values. The rows of a tile are the 8 query heads of one key-value
  head for 2 tokens, and the matmuls skip the positions after the tokens of the tile.
- Chunks of 512 tokens in place of 256. Each expert pads its tokens to a multiple of 16, and in a
  chunk of 256 the padding was a third of the rows.
- The weights in shared memory as [column][row], so that the 32 lanes of a subgroup write to
  different banks.

**Result.** 16.2 ms to 8.1 ms a prompt token at 2000 tokens (7.0 ms without the first compile).
Chunks of 256, 512 and 1024 gave 9.0, 7.6 and 10.2 ms. The gate_up product took 14.5 ms in place
of 16.0 ms with the new layout.

### 8. Q4_0 products with f16 tiles and sums in registers (#30, draft)

**Problem.** After #28, most of a chunk is in the products of the experts (#19). In Vx v0.0.4,
`acc = acc + a @ b` loads the sum from shared memory and stores it back for each 32 columns. The
32 KB of shared memory holds no larger tile, and each other tile shape was slower.

**Change.**
- In Vx: vx-lang/Vx#3477 keeps the sums of `@` in the simdgroup matrices of Metal, when
  `lanes::fill` starts them and `lanes::copy` writes them out.
- In the app: `pf_mm_q4_set` uses that, with the weights and the inputs as f16 tiles. It has 9 KB
  of shared memory in place of 26 KB.

**Result.** 9.3 ms to 7.2 ms a prompt token at 2000 tokens, with the same 40 tokens after it. The
gate_up product alone: 17.0 ms to 12.0 ms. Each of the two changes alone gave little (f16 tiles:
14.5 to 12.4 ms; sums in registers: 14.5 to 11.4 ms), and both gave 8.7 ms. This waits for a Vx
release with vx-lang/Vx#3477, because CI uses v0.0.4.

### 9. A KV cache of f16 (#31)

**Problem.** The KV cache took 196 KB for each token in f32: 48 layers, K and V, 4 heads of 128
values. A Vx guide of 13,000 tokens (#29) took 2.6 GB of cache, on top of the 17.8 GB of weights.

**Change.** The kernels compute in f32, round only the K and V that they store to f16, and read
them back as f32. The cache files of a session are version 2, with f16 rows.

**Result.** 98 KB a token: 1.3 GB for a guide, and cache files half the size. The same 40 tokens
after a 2000-token prompt, with a median logit difference of 0.04. Attention reads half the bytes:
at 2000 tokens of context, 196 MB in place of 392 MB for each generated token. In runs with swap,
the f16 cache was faster in each pair (for example 76 ms against 92 ms at 8000 tokens), but those
runs were not quiet enough for a clean number.

### 10. A profile of a generated token (#35)

**Problem.** It was not clear where the 23 to 30 ms of a generated token went, or how far it was
from the limit of the hardware.

**Change.** `decode/profile_decode.vx` and `decode/bandwidth.m` (see "How to measure").

**Result.** At a short context:

| Part | Bytes a token | Time | Speed |
| --- | --- | --- | --- |
| Experts, 8 of 128 in each layer | 1,019 MB | 10.7 ms | 96 GB/s |
| QKV | 283 MB | 2.7 ms | 104 GB/s |
| lm_head, fp8 | 311 MB | 2.5 ms | about 104 GB/s |
| Attention output | 227 MB | 2.3 ms | 100 GB/s |
| Attention, 100 to 400 positions | about 47 MB | 1.5 ms | about 31 GB/s |
| Router and top-k | 25 MB | 0.5 ms | 54 GB/s |

The kernels that read the weights already run at 92% to 100% of the speed of memory. A token reads
about 1.9 GB, so its floor is 18.4 ms, and it takes 23.4 ms. The rest is small kernels and the time
between about 530 kernels that each wait for the one before (#34).

## What did not help

- **Other tile shapes for the Q4_0 products of a chunk, in Vx v0.0.4** (#19). Tiles of 32 tokens,
  64 columns, or 1 or 2 subgroups were all slower (14.7 to 56.4 ms against 14.5 ms). One `@` of a
  32 x 32 tile of inputs took 4 times as long as two of 16 x 32, probably because 16
  `simdgroup_float8x8` sums leave the registers.
- **Chunks of 1024 tokens.** 10.2 ms a prompt token, against 7.6 ms for 512.
- **Speculative decoding** (#18). The check of several tokens in one pass costs too much on this
  GPU:
  - A draft model (Qwen2.5-Coder-0.5B) takes about 11 ms a token, half the time of the 30B model,
    because the 30B model reads only 1.9 GB a token. A draft must be much faster than the model
    to help.
  - Drafts from the context (prompt lookup) cost nothing. A check of up to 8 drafts kept 5.8
    tokens on an edit of a script, but only 1.1 to 1.4 tokens on new programs.
  - The tokens of a check use different experts: 9 tokens in a row use about 35 experts of a
    layer, not 8.
  - The kernels of the prompt took 106 ms for a check of 1 token, and 207 ms for 9.
  - The M4 GPU has no separate matrix units: `simdgroup_multiply_accumulate` runs on the same ALUs
    as all other work. So each token of a check adds about one multiply-add for each weight. A
    Q4_0 product for 8 tokens took 0.24 ms in Vx v0.0.4 and 0.105 ms in Vx main with tiles of 8
    tokens, against 0.046 ms for one token.
  - With those kernels, an edit would be about 1.3 times as fast, and new code slower. Look again
    on a Mac with matrix units, or for a model where a token costs more.
- **Weights decoded once into registers for several tokens.** In Vx v0.0.4 these kernels were
  slower than the kernels that decode each weight with masks for each token. Their 64 to 128
  floats a lane likely leave the registers.

## Open work

- **The load of the weights** (#27): 51 s, because the program copies each tensor two times. A load
  with no copy needs vx-lang/Vx#3498 (a tensor over memory that the program does not own) and
  vx-lang/Vx#3499 (the Metal runtime gives a mapped file to the GPU once). It would also let macOS
  drop the pages of the weights when other programs need memory.
- **Small kernels** (#34): up to about 5 ms of a 23.4 ms token. Merge the norms into the products
  after them, and the router into top-k, and measure each change with a Metal trace.
- **Fewer bytes for a token** (#33): lm_head in Q4_0 (#9) would save about 1.3 ms, after a check of
  the perplexity.
- **A short turn at a long context** (#26): a turn of 20 tokens still runs almost every expert of
  every layer, so it takes about 2 s.
