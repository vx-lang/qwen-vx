# A journal of the optimizations

This journal records each change that made the Qwen app faster or smaller, with the measurement
that led to it and the result. It also records the changes that did not help, so that nobody
tries them again without a new reason. The details of each entry are in its pull request or issue.

All the numbers are from an M4 Mac with 32 GB of unified memory and a 10-core GPU, unless an entry
says otherwise. The main model is Qwen3-Coder-30B-A3B in Q4_0 (17.8 GB), a mixture of experts
that uses 8 of its 128 experts in each layer for each token.

## The state now

Measured on 2026-11-18 and 2026-11-19 on a quiet Mac, with the drafts of entry 14 on top of
entries 12 and 13, and the vxc of Vx `main` with vx-lang/Vx#3667 and #3673, next to llama.cpp 0.7.0
(build 11669). All the numbers are for one user and one sequence, and greedy decoding.

Generated tokens a second, 256 tokens after a prompt of text from this repository, no chat
template, the best of two rounds in turn (three at 16,000):

| Context | Qwen-vx | llama.cpp |
| --- | --- | --- |
| 60 tokens | 46.8 | 45.0 |
| 2,014 tokens | 41.3 | 40.6 |
| 8,005 tokens | 31.8 | 31.6 |
| 16,026 tokens | 22.5 | 22.4 |

The prompt itself, tokens a second: 113 against 224 at 60 tokens, 369 against 389 at 2,014, 264
against 267 at 8,005, and 176 against 178 at 16,026.

With drafts from the context (`--draft 7`, entry 14), 12 code prompts, 512 tokens each, ms a
token (geometric mean of 3 prompts of each kind, two rounds):

| Prompts | Qwen-vx | Qwen-vx `--draft 7` | llama.cpp | `llama-lookup` |
| --- | --- | --- | --- | --- |
| Edit a file | 24.06 | 13.13 | 25.12 | 12.52 |
| Complete code | 23.64 | 23.62 | 25.33 | 26.27 |
| Write new code | 21.78 | 21.82 | 23.38 | 29.30 |
| Explain code | 23.69 | 21.71 | 25.09 | 26.15 |
| All 12 | 23.27 | 19.58 | 24.72 | 22.41 |

| | Qwen-vx | llama.cpp | Notes |
| --- | --- | --- | --- |
| The start of a session | about 60 s | | The load of the weights is 10.5 s since #53 (#27) |
| The KV cache | 98 KB a token | | f16 since #31; it was 196 KB |

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

### 11. Weights read in place from the mapped file (#27)

**Problem.** `load_weights` copied each of the 628 tensors from the mapped file into a host tensor,
and the transfer to GPU memory copied it again. The load took 29 to 38 s, and the program had a
private copy of the weights: 18.9 GB.

**Change.**
- In Vx: vx-lang/Vx#3606. A transfer to GPU memory of a view of a read-only, shared mapping of a
  file makes no copy: the runtime maps the same pages of the file again, and a write changes only
  the copy. A Metal residency set keeps all GPU memory resident. It also made a generated token
  1.3 to 1.9 ms faster at 2,000 and 8,000 tokens of context (`decode/faults/`).
- In the app: `load_next`, `load_codes`, `load_f16` and `load_scales` transfer a view of the
  mapped file. `copy_out` and its `msync` are gone.

**Result.** The load takes about 10.5 s, and private memory is 0.3 GB at a short context. A
generated token takes the same time as with copied weights (within 0.1 ms at 0 and 2,000 tokens
of context, and ±0.4 ms at 8,000), and the tokens are the same.

### 12. Eight tokens before each read (#34)

**Problem.** Text mode read each token on the CPU before it gave the GPU the next token. A read
waits for all the work that the GPU was given, so the GPU then had nothing to do: about 0.37 ms of
a 21.9 ms token. GPU timestamps of each command buffer split it into 0.17 ms until the CPU wakes
up after the GPU ends, and 0.20 ms until the first command buffer of the next token starts on the
GPU. The 34 command buffers of a token have only 0.03 ms between them.

**Change.** The input of a token is the argmax of the token before it, in GPU memory, so the GPU
can run several tokens without the CPU. Text mode runs 8 tokens and then reads them all. Up to 7
tokens after the end of the text are run and thrown away. `--lines` prints the logit of each
token, so it still reads each token before it runs the next.

**Result.** 0.25 to 0.32 ms less a token, and the same text. 256 tokens of three chat prompts,
two runs of each in turn:

| Prompt | Read each token | 8 tokens before each read |
| --- | --- | --- |
| Write a module and its tests | 22.13, 22.19 ms | 21.85, 21.82 ms |
| Edit a 60-line file | 24.30, 24.27 ms | 23.98, 23.99 ms |
| Explain a function | 23.69, 23.76 ms | 23.45, 23.50 ms |

### 13. Q4_0 products of a token: 2 SIMD groups of 4 rows in each block (#56)

**Problem.** GPU timestamps of each kernel of a generated token (one encoder for each kernel)
showed the products below the 104 GB/s of a plain read:

| Product | Bytes a token | Time | Speed | At 104 GB/s |
| --- | --- | --- | --- | --- |
| Experts, gate and up | 680 MB | 7.52 ms | 90 GB/s | 6.53 ms |
| Experts, down | 340 MB | 3.65 ms | 93 GB/s | 3.27 ms |
| QKV | 283 MB | 3.09 ms | 92 GB/s | 2.72 ms |
| Attention output | 227 MB | 2.65 ms | 86 GB/s | 2.18 ms |
| lm_head, 6 bits | 253 MB | 2.40 ms | 105 GB/s | 2.43 ms |

Each block of the std::nn kernels had one SIMD group of 8 rows: for the attention output, only 256
blocks of 32 threads.

**Change.** `src/products.vx` has copies of `nn_qkv_q4`, `nn_matvec_q4_heads_add`,
`nn_moe_swiglu_q4` and `nn_moe_down_q4_add` with 2 SIMD groups of 4 rows in each block, as in
llama.cpp's Q4_0 kernel. Each row has the same sum, so the results are the same bits.

**Result.** The gate and up products of the experts alone, 48 layers: 6.95 ms in place of 7.3 ms
(2 x 4 rows was about the same, and 8 x 1 row 8.5 ms). The down products gain less: 4.02 ms in
place of 4.1 ms. A generated token, 256 tokens, two runs of each in turn:

| Prompt | std::nn | 2 x 4 rows |
| --- | --- | --- |
| Write a module and its tests | 22.00, 21.80 ms | 21.34, 21.30 ms |
| Edit a 60-line file | 23.85, 23.90 ms | 23.55, 23.48 ms |

The text is the same, and `--lines` gives the same logits.

### 14. Drafts from the context (#57)

**Problem.** #18 found that a check of several tokens costs too much on the M4 for a draft model.
But drafts from the context cost nothing: in an edit, most of the output is text of the prompt.
llama.cpp has the same idea in `llama-lookup`.

**Change.** `--draft K` finds the longest earlier match of the last tokens in the context, and
takes the tokens after it as a draft. A round runs the kernels of n positions at once (the
`_rows` kernels of std::nn, vx-lang/Vx#3673, and `matvec_q6_rows` for the 6-bit `lm_head`), and
keeps each draft token that equals the choice of the model. The round kernels give the same bits
as the kernels of one token (`tests/round.vx`), so the text is the same as without drafts. A round
of n positions costs about 24 + 13 (n - 1) ms at a short context, so a draft must be long and
likely to pay: no draft for a match under 6 tokens, 1 token for a match of 6 or 7, 4 for 8 to 11,
and 7 from 12. After two rounds in a row that keep nothing, there are no drafts until a longer
match comes up. Between rounds, plain steps run 1, 2, 4 and then up to 8 tokens before each read,
as in entry 12.

**Result.** Edits are 1.83 times as fast (24.06 to 13.13 ms a token), and the 12 code prompts of
"The state now" are 1.19 times as fast (23.27 to 19.58 ms), 1.14 times as fast as `llama-lookup`.
Before the plain runs between rounds, drafts on prompts that write new code cost 0.5 ms a token
more than plain steps; with them, nothing (write prompts, two runs of each in turn):

| Prompts | Plain | Drafts, one read for each plain step | Drafts, plain runs |
| --- | --- | --- | --- |
| Write new code | 21.81 ms | 22.36 ms | 21.82 ms |
| Edit a file | 24.08 ms | 13.12 ms | 13.08 ms |

The text is the same as without drafts in all 24 pairs of these runs.

### 15. Expert launches in an order that overlaps (#34)

**Problem.** A GPU trace of a generated token (Metal System Trace with the GPU counters, 512
positions) shows the GPU idle for about 3.4 ms of a 22.8 ms token, mostly between a kernel and the
kernel that reads its output. The concurrent encoder of Vx puts a barrier only where a launch
reads what an earlier launch wrote, so launches that do not depend on each other can run at the
same time. But in a layer of experts, each launch read the output of the one just before it.

**Change.** `add_experts_shared` now starts the SwiGLU of the shared expert first, which needs only
`xn`. So it runs with the router; the gate of the shared expert runs with top-k; and the down
product of the shared expert runs with the SwiGLU of the chosen experts. The shared expert is now
added to `x` before the chosen experts. The rounds of `--draft` use the same order, so they still
give the same bits as one position.

**Result.** 22.36 to 22.22 ms a token at 512 positions, faster in 4 of 4 interleaved runs.

### 16. Drafts from the MTP layer of Qwen3.6 (#109)

**Problem.** Drafts from the context (entry 14) help only when the output repeats text of the
prompt. Qwen3.6-35B-A3B has one more layer for drafts, the MTP layer (multi-token prediction). It
takes h of the model at position p (after `final_norm`) and the token at position p + 1, and gives
logits for position p + 2. llama.cpp uses this layer for its drafts.

**Change.**

- `convert.py` writes the MTP layer at the end of the weight file. The other tensors keep their
  order and bytes. The config integer `mtp_layers` takes slot 25 of the version 2 header, so an
  older file reads 0 and loads as before.
- `src/mtp35.vx` runs the MTP layer: a layer with full attention and its own KV cache. The prompt
  fills the rows of the MTP layer one chunk at a time, after each chunk of the model.
- `--mtp K` drafts with this layer, one MTP row for each draft token, and checks the drafts with
  the rounds of entry 14. So the text is the same as without drafts.
- A draft scores only the tokens with ids below 65,536 (`--mtp-vocab N`), a quarter of the rows of
  `lm_head`.
- A round drafts K tokens while a running mean of the kept drafts is at least 0.85, and 1 token
  otherwise.
- After a round, the rows of the kept tokens run again with the true h of the model. These rows,
  and the row of the first draft of the next round, have all their inputs after the round, so
  `mtp35_rows` runs them in one pass with the kernels of the round.
- With `--mtp K,L`, a round drafts L tokens while the running mean of the kept drafts is at least
  0.95, and K tokens while it is at least 0.85. Long drafts pay in edits, where almost all drafts
  are kept.

**Result.** The numbers come after a run that follows the benchmarking policy.

### 17. The attention and the lm_head of a round, for all positions at once (#110)

**Problem.** With the MTP drafts of entry 16, a round of Qwen3.6 has 2 to 4 positions. Two parts
of a round did more work for each position than they must:

- Each full attention layer ran 5 kernels for each position of the round.
- `matvec_q6_rows` does 2 vectors in each block, so the 6-bit `lm_head` of 3 or 4 positions read
  each weight twice.

**Change.**

- The full attention layers of a round run 3 kernels for all n positions:
  `pf_gated_norm_rope_and_store` (the kernel of the prompt chunks) stores the k and v rows of all
  positions, and `gated_attention_parts_rows` and `gated_attention_merge_rows` do the attention of
  each position over the rows up to its own position. Each position computes its own number of
  parts, as `gated_attention` does, so a round can cross position 2048.
- `matvec_q6_rows4` does 4 vectors in each block, and runs for rounds of 3 or more positions. To
  keep 16 values of x for each vector, and not 32, a lane keeps only x scaled for the low bits, and
  multiplies the high bits of weight t by 16 * 4^-t over that scale. Both factors are powers of 2,
  so each product and each step of each sum is that of `matvec_q6`.

Each position keeps the bits of a plain step (`tests/round35.vx`, `tests/round.vx`).

**Result.** The numbers come after a run that follows the benchmarking policy.

### 18. The lm_head of a round: only the rows that can be the greedy token (#111)

**Problem.** A round needs only the greedy token of each position and its logit. But it computed
all 248,320 logits of each position exactly, and the arithmetic of the 6-bit weights grows with
the number of positions. A profile by subtraction of a round cannot show the `lm_head`, because
without it nothing waits for the round. A separate bench showed the arithmetic, not the bytes, as
the limit of `matvec_q6_rows4` at 4 positions and more.

**Change.** For 2 or more positions, `run_round35` runs three kernels (`src/q6.vx`):

- `lm_rough_q6` computes rough logits a for up to 8 positions at once, with f16 weights and inputs
  on the simdgroup matrices.
- `lm_pick_threshold` finds, for each position, M = the largest a - B over the rows. B is a bound
  of |a - e|, where e is the exact logit. It comes from the norms of x and of the rows of
  `lm_head`, and `src/q6.vx` gives each of its terms. `lm_q6_norms` computes the norms of the rows
  in the first round.
- `matvec_q6_rows4_pick` computes the exact logit of each row that does not have a + B < M, bit for
  bit as `matvec_q6` does. The other rows get the lowest f32, and cannot be the greedy token. A
  block with no such row reads no weights.

The true error is about 3% of B. In the rounds of a bench on the real model, at most 3 of the
248,320 rows of a position needed the exact logit. The greedy tokens and their logits are those
of the plain steps (`tests/round35.vx`). Only the other logits of a round change, and the app does
not read them.

**Result.** The numbers come after a run that follows the benchmarking policy.

## What did not help

- **Other tile shapes for the Q4_0 products of a chunk, in Vx v0.0.4** (#19). Tiles of 32 tokens,
  64 columns, or 1 or 2 subgroups were all slower (14.7 to 56.4 ms against 14.5 ms). One `@` of a
  32 x 32 tile of inputs took 4 times as long as two of 16 x 32, probably because 16
  `simdgroup_float8x8` sums leave the registers.
- **Chunks of 1024 tokens.** 10.2 ms a prompt token, against 7.6 ms for 512.
- **Speculative decoding with a draft model** (#18). The check of several tokens in one pass costs
  too much on this GPU for a draft model. Drafts from the context do help: see entry 14.
  - A draft model (Qwen2.5-Coder-0.5B) takes about 11 ms a token, half the time of the 30B model,
    because the 30B model reads only 1.9 GB a token. A draft must be much faster than the model
    to help.
  - The tokens of a check use different experts: 9 tokens in a row use about 35 experts of a
    layer, not 8.
  - The kernels of the prompt took 106 ms for a check of 1 token, and 207 ms for 9.
  - The M4 GPU has no separate matrix units: `simdgroup_multiply_accumulate` runs on the same ALUs
    as all other work. So each token of a check adds about one multiply-add for each weight. A
    Q4_0 product for 8 tokens took 0.24 ms in Vx v0.0.4 and 0.105 ms in Vx main with tiles of 8
    tokens, against 0.046 ms for one token.
  - With those kernels, an edit would be about 1.3 times as fast, and new code slower. With the
    round kernels of Vx main and drafts only when they pay, edits are 1.84 times as fast (entry
    14). A draft model may still pay on a Mac with matrix units, or for a model where a token
    costs more.
- **The norm before the experts merged into the router and the SwiGLUs** (#34). Each block found
  the scale of the norm from all of `x`, then read `x` times the scale. This removed 40 launches
  from a token, but only 6 of its 417 barriers: the router and the SwiGLUs must still wait for the
  product that writes `x`. With each block reading all of `x` again, the token was about 0.55 ms
  slower (4 of 4 interleaved runs). So the idle gaps between small kernels are waits for data,
  not the cost of a launch. Only work that does not depend on the kernel before it fills a gap.
- **Weights decoded once into registers for several tokens.** In Vx v0.0.4 these kernels were
  slower than the kernels that decode each weight with masks for each token. Their 64 to 128
  floats a lane likely leave the registers.
- **The router of a round with 4 positions in each block.** A bf16 kernel for the router of 4
  positions, as `matvec_q6_rows4` does for `lm_head`, was no faster than the router of each
  position in turn. The router of a round is not limited by the reads of its weights.

## Open work

- **The prompt**: 135 tokens a second at 2,000 tokens, against 361 for llama.cpp. #30 is a first
  step (151).
- **The start of a session** (#27): the load of the weights takes about 10.5 s since entry 11. The
  compile of the program at -O0 is the next part to measure.
- **Small kernels** (#34): the 336 kernels of a token other than the products (7 in each layer)
  take about 2.5 ms for about 0.7 ms of bytes. Each one reads what the kernel before it wrote.
  Fewer launches alone do not help (see "What did not help"); work that runs at the same time
  does (entry 15). A next step: the product that writes `x` also writes the sum of squares of each
  block, so the kernels after it need not read all of `x` for the norm.
- **Fewer bytes for a token** (#33): lm_head in Q4_0 (#9) would save about 1.3 ms, after a check of
  the perplexity.
- **Attention at a long context** (#38): about 4 ms of a token at 8,000 positions, above the
  floor. An earlier measurement of 63.9 ms there was slowed by swap.
- **A short turn at a long context** (#26): a turn of 20 tokens still runs almost every expert of
  every layer, so it takes about 2 s.
