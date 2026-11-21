# Qwen3.6-35B-A3B on an M4: qwen-vx against llama.cpp

Run on 2026-11-20 and 2026-11-21, as `docs/benchmarking_policy.md` of the Vx repository asks. The
app is qwen-vx at the code of PR #98 (prefill kernels), compiled by vxc 0.0.5 with `-O2`. The
baseline is llama.cpp at commit b1b770d (2026-11-13), Release build, Metal.

## Claims

All numbers are for batch 1, greedy decoding (temperature 0), on an M4 MacBook Air with 32 GB.
The model is Qwen3.6-35B-A3B. Both files have the same Q4_0 matrices and a 6-bit output matrix
(6.5 against 6.5625 bits a weight). The token embedding is bf16 in the app and Q4_0 in llama.cpp,
and a token reads one row of it. The "held-out prompts" are 12 chat prompts of 35 to 1,668 tokens.
`make_prompts.py` wrote them for this run (commit 5f57020, 2026-11-20). The draft rule of the app
is older (commit 17ebe25, 2026-11-19), and table T chose the draft lengths on other prompts.

- **Decode:** 44.5 against 38.3 generated tokens a second (1.16×), Qwen3.6-35B-A3B in Q4_0 with a
  6-bit output matrix, on the 12 held-out prompts, 512 generated tokens, on an M4 with 32 GB, batch
  1, against llama.cpp b1b770d (`llama-completion`). qwen-vx is faster on 12 of 12 prompts (1.16×
  to 1.17×).
- **Decode after a long prompt:** 1.12× to 1.21× from 64 to 32,768 prompt tokens (table A), for
  example 41.1 against 35.8 tokens a second after 8,192 tokens.
- **Prompt:** 492 against 404 prompt tokens a second (1.22×) for a prompt of 2,048 tokens, 453
  against 420 (1.08×) at 512, and 400 against 358 (1.12×) at 8,192 (table P, the app's runs after
  its first run in a process).
- **Speculative decoding with drafts from the context:** 51.7 against 45.9 generated tokens a
  second (1.12×), `qwen-vx --draft 7` against `llama-speculative-simple --spec-type ngram-mod`, on
  the 12 held-out prompts. qwen-vx is faster on 10 of 12 prompts (0.61× to 1.49×). Against
  `ngram-cache`: 51.7 against 41.3 (1.25×), faster on 11 of 12.

Where qwen-vx is slower:

- **Drafts from the MTP heads of the model.** llama.cpp with `--spec-type draft-mtp` (3 tokens)
  makes 58.6 tokens a second, against 51.7 for `qwen-vx --draft 7` (qwen-vx is 0.88×). qwen-vx is
  faster on 3 of 12 prompts, one of them (edit1) by 0.1%. The app cannot use the MTP heads yet. Its
  drafts gain on edit prompts (13.19 ms against 15.64 ms a token), and lose on the other kinds.
- **A short first prompt.** A 64-token prompt takes 351 ms against 251 ms (table A). This is the
  first prompt of the app's process, which includes the first launch of each kernel.
  `llama-completion` runs a warm-up first. The run has no warm 64-token prompt of the app.
- Against `ngram-mod`, `qwen-vx --draft` is slower on 2 of 12 held-out prompts (down to 0.61×).

## Setup

What both sides share:

| | |
| --- | --- |
| Machine | MacBook Air Mac16,12, Apple M4 (10-core CPU with 4 performance and 6 efficiency cores, 10-core GPU, Metal 4), 32 GB, no fan, on AC power |
| OS and GPU driver | macOS 27.0.1 (26A434). The GPU driver comes with the OS |
| Model | Qwen3.6-35B-A3B: 40 layers (30 Gated DeltaNet, 10 attention), 256 experts, 8 for each token (`weights.txt`) |
| Q4_0 matrices (4-bit codes, an f16 scale for each 32 weights) | attention q/k/v/gate/out, DeltaNet in and out, experts gate/up/down, shared expert gate/up/down. The bits are the same in both files |
| f32 tensors | norms, DeltaNet conv, A and dt bias |
| KV cache | f16 |
| DeltaNet state | f32 |
| Decoding | greedy (temperature 0), batch 1, end-of-text ignored, thinking off (`<think>\n\n</think>\n\n` in the prompt) |
| Prompts | the same chat text. The llama.cpp files have one more final newline, which `-f` drops. `make_prompts.py` counted the tokens with the `tokenizer.json` of the model (`prompts/tokens.csv`). llama.cpp read the same number of tokens in every run |
| Generated tokens | 256 (table A), 512 (tables T and C) |
| Order | 3 rounds, the order of the programs reversed in round 2 (table T: one run, the order reversed for each second prompt) |
| Idle check | before each run, `run.sh` waits until no process uses more than 20% of a CPU and the GPU shows at most 5%, then waits 30 s. It does not count the processes named claude, Terminal, WindowServer and kernel_task |

What differs:

| | qwen-vx | llama.cpp |
| --- | --- | --- |
| Program | `vxc -O2 src/main.vx`, vxc 0.0.5 (5b92da4f), the release | `llama-completion`, `llama-speculative-simple`, `llama-bench`, b1b770d, Release (`llama.cpp-build.txt`), with a 3-line patch to one clock of `llama-speculative-simple` (below) |
| Weight file | `Qwen3.6-35B-A3B-q4-q6.vxw`, 20,392,312,832 bytes, made by `scripts/convert.py` | `Qwen3.6-35B-A3B-Q4_0.gguf`, 19,715,053,888 bytes, made by `llama-quantize` from the BF16 GGUF of ggml-org |
| Token embedding | bf16 | Q4_0 |
| Output matrix | 6-bit codes, an f16 scale for each 32 weights (6.5 bits a weight) | Q6_K (6.5625 bits a weight) |
| Router and shared-expert gate | bf16 | f32 |
| Weight bytes read for each token | 1.819 GB | 1.865 GB |
| Compile | the app compiles before each run, out of the clock. A run of the app at 64 prompt tokens takes 105 to 108 s in `log.txt`, against 8 to 17 s for `llama-completion`. Most of this is the compile | none |
| Settings | `--host default` (the Mac that compiles the app), `--machine fleet/m4-uma.vx`. No "auto" setting | flash attention: on, which `run.sh` chose in place of the default "auto" (26.37 ms against 27.67 ms a token with it off, medians at 2,048 tokens). CPU threads: 4 (the default). ngram-mod: `n_match` 24, `n_min` 48, `n_max` 64 (the defaults). ngram-cache: 8 tokens, no cache files |
| Batch sizes of the prompt | chunks of up to 2,048 tokens (`CHUNK` of `src/prefill.vx`) | `-b 2048 -ub 512` (the defaults). `-b 4096` for `llama-speculative-simple`, except on the first 8 tuning prompts (`-b 2048`) |

## The same work: bytes for each generated token

A generated token reads one row of the embedding, the matrices of its 8 experts, all other weights,
the KV cache of the 10 attention layers (20,480 bytes for each earlier token), and the DeltaNet
and conv state of the 30 DeltaNet layers. The state is 30 × (32 × 128 × 128 + 3 × 8,192) f32
values, read and written: 131.7 MB. The floor is the time to read these bytes at 120 GB/s, the
memory speed that Apple gives for the M4. The table counts the KV cache of the prompt only. Over 256
generated tokens, the cache grows by 2.6 MB on average, which changes no number of the table by
more than 0.003 GB.

| Prompt tokens | qwen-vx GB | floor ms | measured ms | GB/s | llama.cpp GB | floor ms | measured ms | GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.952 | 16.3 | 22.17 | 88 | 1.998 | 16.7 | 25.81 | 77 |
| 512 | 1.961 | 16.3 | 22.33 | 88 | 2.007 | 16.7 | 25.83 | 78 |
| 2,048 | 1.993 | 16.6 | 22.82 | 87 | 2.039 | 17.0 | 26.40 | 77 |
| 8,192 | 2.118 | 17.7 | 24.32 | 87 | 2.164 | 18.0 | 27.94 | 77 |
| 16,384 | 2.286 | 19.1 | 26.55 | 86 | 2.332 | 19.4 | 32.05 | 73 |
| 32,768 | 2.622 | 21.8 | 43.55 | 60 | 2.668 | 22.2 | 48.97 | 54 |

The weight bytes come from `bytes_per_token.py` (output in `weights.txt`). The llama.cpp file reads
46 MB more for each token, 42 MB of it for the f32 router (2,048 × 256 × 40 × 2 more bytes). At 120
GB/s, this is 0.4 ms, a tenth of the gap of 3.6 ms at 64 tokens.

## The same quality

The two files store three kinds of tensor in other formats. Two of these kinds have other values.
The outputs of the checks are in `weights.txt`.

- **The same bits.** `compare_bits.py` checked the 320 Q4_0 matrices of the 40 layers: experts
  gate, up and down, shared expert gate, up and down, and the matrices of attention or DeltaNet.
  Each holds the codes and scales of the GGUF file. llama.cpp splits the joined input matrices
  (attention q, k and v. DeltaNet qkv, gate, beta and alpha). Their rows are the same rows in
  another order. The DeltaNet output matrix has the same value heads in another order. A run with
  one code or scale changed in 8 matrices reported exactly these 8.
- **The router and the shared-expert gate** have the same values: the GGUF stores the bf16 values
  as f32.
- **The token embedding and the output matrix differ** (bf16 against Q4_0, 6-bit codes against
  Q6_K). One program, `scripts/reference.py` (NumPy, f32), measured both sets of weights. The
  llama.cpp set is the app's file with these two tensors taken from the GGUF and widened to f32.
  The texts are the first 512 tokens of three files: `docs/benchmarking_policy.md` of Vx at
  643f63b5b, and two files of qwen-vx at 3d56b14 (`perplexity.py`):

| Text | Perplexity, qwen-vx weights | Perplexity, llama.cpp weights | KL divergence (llama.cpp from qwen-vx) | Same most likely token |
| --- | ---: | ---: | ---: | ---: |
| `docs/benchmarking_policy.md` (English) | 11.935 | 12.131 | 0.0123 | 94.7% |
| `src/speculate.vx` (Vx) | 7.357 | 7.356 | 0.0051 | 96.1% |
| `scripts/qwen_weights.py` (Python) | 7.124 | 7.157 | 0.0070 | 96.5% |

The app's weights give a lower perplexity on two texts, and a perplexity higher by 0.001 on
`src/speculate.vx`. The greedy texts of the two programs are the same on 3 of the 12 held-out
prompts (edit1, edit2, write1). On the others they differ after 0 to 1,807 characters (`summary.md`).
The weights pick another most likely token at 3.5% to 5.3% of the positions above, and the two
programs also differ in their kernels.

## Results

The full tables are in `summary.md`, and each run is a row of `runs.csv`.

### Table A: generated tokens after a prompt

256 generated tokens after a prompt of 64 to 32,768 tokens, 3 runs. ms for each generated token:
the median, and the fastest and slowest run. The prompt time is that of the first prompt of a
process.

| Prompt tokens | qwen-vx ms | llama.cpp ms | qwen-vx tok/s | llama.cpp tok/s | llama.cpp ms / qwen-vx ms | qwen-vx prompt ms | llama.cpp prompt ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 22.17 (22.14-22.17) | 25.81 (25.63-26.19) | 45.1 | 38.7 | 1.164 | 351 (338-366) | 251 (251-252) |
| 512 | 22.33 (22.29-22.36) | 25.83 (25.81-25.85) | 44.8 | 38.7 | 1.157 | 1180 (1174-1203) | 1217 (1216-1217) |
| 2,048 | 22.82 (22.79-22.84) | 26.40 (26.36-26.53) | 43.8 | 37.9 | 1.157 | 4160 (4157-4168) | 5050 (5044-5056) |
| 8,192 | 24.32 (24.30-24.33) | 27.94 (27.86-27.98) | 41.1 | 35.8 | 1.149 | 18850 (18830-18928) | 22577 (22550-22622) |
| 16,384 | 26.55 (26.54-26.65) | 32.05 (32.02-32.16) | 37.7 | 31.2 | 1.207 | 45572 (45432-45955) | 53550 (53046-53562) |
| 32,768 | 43.55 (43.39-47.50) | 48.97 (48.38-55.01) | 23.0 | 20.4 | 1.124 | 129143 (128628-134961) | 138361 (138249-149488) |

`llama-bench` with the same file, the mean of 5 repetitions as it prints them: tg128 39.09, 38.25,
35.77, 27.83 and 19.68 tokens a second at depth 0, 2,048, 8,192, 16,384 and 32,768.

### Table P: the prompt

Prompt tokens a second: the median, and the slowest and fastest run. qwen-vx ran the prompt 3
times in each of 3 processes. "First run" has the 3 first runs, which include the first launch of
each kernel. "Later runs" has the other 6. `llama-completion` ran the prompt once in each of 3
processes, after its warm-up. `llama-bench` gives the mean and the standard deviation of 5
repetitions.

| Prompt tokens | qwen-vx first run | qwen-vx later runs | llama-completion | llama-bench |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 433 (433-436) | 453 (450-454) | 420 (419-421) | 426.29 ± 3.15 |
| 2,048 | 492 (491-493) | 492 (490-494) | 404 (396-406) | 404.80 ± 3.25 |
| 8,192 | 433 (429-433) | 400 (394-411) | 358 (355-361) | 343.12 ± 5.44 |

At 8,192 tokens, the later runs of the app are slower than its first run. This run does not show
why. The slowest `llama-completion` run at 2,048 tokens (396) started while `clang++` used a whole
CPU (`log.txt`, P.llama.ctx2048.r3).

### Table C: the 12 held-out prompts

512 generated tokens, 3 rounds. ms for each generated token: the geometric mean over the prompts of
each kind, and the fastest and slowest run.

| Kind | Prompt tokens | qwen-vx | qwen-vx --draft 7 | llama-completion | ngram-cache | ngram-mod | draft-mtp (3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edit | 1286, 1605, 1668 | 22.60 (22.49-22.71) | 13.19 (11.45-15.91) | 26.30 (26.11-26.68) | 14.13 (11.98-18.33) | 11.50 (6.88-20.41) | 15.64 (15.15-16.15) |
| complete | 1044, 1086, 1288 | 22.49 (22.41-22.60) | 21.89 (20.27-22.81) | 26.22 (26.10-26.50) | 27.31 (20.96-31.67) | 27.63 (26.33-30.30) | 17.30 (16.22-18.15) |
| write | 35, 44, 52 | 22.26 (22.21-22.35) | 21.79 (20.74-22.44) | 25.79 (25.66-25.94) | 28.91 (26.37-32.53) | 26.35 (25.98-27.36) | 15.92 (14.90-16.48) |
| explain | 1257, 1330, 1503 | 22.56 (22.51-22.69) | 22.31 (21.48-22.86) | 26.22 (26.15-26.29) | 30.77 (27.97-32.53) | 26.88 (26.44-27.73) | 19.74 (18.14-20.77) |
| all 12 | | 22.48 (22.21-22.71) | 19.36 (11.45-22.86) | 26.13 (25.66-26.68) | 24.20 (11.98-32.53) | 21.78 (6.88-30.30) | 17.08 (14.90-20.77) |

The ratios over the 12 prompts, from the mean of the 3 runs of each prompt. A ratio above 1 means
that the first program is faster.

| Programs | Geometric mean | Faster on | Smallest | Largest |
| --- | ---: | ---: | ---: | ---: |
| qwen-vx against llama-completion | 1.162 | 12 of 12 | 1.16 | 1.17 |
| qwen-vx --draft against qwen-vx | 1.161 | 7 of 12 | 0.99 | 1.96 |
| qwen-vx --draft against llama-completion | 1.350 | 12 of 12 | 1.15 | 2.29 |
| qwen-vx --draft against ngram-cache | 1.250 | 11 of 12 | 0.98 | 1.45 |
| qwen-vx --draft against ngram-mod | 1.125 | 10 of 12 | 0.61 | 1.49 |
| qwen-vx --draft against draft-mtp | 0.882 | 3 of 12 | 0.73 | 1.35 |
| ngram-cache against llama-completion | 1.080 | 4 of 12 | 0.79 | 2.12 |
| ngram-mod against llama-completion | 1.200 | 3 of 12 | 0.87 | 3.77 |
| draft-mtp against llama-completion | 1.530 | 12 of 12 | 1.28 | 1.70 |

30 runs of table C, all in rounds 2 and 3, started while the GPU showed 6% to 15%. The busiest
other process at these starts was WindowServer or launchd. Each of these runs is within -0.8% to
+2.3% of the mean of the other 2 rounds of its case. The summary keeps them.

## Speculative decoding

| | qwen-vx --draft | ngram-cache | ngram-mod | draft-mtp |
| --- | --- | --- | --- | --- |
| Source of the drafts | the context: the longest earlier match of its end | the context: short n-grams with counts | the context: hashed n-grams | extra heads: the MTP heads of the model (1.06 GB GGUF) |
| Largest draft | 7 tokens (`--draft 7`, chosen in table T) | 8 tokens (fixed in llama.cpp) | 64 tokens (`n_max`, the default) | 3 tokens (`--spec-draft-n-max 3`, chosen in table T) |
| Accepted / drafted, table C round 1 | 1520 / 1785 (85%) | 3154 / 3154 as printed (see below) | 1274 / 1274 as printed (see below) | 4394 / 5360 (82%) |
| Accepted tokens for each check | 4.79 (edit 6.49, complete 2.95, write 2.53, explain 2.51) | not printed | not printed | not printed |
| Same text as without drafts | 72 of 72 runs in T, 36 of 36 in C | 8 of 12 in T, 27 of 36 in C | 8 of 12 in T, 24 of 36 in C | 42 of 60 in T, 21 of 36 in C |

Temperature 0 everywhere.

**qwen-vx `--draft`** gives each position of a check the bits of a plain step, so its text is the
text of plain qwen-vx in every run. It drafts nothing below a 6-token match, 1 token for a match of
6 or 7, 4 for 8 to 11, and 7 from 12. After 2 checks that accept nothing, it drafts again only after a
longer match. `--draft N` caps this rule, so `--draft 9`, `11` and `15` make the drafts of
`--draft 7`: table T has the same draft counts for them on every prompt.

**llama.cpp with drafts writes other text in some runs.** 36 of the 108 runs with drafts in table
C, and 26 of the 84 in table T, leave the text of `llama-completion`. None has an error line, and
the text stays coherent (for example ", Dict" against "\n\nSOURCE"). On 3 tuning prompts,
ngram-mod, ngram-cache and every MTP length leave the text at the same character. This points to a
near tie that the shape of the batches changes. The summary keeps these runs. `summary.md` lists
each one, with the number of equal characters at the start of the two texts.

**The acceptance of the n-gram drafts of llama.cpp is not comparable.** This model has a recurrent
state. For ngram-cache and ngram-mod, `llama-speculative-simple` cannot drop rejected tokens from
its memory. On a partial accept, it loads a checkpoint and checks the accepted tokens again as a
new draft. Its counts leave out the first check, so they show 100%. The time includes both checks.
draft-mtp keeps a copy of the recurrent state for each draft position, so it uses no checkpoints
(none of its 96 logs has the line "speculative decoding will use checkpoints", and all 96 logs of
ngram-cache and ngram-mod have it). Its counts are comparable.

**The draft programs of llama.cpp stop after the check that passes `-n 512`.** They decoded 513 to
577 tokens. Their speed is the decode time divided by the decoded tokens.

### Table T: the draft lengths

The 12 tuning prompts (`prompts/twelve/`) are other prompts of the same 4 kinds. One run each, 512
generated tokens. ms for each generated token, the geometric mean over the 12 prompts:

| Program | Draft lengths and ms |
| --- | --- |
| qwen-vx --draft | 3: 19.39, 5: 19.04, **7: 18.90**, 9: 18.92, 11: 18.91, 15: 18.96 |
| draft-mtp | 1: 21.62, 2: 19.11, **3: 18.15**, 4: 18.21, 6: 19.22 |
| no drafts, ngram-cache, ngram-mod | qwen-vx 22.59, llama-completion 26.16, ngram-cache 24.38, ngram-mod 19.76 |

For draft-mtp, 3 and 4 are 0.3% apart. The 4 settings of the app that make the same drafts (7, 9,
11, 15) spread over 0.3% too. So table T does not separate 3 from 4. The choice is 3 also on the 7
prompts where every MTP length kept the text of llama-completion (18.20 ms, against 18.40 for 4).

## Where the gain comes from

This run measures the app as a whole. It does not measure each change on its own. Two side
measurements:

- Flash attention gives llama.cpp 1.30 ms a token at 2,048 tokens (26.37 against 27.67 ms, the
  medians of 3 runs each, logs `fa-*`).
- The compile before each run of the app: 3 runs that compiled first gave 22.14, 22.19 and 22.31
  ms a token, and 3 runs of one kept executable gave 22.09, 22.15 and 22.07 ms (512 prompt
  tokens, 256 steps, `extra.tar.xz`). The means differ by 0.5%, and the ranges touch.

## Limits

- One machine, a fanless MacBook Air. A fanless Mac can slow its clock under a long load. The
  programs run interleaved, in a reversed order in round 2, so that a slow period does not fall on
  one program only.
- Batch 1 and greedy decoding only. No serving numbers (section 6 of the policy).
- Decode to 32,768 prompt tokens. Prompt speed to 8,192 tokens in table P. Table A gives the time of
  the first prompt of a process to 32,768 tokens.
- Table T has one run of each case.
- The run does not split the gain of the app into its changes.
- `llama-lookup` is not in the benchmark. In the short test of `run.sh` (`extra.tar.xz`, `lookup/`),
  its text differs from that of `llama-completion`, and its log has errors from `llama_decode`:
  it cannot remove rejected tokens from a recurrent state. `llama-speculative-simple --spec-type
  ngram-cache` uses the same n-gram cache with checkpoints.
- `llama-speculative-simple` has a 3-line patch (`llama-timing.patch`). Without it, the program
  stops the clock of the prompt before the GPU ends the prompt, and counts the rest of the prompt
  in the time of the generated tokens (on edit1, 0.77 s of 1,799 prompt tokens, `extra.tar.xz`,
  `speculative-clock/`). The patch waits for the GPU before it stops the clock. The other programs
  have no patch.
- The two weight files differ in the token embedding and the output matrix. The perplexity check
  above compares the two files with one program.
- ngram-cache and ngram-mod keep the settings of llama.cpp. ngram-cache always drafts 8 tokens.
- llama.cpp does not print its number of checks, so the accepted tokens for each check are only
  for the app.
- The prompts are code tasks in 4 kinds. Other text can give other draft gains.
- The run started 3 times (below). The first 8 tuning prompts of `llama-speculative-simple` ran with
  `-b 2048`, the others with `-b 4096`. The batch size sets only how the prompt goes to the model.
- The restart wrote over the logs of the first 4 runs of edit3 in table T (below).

## How the run went

1. 16:12 on 2026-11-20, all stages. At 22:11, `llama-speculative-simple` refused the tuning prompt
   edit3 (2,251 tokens): "the prompt exceeds the batch size (2251 tokens, batch 2048)". I stopped the
   run. The stages fa, bench, A and P and the first 8 tuning prompts come from this start.
2. 22:13: `STAGES=T PROMPTS="edit3 complete3 write3 explain3" FA=on`, with `-b 4096` for
   `llama-speculative-simple`. This change was not committed yet, so `env.txt` names ef7c58a. The
   `run.sh` that ran differs from ef7c58a only in this line, which commit 69adc71 committed. All
   runs of edit3 ran again, so the programs of that prompt ran together. `parse.py --choose` then
   found no MTP length with the text of llama-completion on all 12 prompts, and stopped.
3. 23:59: `STAGES=C FA=on DRAFT=7 MTP_N=3`, the draft lengths of table T with all runs counted. It
   ended at 05:31 on 2026-11-21. After this start, `parse.py` changed to keep the runs of llama.cpp
   with other text (commit 806bfe0). Stage C of `run.sh` does not call `parse.py`.

`log.txt` keeps the 4 lines of the first edit3 runs: vx and completion (exit code 0), ngram and cache
(exit code 1). The restart wrote over their logs, so `runs.csv` has no rows for them, and the error
message above comes from the log before the restart. `parse.py` uses the last line of each run.

## Commands

llama.cpp:

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b1b770d
    git apply /path/to/qwen-vx/docs/benchmarks/2026-11-20-qwen36-m4/llama-timing.patch
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_MTMD=OFF \
      -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build build -j 8

The model files:

    hf download Qwen/Qwen3.6-35B-A3B --local-dir ~/models/Qwen3.6-35B-A3B
    python3 scripts/convert.py --q4 --q6-lm-head ~/models/Qwen3.6-35B-A3B Qwen3.6-35B-A3B-q4-q6.vxw
    python3 scripts/convert_tokenizer.py ~/models/Qwen3.6-35B-A3B Qwen3.6-35B-A3B.vxt
    hf download ggml-org/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-BF16.gguf mtp-Qwen3.6-35B-A3B-Q4_0.gguf
    llama-quantize Qwen3.6-35B-A3B-BF16.gguf Qwen3.6-35B-A3B-Q4_0.gguf Q4_0

The run, under `caffeinate -dims`, on an idle Mac, from the root of this repository:

    export OUT=docs/benchmarks/2026-11-20-qwen36-m4 VXC=~/.vx/current/bin/vxc FLEET=~/.vx/current/fleet \
      VXW=Qwen3.6-35B-A3B-q4-q6.vxw VXT=Qwen3.6-35B-A3B.vxt GGUF=Qwen3.6-35B-A3B-Q4_0.gguf \
      MTP=mtp-Qwen3.6-35B-A3B-Q4_0.gguf LB=llama.cpp/build/bin
    caffeinate -dims $OUT/run.sh
    python3 $OUT/parse.py $OUT

This run used the first command, then the two restarts above:

    STAGES=T PROMPTS="edit3 complete3 write3 explain3" FA=on caffeinate -dims $OUT/run.sh
    STAGES=C FA=on DRAFT=7 MTP_N=3 caffeinate -dims $OUT/run.sh

`run.sh` prints each command line in the log of its run. `make_prompts.py` wrote `prompts/`.

The checks of the weight files, on the CPU, from the root of this repository, with a checkout of
Vx at 643f63b5b in `$VX`:

    python3 $OUT/bytes_per_token.py $VXW $GGUF scripts llama.cpp/gguf-py
    python3 $OUT/compare_bits.py $VXW $GGUF scripts llama.cpp/gguf-py
    python3 $OUT/perplexity.py $VXW $GGUF scripts llama.cpp/gguf-py ~/models/Qwen3.6-35B-A3B \
      $VX/docs/benchmarking_policy.md src/speculate.vx scripts/qwen_weights.py --max-tokens 512

## Files

- `run.sh`, `parse.py`, `make_prompts.py`, `prompts/`: the script, the parser and the inputs.
- `env.txt`: the machine, OS, versions, files and settings, written at each start of `run.sh`.
- `log.txt`: one line for each run, with its start, end, exit code, the busiest other process and
  the GPU use at the start.
- `logs.tar.xz`: the output of each run (`tar -xJf logs.tar.xz` gives `logs/`, 33 MB).
  `parse.py` gives the same `summary.md` and `runs.csv` from it.
- `runs.csv`, `summary.md`: written by `parse.py`.
- `llama-timing.patch`, `llama.cpp.diff`, `llama.cpp-build.txt`: the patch, the diff of the
  llama.cpp source that ran, and its build settings.
- `bytes_per_token.py`, `compare_bits.py`, `perplexity.py`, `weights.txt`: the checks of the
  weight files, and their output.
- `extra.tar.xz`: the side measurements (`extra/INDEX.txt` lists them): the short test of
  `llama-lookup`, the runs of `llama-speculative-simple` with and without the patch, the compile
  runs of the app, and the test of `-b 4096` on edit3.
