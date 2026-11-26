# Qwen3.6-35B-A3B on an M4: qwen-vx with MTP drafts against llama.cpp

Run on 2026-11-25, as `docs/benchmarking_policy.md` of the Vx repository asks. The app is qwen-vx
at commit 730f1b7 (PR #113 on main, with #109 to #112), compiled by vxc 0.0.5 with `-O2`. The
baseline is llama.cpp at commit b1b770d (2026-11-13), Release build, Metal, the same commit as the
run of 2026-11-20 ([`../2026-11-20-qwen36-m4/`](../2026-11-20-qwen36-m4/README.md)).

This run adds the drafts from the MTP layer of the model to the app (`--mtp`, entries 16 to 19 of
`docs/optimizations.md`). It measures the app with and without drafts against llama.cpp with and
without drafts.

## Claims

All numbers are for batch 1, greedy decoding (temperature 0), on an M4 MacBook Air with 32 GB. The
model is Qwen3.6-35B-A3B. Both files have the same Q4_0 matrices and a 6-bit output matrix (6.5
against 6.5625 bits a weight). The "held-out prompts" are the 12 chat prompts of the 2026-11-20 run
(35 to 1,668 tokens). No setting of this run was tuned on them. Each program ran each prompt 3
times, 512 generated tokens. A speed on the held-out prompts is the geometric mean over the 36 runs.
A ratio is the geometric mean over the 12 prompts of the ratios of the medians of each prompt.

- **Decode with drafts:** 71.2 against 58.5 generated tokens a second (1.22×),
  `qwen-vx --mtp 3,5 --draft 11` against `llama-speculative-simple --spec-type draft-mtp` (3
  tokens), Qwen3.6-35B-A3B in Q4_0 with a 6-bit output matrix, on the 12 held-out prompts, on an M4
  with 32 GB, batch 1, against llama.cpp b1b770d. qwen-vx is faster on 12 of 12 prompts (1.06× to
  1.61×).
- **Decode with MTP drafts only:** 67.8 against 58.5 tokens a second (1.16×), `qwen-vx --mtp 3,5`
  against the same llama.cpp program. Faster on 12 of 12 prompts (1.06× to 1.25×).
- **Decode without drafts:** 44.8 against 38.3 tokens a second (1.17×), `qwen-vx` against
  `llama-completion`. Faster on 12 of 12 prompts (1.16× to 1.18×).
- **Decode after a long prompt:** 1.16× to 1.21× from 64 to 32,768 prompt tokens (table A), for
  example 41.4 against 35.7 tokens a second after 8,192 tokens.
- **Prompt:** 490 against 404 prompt tokens a second (1.21×) for a prompt of 2,048 tokens, 451
  against 419 (1.08×) at 512, and 399 against 356 (1.12×) at 8,192 (table P, the app's runs after
  its first run in a process).
- **Drafts from the context:** 56.3 against 46.0 tokens a second (1.22×), `qwen-vx --draft 11`
  against `llama-speculative-simple --spec-type ngram-mod`. Faster on 11 of 12 prompts (0.77× to
  1.64×).

Each program with drafts of the app gave the text of the app without drafts in 36 of 36 runs.

Where qwen-vx is slower:

- **edit2 against ngram-mod.** ngram-mod takes 6.93 ms a token, against 9.51 ms for
  `qwen-vx --mtp 3,5 --draft 11` and 9.07 ms for `qwen-vx --draft 11`. ngram-mod drafts up to 64
  tokens. The app drafts up to 11. This is the one prompt of 12 where `--mtp 3,5 --draft 11` is
  slower than ngram-mod (0.74×).
- **Drafts from the context only, against draft-mtp:** `qwen-vx --draft 11` is 0.96× of
  llama.cpp's MTP drafts, and faster on 3 of 12 prompts (the 3 edits).
- **A short first prompt.** A 64-token prompt takes 262 ms against 252 ms (table A). This is the
  first prompt of the app's process, which includes the first launch of each kernel.
  `llama-completion` runs a warm-up first.

## Setup

The machine, the OS, the model, the formats, the prompts, the order of the runs and the settings
of llama.cpp are those of the 2026-11-20 run ([Setup](../2026-11-20-qwen36-m4/README.md#setup)).
Its checks of the weight files (the same bits in the 320 Q4_0 matrices, the perplexity of the two
token embeddings and output matrices) hold for this run, because the app's file has the same 745
tensors. What is new:

| | qwen-vx | llama.cpp |
| --- | --- | --- |
| Program | `vxc -O2 src/main.vx` at 730f1b7, vxc 0.0.5 (5b92da4f) | b1b770d with `llama-timing.patch`, built again from the same commit with the same settings (`llama.cpp-build.txt`) |
| Weight file | `Qwen3.6-35B-A3B-q4-q6-mtp.vxw`, 20,868,300,800 bytes: the 745 tensors of the 2026-11-20 file with the same bytes, and the 22 tensors of the MTP layer after them (475,987,968 bytes) | `Qwen3.6-35B-A3B-Q4_0.gguf`, 19,715,053,888 bytes, as on 2026-11-20 |
| MTP layer | in the weight file. `append_mtp.py` added it with the functions of `scripts/convert.py`, which writes the same tensors | `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` of ggml-org, 1,060,038,720 bytes |
| MTP drafts | up to 3 tokens while the running mean of the kept drafts is at least 0.85, up to 5 from 0.95, else 1 (`--mtp 3,5`). A draft scores the 65,536 tokens of the lowest ids | 3 tokens (`--spec-draft-n-max 3`). A draft scores all 248,320 tokens |
| Drafts from the context | `--draft 11`: up to 11 tokens after a match of 16 tokens, 4 after 12 to 15, 1 after 2 to 11. With `--mtp`, a round drafts from the context only after a match of 16 tokens | ngram-mod: `n_match` 24, `n_min` 48, `n_max` 64 (the defaults) |
| Settings | no "auto" setting | flash attention on, which `run.sh` chose again: 26.31 against 27.88 ms a token with it off (medians at 2,048 tokens). CPU threads: 4 (the default) |

The draft settings come from the 12 tuning prompts (`prompts/twelve/`), not from the held-out
prompts: `--mtp 3,5` and the match of 16 from entries 16 and 19 of `docs/optimizations.md`,
`--draft 11` from `draft_length35` of `src/speculate.vx`, and 3 MTP tokens for llama.cpp from table
T of the 2026-11-20 run, with the same build of llama.cpp. This run has no table T.

The idle check of `run.sh` now checks the machine again after the 30 s of cooling, and waits again
when another process started in them.

## Results

The full tables are in `summary.md`, and each run is a row of `runs.csv`. ms for each generated
token; tokens a second are 1,000 divided by the ms.

### Table C: the 12 held-out prompts

The geometric mean over the runs of each kind (3 prompts, 3 runs each), and the fastest and
slowest run:

| Kind | qwen-vx | qwen-vx `--draft 11` | qwen-vx `--mtp 3,5` | qwen-vx `--mtp 3,5 --draft 11` | llama-completion | ngram-mod | draft-mtp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| edit | 22.40 (22.31-22.48) | 11.00 (9.05-13.91) | 12.94 (12.24-13.54) | **10.62** (9.47-12.03) | 26.36 (26.15-26.53) | 11.48 (6.91-20.37) | 15.62 (15.08-16.21) |
| complete | 22.33 (22.23-22.40) | 20.28 (18.36-21.42) | 15.46 (13.97-16.78) | **15.37** (13.78-16.83) | 26.14 (26.03-26.35) | 27.66 (26.29-30.30) | 17.35 (16.42-18.10) |
| write | 22.12 (22.03-22.28) | 20.69 (19.50-21.82) | **14.11** (13.54-15.21) | 14.15 (13.58-15.21) | 25.69 (25.64-25.84) | 26.23 (25.81-26.79) | 16.05 (14.88-17.01) |
| explain | 22.38 (22.29-22.58) | 21.55 (20.40-22.57) | **16.72** (15.71-17.21) | 16.83 (16.02-17.58) | 26.24 (26.10-26.55) | 26.80 (26.35-27.68) | 19.68 (18.11-20.39) |
| all 12 | 22.31 | 17.76 | 14.74 | **14.04** | 26.10 | 21.74 | 17.10 |
| tokens a second | 44.8 | 56.3 | 67.8 | **71.2** | 38.3 | 46.0 | 58.5 |

The median of the 3 runs of each prompt:

| Prompt | qwen-vx | `--draft 11` | `--mtp 3,5` | `--mtp 3,5 --draft 11` | llama-completion | ngram-mod | draft-mtp | draft-mtp / best qwen-vx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| edit1 | 22.45 | 13.89 | 13.54 | 12.02 | 26.29 | 20.35 | 15.96 | 1.33 |
| edit2 | 22.32 | 9.07 | 12.24 | 9.51 | 26.42 | 6.93 | 15.32 | 1.61 |
| edit3 | 22.40 | 10.55 | 13.06 | 10.49 | 26.32 | 10.61 | 15.70 | 1.50 |
| complete1 | 22.27 | 21.18 | 16.77 | 16.78 | 26.11 | 26.33 | 17.89 | 1.07 |
| complete2 | 22.37 | 21.42 | 15.68 | 15.66 | 26.09 | 26.39 | 17.81 | 1.14 |
| complete3 | 22.38 | 18.43 | 14.03 | 13.81 | 26.11 | 30.27 | 16.59 | 1.20 |
| write1 | 22.17 | 20.84 | 15.16 | 15.15 | 25.68 | 26.01 | 16.15 | 1.07 |
| write2 | 22.08 | 21.78 | 13.64 | 13.77 | 25.66 | 25.98 | 16.77 | 1.22 |
| write3 | 22.16 | 19.53 | 13.54 | 13.59 | 25.65 | 26.76 | 14.97 | 1.10 |
| explain1 | 22.37 | 21.80 | 17.19 | 17.20 | 26.24 | 26.53 | 20.26 | 1.18 |
| explain2 | 22.36 | 20.41 | 15.73 | 16.04 | 26.20 | 27.58 | 18.84 | 1.17 |
| explain3 | 22.32 | 22.54 | 17.13 | 17.15 | 26.14 | 26.37 | 20.15 | 1.17 |

The last column compares draft-mtp with `--mtp 3,5 --draft 11`. The 3 runs of a case differ by at
most 5.2% (complete1, draft-mtp).

Speed-ups, the geometric mean of the ratios of the 12 prompts (`summary.md` has all pairs):

| | Ratio | Faster on | Range |
| --- | ---: | ---: | ---: |
| `--mtp 3,5 --draft 11` against draft-mtp | 1.218 | 12 of 12 | 1.06-1.61 |
| `--mtp 3,5` against draft-mtp | 1.161 | 12 of 12 | 1.06-1.25 |
| qwen-vx against llama-completion | 1.170 | 12 of 12 | 1.16-1.18 |
| `--mtp 3,5 --draft 11` against llama-completion | 1.859 | 12 of 12 | 1.51-2.78 |
| `--draft 11` against ngram-mod | 1.224 | 11 of 12 | 0.77-1.64 |
| `--mtp 3,5 --draft 11` against ngram-mod | 1.548 | 11 of 12 | 0.74-2.19 |
| `--mtp 3,5` against qwen-vx (own speed-up) | 1.514 | 12 of 12 | 1.30-1.82 |
| `--mtp 3,5 --draft 11` against qwen-vx (own speed-up) | 1.589 | 12 of 12 | 1.29-2.35 |
| `--mtp 3,5 --draft 11` against `--mtp 3,5` | 1.050 | 6 of 12 | 0.99-1.29 |
| draft-mtp against llama-completion (own speed-up) | 1.526 | 12 of 12 | 1.29-1.73 |
| ngram-mod against llama-completion (own speed-up) | 1.201 | 3 of 12 | 0.87-3.78 |

The drafts from the context in the MTP rounds help the edits (10.62 against 12.94 ms, 1.22×), and
change the other kinds by 1% or less (0.99× to 1.01×).

### Table A: generated tokens after a prompt

256 generated tokens. The median of 3 runs, and the fastest and slowest run:

| Prompt tokens | qwen-vx ms | llama.cpp ms | qwen-vx tok/s | llama.cpp tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 22.05 (22.03-22.08) | 25.82 (25.67-25.90) | 45.4 | 38.7 | 1.171 |
| 512 | 22.20 (22.16-22.20) | 26.03 (25.82-26.49) | 45.1 | 38.4 | 1.173 |
| 2,048 | 22.62 (22.62-22.63) | 26.26 (26.26-26.28) | 44.2 | 38.1 | 1.161 |
| 8,192 | 24.16 (24.11-24.16) | 28.02 (27.88-28.11) | 41.4 | 35.7 | 1.160 |
| 16,384 | 26.60 (26.59-26.76) | 32.27 (32.26-32.31) | 37.6 | 31.0 | 1.213 |
| 32,768 | 50.49 (48.66-51.41) | 58.90 (58.00-65.10) | 19.8 | 17.0 | 1.167 |

`llama-bench` gave 39.05 tokens a second for tg128 and 18.86 at a depth of 32,768 (`summary.md`).

### Table P: the prompt

Tokens of the prompt each second, the median of 3 runs, and the slowest and fastest run. qwen-vx
ran the prompt 3 times in one process. The first run includes the first launch of each kernel.
`llama-completion` ran the prompt once, after its warm-up.

| Prompt tokens | qwen-vx first run | qwen-vx later runs | llama-completion | Ratio, later runs |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 456 (452-456) | 451 (448-453) | 419 (418-420) | 1.08 |
| 2,048 | 498 (497-499) | 490 (487-493) | 404 (403-404) | 1.21 |
| 8,192 | 431 (431-433) | 399 (394-406) | 356 (355-361) | 1.12 |

## Speculative decoding

Accepted and drafted tokens, table C round 1:

| | `--draft 11` | `--mtp 3,5` | `--mtp 3,5 --draft 11` | ngram-mod | draft-mtp |
| --- | --- | --- | --- | --- | --- |
| Accepted / drafted | 2004 / 2631 (76%) | 3963 / 4744 (84%) | 4146 / 4986 (83%) | 1274 / 1274 as printed (below) | 4394 / 5360 (82%) |
| Accepted tokens for each check | 1.73 | 1.84 | 2.10 | not printed | not printed |
| Same text as without drafts | 36 of 36 runs | 36 of 36 runs | 36 of 36 runs | 24 of 36 runs | 21 of 36 runs |

The app checks a draft with the bits of a plain step, so its text with drafts is the text of the
app without drafts in every run.

llama.cpp with drafts writes other text in some runs, as on 2026-11-20: the same 12 runs of
ngram-mod and the same 15 runs of draft-mtp leave the text of `llama-completion`, at the same
characters (`summary.md` lists them). The summary keeps these runs. The acceptance of ngram-mod
shows 100% because `llama-speculative-simple` checks the kept tokens of a partial accept again
from a checkpoint, and its counts leave out the first check. draft-mtp uses no checkpoints, and
its counts are comparable. The draft programs of llama.cpp stop after the check that passes `-n
512`, and their speed is the decode time divided by the decoded tokens. The 2026-11-20 report
explains these points ([Speculative decoding](../2026-11-20-qwen36-m4/README.md#speculative-decoding)).

## Limits

- One machine, a fanless MacBook Air. The programs run interleaved, in a reversed order in round 2.
- Batch 1 and greedy decoding only. No serving numbers (section 6 of the policy).
- Decode to 32,768 prompt tokens. Prompt speed to 8,192 tokens.
- The run does not split the gain of the app into its changes.
- The two weight files differ in the token embedding and the output matrix. The 2026-11-20 run
  measured both with one program.
- The MTP drafts of the app score only the 65,536 tokens of the lowest ids. This changes only the
  drafts, not the text.
- The MTP layers are in other files: the app's is in its weight file, and llama.cpp reads a
  GGUF of ggml-org. This run did not compare their bits.
- At 32,768 prompt tokens, both programs were slower than on 2026-11-20 (qwen-vx 43.55 to 50.49 ms,
  llama.cpp 48.97 to 58.90 ms), and their runs spread more (up to 12%). The ratio was 1.12 then and
  1.17 now. The other prompt lengths of table A are within 1% of 2026-11-20.
- The prompts are code tasks in 4 kinds. Other text can give other draft gains.

## How the run went

1. 06:55, all stages. A browser was open from about 07:48, and some runs started with it using the
   GPU. At 09:14 I stopped the run. Its output is not in this folder.
2. 09:53, all stages again, with the idle check that checks again after the cooling. It ended at
   22:34. While it ran, another job built the Vx compiler and ran its tests on the CPU, three
   times: 17:14 to 17:34, 17:58 to 18:27 and 21:32 to 22:18. The idle check made most runs wait.
   8 runs of table C started in these periods: edit3 round 2 (completion, ngram-mod, draft-mtp),
   complete3 round 2 (`--draft 11`), and write3 round 3 (completion, `--draft 11`, `--mtp 3,5`,
   `--mtp 3,5 --draft 11`).
3. 22:41, these 8 runs again, with `RERUN` and the same `OUT`. They ended at 22:55. Their first
   logs are in `logs-overlapped/` of `logs.tar.xz`. `log.txt` keeps the lines of both runs, and
   `parse.py` uses the last line of each run.

No run started with more than 5% GPU use, and every run exited with code 0.

## Commands

llama.cpp, as on 2026-11-20:

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b1b770d
    git apply /path/to/qwen-vx/docs/benchmarks/2026-11-25-qwen36-m4-mtp/llama-timing.patch
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build build -j 8 --target llama-completion llama-speculative-simple llama-bench

The model files: those of the 2026-11-20 run, and the app's weight file with the MTP layer, which
`scripts/convert.py` now writes:

    python3 scripts/convert.py --q4 --q6-lm-head ~/models/Qwen3.6-35B-A3B Qwen3.6-35B-A3B-q4-q6-mtp.vxw

This run's file came from the 2026-11-20 file and `append_mtp.py`, which writes the same MTP
tensors with the functions of `convert.py`:

    cp -c Qwen3.6-35B-A3B-q4-q6.vxw Qwen3.6-35B-A3B-q4-q6-mtp.vxw
    python3 $OUT/append_mtp.py scripts ~/models/Qwen3.6-35B-A3B Qwen3.6-35B-A3B-q4-q6-mtp.vxw

The run, under `caffeinate -dims`, on an idle Mac, from the root of this repository:

    export OUT=docs/benchmarks/2026-11-25-qwen36-m4-mtp VXC=~/.vx/current/bin/vxc FLEET=~/.vx/current/fleet \
      VXW=Qwen3.6-35B-A3B-q4-q6-mtp.vxw VXT=Qwen3.6-35B-A3B.vxt GGUF=Qwen3.6-35B-A3B-Q4_0.gguf \
      MTP=mtp-Qwen3.6-35B-A3B-Q4_0.gguf LB=llama.cpp/build/bin
    caffeinate -dims $OUT/run.sh
    RERUN="C.edit3.r2.mtp3 C.edit3.r2.ngram C.edit3.r2.completion C.complete3.r2.vxdraft11 \
      C.write3.r3.vxdraft11 C.write3.r3.vxmtp C.write3.r3.vxmtpdraft11 C.write3.r3.completion" \
      STAGES=C FA=on caffeinate -dims $OUT/run.sh
    python3 $OUT/parse.py $OUT

`run.sh` prints each command line in the log of its run.

## Files

- `run.sh`, `parse.py`, `make_prompts.py`, `prompts/`: the script, the parser and the inputs, from
  the 2026-11-20 run with the programs of the app with `--mtp` added.
- `append_mtp.py`: added the MTP layer to the app's weight file.
- `env.txt`: the machine, OS, versions, files and settings, written at each start of `run.sh`.
- `log.txt`: one line for each run, with its start, end, exit code, the busiest other process and
  the GPU use at the start.
- `logs.tar.xz`: the output of each run (`tar -xJf logs.tar.xz` gives `logs/` and
  `logs-overlapped/`). `parse.py` gives the same `summary.md` and `runs.csv` from `logs/`.
- `runs.csv`, `summary.md`: written by `parse.py`.
- `llama-timing.patch`, `llama.cpp.diff`, `llama.cpp-build.txt`: the patch, the diff of the
  llama.cpp source that ran, and its build settings.
