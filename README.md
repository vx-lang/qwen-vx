# Qwen models in Vx

This program runs Qwen models on the GPU of an Apple Silicon Mac. It is written in
[Vx](https://github.com/vx-lang/Vx), and it uses the GPU kernels of Vx's `std::nn`. It generates
tokens with greedy decoding. Its tokenizer is also in Vx, so the prompt and the output are text.

It runs these models:

- **Qwen3.6-35B-A3B**, in Q4_0 with a 6-bit output matrix. This mixture-of-experts (MoE) model
  mixes layers of linear attention (Gated DeltaNet) with layers of full attention. The program can
  draft tokens with the multi-token prediction (MTP) layer of the model.
- Qwen3-Coder-30B-A3B-Instruct, an MoE model, in Q4_0.
- The dense Qwen2.5-Coder models, such as 0.5B and 7B, with bf16, fp8 or Q4_0 weights.

## Speed

Qwen3.6-35B-A3B on a MacBook Air M4 with 32 GB, against llama.cpp b1b770d. Batch 1, greedy
decoding, 512 generated tokens, on 12 held-out code prompts (edit, complete, write and explain).
Generated tokens a second, the geometric mean over 3 runs of each prompt:

| Decoding | qwen-vx | llama.cpp | Ratio | qwen-vx faster on |
| --- | ---: | ---: | ---: | ---: |
| Without drafts | 44.8 | 38.3 | 1.17× | 12 of 12 prompts |
| MTP drafts: `--mtp 3,5 --draft 11` against `draft-mtp` (3 tokens) | 71.2 | 58.5 | 1.22× | 12 of 12 prompts |
| Drafts from the context: `--draft 11` against `ngram-mod` | 56.3 | 46.0 | 1.22× | 11 of 12 prompts |

The prompt runs at 1.08× to 1.21× the speed of llama.cpp from 512 to 8,192 tokens, and generated
tokens at 1.16× to 1.21× after prompts of 64 to 32,768 tokens. With drafts, the text is the same as
without drafts. The run followed the benchmarking policy of the Vx repository. Its report, with
the setup, every run, the cases where qwen-vx is slower and the limits, is
[`docs/benchmarks/2026-11-25-qwen36-m4-mtp/`](docs/benchmarks/2026-11-25-qwen36-m4-mtp/README.md).

## Install Vx

The program needs a Mac with Apple Silicon, and the Vx release that [`VX_VERSION`](VX_VERSION)
names (now v0.0.5). The toolchain also needs LLVM 22 and z3 from Homebrew:

```sh
brew install llvm@22 z3
curl -fsSL https://vxlang.org/install.sh | VX_VERSION=$(cat VX_VERSION) sh
export PATH="$HOME/.vx/bin:$PATH"
```

[`docs/usage.md`](docs/usage.md#install-vx) tells how to install Vx in another folder.

## Run Qwen3.6-35B-A3B

The scripts need Python 3.12 or later and the packages of [`requirements.txt`](requirements.txt).
The download is about 72 GB, and the weight file is about 21 GB:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
hf download Qwen/Qwen3.6-35B-A3B --local-dir ~/models/Qwen3.6-35B-A3B
python3 scripts/convert.py --q4 --q6-lm-head ~/models/Qwen3.6-35B-A3B qwen36.vxw
python3 scripts/convert_tokenizer.py ~/models/Qwen3.6-35B-A3B qwen36.vxt
```

Then ask the model:

```sh
vxc -O2 src/main.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen36.vxw --tokenizer qwen36.vxt \
    --chat --prompt "Write a C++ function that reverses a string." --steps 512 --mtp 3,5 --draft 11
```

`vxc` compiles the program and runs it. It first warns that it cannot check the memory of some
functions with a dynamic shape (W1029). These warnings are expected. Use `-O2` for every run that
you time.

## Options

| Option | What it does |
| --- | --- |
| `--prompt TEXT`, `--prompt-file FILE` | The prompt, as text. It needs `--tokenizer FILE`. |
| `--chat` | Puts the prompt in the chat format, and prints only the reply. |
| `--think` | With `--chat`, lets Qwen3.6 think before it replies. |
| `--steps N` | Generates at most N tokens. The program also stops at the end of the reply. |
| `--ignore-eos` | Does not stop at the end of the reply. |
| `--mtp K[,L]` | Drafts tokens with the MTP layer of Qwen3.6: up to K tokens while the model keeps most drafts (1 token otherwise), and up to L while it keeps almost all of them. |
| `--draft D[,M]` | Drafts up to D tokens from the context, after a match of the last tokens. With `--mtp`, a round drafts from the context only after a match of M tokens (16 by default). |
| `--tokens IDS`, `--lines` | The prompt as token ids, and the output as the lines of `scripts/reference.py`. |
| `--session` | Keeps the model and its cache between the turns of `scripts/chat.py`. |

Drafts do not change the text: the model checks each draft, and keeps only the tokens that it
would generate without drafts. They make the program faster when the model keeps most of them.
`--draft` helps most when the reply repeats the prompt, as an edit of a file does. The benchmark
uses `--mtp 3,5 --draft 11` for Qwen3.6. The first comment of [`src/main.vx`](src/main.vx)
describes all the options.

## Tests

The tests need the packages of [`requirements-test.txt`](requirements-test.txt). They run on tiny
models with random weights, so they need no download:

```sh
pip install -r requirements-test.txt
tests/run.sh
```

## Documentation

| File | What it has |
| --- | --- |
| [`docs/usage.md`](docs/usage.md) | The installation, the dense and the 30B models, sessions and the prompt cache, the scripts, the Vx code and the tests. |
| [`docs/weights.md`](docs/weights.md) | The fp8, Q4_0 and 6-bit formats, the importance matrix, and their perplexity. |
| [`docs/internals.md`](docs/internals.md) | The prompt in chunks, the attention of long contexts, and the tokenizer. |
| [`docs/checks.md`](docs/checks.md) | The checks of the Vx program against the NumPy reference, and of the reference against MLX. |
| [`docs/optimizations.md`](docs/optimizations.md) | A journal of the optimizations: what each one changed, and its measurements. |
| [`docs/benchmarks/`](docs/benchmarks) | The runs against llama.cpp, with their logs and scripts. |
| [`docs/vx-guides/`](docs/vx-guides) | System prompts that teach Qwen3-Coder to write Vx. |
| [`examples/`](examples/README.md) | The 30B model writes programs in Rust, C++ and Python, and a script compiles and checks each one. |

The plan and its milestones are in [#5](https://github.com/vx-lang/qwen-vx/issues/5).
