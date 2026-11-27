# Using the program

How to install Vx, prepare a model, run the program, use a session, and run the tests.

## Install Vx

The program needs a Mac with Apple Silicon for the GPU, and the Vx release that the file
[`VX_VERSION`](../VX_VERSION) names (now v0.0.6). CI tests that release. On a Mac, the Vx toolchain
also needs LLVM 22 and z3 from Homebrew:

```sh
brew install llvm@22 z3
curl -fsSL https://vxlang.org/install.sh | VX_VERSION=$(cat VX_VERSION) sh
```

The toolchain goes in `~/.vx/current`. The commands below call `vxc`, so put the toolchain on
`PATH`, as the installer says:

```sh
export PATH="$HOME/.vx/bin:$PATH"
```

To install Vx in another folder, give it as `VX_HOME`, and keep `VX_HOME` set:

```sh
export VX_HOME="$HOME/tools/vx"
curl -fsSL https://vxlang.org/install.sh | sh
export PATH="$VX_HOME/bin:$PATH"
```

The commands below then name `$VX_HOME` in place of `~/.vx`, as in
`--machine $VX_HOME/current/fleet/m4-uma.vx`.

[Getting started](https://vxlang.org/docs/getting-started.html) has the details. On a computer
with no Apple GPU, the same program runs on the CPU.

## Prepare a model

The scripts need Python 3.12 or later and the packages of [`requirements.txt`](../requirements.txt),
at the versions that it pins. `check_against_mlx.py` also needs those of
[`requirements-mlx.txt`](../requirements-mlx.txt). The Python of Homebrew takes no packages of its
own, so put them in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

1. Download a model from Hugging Face, for example
   [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), which
   is 1 GB:

   ```sh
   hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir ~/models/Qwen2.5-Coder-0.5B-Instruct
   ```

1. Convert it into the weight file and the tokenizer file that the Vx program reads:

   ```sh
   python3 scripts/convert.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen-0.5b.vxw
   python3 scripts/convert_tokenizer.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen.vxt
   ```

1. Generate text with the Vx program, on the GPU:

   ```sh
   vxc -O2 src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw \
       --tokenizer qwen.vxt --prompt "def fibonacci(n):" --steps 64
   ```

   `-O2` optimizes the code that runs on the CPU, which vxc compiles at `-O0` by default. The
   kernels are the same, but the prompt is faster: the CPU makes the tensors of a chunk in about
   1 ms in place of about 50 ms for each 100 MB. Use `-O2` for every run that you time.

   `vxc` first warns that it cannot check the memory of some functions with a dynamic shape
   (W1029). These warnings are expected. The program then prints the prompt, the text that the
   model writes, and the time of a token, about 11 ms on an M4.

   With `--chat`, the prompt is a request to the Instruct model. The program prints only the
   reply, and it stops at the end of the reply. `--prompt-file FILE` reads the prompt from a file:

   ```sh
   vxc -O2 src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw \
       --tokenizer qwen.vxt --chat --prompt "Write a C++ function that reverses a string." --steps 300
   ```

   With `--draft 7`, the program checks drafts from the context: the tokens after the longest
   earlier match of the last tokens. When the model keeps a draft, several tokens come from one
   read of the weights, and the text is the same as without `--draft`. This helps most when the
   reply repeats the prompt, as an edit of a file does: about 1.8 times as fast with the 30B
   model (`docs/optimizations.md`, entry 14). It needs the 30B model in Q4_0 with the 6-bit
   `lm_head` (`scripts/convert.py --q4 --q6-lm-head`), and `--steps` of at least 16. With Qwen3.6,
   use `--draft 11`: its rule (`draft_length35` in `src/speculate.vx`) drafts up to 11 tokens.

   With `--mtp 3`, a Qwen3.6 model drafts tokens with its multi-token prediction (MTP) layer, and
   checks them as `--draft` does, so the text is the same as without `--mtp`. This helps also when
   the reply is new text. A round drafts up to 3 tokens while the model keeps most drafts, and 1
   token otherwise. The drafts come from the 65,536 tokens of the lowest ids; `--mtp-vocab N`
   gives another number, a multiple of 8. It needs a weight file with the MTP layer, which
   `scripts/convert.py --q4 --q6-lm-head` writes for Qwen3.6-35B-A3B.

   `--mtp 3,5` drafts up to 5 tokens while the model keeps almost all drafts. With
   `--mtp 3,5 --draft 11`, a round drafts up to 11 tokens from the context when the context has a
   match of 16 tokens, and from the MTP layer otherwise. This helps in edits.

1. Generate a few tokens with the NumPy reference. It prints the prompt as token ids:

   ```sh
   python3 scripts/reference.py qwen-0.5b.vxw \
       --tokenizer ~/models/Qwen2.5-Coder-0.5B-Instruct --prompt "def fibonacci(n):" --steps 8
   ```

1. Generate the same tokens with the Vx program. With `--tokens`, the prompt is token ids, and the
   program prints the lines of `reference.py`. `--lines` prints the same lines for a text prompt:

   ```sh
   vxc -O2 src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
   vxc -O2 src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw \
       --tokenizer qwen.vxt --prompt "def fibonacci(n):" --lines --steps 8
   ```

   `compare.py` checks the output of `reference.py` and of the Vx program, when each is saved in a
   file:

   ```sh
   python3 scripts/compare.py reference.txt vx.txt
   ```

`make_tiny_model.py` writes a model with the same layers and random weights. It is small enough
for tests that cannot download a real model. With `--moe`, it writes a tiny model with the layers
of Qwen3-Coder-30B-A3B.

## Sessions and the prompt cache

`scripts/chat.py` keeps the model and its KV cache between turns. It runs `src/main.vx --session`,
and sends each turn the whole conversation. The session finds the longest prefix of tokens that the
cache holds, and runs only the tokens after it:

```sh
python3 scripts/convert_tokenizer.py ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3.vxt
python3 scripts/chat.py qwen3-30b-q4.vxw qwen3.vxt --system-file SYSTEM.md --cache system.kv
```

With `--cache`, the first session runs the system prompt once and saves its KV cache to the file.
Later sessions load the file, so the system prompt does not run again. A cache file fits only the
model and the weight file that made it.

vxc does not give its standard input to the program, and it keeps the standard output of the
program until the program ends. So a session reads its requests from one named pipe and writes its
replies to another. The comment at the start of `src/session.vx` describes the requests.

On an M4 with 32 GB, with a system prompt of the first 8000 characters of the README of that
time, and Qwen3-Coder-30B-A3B in Q4_0:

| Turn | Prompt | Tokens that ran | Time of the prompt |
| --- | --- | --- | --- |
| The first, with no cache | 2465 tokens | 2465 | 50.0 s |
| The second, in the same session | 2539 tokens | 29 | 2.3 s |
| The first, in a new session that loaded the cache of the system prompt | 2465 tokens | 20 | 2.5 s |

The KV cache holds f16 values, 98 KB for each token of Qwen3-Coder-30B-A3B, so the cache file of
the 2445 tokens of the system prompt takes 240 MB. With f32 values it took 481 MB, 0.4 s to save
and 1.9 s to load, and the reply after the load was the same text as the reply with no cache. A session
takes about a minute to start, to compile the program and to load the weights.

[`docs/vx-guides`](vx-guides) has system prompts that teach Qwen3-Coder to write Vx, which
it never learned. Each one is a set of chapters of the Vx book for one kind of program, such as
kernels on the GPU of a Mac, of about 13000 tokens. With `--cache`, it runs once.

`tests/session.vx` runs a conversation of three turns on the tiny model with
`scripts/chat.py --check`. Each turn must give the text of a fresh run, and a saved cache must give
the same text in a new session.

## The scripts

| Script | What it does |
| --- | --- |
| `scripts/convert.py` | Converts a Hugging Face Qwen2, Qwen3 MoE or Qwen3.5 MoE folder into a weight file. |
| `scripts/reference.py` | Computes the logits and the greedy tokens with NumPy, in float32. |
| `scripts/check_against_mlx.py` | Compares `reference.py` with MLX on the same model. |
| `scripts/compare.py` | Compares the output of the Vx program with the output of `reference.py`. |
| `scripts/make_tiny_model.py` | Writes a tiny model with random weights: dense, `--moe`, or `--qwen35` with the layers of Qwen3.6-35B-A3B. |
| `scripts/qwen_weights.py` | Reads and writes the weight file. Its first comment describes the format. |
| `scripts/test_qwen_weights.py` | Tests the f8e4m3, Q4_0 and 6-bit codes, their scales, the quantized weight files and the files of Qwen3.5 MoE models. |
| `scripts/importance.py` | Measures the importance matrix of a bf16 model, for `convert.py --importance`. |
| `scripts/test_importance.py` | Tests `importance.py`. |
| `scripts/convert_tokenizer.py` | Converts a tokenizer.json into a tokenizer file. Its first comment describes the format. |
| `scripts/make_tiny_tokenizer.py` | Writes a tiny tokenizer.json, and texts that test each step of the tokenizer. |
| `scripts/test_tokenizer.py` | Compares the tokenizer in Vx with the `tokenizers` package. |
| `scripts/chat.py` | Chats with a model through a session, with a cache of the system prompt. |
| `scripts/make_vx_guides.py` | Makes the guides of `docs/vx-guides` from the Vx book. |
| `scripts/make_draft_prompts.py` | Writes the 12 code prompts that measure decoding with drafts. |
| `scripts/record_experts.cpp` | Records the greedy output and the experts of every position, with llama.cpp. Its first comment says how to build it. |
| `scripts/simulate_drafts.py` | Replays the recorded outputs as rounds of decoding with drafts, and estimates the speedup of each draft policy. |
| `scripts/test_simulate_drafts.py` | Tests `simulate_drafts.py`. |

[`examples/`](../examples/README.md) asks the 30B model for programs in Rust, C++ and Python. It
compiles each one and checks its output.

## The Vx code

| File | What it does |
| --- | --- |
| `src/weights.vx` | Loads a weight file into GPU memory, for the kernels of `std::nn`. |
| `src/model.vx` | Makes the tensors and the KV cache once, and runs one token through all layers. |
| `src/prefill.vx` | Runs the prompt through all layers in chunks of tokens. |
| `src/attention.vx` | The attention of a generated token, with the positions split among blocks. |
| `src/tokenizer.vx` | Turns text into token ids and token ids into text, with a tokenizer file. |
| `src/session.vx` | Keeps the model and the KV cache, and answers the requests of `scripts/chat.py`. |
| `src/main.vx` | Generates tokens with greedy decoding, and prints text or the lines of `reference.py`. |

A program imports the modules as `qwen::weights`, `qwen::model`, `qwen::attention`, `qwen::prefill`,
`qwen::tokenizer` and `qwen::session`, with `vxc --package qwen=src`. A program in another folder
names the `src` folder of this repository.

The tests are in `tests/`. They need the packages of [`requirements-test.txt`](../requirements-test.txt)
(`pip install -r requirements-test.txt`): NumPy, ml_dtypes, and the Python port of `FileCheck`.
A `FileCheck` of LLVM on `PATH` comes first. `tests/run.sh` takes `vxc` and the
machine files from `$VX_HOME/current` when `VX_HOME` is set, else from the `vxc` on `PATH`, else
from `~/.vx/current`:

```sh
tests/run.sh
```

`tests/print_weights.vx` also checks a real model. Its lines must equal those of
`reference.py --tensors`:

```sh
vxc tests/print_weights.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw
python3 scripts/reference.py qwen-0.5b.vxw --tensors
```

## The licenses of the models

The repository has no model weights or tokenizer files. `scripts/convert.py` and
`scripts/convert_tokenizer.py` make them from a model of Hugging Face, which has its own license:
see the model card, for example
[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct).
