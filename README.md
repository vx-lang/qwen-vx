# Qwen coder models in Vx

This program runs Qwen coder models on the GPU of an Apple Silicon Mac. It is written in
[Vx](https://github.com/vx-lang/Vx), and it uses the GPU kernels of Vx's `std::nn`. It runs:

- The dense Qwen2.5-Coder models, such as 0.5B and 7B, with bf16, fp8 or Q4_0 weights.
- The mixture-of-experts (MoE) model Qwen3-Coder-30B-A3B-Instruct, in Q4_0, as the section on the
  MoE model below shows.

The plan and its milestones are in [#5](https://github.com/vx-lang/qwen-vx/issues/5). Milestones 1
to 6 are done. The MoE model was milestone 6,
[#6](https://github.com/vx-lang/qwen-vx/issues/6). Milestone 7 added a tokenizer in Vx, and the
prompt now runs on the GPU in chunks of tokens.

The program generates tokens with greedy decoding. Its tokenizer is also in Vx, so the prompt can
be text or token ids, and the program prints text or token ids.

[`docs/optimizations.md`](docs/optimizations.md) is a journal of the optimizations: what each one
changed, its measurements, and the changes that did not help.

## Install Vx

The program needs Vx v0.0.5 or later:

```sh
curl -fsSL https://vxlang.org/install.sh | sh
```

The toolchain goes in `~/.vx/current`. On a Mac, it also needs LLVM 22 (`brew install llvm@22`).
The commands below call `vxc`, so put the toolchain on `PATH`, as the installer says:

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

The scripts need Python 3 with NumPy. `--prompt` also needs the `tokenizers` package, and
`check_against_mlx.py` needs `mlx-lm`. The Python of Homebrew takes no packages of its own, so
put them in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install numpy tokenizers huggingface_hub
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
   vxc src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw \
       --tokenizer qwen.vxt --prompt "def fibonacci(n):" --steps 64
   ```

   `vxc` first warns that it cannot check the memory of some functions with a dynamic shape
   (W1029). These warnings are expected. The program then prints the prompt, the text that the
   model writes, and the time of a token, about 11 ms on an M4.

   With `--chat`, the prompt is a request to the Instruct model. The program prints only the
   reply, and it stops at the end of the reply. `--prompt-file FILE` reads the prompt from a file:

   ```sh
   vxc src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw \
       --tokenizer qwen.vxt --chat --prompt "Write a C++ function that reverses a string." --steps 300
   ```

1. Generate a few tokens with the NumPy reference. It prints the prompt as token ids:

   ```sh
   python3 scripts/reference.py qwen-0.5b.vxw \
       --tokenizer ~/models/Qwen2.5-Coder-0.5B-Instruct --prompt "def fibonacci(n):" --steps 8
   ```

1. Generate the same tokens with the Vx program. With `--tokens`, the prompt is token ids, and the
   program prints the lines of `reference.py`. `--lines` prints the same lines for a text prompt:

   ```sh
   vxc src/main.vx --package qwen=src --host default \
       --machine ~/.vx/current/fleet/m4-uma.vx -- qwen-0.5b.vxw --tokens 750,75698,1445,1648 --steps 8
   vxc src/main.vx --package qwen=src --host default \
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

## The scripts

| Script | What it does |
| --- | --- |
| `scripts/convert.py` | Converts a Hugging Face Qwen2 or Qwen3 MoE folder into a weight file. |
| `scripts/reference.py` | Computes the logits and the greedy tokens with NumPy, in float32. |
| `scripts/check_against_mlx.py` | Compares `reference.py` with MLX on the same model. |
| `scripts/compare.py` | Compares the output of the Vx program with the output of `reference.py`. |
| `scripts/make_tiny_model.py` | Writes a tiny model with random weights. |
| `scripts/qwen_weights.py` | Reads and writes the weight file. Its first comment describes the format. |
| `scripts/test_qwen_weights.py` | Tests the f8e4m3 and Q4_0 codes, their scales and the quantized weight files. |
| `scripts/importance.py` | Measures the importance matrix of a bf16 model, for `convert.py --importance`. |
| `scripts/test_importance.py` | Tests `importance.py`. |
| `scripts/convert_tokenizer.py` | Converts a tokenizer.json into a tokenizer file. Its first comment describes the format. |
| `scripts/make_tiny_tokenizer.py` | Writes a tiny tokenizer.json, and texts that test each step of the tokenizer. |
| `scripts/test_tokenizer.py` | Compares the tokenizer in Vx with the `tokenizers` package. |
| `scripts/chat.py` | Chats with a model through a session, with a cache of the system prompt. |
| `scripts/make_vx_guides.py` | Makes the guides of `docs/vx-guides` from the Vx book. |

[`examples/`](examples/README.md) asks the 30B model for programs in Rust, C++ and Python. It
compiles each one and checks its output.

## fp8 weights

`convert.py --fp8` stores each weight matrix as f8e4m3 codes, one byte for each weight, with one f32
scale for each block of 128 x 128. A weight is the value of its code times the scale of its block.
The norms, the biases and the embedding table stay bf16, so a token reads about half the bytes of
bf16. The Vx program reads an fp8 file like a bf16 file, and runs the fp8 kernels of `std::nn` for
it.

`reference.py` widens the codes and scales exactly to float32. With `--perplexity`, it measures
how well a model predicts a text. On the first 512 tokens of `scripts/tools/gen_error_index.py` in the Vx repository:

| Model | bf16 perplexity | fp8 perplexity | Greedy tokens of fp8 |
| --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 20.054 | 20.030 | the same 64 as bf16 |
| Qwen2.5-Coder-7B-Instruct | 10.072 | 10.085 | the same 64 as bf16 |

On the fp8 file, the Vx program chose the same 64 greedy tokens as `reference.py`, with each logit
within 6e-05. One token of the 7B model took 95 ms in fp8 and 156 ms in bf16 on a quiet M4, which
is about 10.5 and 6.4 tokens per second.

## Q4_0 weights

`convert.py --q4` stores each weight matrix except `lm_head` in the Q4_0 format of llama.cpp. A
row has one f16 scale for each 32 weights, and each weight is a 4-bit code from 0 to 15. The
weight is (code - 8) times its scale. The codes round as llama.cpp rounds them, and a test checks
them against the loop of llama.cpp. For Qwen2.5-Coder-7B-Instruct, the codes and scales of all
197 Q4_0 tensors were the same bits as in the Q4_0 GGUF file that Qwen publishes. `lm_head` is fp8, as llama.cpp also keeps its output matrix
above 4 bits.

`convert.py --q4 --importance FILE` rounds each group of 32 weights with an importance matrix, as
llama.cpp does with one. `importance.py` measures it: the mean square of each input of each
matrix over a sample text, for each expert on its own tokens. The converter then tries 19 scales
for each group, and keeps the one with the smallest error, where the error of a weight counts as
much as its input matters. A test checks the codes and scales against the loops of llama.cpp.

The Vx program reads a Q4_0 file like the others, and runs the Q4_0 kernels of `std::nn` for it.
Of four 4-bit formats, Q4_0 was the closest to bf16 on the 7B model. The measurement is in
[#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596230).

## The MoE model

Qwen3-Coder-30B-A3B is 61 GB in bf16, so this Mac runs it in Q4_0, at 17.8 GB:

```sh
python3 scripts/convert.py --q4 ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b-q4.vxw
vxc src/main.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen3-30b-q4.vxw --tokens 750,75698,1445,982,257 --steps 64
```

The conversion took 3.4 minutes and 18.6 GB of memory on an M4 with 32 GB. The tokens are
`def fibonacci(n):` and a new line with 4 spaces. The tokenizer of the 30B model gives the same
tokens for that text, so `--tokenizer qwen3.vxt --prompt "$(printf 'def fibonacci(n):\n    ')"` is
the same prompt, after `convert_tokenizer.py ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3.vxt`. The Vx program wrote a recursive
and an iterative `fibonacci` in the 64 steps. `reference.py` on the same file chose the same 64
tokens, with each logit within 5.4e-05.

An importance matrix makes the Q4_0 file closer to bf16. `importance.py` needs the bf16 file, and
took 110 minutes for 150 chunks of 512 tokens on an M4 with 32 GB:

```sh
python3 scripts/convert.py ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b.vxw
python3 scripts/importance.py qwen3-30b.vxw --tokenizer ~/models/Qwen3-Coder-30B-A3B-Instruct \
    --out importance.npz --chunks-per-text 8 TEXT...
python3 scripts/convert.py --q4 --importance importance.npz \
    ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b-q4.vxw
```

The sample text below was the first 8 chunks of each of 19 files:

- `SmallVector.h` and `DenseMap.h` of LLVM 22.
- `argparse.py`, `subprocess.py`, `asyncio/base_events.py`, `http/client.py` and
  `pydoc_data/topics.py` of Python 3.13.
- `src/ahocorasick.rs` of aho-corasick 1.1.4, `src/map.rs` of indexmap 2.14.0, `src/expr.rs` of
  syn 2.0.118, and `src/parser/parser.rs` of clap_builder 4.6.0.
- `src/runtime/scheduler/multi_thread/worker.rs` and `CHANGELOG.md` of tokio 1.52.3.
- `lib/compress/zstd_compress.c` of zstd 1.5.7, and `lib/commands/ls.js` and
  `lib/utils/display.js` of npm 11.16.0.
- `stdlib/std/nn.vx`, `www/book/src/tasks.md` and `www/book/src/ownership.md` of the Vx repository.

Both programs measured the perplexity as `llama-perplexity -c 512 --chunks 8 --no-escape`
measures it, on three files of the Vx repository that are not in the sample:
`src/dialect/MetalImage.cpp` at vx-lang/Vx@7e3cf6a3, `src/hir/check/calls.rs` at
vx-lang/Vx@4df81bf6, and `docs/parallel_compiler_architecture.md` at vx-lang/Vx@f2415488.

| Weights | Program | MetalImage.cpp | calls.rs | architecture.md |
| --- | --- | --- | --- | --- |
| bf16 | `reference.py` | 3.6813 | 5.9254 | 5.6790 |
| Q4_0 of `convert.py --q4` | `reference.py` | 3.7290 | 6.0010 | 5.7670 |
| Q4_0 of `convert.py --q4 --importance` | `reference.py` | 3.6881 | 5.9664 | 5.7472 |
| The Q4_0 codes of unsloth in a file of `convert.py --q4` | `reference.py` | 3.6918 | 5.9535 | 5.7516 |
| Q4_0 of unsloth with an importance matrix | `llama-perplexity` | 3.7091 | not measured | not measured |

The importance matrix removed 86% of the gap to bf16 on `MetalImage.cpp`, 46% on `calls.rs` and
23% on the architecture document.

The fourth file has the codes and scales of the unsloth file for each matrix that is Q4_0 there.
The `experts_down` of the first 6 layers are Q4_1 there, so they stay ours. The unsloth file also
has `lm_head` in Q6_K and the embedding table in Q4_0. So 3.7091 and 3.6918 differ in the program
and in some weights. Compare files with the same program.

A token took 28.0 ms in Vx with the cheaper kernel launches of
[#3152](https://github.com/vx-lang/Vx/pull/3152) and [#3165](https://github.com/vx-lang/Vx/pull/3165),
and 24.3 ms in llama.cpp. Each token runs about 477 kernels. The details are in
[#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596324).

## The prompt in chunks

A generated token reads every weight that it uses, so it waits for memory. The prompt has all its
tokens at the start, so the program runs it in chunks of up to 2048 tokens. A chunk reads each
weight once for all its tokens. `src/prefill.vx` has the kernels of a chunk:

- One product for each weight format. A block decodes 128 rows of the matrix and 32 columns at a
  time into shared memory, and multiplies them with the inputs of 16 tokens, with a matmul of
  tiles (`@`).
- For the experts of a MoE layer, the CPU sorts the tokens of the chunk by expert. Then a tile of
  16 tokens reads the weights of one expert.
- Attention in three passes: the scores, as a matmul of the queries and the keys; their softmax;
  and the sum of the values, as a matmul of the weights and the values. The rows of the matmuls
  are the query heads of one key-value head for two tokens, so a tile of the cache serves 16
  rows. The matmuls skip the positions after the tokens of their tile. Attention takes 128
  tokens of the chunk at a time, so that its scratch tensor of scores does not grow with the
  chunk.

The time of each prompt token, on an M4 with 32 GB, with the prompt as the first tokens of this
README:

| Model | Prompt | One token at a time | In chunks |
| --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct, bf16 | 600 tokens | 11.1 ms | 1.15 ms |
| Qwen3-Coder-30B-A3B-Instruct, Q4_0 | 1000 tokens | 46.7 ms | 10.8 ms |
| Qwen3-Coder-30B-A3B-Instruct, Q4_0 | 2000 tokens | | 8.1 ms |

Both ways chose the same tokens after the prompt. On the 0.5B model, `compare.py` compared 32 steps
with `reference.py`, and the largest logit difference was 5.5e-05. The times include the first
compile of the kernels of a chunk, about 4 seconds. Without it, the prompt of 2000 tokens takes
7.0 ms a token.

Before the attention in tiles, each block did one token, and so each chunk read the whole cache
once for each token in each layer. Then the prompts of 1000 and 2000 tokens took 15.1 ms and
16.2 ms a token. Now the products of the experts take most of the time of a chunk. Their `@`
loads its sum from shared memory and stores it back for each 32 columns, and the 32 KB of shared
memory does not hold larger tiles
([#19](https://github.com/vx-lang/qwen-vx/issues/19)).

`tests/prefill.vx` runs a prompt of 40 tokens through each tiny model one token at a time, and in
chunks of 16 tokens. The logits of the last token must be within 0.0001, and the next 8 tokens
equal.

## Long contexts

A generated token attends to every position before it. The `nn_attention` of Vx v0.0.4 has one
block for each query head, so its time grows with the number of positions. The `nn_attention` of
Vx main splits the positions among blocks
([vx-lang/Vx#3216](https://github.com/vx-lang/Vx/pull/3216)). `src/attention.vx` is a copy of it,
until a Vx release has it ([#4](https://github.com/vx-lang/qwen-vx/issues/4)).

The time of a generated token of Qwen3-Coder-30B-A3B in Q4_0, on an M4 with 32 GB, after a prompt
of the first tokens of this README:

| Positions | One block for each head | Positions split among blocks |
| --- | --- | --- |
| 5 | 26.7 ms | 25.4 ms |
| 1000 | 40.7 ms | 27.5 ms |
| 2000 | 57.3 ms | 30.8 ms |

Both chose the same tokens. On Qwen2.5-Coder-0.5B, 100 tokens after a prompt of 600 went past
position 640, where a block takes 64 positions in place of 32. `compare.py` gave the same tokens
as `reference.py`, with each logit within 5.3e-05. `tests/generate.vx` also runs the tiny models
with 1024 positions after a prompt of 700 tokens.

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

On an M4 with 32 GB, with a system prompt of the first 8000 characters of this README, and
Qwen3-Coder-30B-A3B in Q4_0:

| Turn | Prompt | Tokens that ran | Time of the prompt |
| --- | --- | --- | --- |
| The first, with no cache | 2465 tokens | 2465 | 50.0 s |
| The second, in the same session | 2539 tokens | 29 | 2.3 s |
| The first, in a new session that loaded the cache of the system prompt | 2465 tokens | 20 | 2.5 s |

The KV cache holds f16 values, 98 KB for each token of Qwen3-Coder-30B-A3B, so the cache file of
the 2445 tokens of the system prompt takes 240 MB. With f32 values it took 481 MB, 0.4 s to save
and 1.9 s to load, and the reply after the load was the same text as the reply with no cache. A session
takes about a minute to start, to compile the program and to load the weights.

[`docs/vx-guides`](docs/vx-guides) has system prompts that teach Qwen3-Coder to write Vx, which
it never learned. Each one is a set of chapters of the Vx book for one kind of program, such as
kernels on the GPU of a Mac, of about 13000 tokens. With `--cache`, it runs once.

`tests/session.vx` runs a conversation of three turns on the tiny model with
`scripts/chat.py --check`. Each turn must give the text of a fresh run, and a saved cache must give
the same text in a new session.

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

The tests are in `tests/`. They need a Python 3 that has NumPy and ml_dtypes, and `FileCheck` from
LLVM or the Python port (`pip install numpy ml_dtypes filecheck`). `tests/run.sh` takes `vxc` and the
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

## The tokenizer

`src/tokenizer.vx` is the byte-level BPE tokenizer of Qwen2 and Qwen3. It splits out the added
tokens, such as `<|im_start|>`, and applies Unicode NFC to the text between them. Then it splits
the text with the regex of the pre-tokenizer, and merges the bytes of each piece by the ranks of
tokenizer.json. `convert_tokenizer.py` describes the file and the steps. The file holds sorted
tables, so each lookup is a binary search in the mapped file.

The tokenizer gives the ids of the `tokenizers` package of Hugging Face. On 21.5 MB of text,
`test_tokenizer.py` found the same ids for each of 4783 texts, with both the Qwen2.5-Coder and the
Qwen3-Coder tokenizer. The texts were this repository, Python, Rust and C++ sources, and 3000 texts
of random Unicode characters, marks, Hangul, emoji, white space and added tokens. The Vx program
encoded 3 MB of source files, 820,000 tokens, in about 2.4 seconds.

The NFC step of `tokenizers` has the data of an old Unicode version. It does not know the
combining class of 120 marks that later versions added, and it does not make 21 newer composites.
`convert_tokenizer.py` leaves both out of its tables, so that the ids are the same.
`test_tokenizer.py` measures the two lists again, and fails when they change.

`tests/tokenize.vx` checks a tiny tokenizer from `make_tiny_tokenizer.py` against the ids that
`tokenizers` gave for it. So CI checks the tokenizer without a download. To compare with
`tokenizers` on a large corpus:

```sh
QWEN_TOKENIZER_CORPUS_MB=40 python3 scripts/test_tokenizer.py
```

## The check of the Vx program

`compare.py` compared the Vx program with `reference.py` over 64 greedy steps, after the prompt
`def fibonacci(n):` and a new line. Both chose the same tokens.

| Model | Largest logit difference |
| --- | --- |
| tiny model | 4.7e-05, over 16 steps |
| tiny model, Q4_0 | 4.8e-05, over 16 steps |
| tiny MoE model | 4.7e-05, over 16 steps |
| tiny MoE model, Q4_0 | 5.0e-05, over 16 steps |
| Qwen2.5-Coder-0.5B-Instruct | 5.3e-05 |
| Qwen2.5-Coder-7B-Instruct | 6.0e-05 |

The tiny model runs in CI with `tests/generate.vx`. That test fails when RoPE uses another theta,
when a token runs at the wrong position, and when the two norms of a layer swap their weights.
For the tiny MoE model, it fails when the router reads the hidden state before its norm, when a
token uses one expert in place of two, and when the Q and K norms are missing or swapped.

## The check of the reference

`check_against_mlx.py` compared the reference with MLX in float32 over a 5-token prompt and 32
greedy steps. Both chose the same tokens.

| Model | Largest logit difference | Time on an M4 | Peak memory |
| --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 8.7e-05 | 7 s | 5 GB |
| Qwen2.5-Coder-7B-Instruct | 6.3e-05 | 19 min | 15 GB |

The check fails, as it should, when RoPE uses another theta, when the causal mask lets a token
see the next one, when the program does not add the Q, K and V bias, and when the converter swaps
K and V.

For the tiny MoE model, `make_tiny_model.py --moe --hugging-face DIR` also writes the model as a
Hugging Face folder, which MLX reads:

```sh
python3 scripts/make_tiny_model.py --moe --hugging-face tiny-moe tiny-moe.vxw
python3 scripts/check_against_mlx.py --tokens 17,200,3,95,41 tiny-moe tiny-moe.vxw
```

Over 32 greedy steps, both chose the same tokens, and the largest logit difference was 8.7e-06.
The check fails when the router chooses the smallest experts or one expert too few, or when it
does not scale the weights of the chosen experts to sum to 1. It also fails when an expert uses
the down matrix of the next expert, when the converter puts the up rows before the gate rows, or
when the Q and K norms are missing, swapped or after RoPE.
