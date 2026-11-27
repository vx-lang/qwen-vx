# The weight formats

The fp8, Q4_0 and 6-bit formats of `scripts/convert.py`, and the measurements of the MoE model.

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

`convert.py --q4 --q6-lm-head` stores `lm_head` in 6 bits in place of fp8: a code from 0 to 63 for
each weight, and an f16 scale for each 32 weights of a row. The weight is (code - 32) times its
scale. That is 6.5 bits for each weight, as in the Q6_K of llama.cpp, and the codes round as its
`make_qx_quants` rounds them. The `lm_head` of Qwen3-Coder-30B-A3B is then 253 MB in place of
311 MB, and `src/q6.vx` reads it. See [the MoE model](#the-moe-model) for its cost in perplexity.

The Vx program reads a Q4_0 file like the others, and runs the Q4_0 kernels of `std::nn` for it.
Of four 4-bit formats, Q4_0 was the closest to bf16 on the 7B model. The measurement is in
[#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596230).

## The MoE model

Qwen3-Coder-30B-A3B is 61 GB in bf16, so this Mac runs it in Q4_0, at 17.8 GB:

```sh
python3 scripts/convert.py --q4 ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3-30b-q4.vxw
vxc -O2 src/main.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen3-30b-q4.vxw --tokens 750,75698,1445,982,257 --steps 64
```

The conversion took 3.4 minutes and 18.6 GB of memory on an M4 with 32 GB. The tokens are
`def fibonacci(n):` and a new line with 4 spaces. The tokenizer of the 30B model gives the same
tokens for that text, so `--tokenizer qwen3.vxt --prompt "$(printf 'def fibonacci(n):\n    ')"` is
the same prompt, after `convert_tokenizer.py ~/models/Qwen3-Coder-30B-A3B-Instruct qwen3.vxt`. The Vx program wrote a recursive
and an iterative `fibonacci` in the 64 steps. `reference.py` on the same file chose the same 64
tokens, with each logit within 5.4e-05.

The program gives the GPU the pages of the mapped weight file, with no copy. Vx has this since
v0.0.5 ([vx-lang/Vx#3606](https://github.com/vx-lang/Vx/pull/3606)). On
an M4 with 32 GB, the load takes about 10.5 s, and the program has 0.3 GB of private memory at
a short context and 2.1 GB at 8,000 tokens of context. When it copied the weights, the load took
29 to 38 s and 18.9 to 20.4 GB. The file cache keeps one copy of the weights for all the runs, and
macOS can drop those pages when other programs need memory.

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

`lm_head` in 6 bits (`--q6-lm-head`) costs 0 to 0.25% of perplexity. This table has the Q4_0 file of
`convert.py --q4` with three forms of `lm_head`, all measured again with `reference.py` on
2026-11-17. For the bf16 row, the script took `lm_head` from the bf16 file.

| `lm_head` | Bytes a token | MetalImage.cpp | calls.rs |
| --- | --- | --- | --- |
| bf16 | 622 MB | 3.7395 | 6.0024 |
| fp8 | 311 MB | 3.7283 | 6.0032 |
| 6 bits | 253 MB | 3.7392 | 6.0177 |

On `MetalImage.cpp`, fp8 happens to score below bf16, and 6 bits scores the same as bf16.
A scale for each 16 weights in place of 32 gave 3.7393 on `MetalImage.cpp`, and rounding to the
largest weight of each group, without the search of `make_qx_quants`, gave 3.7392, so neither helps.

The fourth file has the codes and scales of the unsloth file for each matrix that is Q4_0 there.
The `experts_down` of the first 6 layers are Q4_1 there, so they stay ours. The unsloth file also
has `lm_head` in Q6_K and the embedding table in Q4_0. So 3.7091 and 3.6918 differ in the program
and in some weights. Compare files with the same program.

A token took 28.0 ms in Vx with the cheaper kernel launches of
[#3152](https://github.com/vx-lang/Vx/pull/3152) and [#3165](https://github.com/vx-lang/Vx/pull/3165),
and 24.3 ms in llama.cpp. Each token runs about 477 kernels. The details are in
[#6](https://github.com/vx-lang/qwen-vx/issues/6#issuecomment-6879596324).
