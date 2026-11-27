# How the program works

The prompt in chunks, the attention of long contexts, the tokenizer, and the rights of the program.

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

The time of each prompt token, on an M4 with 32 GB, with the prompt as the first tokens of the
README of that time:

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
of the first tokens of the README of that time:

| Positions | One block for each head | Positions split among blocks |
| --- | --- | --- |
| 5 | 26.7 ms | 25.4 ms |
| 1000 | 40.7 ms | 27.5 ms |
| 2000 | 57.3 ms | 30.8 ms |

Both chose the same tokens. On Qwen2.5-Coder-0.5B, 100 tokens after a prompt of 600 went past
position 640, where a block takes 64 positions in place of 32. `compare.py` gave the same tokens
as `reference.py`, with each logit within 5.3e-05. `tests/generate.vx` also runs the tiny models
with 1024 positions after a prompt of 700 tokens.

## The tokenizer

`src/tokenizer.vx` is the byte-level BPE tokenizer of Qwen2, Qwen3 and Qwen3.5. It splits out the added
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
`test_tokenizer.py` measures the two lists again, and fails when they change. The tables come from
the `unicodedata` module of Python, so `convert_tokenizer.py` needs the Unicode version of
`tokenizers`, 16.0, which Python 3.14 has. It refuses another version.

`tests/tokenize.vx` checks a tiny tokenizer from `make_tiny_tokenizer.py` against the ids that
`tokenizers` gave for it. So CI checks the tokenizer without a download. To compare with
`tokenizers` on a large corpus:

```sh
QWEN_TOKENIZER_CORPUS_MB=40 python3 scripts/test_tokenizer.py
```

## The rights of the program

`main` takes the rights of the program from Vx: `fn main(root : Root, args : Vec<String>)`. It
passes `&root.files`, the right to open files, to the functions that open the weight file, the
tokenizer file, a prompt file and a cache file. It passes `root.stdout`, the right to print, to the
functions that print the tokens. Safe code can open a file only with a `&Files`. The `print!` and
`println!` of Vx still need no right. The program declares no `extern` function. It still calls
`malloc` and `free` of `std::libc` for its buffers.
