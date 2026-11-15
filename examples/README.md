# Coding tasks

The files in `prompts/` ask Qwen3-Coder-30B-A3B-Instruct for complete programs in Rust, C++ and
Python. Each program has one correct output, so a test can check it. `run.sh` gives each request
to the Vx program, in the chat format of the Instruct model, and decodes the reply. `check.sh` then
compiles each program, runs it, and compares its output with the correct one.

```sh
examples/run.sh qwen3-30b-q4.vxw ~/models/Qwen3-Coder-30B-A3B-Instruct
SKILLS=1 examples/run.sh qwen3-30b-q4.vxw ~/models/Qwen3-Coder-30B-A3B-Instruct
```

The weight file comes from `convert.py --q4`, as the main README shows. `run.sh` needs a Python
with `tokenizers`, and `check.sh` needs `rustc`, `clang++` and `timeout` (`brew install coreutils`).
With `SKILLS=1`, the Rust tasks get `RUST-SKILLS.md` as the system message.

## Results

The Vx program chose each token greedily, on an M4 with 32 GB. Each reply ended with
`<|im_end|>` before the limit of 700 tokens.

| Task | Language | Without `RUST-SKILLS.md` | With `RUST-SKILLS.md` |
| --- | --- | --- | --- |
| `cpp_lru`: an LRU cache template with `std::list` and `std::unordered_map` | C++ | Pass | |
| `cpp_primes`: the primes below 1000000, in 4 threads | C++ | Pass | |
| `python_nqueens`: the N-Queens counts for n = 1 to 10 | Python | Pass | |
| `rust_lexer`: a lexer whose tokens borrow from the input | Rust | Pass | Pass |
| `rust_wordfreq`: the 3 most common words of standard input | Rust | No `use std::io::Read` | Pass |
| `rust_csv`: people from CSV lines, sorted by age | Rust | No `use std::io::Read` | Skips the first person |
| `rust_bst`: a generic binary search tree | Rust | A lifetime is missing | A lifetime is missing |

In `rust_bst`, the helper `in_order_node(node: &Option<Box<Node<T>>>, result: &mut Vec<&T>)`
needs a named lifetime. With `RUST-SKILLS.md`, `rust_csv` compiles, but it does not clear its line
buffer after the header. So the first person joins the header line and is skipped.

`RUST-SKILLS.md` was written after the first two Rust tasks failed. `rust_csv` and `rust_lexer`
came after it. One earlier version of the file gave `struct Lexer<'a>` as its example, and with that
version `rust_csv` passed. So a small change in the system message can change the result.

A second turn can give the model the error of `rustc`. Then `rust_wordfreq` was correct. For
`rust_bst`, the model wrote the same helper again, although `rustc` showed the fix.
`harness.py encode_fix` makes the prompt of that second turn.

## The mistakes are the model's

The Vx program computes the same tokens as `reference.py`. On `rust_wordfreq`, `reference.py`
chose the same 48 tokens on the same Q4_0 file, with each logit within 7.0e-05. The 48 tokens
include the missing `use`. llama.cpp with the Q4_0 GGUF file of unsloth made the same two errors
in `rust_wordfreq` and `rust_bst`. Its replies differ later from those of the Vx program, because
the two files round some weights differently.

## The time of a token

A token takes longer when the context is longer. In these runs, a token took 23 ms at 70
positions, and 27 to 32 ms at 300 to 750 positions. With `RUST-SKILLS.md`, the prompts are
about 600 tokens long, and a token took 41 to 51 ms. On `rust_wordfreq`, llama.cpp took 24.5 ms
for a token, and the Vx program took 26.7 ms.

The Vx program runs the prompt one token at a time, at about the time of a generated token.
llama.cpp runs the prompt in batches, at about 4 ms for each token of the prompt.
