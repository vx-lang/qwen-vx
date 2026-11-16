# Guides that teach Qwen to write Vx

Qwen3-Coder learned no Vx, because Vx came after the model. Asked for a Vx program, it writes
Rust. Each guide here is a system prompt that teaches it the Vx of one kind of program. A guide
starts with the rules that a model with Rust habits gets wrong, and the chapters "Vx for Rust
programmers" and "Vx in one page" of the Vx book. Then it has the chapters of the book for its
kind of program, with their programs, which are tests of the Vx compiler.

<!-- guides: start -->
| Guide | For | Size | Tokens | KV cache |
| --- | --- | --- | --- | --- |
| [`vx-basics.md`](vx-basics.md) | Programs on the CPU: errors, ownership and generics | 41 KB | about 13,900 | 1.4 GB |
| [`vx-gpu.md`](vx-gpu.md) | Kernels on the GPU of a Mac: topologies, memory, blocks, lanes and tiles | 40 KB | about 13,600 | 1.3 GB |
| [`vx-tasks.md`](vx-tasks.md) | Concurrent programs: tasks and channels | 37 KB | about 12,400 | 1.2 GB |
| [`vx-tensors.md`](vx-tensors.md) | Numeric programs: tensor operations, low precision and autodiff | 37 KB | about 12,600 | 1.2 GB |
| [`vx-unsafe.md`](vx-unsafe.md) | Systems programs: unsafe code, C functions, files, input and time | 38 KB | about 13,000 | 1.3 GB |

Made from the book of vx-lang/Vx at v0.0.4.
<!-- guides: end -->

## Use a guide

Run a session with the guide as the system prompt, and a file for its KV cache:

```sh
VXC=~/.vx/bin/vxc python3 scripts/chat.py qwen3-30b-q4.vxw qwen3.vxt \
    --system-file docs/vx-guides/vx-gpu.md --cache vx-gpu.kv --context 20480 --steps 1024
```

The first session runs the guide once and saves its KV cache to `vx-gpu.kv`. On an M4 with 32 GB,
a guide of 7000 tokens took about a minute, and each guide here is about twice that. Later sessions load the file in a few seconds, so only
the questions run. The cache fits only the model file and the guide that made it: after a change
to either, delete the cache file.

`--context` sets the size of the KV cache in memory, in tokens. It must hold the guide and the
conversation: about 0.1 MB a token for Qwen3-Coder-30B-A3B, so 2 GB for 20480 tokens, on top of
the 17.8 GB of weights. 20480 holds each guide and at least 4000 tokens of conversation.

Compile what the model writes, and give it back the errors:

```sh
~/.vx/bin/vxc program.vx --host default --machine ~/.vx/current/fleet/m4-uma.vx
```

The errors of vxc are in plain English, and the guide tells the model to give the whole program
again with the fix.

## Make the guides again

The guides come from the book in `www/book/src` of a Vx checkout:

```sh
git -C ~/go/Vx worktree add --detach /tmp/vx-v0.0.4 v0.0.4
python3 scripts/make_vx_guides.py /tmp/vx-v0.0.4
```

Use the book of the Vx release that compiles the programs, so that a guide shows no feature that
the compiler does not have yet. A chapter that the book of that release does not have is left out.

The script also writes the table above. The two parts of this repository are in `parts/`:
`header.md` starts each guide, and `apple-gpu.md` ends the GPU guide.
