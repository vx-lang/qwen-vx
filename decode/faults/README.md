# Page faults, memory and CPU time of a generated token

Tools that measure the page faults, the memory and the CPU time of the Qwen app and of llama.cpp,
and their results for Qwen3-Coder-30B-A3B in Q4_0 on an M4 with 32 GB.

The question was why llama.cpp generated a token 1 ms faster at 2,000 tokens of context, and 1 to
2 ms faster at 8,000, but not at short context. The answer:

- **Page faults and the kind of memory do not explain the gap.** Neither program makes page
  faults while it generates, and the GPU reads all kinds of memory at the same speed.
- **The CPU time does not explain it.** The app uses about 3 times the CPU time of llama.cpp for
  a token, but this time does not grow with the context.
- **A Metal residency set closes the gap.** A change to the Vx runtime puts all the buffers from
  `vx_metal_alloc` in one residency set. Then Metal does not make the buffers resident again for
  each command buffer. This alone saves 1.3 to 1.6 ms a token at 2,000 tokens of context and 1.6 to
  1.9 ms at 8,000. llama.cpp also uses a residency set.

The analysis also found that the load copied the weights into private memory, which cost about
30 s and 18 GB. The app now reads the weights in place from the mapped file (see the end).

## The tools

- `sampler.c` is a library for `DYLD_INSERT_LIBRARIES`. Every 200 ms, it writes these values of
  its process to `$FAULTS_DIR/<pid>.tsv`: minor and major page faults (`getrusage`), the faults,
  pageins and copy-on-write faults of the task (`TASK_EVENTS_INFO`), user and sys time, and the
  physical footprint and resident size (`TASK_VM_INFO`).
- `run.sh OUT COMMAND...` builds `sampler.dylib` and runs a command with it. Each process that
  the command starts writes its own file in `OUT`. `vxc` runs the program as a child process. The
  program's file is the one with a footprint of more than 17 GB. The other files are compiler
  processes.
- `delta.py FILE START END [MS_PER_TOKEN]` prints the change of each value between two times. A
  negative time counts back from the end. With the time of a token, it also prints the change
  for each token. Use it on the generation window: the samples after the prefill, where the user
  time grows at a steady rate, and before the last sample.
- `memory_kinds.m` measures how fast a Metal kernel reads three kinds of memory: a Metal buffer
  (`newBufferWithLength`), a NoCopy buffer over anonymous `mmap` (the memory of
  `vx_metal_alloc`), and a NoCopy buffer over a file `mmap` (the memory of llama.cpp). It reads
  4 × 1 GiB of each kind in a stream, and in chunks of 16 KB to 1 MB at scattered offsets.

`DYLD_INSERT_LIBRARIES` has no effect on programs that macOS protects, such as those in
`/usr/bin`.

## Run them

```sh
decode/faults/run.sh out/vx vxc src/main.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen3-30b-q4-q6.vxw \
    --tokens "$(cat p2000.ids)" --steps 64
python3 decode/faults/delta.py out/vx/<pid>.tsv -1.4 -0.2 25.7

decode/faults/run.sh out/llama llama-bench -m Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf \
    -p 0 -n 64 -d 2000 -r 1

clang -O2 -fobjc-arc -framework Foundation -framework Metal decode/faults/memory_kinds.m \
    -o memory_kinds
./memory_kinds Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf
```

`p2000.ids` is a list of 2,000 token ids with commas between them. The app reads the prompt from
it, and then generates 64 tokens. `memory_kinds` maps the first 4 GiB of the file that it gets,
so any file of 4 GiB or more will do.

Run them with no other work on the machine. A browser, or a job that uses the CPU, changes the
times.

## Results

On 2026-11-17, with Qwen-vx `q6-lm-head` (#51) and the 6-bit `lm_head` file. The Vx compilers
are debug builds of `vxc`: 34130be6, and 4eb30c3e with the residency set. llama.cpp is build
620d2225c.

### Page faults, memory and CPU time

2,000 tokens of context, 64 generated tokens, Vx 34130be6:

| | Qwen-vx | llama.cpp |
| --- | --- | --- |
| Time of a generated token | 25.71 ms | 24.69 ms |
| Page faults while it generates | about 690 at the start, then 1 in each 200 ms | about 25 in all |
| Pageins while it generates | 0 | 0 |
| CPU time of a generated token | 2.36 ms user + 1.25 ms sys | 1.03 ms user + 0.22 ms sys |
| Time to the end of the load | about 38 s | about 7 s |
| Page faults in the load | 1.20M minor, 1.08M pageins | 1.1M |
| Physical footprint after the load | 19.6 GB | 0.6 GB |
| Resident size after the load | 19.6 GB | 17.5 GB |

The app makes about 0 page faults in the prefill, too.

**The kind of memory does not change the speed of the GPU.** `memory_kinds` reads all three kinds
at 104 GB/s in a stream. In scattered chunks, all three read at 95 GB/s with 16 KB chunks, and at
103 to 106 GB/s with 64 KB to 1 MB chunks.

**The CPU time does not grow with the context.** CPU time of a generated token, with 200 steps,
Vx 4eb30c3e:

| Context | Residency set | User | Sys |
| --- | --- | --- | --- |
| 0 | on | 2.20 ms | 1.18 ms |
| 0 | off | 2.17 ms | 1.16 ms |
| 2,000 | on | 2.16 ms | 1.21 ms |

One more run at context 0 gave 6.3 ms user and 2.1 ms sys, and a slower token (22.71 ms against
22.0 ms). We count it as an outlier.

### The residency set

One `vxc` at 4eb30c3e with the residency set, which `VX_AB_NO_RESIDENCY=1` turns off. ms for each
generated token, 64 steps, two rounds:

| Context | Set on | Set off | 34130be6 | llama.cpp |
| --- | --- | --- | --- | --- |
| ~0 | 22.07, 22.19 | 21.98, 21.99 | 21.93, 22.11 | 22.78, 22.01 |
| 2,000 | 24.57, 24.51 | 26.14, 25.78 | 26.11, 26.06 | 24.77, 24.70 |
| 8,000 | 31.97, 32.06 | 33.87, 33.67 | 33.84, 33.96 | 32.06, 33.66 |

The 34130be6 and llama.cpp columns are from a separate run of the same commands. All the runs at a
context generate the same 64 tokens.

- With the set off, the times are those of 34130be6. So the other changes between the two
  compilers do not change the time.
- The set saves nothing at short context. Its gain grows with the context. Why is not known yet.

### The load copied the weights

`load_weights` mapped the 17.8 GB file. Then it copied each tensor into a host tensor, and
`transfer(.., Memory::HBM)` copied the host tensor into memory from `vx_metal_alloc`. The 1.08M
pageins are the file, read in 16 KB pages. At the end the app had a private copy of the weights:
its footprint was 19.6 GB. llama.cpp gives Metal the mapped file itself, with
`newBufferWithBytesNoCopy` over the `mmap` and a residency set ("use residency sets = true" in its
log). Its footprint stays at 0.6 GB, and the 17.5 GB of the file is resident as pages of the file.

A second runtime change makes `transfer(&view, Memory::HBM)` of a view of a read-only file mapping
map the file again (`MAP_PRIVATE`) instead of making a copy. With it, the app makes views of the
mapped file in place of host tensors. The load then takes about 10 s when the file is in the
file cache, and the peak private memory is 1.3 GB. A token is a little slower: 24.73 and 24.66 ms
at 2,000 tokens of context, and 32.34 and 32.41 ms at 8,000. Without the residency set, the mapped
weights are much slower: 30.58 and 30.10 ms at 2,000 tokens of context.

Both runtime changes are in Vx since vx-lang/Vx#3606 (c4e57fd7), and the app now makes views of the
mapped file. With the vxc of c4e57fd7, 64 generated tokens after a prompt, two rounds:

| Context | Load, copied | Load, mapped | Private memory, copied | Private memory, mapped | Token, copied | Token, mapped |
| --- | --- | --- | --- | --- | --- | --- |
| ~0 | 37.5, 32.2 s | 10.3, 10.3 s | 18.9 GB | 0.26 GB | 22.15, 22.20 ms | 22.19, 22.30 ms |
| 2,000 | 31.3, 29.0 s | 10.4, 10.4 s | 19.6 GB | 1.27 GB | 24.66, 24.58 ms | 24.70, 24.66 ms |
| 8,000 | 31.7, 31.0 s | 11.2, 11.2 s | 20.4 GB | 2.10 GB | 32.61, 31.89 ms | 32.00, 32.27 ms |

The load is the time of the program less the time of its prompt and of its generated tokens.
Private memory is the peak physical footprint. With mapped weights it holds the KV cache and the
scratch, and the weights are pages of the file. A warm load, right after another mapped run, also
took 10.6 to 10.9 s. llama.cpp gave 22.07 and 22.15 ms, 24.71 and 24.70 ms, and 32.04 and
32.14 ms. All the runs at a context generate the same 64 tokens.

The first measurement of the mapped app, with the vxc of the branch of #3606, was 0.1 to 0.5 ms
slower than copied weights. This measurement shows no such cost.
