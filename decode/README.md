# The time of a generated token

Tools that measure where the time of a generated token goes, and their results for
Qwen3-Coder-30B-A3B in Q4_0 on an M4 with 32 GB.

- `profile_decode.vx` runs 32 tokens with all the parts of a layer, and then 32 tokens with each
  part left out. The difference is the time of the part. A run with a part left out gives wrong
  tokens, but its time is right.
- `bandwidth.m` measures how fast a Metal kernel can read memory. This is the limit of each kernel
  that reads weights.
- `faults/` measures the page faults, the memory and the CPU time of a run. Its README has the
  results, and how a Metal residency set closed the gap to llama.cpp at long context.

## Run them

```sh
vxc decode/profile_decode.vx --package qwen=src --host default \
    --machine ~/.vx/current/fleet/m4-uma.vx -- qwen3-30b-q4.vxw

clang -O2 -fobjc-arc -framework Foundation -framework Metal decode/bandwidth.m -o bandwidth
./bandwidth
```

The profile needs the Q4_0 file of the 30B MoE model. Run it with no other work on the GPU, and
with enough free memory for the 17.8 GB of weights: when the Mac swaps, all the times go up.

## Results

Vx v0.0.4, `main` at c65a820, at a context of 100 to 400 tokens, on 2026-11-15.

**The limit of memory is 104 GB/s.** `bandwidth.m` reads at 104 GB/s with 10K to 655K threads.
The 120 GB/s of the M4 spec is not reachable.

**A token takes 23.4 ms.** The parts:

| Part | Format | Bytes a token | Time | Speed |
| --- | --- | --- | --- | --- |
| Experts, 8 of 128 in each layer | u4, f16 scale for each 32 | 1,019 MB | 10.7 ms | 96 GB/s |
| QKV | u4, f16 scales | 283 MB | 2.7 ms | 104 GB/s |
| lm_head | fp8, f32 scales | 311 MB | 2.5 ms | about 104 GB/s |
| Attention output | u4, f16 scales | 227 MB | 2.3 ms | 100 GB/s |
| Attention, 100 to 400 tokens of KV cache | f32 | about 47 MB | 1.5 ms | about 31 GB/s |
| Router and top-k | bf16 | 25 MB | 0.5 ms | 54 GB/s |
| Norms, RoPE and the store into the KV cache | | small | about 0 ms | |
| Embedding, final norm, argmax and the loop | | small | 0.3 ms | |

The error of each part is about ±0.5 ms, so the parts do not sum to 23.4 ms.

- A token reads about 1.9 GB. At 104 GB/s that is 18.4 ms, the floor of a token with the bytes of
  today.
- The kernels that read the weights run at 96 to 104 GB/s, so a faster kernel can gain little.
  A token gets faster when it reads fewer bytes (#33), or when one read of the weights makes more
  than one token, as speculative decoding does (#18).
- The other 5 ms is in small kernels, and in the time between kernels (#34). A token runs about
  530 kernels, and each one waits for the one before it. 530 launches of a kernel that does almost
  nothing take 1.1 ms, so the cost of a launch on the CPU is small.

## The raw output

```
warm-up: 49.79969561100006 ms a token
all parts: 23.394832387566566 ms a token
without part 1: 23.962687700986862 ms a token, so the part takes -0.5678553134202957 ms
without part 2: 20.674191415309906 ms a token, so the part takes 2.7206409722566605 ms
without part 4: 23.620255291461945 ms a token, so the part takes -0.2254229038953781 ms
without part 8: 21.86981402337551 ms a token, so the part takes 1.5250183641910553 ms
without part 16: 21.132776513695717 ms a token, so the part takes 2.2620558738708496 ms
without part 32: 22.923583164811134 ms a token, so the part takes 0.47124922275543213 ms
without part 64: 12.729949317872524 ms a token, so the part takes 10.664883069694042 ms
without part 128: 20.943213254213333 ms a token, so the part takes 2.4516191333532333 ms
without all parts (embed, final norm, argmax, the loop): 0.2582122106105089 ms a token
530 launches of a kernel that does almost nothing: 1.1274160351604223 ms (warm-up 7.74716679006815 ms for 100)
```

This output is from before the parts had names: 1 is the norms, 2 QKV, 4 RoPE and the store, 8
attention, 16 the attention output, 32 the router and top-k, 64 the experts, and 128 lm_head.
