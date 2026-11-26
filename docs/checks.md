# The checks of the program

How the Vx program and the NumPy reference were checked.

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
