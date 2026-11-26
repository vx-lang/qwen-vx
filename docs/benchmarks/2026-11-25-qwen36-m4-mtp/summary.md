flash attention: on 26.31 ms, off 27.88 ms (medians at 2,048 tokens); chosen: on
draft lengths given: DRAFT=11 MTP_N=3
flash attention: given by FA; chosen: on

## Table A: generated tokens after a prompt

ms for each generated token: the median of the runs, and the fastest and slowest run. The prompt time is that of the first prompt of a process.

| Prompt tokens | qwen-vx ms | llama.cpp ms | qwen-vx tok/s | llama.cpp tok/s | llama.cpp ms / qwen-vx ms | qwen-vx prompt ms | llama.cpp prompt ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 22.05 (22.03-22.08) | 25.82 (25.67-25.90) | 45.4 | 38.7 | 1.171 | 262 (262-263) | 252 (250-252) |
| 512 | 22.20 (22.16-22.20) | 26.03 (25.82-26.49) | 45.1 | 38.4 | 1.173 | 1122 (1121-1122) | 1219 (1216-1219) |
| 2,048 | 22.62 (22.62-22.63) | 26.26 (26.26-26.28) | 44.2 | 38.1 | 1.161 | 4109 (4109-4111) | 5066 (5062-5068) |
| 8,192 | 24.16 (24.11-24.16) | 28.02 (27.88-28.11) | 41.4 | 35.7 | 1.160 | 18947 (18890-18990) | 22709 (22632-22766) |
| 16,384 | 26.60 (26.59-26.76) | 32.27 (32.26-32.31) | 37.6 | 31.0 | 1.213 | 46196 (45968-46380) | 54128 (53659-54465) |
| 32,768 | 50.49 (48.66-51.41) | 58.90 (58.00-65.10) | 19.8 | 17.0 | 1.167 | 147674 (141011-150820) | 161094 (149761-163903) |

## Table P: the prompt

Tokens of the prompt each second: the median of the runs, and the slowest and fastest run. qwen-vx ran the prompt several times in one process. The first run includes the first launch of each kernel. llama-completion ran the prompt once, after its warm-up.

| Prompt tokens | qwen-vx first run | qwen-vx later runs | llama-completion |
| ---: | ---: | ---: | ---: |
| 512 | 456 (452-456) | 451 (448-453) | 419 (418-420) |
| 2,048 | 498 (497-499) | 490 (487-493) | 404 (403-404) |
| 8,192 | 431 (431-433) | 399 (394-406) | 356 (355-361) |

## Table C: the 12 held-out prompts

ms for each generated token: the geometric mean over the prompts of each kind, and the fastest and slowest run.

| Kind | Prompt tokens | qwen-vx | qwen-vx --draft | qwen-vx --mtp | qwen-vx --mtp --draft | llama-completion | llama.cpp ngram-mod | llama.cpp draft-mtp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| edit | 1286, 1605, 1668 | 22.40 (22.31-22.48) | 11.00 (9.05-13.91) | 12.94 (12.24-13.54) | 10.62 (9.47-12.03) | 26.36 (26.15-26.53) | 11.48 (6.91-20.37) | 15.62 (15.08-16.21) |
| complete | 1044, 1086, 1288 | 22.33 (22.23-22.40) | 20.28 (18.36-21.42) | 15.46 (13.97-16.78) | 15.37 (13.78-16.83) | 26.14 (26.03-26.35) | 27.66 (26.29-30.30) | 17.35 (16.42-18.10) |
| write | 35, 44, 52 | 22.12 (22.03-22.28) | 20.69 (19.50-21.82) | 14.11 (13.54-15.21) | 14.15 (13.58-15.21) | 25.69 (25.64-25.84) | 26.23 (25.81-26.79) | 16.05 (14.88-17.01) |
| explain | 1257, 1330, 1503 | 22.38 (22.29-22.58) | 21.55 (20.40-22.57) | 16.72 (15.71-17.21) | 16.83 (16.02-17.58) | 26.24 (26.10-26.55) | 26.80 (26.35-27.68) | 19.68 (18.11-20.39) |
| all 12 |  | 22.31 (22.03-22.58) | 17.76 (9.05-22.57) | 14.74 (12.24-17.21) | 14.04 (9.47-17.58) | 26.10 (25.64-26.55) | 21.74 (6.91-30.30) | 17.10 (14.88-20.39) |

- qwen-vx against llama-completion: 1.170, faster on 12 of 12 prompts (1.16 to 1.18)
- qwen-vx --draft against qwen-vx: 1.256, faster on 11 of 12 prompts (0.99 to 2.46)
- qwen-vx --draft against llama-completion: 1.470, faster on 12 of 12 prompts (1.16 to 2.91)
- qwen-vx --draft against llama.cpp ngram-mod: 1.224, faster on 11 of 12 prompts (0.77 to 1.64)
- qwen-vx --draft against llama.cpp draft-mtp: 0.963, faster on 3 of 12 prompts (0.77 to 1.68)
- qwen-vx --mtp against qwen-vx: 1.514, faster on 12 of 12 prompts (1.30 to 1.82)
- qwen-vx --mtp against qwen-vx --draft: 1.205, faster on 10 of 12 prompts (0.74 to 1.59)
- qwen-vx --mtp against llama-completion: 1.771, faster on 12 of 12 prompts (1.53 to 2.16)
- qwen-vx --mtp against llama.cpp ngram-mod: 1.475, faster on 10 of 12 prompts (0.57 to 2.15)
- qwen-vx --mtp against llama.cpp draft-mtp: 1.161, faster on 12 of 12 prompts (1.06 to 1.25)
- qwen-vx --mtp --draft against qwen-vx: 1.589, faster on 12 of 12 prompts (1.29 to 2.35)
- qwen-vx --mtp --draft against qwen-vx --draft: 1.265, faster on 11 of 12 prompts (0.96 to 1.58)
- qwen-vx --mtp --draft against qwen-vx --mtp: 1.050, faster on 6 of 12 prompts (0.99 to 1.29)
- qwen-vx --mtp --draft against llama-completion: 1.859, faster on 12 of 12 prompts (1.51 to 2.78)
- qwen-vx --mtp --draft against llama.cpp ngram-mod: 1.548, faster on 11 of 12 prompts (0.74 to 2.19)
- qwen-vx --mtp --draft against llama.cpp draft-mtp: 1.218, faster on 12 of 12 prompts (1.06 to 1.61)
- llama.cpp ngram-mod against llama-completion: 1.201, faster on 3 of 12 prompts (0.87 to 3.78)
- llama.cpp draft-mtp against llama-completion: 1.526, faster on 12 of 12 prompts (1.29 to 1.73)

### Drafts, table C, round 1

| Kind | qwen-vx --draft accepted / drafted | qwen-vx --mtp accepted / drafted | qwen-vx --mtp --draft accepted / drafted | llama.cpp ngram-mod accepted / drafted | llama.cpp draft-mtp accepted / drafted |
| --- | ---: | ---: | ---: | ---: | ---: |
| edit | 1221 / 1303 (94%), 5.93 per check | 1166 / 1275 (91%), 3.21 per check | 1326 / 1418 (94%), 6.50 per check | 1199 / 1199 (100%) | 1149 / 1193 (96%) |
| complete | 357 / 599 (60%), 0.91 per check | 934 / 1156 (81%), 1.57 per check | 953 / 1211 (79%), 1.65 per check | 41 / 41 (100%) | 1096 / 1348 (81%) |
| write | 247 / 405 (61%), 0.79 per check | 1027 / 1222 (84%), 2.04 per check | 1033 / 1248 (83%), 2.08 per check | 15 / 15 (100%) | 1118 / 1281 (87%) |
| explain | 179 / 324 (55%), 0.72 per check | 836 / 1091 (77%), 1.21 per check | 834 / 1109 (75%), 1.20 per check | 19 / 19 (100%) | 1031 / 1538 (67%) |
| all 12 | 2004 / 2631 (76%), 1.73 per check | 3963 / 4744 (84%), 1.84 per check | 4146 / 4986 (83%), 2.10 per check | 1274 / 1274 (100%) | 4394 / 5360 (82%) |

- qwen-vx --draft: the text of qwen-vx in 36 of 36 runs
- qwen-vx --mtp: the text of qwen-vx in 36 of 36 runs
- qwen-vx --mtp --draft: the text of qwen-vx in 36 of 36 runs
- llama.cpp ngram-mod: the text of llama-completion in 24 of 36 runs. Other text, in the summary (the number of equal characters at the start): logs/C.complete3.r1.ngram.txt (1395), logs/C.complete3.r2.ngram.txt (1395), logs/C.complete3.r3.ngram.txt (1395), logs/C.explain3.r1.ngram.txt (457), logs/C.explain3.r2.ngram.txt (457), logs/C.explain3.r3.ngram.txt (457), logs/C.write1.r1.ngram.txt (773), logs/C.write1.r2.ngram.txt (773), logs/C.write1.r3.ngram.txt (773), logs/C.write2.r1.ngram.txt (1600), logs/C.write2.r2.ngram.txt (1600), logs/C.write2.r3.ngram.txt (1600)
- llama.cpp draft-mtp: the text of llama-completion in 21 of 36 runs. Other text, in the summary (the number of equal characters at the start): logs/C.complete1.r1.mtp3.txt (1055), logs/C.complete1.r2.mtp3.txt (1055), logs/C.complete1.r3.mtp3.txt (1055), logs/C.complete3.r1.mtp3.txt (1395), logs/C.complete3.r2.mtp3.txt (1395), logs/C.complete3.r3.mtp3.txt (1395), logs/C.explain3.r1.mtp3.txt (457), logs/C.explain3.r2.mtp3.txt (457), logs/C.explain3.r3.mtp3.txt (457), logs/C.write1.r1.mtp3.txt (773), logs/C.write1.r2.mtp3.txt (773), logs/C.write1.r3.mtp3.txt (773), logs/C.write2.r1.mtp3.txt (1581), logs/C.write2.r2.mtp3.txt (1581), logs/C.write2.r3.mtp3.txt (1581)

### qwen-vx and llama-completion: where the text starts to differ

The characters at the start of the generated text that are the same, round 1. The weight files differ in the token embedding and the output matrix, so the greedy text of the two programs can differ.

| Prompt | Same characters | Characters of llama-completion |
| --- | ---: | ---: |
| complete1 | 104 | 2085 |
| complete2 | 0 | 1938 |
| complete3 | 12 | 1427 |
| edit1 | 1768 | 1768 |
| edit2 | 2049 | 2049 |
| edit3 | 1807 | 1909 |
| explain1 | 87 | 1943 |
| explain2 | 5 | 2000 |
| explain3 | 281 | 2121 |
| write1 | 1764 | 1764 |
| write2 | 1077 | 1732 |
| write3 | 419 | 1924 |

## llama-bench-pp

| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |           pp512 |        424.84 ± 3.43 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |          pp2048 |        403.30 ± 3.33 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |          pp8192 |       335.23 ± 12.78 |

## llama-bench-tg

| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |           tg128 |         39.05 ± 0.21 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |   tg128 @ d2048 |         38.08 ± 0.29 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |   tg128 @ d8192 |         35.26 ± 0.51 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |  tg128 @ d16384 |         25.98 ± 0.36 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |  tg128 @ d32768 |         18.86 ± 0.11 |

