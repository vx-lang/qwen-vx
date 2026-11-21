draft lengths tried: app 3 5 7 9 11 15, MTP 1 2 3 4 6
flash attention: on 26.37 ms, off 27.67 ms (medians at 2,048 tokens); chosen: on
flash attention: given by FA; chosen: on
draft lengths given: DRAFT=7 MTP_N=3

## Table A: generated tokens after a prompt

ms for each generated token: the median of the runs, and the fastest and slowest run. The prompt time is that of the first prompt of a process.

| Prompt tokens | qwen-vx ms | llama.cpp ms | qwen-vx tok/s | llama.cpp tok/s | llama.cpp ms / qwen-vx ms | qwen-vx prompt ms | llama.cpp prompt ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 22.17 (22.14-22.17) | 25.81 (25.63-26.19) | 45.1 | 38.7 | 1.164 | 351 (338-366) | 251 (251-252) |
| 512 | 22.33 (22.29-22.36) | 25.83 (25.81-25.85) | 44.8 | 38.7 | 1.157 | 1180 (1174-1203) | 1217 (1216-1217) |
| 2,048 | 22.82 (22.79-22.84) | 26.40 (26.36-26.53) | 43.8 | 37.9 | 1.157 | 4160 (4157-4168) | 5050 (5044-5056) |
| 8,192 | 24.32 (24.30-24.33) | 27.94 (27.86-27.98) | 41.1 | 35.8 | 1.149 | 18850 (18830-18928) | 22577 (22550-22622) |
| 16,384 | 26.55 (26.54-26.65) | 32.05 (32.02-32.16) | 37.7 | 31.2 | 1.207 | 45572 (45432-45955) | 53550 (53046-53562) |
| 32,768 | 43.55 (43.39-47.50) | 48.97 (48.38-55.01) | 23.0 | 20.4 | 1.124 | 129143 (128628-134961) | 138361 (138249-149488) |

## Table P: the prompt

Tokens of the prompt each second: the median of the runs, and the slowest and fastest run. qwen-vx ran the prompt several times in one process. The first run includes the first launch of each kernel. llama-completion ran the prompt once, after its warm-up.

| Prompt tokens | qwen-vx first run | qwen-vx later runs | llama-completion |
| ---: | ---: | ---: | ---: |
| 512 | 433 (433-436) | 453 (450-454) | 420 (419-421) |
| 2,048 | 492 (491-493) | 492 (490-494) | 404 (396-406) |
| 8,192 | 433 (429-433) | 400 (394-411) | 358 (355-361) |

## Table T: draft lengths, on the 12 tuning prompts

ms for each generated token: the geometric mean over the 12 prompts, one run each.

| Program | Draft length | ms | accepted / drafted |
| --- | ---: | ---: | ---: |
| llama-completion |  | 26.16 |  |
| llama.cpp draft-mtp | 1 | 21.62 | 2937 / 3226 (91%) |
| llama.cpp draft-mtp | 2 | 19.11 | 3857 / 4633 (83%) |
| llama.cpp draft-mtp | 3 | 18.15 | 4299 / 5664 (76%) |
| llama.cpp draft-mtp | 4 | 18.21 | 4534 / 6632 (68%) |
| llama.cpp draft-mtp | 6 | 19.22 | 4780 / 8549 (56%) |
| llama.cpp ngram-cache |  | 24.38 | 2794 / 2794 (100%) |
| llama.cpp ngram-mod |  | 19.76 | 1626 / 1626 (100%) |
| qwen-vx |  | 22.59 |  |
| qwen-vx --draft | 3 | 19.39 | 1372 / 1454 (94%) |
| qwen-vx --draft | 5 | 19.04 | 1522 / 1618 (94%) |
| qwen-vx --draft | 7 | 18.90 | 1592 / 1700 (94%) |
| qwen-vx --draft | 9 | 18.92 | 1592 / 1700 (94%) |
| qwen-vx --draft | 11 | 18.91 | 1592 / 1700 (94%) |
| qwen-vx --draft | 15 | 18.96 | 1592 / 1700 (94%) |

- qwen-vx --draft: the text of qwen-vx in 72 of 72 runs
- llama.cpp ngram-cache: the text of llama-completion in 8 of 12 runs. Other text, in the summary (the number of equal characters at the start): logs/T.complete1.r1.cache.txt (231), logs/T.edit1.r1.cache.txt (877), logs/T.explain3.r1.cache.txt (1492), logs/T.write1.r1.cache.txt (803)
- llama.cpp ngram-mod: the text of llama-completion in 8 of 12 runs. Other text, in the summary (the number of equal characters at the start): logs/T.complete1.r1.ngram.txt (231), logs/T.edit1.r1.ngram.txt (877), logs/T.explain3.r1.ngram.txt (1492), logs/T.write1.r1.ngram.txt (803)
- llama.cpp draft-mtp: the text of llama-completion in 42 of 60 runs. Other text, in the summary (the number of equal characters at the start): logs/T.complete1.r1.mtp1.txt (231), logs/T.complete1.r1.mtp2.txt (231), logs/T.complete1.r1.mtp3.txt (231), logs/T.complete1.r1.mtp4.txt (231), logs/T.complete1.r1.mtp6.txt (231), logs/T.complete2.r1.mtp1.txt (1213), logs/T.edit1.r1.mtp1.txt (877), logs/T.edit1.r1.mtp2.txt (877), logs/T.edit1.r1.mtp3.txt (877), logs/T.edit1.r1.mtp4.txt (877), logs/T.edit1.r1.mtp6.txt (877), logs/T.explain3.r1.mtp2.txt (1492), logs/T.explain3.r1.mtp3.txt (1492), logs/T.write1.r1.mtp1.txt (803), logs/T.write1.r1.mtp2.txt (803), logs/T.write1.r1.mtp3.txt (803), logs/T.write1.r1.mtp4.txt (803), logs/T.write1.r1.mtp6.txt (803)

## Table C: the 12 held-out prompts

ms for each generated token: the geometric mean over the prompts of each kind, and the fastest and slowest run.

| Kind | Prompt tokens | qwen-vx | qwen-vx --draft | llama-completion | llama.cpp ngram-cache | llama.cpp ngram-mod | llama.cpp draft-mtp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edit | 1286, 1605, 1668 | 22.60 (22.49-22.71) | 13.19 (11.45-15.91) | 26.30 (26.11-26.68) | 14.13 (11.98-18.33) | 11.50 (6.88-20.41) | 15.64 (15.15-16.15) |
| complete | 1044, 1086, 1288 | 22.49 (22.41-22.60) | 21.89 (20.27-22.81) | 26.22 (26.10-26.50) | 27.31 (20.96-31.67) | 27.63 (26.33-30.30) | 17.30 (16.22-18.15) |
| write | 35, 44, 52 | 22.26 (22.21-22.35) | 21.79 (20.74-22.44) | 25.79 (25.66-25.94) | 28.91 (26.37-32.53) | 26.35 (25.98-27.36) | 15.92 (14.90-16.48) |
| explain | 1257, 1330, 1503 | 22.56 (22.51-22.69) | 22.31 (21.48-22.86) | 26.22 (26.15-26.29) | 30.77 (27.97-32.53) | 26.88 (26.44-27.73) | 19.74 (18.14-20.77) |
| all 12 |  | 22.48 (22.21-22.71) | 19.36 (11.45-22.86) | 26.13 (25.66-26.68) | 24.20 (11.98-32.53) | 21.78 (6.88-30.30) | 17.08 (14.90-20.77) |

- qwen-vx against llama-completion: 1.162, faster on 12 of 12 prompts (1.16 to 1.17)
- qwen-vx --draft against qwen-vx: 1.161, faster on 7 of 12 prompts (0.99 to 1.96)
- qwen-vx --draft against llama-completion: 1.350, faster on 12 of 12 prompts (1.15 to 2.29)
- qwen-vx --draft against llama.cpp ngram-cache: 1.250, faster on 11 of 12 prompts (0.98 to 1.45)
- qwen-vx --draft against llama.cpp ngram-mod: 1.125, faster on 10 of 12 prompts (0.61 to 1.49)
- qwen-vx --draft against llama.cpp draft-mtp: 0.882, faster on 3 of 12 prompts (0.73 to 1.35)
- llama.cpp ngram-cache against llama-completion: 1.080, faster on 4 of 12 prompts (0.79 to 2.12)
- llama.cpp ngram-mod against llama-completion: 1.200, faster on 3 of 12 prompts (0.87 to 3.77)
- llama.cpp draft-mtp against llama-completion: 1.530, faster on 12 of 12 prompts (1.28 to 1.70)

### Drafts, table C, round 1

| Kind | qwen-vx --draft accepted / drafted | llama.cpp ngram-cache accepted / drafted | llama.cpp ngram-mod accepted / drafted | llama.cpp draft-mtp accepted / drafted |
| --- | ---: | ---: | ---: | ---: |
| edit | 1136 / 1172 (97%), 6.49 per check | 1261 / 1261 (100%) | 1199 / 1199 (100%) | 1149 / 1193 (96%) |
| complete | 177 / 292 (61%), 2.95 per check | 737 / 737 (100%) | 41 / 41 (100%) | 1096 / 1348 (81%) |
| write | 119 / 184 (65%), 2.53 per check | 654 / 654 (100%) | 15 / 15 (100%) | 1118 / 1281 (87%) |
| explain | 88 / 137 (64%), 2.51 per check | 502 / 502 (100%) | 19 / 19 (100%) | 1031 / 1538 (67%) |
| all 12 | 1520 / 1785 (85%), 4.79 per check | 3154 / 3154 (100%) | 1274 / 1274 (100%) | 4394 / 5360 (82%) |

- qwen-vx --draft: the text of qwen-vx in 36 of 36 runs
- llama.cpp ngram-cache: the text of llama-completion in 27 of 36 runs. Other text, in the summary (the number of equal characters at the start): logs/C.complete3.r1.cache.txt (1395), logs/C.complete3.r2.cache.txt (1395), logs/C.complete3.r3.cache.txt (1395), logs/C.explain3.r1.cache.txt (983), logs/C.explain3.r2.cache.txt (983), logs/C.explain3.r3.cache.txt (983), logs/C.write1.r1.cache.txt (773), logs/C.write1.r2.cache.txt (773), logs/C.write1.r3.cache.txt (773)
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
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |           pp512 |        426.29 ± 3.15 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |          pp2048 |        404.80 ± 3.25 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |          pp8192 |        343.12 ± 5.44 |

## llama-bench-tg

| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |           tg128 |         39.09 ± 0.17 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |   tg128 @ d2048 |         38.25 ± 0.09 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |   tg128 @ d8192 |         35.77 ± 0.38 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |  tg128 @ d16384 |         27.83 ± 0.59 |
| qwen35moe 35B.A3B Q4_0         |  18.35 GiB |    34.66 B | MTL,BLAS   |       4 |   1 |  tg128 @ d32768 |         19.68 ± 0.05 |

