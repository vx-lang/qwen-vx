#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints the kernels of `round_attention` in src/round.vx: the attention of n positions in a row, at
start to start + n - 1, as Vx code. A kernel for n queries keeps the values of each query in variables
of their own, so that they stay in registers. Vx v0.0.5 keeps a tile in threadgroup memory on Metal,
so a kernel with a tile for the n queries would be slower.

Each query gets the result of `vec_attention` (src/attention.vx) bit for bit. A block does the same
work for each of its n queries that the block of `vec_attention_parts` does for one, in the same
order, but it reads each row of K and V once for all of them. A query skips the blocks of 32
positions after its own position, and a row after its position gets the score -3e38 and so the
weight 0, as in `vec_attention_parts`. The rows after the last query are not read. The kernels call
the small `mlir!` functions of src/attention.vx.

    python3 scripts/round_attention.py

prints the functions; replace them in src/round.vx, and run vx-format on it.
"""
MAX_N = 8
L = []
e = L.append


def parts(n):
    """The kernel of the parts for n queries."""
    T = range(n)
    e('')
    e(f'/// `vec_attention_parts` for {n} queries in a row: query t is at position pos + t, in q[t0 + t], and its part')
    e('/// w goes to part[h][w][132 (t0 + t) ..].')
    e(f'fn round_attention_parts_{n}(part : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, q : &Tensor<f32, [?, ?, ?], Memory::HBM>, k_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, v_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, pos : i32, t0 : i32, q_heads : i32, group : i32, nwg : i32, scale : f32) -> void {{')
    e('  spawn on(Topology::Device) {')
    e('    for h in 0..q_heads across blocks {')
    e('      for w in 0..nwg across blocks {')
    e('        let mut stats = unsafe {')
    e(f'          Tensor<f32, [1, {n}, 2], Memory::SMEM>::uninit()')
    e('        };')
    e('        for s in 0..1 across subgroups {')
    e('          for l in 0..32 across lanes(32) {')
    b = lambda s: e('            ' + s)
    b('let g = h / group;')
    b('let d = 4 * l;')
    b(f'let last = pos + {n - 1};')
    for t in T:
        for j in range(4):
            b(f'let q{t}_{j} = q[t0 + {t}][h][d + {j}] * scale;')
        b(f'let mut big{t} : f32 = -3.0e38;')
        b(f'let mut total{t} : f32 = 0.0;')
        for j in range(4):
            b(f'let mut o{t}_{j} : f32 = 0.0;')
    b('let chunks = (last + 32) / 32;')
    b('let mut c = w;')
    b('while c < chunks {')
    c = lambda s: e('              ' + s)
    d = lambda s: e('                ' + s)
    c('let r0 = c * 32;')
    # the scores of rows r0 to r0 + 31 for each query; lane r keeps the scores of row r0 + r. A row past
    # the last position reads the row of the last position.
    for t in T:
        c(f'let mut mine{t} : f32 = -3.0e38;')
    c('for r in 0..32 {')
    d('let row = r0 + r;')
    d('let mut kept = row;')
    d('if row > last {')
    d('  kept = last;')
    d('}')
    for j in range(4):
        d(f'let k{j} = half_at(k_cache, kept, g, d, {j});')
    for t in T:
        d(f'let mut score{t} = lane_sum(k0 * q{t}_0 + k1 * q{t}_1 + k2 * q{t}_2 + k3 * q{t}_3);')
        d(f'if row > pos + {t} {{')
        d(f'  score{t} = -3.0e38;')
        d('}')
    d('if r == l {')
    for t in T:
        d(f'  mine{t} = score{t};')
    d('}')
    c('}')
    # the online softmax of each query that this block of positions reaches; p is 0 for the others, so
    # that the sum of V leaves them as they are
    for t in T:
        c(f'let mut p{t} : f32 = 0.0;')
        c(f'if c < (pos + {t} + 32) / 32 {{')
        d(f'let top = big{t}.max(lane_max(mine{t}));')
        d(f'let shrink = (big{t} - top).fast_exp();')
        d(f'p{t} = (mine{t} - top).fast_exp();')
        d(f'total{t} = total{t} * shrink + lane_sum(p{t});')
        for j in range(4):
            d(f'o{t}_{j} = o{t}_{j} * shrink;')
        d(f'big{t} = top;')
        c('}')
    # the sum of V: the weight of row r0 + r for query t is in lane r
    c('for r in 0..32 {')
    d('let row = r0 + r;')
    d('let mut kept = row;')
    d('if row > last {')
    d('  kept = last;')
    d('}')
    for t in T:
        d(f'let pr{t} = lane_value(p{t}, r);')
        for j in range(4):
            d(f'o{t}_{j} = fused(half_at(v_cache, kept, g, d, {j}), pr{t}, o{t}_{j});')
    c('}')
    c('c = c + nwg;')
    b('}')
    for t in T:
        b(f'let b{t} = 132 * (t0 + {t});')
        for j in range(4):
            b(f'part[h][w][b{t} + d + {j}] = o{t}_{j};')
    b('// Every lane has the same largest score and sum of the weights of a query. Lanes 0 and 1 keep them')
    b('// for the next loop.')
    b('if l < 2 {')
    for t in T:
        b(f'  let mut x{t} = big{t};')
        b('  if l == 1 {')
        b(f'    x{t} = total{t};')
        b('  }')
        b(f'  stats[s][{t}][l] = x{t};')
    b('}')
    e('          }')
    e('          for l in 0..2 across lanes(32) {')
    for t in T:
        e(f'            part[h][w][132 * (t0 + {t}) + 128 + l] = stats[s][{t}][l];')
    e('          }')
    e('        }')
    e('      }')
    e('    }')
    e('  }')
    e('  return;')
    e('}')


def dispatch():
    """Calls the kernel of the parts for n queries."""
    e('')
    e(f'/// The parts of queries t0 to t0 + n - 1, with the kernel for n queries. n is 1 to {MAX_N}.')
    e('fn round_attention_parts(part : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, q : &Tensor<f32, [?, ?, ?], Memory::HBM>, k_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, v_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, pos : i32, t0 : i32, n : i32, q_heads : i32, group : i32, nwg : i32, scale : f32) -> void {')
    e(f'  assert(n >= 1 && n <= {MAX_N}, "round_attention_parts: n must be 1 to {MAX_N}");')
    for n in range(1, MAX_N + 1):
        word = 'if' if n == 1 else '} else if'
        if n == MAX_N:
            e('  } else {')
        else:
            e(f'  {word} n == {n} {{')
        e(f'    round_attention_parts_{n}(part, q, k_cache, v_cache, pos, t0, q_heads, group, nwg, scale);')
    e('  }')
    e('  return;')
    e('}')


dispatch()
for n in range(1, MAX_N + 1):
    parts(n)
print('\n'.join(L))
