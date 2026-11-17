#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints pf_mm_q4_mlir of src/prefill.vx: the Q4_0 product of a chunk as one threadgroup mlir! block.

The design is that of llama.cpp's kernel_mul_mm_id (ggml/src/ggml-metal/kernels/mul_mm.metal),
MIT License, Copyright (c) 2023-2026 The ggml authors. Edit this script, not the generated MLIR:

    python3 scripts/mlir_kernels/expert_product.py

prints the function; replace it in src/prefill.vx.
"""
import sys
mode = 'app'
hint = True
ACC = '!gpu.mma_matrix<8x8xf32, "COp">'
AOP = '!gpu.mma_matrix<8x8xf16, "AOp">'
BOP = '!gpu.mma_matrix<8x8xf16, "BOp">'
L = []
e = L.append
name = 'pf_mm_q4_mlir' if mode == 'app' else ('mm_mlir2h' if hint else 'mm_mlir2')
e(f'fn {name}(y : &mut Tensor<f32, [?, ?], Memory::HBM>, w : &Tensor<u32, [?], Memory::HBM>, scales : &Tensor<f16, [?], Memory::HBM>, x : &Tensor<f32, [?, ?], Memory::HBM>, list : &Tensor<i32, [?], Memory::HBM>, bases : &Tensor<i32, [?], Memory::HBM>, stride : i32, rows : i32, cols : i32, tiles : i32) -> void {{')
e('  spawn on(Topology::Device) {')
e('    for rb in 0..rows / 64 across blocks {')
e('      for tb in 0..(tiles + 1) / 2 across blocks {' if mode == 'app' else '      for tb in 0..tiles across blocks {')
e('        let mut sw = unsafe { Tensor<f16, [2048], Memory::SMEM>::uninit() };')
e('        let mut sx = unsafe { Tensor<f16, [1024], Memory::SMEM>::uninit() };')
e('        let _d = mlir!( threads : 128, inputs : (%sw = &mut sw : memref<2048xf16, 3>, %sx = &mut sx : memref<1024xf16, 3>, %y = y : memref<?x?xf32>, %w = w : memref<?xi32>, %sc = scales : memref<?xf16>, %x = x : memref<?x?xf32>, %list = list : memref<?xi32>, %bases = bases : memref<?xi32>, %stride = stride : i32, %cols = cols : i32, %rb = rb : i32, %tb = tb : i32), returns : i32, dialects : ['+('"arith", "memref", "scf", "gpu", "vector", "vx"' if hint else '"arith", "memref", "scf", "gpu", "vector"')+'], ensures : [writes_only(sw), writes_only(sx), reads_only(w), reads_only(scales), reads_only(x), reads_only(list), reads_only(bases), writes_only(y[tb * 32..tb * 32 + 32][rb * 64..rb * 64 + 64])]) {')
b = lambda s: e('          ' + s)
consts = {}
def c(n):
    consts.setdefault(n, f'%k{n}')
    return consts[n]
for n in [0, 1, 2, 4, 8, 16, 32, 64]:
    c(n)
b('%tid = gpu.thread_id x')
b('%sg = arith.divui %tid, %k32 : index')
b('%l = arith.remui %tid, %k32 : index')
b('%th = arith.divui %sg, %k2 : index')
b('%rh = arith.remui %sg, %k2 : index')
b('%rbi = arith.index_cast %rb : i32 to index')
b('%tbi = arith.index_cast %tb : i32 to index')
b('%stridei = arith.index_cast %stride : i32 to index')
b('%colsi = arith.index_cast %cols : i32 to index')
b('%G = arith.divui %colsi, %k32 : index')
b('%WPR = arith.divui %colsi, %k8 : index')
if mode == 'app':
    b('%tbb = arith.muli %tbi, %k2 : index')
else:
    b('%tbb = arith.addi %tbi, %k0 : index')
b('%base32 = memref.load %bases[%tbb] : memref<?xi32>')
b('%zi = arith.constant 0 : i32')
b('%bok = arith.cmpi sge, %base32, %zi : i32')
b('%basec = arith.maxsi %base32, %zi : i32')
b('%base = arith.index_cast %basec : i32 to index')
b('%Gl = arith.select %bok, %G, %k0 : index')
# weights: lane decodes 16 columns (half kh = l % 2) of row rl = sg * 16 + l / 2
b('%sg16 = arith.muli %sg, %k16 : index')
b('%l2 = arith.divui %l, %k2 : index')
b('%kh = arith.remui %l, %k2 : index')
b('%rl = arith.addi %sg16, %l2 : index')
b('%rb64 = arith.muli %rbi, %k64 : index')
b('%r = arith.addi %rb64, %rl : index')
b('%bs = arith.muli %base, %stridei : index')
b('%row = arith.addi %bs, %r : index')
b('%rw = arith.muli %row, %WPR : index')
b('%kh2 = arith.muli %kh, %k2 : index')
b('%wbase = arith.addi %rw, %kh2 : index')
b('%sbase = arith.muli %row, %G : index')
b(f'%kh1024 = arith.muli %kh, {c(1024)} : index')
b('%swd = arith.addi %kh1024, %rl : index')          # sw[(kh * 16 + k) * 64 + rl]
# inputs: lane copies 8 columns cl = (l % 4) * 8 of token tl = sg * 8 + l / 4
b('%sg8 = arith.muli %sg, %k8 : index')
b('%l4 = arith.divui %l, %k4 : index')
b('%lm4 = arith.remui %l, %k4 : index')
b('%tl = arith.addi %sg8, %l4 : index')
b('%cl = arith.muli %lm4, %k8 : index')
b('%tb32 = arith.muli %tbi, %k32 : index')
b('%li = arith.addi %tb32, %tl : index')
b('%from32 = memref.load %list[%li] : memref<?xi32>')
b('%live = arith.cmpi sge, %from32, %zi : i32')
b('%fromc = arith.maxsi %from32, %zi : i32')
b('%from = arith.index_cast %fromc : i32 to index')
b('%tl32 = arith.muli %tl, %k32 : index')
b('%sxd = arith.addi %tl32, %cl : index')
b('%sxd4 = arith.addi %sxd, %k4 : index')
b('%shl = arith.constant dense<[0, 4, 8, 12]> : vector<4xi32>')
b('%shh = arith.constant dense<[16, 20, 24, 28]> : vector<4xi32>')
b('%m15 = arith.constant dense<15> : vector<4xi32>')
b('%vz = arith.constant dense<0.0> : vector<4xf32>')
b('%m8 = arith.constant -8.0 : f16')
b('%zf = arith.constant 0.0 : f32')
b('%z = gpu.subgroup_mma_constant_matrix %zf : ' + ACC)
# the MMA operands of this subgroup: tokens th * 16.., rows rh * 32..
b(f'%th512 = arith.muli %th, {c(512)} : index')       # sx[(th * 16) * 32]
b('%rh32 = arith.muli %rh, %k32 : index')              # sw[rh * 32]
accs = ', '.join(f'%a{i} = %z' for i in range(8))
accty = ', '.join([ACC] * 8)
b(f'%res:8 = scf.for %g = %k0 to %Gl step %k1 iter_args({accs}) -> ({accty}) {{')
d = lambda s: e('            ' + s)
d('%g4 = arith.muli %g, %k4 : index')
d('%wa = arith.addi %wbase, %g4 : index')
d('%q = vector.load %w[%wa] {alignment = 8} : memref<?xi32>, vector<2xi32>')
d('%sa = arith.addi %sbase, %g : index')
d('%s = memref.load %sc[%sa] : memref<?xf16>')
d('%m = arith.mulf %s, %m8 : f16')
d('%sv = vector.broadcast %s : f16 to vector<4xf16>')
d('%mv = vector.broadcast %m : f16 to vector<4xf16>')
for wi in range(2):
    d(f'%q{wi} = vector.extract %q[{wi}] : i32 from vector<2xi32>')
    d(f'%qb{wi} = vector.broadcast %q{wi} : i32 to vector<4xi32>')
    for hi, sh in enumerate(['%shl', '%shh']):
        t = f'{wi}{hi}'
        d(f'%s{t} = arith.shrui %qb{wi}, {sh} : vector<4xi32>')
        d(f'%n{t} = arith.andi %s{t}, %m15 : vector<4xi32>')
        d(f'%f{t} = arith.uitofp %n{t} : vector<4xi32> to vector<4xf16>')
        d(f'%d{t} = vector.fma %f{t}, %sv, %mv : vector<4xf16>')
d('%g32 = arith.muli %g, %k32 : index')
d('%xc = arith.addi %g32, %cl : index')
d('%xc4 = arith.addi %xc, %k4 : index')
d('%x0r = vector.load %x[%from, %xc] {alignment = 16} : memref<?x?xf32>, vector<4xf32>')
d('%x1r = vector.load %x[%from, %xc4] {alignment = 16} : memref<?x?xf32>, vector<4xf32>')
d('%x0 = arith.select %live, %x0r, %vz : vector<4xf32>')
d('%x1 = arith.select %live, %x1r, %vz : vector<4xf32>')
d('%h0 = arith.truncf %x0 : vector<4xf32> to vector<4xf16>')
d('%h1 = arith.truncf %x1 : vector<4xf32> to vector<4xf16>')
d('gpu.barrier')
for k in range(16):
    wi, rem = divmod(k, 8)
    hi, j = divmod(rem, 4)
    d(f'%e{k} = vector.extract %d{wi}{hi}[{j}] : f16 from vector<4xf16>')
    if k == 0:
        d(f'memref.store %e0, %sw[%swd] : memref<2048xf16, 3>')
    else:
        d(f'%o{k} = arith.addi %swd, {c(k * 64)} : index')
        d(f'memref.store %e{k}, %sw[%o{k}] : memref<2048xf16, 3>')
d('vector.store %h0, %sx[%sxd] {alignment = 8} : memref<1024xf16, 3>, vector<4xf16>')
d('vector.store %h1, %sx[%sxd4] {alignment = 8} : memref<1024xf16, 3>, vector<4xf16>')
d('gpu.barrier')
prev = [f'%a{i}' for i in range(8)]
for k8 in (0, 8, 16, 24):
    if hint:
        d('vx.subgroup_schedule_barrier')
    for j in range(4):
        d(f'%bo{j}_{k8} = arith.addi %rh32, {c(k8 * 64 + j * 8)} : index')
        d(f'%mb{j}_{k8} = gpu.subgroup_mma_load_matrix %sw[%bo{j}_{k8}] {{leadDimension = 64 : index}} : memref<2048xf16, 3> -> {BOP}')
    if hint:
        d('vx.subgroup_schedule_barrier')
    for i in range(2):
        d(f'%ao{i}_{k8} = arith.addi %th512, {c(i * 8 * 32 + k8)} : index')
        d(f'%ma{i}_{k8} = gpu.subgroup_mma_load_matrix %sx[%ao{i}_{k8}] {{leadDimension = 32 : index}} : memref<1024xf16, 3> -> {AOP}')
    if hint:
        d('vx.subgroup_schedule_barrier')
    cur = []
    for i in range(2):
        for j in range(4):
            n = i * 4 + j
            d(f'%r{n}_{k8} = gpu.subgroup_mma_compute %ma{i}_{k8}, %mb{j}_{k8}, {prev[n]} : {AOP}, {BOP} -> {ACC}')
            cur.append(f'%r{n}_{k8}')
    prev = cur
d('scf.yield ' + ', '.join(prev) + ' : ' + accty)
b('}')
b('%th16 = arith.muli %th, %k16 : index')
b('%yr0 = arith.addi %tb32, %th16 : index')
b('%yr1 = arith.addi %yr0, %k8 : index')
b('%yc0 = arith.addi %rb64, %rh32 : index')
for j in range(1, 4):
    b(f'%yc{j} = arith.addi %yc0, {c(8 * j)} : index')
for i in range(2):
    for j in range(4):
        b(f'gpu.subgroup_mma_store_matrix %res#{i * 4 + j}, %y[%yr{i}, %yc{j}] {{leadDimension = 0 : index}} : {ACC}, memref<?x?xf32>')
b('macro.yield %tb : i32')
e('        };')
e('      }')
e('    }')
e('  }')
e('  return;')
e('}')
hdr = next(i for i, x in enumerate(L) if 'mlir!(' in x)
L[hdr + 1:hdr + 1] = ['          ' + f'%k{n} = arith.constant {n} : index' for n in sorted(consts)]
out = '\n'.join(L) + '\n'
print(out, end='')
