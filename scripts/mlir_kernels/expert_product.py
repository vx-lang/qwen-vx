#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints pf_mm_q4_mlir of src/prefill.vx: the Q4_0 product of a chunk as one threadgroup mlir! block.
With --bf16, prints pf_mm_bf16_mlir: the same product for a bf16 matrix, whose weights go to f16.

The design is that of llama.cpp's kernel_mul_mm_id (ggml/src/ggml-metal/kernels/mul_mm.metal),
MIT License, Copyright (c) 2023-2026 The ggml authors. Edit this script, not the generated MLIR:

    python3 scripts/mlir_kernels/expert_product.py [--bf16]

prints the function; replace it in src/prefill.vx.
"""
import sys
mode = 'app'
bf16 = '--bf16' in sys.argv[1:]
# Tokens of a block: 32, or 64 with --t64 (Q4_0 only, for the dense products, with no skip of
# padding). With 64, each subgroup makes 32 tokens x 32 rows.
T = 64 if '--t64' in sys.argv[1:] else 32
NI = T // 16      # rows of 8 tokens of a subgroup
NA = 4 * NI       # sums of a subgroup
hint = True
ACC = '!gpu.mma_matrix<8x8xf32, "COp">'
AOP = '!gpu.mma_matrix<8x8xf16, "AOp">'
BOP = '!gpu.mma_matrix<8x8xf16, "BOp">'
L = []
e = L.append
name = 'pf_mm_q4_mlir' if mode == 'app' else ('mm_mlir2h' if hint else 'mm_mlir2')
if bf16:
    name = 'pf_mm_bf16_mlir'
if T == 64:
    name = 'pf_mm_q4_mlir64'
WT = 'w : &Tensor<bf16, [?], Memory::HBM>' if bf16 else 'w : &Tensor<u32, [?], Memory::HBM>, scales : &Tensor<f16, [?], Memory::HBM>'
e(f'fn {name}(y : &mut Tensor<f32, [?, ?], Memory::HBM>, {WT}, x : &Tensor<f32, [?, ?], Memory::HBM>, list : &Tensor<i32, [?], Memory::HBM>, bases : &Tensor<i32, [?], Memory::HBM>, stride : i32, rows : i32, cols : i32, tiles : i32) -> void {{')
e('  spawn on(Topology::Device) {')
e('    for rb in 0..(rows + 63) / 64 across blocks {' if bf16 else '    for rb in 0..rows / 64 across blocks {')
e(f'      for tb in 0..(tiles + {T // 16 - 1}) / {T // 16} across blocks {{' if mode == 'app' else '      for tb in 0..tiles across blocks {')
e('        let mut sw = unsafe { Tensor<f16, [2048], Memory::SMEM>::uninit() };')
e(f'        let mut sx = unsafe {{ Tensor<f16, [{T * 32}], Memory::SMEM>::uninit() }};')
SX = f'memref<{T * 32}xf16, 3>'
e('        let _d = mlir!( threads : 128, inputs : (%sw = &mut sw : memref<2048xf16, 3>, %sx = &mut sx : ' + SX + ', %y = y : memref<?x?xf32>, ' + ('%w = w : memref<?xbf16>, %rows = rows : i32' if bf16 else '%w = w : memref<?xi32>, %sc = scales : memref<?xf16>') + ', %x = x : memref<?x?xf32>, %list = list : memref<?xi32>, %bases = bases : memref<?xi32>, %stride = stride : i32, %cols = cols : i32, %rb = rb : i32, %tb = tb : i32), returns : i32, dialects : ['+('"arith", "memref", "scf", "gpu", "vector", "vx"' if hint else '"arith", "memref", "scf", "gpu", "vector"')+'], ensures : [writes_only(sw), writes_only(sx), reads_only(w), ' + ('' if bf16 else 'reads_only(scales), ') + 'reads_only(x), reads_only(list), reads_only(bases), writes_only(y[tb * ' + str(T) + '..tb * ' + str(T) + ' + ' + str(T) + '][rb * 64..rb * 64 + 64])]) {')
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
    b(f'%tbb = arith.muli %tbi, {c(T // 16)} : index')
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
if bf16:
    # A row past the last reads row 0 of the matrix and gets zeros.
    b('%rowsi = arith.index_cast %rows : i32 to index')
    b('%rok = arith.cmpi ult, %r, %rowsi : index')
    b('%rowc = arith.select %rok, %row, %k0 : index')
    b('%rw = arith.muli %rowc, %colsi : index')
    b('%kh16 = arith.muli %kh, %k16 : index')
    b('%wbase = arith.addi %rw, %kh16 : index')
    b('%hz = arith.constant dense<0.0> : vector<4xf16>')
else:
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
b(f'%tb32 = arith.muli %tbi, {c(T)} : index')     # the first token of the block
# With 64 tokens, lane l also copies token tl + 32 (suffix _1).
XS = [''] + [f'_{xi}' for xi in range(1, T // 32)]
for xi, u in enumerate(XS):
    tlu = '%tl'
    if xi:
        b(f'%tl{u} = arith.addi %tl, {c(32 * xi)} : index')
        tlu = f'%tl{u}'
    b(f'%li{u} = arith.addi %tb32, {tlu} : index')
    b(f'%from32{u} = memref.load %list[%li{u}] : memref<?xi32>')
    b(f'%live{u} = arith.cmpi sge, %from32{u}, %zi : i32')
    b(f'%fromc{u} = arith.maxsi %from32{u}, %zi : i32')
    b(f'%from{u} = arith.index_cast %fromc{u} : i32 to index')
    b(f'%tl32{u} = arith.muli {tlu}, %k32 : index')
    b(f'%sxd{u} = arith.addi %tl32{u}, %cl : index')
    b(f'%sxd4{u} = arith.addi %sxd{u}, %k4 : index')
b('%shl = arith.constant dense<[0, 4, 8, 12]> : vector<4xi32>')
b('%shh = arith.constant dense<[16, 20, 24, 28]> : vector<4xi32>')
b('%m15 = arith.constant dense<15> : vector<4xi32>')
b('%vz = arith.constant dense<0.0> : vector<4xf32>')
b('%m8 = arith.constant -8.0 : f16')
b('%zf = arith.constant 0.0 : f32')
b('%z = gpu.subgroup_mma_constant_matrix %zf : ' + ACC)
# the MMA operands of this subgroup: tokens th * 16.., rows rh * 32..
b(f'%th512 = arith.muli %th, {c(T * 16)} : index')       # sx[(th * T / 2) * 32]
b(f'%th16 = arith.muli %th, {c(T // 2)} : index')
b('%rh32 = arith.muli %rh, %k32 : index')              # sw[rh * 32]
# The tokens of an expert come first in its tiles, so when token t of the block is padding, so are
# the tokens after it. A subgroup skips the multiplies of its 8 tokens th * 16 + 8 to + 15 when the
# first of them is padding (busy1 false), and all its multiplies when token th * 16 is padding.
if T == 32:
    b('%tf0 = arith.addi %tb32, %th16 : index')
    b('%tf1 = arith.addi %tf0, %k8 : index')
    for i in range(2):
        b(f'%from_t{i} = memref.load %list[%tf{i}] : memref<?xi32>')
        b(f'%busy{i} = arith.cmpi sge, %from_t{i}, %zi : i32')
accs = ', '.join(f'%a{i} = %z' for i in range(NA))
accty = ', '.join([ACC] * NA)
b(f'%res:{NA} = scf.for %g = %k0 to %Gl step %k1 iter_args({accs}) -> ({accty}) {{')
d = lambda s: e('            ' + s)
if bf16:
    # Lane l loads columns kh * 16 to + 15 of its row, as 4 vectors of 4.
    d('%g32w = arith.muli %g, %k32 : index')
    d('%wa = arith.addi %wbase, %g32w : index')
    for wi in range(2):
        for hi in range(2):
            t = f'{wi}{hi}'
            d(f'%wa{t} = arith.addi %wa, {c(wi * 8 + hi * 4)} : index')
            d(f'%b{t} = vector.load %w[%wa{t}] {{alignment = 8}} : memref<?xbf16>, vector<4xbf16>')
            d(f'%we{t} = arith.extf %b{t} : vector<4xbf16> to vector<4xf32>')
            d(f'%t{t} = arith.truncf %we{t} : vector<4xf32> to vector<4xf16>')
            d(f'%d{t} = arith.select %rok, %t{t}, %hz : vector<4xf16>')
if not bf16:
    d('%g4 = arith.muli %g, %k4 : index')
    d('%wa = arith.addi %wbase, %g4 : index')
    d('%q = vector.load %w[%wa] {alignment = 8} : memref<?xi32>, vector<2xi32>')
    d('%sa = arith.addi %sbase, %g : index')
    d('%s = memref.load %sc[%sa] : memref<?xf16>')
    d('%m = arith.mulf %s, %m8 : f16')
    d('%sv = vector.broadcast %s : f16 to vector<4xf16>')
    d('%mv = vector.broadcast %m : f16 to vector<4xf16>')
for wi in range(0 if bf16 else 2):
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
for u in XS:
    d(f'%x0r{u} = vector.load %x[%from{u}, %xc] {{alignment = 16}} : memref<?x?xf32>, vector<4xf32>')
    d(f'%x1r{u} = vector.load %x[%from{u}, %xc4] {{alignment = 16}} : memref<?x?xf32>, vector<4xf32>')
    d(f'%x0{u} = arith.select %live{u}, %x0r{u}, %vz : vector<4xf32>')
    d(f'%x1{u} = arith.select %live{u}, %x1r{u}, %vz : vector<4xf32>')
    d(f'%h0{u} = arith.truncf %x0{u} : vector<4xf32> to vector<4xf16>')
    d(f'%h1{u} = arith.truncf %x1{u} : vector<4xf32> to vector<4xf16>')
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
for u in XS:
    d(f'vector.store %h0{u}, %sx[%sxd{u}] {{alignment = 8}} : {SX}, vector<4xf16>')
    d(f'vector.store %h1{u}, %sx[%sxd4{u}] {{alignment = 8}} : {SX}, vector<4xf16>')
d('gpu.barrier')
def mma(d, tag, rows):
    """The multiplies of a step for the token rows `rows` (0: tokens th * 16 to + 7, 1: + 8 to + 15).
    Yields the sums; those of the other rows go through as they are."""
    prev = [f'%a{i}' for i in range(NA)]
    for k8 in (0, 8, 16, 24):
        if hint:
            d('vx.subgroup_schedule_barrier')
        for j in range(4):
            d(f'%bo{j}_{k8}{tag} = arith.addi %rh32, {c(k8 * 64 + j * 8)} : index')
            d(f'%mb{j}_{k8}{tag} = gpu.subgroup_mma_load_matrix %sw[%bo{j}_{k8}{tag}] {{leadDimension = 64 : index}} : memref<2048xf16, 3> -> {BOP}')
        if hint:
            d('vx.subgroup_schedule_barrier')
        for i in rows:
            d(f'%ao{i}_{k8}{tag} = arith.addi %th512, {c(i * 8 * 32 + k8)} : index')
            d(f'%ma{i}_{k8}{tag} = gpu.subgroup_mma_load_matrix %sx[%ao{i}_{k8}{tag}] {{leadDimension = 32 : index}} : {SX} -> {AOP}')
        if hint:
            d('vx.subgroup_schedule_barrier')
        cur = list(prev)
        for i in rows:
            for j in range(4):
                n = i * 4 + j
                d(f'%r{n}_{k8}{tag} = gpu.subgroup_mma_compute %ma{i}_{k8}{tag}, %mb{j}_{k8}{tag}, {prev[n]} : {AOP}, {BOP} -> {ACC}')
                cur[n] = f'%r{n}_{k8}{tag}'
        prev = cur
    d('scf.yield ' + ', '.join(prev) + ' : ' + accty)
if T == 64:
    mma(d, '', tuple(range(NI)))
else:
    d0 = d
    d1 = lambda s: d0('  ' + s)
    d2 = lambda s: d0('    ' + s)
    d(f'%mm:8 = scf.if %busy1 -> ({accty}) {{')
    mma(d1, 'f', (0, 1))
    d('} else {')
    d1(f'%mh:8 = scf.if %busy0 -> ({accty}) {{')
    mma(d2, 'h', (0,))
    d1('} else {')
    d1('  scf.yield ' + ', '.join(f'%a{i}' for i in range(8)) + ' : ' + accty)
    d1('}')
    d1('scf.yield ' + ', '.join(f'%mh#{i}' for i in range(8)) + ' : ' + accty)
    d('}')
    d('scf.yield ' + ', '.join(f'%mm#{i}' for i in range(8)) + ' : ' + accty)
b('}')
b('%yr0 = arith.addi %tb32, %th16 : index')
for i in range(1, NI):
    b(f'%yr{i} = arith.addi %yr0, {c(8 * i)} : index')
b('%yc0 = arith.addi %rb64, %rh32 : index')
for j in range(1, 4):
    b(f'%yc{j} = arith.addi %yc0, {c(8 * j)} : index')
for i in range(NI):
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
