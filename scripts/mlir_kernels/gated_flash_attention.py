#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints pf_gated_flash_attention of src/prefill35.vx: the gated attention of a chunk of Qwen3.5 and
Qwen3.6 (heads of 256 values) as one threadgroup mlir! block.

It is scripts/mlir_kernels/flash_attention.py for heads of 256 values, with the output gate
sigmoid(gate) in the last step. The design is that of llama.cpp's kernel_flash_attn_ext
(ggml/src/ggml-metal/kernels/fa_common.metal), MIT License, Copyright (c) 2023-2026 The ggml
authors. Edit this script, not the generated MLIR:

    python3 scripts/mlir_kernels/gated_flash_attention.py

prints the function; replace it in src/prefill35.vx.
"""
D = 256
ACC = '!gpu.mma_matrix<8x8xf32, "COp">'
AOP = '!gpu.mma_matrix<8x8xf16, "AOp">'
BOP = '!gpu.mma_matrix<8x8xf16, "BOp">'
KV = 'memref<?x?xf16, strided<[?, 1]>>'
SQ = f'memref<{8 * D}xf16, 3>'
SO = f'memref<{8 * D}xf32, 3>'
L = []
e = L.append
e(f'/// y[t][h * {D} + d] = the attention of query head h of token t (at position start + t) over the')
e('/// positions up to start + t, times sigmoid(qkvg[t][gate_at + h * 256 + d]), for t below n: the')
e('/// gated attention of `gated_attention`. A threadgroup does 8 tokens of one query head, in steps of 64')
e('/// positions, as `pf_flash_attention` and llama.cpp\'s kernel_flash_attn_ext. The caches are')
e(f'/// [position, kv_heads, {D}] f16, with a multiple of 64 positions. K and V go from the cache straight')
e('/// into simdgroup matrices, K transposed; Q (as f16), the scores, the weights (as f16) and the sums')
e('/// of the 8 tokens are in shared memory. The softmax is online, in all the lanes of a subgroup.')
e('///')
e('/// The design is that of llama.cpp (ggml/src/ggml-metal/kernels/fa_common.metal), MIT License,')
e('/// Copyright (c) 2023-2026 The ggml authors.')
e('///')
e('/// scripts/mlir_kernels/gated_flash_attention.py writes this function: change that script, not the')
e('/// MLIR.')
e('fn pf_gated_flash_attention(y : &mut Tensor<f32, [?, ?], Memory::HBM>, q : &Tensor<f32, [?, ?, ?], Memory::HBM>, k_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, v_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, qkvg : &Tensor<f32, [?, ?], Memory::HBM>, start : i32, n : i32, q_heads : i32, group : i32, gate_at : i32, scale : f32) -> void {')
e(f'  assert(q.extent(2) == {D} && k_cache.extent(2) == {D} && k_cache.extent(0) % 64 == 0 && start + n <= k_cache.extent(0), "pf_gated_flash_attention: heads of {D} values, and a KV cache of a multiple of 64 positions");')
e('  spawn on(Topology::Device) {')
e('    for qb in 0..(n + 7) / 8 across blocks {')
e('      for h in 0..q_heads across blocks {')
e(f'        let mut sq = unsafe {{ Tensor<f16, [{8 * D}], Memory::SMEM>::uninit() }};')
e(f'        let mut so = unsafe {{ Tensor<f32, [{8 * D}], Memory::SMEM>::uninit() }};')
e('        let mut ss = unsafe { Tensor<f32, [512], Memory::SMEM>::uninit() };')
e('        let mut sp = unsafe { Tensor<f16, [512], Memory::SMEM>::uninit() };')
e(f'        let _d = mlir!( threads : 128, inputs : (%sq = &mut sq : {SQ}, %so = &mut so : {SO}, %ss = &mut ss : memref<512xf32, 3>, %sp = &mut sp : memref<512xf16, 3>, %y = y : memref<?x?xf32>, %q = q : memref<?x?x?xf32>, %k3 = k_cache : memref<?x?x?xf16>, %v3 = v_cache : memref<?x?x?xf16>, %qkvg = qkvg : memref<?x?xf32>, %start = start : i32, %n = n : i32, %group = group : i32, %gate_at = gate_at : i32, %scale = scale : f32, %qb = qb : i32, %h = h : i32), returns : i32, dialects : ["arith", "memref", "scf", "gpu", "vector", "math"], ensures : [writes_only(sq), writes_only(so), writes_only(ss), writes_only(sp), reads_only(q), reads_only(k_cache), reads_only(v_cache), reads_only(qkvg), writes_only(y[qb * 8..qb * 8 + 8][h * {D}..h * {D} + {D}])]) {{')
b = lambda s: e('          ' + s)
for c in [0, 1, 2, 4, 8, 12, 16, 32, 64, D]:
    b(f'%c{c} = arith.constant {c} : index')
b('%tid = gpu.thread_id x')
b('%sg = arith.divui %tid, %c32 : index')
b('%l = arith.remui %tid, %c32 : index')
b('%qbi = arith.index_cast %qb : i32 to index')
b('%hi = arith.index_cast %h : i32 to index')
b('%ni = arith.index_cast %n : i32 to index')
b('%starti = arith.index_cast %start : i32 to index')
b('%groupi = arith.index_cast %group : i32 to index')
b('%gatei = arith.index_cast %gate_at : i32 to index')
b('%g = arith.divui %hi, %groupi : index')
b(f'%gD = arith.muli %g, %c{D} : index')
b('%P = memref.dim %k3, %c0 : memref<?x?x?xf16>')
b('%KH = memref.dim %k3, %c1 : memref<?x?x?xf16>')
b(f'%Wd = arith.muli %KH, %c{D} : index')
b(f'%k = memref.reinterpret_cast %k3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
b(f'%v = memref.reinterpret_cast %v3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
b('%t0 = arith.muli %qbi, %c8 : index')
# Q: thread loads 16 floats of row j = tid / 16, columns (tid % 16) * 16, as f16 into sq; zeros so
b('%qj = arith.divui %tid, %c16 : index')
b('%qm = arith.remui %tid, %c16 : index')
b('%qc = arith.muli %qm, %c16 : index')
b('%qt = arith.addi %t0, %qj : index')
b('%nm1 = arith.subi %ni, %c1 : index')
b('%qtc = arith.minui %qt, %nm1 : index')
b('%qlive = arith.cmpi ult, %qt, %ni : index')
b('%vz4 = arith.constant dense<0.0> : vector<4xf32>')
b(f'%sqo = arith.muli %qj, %c{D} : index')
b('%sqa = arith.addi %sqo, %qc : index')
for i in range(4):
    b(f'%qc{i} = arith.addi %qc, %c{4 * i} : index')
    b(f'%sq{i} = arith.addi %sqa, %c{4 * i} : index')
    b(f'%q{i}r = vector.load %q[%qtc, %hi, %qc{i}] : memref<?x?x?xf32>, vector<4xf32>')
    b(f'%q{i} = arith.select %qlive, %q{i}r, %vz4 : vector<4xf32>')
    b(f'%q{i}h = arith.truncf %q{i} : vector<4xf32> to vector<4xf16>')
    b(f'vector.store %q{i}h, %sq[%sq{i}] : {SQ}, vector<4xf16>')
    b(f'vector.store %vz4, %so[%sq{i}] : {SO}, vector<4xf32>')
b('gpu.barrier')
# the positions: up to the last query of the tile
b('%tl = arith.addi %t0, %c8 : index')
b('%tlc = arith.minui %tl, %ni : index')
b('%lastq = arith.subi %tlc, %c1 : index')
b('%lastp = arith.addi %starti, %lastq : index')
b('%nblk0 = arith.divui %lastp, %c64 : index')
b('%nblk = arith.addi %nblk0, %c1 : index')
# rows of this simdgroup: j0 = sg, j1 = sg + 4; query positions
b('%j0 = arith.addi %sg, %c0 : index')
b('%j1 = arith.addi %sg, %c4 : index')
for jj in range(2):
    b(f'%qt{jj} = arith.addi %t0, %j{jj} : index')
    b(f'%qp{jj} = arith.addi %starti, %qt{jj} : index')
b('%neg = arith.constant -3.0e38 : f32')
b('%zero = arith.constant 0.0 : f32')
b('%l2 = arith.muli %l, %c2 : index')
b('%l8 = arith.muli %l, %c8 : index')
b('%zacc = gpu.subgroup_mma_constant_matrix %zero : ' + ACC)
b('%st:4 = scf.for %kb = %c0 to %nblk step %c1 iter_args(%M0 = %neg, %S0 = %zero, %M1 = %neg, %S1 = %zero) -> (f32, f32, f32, f32) {')
c = lambda s: e('            ' + s)
d = lambda s: e('              ' + s)
c('%p0 = arith.muli %kb, %c64 : index')
c('%sg8 = arith.muli %sg, %c8 : index')
# Q K^T: simdgroup sg makes the 8 x 8 score tiles at key columns 8 sg and 8 sg + 32
for nc in range(2):
    c(f'%kc{nc} = arith.addi %sg8, %c{32 * nc} : index')
    c(f'%kp{nc} = arith.addi %p0, %kc{nc} : index')
    c(f'%qk{nc} = scf.for %i = %c0 to %c{D // 8} step %c1 iter_args(%a = %zacc) -> ({ACC}) {{')
    d('%d0 = arith.muli %i, %c8 : index')
    d(f'%mq = gpu.subgroup_mma_load_matrix %sq[%d0] {{leadDimension = {D} : index}} : {SQ} -> {AOP}')
    d('%kd = arith.addi %gD, %d0 : index')
    d(f'%mk = gpu.subgroup_mma_load_matrix %k[%kp{nc}, %kd] {{leadDimension = 0 : index, transpose}} : {KV} -> {BOP}')
    d(f'%a2 = gpu.subgroup_mma_compute %mq, %mk, %a : {AOP}, {BOP} -> {ACC}')
    d(f'scf.yield %a2 : {ACC}')
    c('}')
    c(f'gpu.subgroup_mma_store_matrix %qk{nc}, %ss[%kc{nc}] {{leadDimension = 64 : index}} : {ACC}, memref<512xf32, 3>')
c('gpu.barrier')
# online softmax: rows j0, j1 of this simdgroup; lane l takes columns 2l, 2l + 1
outs = []
for jj in range(2):
    c(f'%r{jj} = arith.muli %j{jj}, %c64 : index')
    c(f'%ia{jj} = arith.addi %r{jj}, %l2 : index')
    c(f'%ib{jj} = arith.addi %ia{jj}, %c1 : index')
    c(f'%sa{jj}r = memref.load %ss[%ia{jj}] : memref<512xf32, 3>')
    c(f'%sb{jj}r = memref.load %ss[%ib{jj}] : memref<512xf32, 3>')
    c(f'%sa{jj}s = arith.mulf %sa{jj}r, %scale : f32')
    c(f'%sb{jj}s = arith.mulf %sb{jj}r, %scale : f32')
    c(f'%pa{jj} = arith.addi %p0, %l2 : index')
    c(f'%pb{jj} = arith.addi %pa{jj}, %c1 : index')
    c(f'%oka{jj} = arith.cmpi ule, %pa{jj}, %qp{jj} : index')
    c(f'%okb{jj} = arith.cmpi ule, %pb{jj}, %qp{jj} : index')
    c(f'%sa{jj} = arith.select %oka{jj}, %sa{jj}s, %neg : f32')
    c(f'%sb{jj} = arith.select %okb{jj}, %sb{jj}s, %neg : f32')
    c(f'%mx{jj} = arith.maxnumf %sa{jj}, %sb{jj} : f32')
    c(f'%bm{jj} = gpu.subgroup_reduce maxnumf %mx{jj} : (f32) -> f32')
    c(f'%Mn{jj} = arith.maxnumf %M{jj}, %bm{jj} : f32')
    c(f'%dm{jj} = arith.subf %M{jj}, %Mn{jj} : f32')
    c(f'%ms{jj} = math.exp %dm{jj} : f32')
    c(f'%ea{jj} = arith.subf %sa{jj}, %Mn{jj} : f32')
    c(f'%eb{jj} = arith.subf %sb{jj}, %Mn{jj} : f32')
    c(f'%xa{jj} = math.exp %ea{jj} : f32')
    c(f'%xb{jj} = math.exp %eb{jj} : f32')
    c(f'%xs{jj} = arith.addf %xa{jj}, %xb{jj} : f32')
    c(f'%bs{jj} = gpu.subgroup_reduce add %xs{jj} : (f32) -> f32')
    c(f'%Ss{jj} = arith.mulf %S{jj}, %ms{jj} : f32')
    c(f'%Sn{jj} = arith.addf %Ss{jj}, %bs{jj} : f32')
    c(f'%ha{jj} = arith.truncf %xa{jj} : f32 to f16')
    c(f'%hb{jj} = arith.truncf %xb{jj} : f32 to f16')
    c(f'memref.store %ha{jj}, %sp[%ia{jj}] : memref<512xf16, 3>')
    c(f'memref.store %hb{jj}, %sp[%ib{jj}] : memref<512xf16, 3>')
    # rescale this row of O: lane l takes 8 values
    c(f'%or{jj} = arith.muli %j{jj}, %c{D} : index')
    c(f'%oi{jj} = arith.addi %or{jj}, %l8 : index')
    c(f'%oj{jj} = arith.addi %oi{jj}, %c4 : index')
    c(f'%msv{jj} = vector.broadcast %ms{jj} : f32 to vector<4xf32>')
    for half, idx in (('a', f'%oi{jj}'), ('b', f'%oj{jj}')):
        c(f'%ov{jj}{half} = vector.load %so[{idx}] : {SO}, vector<4xf32>')
        c(f'%ow{jj}{half} = arith.mulf %ov{jj}{half}, %msv{jj} : vector<4xf32>')
        c(f'vector.store %ow{jj}{half}, %so[{idx}] : {SO}, vector<4xf32>')
    outs += [f'%Mn{jj}', f'%Sn{jj}']
c('gpu.barrier')
# P V: simdgroup sg owns the output columns 8 sg + 32 ii
for ii in range(D // 32):
    c(f'%dv{ii}c = arith.constant {32 * ii} : index')
    c(f'%dv{ii} = arith.addi %sg8, %dv{ii}c : index')
    c(f'%o{ii}i = gpu.subgroup_mma_load_matrix %so[%dv{ii}] {{leadDimension = {D} : index}} : {SO} -> {ACC}')
    c(f'%vd{ii} = arith.addi %gD, %dv{ii} : index')
    c(f'%o{ii} = scf.for %kk = %c0 to %c8 step %c1 iter_args(%o = %o{ii}i) -> ({ACC}) {{')
    d('%k8 = arith.muli %kk, %c8 : index')
    d(f'%mp = gpu.subgroup_mma_load_matrix %sp[%k8] {{leadDimension = 64 : index}} : memref<512xf16, 3> -> {AOP}')
    d('%vp = arith.addi %p0, %k8 : index')
    d(f'%mv = gpu.subgroup_mma_load_matrix %v[%vp, %vd{ii}] {{leadDimension = 0 : index}} : {KV} -> {BOP}')
    d(f'%onx = gpu.subgroup_mma_compute %mp, %mv, %o : {AOP}, {BOP} -> {ACC}')
    d(f'scf.yield %onx : {ACC}')
    c('}')
    c(f'gpu.subgroup_mma_store_matrix %o{ii}, %so[%dv{ii}] {{leadDimension = {D} : index}} : {ACC}, {SO}')
c('gpu.barrier')
c('scf.yield ' + ', '.join(outs) + ' : f32, f32, f32, f32')
b('}')
# epilogue: the rows of this simdgroup, divided by their sums, times sigmoid of the gate
b(f'%hD = arith.muli %hi, %c{D} : index')
b('%gh = arith.addi %gatei, %hD : index')
b('%one = arith.constant 1.0 : f32')
b('%one4 = arith.constant dense<1.0> : vector<4xf32>')
for jj in range(2):
    b(f'%inv{jj} = arith.divf %one, %st#{2 * jj + 1} : f32')
    b(f'%iv{jj} = vector.broadcast %inv{jj} : f32 to vector<4xf32>')
    b(f'%fr{jj} = arith.muli %j{jj}, %c{D} : index')
    b(f'%live{jj} = arith.cmpi ult, %qt{jj}, %ni : index')
    b(f'scf.if %live{jj} {{')
    for half, off in (('a', 0), ('b', 4)):
        b(f'  %fl{jj}{half} = arith.addi %l8, %c{off} : index')
        b(f'  %fi{jj}{half} = arith.addi %fr{jj}, %fl{jj}{half} : index')
        b(f'  %fv{jj}{half} = vector.load %so[%fi{jj}{half}] : {SO}, vector<4xf32>')
        b(f'  %fw{jj}{half} = arith.mulf %fv{jj}{half}, %iv{jj} : vector<4xf32>')
        b(f'  %gi{jj}{half} = arith.addi %gh, %fl{jj}{half} : index')
        b(f'  %gv{jj}{half} = vector.load %qkvg[%qt{jj}, %gi{jj}{half}] : memref<?x?xf32>, vector<4xf32>')
        b(f'  %gn{jj}{half} = arith.negf %gv{jj}{half} : vector<4xf32>')
        b(f'  %ge{jj}{half} = math.exp %gn{jj}{half} : vector<4xf32>')
        b(f'  %gd{jj}{half} = arith.addf %ge{jj}{half}, %one4 : vector<4xf32>')
        b(f'  %fg{jj}{half} = arith.divf %fw{jj}{half}, %gd{jj}{half} : vector<4xf32>')
        b(f'  %yc{jj}{half} = arith.addi %hD, %fl{jj}{half} : index')
        b(f'  vector.store %fg{jj}{half}, %y[%qt{jj}, %yc{jj}{half}] : memref<?x?xf32>, vector<4xf32>')
    b('}')
b('macro.yield %qb : i32')
e('        };')
e('      }')
e('    }')
e('  }')
e('  return;')
e('}')
print('\n'.join(L))
