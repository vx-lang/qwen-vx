#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints vec_attention_parts of src/attention.vx: the attention of a generated token as an mlir! block.
vec_attention_merge, which merges the parts, is Vx code in src/attention.vx.

The design is that of llama.cpp's kernel_flash_attn_ext_vec (ggml/src/ggml-metal/ggml-metal.metal) for
head size 128 with NE 1, MIT License, Copyright (c) 2023-2026 The ggml authors. Edit this script, not the
generated MLIR:

    python3 scripts/mlir_kernels/decode_attention.py

prints the function; replace it in src/attention.vx.
"""
KV = 'memref<?x?xf16, strided<[?, 1]>>'
L = []
e = L.append
e('/// The parts of `vec_attention`, as llama.cpp\'s kernel_flash_attn_ext_vec. Block (h, w) is a SIMD group for')
e('/// query head h. It does the blocks of 32 positions w, w + nwg, w + 2 nwg, ... up to `pos`, and writes')
e('/// part[h][w]. Lane l owns values 4 l to 4 l + 3 of the head, so that the 32 lanes read a row of K or V in')
e('/// one 256-byte load. For each row, the lanes add up their parts of q . k; then each lane keeps the score')
e('/// of one row, and the lanes take the weights of the rows from each other for the sum of V. A part is the')
e('/// sum of V, not divided, then the largest score and the sum of the weights.')
e('fn vec_attention_parts(part : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, q : &Tensor<f32, [?, ?], Memory::HBM>, k_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, v_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, pos : i32, q_heads : i32, group : i32, nwg : i32, scale : f32) -> void {')
e('  spawn on(Topology::Device) {')
e('    for h in 0..q_heads across blocks {')
e('      for w in 0..nwg across blocks {')
e('        let g = h / group;')
e('        let _d = mlir!( threads : 32, inputs : (%part = part : memref<?x?x?xf32>, %q = q : memref<?x?xf32>, %k3 = k_cache : memref<?x?x?xf16>, %v3 = v_cache : memref<?x?x?xf16>, %pos = pos : i32, %nwg = nwg : i32, %scale = scale : f32, %h = h : i32, %w = w : i32, %g = g : i32), returns : i32, dialects : ["arith", "memref", "scf", "gpu", "vector", "math"], ensures : [reads_only(q), reads_only(k_cache), reads_only(v_cache), writes_only(part[h][w])]) {')
b = lambda s: e('          ' + s)
for c in [0, 1, 4, 32, 128, 129]:
    b(f'%c{c} = arith.constant {c} : index')
b('%c32i = arith.constant 32 : i32')
b('%neg = arith.constant -3.0e38 : f32')
b('%zero = arith.constant 0.0 : f32')
b('%vz4 = arith.constant dense<0.0> : vector<4xf32>')
b('%l = gpu.thread_id x')
b('%l4 = arith.muli %l, %c4 : index')
for v in ['h', 'w', 'g', 'pos', 'nwg']:
    b(f'%{v}i = arith.index_cast %{v} : i32 to index')
b('%g128 = arith.muli %gi, %c128 : index')
b('%kd = arith.addi %g128, %l4 : index')
b('%P = memref.dim %k3, %c0 : memref<?x?x?xf16>')
b('%KH = memref.dim %k3, %c1 : memref<?x?x?xf16>')
b('%Wd = arith.muli %KH, %c128 : index')
b(f'%k = memref.reinterpret_cast %k3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
b(f'%v = memref.reinterpret_cast %v3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
# q of this lane, times the scale
b('%qv = vector.load %q[%hi, %l4] {alignment = 16} : memref<?x?xf32>, vector<4xf32>')
b('%scv = vector.broadcast %scale : f32 to vector<4xf32>')
b('%qs = arith.mulf %qv, %scv : vector<4xf32>')
b('%nkv = arith.addi %posi, %c32 : index')
b('%chunks = arith.divui %nkv, %c32 : index')
b('%st:3 = scf.for %ic = %wi to %chunks step %nwgi iter_args(%M = %neg, %S = %zero, %o = %vz4) -> (f32, f32, vector<4xf32>) {')
c = lambda s: e('            ' + s)
d = lambda s: e('              ' + s)
c('%r0 = arith.muli %ic, %c32 : index')
# the scores of rows r0 to r0 + 31; lane cc keeps the score of row r0 + cc. A row past `pos` reads row `pos`.
c('%my = scf.for %cc = %c0 to %c32 step %c1 iter_args(%m = %neg) -> (f32) {')
d('%row = arith.addi %r0, %cc : index')
d('%rowc = arith.minui %row, %posi : index')
d(f'%kh = vector.load %k[%rowc, %kd] {{alignment = 8}} : {KV}, vector<4xf16>')
d('%kf = arith.extf %kh : vector<4xf16> to vector<4xf32>')
d('%pr = arith.mulf %kf, %qs : vector<4xf32>')
d('%pl = vector.reduction <add>, %pr : vector<4xf32> into f32')
d('%sc = gpu.subgroup_reduce add %pl : (f32) -> f32')
d('%live = arith.cmpi ule, %row, %posi : index')
d('%scm = arith.select %live, %sc, %neg : f32')
d('%mine = arith.cmpi eq, %cc, %l : index')
d('%m2 = arith.select %mine, %scm, %m : f32')
d('scf.yield %m2 : f32')
c('}')
# online softmax over the 32 rows
c('%bm = gpu.subgroup_reduce maxnumf %my : (f32) -> f32')
c('%Mn = arith.maxnumf %M, %bm : f32')
c('%dm = arith.subf %M, %Mn : f32')
c('%ms = math.exp %dm : f32')
c('%dp = arith.subf %my, %Mn : f32')
c('%p = math.exp %dp : f32')
c('%bs = gpu.subgroup_reduce add %p : (f32) -> f32')
c('%Ss = arith.mulf %S, %ms : f32')
c('%Sn = arith.addf %Ss, %bs : f32')
c('%msv = vector.broadcast %ms : f32 to vector<4xf32>')
c('%o1 = arith.mulf %o, %msv : vector<4xf32>')
# the sum of V: the weight of row r0 + cc is in lane cc
c('%o2 = scf.for %cc = %c0 to %c32 step %c1 iter_args(%oa = %o1) -> (vector<4xf32>) {')
d('%row = arith.addi %r0, %cc : index')
d('%rowc = arith.minui %row, %posi : index')
d(f'%vh = vector.load %v[%rowc, %kd] {{alignment = 8}} : {KV}, vector<4xf16>')
d('%vf = arith.extf %vh : vector<4xf16> to vector<4xf32>')
d('%cci = arith.index_cast %cc : index to i32')
d('%pc, %pv = gpu.shuffle idx %p, %cci, %c32i : f32')
d('%pcv = vector.broadcast %pc : f32 to vector<4xf32>')
d('%on = vector.fma %vf, %pcv, %oa : vector<4xf32>')
d('scf.yield %on : vector<4xf32>')
c('}')
c('scf.yield %Mn, %Sn, %o2 : f32, f32, vector<4xf32>')
b('}')
b('vector.store %st#2, %part[%hi, %wi, %l4] : memref<?x?x?xf32>, vector<4xf32>')
b('%lane0 = arith.cmpi eq, %l, %c0 : index')
b('scf.if %lane0 {')
b('  memref.store %st#0, %part[%hi, %wi, %c128] : memref<?x?x?xf32>')
b('  memref.store %st#1, %part[%hi, %wi, %c129] : memref<?x?x?xf32>')
b('}')
b('macro.yield %h : i32')
e('        };')
e('      }')
e('    }')
e('  }')
e('  return;')
e('}')

print('\n'.join(L))
