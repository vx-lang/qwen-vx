#!/usr/bin/env python3
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Prints the kernels of `round_attention` in src/round.vx: the attention of n positions in a row, at
start to start + n - 1, as mlir! blocks.

Each query gets the result of `vec_attention` (src/attention.vx) bit for bit. A block does the same
work for each of its n queries that the block of `vec_attention_parts` does for one, in the same
order, but it reads each row of K and V once for all of them. A query skips the blocks of 32
positions after its own position, and a row after its position gets the score -3e38 and so the
weight 0, as in `vec_attention_parts`. The rows after the last query are not read.

    python3 scripts/mlir_kernels/round_attention.py

prints the functions; replace them in src/round.vx.
"""
KV = 'memref<?x?xf16, strided<[?, 1]>>'
MAX_N = 8
L = []
e = L.append


def parts(n):
    """The kernel of the parts for n queries."""
    e('')
    e(f'/// `vec_attention_parts` for {n} queries in a row: query t is at position pos + t, in q[t0 + t], and its part')
    e('/// w goes to part[h][w][132 (t0 + t) ..].')
    e(f'fn round_attention_parts_{n}(part : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, q : &Tensor<f32, [?, ?, ?], Memory::HBM>, k_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, v_cache : &Tensor<f16, [?, ?, ?], Memory::HBM>, pos : i32, t0 : i32, q_heads : i32, group : i32, nwg : i32, scale : f32) -> void {{')
    e('  spawn on(Topology::Device) {')
    e('    for h in 0..q_heads across blocks {')
    e('      for w in 0..nwg across blocks {')
    e('        let g = h / group;')
    e('        let _d = mlir!( threads : 32, inputs : (%part = part : memref<?x?x?xf32>, %q = q : memref<?x?x?xf32>, %k3 = k_cache : memref<?x?x?xf16>, %v3 = v_cache : memref<?x?x?xf16>, %pos = pos : i32, %t0 = t0 : i32, %nwg = nwg : i32, %scale = scale : f32, %h = h : i32, %w = w : i32, %g = g : i32), returns : i32, dialects : ["arith", "memref", "scf", "gpu", "vector", "math"], ensures : [reads_only(q), reads_only(k_cache), reads_only(v_cache), writes_only(part[h][w])]) {')
    b = lambda s: e('          ' + s)
    for c in [0, 1, 4, 32, 128, 129, 132]:
        b(f'%c{c} = arith.constant {c} : index')
    b('%c32i = arith.constant 32 : i32')
    b('%neg = arith.constant -3.0e38 : f32')
    b('%zero = arith.constant 0.0 : f32')
    b('%vz4 = arith.constant dense<0.0> : vector<4xf32>')
    b('%l = gpu.thread_id x')
    b('%l4 = arith.muli %l, %c4 : index')
    for v in ['h', 'w', 'g', 'pos', 'nwg', 't0']:
        b(f'%{v}i = arith.index_cast %{v} : i32 to index')
    b('%g128 = arith.muli %gi, %c128 : index')
    b('%kd = arith.addi %g128, %l4 : index')
    b('%P = memref.dim %k3, %c0 : memref<?x?x?xf16>')
    b('%KH = memref.dim %k3, %c1 : memref<?x?x?xf16>')
    b('%Wd = arith.muli %KH, %c128 : index')
    b(f'%k = memref.reinterpret_cast %k3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
    b(f'%v = memref.reinterpret_cast %v3 to offset: [0], sizes: [%P, %Wd], strides: [%Wd, 1] : memref<?x?x?xf16> to {KV}')
    b('%scv = vector.broadcast %scale : f32 to vector<4xf32>')
    for t in range(n):
        # the position of query t, its blocks of 32 positions, and its q times the scale
        b(f'%ct{t} = arith.constant {t} : index')
        b(f'%pos{t} = arith.addi %posi, %ct{t} : index')
        b(f'%n{t} = arith.addi %pos{t}, %c32 : index')
        b(f'%chunks{t} = arith.divui %n{t}, %c32 : index')
        b(f'%qt{t} = arith.addi %t0i, %ct{t} : index')
        b(f'%qv{t} = vector.load %q[%qt{t}, %hi, %l4] {{alignment = 16}} : memref<?x?x?xf32>, vector<4xf32>')
        b(f'%qs{t} = arith.mulf %qv{t}, %scv : vector<4xf32>')
    last = n - 1
    init = ', '.join(f'%M{t} = %neg, %S{t} = %zero, %o{t} = %vz4' for t in range(n))
    types = ', '.join('f32, f32, vector<4xf32>' for _ in range(n))
    b(f'%st:{3 * n} = scf.for %ic = %wi to %chunks{last} step %nwgi iter_args({init}) -> ({types}) {{')
    c = lambda s: e('            ' + s)
    d = lambda s: e('              ' + s)
    c('%r0 = arith.muli %ic, %c32 : index')
    # the scores of rows r0 to r0 + 31 for each query; lane cc keeps the scores of row r0 + cc. A row past
    # the last position reads the row of the last position.
    minit = ', '.join(f'%m{t} = %neg' for t in range(n))
    c(f'%my:{n} = scf.for %cc = %c0 to %c32 step %c1 iter_args({minit}) -> ({", ".join("f32" for _ in range(n))}) {{')
    d('%row = arith.addi %r0, %cc : index')
    d(f'%rowc = arith.minui %row, %pos{last} : index')
    d(f'%kh = vector.load %k[%rowc, %kd] {{alignment = 8}} : {KV}, vector<4xf16>')
    d('%kf = arith.extf %kh : vector<4xf16> to vector<4xf32>')
    d('%mine = arith.cmpi eq, %cc, %l : index')
    for t in range(n):
        d(f'%pr{t} = arith.mulf %kf, %qs{t} : vector<4xf32>')
        d(f'%pl{t} = vector.reduction <add>, %pr{t} : vector<4xf32> into f32')
        d(f'%sc{t} = gpu.subgroup_reduce add %pl{t} : (f32) -> f32')
        d(f'%live{t} = arith.cmpi ule, %row, %pos{t} : index')
        d(f'%scm{t} = arith.select %live{t}, %sc{t}, %neg : f32')
        d(f'%mm{t} = arith.select %mine, %scm{t}, %m{t} : f32')
    d(f'scf.yield {", ".join(f"%mm{t}" for t in range(n))} : {", ".join("f32" for _ in range(n))}')
    c('}')
    # the online softmax of each query that this block of positions reaches; p is 0 for the others, so
    # that the sum of V leaves them as they are
    for t in range(n):
        c(f'%act{t} = arith.cmpi ult, %ic, %chunks{t} : index')
        c(f'%u{t}:4 = scf.if %act{t} -> (f32, f32, vector<4xf32>, f32) {{')
        d(f'%bm{t} = gpu.subgroup_reduce maxnumf %my#{t} : (f32) -> f32')
        d(f'%Mn{t} = arith.maxnumf %M{t}, %bm{t} : f32')
        d(f'%dm{t} = arith.subf %M{t}, %Mn{t} : f32')
        d(f'%ms{t} = math.exp %dm{t} : f32')
        d(f'%dp{t} = arith.subf %my#{t}, %Mn{t} : f32')
        d(f'%p{t} = math.exp %dp{t} : f32')
        d(f'%bs{t} = gpu.subgroup_reduce add %p{t} : (f32) -> f32')
        d(f'%Ss{t} = arith.mulf %S{t}, %ms{t} : f32')
        d(f'%Sn{t} = arith.addf %Ss{t}, %bs{t} : f32')
        d(f'%msv{t} = vector.broadcast %ms{t} : f32 to vector<4xf32>')
        d(f'%ox{t} = arith.mulf %o{t}, %msv{t} : vector<4xf32>')
        d(f'scf.yield %Mn{t}, %Sn{t}, %ox{t}, %p{t} : f32, f32, vector<4xf32>, f32')
        c('} else {')
        d(f'scf.yield %M{t}, %S{t}, %o{t}, %zero : f32, f32, vector<4xf32>, f32')
        c('}')
    # the sum of V: the weight of row r0 + cc for query t is in lane cc
    oinit = ', '.join(f'%oa{t} = %u{t}#2' for t in range(n))
    vtypes = ', '.join('vector<4xf32>' for _ in range(n))
    c(f'%ov:{n} = scf.for %cc = %c0 to %c32 step %c1 iter_args({oinit}) -> ({vtypes}) {{')
    d('%row = arith.addi %r0, %cc : index')
    d(f'%rowc = arith.minui %row, %pos{last} : index')
    d(f'%vh = vector.load %v[%rowc, %kd] {{alignment = 8}} : {KV}, vector<4xf16>')
    d('%vf = arith.extf %vh : vector<4xf16> to vector<4xf32>')
    d('%cci = arith.index_cast %cc : index to i32')
    for t in range(n):
        d(f'%pc{t}, %pv{t} = gpu.shuffle idx %u{t}#3, %cci, %c32i : f32')
        d(f'%pcv{t} = vector.broadcast %pc{t} : f32 to vector<4xf32>')
        d(f'%on{t} = vector.fma %vf, %pcv{t}, %oa{t} : vector<4xf32>')
    d(f'scf.yield {", ".join(f"%on{t}" for t in range(n))} : {vtypes}')
    c('}')
    out = ', '.join(f'%u{t}#0, %u{t}#1, %ov#{t}' for t in range(n))
    c(f'scf.yield {out} : {types}')
    b('}')
    b('%lane0 = arith.cmpi eq, %l, %c0 : index')
    for t in range(n):
        b(f'%base{t} = arith.muli %qt{t}, %c132 : index')
        b(f'%at{t} = arith.addi %base{t}, %l4 : index')
        b(f'vector.store %st#{3 * t + 2}, %part[%hi, %wi, %at{t}] : memref<?x?x?xf32>, vector<4xf32>')
        b(f'%am{t} = arith.addi %base{t}, %c128 : index')
        b(f'%as{t} = arith.addi %base{t}, %c129 : index')
        b('scf.if %lane0 {')
        b(f'  memref.store %st#{3 * t}, %part[%hi, %wi, %am{t}] : memref<?x?x?xf32>')
        b(f'  memref.store %st#{3 * t + 1}, %part[%hi, %wi, %as{t}] : memref<?x?x?xf32>')
        b('}')
    b('macro.yield %h : i32')
    e('        };')
    e('      }')
    e('    }')
    e('  }')
    e('  return;')
    e('}')


def merge():
    """`vec_attention_merge` for query t0 + t, t below n, with its parts at part[h][w][132 (t0 + t) ..]."""
    e('')
    e('/// `vec_attention_merge` for queries t0 to t0 + n - 1, whose parts are at part[h][w][132 t ..]: out[t][h]')
    e('/// gets the attention of head h of query t.')
    e('fn round_attention_merge(out : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, part : &Tensor<f32, [?, ?, ?], Memory::HBM>, parts : i32, t0 : i32, n : i32, q_heads : i32) -> void {')
    e('  spawn on(Topology::Device) {')
    e('    for i in 0..n across blocks {')
    e('      for h in 0..q_heads across blocks {')
    e('        let t = t0 + i;')
    e('        let _d = mlir!( threads : 128, inputs : (%out = out : memref<?x?x?xf32>, %part = part : memref<?x?x?xf32>, %parts = parts : i32, %h = h : i32, %t = t : i32), returns : i32, dialects : ["arith", "memref", "scf", "gpu", "vector", "math"], ensures : [reads_only(part), writes_only(out[t][h])]) {')
    b = lambda s: e('          ' + s)
    for c in [0, 1, 2, 3, 4, 16, 32, 64, 96, 128, 129, 132]:
        b(f'%c{c} = arith.constant {c} : index')
    b('%neg = arith.constant -3.0e38 : f32')
    b('%zero = arith.constant 0.0 : f32')
    b('%one = arith.constant 1.0 : f32')
    b('%tid = gpu.thread_id x')
    b('%sg = arith.divui %tid, %c32 : index')
    b('%l = arith.remui %tid, %c32 : index')
    b('%hi = arith.index_cast %h : i32 to index')
    b('%ti = arith.index_cast %t : i32 to index')
    b('%base = arith.muli %ti, %c132 : index')
    b('%bm = arith.addi %base, %c128 : index')
    b('%bs = arith.addi %base, %c129 : index')
    b('%np = arith.index_cast %parts : i32 to index')
    b('%last = arith.subi %np, %c1 : index')
    for j in range(4):
        b(f'%p{j} = arith.addi %l, %c{32 * j} : index')
        b(f'%live{j} = arith.cmpi ult, %p{j}, %np : index')
        b(f'%pc{j} = arith.minui %p{j}, %last : index')
        b(f'%mr{j} = memref.load %part[%hi, %pc{j}, %bm] : memref<?x?x?xf32>')
        b(f'%m{j} = arith.select %live{j}, %mr{j}, %neg : f32')
    b('%m01 = arith.maxnumf %m0, %m1 : f32')
    b('%m23 = arith.maxnumf %m2, %m3 : f32')
    b('%mx = arith.maxnumf %m01, %m23 : f32')
    b('%big = gpu.subgroup_reduce maxnumf %mx : (f32) -> f32')
    for j in range(4):
        b(f'%dm{j} = arith.subf %m{j}, %big : f32')
        b(f'%w{j} = math.exp %dm{j} : f32')
        b(f'%s{j} = memref.load %part[%hi, %pc{j}, %bs] : memref<?x?x?xf32>')
        b(f'%ws{j} = arith.mulf %w{j}, %s{j} : f32')
        b(f'%wv{j} = vector.broadcast %w{j} : f32 to vector<4xf32>')
    b('%t01 = arith.addf %ws0, %ws1 : f32')
    b('%t23 = arith.addf %ws2, %ws3 : f32')
    b('%ts = arith.addf %t01, %t23 : f32')
    b('%total = gpu.subgroup_reduce add %ts : (f32) -> f32')
    b('%inv = arith.divf %one, %total : f32')
    b('%lane0 = arith.cmpi eq, %l, %c0 : index')
    b('%sg4 = arith.muli %sg, %c4 : index')
    b('%vz4 = arith.constant dense<0.0> : vector<4xf32>')
    b('scf.for %n = %c0 to %c128 step %c16 {')
    c = lambda s: e('            ' + s)
    c('%d = arith.addi %n, %sg4 : index')
    c('%pd = arith.addi %base, %d : index')
    acc = '%vz4'
    for j in range(4):
        c(f'%x{j} = vector.load %part[%hi, %pc{j}, %pd] {{alignment = 16}} : memref<?x?x?xf32>, vector<4xf32>')
        c(f'%a{j} = vector.fma %x{j}, %wv{j}, {acc} : vector<4xf32>')
        acc = f'%a{j}'
    for i in range(4):
        c(f'%e{i} = vector.extract {acc}[{i}] : f32 from vector<4xf32>')
        c(f'%r{i} = gpu.subgroup_reduce add %e{i} : (f32) -> f32')
        c(f'%o{i} = arith.mulf %r{i}, %inv : f32')
    c('scf.if %lane0 {')
    for i in range(4):
        c(f'  %d{i} = arith.addi %d, %c{i} : index')
        c(f'  memref.store %o{i}, %out[%ti, %hi, %d{i}] : memref<?x?x?xf32>')
    c('}')
    b('}')
    b('macro.yield %h : i32')
    e('        };')
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
merge()
print('\n'.join(L))
