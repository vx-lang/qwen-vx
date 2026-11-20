# Prints pf_dn_recur of src/prefill35.vx: the DeltaNet state update over the tokens of a chunk, with
# 8 values of v for each SIMD group and the 32 values of S of each lane in variables, so that they
# stay in registers. Run: python3 scripts/deltanet_chunk.py, and put the output in src/prefill35.vx.
P = 8
L = []
p = L.append
p("/// The update of the state and the output q . S for the n tokens of the chunk, in order. Block (h, vq)")
p("/// is one SIMD group for values vq * 8 to vq * 8 + 7 of v of head h. Lane l owns values 4 l to 4 l + 3")
p("/// of k, and keeps its 32 values of S in registers for the whole chunk: S is read and written once.")
p("/// For each token and value of v, the operations are those of `deltanet_recur`, in the same order.")
p("/// o[t][h][vq][m] is the output of value vq * 8 + m.")
p("fn pf_dn_recur(o : &mut Tensor<f32, [?, ?, ?, ?], Memory::HBM>, state : &mut Tensor<f32, [?, ?, ?], Memory::HBM>, qn : &Tensor<f32, [?, ?, ?], Memory::HBM>, kn : &Tensor<f32, [?, ?, ?], Memory::HBM>, vv : &Tensor<f32, [?, ?, ?], Memory::HBM>, beta : &Tensor<f32, [?, ?], Memory::HBM>, decay : &Tensor<f32, [?, ?], Memory::HBM>, group : i32, n : i32, v_heads : i32) -> void {")
p("  spawn on(Topology::Device) {")
p("    for h in 0..v_heads across blocks {")
p("      for vq in 0..16 across blocks {")
p("        for s in 0..1 across subgroups {")
p("          for l in 0..32 across lanes(32) {")
p("            let g = h / group;")
p("            let d = 4 * l;")
for m in range(P):
    for j in range(4):
        p(f"            let mut s{m}_{j} = state[h][vq * 8 + {m}][d + {j}];")
p("            for t in 0..n {")
for j in range(4):
    p(f"              let q{j} = qn[t][g][d + {j}];")
for j in range(4):
    p(f"              let k{j} = kn[t][g][d + {j}];")
p("              let b = beta[t][h];")
p("              let dec = decay[t][h];")
p("              let mut mine : f32 = 0.0;")
for m in range(P):
    for j in range(4):
        p(f"              let a{m}_{j} = s{m}_{j} * dec;")
    p(f"              let kv{m} = simd_sum(a{m}_0 * k0 + a{m}_1 * k1 + a{m}_2 * k2 + a{m}_3 * k3);")
    p(f"              let delta{m} = (vv[t][h][vq * 8 + {m}] - kv{m}) * b;")
    for j in range(4):
        p(f"              s{m}_{j} = a{m}_{j} + k{j} * delta{m};")
    p(f"              let out{m} = simd_sum(s{m}_0 * q0 + s{m}_1 * q1 + s{m}_2 * q2 + s{m}_3 * q3);")
    p(f"              if l == {m} {{")
    p(f"                mine = out{m};")
    p("              }")
p("              if l < 8 {")
p("                o[t][h][vq][l] = mine;")
p("              }")
p("            }")
for m in range(P):
    for j in range(4):
        p(f"            state[h][vq * 8 + {m}][d + {j}] = s{m}_{j};")
p("          }")
p("        }")
p("      }")
p("    }")
p("  }")
p("  return;")
p("}")
print("\n".join(L))
