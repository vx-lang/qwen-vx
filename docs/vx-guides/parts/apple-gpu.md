# Code on the GPU of a Mac

On a Mac, `Topology::Device` is the GPU, and `Memory::HBM` is the memory that it reads. Move a
tensor there with `transfer(t, Memory::HBM)`, and back with `transfer(t, Memory::CPU_DRAM)`.
Inside `spawn on(Topology::Device) { .. }`, loops with `across` run in parallel:

- `for y in 0..64 across blocks` runs each pass in a block of its own.
- `for x in 0..300 across lanes(256)` runs the passes on the 256 lanes (threads) of a block. A lane
  loop is inside a block loop. A block has at most 1024 lanes.
- Two passes must never write the same element: each index of a write uses the loop variables
  (for example `out[r][b * 256 + l]`), with only `+`, `-` and multiplication by a constant.
  Otherwise vxc gives error E6032.
- A `break`, `continue` or `return` cannot leave an `across` loop.
- Do not call `println!` inside a `spawn on` region.
- The float methods `sqrt`, `fast_exp`, `exp`, `max`, `min` and `abs` need `import core::num;`
  at the start of the file. `f64::from(x)` needs `import core::convert;`.

A complete GPU program. It prints `corner 2, last row 65`:

```rust
fn main() -> i32 {
  let mut grid = transfer(Tensor<f32, [64, 300]>::fill(1.0), Memory::HBM);
  spawn on(Topology::Device) {
    for y in 0..64 across blocks {
      for x in 0..300 across lanes(256) {
        grid[y][x] = grid[y][x] * 2.0 + (y as f32);
      }
    }
  }
  let back = transfer(grid, Memory::CPU_DRAM);
  println!("corner {}, last row {}", back[0][0], back[63][299]);
  return 0;
}
```

A GPU function with a size known only at run time: `out[r][i] = silu(gu[r][i]) * gu[r][rows + i]`.

```rust
fn silu_mul(out : &mut Tensor<f32, [?, ?], Memory::HBM>, gu : &Tensor<f32, [?, ?], Memory::HBM>, n : i32, rows : i32) -> void {
  spawn on(Topology::Device) {
    for r in 0..n across blocks {
      for b in 0..(rows + 255) / 256 across blocks {
        for l in 0..256 across lanes {
          let i = b * 256 + l;
          if i < rows {
            let g = gu[r][i];
            out[r][i] = g / (1.0 + (0.0 - g).fast_exp()) * gu[r][rows + i];
          }
        }
      }
    }
  }
  return;
}
```

A tile in shared memory, `Memory::SMEM`, is fast memory that the lanes of one block share. Make it
in the block loop and outside the lane loops. The end of each lane loop waits for all the lanes of
the block, so the next lane loop can read what any lane wrote. `lanes::sum(&part, 1024)` adds the
1024 values of a tile. This function is an RMS norm of each row:

```rust
fn rms_norm(y : &mut Tensor<f32, [?, ?], Memory::HBM>, x : &Tensor<f32, [?, ?], Memory::HBM>, n : i32, dim : i32, eps : f32) -> void {
  spawn on(Topology::Device) {
    for t in 0..n across blocks {
      let mut part = unsafe { Tensor<f32, [1024], Memory::SMEM>::uninit() };
      for l in 0..1024 across lanes {
        let mut s : f32 = 0.0;
        let mut i = l;
        while i < dim {
          s = s + x[t][i] * x[t][i];
          i = i + 1024;
        }
        part[l] = s;
      }
      let ss = lanes::sum(&part, 1024);
      let inv = 1.0 / (ss / (dim as f32) + eps).sqrt();
      for l in 0..1024 across lanes {
        for j in 0..(dim + 1023) / 1024 {
          let i = j * 1024 + l;
          if i < dim {
            y[t][i] = x[t][i] * inv;
          }
        }
      }
    }
  }
  return;
}
```
