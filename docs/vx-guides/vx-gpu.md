You write programs in Vx, a new programming language. Vx looks like Rust, but it is not Rust, and
Rust that is almost right does not compile. Follow the rules in this guide exactly:

- Always write the return type of a function, `-> void` when there is no result.
- `fn main() -> i32` ends with `return 0;`.
- `import`, not `use`. Write the type arguments of generic types: `Vec<i32>::new()`,
  `Option<i64>::Some(3)`, `Option<i64>::None()`.
- Each `match` arm body is a block: `X => { .. },`. There is no `if let` and no `while let`.
- Closures write the types of their parameters: `| x : i64 | x + 1`.
- Print with `println!("{} {}", a, b)`.
- Write `let mut total : i64 = 0;` when a variable is not an `i32`.

Give a complete program in one ```rust block. The user compiles and runs it on a Mac with Apple silicon:

    ~/.vx/bin/vxc program.vx --host default --machine ~/.vx/current/fleet/m4-uma.vx

When the user gives you an error from vxc, read it carefully, and give the whole corrected program.

# Vx for Rust programmers

Vx looks like Rust, so Rust that almost compiles is the first Vx most people write. This chapter
lists what to change. Each item shows the Rust form and the Vx form.

## Modules and items

- **`import`, not `use`.** `use core::option;` is `import core::option;`. An import brings in
  everything the module declares.
- **`pub` exports a symbol.** Every top-level item is visible to a module that imports it, with or
  without `pub`. A `pub fn` is also exported from the object file, so C can call it.
- **No attributes.** There is no `#[derive(..)]`, `#[test]` or `#[inline]`.

## Functions and methods

- **The return type is always written.** `fn log(x : i32)` is `fn log(x : i32) -> void`, and
  `main` returns `i32`: `fn main() -> i32 { .. return 0; }`.
- **`self` has a type.** `fn get(&self)` is `fn get(self : &Self)`, `&mut self` is
  `self : &mut Self`, and `self` is `self : Self`. The type's own name works too:
  `self : &Point`.
- **`#[track_caller]` is a parameter.** End the parameters with `at : Caller`
  (`import core::panic;`). A call that leaves the argument out passes its own place, in
  `at.location`: a `Location` with `file`, `line` and `column`. To report the place of your own
  caller, give your `at` to the call. `Location::here()` is Rust's `Location::caller()` and
  `file!()`, `line!()` and `column!()` in one.
- **Closure parameters have types.** `|x| x + 1` is `| x : i64 | x + 1`.
- **No `impl Trait` parameters.** `fn show(g : &impl Area)` is `fn show<T : Area>(g : &T)`.

## Types and values

- **No turbofish.** `Vec::<i32>::new()` is `Vec<i32>::new()`.
- **An enum variant is written with its type.** `Some(3)` is `Option<i64>::Some(3)`, and `None` is
  `Option<i64>::None()`. `Option` comes from `import core::option;`.
- **An integer variable with no written type is `i32`.** Rust reads `let mut total = 0;` as an
  `i64` when `i64`s are added to it later. Vx does not, so write `let mut total : i64 = 0;`.
- **Arrays are tensors.** `[i32; 4]` is `Tensor<i32, [4]>`. The shape is part of the type, and a
  `?` marks a size known only at run time: `Tensor<f32, [?, ?]>`.
- **The atomic orders are `AtomicOrdering`**, because `Ordering` is the result of a comparison.
  A `Relaxed` read is compiled as an `Acquire` read, so no value comes out of thin air.
  `-fweak-relaxed-atomics` gives Rust's `Relaxed`. See
  Memory orders.
- **Uninitialized storage is `unsafe`.** `MaybeUninit` has no equivalent. Write
  `unsafe { Tensor<f32, [4]>::uninit() }` and set every element before reading it, or use
  `Tensor<f32, [4]>::new()`, which is zeroed.

## Control flow

- **A `match` arm's body is a block.** `0 => 7,` is `0 => { 7 },`.
- **No `if let` or `while let`.** Use `match`.
- **`?` does not convert the error.** `x?` passes the `Err` up unchanged, so its type must be
  the error type of the function. Use `.map_err(..)` to convert it. `?` takes only a `Result`.
- **`?` in a `tasks` block fails the block**, not the function. The block gives that `Err`, and it
  cancels its tasks. In a task, `?` ends the task, and the task fails. In a closure, `?` needs
  the closure's type written: `|x : i32| -> Result<i32, E> { .. }`.

## Printing

- `print(x)` prints one value with no newline.
- `print!("text")` prints text. `println!("{} and {}", a, b)` puts each value in place of a `{}`
  and ends the line. `eprint!` and `eprintln!` do the same on standard error.
- There is no function called `println`: `println(x)` is an error. Write `println!("{}", x)`.

## What Rust does not have

Rust has one kind of memory. Vx describes several, and checks where data lives:

- A tensor's third type argument is the memory it lives in:
  `Tensor<f32, [4, 4], Memory::NPU_HBM>`.
- `transfer(t, Memory::NPU_HBM)` moves data to another memory. It is required even when the move
  costs nothing on the hardware.
- `spawn on(Topology::NPU[0]) { .. }` runs a block on a device. Everything the block uses must
  already be in memory that device can reach.
- `unsafe` allows raw pointers and calls to `unsafe fn`, as in Rust. It does not allow reading
  device memory from the host: that stays a compile error.

Topologies and memory explains these.

## A complete program

Most of the differences above, in one program:

```rust
import core::iter;
import core::option;

trait Area {
    fn area(self : &Self) -> i64;
}

struct Rect {
    w : i64,
    h : i64,
}

impl Rect {
    fn grow(self : &mut Rect, by : i64) -> void {
        self.w = self.w + by;
        self.h = self.h + by;
    }
}

impl Area for Rect {
    fn area(self : &Rect) -> i64 {
        return self.w * self.h;
    }
}

fn total<T : Area>(a : &T, b : &T) -> i64 {
    return a.area() + b.area();
}

fn half(n : i64) -> Option<i64> {
    if n % 2 == 0 {
        return Option<i64>::Some(n / 2);
    }
    return Option<i64>::None();
}

// Rust would write `half(half(n)?)`.
fn quarter(n : i64) -> Option<i64> {
    match half(n) {
        Option<i64>::Some(h) => { return half(h); },
        Option<i64>::None() => { return Option<i64>::None(); }
    }
}

fn main() -> i32 {
    let mut r = Rect { w : 2, h : 3 };
    r.grow(1);
    let s = Rect { w : 1, h : 1 };
    println!("area {}", total(&r, &s));

    let squares = range(1i64, 4i64).map(| x : i64 | x * x).sum();
    println!("squares {}", squares);

    match quarter(12) {
        Option<i64>::Some(q) => { println!("quarter {}", q); },
        Option<i64>::None() => { println!("no quarter"); }
    }

    let mut t : Tensor<f32, [3]> = unsafe { Tensor<f32, [3]>::uninit() };
    for i in 0..3 {
        t[i] = 0.5 * (i as f32);
    }
    println!("last {}", t[2]);
    return 0;
}
```

It prints:

```text
area 13
squares 14
quarter 3
last 1
```

# Vx in one page

A short reference for someone who already programs: the shape of a program, the types, the
statements, and three complete programs with what they print. A tour of Vx explains
each part at length, and Vx for Rust programmers lists what to change
when writing Vx with Rust habits.

## A program

- One file is one module. `import core::option;` brings in everything that module declares.
- `fn main() -> i32` is where the program starts. Its result is the exit code.
- Comments are `//` to the end of the line, and `///` for a doc comment.
- Statements end in `;`. Blocks are `{ .. }`, and indentation means nothing.

## Declarations

- `const N : i64 = 4;` is a value known while compiling. The type is required.
- `fn add(a : i64, b : i64) -> i64 { .. }`. Parameter and return types are required, and a
  function with no result returns `void`.
- `struct Point { x : i64, y : i64 }`, built with `Point { x : 1, y : 2 }`.
- `enum Shape { Square(i64), Rect(i64, i64) }`, built with `Shape::Square(3)`.
- `trait Describe { fn size(self : &Self) -> i64; }`. A trait method can have a default body.
- `impl Point { .. }` and `impl Describe for Shape { .. }`. Each method writes its receiver's
  type: `self : &Self`, `self : &mut Self` or `self : Self`. Inside an impl, `Self` is the impl's
  type.
- Generics: `struct Pair<T>`, `fn largest<T : Describe>(a : &T) -> i64`, `impl<T> Pair<T>`. A
  generic type is always written with its arguments: `Pair<i64> { .. }`, `Vec<i64>::new()`,
  `Option<i64>::Some(3)`.
- Const generics: `fn dot<const N : i32>(a : Tensor<f32, [N]>) -> f32`.
- `macro_rules name { ($x : expr) => { .. }; }`, called as `name!(..)`.
- `extern "C" { fn abs(x : i32) -> i32; }` declares a C function.

## Types

- Integers `i4` `i8` `i16` `i32` `i64`, `u4` `u8` `u16` `u32` `u64`, `usize`; floats `f16` `bf16`
  `f32` `f64`; `bool`. The fp8 and fp4 types are in Low-precision numbers.
  Nothing converts by itself: write `x as i64`. An integer variable with no written type is `i32`.
- An `as` cast to a narrower integer type wraps: `300 as u8` is 44. A `usize` holds 0 up to the
  largest `i64`. A cast to `usize` never wraps. It stops the program when the value does not fit,
  for example `(-1i64) as usize`.
- Tuples `(i64, f32)`, read as `t.0` or taken apart with `let (a, b) = t;`.
- References `&T` and `&mut T`, checked by the borrow checker; raw pointers `*const T` and
  `*mut T`, used inside `unsafe`.
- `Tensor<f32, [2, 3]>`: the shape is part of the type. `?` is a size known only at run time,
  `Tensor<f32, [?, ?]>`, read with `t.extent(0)`. A third argument names the memory it lives in:
  `Tensor<f32, [4], Memory::NPU_HBM>`. Arrays are tensors: `let a : Tensor<f32, [3]> = [1.0, 2.0, 3.0];`.
- `Option<T>` and `Result<T, E>` (`import core::option;`, `import core::result;`), `Vec<T>`
  (`import std::vec;`), `Box<T>` (`import std::box;`).
- A function that never returns has the type `-> !`, like `panic("..")`.
- A last parameter `at : Caller` (`import core::panic;`) gets the place of each call that leaves
  it out: `at.location.file`, `.line` and `.column`. `Location::here()` is the place of that call.
  `panic(msg, at)` reports that place instead of its own.

## Statements and expressions

- `let x = 1;` and `let mut y : i64 = 2;`. Variables are read-only unless `mut`.
- `if c { .. } else { .. }`, which is also an expression: `if c { 1 } else { 2 }`.
- `for i in 0..n { .. }` (`n` not included), `while c { .. }`, `loop { .. }`, with `break`
  and `continue`.
- `match v { Shape::Square(s) => { .. }, _ => { .. } }`. Each arm's body is a block, and the
  last arm must cover every value that is left.
- `return x;`, or a final expression with no `;`.
- `let v = parse(s)?;` gives the value of an `Ok`, or returns the `Err` unchanged.
- Arithmetic `+ - * / %`, comparisons `== != < <= > >=`, `&&` `||` `!` on `bool`, and `+=`,
  `-=`, `*=` and the like.
- Closures: `| x : i64 | x * 2`. Parameter types are required.
- `unsafe { .. }` allows raw pointer use, calls to `unsafe fn`, and uninitialized tensors:
  `unsafe { Tensor<f32, [4]>::uninit() }`.

## Printing and stopping

- `print(x)` prints one value. `print!("text")` prints text. `println!("{} and {}", a, b)` puts
  each value in place of a `{}` and ends the line.
- `eprint!` and `eprintln!` do the same on standard error.
- `panic("message")` stops the program with a message on standard error, and `abort()` without
  one.

## Memory and devices

Code runs on a topology, and data lives in a memory. Moving data is written out, and a device
block can only use data in memory that device can reach:

```rust
fn main() -> i32 {
    let host : Tensor<f32, [4]> = Tensor<f32, [4]>::new();
    let mut device = transfer(host, Memory::NPU_HBM);
    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            device[i] += 1.0;
        }
    }
    return 0;
}
```

Topologies and memory explains the checks, and
Machine files how a machine is described.

## The standard library

- `core::iter`: `range(a, b)` and the iterator methods `map`, `filter`, `zip`, `enumerate`,
  `chain`, `take`, `skip`, `rev`, `fold`, `sum`, `count`, `min`, `max`, `any`, `all`, `find`,
  `nth`, `collect` and more.
- `core::option`, `core::result`, `core::cmp`, `core::clone`, `core::convert`, `core::panic`.
- `std::vec` (`Vec<T>`: `push`, `pop`, `get`, `set`, `len`, `iter`), `std::box`, `std::io`,
  `std::fs`, `std::tensor`.

## Three programs

Declarations, traits, enums and `Option`:

```rust
import core::option;

const SIDES : i64 = 4;

struct Pair<T> {
  first : T,
  second : T,
}

enum Shape {
  Square(i64),
  Rect(i64, i64),
}

trait Describe {
  fn size(self : &Self) -> i64;

  // A default method: an impl that does not write it gets this one.
  fn doubled(self : &Self) -> i64 {
    return self.size() * 2;
  }
}

impl Describe for Shape {
  fn size(self : &Shape) -> i64 {
    match *self {
      Shape::Square(s) => {
        return s * s;
      },
      Shape::Rect(w, h) => {
        return w * h;
      }
    }
  }
}

impl<T> Pair<T> {
  fn swap(self : Pair<T>) -> Pair<T> {
    return Pair<T> {
      first : self.second, second : self.first
    };
  }
}

fn largest<T : Describe>(a : &T, b : &T) -> i64 {
  let x = a.size();
  let y = b.size();
  return if x > y {
    x
  } else {
    y
  };
}

fn divmod(a : i64, b : i64) -> (i64, i64) {
  return (a / b, a % b);
}

fn first_even(a : i64, b : i64) -> Option<i64> {
  if a % 2 == 0 {
    return Option<i64>::Some(a);
  }
  if b % 2 == 0 {
    return Option<i64>::Some(b);
  }
  return Option<i64>::None();
}

fn main() -> i32 {
  let sq : Shape = Shape::Square(3);
  let r : Shape = Shape::Rect(2, SIDES);
  println!("{} {} {}", sq.size(), r.doubled(), largest(&sq, &r));

  let p = Pair<i64> {
    first : 1, second : 2
  }.swap();
  let (q, rem) = divmod(17, 5);
  println!("{} {} {} {}", p.first, p.second, q, rem);

  match first_even(3, 8) {
    Option<i64>::Some(v) => {
      println!("even {}", v);
    },
    Option<i64>::None() => {
      println!("none");
    }
  }
  return 0;
}
```

It prints:

```text
9 16 9
2 1 3 2
even 8
```

Tensors with a fixed shape and with a shape known only at run time:

```rust
// A shape known while compiling is checked while compiling.
fn dot<const N : i32>(a : Tensor<f32, [N]>, b : Tensor<f32, [N]>) -> f32 {
  let mut acc = 0.0;
  for i in 0..N {
    acc += a[i] * b[i];
  }
  return acc;
}

// `?` is a size known only when the program runs.
fn matmul(a : &Tensor<f32, [?, ?]>, b : &Tensor<f32, [?, ?]>) -> Tensor<f32, [?, ?]> {
  let rows = a.extent(0);
  let cols = b.extent(1);
  let mut c : Tensor<f32, [?, ?]> = unsafe {
    Tensor<f32, [?, ?]>::uninit([rows, cols])
  };
  for i in 0..rows {
    for j in 0..cols {
      c[i][j] = 0.0;
      for k in 0..a.extent(1) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

fn main() -> i32 {
  let a : Tensor<f32, [3]> = [1.0, 2.0, 3.0];
  let b : Tensor<f32, [3]> = [4.0, 5.0, 6.0];
  println!("dot {}", dot(a, b));

  let m_flat : Tensor<f32, [4]> = [1.0, 2.0, 3.0, 4.0];
  let id_flat : Tensor<f32, [4]> = [1.0, 0.0, 0.0, 1.0];
  let m : Tensor<f32, [2, 2]> = m_flat.reshape([2, 2]);
  let id : Tensor<f32, [2, 2]> = id_flat.reshape([2, 2]);
  let p = matmul(&m, &id);
  println!("{} {} {} {}", p[0][0], p[0][1], p[1][0], p[1][1]);

  let z : Tensor<i32, [2, 3]> = Tensor<i32, [2, 3]>::new();
  let flat : Tensor<i32, [6]> = z.reshape([6]);
  println!("zeros {}", flat[5]);
  return 0;
}
```

It prints:

```text
dot 32
1 2 3 4
zeros 0
```

`Vec`, iterators and `Result`:

```rust
import core::iter;
import core::option;
import core::result;
import std::vec;

fn parse_digit(c : i64) -> Result<i64, i64> {
  if c >= 48 && c <= 57 {
    return Result<i64, i64>::Ok(c - 48);
  }
  return Result<i64, i64>::Err(c);
}

fn main() -> i32 {
  let mut v = Vec<i32>::new();
  for i in 1..6 {
    v.push(i * i);
  }
  let mut total = 0;
  for x in v.iter() {
    total += *x;
  }
  println!("len {} total {}", v.len(), total);

  let odd_squares = range(1i64, 10i64).filter(| x : &i64 | *x % 2 == 1).map(| x : i64 | x * x).sum();
  println!("odd squares {}", odd_squares);

  match parse_digit(55) {
    Result<i64, i64>::Ok(d) => {
      println!("digit {}", d);
    },
    Result<i64, i64>::Err(c) => {
      println!("not a digit: {}", c);
    }
  }
  match v.pop() {
    Option<i64>::Some(last) => {
      println!("popped {}", last);
    },
    Option<i64>::None() => {
      println!("empty");
    }
  }
  return 0;
}
```

It prints:

```text
len 5 total 55
odd squares 165
digit 7
popped 25
```

# Topologies and memory

This is the chapter that makes Vx different from other systems languages. Everything up to here you
could have done in Rust or C++; none of it needed a new language.

## The two vocabularies

Vx describes a machine with two kinds of declaration, and they answer different questions:

- **`Memory`** — *where data lives.* `Memory::CPU_DRAM`, `Memory::NPU_HBM`, a scratchpad, a cache
  level. Each has a capacity, a bandwidth, and a scope.
- **`Topology`** — *where code runs.* `Topology::CPU`, `Topology::NPU[0]`, `Topology::GPU`. A
  topology is bound to the memory it can address.

A tensor's type records which memory space it is in. A region of code records which topology it runs
on. The checker's job is to make sure those two agree everywhere.

## Placement and transfer

Moving data between memory spaces is explicit:

```rust
let a = transfer(a_host, Memory::NPU_HBM);
let b = transfer(b_host, Memory::NPU_HBM);
```

`transfer` yields a value whose type says it lives in `NPU_HBM`. The original is still typed as
living where it was.

The explicitness is the point, and it is required **even when the hardware boundary costs nothing**.
On Apple's unified memory the CPU and the GPU address the same physical DRAM, so the copy compiles
away to nearly nothing — and you still write it. The reason is that data locality should be provable
by reading the source, not by profiling the binary. A `transfer` you cannot find in the text is a
transfer you cannot reason about.

Two things the compiler checks here:

- **Reachability** — there must be a declared path between the two spaces. A transfer between
  memories with no route between them is an error, not a runtime hang.
- **Admission** — the destination must have room. This is checked against the
  machine file, before any binary exists.

## Running code somewhere else

`spawn on` runs a block on a named topology:

```rust
fn main() -> i32 {
    let mut host : Tensor<f32, [4, 4]> = unsafe { Tensor<f32, [4, 4]>::uninit() };
    for i in 0..4 {
        for j in 0..4 {
            host[i][j] = 1.0;
        }
    }

    let mut device = transfer(host, Memory::NPU_HBM);

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                device[i][j] += 1.0;
            }
        }
    }

    return 0;
}
```

`Tensor<f32, [4, 4]>::uninit()` takes no arguments: the shape is already part of the type. Only the
dynamic form needs extents passed — `Tensor<f32, [?, ?]>::uninit([rows, cols])`.

The block is outlined into a kernel and handed to the dispatcher for that topology. On Apple Silicon
that means CoreML and the neural engine; on an NVIDIA box it means PTX. The source does not change
between the two — the machine file does.

Every value the block touches must already live in a memory the target topology can address. That is
why the `transfer` comes first. Skip it, and the error names the value and the space it is in rather
than crashing inside a vendor runtime.

The block also calls no helper function. A function declared without a topology belongs to the host,
and calling it from inside a device region is a compile error:

```
Error[E6001]: Type error: Function 'f' requires topology 'CPU', but is called from 'ANE'
```

That is the address-space rule doing its job. Code meant for a device is written in the region, or
in a function declared for that topology.

`spawn on` is a statement. To start a copy and do other work before it ends, use
`transfer_async(x, M)`. It gives a token, and `.wait()` on the token gives the placed tensor:

```rust
let first = transfer_async(next_tile, Memory::W);
// Other work here.
let tile = first.wait();
```

A program must wait for each token. A token that the program drops with no `.wait()` gets the
error E6036. Today the copy ends before `transfer_async` returns, so `.wait()` only gives the
tensor. A later runtime will run the copy in the background, and the program will not change.

A `spawn on` can also be in a task. So a `tasks` block can give work to several devices, one task
for each device (see Tasks and channels).

## Functions for any topology

The last section said that a device region cannot call a plain function. A plain function
belongs to the host. Often you want one helper for both: a clamp, a sum, a small formula. Vx
lets a function be generic over the topology that runs it, and over the memory of its data.

### One helper for host and device code

Write `on Topology::Any` after the parameters. Then the function runs on the topology of the code
that calls it. Each topology that calls it gets its own copy of the function.

`on Topology::Any` is short for a topology parameter that no argument binds. A topology
parameter is a generic parameter with the bound `Topology`, such as `<D : Topology>`. Inside the
function, `Topology::D` names it. So `fn f<D : Topology>(..) on Topology::D` means the same thing.
This program uses both forms:

```rust
fn clamp(v : i32, lo : i32, hi : i32) on Topology::Any -> i32 {
  if v < lo {
    return lo;
  }
  if v > hi {
    return hi;
  }
  return v;
}

fn add_one<D : Topology>(v : i32) on Topology::D -> i32 {
  return v + 1;
}

fn main() -> i32 {
  let a = spawn on(Topology::GPU) {
    clamp(12, 0, 5)
  };
  let b = spawn on(Topology::GPU) {
    add_one(12)
  };
  println!("gpu {} {} host {} {}", a, b, clamp(-3, 0, 5), add_one(-3));
  return 0;
}
```

It prints `gpu 5 13 host 0 -2`. With no GPU in the machine, the GPU region runs on the host, so
the program runs on any computer. Without `on Topology::Any`, the call `clamp(12, 0, 5)` in the
GPU region gets the error E6001.

`Topology::Any` is correct only after `on` in a function signature. A program cannot declare a
topology with the name `Any` (E6016).

### A function for any memory

A tensor's type records its memory. A parameter of type `&Tensor<i32, [?]>` names no memory, and
the host cannot give it a tensor in GPU memory (E3003). To accept a tensor in any memory, give the
function a memory parameter, such as `<M : Memory>`. Write the tensor type with `Memory::M` as
its memory:

```rust
fn sum_three<M : Memory>(t : &Tensor<i32, [?], Memory::M>) on Topology::Any -> i32 {
  return t[0] + t[1] + t[2];
}

fn main() -> i32 {
  let h = Tensor<i32, [3]>::fill(2);
  let g = transfer(Tensor<i32, [3]>::fill(10), Memory::GPU_HBM);
  let mut out = transfer(Tensor<i32, [1]>::fill(0), Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    out[0] = sum_three(&g);
  }
  let o = transfer(out, Memory::CPU_DRAM);
  println!("host {} gpu {}", sum_three(&h), o[0]);
  return 0;
}
```

It prints `host 6 gpu 30`. Each call binds `M` to the memory of its tensor argument. Two
arguments that both name `Memory::M` must be in the same memory. A call must also give `M` a
memory through a tensor argument (E3016).

The memory parameter does not change where the code runs. The host still cannot read a tensor in
GPU memory. If `main` calls `sum_three(&g)`, the compiler reports E6003 and asks for a `transfer`.

### Your own topology

A program can declare a topology of its own with a `Topology` block:

- `memory:` names the default memory of the topology. Every `Topology` block must have it.
  The memory can be a new name.
- `visible:` lists the memories that its code can read. It defaults to the `memory:` space.
- Each `transfer A -> B : cost` line adds a route for data, in one direction only.

`Reachable<A, B>` is true when data can go from topology `A` to topology `B` on the declared
routes. In an `if comptime` statement, the compiler tests it while it compiles. It compiles only
the branch that applies. A topology variable such as `D` can be an argument:

```rust
Topology Acme {
  memory: Memory::AcmeSRAM,
  visible: [Memory::AcmeSRAM, Memory::CPU_DRAM],
  transfer Memory::CPU_DRAM -> Memory::AcmeSRAM : 40 sync,
}

fn host_can_reach<D : Topology>() on Topology::D -> i32 {
  if comptime Reachable<Topology::CPU, D> {
    return 1;
  }
  return 0;
}

fn main() -> i32 {
  let a = spawn on(Topology::Acme) {
    host_can_reach()
  };
  println!("host to acme {}", a);
  if comptime Reachable<Topology::Acme, Topology::CPU> {
    println!("acme to host: yes");
  } else {
    println!("acme to host: no");
  }
  if comptime Reachable<Topology::GPU, Topology::Acme> {
    println!("gpu to acme: yes");
  } else {
    println!("gpu to acme: no");
  }
  return 0;
}
```

It prints:

```text
host to acme 1
acme to host: no
gpu to acme: yes
```

The host can send data to `Acme`, but `Acme` has no route back. GPU data can go to `Acme`
through the host. If a topology has no route from the host, the compiler gives warning W1026.

### Asking for a route

A generic function that moves data between two topology parameters must say that a route exists.
Write `where Reachable<S, D>` after the return type. The compiler checks the clause at each call,
for the topologies of that call. A call with no route is a compile error. To give more than one
clause, separate them with commas.

```rust
fn fetch_sum<S : Topology, D : Topology>(x : Pinned<Tensor<i32, [2]>, Topology::S>) on Topology::D -> i32 where Reachable<S, D> {
  let here = transfer(x, Memory::CPU_DRAM);
  return here[0] + here[1];
}

fn main() -> i32 {
  let g : Pinned<Tensor<i32, [2]>, Topology::GPU> = transfer(Tensor<i32, [2]>::fill(7), Memory::GPU_HBM);
  println!("sum {}", fetch_sum(g));
  return 0;
}
```

It prints `sum 14`. The argument binds `S` to the GPU, and the caller `main` binds `D` to the
host. If `x` is in `Acme` memory, the call fails with
`` unsatisfied `where Reachable<S, D>` in call to 'fetch_sum' ``.

Name only topology parameters in a `where` clause. The compiler does not check a clause that names
anything else yet ([Vx#3230](https://github.com/vx-lang/Vx/issues/3230)).

## What gets rejected

The point of putting placement in the type system is the errors you get for free.

**Dereferencing a device pointer from the host.** A `Pinned<T, Topology::NPU>` that escapes into a host
expression is a type error with a source span. This is the error that motivates the whole design: in
C++ with CUDA it is a segfault, and in Python it is a silent wrong answer.

**Working set overflow.** A tile you place in a scratchpad that cannot hold it is rejected at compile
time, with the required and available figures in the diagnostic. See
machine files.

**Use-after-move.** Buffers are linear values. Consuming one and then reading it again is an error.

**Unvisible transfers.** An asynchronous `transfer` whose completion has not been made visible before
the buffer is read is a *seam* violation. Turn the check on with `--verify-seams`; it discharges the
obligation with an SMT solver, and needs `z3` on your `PATH`.

## Verified values

`Verified<T>` marks a value whose computation carried its proof obligations all the way through. A
function returning `Verified<Tensor>` is asserting that the placement, capacity and visibility
conditions on the path that produced it were all discharged, not merely unchecked.

```rust
fn custom_matmul(a : Pinned<Tensor<f32, [4, 4]>, Topology::NPU[0]>,
                 b : Pinned<Tensor<f32, [4, 4]>, Topology::NPU[0]>)
    -> Verified<Tensor<f32, [4, 4], Memory::NPU_HBM>> {

    let mut result = unsafe { Tensor<f32, [4, 4], Memory::NPU_HBM>::uninit() };

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0.0;
                for k in 0..4 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return Verified(result);
}
```

`Pinned<T, Topology>` says the value is resident on a particular device. Note the third type
argument on `Tensor` — the memory space it lives in — and that `Verified(...)` wraps the result
*after* the region, not inside it.

## Choosing a machine at compile time

The same program compiles and runs against different hardware by swapping the machine file:

```bash
vxc --machine fleet/h100-sxm.vx --host default program.vx
vxc --machine fleet/m4-uma.vx --host default program.vx
```

`--host default` says which machine runs `main` (Machine files).

You can ask what the compiler concluded, as JSON, rather than reading it out of diagnostics:

```bash
vxc --machine fleet/h100-sxm.vx --host default program.vx --diagnostics-json out.json
```

That record carries every diagnostic with its structured fields — a capacity rejection includes the
space, the amount required, the amount available and the margin — plus the staging routes and
per-edge costs that an admitted program resolved to.

## Next

Machine files covers how a real part gets described, and what the compiler
derives from that description.

# Parallel code on a GPU

A GPU runs one piece of code on many threads at the same time. Vx writes that code as plain loops
in a `spawn on` region. A word after the range of a loop, `across blocks` or `across lanes`, says
which level of the GPU runs its passes. The compiler checks that two passes cannot write the same
element, so the loops give the same result on the GPU as they do one pass at a time.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/optimizations/pass/` with the name that its section gives. The programs use the machine
file of an NVIDIA H100, so run them with `--machine fleet/h100-sxm.vx --host default`. On a computer with no GPU,
the runtime prints `no CUDA device` and runs each kernel on the CPU. The result is the same.

## Blocks and lanes

A GPU has two levels of threads. A block is a group of threads that runs on one part of the chip,
and a lane is one thread in a block. The lanes of a block can share a fast memory, and wait for each
other. Two blocks cannot.

`for y in 0..64 across blocks` runs each pass in a block of its own. `for x in 0..300 across lanes(256)` runs the passes on 256 lanes, so some lanes run two passes. This program doubles each
element of a tensor in GPU memory, and adds the number of its row, `y`
(`book_gpu_blocks_and_lanes.vx`):

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

It prints `corner 2, last row 65`.

The meaning of the program does not depend on `across`. Remove each `across`, and the loops run one
pass at a time with the same result. The words only tell the compiler where to run the passes.

The compiler checks these rules, and gives the error E6032 when a rule is broken:

- An `across` loop is in a `spawn on` region for a device that has blocks and lanes, such as an
  NVIDIA or an Apple GPU. Topologies and memory explains `spawn on`.
- The block loop is directly in the region. A lane loop is in the block loop.
- A block has at most 1024 lanes. The program writes their number as a constant, as in
  `lanes(256)`.
- A `break`, `continue` or `return` cannot leave an `across` loop, because each lane must get to
  the end of the loop.
- At most three block loops can be in each other, and at most three lane loops, because the grid
  and the block of a GPU have three dimensions.

## Two passes cannot write the same element

The passes of a block loop run at the same time, in any order. So each write in the loop must use
the variable of the loop in its index, so that each pass writes other elements. In `grid[y][x]`,
`y` is different in each pass, so no two passes write the same element. A write to `grid[0][x]`
does not compile:

```text
Error[E6032]: two passes of this block loop may write the same element of `grid`: an index of the
write must fix `y`
```

The same check refuses a read of an element that another pass writes, a write to a variable from
outside the loop, `println!`, and a call that can write through its argument.

## A tile in shared memory

`Memory::SMEM` is the fast memory of a block. A tensor in it, a tile, lives as long as the code of
the block that makes it. The code of a block is the code in the block loop and outside its lane
loops. The end of each lane loop waits for all the lanes of the block. So the next lane loop
can read what any lane wrote. Code between two lane loops runs once for each block.

`lanes::sum(&part, n)` adds the first `n` elements of a tile, with all the lanes of the block. This
program adds each row of a matrix in a block of its own (`book_gpu_row_sums_in_shared_memory.vx`):

```rust
fn main() -> i32 {
  let mut m = Tensor<f32, [8, 256]>::fill(0.0);
  for r in 0..8 {
    for c in 0..256 {
      m[r][c] = (r + 1) as f32;
    }
  }
  let rows = transfer(m, Memory::HBM);
  let mut sums = transfer(Tensor<f32, [8]>::fill(0.0), Memory::HBM);
  spawn on(Topology::Device) {
    for b in 0..8 across blocks {
      let mut part = unsafe { Tensor<f32, [256], Memory::SMEM>::uninit() };
      for t in 0..256 across lanes {
        part[t] = rows[b][t];
      }
      let s = lanes::sum(&part, 256);
      sums[b] = s;
    }
  }
  let back = transfer(sums, Memory::CPU_DRAM);
  println!("row 0: {}, row 7: {}", back[0], back[7]);
  return 0;
}
```

It prints `row 0: 256, row 7: 2048`.

Here each lane reads only the element that it wrote, so the compiler keeps `part` in registers.

The library has other routines of this kind. `lanes::max` gives the largest element, and
`lanes::argmax` its index. Write each call as the whole value of a `let`, in the code of a block.
`lanes::sum_unordered` lets the compiler choose the order of the adds, so the last bits of the sum
can differ from `lanes::sum`. `lanes::sort(&mut t, n)` sorts the first `n` elements of a tile of
`i64`, whose size is a power of two.

## Atomics on the elements of a tensor

Many passes can add to the same element with an atomic operation from `core::sync::atomic`. Each
lane here adds 1 to the bin of its value (`book_gpu_histogram_with_atomics.vx`):

```rust
import core::sync::atomic;

fn main() -> i32 {
  let mut v = Tensor<i32, [1024]>::fill(0);
  for i in 0..1024 {
    v[i] = (i * 7) % 8;
  }
  let values = transfer(v, Memory::HBM);
  let bins = transfer(Tensor<u32, [8]>::fill(0u32), Memory::HBM);
  spawn on(Topology::Device) {
    for b in 0..4 across blocks {
      for l in 0..256 across lanes {
        let x = values[b * 256 + l];
        let _old = atomic_fetch_add_u32(&bins, x, 1u32, AtomicOrdering::Relaxed);
      }
    }
  }
  let back = transfer(bins, Memory::CPU_DRAM);
  println!("bin 0: {}, bin 7: {}", back[0], back[7]);
  return 0;
}
```

It prints `bin 0: 128, bin 7: 128`.

The atomic operations are `atomic_load`, `atomic_store`, `atomic_swap`, `atomic_fetch_add`,
`atomic_fetch_sub`, `atomic_fetch_max`, `atomic_fetch_min` and `atomic_compare_and_swap`. Each name
ends with the type of the element, such as `atomic_fetch_add_u32`. Each one also takes a memory
order, such as `AtomicOrdering::Relaxed`, as the atomics of Tasks and channels do.

When a loop changes a tensor with atomics, each access to that tensor in the loop must be atomic:

```text
Error[E6032]: the passes of this block loop change `bins` with atomic operations, so every access
to it in the loop must be one
```

`scoped_fence(order, FenceScope::Block)` orders the memory operations of the lanes of one block.
`FenceScope::Device` orders them for the whole GPU, and `FenceScope::System` also for the CPU.

On an Apple GPU, the atomics work on `i32` and `u32` with `Relaxed` only.

## Copies in the background

`lanes::copy_async(&mut tile, &x, [i, j])` starts a copy of a part of `x` into `tile`, and gives a
token. The part starts at the element `[i, j]` of `x`, and has the shape of `tile`. `.wait()` on
the token waits until the copy ends.
Between the two, the lanes can work on another tile. So with two tiles, the next copy runs while
the lanes use the last one:

```rust
let mut tiles = unsafe { Tensor<f32, [2, 64, 64], Memory::SMEM>::uninit() };
for k in 0..steps / 2 {
  let even = lanes::copy_async(&mut tiles[0], &x, [0, k * 128]);
  let odd = lanes::copy_async(&mut tiles[1], &x, [0, k * 128 + 64]);
  even.wait();
  // The lanes read tiles[0] here, while the copy into tiles[1] goes on.
  odd.wait();
  // The lanes read tiles[1] here.
}
```

The token holds the borrow of the tile. So a read of the tile before its `.wait()` is a borrow
error, and a token with no `.wait()` gets the error E6036. On an H100 the copy uses TMA, and on an
older NVIDIA GPU `cp.async`. Today the copy does not work on tiles of `f64` or of integers.
`tests/optimizations/pass/lane_copies_on_the_cpu.vx` is a complete program.

## What runs where

- An NVIDIA GPU runs each feature of this chapter. The compiler generates PTX.
- An Apple GPU runs block and lane loops, shared memory, and atomics on `i32` and `u32`.
- Each kernel also has a copy for the CPU, which runs when there is no GPU. The threads of the host
  share the blocks. `VX_HOST_THREADS=n` sets their number.

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
