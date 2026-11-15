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

# Tensor operations

The module `std::tensor` gives a tensor the operations of numerical code. It has sums, statistics,
element-wise changes, reductions along an axis and a matrix product. A tour of
Vx shows how to make a tensor, read its shape, and `reshape` or `pad`
it. This chapter starts from there.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Sums of a 1-D tensor

A reduction turns many elements into one value. A sum is the most common one. Add
`import std::tensor;` to get these methods on a 1-D tensor of `f16`, `bf16`, `f32` or `f64`:

- `a.sum()` adds the elements, and `a.mean()` divides that sum by the length.
- `a.dot(&b)` multiplies the elements of `a` and `b` in pairs and adds the products.
- `a.norm()` is the length of `a` as a vector: the square root of `a.dot(&a)`.
- `a.max()` and `a.min()` give the largest and the smallest element. They skip a NaN.

This program calls each one (`book_tensor_ops_sums.vx`):

```rust
import std::tensor;

fn main() -> i32 {
  let a : Tensor<f32, [4]> = [ 1.0, -2.0, 3.0, 2.0 ];
  let b : Tensor<f32, [4]> = [ 2.0, 1.0, 0.5, -1.0 ];
  println!("sum {} dot {}", a.sum(), a.dot(&b));
  println!("max {} min {}", a.max(), a.min());
  println!("mean {} norm {}", a.mean(), a.norm());
  return 0;
}
```

It prints `sum 4 dot -0.5`, then `max 3 min -2`, then `mean 1 norm 4.2426405`.

The length can be fixed in the type, as in `Tensor<f32, [4]>`, or known only at run time, as in
`Tensor<f32, [?]>`. A row of a 2-D tensor is a 1-D tensor too, so `m[1].sum()` adds row 1.

These methods add the elements in an order that they choose, so that they can use vector
instructions. The answer can differ in the last bits from a plain loop that adds one element at a
time. The methods add `f16` or `bf16` elements in `f32`, and they give an `f32`. An `f16` total
above 65504 does not fit in an `f16`.

In earlier versions of Vx, the compiler had `dot`, `sum`, `max` and `min` as built-in functions.
Now they are methods. So `dot(q[i], k[j])` is now `q[i].dot(&k[j])`, and the program needs `import std::tensor;`.

## Statistics and softmax

`std::tensor` has these methods on a 1-D tensor too:

- `a.argmax()` and `a.argmin()` give the index of the largest or smallest element. They give an
  `Option<i32>`: `None` for an empty tensor. When two elements are equal, they give the first one.
- `a.variance()` is the mean of the squared distances from the mean. It divides by the length `n`,
  not `n - 1`. `a.std_dev()` is its square root.
- `a.softmax_inplace()` replaces each element `x` with `exp(x)` divided by the sum of `exp` over
  all the elements. After it, the elements are positive and their sum is 1.

`softmax_inplace` subtracts the largest element before it calls `exp`. The answer is the same, but
`exp(1000.0)` does not overflow. This program uses each method (`book_tensor_ops_statistics.vx`):

```rust
import std::tensor;

fn main() -> i32 {
  let scores : Tensor<f64, [5]> = [ 2.0, 4.0, 4.0, 1.0, 9.0 ];
  match scores.argmax() {
    Option<i32>::Some(i) => println!("argmax {}", i),
    Option<i32>::None => println!("argmax none"),
  }
  println!("argmin {}", scores.argmin().unwrap());
  println!("variance {} std_dev {}", scores.variance(), scores.std_dev());

  let mut logits : Tensor<f64, [3]> = [ 1000.0, 1001.0, 1002.0 ];
  logits.softmax_inplace();
  println!("softmax {} {} {}", logits[0], logits[1], logits[2]);

  let empty = Tensor<f64>([0]);
  println!("empty argmax is none: {}", empty.argmax().is_none());
  return 0;
}
```

It prints `argmax 4`, `argmin 3`, `variance 7.6 std_dev 2.756809750418044`, then
`softmax 0.09003057317038046 0.24472847105479764 0.6652409557748218` and
`empty argmax is none: true`. The last digits of the softmax line can differ on another machine,
because `exp` comes from the system's math library.

`softmax_inplace` changes the tensor, so the tensor must be `mut`. The other methods of this
section do not change the tensor.

## Element-wise operations in place

An element-wise operation does the same thing to each element. The methods of `std::tensor`
change the tensor in place, so they make no new tensor. A chain of them allocates no memory.

| Method | What it does to each element `y[i]` |
| --- | --- |
| `y.add_assign(&x)` | `y[i] + x[i]`. `sub_assign`, `mul_assign` and `div_assign` do the same with `-`, `*` and `/`. |
| `y.add_scalar(a)` | `y[i] + a` |
| `y.scale(a)` | `y[i] * a` |
| `y.scaled_add(a, &x)` | `y[i] + a * x[i]` |
| `y.clamp_inplace(lo, hi)` | holds it between `lo` and `hi` |
| `y.abs_inplace()`, `y.sqrt_inplace()` | its absolute value, its square root |
| `y.map_inplace(f)` | `f(y[i])`, for a closure `f` |

The `_assign` methods take the place of `+=` and the other compound operators, because Vx cannot
overload them on a tensor yet. Both tensors must have the same length. A different length stops
the program with an error message. This program uses them one after the other
(`book_tensor_ops_in_place.vx`):

```rust
import std::tensor;

fn show(t : &Tensor<f32, [?]>) -> i32 {
  for i in 0..t.len() {
    print!(" {}", t[i]);
  }
  println!("");
  return 0;
}

fn main() -> i32 {
  let mut y : Tensor<f32, [4]> = [ 1.0, 2.0, 3.0, 4.0 ];
  let x : Tensor<f32, [4]> = [ 10.0, 20.0, 30.0, 40.0 ];
  y.add_assign(&x);
  print!("add_assign");
  show(&y);
  y.scale(0.5);
  print!("scale");
  show(&y);
  y.scaled_add(-1.0, &x);
  print!("scaled_add");
  show(&y);
  y.clamp_inplace(-10.0, 0.0);
  print!("clamp_inplace");
  show(&y);
  y.map_inplace(| v : f32 | v * v);
  print!("map_inplace");
  show(&y);
  y.sqrt_inplace();
  print!("sqrt_inplace");
  show(&y);
  return 0;
}
```

It prints one line for each step, from `add_assign 11 22 33 44` to `sqrt_inplace 4.5 9 10 10`.

`map_inplace` calls the closure once for each element. When one of the named methods does the same
work, use it. It is faster.

## Your own reduction with `reduce`

`t.reduce(init, |acc, x| ..)` is a reduction that you write. It starts with `acc` equal to `init`.
For each element `x`, it calls the closure and makes the result the new `acc`. At the end, it gives
the last `acc`. `reduce` is part of the language, so it needs no import. It works on a tensor of
any rank and of any element type, integers too. The closure can read a variable from outside it.

This program finds a total, a largest value, a total of reduced prices, and a product
(`book_tensor_ops_reduce.vx`):

```rust
fn main() -> i32 {
  let prices : Tensor<f64, [4]> = [ 3.0, 7.5, 2.0, 5.5 ];
  let total = prices.reduce(0.0, | acc, x | acc + x);
  let most = prices.reduce(0.0, | acc, x | if x > acc {
    x
  } else {
    acc
  });
  let discount : f64 = 0.5;
  let sale = prices.reduce(0.0, | acc, x | acc + x * discount);
  println!("total {} most {} sale {}", total, most, sale);

  let counts : Tensor<i32, [2, 3]> = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];
  println!("product {}", counts.reduce(1, | acc, x | acc * x));
  return 0;
}
```

It prints `total 18 most 7.5 sale 9`, then `product 720`.

`init` must have the type of the elements, and the closure must give that type too. The compiler
refuses a program that breaks either rule. `reduce` can combine the elements in any order, as `sum` does. So use a closure
for which the order does not change the answer, such as `+`, `*` or a maximum.

## Matrices: reductions along an axis and `matmul`

A 2-D tensor of a float type has `sum`, `mean`, `max` and `min` over all its elements. It has the
element-wise methods of the table above, except `scaled_add` and `map_inplace`. It also has
reductions along an axis, as numpy's `axis=` argument does:

- `m.sum_axis(0)` reduces each column, so it gives one value for each column.
- `m.sum_axis(1)` reduces each row, so it gives one value for each row.
- `mean_axis`, `max_axis` and `min_axis` work in the same way.

Each one gives a new 1-D tensor. An axis other than 0 or 1 stops the program.

`a.matmul(&b)` is the matrix product for `f32` and `f64`. It makes a new tensor with the rows of
`a` and the columns of `b`. Element `[i][j]` is the sum over `k` of `a[i][k] * b[k][j]`. This
program uses both (`book_tensor_ops_matrices.vx`):

```rust
import std::tensor;

fn show(t : &Tensor<f64, [?]>) -> i32 {
  for i in 0..t.len() {
    print!(" {}", t[i]);
  }
  println!("");
  return 0;
}

fn main() -> i32 {
  let m : Tensor<f64, [2, 3]> = [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ];
  println!("sum {} max {}", m.sum(), m.max());
  print!("sum_axis(0)");
  show(&m.sum_axis(0));
  print!("sum_axis(1)");
  show(&m.sum_axis(1));
  print!("mean_axis(1)");
  show(&m.mean_axis(1));

  let k : Tensor<f64, [3, 2]> = [ [ 1.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ] ];
  let p = m.matmul(&k);
  println!("matmul is {} by {}", p.extent(0), p.extent(1));
  for i in 0..2 {
    println!("{} {}", p[i][0], p[i][1]);
  }
  return 0;
}
```

It prints `sum 21 max 6`, `sum_axis(0) 5 7 9`, `sum_axis(1) 6 15` and `mean_axis(1) 2 5`. Then it
prints `matmul is 2 by 2` and the two rows `4 5` and `10 11`.

The number of columns of `a` must be equal to the number of rows of `b`. `matmul` checks this when
the program runs, not when it compiles, even when the two shapes are in the types. A wrong shape
stops the program with the message
`matmul needs as many columns on the left as rows on the right`.

For code on a GPU, see Parallel code on a GPU.

# Low-precision numbers

Machine-learning hardware stores numbers in fewer bits than `f32`, so that more of them fit in
memory and more of them move per second. Vx has these types as part of the language. A machine
file lists the ones each device supports (`dtypes:`), and the compiler checks a program against
that list.

## The types

| Type | Bits | Values | Infinity | NaN | Status |
| :--- | ---: | :--- | :---: | :---: | :--- |
| `f16` | 16 | up to 65504, about 3 decimal digits | yes | yes | works |
| `bf16` | 16 | the range of `f32`, about 2 decimal digits | yes | yes | works |
| `f8e4m3` | 8 | up to 448 | no | yes | works, except matmul ([#1876]) |
| `f8e5m2` | 8 | up to 57344 | yes | yes | works, except matmul ([#1876]) |
| `f4e2m1` | 4 | 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives | no | no | works, except matmul ([#1876]) |
| `f8e8m0` | 8 | powers of two, 2⁻¹²⁷ to 2¹²⁷ | no | yes | planned ([#1877]), block scales only |
| `i4` | 4 | -8 to 7 | | | works |
| `u4` | 4 | 0 to 15 | | | works |

In a float type's name, `e` is the number of exponent bits and `m` the number of mantissa bits.
More exponent bits give a wider range. More mantissa bits give more precision.

## Conversions

Nothing converts by itself. You write `as`:

```rust
let x : f32 = 2.7;
let h = x as bf16;
let back = h as f32;
```

`as` also converts each element of a float tensor, and gives a new tensor of the same shape. The
cast only reads the tensor, so you can use the tensor again after it:

```rust
let w = Tensor<bf16, [64, 64]>::new();
let wide = w as f32;     // a Tensor<f32, [64, 64]>
let narrow = wide as f16;
```

A conversion to a narrower float rounds to the nearest value it can hold. On a tie it picks the
value whose last bit is 0.

A value too large for the type gives infinity in `f16` and `bf16`. `f8e4m3` and `f4e2m1` have no
infinity, so a value too large for them gives the largest value they hold: 448 or 6. `f8e5m2`
does the same, and gives 57344. NVIDIA's conversion instructions also work this way when
saturation is on.

Integer arithmetic wraps around. In `i4`, `3 * 2 + 3` is 9, which does not fit, so the answer is
-7. In `u4`, `15 + 1` is 0.

## Arithmetic on fp8 and fp4

`f16`, `bf16`, `i4` and `u4` compute like the other number types. An operation on two `bf16`
values gives a `bf16`.

An operation on `f8e4m3`, `f8e5m2` or `f4e2m1` values needs a result type that you write:

```rust
let c : f32 = a * b + d;     // a, b and d are f4e2m1
let e : f8e4m3 = a * b;
let f = a * b;               // error: the result type is not written
```

The compiler computes the whole expression in `f32`, and rounds once, to the type you wrote. This
is what the hardware does. The tensor cores of an NVIDIA Blackwell GPU multiply fp4 values, add the
products in `f32`, and convert the sum in a separate step.

This rule does not break the rule that nothing converts by itself. The result type is in the
program text, so the reader sees where the precision changes.

A matmul will work the same way. The compiler does not support it yet ([#1876]):

```rust
let c : Tensor<f32, [64, 64]> = a @ b;      // a and b are f4e2m1 tensors
```

The result type can be `f32`, `f16`, `bf16`, `f8e4m3`, `f8e5m2` or `f4e2m1`. The operands can be of
two different narrow types, for example an `f4e2m1` tensor times an `f8e4m3` tensor.

## Block-scaled fp4: NVFP4 and MXFP4

The compiler does not support this part yet ([#1877]).

An `f4e2m1` value is a number from -6 to 6. To hold real weights, NVFP4 and MXFP4
put the values in blocks, and give each block a scale. A value is its `f4e2m1` number times the
scale of its block.

| Format | Values per block | Scale per block | Scale per tensor |
| :--- | ---: | :--- | :--- |
| NVFP4 | 16 | `f8e4m3` | one `f32` |
| MXFP4 | 32 | `f8e8m0`, a power of two | none |

The scale is part of the tensor type:

```rust
let w : Tensor<f4e2m1, [4096, 4096], Scale<f8e4m3, block: 16>>;   // NVFP4
let m : Tensor<f4e2m1, [4096, 4096], Scale<f8e8m0, block: 32>>;   // MXFP4
```

`quantize` makes such a tensor from an `f32` tensor, and chooses each block's scale. `dequantize`
turns it back into `f32`. `@` applies the scales and adds the products in `f32`.

The memory capacity checks count the scales too. A 4096 by 4096 NVFP4 tensor is 8 MiB of values
and 1 MiB of scales, so it needs 9 MiB.

[#1876]: https://github.com/vx-lang/Vx/issues/1876
[#1877]: https://github.com/vx-lang/Vx/issues/1877

# Automatic differentiation

Vx can differentiate a function you wrote, at compile time. The derivative is generated from the
function's own code: the program does not record its operations as it runs, builds no graph, and
needs no separate framework.

Three forms, matching the three things people usually want.

## grad

`grad` gives the derivative of a function with respect to its input.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let d = grad(square, 3.0);
    return 0;
}
```

`grad(square, 3.0)` is the derivative of `square` evaluated at `3.0`. For `x * x` that is `2 * x`,
so `6.0`.

## vjp — reverse mode

A vector-Jacobian product. This is what backpropagation computes, and it is the efficient choice
when a function has many inputs and few outputs — the usual shape of a loss function.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let g = vjp(square, 2.0, 1.0);
    return 0;
}
```

The arguments are the function, the point to evaluate at, and the seed — the vector to multiply the
Jacobian by, which for a scalar loss is `1.0`.

## jvp — forward mode

A Jacobian-vector product. The efficient choice in the opposite case: few inputs, many outputs.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let d = jvp(square, 2.0, 1.0);
    return 0;
}
```

Same arguments, but the seed is a direction in the *input* space, and the result is how the outputs
move in that direction.

## Choosing between them

| You have | Use |
| --- | --- |
| Many inputs, one output (a loss) | `vjp` |
| One input, many outputs | `jvp` |
| A scalar function of a scalar | `grad` |

One call to `vjp` or `jvp` costs a small multiple of one call to the function. In general, to get
the whole Jacobian — every partial derivative — of a function with n inputs and m outputs, you need
one `vjp` for each output, or one `jvp` for each input. That is the whole reason both exist.

So far Vx differentiates only a function whose first parameter and result are each a single
number; see Limits worth knowing.

## Larger examples

These use `f64`, with `exp`, `ln_1p` and `cos` from `core::num`.

### Through calls to other functions

```rust
import core::num;

fn log_one_plus(t : f64) -> f64 {
    return t.ln_1p();
}

// softplus(x) = ln(1 + e^x), a smooth version of max(0, x).
fn softplus(x : f64) -> f64 {
    return log_one_plus(x.exp());
}

fn sigmoid(x : f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn main() -> i32 {
    let x : f64 = 0.5;
    print(grad(softplus, x));
    print!(" ");
    print(sigmoid(x));
    return 0;
}
```

The derivative of softplus is the sigmoid, and the program prints both:
`0.6224593312018545 0.6224593312018546`. The derivative goes through `exp` and then through
`log_one_plus` by the chain rule, and neither function had to be marked. The last digit differs
because the two numbers are computed in different ways.

### Branches and loops

The derivative follows whichever branches and loop iterations the function takes at that point.

```rust
// The Huber loss: quadratic near zero, linear further out.
fn huber(x : f64) -> f64 {
    if x < -1.0 {
        return -x - 0.5;
    }
    if x > 1.0 {
        return x - 0.5;
    }
    return 0.5 * x * x;
}

fn main() -> i32 {
    let near : f64 = 0.25;
    let far : f64 = 3.0;
    print(grad(huber, near));
    print!(" ");
    print(grad(huber, far));
    print!(" ");
    print(grad(huber, -far));
    return 0;
}
```

This prints `0.25 1 -1`, the slope of the piece each point falls in.

```rust
// x raised to the n, by multiplying n times.
fn power(x : f64, n : i32) -> f64 {
    let mut result : f64 = 1.0;
    for i in 0..n {
        result = result * x;
    }
    return result;
}

fn main() -> i32 {
    // The derivative of x^5 is 5 x^4, which is 80 at x = 2.
    let x : f64 = 2.0;
    print(grad(power, x, 5));
    return 0;
}
```

This prints `80`. The count `n` comes after `x`, so it is passed through unchanged and the loop
runs five times.

### Partial derivatives

`grad` moves the first parameter and holds the others fixed. To differentiate in another
parameter, write a function that takes it first.

```rust
fn f(x : f64, y : f64) -> f64 {
    return x * x * y;
}

// f with y as its first parameter, so grad moves y.
fn f_in_y(y : f64, x : f64) -> f64 {
    return f(x, y);
}

fn main() -> i32 {
    let x : f64 = 3.0;
    let y : f64 = 2.0;
    print(grad(f, x, y));
    print!(" ");
    print(grad(f_in_y, y, x));
    return 0;
}
```

This prints `12 9`: the derivative in `x` is `2xy`, and in `y` it is `x * x`.

### Using the derivative in a loop

A derivative is an ordinary function call, so a program can call it as often as it needs.
Newton's method uses it to find where a function is zero:

```rust
import core::num;

fn gap(x : f64) -> f64 {
    return x.cos() - x;
}

fn main() -> i32 {
    // Newton's method for the x where cos x = x.
    let mut x : f64 = 1.0;
    for step in 0..6 {
        x = x - gap(x) / grad(gap, x);
    }
    print(x);
    return 0;
}
```

This prints `0.7390851332151607`, the number whose cosine is itself.

Gradient descent uses it to fit a model. Here the model is a line through the origin, `y = w x`,
and the loss measures how far it is from three points:

```rust
// How far the line y = w x is from three measured points, squared and added up.
fn loss(w : f64) -> f64 {
    let e1 = w * 1.0 - 2.0;
    let e2 = w * 2.0 - 4.1;
    let e3 = w * 3.0 - 5.9;
    return e1 * e1 + e2 * e2 + e3 * e3;
}

fn main() -> i32 {
    let mut w : f64 = 0.0;
    for step in 0..50 {
        w = w - 0.02 * grad(loss, w);
    }
    print(w);
    return 0;
}
```

This prints `1.9928571428571429`, the best slope for those points (27.9 / 14).

## How it works

Differentiation is done by [Enzyme](https://enzyme.mit.edu/), which differentiates LLVM IR. Because
it works on the IR rather than on source, it differentiates through the optimiser's view of your
code, including calls into other functions.

The toolchains the installer sets up include Enzyme. A compiler built from source uses the Enzyme
plugin that `ENZYME_LIB` names; Building from source covers building one.

## Limits worth knowing

**Only floating-point values can be differentiated.** At each `grad`, `vjp` or `jvp` call, the
compiler checks the types of two things:

- the **result**, and
- the **first parameter**, which is the value the derivative is taken with respect to.

Both must be floating-point types. An `i32` or a `bool` takes separated values: between any two of
them there is no limit to take, so there is no derivative.

```
Error: Function 'discrete_func' cannot be differentiated because it returns the discrete type i32
```

The check is on types only. It does not look for corners or jumps inside the function: a function
built from pieces, like `huber` above, is differentiated piece by piece.

**Only single numbers, so far.** The first parameter and the result must each be one number, such
as an `f64`, not a tensor. For a function like that the three forms agree: `grad` gives the slope,
and `vjp` and `jvp` give the slope times the seed.

**A later parameter may be discrete, for `grad` and `vjp`.** A function of an `f64` that also takes
an index or a loop count is an ordinary thing to differentiate, and only the first argument varies.
`jvp` takes only a function with one parameter.

## Where to next

- Compile-time evaluation — the other thing that happens before the program runs
- Topologies and memory — running the result on an accelerator
