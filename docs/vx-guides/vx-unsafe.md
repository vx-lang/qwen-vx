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

# Unsafe and FFI

Vx checks a lot at compile time: ownership, borrowing, memory placement, contracts. Some things
cannot be checked — a pointer that came from C, a hardware register, a cast the compiler has no way
to justify. `unsafe` is where you take responsibility for those.

`unsafe` does not switch the checks off. It permits a small, specific set of operations that are
otherwise refused.

## unsafe blocks

An `unsafe` block is an expression. It can produce a value.

```rust
fn main() -> i32 {
    let x : i32 = 7;
    let p : *const i32 = &x;
    let v = unsafe { *p };
    return v - 7;
}
```

Reading through a raw pointer — `*p` — is the operation that needs the block. So does indexing
through one, or reading a field through one.

## Raw pointers

Two kinds, matching the two kinds of reference:

| Type | Meaning |
| --- | --- |
| `*const T` | A raw pointer you may read through. |
| `*mut T` | A raw pointer you may read and write through. |

Unlike `&T` and `&mut T`, raw pointers are not tracked by the borrow checker. Nothing stops two
`*mut T` pointing at the same value. That is exactly why dereferencing one needs `unsafe`.

A raw pointer can name a memory, as in `*mut f32 in Memory::GPU_HBM`. The functions of `core::ptr`
that move a pointer keep its memory: `offset`, `add` and `sub` and their `_mut` forms take a pointer
in any memory `M` and give a pointer in `M`. `read` and `write` take only a host pointer.

## unsafe functions

A function whose *body* is not the dangerous part — but whose *contract* is — should be marked
`unsafe` itself. A caller then has to opt in.

```rust
unsafe fn read_at(p : *const i32) -> i32 {
    return unsafe { *p };
}

fn main() -> i32 {
    return 0;
}
```

Unsafe-ness is part of the function's type. An `unsafe fn` cannot be passed where a safe function
is expected, so the obligation cannot be lost by storing the function in a variable.

The rule for when to mark a function `unsafe`: if a caller can make it misbehave by passing
something the compiler cannot check — a dangling pointer, a wrong length — it is `unsafe`. If the
function checks everything itself, it is not, even if its body uses `unsafe` internally.

## Pointer clauses

An `unsafe fn` can state what its caller must give it as `requires` clauses about its raw
pointers:

```rust
unsafe fn sum(p : *mut i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    let mut i = 0i64;
    while i < n {
        total = total + unsafe { p[i] };
        i = i + 1i64;
    }
    return total;
}

fn main() -> i32 {
    return 0;
}
```

- `valid(p, n)`: `p` is not null, is aligned, and points at `n` elements of one live allocation.
  The elements do not have to hold values yet.
- `initialized(p, n)`: each of the first `n` elements at `p` holds a value.
- `!overlaps(p, q, n)`: the first `n` elements at `p` and at `q` share no byte.

At a call in safe code, the compiler proves each clause from what it knows there: the clauses
of the caller, a pointer made from `&x` for a local variable `x`, and the pointer functions
`offset`, `add` and `sub`. A write `p[k] = x` or `*p = x` in `unsafe` code gives that one element
a value, so after `p[0] = 7i64` the compiler knows `initialized(p, 1i64)`. Two facts whose ranges
touch prove a clause together, so after `p[0] = 7i64; p[1] = 8i64;` it knows
`initialized(p, 2i64)`. A call that gets a raw pointer can free its memory, so after the call
the compiler knows nothing about memory that the caller does not own. When such code ended a
fact that the call needs, the error has a note with its line. In a loop, such code anywhere in
the body ends the facts at the start of the body, because an earlier pass can run it. A call in
`unsafe` code needs no proof: its author keeps the promise. A safe function with a pointer
clause cannot be a function value yet, because a call through the value would not prove the
clause.

`ensures !frees(p)` says that a call frees no memory that it reaches through `p`. Then a call
keeps the facts about the memory of its argument, and the compiler checks the body against the
clause. The compiler also finds `!frees(p)` itself when the body only reads and writes through
`p`, moves `p` with `offset`, compares it, and gives it only to functions with `!frees`. So
`split` compiles:

```rust
import core::ptr;

fn sum(p : *const i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    for i in 0i64..n {
        total = total + unsafe { p[i] };
    }
    return total;
}

fn split(p : *const i64, n : i64, k : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
    requires 0i64 <= k
    requires k <= n
{
    return sum(p, k) + sum(offset(p, k), n - k);
}

fn main() -> i32 {
    return 0;
}
```

`ensures frees(p)` says that the call can free that memory, so the compiler does not find
`!frees(p)` for it. An `extern` function can say `!frees(p)` after its return type, and the
compiler trusts it. It can also say a clause about numbers, such as `ensures 0i64 <= return`.
Each call then knows the clause, and `vxc --audit` lists it.

A function can also promise a pointer clause in an `ensures`, where `return` is the result. A
caller gets each such clause as a fact when the call returns. The body proves each one at each
`return` and at its end, after the storage of its local variables ends. So a function cannot
promise a pointer to one of its local variables. In an `unsafe` block, `assume e;` adds the fact
`e` without a proof, and it generates no code. Use it for a fact that the compiler cannot see,
such as the size of the memory that `alloc` gives. A wrong `assume` is undefined behavior.

```rust
import core::ptr;
import std::alloc;

fn alloc_i64(n : i64) -> *mut i64
    requires n > 0i64
    ensures valid(return, n)
{
    let p : *mut i64 = alloc(n * 8i64, 16i64);
    if is_null_mut(p) {
        panic("out of memory");
    }
    unsafe {
        assume valid(p, n);
    }
    return p;
}

fn first(p : *const i64) -> i64
    requires valid(p, 1i64)
    requires initialized(p, 1i64)
{
    return unsafe { p[0] };
}

fn main() -> i32 {
    let p = alloc_i64(1i64);
    unsafe {
        p[0] = 7i64;
    }
    let x = first(p as *const i64);
    let bytes : *mut i8 = p;
    dealloc(bytes, 16i64);
    return (x - 7i64) as i32;
}
```

The methods of `Vec` state such clauses. `as_mut_ptr` ensures `valid` and `initialized` for the
`len` elements of the vector, and `len` ensures that it returns that count. The methods that only
read the vector have `!frees(self)`. So safe code can give the pointer to `sum`. A `push` can move
the buffer, so after a `push` the pointer has no facts.

```rust
import core::ptr;
import std::vec;

fn sum(p : *const i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    for i in 0i64..n {
        total = total + unsafe { p[i] };
    }
    return total;
}

fn total(v : &Vec<i64>) -> i64 {
    let p = v.as_mut_ptr();
    return sum(p as *const i64, v.len() as i64);
}

fn main() -> i32 {
    let mut v = Vec<i64>::new();
    v.push(2i64);
    v.push(3i64);
    return (total(&v) - 5i64) as i32;
}
```

A vector that a function makes with `Vec<i64>::new()` owns only new memory. So a `push` on it, or
its drop, keeps the facts about a pointer parameter. That stops at a pointer or a reference into
the vector, and at a loop or a closure that uses it. It also stops at a call that can put memory of
the caller into the vector.

From a `spawn` until its `join()` or the end of its `tasks` block, the task can free memory, so
no fact about other memory starts in that time.

For `requires valid(p, n)`, the compiler tells LLVM that `p` is not null and aligned, and that
`n` elements at `p` can be read. For `requires !overlaps(p, q, n)`, it tells LLVM that an access
`p[i]` does not alias an access `q[i]`, where `i` is the variable of a `for i in 0..n` loop. After
each call of a function with `ensures valid(return, n)`, it tells LLVM that `n` elements at the
result can be read. A clause that does not hold is undefined behavior, as a wrong `*p` is.

`vxc --audit program.vx` prints each fact that the compiler trusts without a proof, one line each:

- each `assume`
- each pointer clause that a call in `unsafe` code trusts, because the compiler cannot prove it
- each `ensures` of an `extern` function
- each pointer `requires` of a `pub fn`, because a call from C proves nothing

Each line starts with the file, the line and the column, and the lines are in that order. So a
review can compare the audit of two versions of a program.

## Calling C

An `extern` block declares functions that exist outside Vx.

```rust
extern "C" {
    safe fn abs(x : i32) -> i32;
}

fn main() -> i32 {
    return abs(0);
}
```

By default an `extern` function is unsafe to call — the compiler knows nothing about what is on the
other side. `safe` marks one that genuinely is safe to call with any arguments its types allow, so
callers do not need an `unsafe` block.

`abs` qualifies: every `i32` is a valid input and it cannot misuse memory. A function taking a
pointer and a length would not qualify, because passing a mismatched pair breaks it.

Use `safe` sparingly. It is a promise the compiler cannot verify, and it is the one place in an
`extern` block where a mistake is silent.

## Calling Vx from C

An object file exports only the functions marked `pub`. Every other function the program defines
is hidden, so C cannot call it.

```rust
pub fn add_one(x : i32) -> i32 {
    return x + 1;
}
```

```sh
vxc add_one.vx --action emit-obj -o add_one.o
cc -shared -o libadd_one.so add_one.o
```

C, or Python's `ctypes`, can now load `libadd_one.so` and call `add_one`. Give a `pub` function
parameters and a result that C can pass: numbers, `bool` and raw pointers. Only a top-level
function that is not generic can be `pub` for now.

## Inline MLIR

The compiler turns a Vx program into MLIR, a set of small languages for compilers. Each of these
languages is a *dialect*: `arith` has integer and float arithmetic, `math` has functions such as
`ctpop` (count the bits that are 1), and `llvm` has the operations of LLVM. An `mlir!` block puts
MLIR text directly into a Vx function. Use it for one operation that Vx has no syntax for.

```vx
mlir!(inputs : (%a = x : i32), returns : i32, dialects : ["math"]) {
  %r = math.ctpop %a : i32
  macro.yield %r : i32
}
```

- `inputs` gives each Vx value to the block. `%a = x : i32` names the value of `x` as `%a` in the
  MLIR, with the MLIR type `i32`.
- `returns` gives the Vx type of the value of the block. `macro.yield` gives that value. A block
  that gives no value says `returns : void`.
- `dialects` lists each dialect whose operations the block uses.

This program has two functions whose bodies are `mlir!` blocks
(`book_unsafe_ffi_inline_mlir.vx`):

```rust
fn count_ones(x : i32) -> i32 {
  return mlir!(inputs : (%a = x : i32), returns : i32, dialects : ["math"]) {
    %r = math.ctpop %a : i32
    macro.yield %r : i32
  };
}

fn larger(x : i64, y : i64) -> i64 {
  return mlir!(inputs : (%a = x : i64, %b = y : i64), returns : i64, dialects : ["arith"]) {
    %r = arith.maxsi %a, %b : i64
    macro.yield %r : i64
  };
}

fn main() -> i32 {
  println!("ones in 255: {}", count_ones(255));
  println!("ones in 1024: {}", count_ones(1024));
  println!("larger of 3 and 7: {}", larger(3, 7));
  return 0;
}
```

It prints:

```text
ones in 255: 8
ones in 1024: 1
larger of 3 and 7: 7
```

The `dialects` list must agree with the body. If `count_ones` listed `"arith"` and not `"math"`,
the compiler would give the error E8010: the block uses `math.ctpop`, but the list does not name
`math`. A name in the list that is not a dialect is also E8010.

The compiler does not check what the MLIR operations do. So treat each `mlir!` block as code that
the compiler trusts, as you treat an `unsafe` block. The standard library uses `mlir!` in this way.
`core::ptr` moves a raw pointer and compares two raw pointers with one LLVM operation each.

The compiler assumes one more thing about a block: a block that gets a raw pointer can free the
memory behind it. So a function that says `ensures !frees(p)` (see Pointer
clauses) cannot give `p` to an `mlir!` block (E8012). `clobbers : [n]` tells the compiler that the block writes the variable `n`.

## What unsafe does not unlock

`unsafe` covers operations the type system cannot justify. It does **not** override placement.

A pointer into device memory still cannot be dereferenced from the host inside an `unsafe` block —
that is not an unchecked operation, it is a wrong one, and it stays a compile error. See
Topologies and memory.

## Where to next

- Ownership and borrowing — the checks `unsafe` steps around
- Topologies and memory — the checks it does not
- Standard library reference — which `std` functions are `unsafe`, and why

# Files, input and time

A program often needs text, files, the clock or the network. This chapter shows the parts of the
standard library for these jobs: `std::string`, `std::fs`, `std::io`, `std::time` and `std::net`.

Many of these functions take a raw pointer, such as a string literal or a buffer. The compiler
cannot check a raw pointer, so you call these functions inside an `unsafe { .. }` block.
Unsafe and FFI explains raw pointers.

Each complete program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Strings

`String` is a text that can grow. `String::new()` makes an empty string. `push_c_str` adds a
string literal to the end. `len()` gives the length in bytes, not in characters.

`to_string()` on an integer gives its decimal digits as a `String`. `String::from_c_str` copies a
string literal. `==` on two strings compares their bytes. `parse_int` reads the digits at the
start of a text as an `i32` (`book_io_build_a_string.vx`):

```rust
import std::string;

fn main() -> i32 {
  let mut greeting = String::new();
  unsafe { greeting.push_c_str("hello, "); greeting.push_c_str("world"); }
  let size = greeting.len();
  println!("{} has {} bytes", greeting, size);
  let n : i32 = 42;
  let text = n.to_string();
  let same = unsafe { String::from_c_str("42") };
  println!("equal: {}", text == same);
  let parsed = unsafe { parse_int("123 apples") };
  println!("parsed {}", parsed + 1);
  return 0;
}
```

It prints:

```text
hello, world has 12 bytes
equal: true
parsed 124
```

`println!` moves a `String` argument ([Vx#3235](https://github.com/vx-lang/Vx/issues/3235)). So the
program gets the length before it prints `greeting`.

`parse_int` stops at the first byte that is not a digit. So `"12abc"` gives 12, and `"abc"`
gives 0. A `-` is not a sign. `parse_int` does not check for overflow.

## Files

`File::open(path, mode)` opens a file. `path` is a text that ends with a zero byte, such as a
string literal. The mode is one of the three values of `OpenMode`:

- `OpenMode::Read()` opens a file that exists, for reading.
- `OpenMode::Write()` makes the file, or makes an existing file empty, for writing.
- `OpenMode::ReadWriteCreate()` opens the file for both, and makes it if it does not exist. It
  keeps the bytes that the file has.

`write(buffer, len)` writes `len` bytes from a buffer and gives how many it wrote. `read(buffer, len)` reads up to `len` bytes into a buffer and gives how many it read. 0 means the end of the
file. `seek(offset, whence)` moves the position of the next read or write, and gives the new
position. `whence` is 0 for the start of the file, 1 for the current position and 2 for the end.

`File` implements `Drop`, so a file closes at the end of its scope. You do not close it by hand.
Ownership and borrowing explains `Drop`.

This program writes three lines to a file, and reads the file back one byte at a time. The test
gives the path of the file in the environment variable `NOTES` (`book_io_write_and_read_a_file.vx`):

```rust
import std::fs;
import std::string;
import std::vec;

extern {
  fn getenv(name : *const i8) -> *const i8;
}

// Writes `text` to a new file at `path`. The file closes when `out` goes out of scope.
fn write_text(path : *const i8, text : *const i8) -> void {
  let mut bytes = Vec<u8>::new();
  let length = unsafe { string_length(text) };
  for i in 0..length {
    bytes.push(unsafe { text[i] } as u8);
  }
  let mut out = unsafe { File::open(path, OpenMode::Write()) };
  let written = unsafe {
    out.write(bytes.as_mut_ptr() as *const u8, length as u64)
  };
  println!("wrote {} bytes", written);
}

fn main() -> i32 {
  let path = unsafe { getenv("NOTES") };
  write_text(path, "one\ntwo\nthree\n");

  let mut input = unsafe { File::open(path, OpenMode::Read()) };
  let mut byte : u8 = 0;
  let mut lines = 0;
  loop {
    let n = unsafe { input.read((&mut byte) as *mut u8, 1) };
    if n == 0 {
      break;
    }
    if byte == 10 {
      lines = lines + 1;
    }
  }
  println!("read {} lines", lines);
  println!("the file ends at byte {}", input.seek(0, 2));

  input.seek(4, 0);
  unsafe { input.read((&mut byte) as *mut u8, 1) };
  println!("byte 4 is {}, the letter t", byte);
  return 0;
}
```

It prints:

```text
wrote 14 bytes
read 3 lines
the file ends at byte 14
byte 4 is 116, the letter t
```

`write_text` copies the literal into a `Vec<u8>`, because `write` takes a pointer to `u8` bytes.
`out` closes at the end of `write_text`. So `main` reads all 14 bytes.

A failed `open` gives a `File` that does nothing: each later `read` and `write` gives 0. A missing
file and an empty file look the same. So check the count that `read` and `write` give.

## Standard input and output

`std::io` has three functions. `stdin_read(buffer, len)` reads up to `len` bytes of standard input
and gives the count. 0 means the end of the input. `stdout_write` and `stderr_write` write bytes to
standard output and standard error. This loop counts the lines of the input:

```vx
import std::io;

fn main() -> i32 {
  let mut byte : u8 = 0;
  let mut lines = 0;
  loop {
    let n = unsafe { stdin_read((&mut byte) as *mut u8, 1) };
    if n == 0 {
      break;
    }
    if byte == 10 {
      lines = lines + 1;
    }
  }
  println!("{} lines", lines);
  return 0;
}
```

When `vxc program.vx` runs a program, the program gets no standard input
([Vx#3236](https://github.com/vx-lang/Vx/issues/3236)). So this loop stops at once and prints
`0 lines`. Compile the program with `-c` and run the executable to give it input
(Your first program).

## Time

`std::time` measures time. `Instant::now()` reads a clock that never goes back. `elapsed()` gives
the time since a reading, as a `Duration`. The difference `later - earlier` of two readings is a
`Duration` too. `sleep(d)` stops the thread for at least the `Duration` `d`.

`Duration` comes from `core::time`. `Duration::from_secs`, `from_millis`, `from_micros` and
`from_nanos` make one. `as_secs`, `as_millis` and `as_nanos` give its length, and `subsec_millis`
gives the part that is less than a second. `+` adds two durations
(`book_io_sleep_and_measure.vx`):

```rust
import core::time;
import std::time;

fn main() -> i32 {
  let start = Instant::now();
  sleep(Duration::from_millis(50));
  let waited = start.elapsed();
  println!("slept at least 50 ms: {}", waited.as_millis() >= 50);

  let step = Duration::from_millis(1500);
  println!("{} s and {} ms", step.as_secs(), step.subsec_millis());
  let total = step + Duration::from_secs(2);
  println!("total {} ms", total.as_millis());
  return 0;
}
```

It prints:

```text
slept at least 50 ms: true
1 s and 500 ms
total 3500 ms
```

The program does not print the time that it slept, because that time is different in each run.

## Networks

`std::net` has three types. `TcpListener` waits for TCP connections. `TcpStream` is one TCP
connection. `UdpSocket` sends and receives UDP datagrams, which are single messages. Each address
is a text of the form `host:port`, for example `"127.0.0.1:7878"`.

Each call that can fail gives a `Result`. Its `Err` holds a `NetError`, which says why the call
failed:

- `BadAddress`: the address is not `host:port`, or the host has no address.
- `Refused`: nothing listens at the address.
- `Reset`: the other end closed the connection.
- `InUse`: another socket uses the address.
- `Cancelled`: the program cancelled the `tasks` block of the task while the call waited.
- `Os(code)`: another failure, with the error number of the system.

This program makes two calls. The address of the first has no port. Port 0 in the second asks the
system for any free port (`book_io_network_errors.vx`):

```rust
import core::result;
import std::net;

// Prints why a network call did not work.
fn describe(e : NetError) -> void {
  match e {
    NetError::BadAddress => {
      println!("bad address");
    }
    NetError::Refused => {
      println!("refused");
    }
    NetError::Os(code) => {
      println!("system error {}", code);
    }
    _ => {
      println!("another error");
    }
  }
}

fn main() -> i32 {
  // An address must be `host:port`. This one has no port.
  match unsafe { TcpStream::connect("localhost") } {
    Result<TcpStream, NetError>::Ok(_stream) => {
      println!("connected");
    }
    Result<TcpStream, NetError>::Err(e) => {
      describe(e);
    }
  }
  // Port 0 asks the system for any free port.
  match unsafe { UdpSocket::bind("127.0.0.1:0") } {
    Result<UdpSocket, NetError>::Ok(_socket) => {
      println!("bound a UDP socket");
    }
    Result<UdpSocket, NetError>::Err(e) => {
      describe(e);
    }
  }
  return 0;
}
```

It prints:

```text
bad address
bound a UDP socket
```

The sockets close at the end of their scope, as a `File` does.

A server calls `accept()` on a `TcpListener`. `accept` waits until a client connects, and gives a
`TcpStream`. `read` and `write` on the stream work as they do on a `File`. This function sends one
byte back to the client. `?` gives the `Err` to the caller:

```vx
fn echo_one_byte(listener : TcpListener) -> Result<u64, NetError> {
  let mut server = listener;
  let mut client = server.accept()?;
  let mut byte : u8 = 0;
  let got = unsafe { client.read((&mut byte) as *mut u8, 1) };
  let _count = got?;
  return unsafe { client.write((&byte) as *const u8, 1) };
}
```

TCP is a stream of bytes. A message can arrive in parts, so a `read` can give fewer bytes than the
other end wrote. To get a whole message, read in a loop until you have all of its bytes.
