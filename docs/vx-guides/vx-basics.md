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

# Errors, Option and Result

A function can fail to give an answer. A search can find nothing, and a parse can meet a bad
character. Vx has no exceptions. A function that can fail says so in its return type, with
`Option` or `Result`. This chapter shows how to make these values, how to read them, and how to
pass an error up to the caller.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Option: a value or nothing

`Option<T>` holds either one value of type `T` (`Some`) or no value (`None`). It comes from
`import core::option;`. Write a variant with its full type: `Option<i64>::Some(3)` and
`Option<i64>::None()`. Vx for Rust programmers lists this and the
other differences from Rust.

Read an `Option` with `match`, as Control flow shows for an enum. Each
arm handles one variant. A method is shorter when you need only one fact:

- `o.is_some()` and `o.is_none()` tell which variant `o` is. They borrow `o`, so you can still use
  it after the call.
- `o.unwrap_or(d)` gives the value, or `d` for a `None`.

This program searches for the first square above a limit (`book_errors_find_a_value.vx`):

```rust
import core::option;

fn first_square_above(limit : i64) -> Option<i64> {
  for i in 1..10 {
    let sq = (i * i) as i64;
    if sq > limit {
      return Option<i64>::Some(sq);
    }
  }
  return Option<i64>::None();
}

fn report(limit : i64) -> void {
  match first_square_above(limit) {
    Option<i64>::Some(sq) => {
      println!("first square above {} is {}", limit, sq);
    }
    Option<i64>::None() => {
      println!("no square above {}", limit);
    }
  }
}

fn main() -> i32 {
  report(20);
  report(200);
  let missing = first_square_above(500);
  println!("found {}", missing.is_some());
  println!("or 0 gives {}", missing.unwrap_or(0));
  return 0;
}
```

It prints:

```text
first square above 20 is 25
no square above 200
found false
or 0 gives 0
```

## Result: a value or an error

`Result<T, E>` holds either a value of type `T` (`Ok`) or an error of type `E` (`Err`). It comes
from `import core::result;`. Use it when the caller must know why the operation failed. The error
type is often an enum of your own, with one variant for each way to fail.

`r.is_ok()` and `r.is_err()` tell which variant `r` is. They also borrow `r`, so they work for a
payload that is not `Copy`. `r.unwrap_or(d)` gives the value, or `d` for an `Err`.

## The `?` operator

`r?` gives the value of an `Ok`. For an `Err`, it returns that `Err` from the function at once.
The function must return a `Result` with the same error type. So `?` replaces a `match` that only
passes the error up.

`?` does not convert the error. If the error types differ, the compiler reports E3063. Use
`r.map_err(f)` first: it calls the closure `f` on the error of an `Err`, and it keeps an `Ok` as it
is. Write the type that the closure returns, as in `| c : i64 | -> AddError { .. }`. Without it,
the `?` after `map_err` gets a wrong E3063 ([Vx#3233](https://github.com/vx-lang/Vx/issues/3233)).

In this program, `digit` reports a bad character with its code, an `i64`. `two_digits` converts
that code to an `AddError` and passes it up with `?` (`book_errors_question_mark.vx`):

```rust
import core::result;

enum AddError {
  NotADigit(i64),
  TooLarge(i64),
}

// The character code of a digit, such as 50 for '2', gives its value.
fn digit(c : i64) -> Result<i64, i64> {
  if c >= 48 && c <= 57 {
    return Result<i64, i64>::Ok(c - 48);
  }
  return Result<i64, i64>::Err(c);
}

// A number with two digits, from the codes of its characters.
fn two_digits(tens : i64, ones : i64) -> Result<i64, AddError> {
  let t = digit(tens).map_err(| c : i64 | -> AddError {
    AddError::NotADigit(c)
  })?;
  let o = digit(ones).map_err(| c : i64 | -> AddError {
    AddError::NotADigit(c)
  })?;
  return Result<i64, AddError>::Ok(t * 10 + o);
}

fn add(a : Result<i64, AddError>, b : Result<i64, AddError>) -> Result<i64, AddError> {
  let sum = a? + b?;
  if sum > 99 {
    return Result<i64, AddError>::Err(AddError::TooLarge(sum));
  }
  return Result<i64, AddError>::Ok(sum);
}

fn report_error(e : AddError) -> void {
  match e {
    AddError::NotADigit(c) => {
      println!("error: the character {} is not a digit", c);
    }
    AddError::TooLarge(n) => {
      println!("error: the sum {} is too large", n);
    }
  }
}

fn report(r : Result<i64, AddError>) -> void {
  match r {
    Result<i64, AddError>::Ok(sum) => {
      println!("sum {}", sum);
    }
    Result<i64, AddError>::Err(e) => {
      report_error(e);
    }
  }
}

fn main() -> i32 {
  // The codes 49 and 50 are '1' and '2'. The code 120 is 'x'.
  report(add(two_digits(49, 50), two_digits(51, 48)));
  report(add(two_digits(49, 120), two_digits(51, 48)));
  report(add(two_digits(53, 52), two_digits(53, 52)));
  return 0;
}
```

It prints:

```text
sum 42
error: the character 120 is not a digit
error: the sum 108 is too large
```

A pattern in a `match` cannot test a variant inside a payload. `Err(AddError::TooLarge(n))` gets
the error E3050. So `report` binds the error to `e`, and `report_error` matches on `e`.

`?` takes only a `Result`. A `?` on an `Option` gets the error E3063. Convert the `Option` first:
`o.ok_or(e)` gives `Ok` with the value, or `Err(e)` for a `None`.

```vx
fn quarter(n : i64) -> Result<i64, i64> {
  let h = half(n).ok_or(n)?;
  let q = half(h).ok_or(h)?;
  return Result<i64, i64>::Ok(q);
}
```

A `?` in a task or a `tasks` block passes the error to the block, not to the function.
Tasks and channels shows this.

## Take a value out with `take` and `replace`

A field of type `Option` can hold a value for a time. `o.take()` moves the value out and leaves
`None` in `o`. `o.replace(v)` puts `v` in `o` and gives the old contents. Both borrow `o` mutably,
so `o` must be a `let mut` variable or a field of one. Both work for a value that is not `Copy`.

In this program, a desk holds at most one ticket (`book_errors_take_and_replace.vx`). A message
such as `"took"` is a `*const i8`, the type of a string literal:

```rust
import core::option;

// A struct with no `Copy` impl. A move of a ticket leaves nothing behind.
struct Ticket {
  id : i64,
}

struct Desk {
  waiting : Option<Ticket>,
}

fn show(what : *const i8, o : Option<Ticket>) -> void {
  match o {
    Option<Ticket>::Some(t) => {
      println!("{} ticket {}", what, t.id);
    }
    Option<Ticket>::None() => {
      println!("{} nothing", what);
    }
  }
}

fn main() -> i32 {
  let mut desk = Desk {
    waiting: Option<Ticket>::None()
  };
  show("replaced", desk.waiting.replace(Ticket {
    id: 1
  }));
  show("replaced", desk.waiting.replace(Ticket {
    id: 2
  }));
  show("took", desk.waiting.take());
  println!("empty {}", desk.waiting.is_none());
  return 0;
}
```

It prints:

```text
replaced nothing
replaced ticket 1
took ticket 2
empty true
```

## Errors that the program cannot handle

Some errors mean that the program has a bug. Then the best action is to stop the program with a
clear message. A tour of Vx explains `panic`, `abort` and `-> !`.

- `o.unwrap()` and `r.unwrap()` give the value. For a `None` or an `Err`, they stop the program.
- `o.expect(msg)` and `r.expect(msg)` do the same, but they print `msg`.
- `panic(msg)` stops the program with `msg`. `abort()` stops it with no message.

Each of these prints the file and line of the call, not a line in the standard library. Use
`expect` rather than `unwrap`, so that the message tells the reader what went wrong.

This program divides twice. The second division fails (`book_errors_expect_stops_the_program.vx`):

```rust
import core::result;

fn divide(a : i64, b : i64) -> Result<i64, i64> {
  if b == 0 {
    return Result<i64, i64>::Err(a);
  }
  return Result<i64, i64>::Ok(a / b);
}

fn main() -> i32 {
  let good = divide(10, 2);
  println!("ok {}, err {}", good.is_ok(), good.is_err());
  println!("10 / 2 = {}", good.unwrap());
  let bad = divide(10, 0);
  let q = bad.expect("the divisor must not be zero");
  println!("never printed {}", q);
  return 0;
}
```

It prints the lines below and ends with an error status. The last line names the line of the
`expect` call.

```text
ok true, err false
10 / 2 = 5
panic: the divisor must not be zero
  at book_errors_expect_stops_the_program.vx:36
```

`unwrap` and `expect` take the `Option` or `Result` by value. So call `is_ok` before them, as this
program does, and not after.

## More methods

`core/option.vx` and `core/result.vx` in the standard library have more methods, each with a
comment. Some of them are `map`, `and_then`, `or`, `filter`, `ok`, `err`, `unwrap_err` and
`unwrap_or_else`.

# Ownership and borrowing

Vx has no garbage collector and no mandatory reference counting. Memory is managed by ownership,
checked at compile time, in the same family as Rust's model.

## Owning and borrowing

A binding owns its value. Passing it to a function by value moves it, and the original binding is
no longer usable:

```rust
let v = make_buffer();
consume(v);      // v is moved
// reading v here is an error
```

Borrow instead of moving to keep the original alive:

```rust
fn total(v : &Vec<i32>) -> i32 { /* ... */ }
fn append(v : &mut Vec<i32>, x : i32) -> void { /* ... */ }
```

`&T` is a shared borrow, `&mut T` is an exclusive one. The usual rule applies: any number of shared
borrows, or exactly one exclusive borrow, never both at once. The checker tracks variance and
regions, so a borrow cannot outlive what it points into.

## The rules the checker enforces

Each rule below is a compile error, with the code the error index lists it under.

**A moved value cannot be used (E4001).** Assigning a value, or passing it by value, moves it
unless its type is `Copy`; numbers are. Giving the variable a new value makes it usable again. A
method that takes `self` by value moves the value it is called on: `v.into_iter()` moves `v`.

```rust
let a = S { v : 1 };
let b = a;          // a is moved
print(a.v);         // E4001
```

Only one branch of an `if` or arm of a `match` runs, so each may move the same value. After the
`if`, the value counts as moved if any branch that reaches the next line moved it.

Inside a loop, a value moved in the body is used again on the next pass, so it is refused at the
loop unless the body gives it a new value first. A `continue` goes round again as well. A move
just before `break` is fine; the value then counts as moved after the loop.

```rust
let a = S { v : 1 };
while more() {      // E4001: a is moved on every pass
    take(a);
}
```

**A value cannot move while it is borrowed (E4007).** If a borrow of `a` is used after `a` moves,
the borrow would point at a value that has gone.

```rust
let r = &a;
let b = a;          // E4007: r is used below
print(r.v);
```

**A reference cannot give away what it points at (E4008).** `let b = *r;` would move the value out
of `r`, which only borrows it. Copy or clone it instead; for a number, `*r` is a copy and is fine.

**Reading a field that has something to drop moves it out of its struct.** After `let x = s.a;`,
`s.a` has no value and `s` cannot be used whole, but its other fields can, and `s` drops only the
fields it still holds. Assigning `s.a` gives it a value again. A field cannot be moved out through
a reference (E4008), out of a struct whose type implements `Drop`, since its `drop` needs every
field, or out of a value with no name, such as `make().a`, which drops all its fields at the end
of the statement (E4013). Borrow it (`&s.a`) or clone it (`s.a.clone()`) instead. A field with
nothing to drop is copied.

**A variable without `mut` is read-only (E4010).** As in Rust, `let x = ..` and a parameter `x : T`
cannot be assigned, nor can a field or element of `x`; `x` cannot be borrowed `&mut`; and a method
that takes `&mut self` cannot be called on it. Write `let mut x = ..`, or `mut x : T` for a
parameter (`mut self : T` for a method that takes `self` by value). A `&mut` held in a variable
without `mut` can still be written through, since that changes what it points at, not the variable.

```rust
let x = 1;
x = 2;              // E4010
let mut y = 1;
y = 2;              // fine
let r = &mut y;
*r = 3;             // fine: writes y, not r
```

**Shared or exclusive, never both (E4002, E4003, E4004).** While a `&mut` borrow of `x` is still
going to be used, `x` cannot be read, written or borrowed again. While a `&` borrow is still going
to be used, `x` cannot be borrowed `&mut`. Two borrows passed to one call count as alive together.

```rust
let m = &mut x;
print(x);           // E4002: m is used below
*m = 2;
```

**A borrow ends at its last use.** It does not last to the end of the block, so this is fine:

```rust
let m = &mut x;
*m = 2;
print(x);           // m is not used again
```

**Different fields are borrowed separately.** `&mut p.a` and `&mut p.b` can be alive together;
`&mut p.a` twice, or `&p.a` while `&mut p` is alive, cannot.

**A reborrow borrows the reference (E4002, E4004).** `let n = &mut *m;` borrows `m` for as long as
`n` is used. Until then `m` cannot be read, written or reborrowed `&mut` again. After `n`'s last use,
`m` works as before.

```rust
let m = &mut x;
let n = &mut *m;
*m = 4;             // E4002: n is used below
*n = 5;
```

**A borrowed variable cannot be assigned (E4009).** While a `&` borrow of `x`, or of a field or
element of it, is still going to be used, that part of `x` cannot be given a new value. A borrow
passed only to a call, as in `v.cmp(&best)`, ends with the call.

```rust
let r = &x;
x = 2;              // E4009: r is used below
print(*r);
```

**Only a `&mut` can be written through (E4006).** `*r = v`, `r.f = v` and `r[i] = v` need `r` to be
a `&mut`. So does a `&mut` field reached through a `&`. A `&` cannot be turned into a `&mut` either:
`&mut *r`, or calling a method that takes `&mut self` through `r`, needs `r` to be a `&mut`.

**A reference cannot outlive what it points at (E4005).** A reference to a local, or a value or
closure holding one, cannot be returned, stored through a reference, or passed to a call that could
store it. Inside a function, a variable declared outside a block cannot be given a reference to a
variable declared inside it.

## Views of a tensor

A row `q[i]`, a field `h.t` that holds a tensor, and a view of either are *views*: they share
their owner's memory instead of copying it. Writing `r[0]` through a row `r = q[1]` changes
`q[1][0]`, and the other way round. So a view is a borrow of its owner, exactly as `&q` is, and the
rules above apply to it.

```rust
let mut q = Tensor<f32, [2, 4]>::new();
let mut r = q[1];   // borrows q mutably, since r is `mut`
r[0] = 5.0;         // writes q[1][0]
print(q[1][0]);     // fine: r is not used again
```

- A view is made by indexing to a row (`q[1]`), reading a tensor field (`h.t`), reshaping
  (`q.reshape([8])`), or taking a view of a view. A view chosen by an `if` or a `match` borrows
  every tensor it may come from, and a view taken through a reference (`rq[1]` with `rq = &q`)
  borrows the tensor behind the reference. A closure that uses a view keeps the view's tensor
  borrowed for as long as the closure is used.
- A view declared `mut` is a `&mut` borrow, and its owner must be `mut` (E4010). Any other view
  is a `&` borrow, so two of them can be used together.
- The owner cannot be moved (E4007), assigned (E4009) or, under a `mut` view, read (E4002) while
  the view is still going to be used. As with `&`, the borrow ends at the view's last use.
- A view of a tensor the function owns, a local or a parameter taken by value, cannot be
  returned (E4005). A view of a tensor taken by reference (`q : &Tensor<..>`) can be returned
  as a reference: `-> &Tensor<..>` or `-> &mut Tensor<..>`, with `return &q[1]`. The result
  borrows the caller's tensor, and a write through a `&mut` result changes that tensor. A view
  of memory from a raw pointer, `tensor_view_2d(p, ..)`, is returned the same way.
- A function that returns a tensor by value (`-> Tensor<..>`) cannot return a view (E4011): the
  caller would get a copy, and its writes would not reach the owner. Return a reference, or
  return a copy with `q[1].clone()`.
- A view cannot be stored where a tensor of its own is held: a variable that already exists
  (`keep = q[1]`) or a struct field (`h.t = q[1]`, `Holder { t : q[1] }`) (E4011). It would
  replace that tensor with a window onto another. Bind it to a new variable with `let` instead.
  Writing a row into another row, `p[0] = q[1]`, copies the elements and is fine.
- A view cannot be passed to a function that takes a tensor by value (E4011): the function
  would own memory that is the owner's. Pass a copy, `f(q[1].clone())`, or make the function
  take the tensor by reference, `f(&q[1])`. `t.clone()` makes a new tensor of `t`'s type with
  a copy of its elements.

**Why views are checked this strictly.** Vx frees memory automatically
(Dropping values). It frees a tensor that only holds memory after the last use
of the tensor *and of every view of it*. It drops a type with a `Drop` of its own at the end of
its block. Freeing at the last use is only safe if the checker sees every view, so
a free that comes too early is a borrow checker bug, not a rule for the programmer to remember.
The checker tells a view from an owner by where the value came from, since both have the type
`Tensor`; a separate view type (#400) may replace this later.

## Dropping values

When its owner goes away, a value is *dropped*: what it owns is released. A value that was moved
is not dropped, because its new owner drops it. Giving a variable a new value drops the old one.

- **Memory only:** a tensor, or a `Vec` of numbers, is freed after its last use.
- **A `Drop` implementation:** a value whose type implements `Drop` waits for the end of its block,
  and values in one block are dropped in reverse order of declaration. Its `drop` runs first, then
  its fields are dropped in the order they are declared. Dropping an enum drops the payload of the
  variant it holds.

```rust
import core::ops;

struct Noisy {
    id : i32,
}

impl Drop for Noisy {
    fn drop(self : &mut Noisy) -> void {
        print(self.id);
    }
}

fn main() -> i32 {
    let a = Noisy { id : 1 };
    let b = Noisy { id : 2 };
    return 0;   // prints 2, then 1
}
```

A `match` arm that moves a part out of the matched value takes all of the value. For example,
`match t { Two::Pair(a, _) => .. }` moves all of `t` into the arm. An arm that moves nothing leaves
`t` as it is, and when that arm runs, `t` drops at the end of its block. Each arm of a `match` on a
value with no name, such as `match make() { .. }`, takes the value too. At its end, the arm drops the
names that it binds, and then the parts that it does not name, such as the `_` part. Rust keeps
the `_` part in `t` and drops it at the end of the block of `t`.

In a `let`, `_` is a name like any other. `let _ = e;` drops the value at the same point as
`let _x = e;`, and `let _ = a;` moves `a`. In Rust, `let _ = e;` drops the value at the end of the
statement, and `let _ = a;` moves nothing. The reasons are in
[the language semantics](https://github.com/vx-lang/Vx/blob/main/docs/lang/semantics.md#26-drops).

`Vec`, `Box`, `String`, `File` and the socket types implement `Drop`, so none of them is freed by
hand. `drop` cannot be called directly (E4012). To end a value's life early, `core::mem` has
`drop(x)`; `forget(x)` ends it without dropping anything, and `needs_drop<T>()` says whether
dropping a `T` does anything.

## Linear values

Some values are *linear*: they must be consumed exactly once, and the checker enforces it. Device
buffers are the motivating case. A buffer that has been handed off to an accelerator has left your
control, and reading it again is a use-after-move — reported as a compile error rather than as
corrupted data.

This is stronger than an ordinary move check. A linear value cannot be quietly dropped either,
because dropping a device allocation without releasing it is a leak the runtime cannot detect for
you.

## Boxing

Recursive types must be boxed. `Box<T>` is a heap allocation with a single owner:

```rust
import std::box;

struct Node {
    value: i32,
    next: Box<Node>,
}
```

The requirement is not an oversight — it is what lets the compiler give every nominal type a size
without solving a fixpoint across module boundaries, which in turn is what lets the frontend compile
modules in parallel with no shared state.

## Raw pointers

`*const T` and `*mut T` are raw pointers, and they opt out of all of the above. Because of that,
the operations that can go wrong with them require `unsafe`:

- dereferencing one
- indexing through one
- reading a field through one
- calling an `unsafe fn`

A function that takes a caller's raw pointer and dereferences it should be declared `unsafe fn`, so
the obligation is visible in the signature rather than buried in the body. Marking a function
`unsafe` does not allow raw pointer operations in its body. Each one still needs an `unsafe` block,
as in Unsafe and FFI.

```rust
unsafe fn read_first(p : *const i32) -> i32 {
    return unsafe { *p };
}

fn main() -> i32 {
    let x : i32 = 42;
    unsafe {
        return read_first(&x) - 42;
    }
}
```

Keep `unsafe` blocks small. The point of the annotation is that the region a human has to verify by
hand is written down and searchable.

# Generics and traits

Generics in Vx are monomorphized: each instantiation becomes its own concrete function or type at
compile time. There is no boxing, no vtable and no dynamic dispatch unless you ask for it.

## Generic types and functions

```rust
struct Pair<T> {
    first: T,
    second: T,
}

enum Maybe<T> {
    Just(T),
    Nothing,
}
```

Instantiate by naming the argument:

```rust
enum Maybe<T> {
    Just(T),
    Nothing,
}

fn main() -> i32 {
    let m = Maybe<i32>::Just(42);

    let mut result = 0;
    match m {
        Maybe<i32>::Just(val) => { result = val; },
        Maybe<i32>::Nothing => { result = 0; },
    }
    return result;
}
```

## Generic impls

An `impl` block can be generic, or specific to one instantiation:

```rust
impl<T> Pair<T> {
    fn first(self: &Pair<T>) -> T {
        return self.first;
    }
}

impl Pair<i32> {
    fn sum(self: &Pair<i32>) -> i32 {
        return self.first + self.second;
    }
}
```

When several impls could apply, the most specific one wins.

## Bounds

Constrain a parameter with `:`, and write several with `+`:

```rust
impl<T : Float> Tensor<T, [?, ?]> {
    // ...
}
```

```rust
fn describe<T : Doubles + Tags>(x : T) -> i32 {
    return x.twice() + x.tag();
}
```

A bound can give the trait type arguments, or say what its associated types must be. A
bound may also name another parameter's associated type:

```rust
fn first<I : Iterator<Item = i64>>(it : I) -> i64 { /* ... */ }
fn total<I : Iterator, S : Sum<I::Item>>(it : I) -> S { /* ... */ }
```

Every bound has to hold at the call site, and each one that does not is reported. Note
that a bound constrains *callers*, not the body: a generic body is checked after
monomorphization against the concrete type, so it can call any method that type has,
whether or not a bound named it.

## Traits

A trait method may carry a body. An impl that does not write that method inherits it:

```rust
trait Counts {
    fn value(self : Self) -> i32;
    fn doubled(self : Self) -> i32 {
        return self.value() * 2;
    }
}
```

`self.value()` inside the default dispatches to whichever impl inherited it, so one body
serves every implementor. An impl that writes `doubled` itself keeps its own.

Traits describe shared behaviour, and are implemented with `impl ... for`:

```rust
impl<T> Iterator for VecIter<T> {
    type Item = &T;
    // fn next(self : &mut VecIter<T>) -> Option<&T> { ... }
}
```

That is how `Vec` participates in `for` loops and in the iterator adaptors — `map` and friends are
ordinary generic functions over the `Iterator` trait rather than compiler magic.

## Const generics

A *value* can be a type parameter, not only a type. This is what makes statically-shaped tensors
work:

```rust
fn dot<const N : i32>(a : Tensor<f32, [N]>, b : Tensor<f32, [N]>) -> f32 {
    let mut acc = 0.0;
    for i in 0..N {
        acc += a[i] * b[i];
    }
    return acc;
}
```

Because `N` is part of the type, passing two tensors of different lengths to `dot` is a compile
error rather than a runtime check you forgot to write. Each distinct `N` monomorphizes to its own
function, so the loop bound is a constant the optimizer can see.

## How this stays fast to compile

Every symbol, nominal type and monomorphized instantiation is identified by a flat 256-bit
identifier rather than by a pointer into a shared tree. Combined with a nominal type system and
mandatory boxing for recursive types, that decouples modules from one another: the frontend
resolves and checks them in parallel across cores, with no query engine, no locks and no shared
mutable state.

The observable consequence is that the same source produces byte-identical MLIR whether it is
compiled serially or in parallel — which is asserted in the test suite rather than assumed.
