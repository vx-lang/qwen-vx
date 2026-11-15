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

# Tasks and channels

A Vx program can do several things at the same time on the CPU. It starts each one as a task. This
chapter shows how to start tasks, how they share data, and how they send values to each other.
What Vx guarantees about concurrency lists what you can rely on when tasks run at
the same time, and how we check each promise.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## A `tasks` block and `spawn`

A `tasks` block holds tasks. `spawn { .. }` starts a task, and the task runs on a thread of its
own. The block ends only after each of its tasks ends. So a task can borrow the locals of the
code around it: the locals live longer than the task.

`spawn` gives a handle to the task. `join()` waits for the task and gives the value of its last
expression. A `return` in the task ends the task early, and its value is then the value of the
task. The value of the block is the value of its last expression too.

This program adds the numbers 1 to 100 with two tasks. Each task borrows the vector `v` from `main`
and adds one half of it (`book_tasks_split_the_work.vx`):

```rust
import std::vec;

fn sum_range(v : &Vec<i64>, from : i32, to : i32) -> i64 {
  let mut sum : i64 = 0;
  for i in from..to {
    sum = sum + *v.get_ref(i);
  }
  return sum;
}

fn main() -> i32 {
  let mut v = Vec<i64>::new();
  for i in 1..101 {
    v.push(i as i64);
  }
  let total = tasks {
    let low = spawn {
      sum_range(&v, 0, 50)
    };
    let high = spawn {
      sum_range(&v, 50, 100)
    };
    low.join() + high.join()
  };
  println!("total {}", total);
  return 0;
}
```

It prints `total 5050`.

A task must not outlive its block, so the compiler checks two rules:

- A `spawn` outside a `tasks` block gets the error E3056. A `spawn` in a closure, or directly in
  another task, is outside the block too. Give the inner tasks a `tasks` block of their own.
- The only use of a task handle is `h.join()`. A program cannot return a handle, store it, or pass
  it to a function (E3057).

## A task that fails

A task fails when its value is the `Err` of a `Result`. Then its `join()` leaves the block, and that
`Err` is the value of the block. A `join()` of a task that did not fail gives the value inside the
`Ok`. So the code of the block works with plain values, and the block gives the first failure
(`book_tasks_join_a_failure.vx`). A `?` in a task ends the task with the `Err`:

```rust
import core::result;

// A read that fails for an even number above 2.
fn read(n : i32) -> Result<i32, i32> {
  if n > 2 && n % 2 == 0 {
    return Result<i32, i32>::Err(n);
  }
  return Result<i32, i32>::Ok(n * 10);
}

fn read_two(a : i32, b : i32) -> Result<i32, i32> {
  let r = tasks {
    let x = spawn {
      read(a)
    };
    let y = spawn {
      read(b)
    };
    // Each `join` gives the `Ok` value. An `Err` leaves the block, and is its value.
    let a = x.join();
    let b = y.join();
    Result<i32, i32>::Ok(a + b)
  };
  return r;
}

fn show(a : i32, b : i32) -> void {
  match read_two(a, b) {
    Result<i32, i32>::Ok(v) => {
      println!("read {} and {}: Ok {}", a, b, v);
    }
    Result<i32, i32>::Err(e) => {
      println!("read {} and {}: Err {}", a, b, e);
    }
  }
}

fn main() -> i32 {
  show(1, 2);
  show(1, 4);
  return 0;
}
```

It prints:

```text
read 1 and 2: Ok 30
read 1 and 4: Err 4
```

The block must join each task that can fail, in a statement of its own that always runs. Otherwise
the compiler gives the error E3060. A task that fails also cancels the other tasks of its block, so
that they do not wait for nothing. The section on channels below says what a cancel does.

## What a task can take

A task can take a value from the code around it in two ways:

- It moves the value. Then the type of the value must be `Send`.
- It borrows the value, while the code around it can use the value too. Then the type must be
  `Sync`, because two threads can read it at the same time.

The compiler finds both traits from the fields of a type. A number, a `bool` and a struct of them
are `Send` and `Sync`. A raw pointer is neither, so a type with a raw pointer field must say what
it allows with `unsafe impl`. The library does this for `Vec<T>`, the atomics and the types below.

For example, two tasks cannot borrow one `Receiver` of a channel, because two `recv` calls at the
same time would take the same value. The compiler says:

```text
Error[E3055]: the task reads 'rx', which the code around the task can read at the same time, so
its type 'Receiver<i64>' must be `Sync`: field 'words' of 'Receiver<i64>' has type
'*mut AtomicI64', and '*mut AtomicI64' is a raw pointer, which is not `Sync`
```

The rule depends only on the types, never on the body of a function. So the compiler can check each
function on its own, and in parallel.

## Atomics that tasks share

An atomic is a number that many tasks can change at the same time, from `core::sync::atomic`. Each
task here borrows the same counter and adds 1 to it 10000 times (`book_tasks_share_an_atomic.vx`):

```rust
import core::sync::atomic;

fn add_ones(counter : &AtomicI64, n : i32) -> void {
  for _i in 0..n {
    counter.fetch_add(1, AtomicOrdering::Relaxed);
  }
}

fn main() -> i32 {
  let counter = AtomicI64::new(0);
  tasks {
    for _t in 0..4 {
      spawn {
        add_ones(&counter, 10000);
      };
    }
  };
  println!("count {}", counter.load(AtomicOrdering::SeqCst));
  return 0;
}
```

It prints `count 40000`. A plain `i64` in place of the atomic does not compile: a task cannot
write a value that another task can read at the same time.

### Memory orders

Each operation on an atomic takes an `AtomicOrdering`: `Relaxed`, `Release`, `Acquire`, `AcqRel`
or `SeqCst`. They mean what they mean in Rust and C++. The counter above uses `Relaxed`, because
only the count matters. To pass other data from one task to another, write it, and then store a
flag with `Release`. The task that reads the flag with `Acquire` then sees the data
(`book_release_and_acquire.vx`):

```rust
import core::sync::atomic;

fn main() -> i32 {
  let data = AtomicI64::new(0);
  let ready = AtomicBool::new(false);
  tasks {
    spawn {
      data.store(42, AtomicOrdering::Relaxed);
      ready.store(true, AtomicOrdering::Release);
    };
    spawn {
      while !ready.load(AtomicOrdering::Acquire) {}
      println!("data {}", data.load(AtomicOrdering::Relaxed));
    };
  };
  return 0;
}
```

It prints `data 42`.

One difference from Rust: Vx compiles each `Relaxed` read as an `Acquire` read. This rules out
values "out of thin air", which C++ and Rust allow on AArch64 and on GPUs. On x86 it costs
nothing. Do not rely on it for order: ask for `Acquire` where you need it.
`-fweak-relaxed-atomics` keeps `Relaxed` reads relaxed. Use it with `--sanitize=thread` to check
that a program does not depend on the stronger read. The full rules are in
[the semantics of Vx](https://github.com/vx-lang/Vx/blob/main/docs/lang/semantics.md#25-atomic-orders).

## Channels

A channel moves values from tasks that send to one task that receives. `channel<T>(n)` makes a
channel with room for `n` values, and gives a `Sender<T>` and a `Receiver<T>`:

- `send(v)` puts `v` in the channel. It waits while the channel is full.
- `recv()` takes the next value. It waits while the channel is empty. It gives `Closed` when the
  program dropped each `Sender` and the channel holds no value.
- Clone the `Sender` to send from many tasks. A channel has one `Receiver`.

The values of one sender arrive in the order that it sent them. The type `T` must be `Send`, and it
must not hold a reference: a value in a channel can outlive the data that a reference points to.

In this program, four tasks send squares, and the code of the block receives them
(`book_channel_from_workers.vx`):

```rust
import std::sync::channel;

fn send_squares(tx : Sender<i64>, from : i64) -> void {
  for i in from..from + 10 {
    match tx.send(i * i) {
      Result<void, SendError<i64>>::Ok() => {}
      Result<void, SendError<i64>>::Err(_e) => {
        println!("the receiver is gone");
      }
    }
  }
}

fn main() -> i32 {
  let (tx, rx) = channel<i64>(4);
  let mut count : i64 = 0;
  let mut sum : i64 = 0;
  tasks {
    for w in 0..4 {
      let mine = tx.clone();
      spawn {
        send_squares(mine, (w as i64) * 10);
      };
    }
    // The channel closes when the last `Sender` is dropped.
    drop(tx);
    let mut open = true;
    while open {
      match rx.recv() {
        Result<i64, RecvError>::Ok(v) => {
          count = count + 1;
          sum = sum + v;
        }
        Result<i64, RecvError>::Err(_e) => {
          open = false;
        }
      }
    }
  };
  println!("got {} values, sum {}", count, sum);
  return 0;
}
```

It prints `got 40 values, sum 20540`.

### Tensors in a channel

A channel can carry tensors, also tensors in the memory of a GPU. A `send` moves only the handle of
the tensor, and its data stays where it is. Here a task sends six tensors that live in the GPU
memory `W`, and the code of the block copies each one back to read it
(`book_channel_of_gpu_tensors.vx`):

```rust
import std::sync::channel;

Memory CPU_DRAM {}
Memory W {
  within: Memory::CPU_DRAM, capacity: 64 KiB, managed: explicit, scope: device
}
Topology Dev {
  arch: nvptx64,
  memory: Memory::W,
  visible: [Memory::W],
  transfer Memory::CPU_DRAM -> Memory::W : 10,
  transfer Memory::W -> Memory::CPU_DRAM : 10
}

fn main() -> i32 {
  // Room for 4 tensors of 4 KiB each.
  let (tx, rx) = channel<Tensor<f32, [1024], Memory::W>>(4);
  let mut count = 0;
  let mut sum = 0.0;
  tasks {
    spawn {
      for i in 0..6 {
        let t = transfer(Tensor<f32, [1024]>::fill(i as f32), Memory::W);
        let _ = tx.send(t);
      }
      drop(tx);
    };
    let mut open = true;
    while open {
      match rx.recv() {
        Result<Tensor<f32, [1024], Memory::W>, RecvError>::Ok(t) => {
          let here = transfer(t, Memory::CPU_DRAM);
          count = count + 1;
          sum = sum + here[0];
        }
        Result<Tensor<f32, [1024], Memory::W>, RecvError>::Err(_e) => {
          open = false;
        }
      }
    }
  };
  println!("got {} tensors, sum {}", count, sum);
  return 0;
}
```

It prints `got 6 tensors, sum 15`.

The channel can hold 4 tensors at the same time, so the compiler counts 4 tensors in `W` from the
`channel` call. The check of the memory capacity adds them to the other tensors in `W`. With room
for 20 tensors, the program does not compile:

```text
Error[E6009]: the room of this channel, 20 of `Tensor<f32, [1024], Memory::W>`, needs 81920 bytes
but memory space 'W' has capacity 65536 bytes
```

### Cancellation

When the code of a `tasks` block leaves it early, with `return` or `break`, or when a task fails,
the block cancels its tasks. A wait that can be cancelled then gives an error at once: `recv()`
gives `Cancelled`, and `send(v)` gives `v` back with `cancelled` set. The task should then end. The
compiler warns about a loop that waits again after a `Cancelled` (W1035), because each wait in a
cancelled task gives `Cancelled` at once, and such a loop never ends.

A `join()` and a `lock()` cannot be cancelled.

When each task waits and nothing can wake one, the program has a deadlock. The runtime then stops
the program with the message `vx: deadlock: ...`, so that it does not hang. It never reports a
deadlock for a program that can still continue.

A `drop` runs with cancellation off, so that a cleanup can finish. So a wait in a `drop` should
have a time limit: `send_for(v, limit)` and `recv_for(limit)` take a `Duration`, and give
`timed_out` and `TimedOut` after it. The compiler warns about a `send` or `recv` in a `drop` that
has no limit (W1036).

A `drop` also has a limit of its own. Its type says how long its waits can take, with a method of
`Drop` that returns 10 seconds unless the type writes its own:

```rust
impl Drop for SlotGuard {
  fn drop(self : &mut SlotGuard) -> void {
    // The cleanup. Its waits get the limit below.
  }

  fn wait_limit(self : &SlotGuard) -> Duration {
    return Duration::from_millis(50);
  }
}
```

After the limit, a `send` or `recv` in the `drop` gives up as in a cancelled block. This includes
the waits of the tasks that the `drop` spawns.
`VX_DROP_WAIT_SECONDS` can make the limit of each `drop` shorter, for a test or to find a `drop`
that hangs, but never longer.

`within(limit, body)`, from `std::sync::wait`, gives the waits in the closure `body` one time
limit, in a `drop` or anywhere else. It gives the value of `body`:

```rust
let flushed = within(Duration::from_millis(50), || -> bool {
  // A flush, a close and a send here share the 50 ms.
  true
});
```

The limit also holds in the functions that `body` calls, and in the tasks that it spawns. After the limit, a wait gives up as in a
cancelled block, and `cancelled()` is true until `body` returns. Outside a `drop`, a cancel still
stops the waits in `body`. A `return` in `body` ends `body`, and not the function around the call.

## `select`

`Selector::new(receivers)` takes a list of receivers, and `select()` on it takes the first value
from any of them. It gives the value and the index of its channel in the list. It gives `Closed`
when each channel is closed and empty. Each search starts after the channel of the last value, so a
channel that always has a value does not hide the others (`book_select_from_two_channels.vx`):

```rust
import std::sync::channel;
import std::vec;

fn send_three(tx : Sender<i64>) -> void {
  let first : i64 = 0;
  for i in first..3 {
    match tx.send(i) {
      Result<void, SendError<i64>>::Ok() => {}
      Result<void, SendError<i64>>::Err(_e) => {}
    }
  }
}

fn main() -> i32 {
  let (tx0, rx0) = channel<i64>(1);
  let (tx1, rx1) = channel<i64>(1);
  let mut rs = Vec<Receiver<i64>>::new();
  rs.push(rx0);
  rs.push(rx1);
  let mut sel = Selector<i64>::new(rs);
  let mut counts : Tensor<i64, [2]> = [0, 0];
  tasks {
    spawn {
      send_three(tx0);
    };
    spawn {
      send_three(tx1);
    };
    let mut open = true;
    while open {
      match sel.select() {
        Result<Selected<i64>, RecvError>::Ok(s) => {
          counts[s.index] = counts[s.index] + 1;
        }
        // Both channels are closed and empty.
        Result<Selected<i64>, RecvError>::Err(_e) => {
          open = false;
        }
      }
    }
  };
  println!("from 0: {} values, from 1: {} values", counts[0], counts[1]);
  return 0;
}
```

It prints `from 0: 3 values, from 1: 3 values`.

So for `n` channels, if one channel has a value at each of `n` calls in a row, one of those calls
takes a value from it. What Vx guarantees about concurrency lists this promise
and the others about tasks, and says how we check each one.

## `Mutex`

A `Mutex<T>` guards a value that many tasks change. `lock()` waits until no other task holds the
lock, and gives a guard. `get()` on the guard gives a pointer to the value. The drop of the guard
unlocks the `Mutex` (`book_mutex_guards_a_total.vx`):

```rust
import std::sync::mutex;

fn add_to(total : &Mutex<i64>, n : i32) -> void {
  for _i in 0..n {
    // The guard unlocks the `Mutex` when it is dropped, at the end of each pass.
    let mut guard = total.lock();
    let value = guard.get();
    *value = *value + 1;
  }
}

fn main() -> i32 {
  let total = Mutex<i64>::new(0);
  tasks {
    for _t in 0..4 {
      spawn {
        add_to(&total, 5000);
      };
    }
  };
  println!("total {}", total.into_inner());
  return 0;
}
```

It prints `total 20000`. Use an atomic for one number, and a `Mutex` when a change touches more
than one number.

## `Arc`

A task can borrow the locals of its parent, so most programs need no `Arc`. Use an `Arc<T>` when a
value must live longer than its `tasks` block. `clone()` gives another `Arc` that shares the value,
and `get()` gives `&T`. The drop of the last `Arc` drops the value. It works as Rust's `Arc` does.
An `Arc<T>` can move to another task only when `T` is `Send` and `Sync`
(`book_arc_outlives_its_task.vx`):

```rust
import std::sync::arc;
import std::sync::channel;

struct Settings {
  scale : i64,
}

fn load(tx : Sender<Arc<Settings>>) -> void {
  let settings = Arc<Settings>::new(Settings {
    scale : 10
  });
  // A clone shares the value and adds 1 to the count of owners.
  match tx.send(settings.clone()) {
    Result<void, SendError<Arc<Settings>>>::Ok() => {}
    Result<void, SendError<Arc<Settings>>>::Err(_e) => {}
  }
  match tx.send(settings) {
    Result<void, SendError<Arc<Settings>>>::Ok() => {}
    Result<void, SendError<Arc<Settings>>>::Err(_e) => {}
  }
}

fn receive(rx : &Receiver<Arc<Settings>>) -> Arc<Settings> {
  match rx.recv() {
    Result<Arc<Settings>, RecvError>::Ok(got) => {
      return got;
    }
    Result<Arc<Settings>, RecvError>::Err(_e) => {
      panic("the task sent nothing");
    }
  }
}

fn main() -> i32 {
  let (tx, rx) = channel<Arc<Settings>>(2);
  tasks {
    spawn {
      load(tx);
    };
  };
  let a = receive(&rx);
  let b = receive(&rx);
  println!("scale {}, owners {}", a.get().scale, a.strong_count());
  // The drop of the last owner drops the value.
  drop(b);
  println!("owners {}", a.strong_count());
  return 0;
}
```

It prints `scale 10, owners 2`, and then `owners 1`. To change a shared value, put a `Mutex` in the
`Arc`.

## How tasks run

By default, each task runs on an OS thread of its own. With `VX_TASK_RUNTIME=shared`, the program
runs many tasks on a few OS threads instead. Each task then gets a stack of its own, and a task
that waits gives its thread to another task. A program needs no change to use the shared runtime.
Set these variables when the program starts:

- `VX_TASK_RUNTIME=threads` gives the default: one OS thread for each task. A value other than
  `shared` or `threads` stops the program with a message.
- `VX_TASK_WORKERS=n` starts `n` threads. Without it, the runtime starts one thread for each CPU.
- A task that blocks in a C function, or runs a long loop with no wait, holds its thread. When
  tasks are ready and each thread is busy, the runtime starts one more thread.
  `VX_TASK_MAX_WORKERS=n` sets the largest number of threads.
- `VX_TASK_SEED=n` runs all tasks on one thread, and the seed picks the next task to run. The same
  seed gives the same order each time, so a bug that depends on the order happens again in the
  next run. Try other seeds to find such a bug. A seed cannot go with `VX_TASK_WORKERS` or
  `VX_TASK_MAX_WORKERS`.

## Finding a data race at run time

The compiler cannot see a race through a raw pointer in `unsafe` code. To find one, run the program
with `vxc --sanitize=thread program.vx`. LLVM's ThreadSanitizer then checks each memory access of
the program. It prints `WARNING: ThreadSanitizer: data race` for each race, and the program fails.

ThreadSanitizer sees the start and the end of a task, the atomics, and the waits of channels and
`Mutex`, with both runtimes. It finds only the races that happen in that run. The option works
only when `vxc` runs the program. On Ubuntu, the runtime of ThreadSanitizer is in the package
`libclang-rt-22-dev`.

## Why Vx has no `async`

In Vx, a function that waits is a normal function. `recv`, `join`, `lock` and a network read wait
for an event, and the task continues after it. A function that calls them is a normal function too.
So Vx has no `async` and no `await`.

In a language with `async`, a function that waits must be `async`, and so must each function that
calls it. Such a function has a "color", and a program must keep the two colors apart. Vx does not
have that problem. A borrow that lives across a wait is a borrow across a call, so the borrow
checker needs no new rule.

A task that waits holds no OS thread in the shared runtime, so a program can have many waiting
tasks. The language is the same for both runtimes.

## Compared with Rust and Go

| | Vx | Rust | Go |
| --------------------------- | ----------------------------------- | ------------------------------------ | -------------------------------- |
| Start a task | `spawn` in a `tasks` block | `thread::scope` and `spawn` | `go f()` |
| A task borrows locals | Yes, the block outlives its tasks | Yes, in `thread::scope` | No borrows, a race is possible |
| What can move to a task | `Send`, from the types | `Send`, from the types | Anything |
| A task that fails | Its `join` leaves the block | `join` gives a `Result` | Each program writes it |
| Cancel the other tasks | When the block leaves or a task fails | Each program writes it | `context`, by hand |
| Functions that wait | Normal functions | `async fn`, or a blocked thread | Normal functions |
| Wait on many channels | `select` over a list of receivers | A crate, such as `crossbeam` | `select` statement |
