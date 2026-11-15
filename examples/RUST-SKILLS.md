# Rust skills

Rules for Rust code that compiles the first time, with no warnings.

## Import the trait of each method

Many methods come from a trait, and a call compiles only when that trait is imported.

- `read_to_string`, `read_exact` come from `std::io::Read`.
- `lines`, `read_line` on a reader come from `std::io::BufRead`.
- `write_all`, `flush`, and `write!` on a writer come from `std::io::Write`.
- `write!` into a `String` comes from `std::fmt::Write`.

So start a program that reads standard input with `use std::io::{self, BufRead, Read, Write};`,
and remove the imports it does not use.

## Name the lifetime of a stored reference

- A method such as `fn items(&self) -> Vec<&T>` needs no lifetime names.
- A helper that puts a reference from one argument into another argument needs a named
  lifetime: `fn walk<'a>(node: &'a Option<Box<Node<T>>>, out: &mut Vec<&'a T>)`.
- A struct that holds a reference has a lifetime parameter: `struct Words<'a> { rest: &'a str }`.
  Its `impl<'a> Iterator for Words<'a>` can return `&'a str` items that borrow from the input.

## Ownership

- Move a value out of an `Option` field with `.take()`.
- Walk an `Option<Box<Node<T>>>` with `as_ref()` to read and `as_mut()` to change it.
- Write `for x in &v` to borrow a vector, and `for x in v` only to consume it.

## Collections

- Count with `*map.entry(key).or_insert(0) += 1;`.
- Sort on two keys with `sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)))`.

## Before you reply

- For each method call, name the trait that provides it, and check that it is imported.
- For each reference that goes into a struct or a collection, check that it lives long enough.
- Remove each `mut` that the code does not need.
