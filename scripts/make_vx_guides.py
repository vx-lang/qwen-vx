#!/usr/bin/env python3
"""Make the Vx guides of docs/vx-guides from the book of a Vx checkout.

    python3 scripts/make_vx_guides.py ~/go/Vx

Give the checkout of the Vx release that runs the programs, such as the tag v0.0.4, so that the
guide shows no feature that the compiler does not have yet. A chapter that the book of that
release does not have is left out.

Qwen3-Coder learned no Vx: the language came after the model. A guide is a system prompt that
teaches it the Vx of one kind of program, with the chapters of the book (www/book/src) for it.
Each guide starts with docs/vx-guides/parts/header.md, the rules that a model with Rust habits
gets wrong, and the chapters "Vx for Rust programmers" and "Vx in one page". The GPU guide ends
with docs/vx-guides/parts/apple-gpu.md.

The book takes its programs from the tests of Vx with `{{#include FILE:program}}`, and this
script puts the lines between `// ANCHOR: program` and `// ANCHOR_END: program` in their place.
A link to another chapter keeps only its text, because the model cannot follow it. The script
also writes the table of docs/vx-guides/README.md: the size of each guide, its tokens at about
3 characters a token, and its KV cache.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDES = ROOT / "docs" / "vx-guides"

# The chapters that start every guide.
BASE = ["for-rust-programmers.md", "quick-reference.md"]

# Each guide: its file, what it is for, and its chapters after BASE. "apple-gpu" is the part of
# this repository.
TOPICS = [
    ("vx-basics.md", "Programs on the CPU: errors, ownership and generics",
     ["errors.md", "ownership.md", "generics.md"]),
    ("vx-gpu.md", "Kernels on the GPU of a Mac: topologies, memory, blocks, lanes and tiles",
     ["heterogeneous.md", "gpu.md", "apple-gpu"]),
    ("vx-tasks.md", "Concurrent programs: tasks and channels",
     ["tasks.md", "writing-parallel-code.md"]),
    ("vx-tensors.md", "Numeric programs: tensor operations, low precision and autodiff",
     ["tensor-ops.md", "low-precision-numbers.md", "exact-and-fast-math.md", "autodiff.md"]),
    ("vx-unsafe.md", "Systems programs: unsafe code, C functions, files, input and time",
     ["unsafe-and-ffi.md", "files-input-time.md"]),
]

# The tokens of Qwen for each character of these guides, from the first guide: 20374 characters
# gave 6870 tokens in the chat format.
CHARS_PER_TOKEN = 3.0
# Bytes of KV cache for each token of Qwen3-Coder-30B-A3B: 48 layers, K and V, 4 heads of 128
# values, f16.
KV_BYTES_PER_TOKEN = 48 * 2 * 4 * 128 * 2

INCLUDE = re.compile(r"\{\{#include ([^}:]+):([A-Za-z0-9_-]+)\}\}")
LINK = re.compile(r"\[([^\]]+)\]\((?!https?:)[^)]*\)")
DOCTEST = re.compile(r"^<!-- vx-doctest:.*-->\n\n?", re.M)


def anchored(path, anchor):
    """The lines of `path` between the ANCHOR and ANCHOR_END lines of `anchor`."""
    lines, inside = [], False
    for line in path.read_text().splitlines():
        if line.strip() == f"// ANCHOR: {anchor}":
            inside = True
        elif line.strip() == f"// ANCHOR_END: {anchor}":
            inside = False
        elif inside and "// ANCHOR" not in line:
            lines.append(line)
    if not lines:
        sys.exit(f"{path}: no lines for the anchor {anchor}")
    return "\n".join(lines)


def chapter(book, name):
    path = book / name
    text = path.read_text()
    text = INCLUDE.sub(lambda m: anchored((path.parent / m.group(1)).resolve(), m.group(2)), text)
    text = DOCTEST.sub("", text)
    return LINK.sub(r"\1", text).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("vx", help="a checkout of vx-lang/Vx")
    args = parser.parse_args()
    vx = Path(args.vx).expanduser()
    book = vx / "www" / "book" / "src"
    if not book.is_dir():
        sys.exit(f"{book} is not a folder: give a checkout of vx-lang/Vx")
    commit = subprocess.run(["git", "-C", str(vx), "describe", "--tags", "--always"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    parts = GUIDES / "parts"
    rows = []
    for name, purpose, chapters in TOPICS:
        pieces = [(parts / "header.md").read_text()]
        for c in BASE + chapters:
            if c == "apple-gpu":
                pieces.append((parts / "apple-gpu.md").read_text())
            elif (book / c).exists():
                pieces.append(chapter(book, c))
            else:
                # An older book may not have the chapter yet.
                print(f"{name}: the book has no {c}, so the guide goes on without it")
        text = "\n".join(pieces)
        (GUIDES / name).write_text(text)
        tokens = round(len(text) / CHARS_PER_TOKEN, -2)
        rows.append(f"| [`{name}`]({name}) | {purpose} | {len(text) // 1000} KB | about {int(tokens):,} "
                    f"| {tokens * KV_BYTES_PER_TOKEN / 1e9:.1f} GB |")
        print(f"wrote {GUIDES / name}: {len(text)} characters")
    readme = GUIDES / "README.md"
    text = readme.read_text()
    start, end = "<!-- guides: start -->\n", "<!-- guides: end -->\n"
    table = ("| Guide | For | Size | Tokens | KV cache |\n| --- | --- | --- | --- | --- |\n"
             + "\n".join(rows) + f"\n\nMade from the book of vx-lang/Vx at {commit}.\n")
    a, b = text.index(start) + len(start), text.index(end)
    readme.write_text(text[:a] + table + text[b:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
