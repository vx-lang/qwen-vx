#!/usr/bin/env python3
"""Write the 12 code prompts that measure decoding with drafts, as full chat text.

    python3 scripts/make_draft_prompts.py VX_CHECKOUT OUT_DIR

There are 3 prompts of each of 4 kinds: edit a file, complete a file, write new code and explain
code. The files to edit, complete and explain come from the Vx repository. So the prompts depend
on the Vx commit: the benchmark of the design (docs/discussions/predictive_pipelining.md in the Vx
repository) used Vx at 7fb49be57. llama.cpp's `-f` drops the last newline of a prompt file, so
give llama.cpp a copy with one more newline at the end.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import pathlib
import sys


def chat(user):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


def prompts(vx):
    def source(rel):
        return (vx / rel).read_text()

    def head(rel, share=0.6):
        lines = source(rel).splitlines(keepends=True)
        return "".join(lines[: int(len(lines) * share)])

    def fenced(rel, text):
        lang = {"py": "python", "rs": "rust", "c": "c"}[rel.rsplit(".", 1)[1]]
        return f"`{rel}`:\n\n```{lang}\n{text}```"

    edit = "Output the whole file, and change nothing else."
    complete = (
        "This file stops in the middle. Continue it from exactly where it stops. "
        "Output only the rest of the file."
    )
    explain = "Explain what this code does, function by function."
    return {
        "edit1": f"{fenced('scripts/tools/gen_error_index.py', source('scripts/tools/gen_error_index.py'))}\n\n"
        f"Add a type hint to every function parameter and return value. {edit}",
        "edit2": f"{fenced('src/hash.rs', source('src/hash.rs'))}\n\n"
        f"Add a `///` doc comment above every function that has none. {edit}",
        "edit3": f"{fenced('runtime/vx_mlir_shims.c', source('runtime/vx_mlir_shims.c'))}\n\n"
        f"Add a one-line comment above every function that says what it does. {edit}",
        "complete1": f"{fenced('src/report.rs', head('src/report.rs'))}\n\n{complete}",
        "complete2": f"{fenced('scripts/tools/check_doc_examples.py', head('scripts/tools/check_doc_examples.py'))}\n\n"
        f"{complete}",
        "complete3": f"{fenced('src/session.rs', head('src/session.rs'))}\n\n{complete}",
        "write1": "Write a Python module with a class LRUCache that has get, put and a capacity, "
        "built on OrderedDict. Add pytest tests for it.",
        "write2": "Write a C function `int parse_csv_line(const char *line, char **fields, int max_fields)` "
        "that handles quoted fields and doubled quotes inside them. Add a main() that tests it.",
        "write3": "Write a Rust struct RingBuffer<T> with new, push, pop, len, is_empty and an iterator. "
        "Add unit tests.",
        "explain1": f"{fenced('src/suggest.rs', source('src/suggest.rs'))}\n\n{explain}",
        "explain2": f"{fenced('src/decline.rs', source('src/decline.rs'))}\n\n{explain}",
        "explain3": f"{fenced('src/error.rs', source('src/error.rs'))}\n\n{explain}",
    }


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    vx, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    texts = prompts(vx)
    for name, user in texts.items():
        (out / f"{name}.txt").write_text(chat(user))
    print(len(texts), "prompts in", out)


if __name__ == "__main__":
    main()
