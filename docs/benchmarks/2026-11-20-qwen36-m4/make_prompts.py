#!/usr/bin/env python3
"""Write the prompts of this benchmark into prompts/ beside this script.

    python3 make_prompts.py MODEL_DIR VX_CHECKOUT

MODEL_DIR is the Hugging Face folder of Qwen3.6-35B-A3B. VX_CHECKOUT is a clone of the Vx
repository. The script reads its files at two commits with `git show`, so the clone can be at any
commit.

- context/ctxN.txt: the first N tokens of the sources of this app, at QWEN_VX_COMMIT.
- twelve/: the 12 prompts of scripts/make_draft_prompts.py, on Vx at TWELVE_COMMIT. Table T of
  run.sh chooses the draft lengths with these prompts.
- heldout/: 12 other prompts of the same 4 kinds, on Vx at HELDOUT_COMMIT. The benchmark chooses
  no setting with them.

The prompts of twelve/ and heldout/ are chat text, as the chat template of Qwen3.6 writes it with
enable_thinking=false. The script checks this against the template. Each prompt is in vx/ and in
llama/. The -f option of llama.cpp drops the last new line of a file, so the llama/ copy has one
more new line at the end. prompts/tokens.csv gives the number of tokens of each prompt.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import csv
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import make_draft_prompts  # noqa: E402

CONTEXTS = [64, 512, 2048, 8192, 16384, 32768]
QWEN_VX_COMMIT = "8b12928"
TWELVE_COMMIT = "7fb49be57"
HELDOUT_COMMIT = "9d7cf3a49"


class GitTree:
    """The files of `repo` at `commit`, with the `/` and `read_text` of a pathlib.Path."""

    def __init__(self, repo, commit, rel=""):
        self.repo, self.commit, self.rel = repo, commit, rel

    def __truediv__(self, rel):
        return GitTree(self.repo, self.commit, f"{self.rel}/{rel}" if self.rel else rel)

    def read_text(self):
        cmd = ["git", "-C", str(self.repo), "show", f"{self.commit}:{self.rel}"]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def heldout_prompts(vx):
    """12 prompts of the same kinds as those of make_draft_prompts.py, on other files and tasks."""

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
    e1 = "scripts/tools/differential_fuzz/generators/tensors.py"
    e2 = "src/hir/check/host_waits.rs"
    e3 = "utils/memalg/measure_link.c"
    return {
        "edit1": f"{fenced(e1, source(e1))}\n\nAdd a type hint to every function parameter and return value. {edit}",
        "edit2": f"{fenced(e2, source(e2))}\n\nAdd a `///` doc comment above every function that has none. {edit}",
        "edit3": f"{fenced(e3, source(e3))}\n\nAdd a one-line comment above every function that says what it does. {edit}",
        "complete1": f"{fenced('src/hir/flatten/affine.rs', head('src/hir/flatten/affine.rs'))}\n\n{complete}",
        "complete2": f"{fenced('utils/memalg/shakedown.py', head('utils/memalg/shakedown.py'))}\n\n{complete}",
        "complete3": f"{fenced('stdlib/rust_core/src/io.rs', head('stdlib/rust_core/src/io.rs'))}\n\n{complete}",
        "write1": "Write a Python function merge_intervals(intervals) that merges overlapping closed intervals "
        "and returns them sorted. Add pytest tests for it, with empty input and touching intervals.",
        "write2": "Write a C function `void reverse_words(char *s)` that reverses the order of the words in a "
        "string in place, with single spaces between words. Add a main() that tests it.",
        "write3": "Write a Rust struct Trie with new, insert, contains and starts_with for lowercase ASCII words. "
        "Add unit tests.",
        "explain1": f"{fenced('src/hir/places.rs', source('src/hir/places.rs'))}\n\n{explain}",
        "explain2": f"{fenced('src/hir/check/drop_waits.rs', source('src/hir/check/drop_waits.rs'))}\n\n{explain}",
        "explain3": f"{fenced('src/hir/decl_check/bounds.rs', source('src/hir/decl_check/bounds.rs'))}\n\n{explain}",
    }


def app_sources():
    """The .vx files of src/ at QWEN_VX_COMMIT, one after the other, model.vx first."""
    repo = HERE.parents[2]
    cmd = ["git", "-C", str(repo), "ls-tree", "--name-only", QWEN_VX_COMMIT, "src/"]
    names = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.split()
    names = sorted((n for n in names if n.endswith(".vx")), key=lambda n: (n != "src/model.vx", n))
    tree = GitTree(repo, QWEN_VX_COMMIT)
    return "".join((tree / n).read_text() for n in names)


def first_tokens(tok, text, n):
    """The text of the first n tokens of `text`. Asserts that this text has exactly those tokens."""
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= n, f"the sources have only {len(ids)} tokens, fewer than {n}"
    cut = tok.decode(ids[:n], skip_special_tokens=False)
    assert tok.encode(cut, add_special_tokens=False).ids == ids[:n], f"the first {n} tokens do not encode back to themselves"
    return cut


def check_chat_format(model_dir, user):
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(model_dir)
    messages = [{"role": "user", "content": user}]
    want = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert chat(user) == want, f"the chat text differs from the template of {model_dir}:\n{want!r}"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    from tokenizers import Tokenizer

    model_dir, vx = pathlib.Path(sys.argv[1]).expanduser(), pathlib.Path(sys.argv[2]).expanduser()
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    out = HERE / "prompts"
    texts = {}
    sources = app_sources()
    for n in CONTEXTS:
        texts[f"context/ctx{n}"] = first_tokens(tok, sources, n)
    twelve = make_draft_prompts.prompts(GitTree(vx, TWELVE_COMMIT))
    check_chat_format(model_dir, twelve["write1"])
    for name, user in twelve.items():
        texts[f"twelve/{name}"] = chat(user)
    for name, user in heldout_prompts(GitTree(vx, HELDOUT_COMMIT)).items():
        texts[f"heldout/{name}"] = chat(user)
    rows = []
    for key, text in texts.items():
        kind, name = key.split("/")
        for side, extra in (("vx", ""), ("llama", "\n")):
            path = out / kind / side / f"{name}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + extra)
        rows.append({"prompt": key, "tokens": len(tok.encode(text, add_special_tokens=False).ids)})
    with open(out / "tokens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["prompt", "tokens"])
        w.writeheader()
        w.writerows(rows)
    print(len(texts), "prompts in", out)


if __name__ == "__main__":
    main()
