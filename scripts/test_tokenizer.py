#!/usr/bin/env python3
"""Compare the tokenizer of `src/tokenizer.vx` with the `tokenizers` package of Hugging Face.

    python3 scripts/test_tokenizer.py

For the tokenizer of each model folder in MODELS, the test converts tokenizer.json, encodes a
corpus with `tests/tokenize.vx`, and asserts that each text gets the ids of `tokenizers`. Then it
decodes those ids and asserts that each text comes back in its NFC form. The corpus has the files
of this repository, Python, Rust and C++ sources, and texts made from random Unicode characters.
QWEN_TOKENIZER_CORPUS_MB sets the size of the source files (default 4 MB).

The tests skip when `tokenizers`, `vxc` or the model folders are missing, as in CI.
`tests/tokenize.vx` checks a tiny tokenizer without them.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import bisect
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unicodedata
import unittest
from pathlib import Path

import convert_tokenizer

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

ROOT = Path(__file__).resolve().parent.parent
MODELS = [Path("~/models/Qwen2.5-Coder-0.5B-Instruct").expanduser(),
          Path("~/models/Qwen3-Coder-30B-A3B-Instruct").expanduser()]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|fim_prefix|>", "<think>", "</tool_call>"]


def vx_fleet():
    fleet = os.environ.get("VX_FLEET")
    if fleet:
        return Path(fleet)
    home = Path(os.environ.get("VX_HOME", "~/.vx")).expanduser()
    return home / "current" / "fleet" if (home / "current").is_dir() else home / "fleet"


def source_files(limit):
    """The texts of this repository and of Python, Rust and C++ sources, up to `limit` bytes."""
    groups = [
        sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "worktrees" not in p.parts),
        sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")),
        sorted(Path(p) for p in glob.glob(os.path.expanduser("~/.rustup/toolchains/*/lib/rustlib/src/rust/library/*/src/**/*.rs"), recursive=True)),
        sorted(Path(p) for p in glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1/**/*", recursive=True) if Path(p).is_file()),
        sorted(Path(p) for p in glob.glob("/usr/include/c++/*/**/*", recursive=True) if Path(p).is_file()),
    ]
    texts = []
    for files in groups:
        size = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            text = text.replace("\0", "")
            texts.append(text)
            size += len(text.encode("utf-8"))
            if size > limit // len(groups):
                break
    return texts


def random_texts(count, seed=1):
    """Texts of random characters, chosen so that each step of the tokenizer gets hard cases."""
    rng = random.Random(seed)
    white = [chr(c) for c in [9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]]
    marks = [chr(c) for c in [0x300, 0x301, 0x302, 0x303, 0x308, 0x323, 0x327, 0x328, 0x331, 0x345, 0x5B0, 0x93C, 0x3099, 0x309A]]
    pools = [
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
        white,
        marks,
        ["'s", "'S", "'t", "'re", "'RE", "'ve", "'m", "'ll", "'LL", "'d", "'ſ", "'K"],
        [chr(c) for c in range(0xC0, 0x250)],
        [chr(c) for c in range(0x370, 0x530)],
        [chr(c) for c in range(0x4E00, 0x4E00 + 400)] + [chr(c) for c in range(0x3040, 0x30FF)],
        [chr(c) for c in range(0xAC00, 0xAC00 + 300)] + [chr(c) for c in range(0x1100, 0x1200)],
        [chr(c) for c in [0x660, 0x661, 0x6F9, 0x966, 0xFF11, 0x2160, 0x2167, 0xB2, 0xBD, 0x2155, 0x1D7CE]],
        [chr(c) for c in [0x1F600, 0x1F44D, 0x1F3FD, 0x200D, 0x2764, 0xFE0F, 0x1F468, 0x1F469, 0x1F467]],
        [chr(c) for c in [0x212B, 0x2126, 0xF900, 0xFB01, 0x958, 0x2ADC, 0x1E9B, 0x344, 0xFEFF, 0xAD]],
        SPECIALS + ["<|im_start", "im_end|>", "<|", "|>", "<think"],
    ]
    texts = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randrange(1, 60)):
            if rng.random() < 0.05:
                c = rng.randrange(1, 0x110000)
                if not 0xD800 <= c <= 0xDFFF:
                    parts.append(chr(c))
            else:
                pool = rng.choice(pools)
                parts.append("".join(rng.choice(pool) for _ in range(rng.randrange(1, 6))))
        texts.append("".join(parts))
    texts += [" " * 5000, "\n" * 300 + "x", "=" * 3000, "9" * 1000, " \t" * 400 + "!", "a" * 4000,
              "".join(chr(c) for c in range(0x20, 0x3000) if not 0xD800 <= c <= 0xDFFF)]
    return texts


def run_vx(*args):
    out = subprocess.run(["vxc", str(ROOT / "tests" / "tokenize.vx"), "--package", f"qwen={ROOT / 'src'}",
                          "--host", "default", "--machine", str(vx_fleet() / "m4-uma.vx"), "--", *map(str, args)],
                         capture_output=True, check=True)
    return out.stdout


@unittest.skipIf(Tokenizer is None, "needs the tokenizers package")
@unittest.skipIf(shutil.which("vxc") is None, "needs vxc on PATH")
class TokenizerAgainstHuggingFace(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        limit = int(float(os.environ.get("QWEN_TOKENIZER_CORPUS_MB", "4")) * 1e6)
        cls.texts = source_files(limit) + random_texts(3000)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_model(self, model):
        if not (model / "tokenizer.json").is_file():
            self.skipTest(f"needs {model}")
        tmp = Path(self.tmp.name)
        vxt = tmp / (model.name + ".vxt")
        vxt.write_bytes(convert_tokenizer.build(json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))))
        texts_file = tmp / "texts.bin"
        texts_file.write_bytes(b"\0".join(t.encode("utf-8") for t in self.texts))
        lines = [l for l in run_vx(vxt, texts_file).decode().split("\n") if not l.startswith("[JIT]")]
        end = lines.index(f"texts={len(self.texts)};")
        got = [[int(x) for x in l.split(",")] if l else [] for l in lines[:end]]
        expected = [e.ids for e in Tokenizer.from_file(str(model / "tokenizer.json")).encode_batch(self.texts)]
        self.assertEqual(len(got), len(expected))
        bad = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        tokens = sum(len(e) for e in expected)
        size = sum(len(t.encode("utf-8")) for t in self.texts)
        print(f"\n{model.name}: {len(self.texts) - len(bad)} of {len(self.texts)} texts equal, "
              f"{size} bytes, {tokens} tokens", file=sys.stderr)
        for i in bad[:3]:
            k = next(j for j in range(min(len(got[i]), len(expected[i])) + 1) if j >= min(len(got[i]), len(expected[i])) or got[i][j] != expected[i][j])
            print(f"text {i} {self.texts[i][:80]!r}: first difference at id {k}: "
                  f"vx {got[i][k:k + 8]} tokenizers {expected[i][k:k + 8]}", file=sys.stderr)
        self.assertEqual(bad, [], "the Vx tokenizer gave other ids than tokenizers")

        # Decoding gives the text after the NFC step of `tokenizers`. The JIT of vxc prints its
        # lines before the program's output.
        ids_file = tmp / "ids.txt"
        ids_file.write_text("".join(",".join(map(str, e)) + "\n" for e in expected))
        out = run_vx(vxt, ids_file, "--decode")
        while out.startswith(b"[JIT]"):
            out = out[out.index(b"\n") + 1:]
        decoded = out.split(b"\0")[:len(self.texts)]
        normalizer = Tokenizer.from_file(str(model / "tokenizer.json")).normalizer
        normalized = [normalizer.normalize_str(t).encode("utf-8") for t in self.texts]
        self.assertEqual(len(decoded), len(self.texts))
        self.assertEqual([i for i, (a, b) in enumerate(zip(decoded, normalized)) if a != b], [], "decoding did not give the normalized text")

    def test_qwen25_coder(self):
        self.check_model(MODELS[0])

    def test_qwen3_coder(self):
        self.check_model(MODELS[1])


@unittest.skipIf(Tokenizer is None, "needs the tokenizers package")
class UnicodeTablesAgainstHuggingFace(unittest.TestCase):

    def test_classes_equal_those_of_the_regex_engine(self):
        """The regex engine of `tokenizers` puts each code point in the class that the file says."""
        path = next((m / "tokenizer.json" for m in MODELS if (m / "tokenizer.json").is_file()), None)
        if path is None:
            self.skipTest("needs a Qwen tokenizer.json")
        pre = Tokenizer.from_file(str(path)).pre_tokenizer
        letters, numbers, spaces, *_ = convert_tokenizer.unicode_tables()

        def inside(ranges, c):
            i = bisect.bisect_right(ranges, (c, 0x110000)) - 1
            return i >= 0 and ranges[i][0] <= c <= ranges[i][1]

        # "a" + c is one piece only for a letter c, c + c is two pieces only for a number, and
        # c + "!" is two pieces only for white space (and " !" is one).
        wrong = []
        for c in range(0x110000):
            if 0xD800 <= c <= 0xDFFF:
                continue
            ch = chr(c)
            letter = len(pre.pre_tokenize_str("a" + ch)) == 1
            number = not letter and len(pre.pre_tokenize_str(ch + ch)) == 2
            space = c == 32 or (not letter and not number and len(pre.pre_tokenize_str(ch + "!")) == 2)
            want = (inside(letters, c), inside(numbers, c), inside(spaces, c))
            if (letter, number, space) != want:
                wrong.append(hex(c))
        self.assertEqual(wrong[:20], [])

    def test_old_nfc_data_lists_what_tokenizers_lacks(self):
        """The marks and composites that the NFC step of `tokenizers` treats as unknown are those
        that `convert_tokenizer.py` leaves out."""
        path = next((m / "tokenizer.json" for m in MODELS if (m / "tokenizer.json").is_file()), None)
        if path is None:
            self.skipTest("needs a Qwen tokenizer.json")
        norm = Tokenizer.from_file(str(path)).normalizer

        def differs(text):
            return norm.normalize_str(text) != unicodedata.normalize("NFC", text)

        # A mark of class k > 0 moves before U+0301 (230) when k < 230, and after U+0316 (220)
        # when k > 220. A mark of class 0 does neither. "#" composes with no mark.
        marks = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and unicodedata.combining(chr(c))
                 and (differs("#́" + chr(c)) or differs("#" + chr(c) + "̖"))]
        composites = []
        for c in range(0x110000):
            if 0xD800 <= c <= 0xDFFF:
                continue
            mapping = unicodedata.decomposition(chr(c))
            parts = [int(x, 16) for x in mapping.split()] if mapping and not mapping.startswith("<") else []
            if len(parts) == 2 and differs(chr(parts[0]) + chr(parts[1])):
                composites.append(c)
        listed = [c for first, last in convert_tokenizer.OLD_NFC_MARKS for c in range(first, last + 1)]
        self.assertEqual([hex(c) for c in marks], [hex(c) for c in listed])
        self.assertEqual([hex(c) for c in composites], [hex(c) for c in convert_tokenizer.OLD_NFC_COMPOSITES])


if __name__ == "__main__":
    unittest.main()
