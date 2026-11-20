#!/usr/bin/env python3
"""Write a tiny tokenizer.json with the pre-tokenizer of Qwen, for tests that cannot download one.

    python3 scripts/make_tiny_tokenizer.py tiny.json
    python3 scripts/make_tiny_tokenizer.py --texts texts.bin
    python3 scripts/make_tiny_tokenizer.py --check-decoded decoded.bin

The tokenizer has the 256 byte tokens, 160 merges learned from the text TRAINING, 3 merges across
the ends of pieces, and the added tokens <|endoftext|>, <|im_start|>, <|im_end|> and <think>. Only the standard library of Python
makes it, so it has the same bytes on every machine. The `tokenizers` package reads it as it reads
the tokenizer of Qwen.

--texts writes the texts of TEXTS, separated by NUL bytes, for `tests/tokenize.vx`. Together they
use each alternative of the regex, NFC, and the added tokens. --check-decoded compares what
`tests/tokenize.vx --decode` wrote with the NFC form of each text.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

import convert_tokenizer

TRAINING = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fn main() {
    let total = price * 3 + tax_rate;
    println!("{}", total);
}

The tokenizer splits the text into pieces, and then merges the bytes of each piece.
It's the same tokenizer that they're using, and we'll test it here.
"""

ADDED = [("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True), ("<think>", False)]

# Each text uses a part of the tokenizer that the others do not.
TEXTS = [
    "def fibonacci(n):\n    return n",
    "It's here. THEY'RE 'LL 'Ve 'd 'm 't 'S and the long s'ſr, x'Sreturn x'sthe",
    "x=1;y==2!!\n\n\nz 1abc if n <= 1 {\n",
    "   three spaces, then a tab\tand two at the end  ",
    "a\r\n\r\n  b \n\n  c\n",
    "<|im_start|>user\nWrite Rust.<|im_end|>\n<|im_start|>assistant\n<think>",
    "<|im_start <|im_end|><|im_end|>x<|endoftext|>",
    "café café ậ ậ a̅́ Å 각 각 क़",
    "日本語 123 ١٢٣ Ⅷ x² \U0001f600\U0001f44d\U0001f3fd  　end",
    "Αθήνα Москва שלום مرحبا",
    "",
]


def train(text, merges):
    """BPE merges learned from `text`: each time, the most frequent pair in its pieces. The pieces
    come from the regex of Qwen for ASCII text, so that merges join spaces and new lines to words,
    and a piece with the wrong end gets other tokens."""
    to_char = convert_tokenizer.bytes_to_unicode()
    ascii_pattern = r"'s|'t|'re|'ve|'m|'ll|'d|[^\r\na-zA-Z0-9]?[a-zA-Z]+|[0-9]| ?[^\sa-zA-Z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    words = collections.Counter(re.findall(ascii_pattern, text))
    split = {w: [to_char[b] for b in w.encode("utf-8")] for w in words if w}
    vocab = [to_char[b] for b in range(256)]
    learned = []
    for _ in range(merges):
        pairs = collections.Counter()
        for w, parts in split.items():
            for a, b in zip(parts, parts[1:]):
                pairs[(a, b)] += words[w]
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        learned.append(best)
        vocab.append(best[0] + best[1])
        for w, parts in split.items():
            out, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                    out.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            split[w] = out
    return vocab, learned


def tokenizer_json():
    vocab, merges = train(TRAINING, 160)
    # Merges across the ends of two pieces: a digit and a letter, and the long s and a letter.
    # The regex never puts them in one piece, so with a correct pre-tokenizer they never apply.
    # The vocabularies of Qwen have no such tokens, so only this tokenizer shows a wrong end there.
    to_char = convert_tokenizer.bytes_to_unicode()
    long_s = to_char[0xC5] + to_char[0xBF]
    for left, right in [("1", "a"), (to_char[0xC5], to_char[0xBF]), (long_s, "r")]:
        merges.append((left, right))
        vocab.append(left + right)
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": len(vocab) + i, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": special}
            for i, (content, special) in enumerate(ADDED)
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": convert_tokenizer.QWEN2_PATTERN}, "behavior": "Isolated", "invert": False},
            byte_level,
        ]},
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": {token: i for i, token in enumerate(vocab)},
            "merges": [f"{a} {b}" for a, b in merges],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", nargs="?", help="the tokenizer.json to write")
    parser.add_argument("--texts", help="write the test texts, separated by NUL bytes, to this file")
    parser.add_argument("--check-decoded", help="compare this output of tests/tokenize.vx --decode with the texts")
    args = parser.parse_args()
    if args.output:
        Path(args.output).write_text(json.dumps(tokenizer_json(), ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
    if args.texts:
        Path(args.texts).write_bytes(b"\0".join(t.encode("utf-8") for t in TEXTS))
    if args.check_decoded:
        decoded = Path(args.check_decoded).read_bytes().split(b"\0")
        # Each text ends with a NUL byte, so the last part is empty.
        assert decoded[-1].strip() == b"", "the decoded output does not end with a NUL byte"
        decoded = decoded[:-1]
        expected = [unicodedata.normalize("NFC", t).encode("utf-8") for t in TEXTS]
        same = sum(a == b for a, b in zip(decoded, expected))
        print(f"decoded {len(decoded)} texts, {same} equal to the NFC form of the text")
        return 0 if same == len(expected) == len(decoded) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
