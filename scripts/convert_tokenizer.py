#!/usr/bin/env python3
"""Convert the tokenizer.json of a Qwen2 or Qwen3 model into the tokenizer file of the Qwen app.

    python3 scripts/convert_tokenizer.py ~/models/Qwen2.5-Coder-0.5B-Instruct qwen.vxt

The Vx program `src/tokenizer.vx` reads the file. It turns text into token ids as the Hugging Face
`tokenizers` package does for these models:

1. It splits out each added token, such as <|im_start|>, where the text has it. When two added
   tokens start at the same place, the longer one wins.
2. It applies Unicode NFC normalization to the text between the added tokens.
3. It splits that text with the pre-tokenizer regex of tokenizer.json (below as PATTERN).
4. It turns each piece into its UTF-8 bytes, and each byte into its token.
5. It merges the pair of neighbor tokens with the lowest merge rank, again and again, until no
   pair of the piece has a merge. Of equal pairs, the leftmost goes first.

This script reads only tokenizer.json, with no package beyond the standard library of Python. The
Unicode tables come from `unicodedata`. The classes of the regex equal those of the regex engine
of `tokenizers`. Its NFC step has older data, so the NFC tables leave out the marks and composites
that the older data lacks (OLD_NFC_MARKS and OLD_NFC_COMPOSITES). `scripts/test_tokenizer.py`
checks both.

The file is little-endian. It starts with 32 i64 words:

    0      the bytes "VXQTOK01"
    1      the format version, 1
    2      the number of token ids, n_tokens
    3      the number of merges
    4      the number of added tokens
    5-7    the number of ranges of letters (\\p{L}), numbers (\\p{N}) and white space (\\s)
    8      the number of code points with a canonical decomposition
    9      the number of code points in all those decompositions
    10     the number of code points with a canonical combining class above 0
    11     the number of canonical compositions
    12     the number of bytes of all tokens
    13-30  the byte offset of each table below, in this order
    31     0

The tables, each at a multiple of 8 bytes:

    13  byte_tokens       i32 [256]          the token of each byte
    14  token_starts      i32 [n_tokens + 1] where the bytes of each token start in token_bytes
    15  token_bytes       u8                 the bytes of every token, one token after another
    16  merge_pairs       i64 [merges]       left << 20 | right, sorted
    17  merge_ranks       i32 [merges]       the rank of each pair: its line in tokenizer.json
    18  merge_tokens      i32 [merges]       the token that the merge makes
    19  added_tokens      i32 [added]        the id of each added token
    20  added_special     i32 [added]        1 for a special token, such as <|im_end|>, else 0
    21  letters           i32 [2 * ranges]   the first and last code point of each range, sorted
    22  numbers           i32 [2 * ranges]
    23  spaces            i32 [2 * ranges]
    24  decomposed        i32 [count]        code points with a canonical decomposition, sorted
    25  decomposition_starts  i32 [count + 1] where each decomposition starts in decompositions
    26  decompositions    i32                the full canonical decompositions, one after another
    27  combining         i32 [count]        code points with a combining class above 0, sorted
    28  combining_class   i32 [count]        their combining classes
    29  composition_pairs i64 [count]        first << 21 | second, sorted
    30  compositions      i32 [count]        the code point that each pair composes to

The Hangul syllables are not in the decomposition and composition tables. `tokenizer.vx`
computes them, as the Unicode standard describes.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path

MAGIC = b"VXQTOK01"
VERSION = 1
HEADER_WORDS = 32

# The pre-tokenizer regex of the Qwen2 and Qwen3 tokenizers. `tokenizer.vx` matches it with
# hand-written code, so this script refuses a tokenizer with another one.
PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
           r"|\s*[\r\n]+|\s+(?!\S)|\s+")

# The code points of the Unicode property White_Space, which \s matches.
WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680),
               (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F),
               (0x3000, 0x3000)]

HANGUL_FIRST, HANGUL_LAST = 0xAC00, 0xD7A3

# The NFC step of `tokenizers` 0.23 has the data of an old version of Unicode. It gives combining
# class 0 to these marks, which later versions added, and it does not compose to these
# composites. The tables leave out both, so that NFC here gives the text of `tokenizers`.
# `test_tokenizer.py` measures both lists again.
OLD_NFC_MARKS = [
    (0x7FD, 0x7FD), (0x897, 0x89F), (0x8CA, 0x8D3), (0x9FE, 0x9FE), (0xC3C, 0xC3C), (0xD3B, 0xD3C),
    (0xEBA, 0xEBA), (0x1715, 0x1715), (0x1ABF, 0x1ACE), (0x1DF6, 0x1DFA), (0xA82C, 0xA82C),
    (0x10D24, 0x10D27), (0x10D69, 0x10D6D), (0x10EAB, 0x10EAC), (0x10EFD, 0x10EFF), (0x10F46, 0x10F50),
    (0x10F82, 0x10F85), (0x11070, 0x11070), (0x1133B, 0x1133B), (0x113CE, 0x113D0), (0x1145E, 0x1145E),
    (0x11839, 0x1183A), (0x1193D, 0x1193E), (0x11943, 0x11943), (0x119E0, 0x119E0), (0x11A34, 0x11A34),
    (0x11A47, 0x11A47), (0x11A99, 0x11A99), (0x11D42, 0x11D42), (0x11D44, 0x11D45), (0x11D97, 0x11D97),
    (0x11F41, 0x11F42), (0x1612F, 0x1612F), (0x16FF0, 0x16FF1), (0x1E08F, 0x1E08F), (0x1E130, 0x1E136),
    (0x1E2AE, 0x1E2AE), (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF), (0x1E5EE, 0x1E5EF),
]
OLD_NFC_COMPOSITES = [
    0x105C9, 0x105E4, 0x11383, 0x11385, 0x1138E, 0x11391, 0x113C5, 0x113C7, 0x113C8, 0x11938,
    0x16121, 0x16122, 0x16123, 0x16124, 0x16125, 0x16126, 0x16127, 0x16128, 0x16D68, 0x16D69, 0x16D6A,
]


def bytes_to_unicode():
    """The character that the byte-level step of GPT-2 gives each byte, as in `tokenizers`."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def check_structure(t):
    """Asserts that tokenizer.json describes the tokenizer that `tokenizer.vx` computes."""
    assert t["normalizer"] == {"type": "NFC"}, f"the normalizer must be NFC, not {t['normalizer']}"
    pre = t["pre_tokenizer"]
    assert pre["type"] == "Sequence" and len(pre["pretokenizers"]) == 2, "the pre-tokenizer must be a split and a byte-level step"
    split, byte_level = pre["pretokenizers"]
    assert split["type"] == "Split" and split["pattern"] == {"Regex": PATTERN}, "the pre-tokenizer has another regex"
    assert split["behavior"] == "Isolated" and not split["invert"], "the split must keep each match as a piece"
    assert byte_level["type"] == "ByteLevel" and not byte_level["add_prefix_space"] and not byte_level["use_regex"], "the byte-level step must add no space and use no regex"
    model = t["model"]
    assert model["type"] == "BPE" and model.get("dropout") is None, "the model must be BPE without dropout"
    assert not model.get("byte_fallback") and not model.get("ignore_merges"), "the BPE model must merge every piece, with no byte fallback"
    assert not model.get("continuing_subword_prefix") and not model.get("end_of_word_suffix"), "the BPE model must have no prefix or suffix"
    for a in t["added_tokens"]:
        assert not (a["normalized"] or a["lstrip"] or a["rstrip"] or a["single_word"]), f"the added token {a['content']} must match the text as it is"


def ranges(points):
    """The sorted code points as a list of (first, last) ranges."""
    out = []
    for p in sorted(points):
        if out and out[-1][1] == p - 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


def unicode_tables():
    """The character classes of the regex, and the tables of NFC."""
    letters, numbers = [], []
    decomposed, combining, compositions = {}, {}, {}
    old_marks = {cp for first, last in OLD_NFC_MARKS for cp in range(first, last + 1)}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        category = unicodedata.category(c)
        if category.startswith("L"):
            letters.append(cp)
        elif category.startswith("N"):
            numbers.append(cp)
        if unicodedata.combining(c) and cp not in old_marks:
            combining[cp] = unicodedata.combining(c)
        if HANGUL_FIRST <= cp <= HANGUL_LAST or cp in OLD_NFC_COMPOSITES:
            continue
        nfd = unicodedata.normalize("NFD", c)
        if nfd != c:
            decomposed[cp] = [ord(x) for x in nfd]
        mapping = unicodedata.decomposition(c)
        if mapping and not mapping.startswith("<"):
            parts = [int(x, 16) for x in mapping.split()]
            # A primary composite: NFC turns its two-part decomposition back into it.
            if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == c:
                compositions[(parts[0] << 21) | parts[1]] = cp
    return ranges(letters), ranges(numbers), WHITE_SPACE, decomposed, combining, compositions


def build(t):
    """The bytes of the tokenizer file for the parsed tokenizer.json `t`."""
    check_structure(t)
    to_char = bytes_to_unicode()
    to_byte = {c: b for b, c in to_char.items()}
    vocab = t["model"]["vocab"]
    tokens = {}
    for text, i in vocab.items():
        tokens[i] = bytes(to_byte[c] for c in text)
    added = sorted(t["added_tokens"], key=lambda a: a["id"])
    for a in added:
        content = a["content"].encode("utf-8")
        assert a["id"] not in tokens or tokens[a["id"]] == content, f"the added token {a['content']} has the id of another token"
        tokens[a["id"]] = content
    n_tokens = max(tokens) + 1
    byte_tokens = [vocab[to_char[b]] for b in range(256)]

    # A later line of a pair overwrites an earlier one, as in `tokenizers`.
    merges = {}
    for rank, m in enumerate(t["model"]["merges"]):
        left, right = m.split(" ") if isinstance(m, str) else m
        assert left in vocab and right in vocab and left + right in vocab, f"the merge {left} {right} has a token that is not in the vocabulary"
        merges[(vocab[left] << 20) | vocab[right]] = (rank, vocab[left + right])
    merge_pairs = sorted(merges)

    letters, numbers, spaces, decomposed, combining, compositions = unicode_tables()
    starts, blob = [], bytearray()
    for i in range(n_tokens):
        starts.append(len(blob))
        blob += tokens.get(i, b"")
    starts.append(len(blob))
    decomposed_keys = sorted(decomposed)
    decomposition_starts, decompositions = [], []
    for cp in decomposed_keys:
        decomposition_starts.append(len(decompositions))
        decompositions += decomposed[cp]
    decomposition_starts.append(len(decompositions))
    combining_keys = sorted(combining)
    composition_keys = sorted(compositions)

    def i32(values):
        return struct.pack(f"<{len(values)}i", *values)

    def i64(values):
        return struct.pack(f"<{len(values)}q", *values)

    tables = [
        i32(byte_tokens),
        i32(starts),
        bytes(blob),
        i64(merge_pairs),
        i32([merges[k][0] for k in merge_pairs]),
        i32([merges[k][1] for k in merge_pairs]),
        i32([a["id"] for a in added]),
        i32([1 if a["special"] else 0 for a in added]),
        i32([x for r in letters for x in r]),
        i32([x for r in numbers for x in r]),
        i32([x for r in spaces for x in r]),
        i32(decomposed_keys),
        i32(decomposition_starts),
        i32(decompositions),
        i32(combining_keys),
        i32([combining[k] for k in combining_keys]),
        i64(composition_keys),
        i32([compositions[k] for k in composition_keys]),
    ]
    offsets, body = [], bytearray()
    position = HEADER_WORDS * 8
    for table in tables:
        offsets.append(position + len(body))
        body += table
        body += b"\0" * (-len(body) % 8)
    counts = [n_tokens, len(merge_pairs), len(added), len(letters), len(numbers), len(spaces),
              len(decomposed_keys), len(decompositions), len(combining_keys), len(composition_keys), len(blob)]
    header = [int.from_bytes(MAGIC, "little"), VERSION] + counts + offsets + [0]
    assert len(header) == HEADER_WORDS
    return i64(header) + bytes(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", help="a Hugging Face model folder with tokenizer.json, or the file itself")
    parser.add_argument("output", help="the tokenizer file to write")
    args = parser.parse_args()
    path = Path(args.model_dir).expanduser()
    if path.is_dir():
        path = path / "tokenizer.json"
    data = build(json.loads(path.read_text(encoding="utf-8")))
    Path(args.output).write_bytes(data)
    print(f"wrote {args.output}: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
