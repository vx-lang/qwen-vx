#!/usr/bin/env python3
"""Replay the greedy outputs that record_experts.cpp recorded, as rounds of decoding with drafts.

    python3 scripts/simulate_drafts.py DATA_DIR [EXTRA_MS]

For each prompt it reads <name>.tokens and <name>.experts, and simulates speculative decoding with
a draft copied from the context. A draft token is accepted while it equals the model's own greedy
token. The experts of a round are the union of the recorded experts of its positions. The cost of
a round comes from the decode profile of 2026-11-17 (#34), plus EXTRA_MS (0.25 if not given) for
each position after the first.

It prints the experts that n positions in a row use, the speedup of each draft policy for each
kind of prompt, and the tokens that a round makes. Step 0.2 of the design explains the cost
model: docs/discussions/predictive_pipelining.md in the Vx repository.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import pathlib
import sys

import numpy as np

LAYERS, USED, EXPERTS = 48, 8, 128
# From the profile: a token takes 22.0 ms at a short context and 32.0 ms at 8,000 positions.
# The experts take 10.7 ms of it, for 8 x 48 = 384 experts.
PER_EXPERT = 10.7 / (USED * LAYERS)
REST = {"short": 22.0 - 10.7, "8k": 32.0 - 10.7}
PLAIN = {"short": 22.0, "8k": 32.0}
CLOSERS = set(" \t\n)]}>;,:.\"'`")


def load(data, name):
    raw = np.fromfile(data / f"{name}.tokens", dtype=np.int32)
    prompt, tokens = int(raw[0]), raw[1:]
    experts = np.fromfile(data / f"{name}.experts", dtype=np.int32)
    experts = experts.reshape(-1, LAYERS, USED)
    # Row r holds the position prompt - 1 + r.
    assert len(experts) == len(tokens) - prompt, (name, len(experts), len(tokens), prompt)
    return prompt, tokens.tolist(), experts


def load_pieces(data):
    """The text of each token id, from vocab.bin."""
    raw = (data / "vocab.bin").read_bytes()
    pieces, at = [], 0
    while at < len(raw):
        n = int.from_bytes(raw[at : at + 4], "little")
        pieces.append(raw[at + 4 : at + 4 + n].decode("utf-8", "replace"))
        at += 4 + n
    return pieces


class CopyDraft:
    """Finds the latest earlier match of the last tokens, for the longest of LENGTHS, and copies
    the tokens that followed it."""

    LENGTHS = (16, 12, 8, 6, 4, 3, 2, 1)

    def __init__(self, tokens):
        self.seq = tokens
        self.index = {n: {} for n in self.LENGTHS}
        self.known = 0  # n-grams that end before this position are in the index

    def _grow(self, upto):
        # Index each n-gram that ends at e < upto - 1, with the position after it.
        for e in range(self.known, upto - 1):
            for n in self.LENGTHS:
                if e - n + 1 >= 0:
                    self.index[n][tuple(self.seq[e - n + 1 : e + 1])] = e + 1
        self.known = max(self.known, upto - 1)

    def draft(self, i, k):
        """A draft of up to k tokens after seq[:i], and the length of the match it came from."""
        self._grow(i)
        for n in self.LENGTHS:
            if i - n < 0:
                continue
            at = self.index[n].get(tuple(self.seq[i - n : i]))
            if at is None:
                continue
            out = []
            src = at
            while len(out) < k:
                tok = self.seq[src] if src < i else out[src - i]
                out.append(tok)
                src += 1
            return out, n
        return [], 0


def union(experts, first, n):
    """The sum over layers of the number of different experts at positions first..first+n-1."""
    window = experts[first : first + n]
    return sum(len(np.unique(window[:, layer, :])) for layer in range(LAYERS))


def predictable(tokens, prompt, pieces):
    """For each position, whether a copy draft of one token predicts it, or it is only spaces and
    closing punctuation."""
    draft = CopyDraft(tokens)
    flags = [False] * len(tokens)
    for i in range(prompt + 1, len(tokens)):
        d, matched = draft.draft(i, 1)
        copy = matched >= 2 and d and d[0] == tokens[i]
        flags[i] = bool(copy) or set(pieces[tokens[i]]) <= CLOSERS
    return flags


def replay(prompt, tokens, experts, policy, extra=0.25, pieces=None):
    """Rounds over the output. Returns (tokens made, ms at short context and at 8k, the tokens that
    each round made).

    A policy is (k, minimum): a copy draft of k tokens when the match has at least `minimum`
    tokens. k can be a function of the match length. ("perfect", k) drafts the next k tokens of
    the output, and ("copy+syntax", k) drafts up to k tokens that `predictable` marks."""
    draft = CopyDraft(tokens)
    end = len(tokens)
    i = prompt + 1  # tokens[prompt] came from the prompt's pass
    made = 0
    ms = {"short": 0.0, "8k": 0.0}
    per_round = []
    flags = predictable(tokens, prompt, pieces) if policy[0] == "copy+syntax" else None
    while i < end:
        k, minimum = policy
        if k == "perfect":
            d = tokens[i : i + minimum]
        elif k == "copy+syntax":
            d = []
            while len(d) < minimum and i + len(d) < end and flags[i + len(d)]:
                d.append(tokens[i + len(d)])
        else:
            d, matched = draft.draft(i, 8 if callable(k) else k)
            if matched < minimum:
                d = []
            elif callable(k):
                d = d[: k(matched)]
        d = d[: max(0, end - i - 1)]
        m = 0
        while m < len(d) and d[m] == tokens[i + m]:
            m += 1
        n = len(d) + 1
        row = i - 1 - (prompt - 1)
        u = union(experts, row, min(n, len(experts) - row))
        for c in ms:
            ms[c] += REST[c] + PER_EXPERT * u + extra * (n - 1)
        got = min(m + 1, end - i)
        made += got
        per_round.append(got)
        i += got
    return made, ms, per_round


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__.split("\n\n")[1])
    data = pathlib.Path(sys.argv[1])
    extra = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    pieces = load_pieces(data)
    names = sorted(p.stem for p in data.glob("*.tokens"))
    loaded = {n: load(data, n) for n in names}

    print("Experts in a layer for n positions in a row, measured against independent picks:")
    for n in (1, 2, 3, 4, 6, 8):
        vals = []
        for prompt, tokens, experts in loaded.values():
            for first in range(0, len(experts) - n + 1, 3):
                vals.append(union(experts, first, n) / LAYERS)
        indep = EXPERTS * (1 - (1 - USED / EXPERTS) ** n)
        print(f"  n={n}: {np.mean(vals):5.1f} measured, {indep:5.1f} independent")

    policies = {f"k={k}, match>={m}": (k, m) for k in (1, 2, 3, 4, 6, 8) for m in (1, 2, 3, 4)}
    policies["k by match length"] = (lambda n: 0 if n == 0 else (8 if n >= 8 else (4 if n >= 4 else 2)), 2)
    policies["k by match, no 1-2"] = (lambda n: 0 if n == 0 else (8 if n >= 8 else (5 if n >= 4 else 2)), 3)
    policies["BOUND copy+syntax 8"] = ("copy+syntax", 8)
    policies["BOUND perfect 4"] = ("perfect", 4)
    policies["BOUND perfect 8"] = ("perfect", 8)

    kinds = sorted({n.rstrip("0123456789") for n in names})
    print(f"\nTokens per second against plain decoding (extra {extra} ms for each extra position):")
    print("policy".ljust(22) + "".join(f"{k:>16}" for k in kinds) + f"{'all':>16}")
    print(" " * 22 + "".join(f"{'short   8k':>16}" for _ in kinds + ["all"]))
    best = {}
    for label, policy in policies.items():
        row = label.ljust(22)
        for kind in kinds + ["all"]:
            made = 0
            ms = {"short": 0.0, "8k": 0.0}
            for name in names:
                if kind != "all" and not name.startswith(kind):
                    continue
                m, t, _ = replay(*loaded[name], policy, extra, pieces)
                made += m
                for c in ms:
                    ms[c] += t[c]
            s, l = PLAIN["short"] * made / ms["short"], PLAIN["8k"] * made / ms["8k"]
            row += f"{s:8.2f}{l:8.2f}"
            best.setdefault(kind, []).append((s, label))
        print(row)
    print("\nBest policy for each kind at a short context:")
    for kind, vals in best.items():
        s, label = max(vals)
        print(f"  {kind}: {s:.2f} with {label}")

    print("\nTokens a round made, with 'k by match length':")
    for name in names:
        made, _, per = replay(*loaded[name], policies["k by match length"], extra, pieces)
        per = np.array(per)
        print(f"  {name}: {made} tokens in {len(per)} rounds, mean {per.mean():.2f}, "
              f"rounds of 4 or more {np.mean(per >= 4):.0%}")

    print("\nEach generated token: a copy draft (match of 2 or more) predicts it, or it is only "
          "spaces and closing punctuation, or neither:")
    for kind in kinds + ["all"]:
        counts = {"copy": 0, "syntax": 0, "neither": 0}
        for name in names:
            if kind != "all" and not name.startswith(kind):
                continue
            prompt, tokens, _ = loaded[name]
            draft = CopyDraft(tokens)
            for i in range(prompt + 1, len(tokens)):
                d, matched = draft.draft(i, 1)
                if matched >= 2 and d and d[0] == tokens[i]:
                    counts["copy"] += 1
                elif set(pieces[tokens[i]]) <= CLOSERS:
                    counts["syntax"] += 1
                else:
                    counts["neither"] += 1
        total = sum(counts.values())
        print(f"  {kind}: " + ", ".join(f"{k} {v / total:.0%}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
