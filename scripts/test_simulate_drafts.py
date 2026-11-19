"""Tests of simulate_drafts.py: the copy draft, the union of experts and the cost of the rounds.

    python3 -m unittest discover -s scripts -p 'test_*.py'

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import unittest

import numpy as np

import simulate_drafts as sim


def same_experts(positions):
    """Experts 0 to 7 in every layer at every position."""
    return np.tile(np.arange(sim.USED, dtype=np.int32), (positions, sim.LAYERS, 1))


class Draft(unittest.TestCase):
    def test_it_copies_after_the_latest_earlier_match(self):
        # [1, 2, 3] comes twice before the end. The tokens after the later one are 5, 7.
        tokens = [1, 2, 3, 4, 9, 1, 2, 3, 5, 7, 1, 2, 3]
        self.assertEqual(sim.CopyDraft(tokens).draft(13, 2), ([5, 7], 3))

    def test_a_draft_can_copy_its_own_tokens(self):
        # The match [5, 6] ends 2 tokens before the end, so the draft repeats 5, 6.
        self.assertEqual(sim.CopyDraft([5, 6, 5, 6]).draft(4, 4), ([5, 6, 5, 6], 2))

    def test_no_match_gives_no_draft(self):
        self.assertEqual(sim.CopyDraft([1, 2, 3, 4]).draft(4, 3), ([], 0))


class Union(unittest.TestCase):
    def test_it_counts_the_different_experts_of_each_layer(self):
        experts = same_experts(3)
        experts[1] += 4  # experts 4 to 11, so positions 0 and 1 use 12 experts in each layer
        experts[2] += 20  # experts 20 to 27, so positions 1 and 2 use 16
        self.assertEqual(sim.union(experts, 0, 2), 12 * sim.LAYERS)
        self.assertEqual(sim.union(experts, 1, 2), 16 * sim.LAYERS)


class Replay(unittest.TestCase):
    # 2 prompt tokens, then 11 output tokens. The first output token comes from the prompt's pass,
    # so the rounds make the other 10.
    TOKENS = [0, 0] + list(range(100, 111))
    EXPERTS = same_experts(11)

    def test_a_round_of_4_correct_draft_tokens_makes_5_tokens(self):
        made, ms, rounds = sim.replay(2, self.TOKENS, self.EXPERTS, ("perfect", 4), extra=0.25)
        self.assertEqual((made, rounds), (10, [5, 5]))
        # A round costs a plain token (22.0 ms), plus 0.25 ms for each of its 4 draft tokens.
        self.assertAlmostEqual(ms["short"], 2 * (22.0 + 4 * 0.25))

    def test_with_no_match_each_round_is_a_plain_token(self):
        made, ms, rounds = sim.replay(2, self.TOKENS, self.EXPERTS, (4, 1))
        self.assertEqual((made, rounds), (10, [1] * 10))
        self.assertAlmostEqual(ms["short"], 10 * 22.0)
        self.assertAlmostEqual(ms["8k"], 10 * 32.0)


if __name__ == "__main__":
    unittest.main()
