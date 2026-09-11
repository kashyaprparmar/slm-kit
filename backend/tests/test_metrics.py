from __future__ import annotations

import unittest

from app.eval.metrics import exact_match, token_f1


class MetricTests(unittest.TestCase):
    def test_exact_match_normalizes_case_spacing_and_punctuation(self):
        self.assertEqual(exact_match(["  Gross Margin! "], ["gross   margin"]), 1.0)

    def test_token_f1_counts_duplicate_tokens_once_each(self):
        self.assertAlmostEqual(token_f1(["a a b"], ["a b b"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
