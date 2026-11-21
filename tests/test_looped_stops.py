import unittest

import torch

from echo.inference.looped_stop import GenerationStops


class PieceTokenizer:
    eos_token_id = None
    pieces = {
        0: "x",
        1: "\n`",
        2: "``",
        3: "Q",
        4: ":",
        5: "after",
        6: " ",
        7: "<eos>",
        9: "<end>",
    }

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[x] for x in ids)


class GenerationStopsTest(unittest.TestCase):
    def test_stop_string_across_token_and_acceptance_block_boundaries(self):
        stops = GenerationStops(PieceTokenizer(), strings=("\n```",))
        self.assertIsNone(stops.observe(torch.tensor([0, 1])))
        result = stops.observe(torch.tensor([0, 1, 2, 5, 5]))
        self.assertEqual(result.generated_count, 3)
        self.assertEqual(result.reason, dict(type="stop", matched="\n```"))

    def test_earliest_stop_inside_a_long_verified_block(self):
        stops = GenerationStops(PieceTokenizer(), strings=("Q:",))
        result = stops.observe(torch.tensor([0, 3, 4] + [5] * 100))
        self.assertEqual(result.generated_count, 3)

    def test_all_frozen_eos_ids_are_used(self):
        stops = GenerationStops(PieceTokenizer(), eos_token_ids=(7, 9), strings=("Q:",))
        result = stops.observe(torch.tensor([0, 9, 3, 4]))
        self.assertEqual(result.generated_count, 2)
        self.assertEqual(result.reason["matched"], 9)

    def test_token_sequence_retains_separate_raw_and_visible_counts(self):
        stops = GenerationStops(PieceTokenizer(), token_sequences=((3, 4),))
        result = stops.observe(torch.tensor([0, 3, 4]))
        self.assertEqual((result.generated_count, result.visible_count), (3, 1))


if __name__ == "__main__":
    unittest.main()
