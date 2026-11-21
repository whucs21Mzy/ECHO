import random
import subprocess
import sys
import unittest

try:
    from echo.automaton import Automaton
except ImportError:
    Automaton = None


@unittest.skipUnless(
    Automaton is not None, "build echo/automaton to test C++ recycling"
)
class AutomatonRecyclingTest(unittest.TestCase):
    def test_capacity_error_leaves_a_valid_trie(self):
        program = """
from echo.automaton import Automaton
a = Automaton(4)
try:
    a.insert([2, 3, 4, 5, 6, 7])
except RuntimeError:
    pass
else:
    raise AssertionError('Expected a capacity error')
a.build()
assert a.retrieve(2, 4).tree_candidates[0] == 2
"""
        subprocess.run([sys.executable, "-c", program], check=True, timeout=5)

    def test_long_lived_trie_survives_capacity_reuse(self):
        rng = random.Random(1234)
        automaton = Automaton(128)
        automaton.init_logits(64, 8)
        for _ in range(1000):
            pattern = [rng.randrange(2, 64) for _ in range(10)]
            automaton.insert(pattern)
            automaton.build()
            automaton.trans_tokens(pattern[-4:])
            automaton.update([pattern[-1]], [[rng.randrange(2, 64) for _ in range(8)]])
            result = automaton.retrieve(pattern[-1], 32)
            self.assertEqual(result.tree_candidates[0], pattern[-1])
            self.assertLessEqual(len(result.tree_candidates), 32)


if __name__ == "__main__":
    unittest.main()
