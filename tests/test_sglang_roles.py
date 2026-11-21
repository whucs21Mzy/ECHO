"""KV ownership invariants of the shared SGLang recurrent execution path."""

import unittest
from types import SimpleNamespace

from echo.sglang.backend import OuroSGLangBackend, RavenSGLangBackend


def covered(ranges):
    slots = [slot for offset, count in ranges for slot in range(offset, offset + count)]
    if len(slots) != len(set(slots)):
        raise AssertionError("A forward would write the same logical KV slot twice")
    return set(slots)


class RecurrentRolesTest(unittest.TestCase):
    def test_ouro_split_covers_each_layer_once(self):
        model = SimpleNamespace(config=None, total_steps=4, num_layers=2)
        backend = OuroSGLangBackend(SimpleNamespace(model=model), 1, 16, 8)
        early = covered(
            backend._cache_ranges(0, 1, from_tokens=True, fill_shallow=True)
        )
        deep = covered(
            backend._cache_ranges(1, 4, from_tokens=False, fill_shallow=True)
        )
        self.assertFalse(early & deep)
        self.assertEqual(early, set(range(backend.shallow_slot_end)))
        self.assertEqual(early | deep, set(range(backend.num_cache_slots)))

    def test_raven_ar_skips_q1_coda_and_split_covers_all_roles_once(self):
        class Model:
            config = None
            total_steps = 32
            num_layers = 2
            pre_count = post_count = 1

            def configure_split(self, steps, context, seed):
                self.steps = steps
                self.shallow_slot_end = (
                    self.pre_count + steps * self.num_layers + self.post_count
                )
                self.end_layer = (
                    self.pre_count
                    + self.total_steps * self.num_layers
                    + 2 * self.post_count
                )

            def core_offset(self, step):
                return (
                    self.pre_count
                    + step * self.num_layers
                    + (self.post_count if step >= self.steps else 0)
                )

        for k in (2, 4, 8):
            with self.subTest(shallow_steps=k):
                model = Model()
                backend = RavenSGLangBackend(SimpleNamespace(model=model), k, 64, 8)
                early = covered(
                    backend._cache_ranges(0, k, from_tokens=True, fill_shallow=True)
                )
                deep = covered(
                    backend._cache_ranges(k, 32, from_tokens=False, fill_shallow=True)
                )
                echo = covered(
                    backend._cache_ranges(0, 32, from_tokens=True, fill_shallow=True)
                )
                ar = covered(
                    backend._cache_ranges(0, 32, from_tokens=True, fill_shallow=False)
                )
                q1 = set(
                    range(
                        backend.shallow_slot_end - model.post_count,
                        backend.shallow_slot_end,
                    )
                )
                self.assertFalse(early & deep)
                self.assertEqual(early | deep, echo)
                self.assertEqual(early, set(range(backend.shallow_slot_end)))
                self.assertEqual(echo, set(range(backend.num_cache_slots)))
                self.assertEqual(echo - ar, q1)


if __name__ == "__main__":
    unittest.main()
