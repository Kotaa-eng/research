"""事前の座標入り因子と、復元画像の線形移動の数学的等価性を確認。"""
import unittest

import numpy as np

from legacy_vs_proposed_rank80 import (
    legacy_field, prepare_placed_factors, proposed_cached_field, shift_x_linear,
)
import lsh_digit1_proposed as proposed
from stage_a_validation import _spatial_factors, _x_shift_core_operator


class FixedCoordinatesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        uy, _ = np.linalg.qr(rng.normal(size=(16, 5)))
        ux, _ = np.linalg.qr(rng.normal(size=(16, 5)))
        vy, vx = _spatial_factors(uy, ux)
        self.library = {"uy": uy, "ux": ux, "vy": vy, "vx": vx,
                        "cores": rng.normal(size=(3, 5, 5))+1j*rng.normal(size=(3, 5, 5))}
        self.weights = rng.normal(size=(3, 2))+1j*rng.normal(size=(3, 2))
        self.positions = np.array([.13, -.27])

    def test_placed_factors_equal_shift_after_reconstruction(self):
        lib = self.library
        placed = prepare_placed_factors(lib, self.positions, .1)
        actual, times = legacy_field(lib, self.weights, self.positions, .1, placed)
        expected = np.zeros_like(actual)
        for q, position in enumerate(self.positions):
            core = np.sum(self.weights[:, q, None, None]*lib["cores"], axis=0)
            centered = lib["vy"] @ core @ lib["vx"].T
            expected += shift_x_linear(centered, position/.1)
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        self.assertEqual(times["shift_seconds"], 0.0)

    def test_prepared_proposed_operators_equal_online_operators(self):
        lib = self.library
        k = 2*np.pi*np.fft.fftfreq(16, d=.1)
        operators = np.stack([_x_shift_core_operator(lib["ux"], np.exp(-1j*k*x)) for x in self.positions])
        actual, times = proposed_cached_field(lib, self.weights, operators)
        expected, _, _ = proposed.proposed_field(lib, self.weights, self.positions, .1)
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        self.assertEqual(times["shift_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
