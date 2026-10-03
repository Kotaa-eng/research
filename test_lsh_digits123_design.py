"""設計式・両手法の係数一致を確認するテスト。研究計算は実行しない。

ユーザーが実行する場合: python -m unittest test_lsh_digits123_design
"""
import unittest
from dataclasses import replace

import numpy as np
from scipy.special import j0

import lsh_digit1_theory as theory
import lsh_digit1_proposed as proposed
from lsh_digits123_design import design_candidates, design_report


class Digits123DesignTests(unittest.TestCase):
    def test_wider_band_and_denser_threads_preserve_target_extent(self):
        cases = design_candidates()
        base, wide = cases["baseline"], cases["wide_dense_coupled"]
        report = design_report(wide)
        self.assertGreater(theory.mode_design(wide).count, theory.mode_design(base).count)
        self.assertGreater(report["kz_bandwidth_per_m"], design_report(base)["kz_bandwidth_per_m"])
        self.assertEqual(wide.thread_spacing_m, base.thread_spacing_m/2)
        self.assertAlmostEqual((wide.thread_count-1)*wide.thread_spacing_m,
                               (base.thread_count-1)*base.thread_spacing_m)
        self.assertGreaterEqual(report["geometric_all_mode_overlap_estimate_mm"], 500)
        self.assertGreaterEqual(wide.grid_size*wide.pixel_pitch_m/2,
                                wide.aperture_radius_m+(wide.thread_count-1)*wide.thread_spacing_m/2)

    def test_regularized_solver_recovers_a_known_coherent_superposition(self):
        x = np.linspace(-.5e-3, .5e-3, 101)
        positions = np.array([-.13e-3, .17e-3])
        kr = np.array([30000., 60000.])
        expected = np.array([[.5+.3j, -.2+.1j], [.1-.7j, .8+.2j]])
        target_modes = np.stack([j0(k*np.abs(x[:, None]-positions)) @ row
                                 for k, row in zip(kr, expected)])
        actual = theory.fit_transverse_coefficients(x, positions, kr, target_modes, 1e-12)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_coupled_coefficients_minimize_coherent_regularized_objective(self):
        config = theory.Digit1Config(target_digits="123", digit_axes="zx",
                                    coefficient_samples=128, coefficient_fit_x_samples=51)
        design = theory.mode_design(config)
        positions = np.array([-.3e-3, 0, .3e-3])
        old = theory.target_coefficients(config, positions, design)
        coupled = replace(config, coefficient_design="coupled_least_squares")
        fitted = theory.target_coefficients(coupled, positions, design)
        x = np.linspace(-config.evaluation_x_half_width_m, config.evaluation_x_half_width_m, 51)
        z = (np.arange(128)+.5)*config.design_length_m/128
        target_modes = theory.fourier_coefficients(np.sqrt(theory.digit_target(x, z, config)),
                                                  z, design.m, config.design_length_m)
        for index, kr in enumerate(design.kr_per_m):
            basis = j0(kr*np.abs(x[:, None]-positions))
            penalty = config.coefficient_ridge_relative*np.linalg.norm(basis, 2)**2
            def objective(coeff):
                return np.linalg.norm(basis @ coeff-target_modes[index])**2 + penalty*np.linalg.norm(coeff)**2
            self.assertLessEqual(objective(fitted[index]), objective(old[index])+1e-11)
        # 圧縮側も理論側と同じ設計を使い、係数だけ別に最適化しない。
        _, shared = proposed.design_coefficients(coupled, positions)
        np.testing.assert_array_equal(shared, fitted)

    def test_new_target_fit_reuses_library_but_rejects_changed_kr_or_placement(self):
        cases = design_candidates()
        independent = cases["wider_kr_finer_threads"]
        coupled = cases["wide_dense_coupled"]
        self.assertEqual(proposed.library_spec(independent, 100, 42),
                         proposed.library_spec(coupled, 100, 42))
        self.assertNotEqual(proposed.library_spec(cases["baseline"], 100, 42),
                            proposed.library_spec(coupled, 100, 42))


if __name__ == "__main__":
    unittest.main()
