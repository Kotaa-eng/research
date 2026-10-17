"""2軸の符号・左右の乗算・複素加算・評価のスケールを確認する小規模テスト。"""
import unittest

import numpy as np
from scipy.special import j0

import lsh_multisheet as multisheet
import lsh_digit1_proposed as single_sheet
from stage_a_validation import _spatial_factors


class MultiSheetTests(unittest.TestCase):
    def library(self, n=16, rank=5, modes=3):
        rng = np.random.default_rng(103)
        uy, _ = np.linalg.qr(rng.normal(size=(n, rank)))
        ux, _ = np.linalg.qr(rng.normal(size=(n, rank)))
        vy, vx = _spatial_factors(uy, ux)
        cores = rng.normal(size=(modes, rank, rank))+1j*rng.normal(size=(modes, rank, rank))
        return {"uy": uy, "ux": ux, "vy": vy, "vx": vx,
                "cores": np.ascontiguousarray(cores)}

    def test_two_axis_core_shifts_equal_explicit_fourier_projection(self):
        library = self.library()
        coefficients = np.array([[1+.2j, -.5j], [.3j, .8], [-.2, .7j]])
        pitch = .1
        xs, ys = np.array([.13, -.27]), np.array([-.17, .04, .22])
        gains = np.exp(1j*np.array([0, .7, -1.2]))
        actual, _ = multisheet.proposed_multisheet_field(library, coefficients, xs, ys, gains, pitch)
        k = 2*np.pi*np.fft.fftfreq(len(library["ux"]), d=pitch)
        exact = np.zeros((len(k), len(k)), dtype=complex)
        for y, gain in zip(ys, gains):
            for q, x in enumerate(xs):
                core = np.sum(coefficients[:, q, None, None]*library["cores"], axis=0)
                spectrum = library["uy"] @ core @ library["ux"].T
                exact += gain*np.exp(-1j*k*y)[:, None]*spectrum*np.exp(-1j*k*x)[None, :]
        py, px = library["uy"] @ library["uy"].T, library["ux"] @ library["ux"].T
        expected = np.fft.ifft2(py @ exact @ px, norm="ortho")
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_full_rank_is_exact_periodic_shift_in_both_axes(self):
        library = self.library(rank=16)
        coefficients = np.array([[1+.2j], [.3j], [-.2]])
        pitch = .1
        xs, ys, gains = np.array([.13]), np.array([-.17, .22]), np.array([1, .4+.6j])
        actual, _ = multisheet.proposed_multisheet_field(library, coefficients, xs, ys, gains, pitch)
        k = 2*np.pi*np.fft.fftfreq(16, d=pitch)
        core = np.sum(coefficients[:, 0, None, None]*library["cores"], axis=0)
        spectrum = library["uy"] @ core @ library["ux"].T
        full = sum(gain*np.exp(-1j*k*y)[:, None]*spectrum*np.exp(-1j*k*xs[0])[None, :]
                   for y, gain in zip(ys, gains))
        np.testing.assert_allclose(actual, np.fft.ifft2(full, norm="ortho"), atol=3e-12, rtol=3e-12)

    def test_one_sheet_y_zero_matches_existing_online_method(self):
        library = self.library()
        coefficients = np.array([[1+.2j, -.5j], [.3j, .8], [-.2, .7j]])
        xs = np.array([.13, -.27])
        actual, _ = multisheet.proposed_multisheet_field(
            library, coefficients, xs, np.array([0.0]), np.array([1.0]), .1)
        expected, _, _ = single_sheet.proposed_field(library, coefficients, xs, .1)
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_direct_uses_both_displaced_coordinates_and_complex_gains(self):
        axis = (np.arange(16)-8)*.1
        xs, ys = np.array([.13]), np.array([-.17, .22])
        kr, coefficients, gains = np.array([3.0]), np.array([[1+.2j]]), np.array([1, .4+.6j])
        actual = multisheet.direct_multisheet_field(axis, xs, ys, kr, coefficients, gains)
        expected = sum(gain*coefficients[0, 0]*j0(3*np.hypot(axis[:, None]-y, axis[None, :]-.13))
                       for y, gain in zip(ys, gains))
        np.testing.assert_allclose(actual, expected)

    def test_volume_metrics_do_not_fit_away_power_loss(self):
        reference = np.ones((3, 5, 7), dtype=complex)*(1+.2j)
        result = multisheet._voxel_scores(reference, reference*.5)
        self.assertAlmostEqual(result["complex_relative_l2"], .5)
        self.assertAlmostEqual(result["intensity_relative_l2"], .75)
        self.assertAlmostEqual(result["intensity_sum_ratio"], .25)

    def test_sheet_specific_coefficients_match_full_fourier_projection(self):
        library = self.library()
        weights = np.array([[[1+.2j, -.5j], [-.7j, .2]],
                            [[.3j, .8], [.4, -.1j]],
                            [[-.2, .7j], [.6j, -.3]]])
        xs, ys, gains = np.array([.13, -.27]), np.array([-.17, .22]), np.ones(2)
        actual, _ = multisheet.proposed_multisheet_field(library, weights, xs, ys, gains, .1)
        k = 2*np.pi*np.fft.fftfreq(16, d=.1)
        full = np.zeros((16, 16), dtype=complex)
        for s, y in enumerate(ys):
            for q, x in enumerate(xs):
                core = np.sum(weights[:, s, q, None, None]*library["cores"], axis=0)
                spectrum = library["uy"] @ core @ library["ux"].T
                full += np.exp(-1j*k*y)[:, None]*spectrum*np.exp(-1j*k*x)[None, :]
        py, px = library["uy"] @ library["uy"].T, library["ux"] @ library["ux"].T
        np.testing.assert_allclose(actual, np.fft.ifft2(py @ full @ px, norm="ortho"), atol=3e-12)

    def test_explicit_repeated_weights_agree_with_shared_sheet_shortcut(self):
        library = self.library()
        weights = np.array([[1+.2j, -.5j], [.3j, .8], [-.2, .7j]])
        xs, ys, gains = np.array([.13, -.27]), np.array([-.17, .22]), np.array([1, .4+.6j])
        common, _ = multisheet.proposed_multisheet_field(library, weights, xs, ys, gains, .1)
        explicit, _ = multisheet.proposed_multisheet_field(
            library, np.repeat(weights[:, None, :], 2, axis=1), xs, ys, gains, .1)
        np.testing.assert_allclose(explicit, common, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
