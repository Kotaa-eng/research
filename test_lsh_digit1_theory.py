"""小さい格子で理論式・角スペクトル・CGHを確認するテスト。

ユーザーが実行する場合: python -m unittest test_lsh_digit1_theory
2048格子の研究計算や画像生成は行わない。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lsh_digit1_theory import (
    AngularSpectrum, Digit1Config, encode_and_replay, fourier_coefficients,
    mode_design, run_digit1,
)


class Digit1TheoryTests(unittest.TestCase):
    def test_coefficients_use_amplitude_and_correct_fourier_sign(self):
        length, count = 0.5, 128
        z = (np.arange(count) + .5) * length / count
        amplitude = np.column_stack((np.ones(count),
                                     1 + .4 * np.cos(2 * np.pi * z / length)))
        coefficients = fourier_coefficients(amplitude, z, np.arange(-2, 3), length)
        np.testing.assert_allclose(coefficients[:, 0], [0, 0, 1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(coefficients[:, 1], [0, .2, 1, .2, 0], atol=1e-14)

    def test_padded_zero_distance_and_row_equal_sensor_frame(self):
        rng = np.random.default_rng(42)
        field = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
        plan = AngularSpectrum(32, 10e-6, 532e-9, 2)
        spectrum = plan.spectrum(field)
        transfer, _ = plan.transfer(0)
        np.testing.assert_allclose(plan.frame(spectrum, transfer), field, rtol=2e-6, atol=2e-6)
        for z in (0, .002):
            transfer, _ = plan.transfer(z)
            np.testing.assert_allclose(plan.row(spectrum, transfer),
                                       plan.frame(spectrum, transfer)[16],
                                       rtol=3e-6, atol=2e-6)

    def test_plane_wave_follows_helmholtz_dispersion(self):
        n, pitch, wavelength, z = 32, 10e-6, 532e-9, .01
        axis = np.arange(n) - n // 2
        u, v = 2, -1
        field = np.exp(2j * np.pi * (v * axis[:, None] + u * axis[None, :]) / n)
        plan = AngularSpectrum(n, pitch, wavelength, 1, bandlimit=False)
        spectrum = plan.spectrum(field)
        h, _ = plan.transfer(z)
        k0 = 2 * np.pi / wavelength
        transverse2 = (2 * np.pi / (n * pitch))**2 * (u**2 + v**2)
        delta = -transverse2 / (np.sqrt(k0**2 - transverse2) + k0)
        expected = field[n // 2] * np.exp(1j * delta * z)
        np.testing.assert_allclose(plan.row(spectrum, h), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(np.abs(plan.frame(spectrum, h))**2, 1,
                                   rtol=3e-6, atol=3e-6)

    def test_constant_complex_field_first_order_cgh_replay(self):
        config = Digit1Config(grid_size=64, pixel_pitch_m=10e-6,
                              aperture_radius_m=.25e-3,
                              evaluation_x_half_width_m=.2e-3, digit_width_m=.12e-3,
                              thread_count=9, thread_spacing_m=25e-6)
        field = np.full((64, 64), .6 * np.exp(.7j))
        codes, replay, report = encode_and_replay(field, np.ones((64, 64)), config, 20000.)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertLess(report["initial_plane_complex_relative_l2"], .025)
        self.assertLess(np.linalg.norm(replay - field) / np.linalg.norm(field), .025)
        self.assertGreater(report["first_order_power_fraction_of_incident"], 0)

    def test_small_theory_pipeline_saves_sensor_data_without_compression(self):
        config = Digit1Config(grid_size=64, pixel_pitch_m=10e-6,
                              aperture_radius_m=.25e-3, evaluation_x_half_width_m=.2e-3,
                              digit_width_m=.12e-3, thread_count=9, thread_spacing_m=25e-6,
                              coefficient_samples=64, z_samples=5, padding_factor=1,
                              include_cgh_replay=False)
        self.assertEqual(mode_design(config).count, 15)
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "theory"
            result = run_digit1(out, config, save_images=False)
            self.assertEqual(result["study"], "theory_only_digit1_single_xz_sheet")
            self.assertGreater(result["active_thread_count"], 0)
            self.assertLess(result["metrics"]["analytic_vs_initial_y0_complex_relative_l2"], 1e-12)
            self.assertEqual(result["padding_validation"]["status"], "not_checked")
            with np.load(out / "digit1_sheet_data.npz") as data:
                self.assertEqual(data["finite_complex_intensity"].shape, (5, 64))
                self.assertNotIn("cgh_replay_intensity", data.files)
            with np.load(out / "digit1_design.npz") as data:
                self.assertEqual(data["coefficients"].shape, (15, 9))
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("sensor_frames.npz", manifest["files"])
            self.assertFalse((out / "compressed_library.npz").exists())
            with self.assertRaises(FileExistsError):
                run_digit1(out, config, save_images=False)


if __name__ == "__main__":
    unittest.main()
