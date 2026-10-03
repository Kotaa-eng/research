"""小さい格子で理論式・角スペクトル・CGHを確認するテスト。

ユーザーが実行する場合: python -m unittest test_lsh_digit1_theory
2048格子の研究計算や画像生成は行わない。
"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lsh_digit1_theory import (
    AngularSpectrum, Digit1Config, encode_and_replay, fourier_coefficients,
    mode_design, run_digit1,
    config_payload, digit_target, sheet_display,
)


class Digit1TheoryTests(unittest.TestCase):
    def test_zx_changes_target_and_preserves_internal_array_order(self):
        config = Digit1Config()
        q = (np.arange(23)+.5)/23
        x = (q-.5)*config.digit_width_m
        z = config.digit_z_start_m+q*(config.digit_z_end_m-config.digit_z_start_m)
        xz = digit_target(x, z, config)
        zx = digit_target(x, z, replace(config, digit_axes="zx"))
        np.testing.assert_array_equal(zx, xz.T)
        self.assertFalse(np.array_equal(xz, zx))
        values, extent, xlabel, ylabel = sheet_display(zx, x, z, np.ones(len(x), dtype=bool), "zx")
        np.testing.assert_array_equal(values, zx.T)
        self.assertEqual((xlabel, ylabel), ("z [mm]", "x [mm]"))
        self.assertAlmostEqual(extent[0], z[0]*1000)

    def test_old_config_without_orientation_remains_compatible(self):
        current = config_payload(Digit1Config())
        old = dict(current)
        old.pop("digit_axes")
        old.pop("target_digits")
        old.pop("digit_gap_fraction")
        old.pop("coefficient_design")
        old.pop("coefficient_fit_x_samples")
        old.pop("coefficient_ridge_relative")
        self.assertEqual(config_payload(old), current)

    def test_123_is_three_separated_digits_on_one_zx_sheet(self):
        config = Digit1Config(target_digits="123", digit_axes="zx")
        # 上部の横断線が、独立した3文字と背景の隙間を通ることを確認。
        u = (np.arange(1201) + .5) / 1201
        x = np.array([.4 * config.digit_width_m])
        z = config.digit_z_start_m + u * (config.digit_z_end_m-config.digit_z_start_m)
        target = digit_target(x, z, config)
        self.assertEqual(target.shape, (1201, 1))
        transitions = np.diff(np.r_[False, target[:, 0].astype(bool), False].astype(int))
        self.assertEqual(np.count_nonzero(transitions == 1), 3)
        self.assertEqual(np.count_nonzero(transitions == -1), 3)
        square_x = (u-.5)*config.digit_width_m
        xz = digit_target(square_x, z, replace(config, digit_axes="xz"))
        zx = digit_target(square_x, z, config)
        np.testing.assert_array_equal(zx, xz.T)

    def test_two_and_three_have_different_lower_stems(self):
        config = Digit1Config()
        x = (np.array([.2, .8])-.5)*config.digit_width_m
        z = np.array([config.digit_z_start_m + .3*(config.digit_z_end_m-config.digit_z_start_m)])
        np.testing.assert_array_equal(digit_target(x, z, replace(config, target_digits="2")), [[1, 0]])
        np.testing.assert_array_equal(digit_target(x, z, replace(config, target_digits="3")), [[0, 1]])

    def test_longer_design_adds_modes_with_consistent_dispersion(self):
        config = Digit1Config(target_digits="123", digit_axes="zx")
        short = mode_design(config)
        long = mode_design(replace(config, design_length_m=1.0,
                                  digit_z_start_m=.15, digit_z_end_m=.85))
        self.assertGreater(long.count, short.count)
        np.testing.assert_allclose(long.kz_per_m**2 + long.kr_per_m**2,
                                   long.k0_per_m**2, rtol=1e-14)
        self.assertAlmostEqual(long.kr_center_per_m, short.kr_center_per_m)

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
