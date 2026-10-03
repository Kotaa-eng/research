"""小さい格子で提案法の数学的一致と辞書の取り違え防止を検証する。

未実行。ユーザー側で python -m unittest test_lsh_digit1_proposed を実行可能。
2048画素の研究計算はしない。
"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lsh_digit1_proposed import (
    design_coefficients, library_spec, load_or_build_library,
    mode_design_for_spacing, proposed_field, truncate_library,
    _shift_aware_x_basis,
)
from lsh_digit1_theory import Digit1Config
from stage_a_parameters import StageAConfig
from stage_a_validation import _spatial_factors


class Digit1ProposedTests(unittest.TestCase):
    def test_zx_reuses_dictionary_but_recomputes_thread_coefficients(self):
        config = Digit1Config()
        zx = replace(config, digit_axes="zx")
        self.assertEqual(library_spec(config, 100, 42), library_spec(zx, 100, 42))
        positions = np.array([-.225e-3, 0, .225e-3])
        _, old = design_coefficients(config, positions)
        _, new = design_coefficients(zx, positions)
        self.assertGreater(np.linalg.norm(new-old)/np.linalg.norm(old), .1)

    def test_shift_aware_basis_keeps_odd_translation_component(self):
        n, pitch = 32, 4e-6
        basis = np.zeros((n, 3))
        basis[0, 0] = 1
        basis[[1, -1], 1] = 1/np.sqrt(2)
        basis[[2, -2], 2] = 1/np.sqrt(2)
        cores = np.zeros((1, 3, 3), dtype=complex)
        cores[0, 0, 0], cores[0, 1, 1] = 2, 1
        shift = n*pitch/4
        kx = 2*np.pi*np.fft.fftfreq(n, d=pitch)
        translated = np.exp(-1j*kx*shift)*basis[:, 1]
        # 中心の偶基底では四分周期移動後の奇成分が消える。
        old_error = np.linalg.norm(translated-basis @ (basis.T @ translated))
        new_basis = _shift_aware_x_basis(basis, cores, np.array([-shift, 0, shift]),
            pitch, StageAConfig(grid_size=n, rank=3, selected_mode_count=1))
        new_error = np.linalg.norm(translated-new_basis @ (new_basis.T @ translated))
        self.assertGreater(old_error, .99)
        self.assertLess(new_error, 2e-13)

    def test_finer_modes_preserve_kr_endpoints_and_propagation_length(self):
        config = Digit1Config()
        base = mode_design_for_spacing(config)
        for factor in (2, 4):
            fine = mode_design_for_spacing(config, factor)
            self.assertEqual(fine.count, (base.count-1)*factor+1)
            np.testing.assert_allclose(fine.kr_per_m[[0, -1]], base.kr_per_m[[0, -1]], rtol=1e-12)
            np.testing.assert_allclose(np.diff(fine.kz_per_m),
                                       2*np.pi/(config.design_length_m*factor), atol=3e-9)
            self.assertEqual(config.design_length_m, .5)

    def test_finer_coefficient_period_zero_extends_target(self):
        config = Digit1Config()
        base, a = design_coefficients(config, np.array([0.0]), 1)
        fine, b = design_coefficients(config, np.array([0.0]), 2)
        # DCは積分値/周期。目標の積分値を維持し、周期を2倍にする。
        np.testing.assert_allclose(b[fine.m == 0], a[base.m == 0]/2, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            mode_design_for_spacing(config, 1.5)

    @staticmethod
    def library(n, rank):
        rng = np.random.default_rng(57)
        uy = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank]
        ux = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank]
        cores = rng.normal(size=(3, rank, rank)) + 1j*rng.normal(size=(3, rank, rank))
        vy, vx = _spatial_factors(uy, ux)
        return {"uy": uy, "ux": ux, "vy": vy, "vx": vx, "cores": cores}

    def test_full_rank_matches_integer_pixel_shift_and_thread_weights(self):
        n, pitch = 16, 4e-6
        library = self.library(n, n)
        weights = np.array([[1, .2j], [.4j, -.3], [.8, .6+.1j]])
        expected = np.zeros((n, n), dtype=complex)
        for q, pixels in enumerate((0, 3)):
            for m in range(3):
                beam = np.fft.ifft2(library["uy"] @ library["cores"][m] @ library["ux"].T,
                                    norm="ortho")
                expected += weights[m, q]*np.roll(beam, pixels, axis=1)
        # オンラインに全画素IFFTが入っていないことも確認する。
        with patch("numpy.fft.ifft2", side_effect=AssertionError("online full IFFT")):
            actual, timings, _ = proposed_field(library, weights, np.array([0, 3*pitch]), pitch)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
        self.assertGreaterEqual(timings["field_generation_seconds"], 0)

    def test_reduced_rank_matches_projection_of_full_shifted_spectrum(self):
        n, pitch = 16, 4e-6
        parent = self.library(n, 8)
        library = truncate_library(parent, 4)
        # 実際の親基底からの切り出しでも、配列配置以外の値を変えない。
        self.assertFalse(parent["cores"][:, :4, :4].flags.c_contiguous)
        for key in ("uy", "ux", "vy", "vx", "cores"):
            self.assertTrue(library[key].flags.c_contiguous)
            expected_slice = parent[key][:, :4, :4] if key == "cores" else parent[key][:, :4]
            np.testing.assert_array_equal(library[key], expected_slice)
            self.assertEqual(library[key].dtype, parent[key].dtype)
        weights = np.array([[1, .2j], [.4j, -.3], [.8, .6+.1j]])
        positions = np.array([-1.3, 2.7])*pitch
        kx = 2*np.pi*np.fft.fftfreq(n, d=pitch)
        shifted = np.zeros((n, n), dtype=complex)
        for q, pos in enumerate(positions):
            for m in range(3):
                shifted += weights[m, q]*(library["uy"] @ library["cores"][m]
                           @ library["ux"].T)*np.exp(-1j*kx*pos)[None, :]
        uy, ux = library["uy"], library["ux"]
        projected = uy @ (uy.T @ shifted @ ux) @ ux.T
        expected = np.fft.ifft2(projected, norm="ortho")
        actual, _, _ = proposed_field(library, weights, positions, pitch)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
        # 射影は共有部分空間内の別の候補よりL2残差が小さい。
        rng = np.random.default_rng(32)
        alternative = projected + uy @ rng.normal(size=(4, 4)) @ ux.T
        self.assertLessEqual(np.linalg.norm(shifted-projected), np.linalg.norm(shifted-alternative))

    def test_mismatched_library_is_rejected_without_overwriting(self):
        config = Digit1Config(grid_size=64, pixel_pitch_m=10e-6, aperture_radius_m=.25e-3,
                              evaluation_x_half_width_m=.2e-3, digit_width_m=.12e-3,
                              thread_count=3, thread_spacing_m=30e-6)
        spec = library_spec(config, 4, 42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"library.npz"
            np.savez(path, **{key: np.zeros(1) for key in ("uy", "ux", "vy", "vx", "cores", "m", "kr")},
                     spec_json=json.dumps(spec))
            old_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Library settings differ"):
                load_or_build_library(path, replace(config, spot_radius_m=60e-6), rank=4)
            self.assertEqual(path.read_bytes(), old_bytes)


if __name__ == "__main__":
    unittest.main()
