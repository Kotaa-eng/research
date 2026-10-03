"""Stage A の数式的な整合性を小さい格子で確認するテスト。

2048×2048 の研究計算や画像出力は行わない。
実行する場合: python -m unittest test_stage_a_validation
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import j0

from stage_a_parameters import StageAConfig, design_modes
from stage_a_validation import (
    _cgh_metrics,
    _field_metrics,
    _image_metrics,
    _json_compatible,
    _project_core,
    _randomized_shared_basis,
    _reconstruct_fft,
    _reconstruct_spatial,
    _spatial_factors,
    _relative_l2,
    _x_shift_core_operator,
    METRIC_SCHEMA_VERSION,
    run_stage_a,
)


class StageAMathematicsTests(unittest.TestCase):
    def test_spatial_factors_equal_full_ifft_for_coaxial_and_shifted_cores(self):
        # Distinct, non-square factors expose wrong axes, conjugation or FFT scaling.
        rng = np.random.default_rng(914)
        uy, _ = np.linalg.qr(rng.normal(size=(12, 3)))
        ux, _ = np.linalg.qr(rng.normal(size=(18, 5)))
        cores = rng.normal(size=(4, 3, 5)) + 1j * rng.normal(size=(4, 3, 5))
        weights = rng.normal(size=4) + 1j * rng.normal(size=4)
        summed = np.tensordot(weights, cores, axes=(0, 0))
        shifted = np.zeros_like(summed)
        for displacement in (-1.25, 0.4, 2.7):
            phase = np.exp(-2j * np.pi * np.fft.fftfreq(18) * displacement)
            shifted += summed @ _x_shift_core_operator(ux, phase)
        vy, vx = _spatial_factors(uy, ux)
        for core in (summed, shifted):
            reference = np.fft.ifft2(_reconstruct_fft(core, uy, ux), norm="ortho")
            np.testing.assert_allclose(
                _reconstruct_spatial(core, vy, vx), reference, rtol=1e-12, atol=1e-12)

    def test_image_quality_metrics_identical_and_controlled_error(self):
        reference = np.linspace(0.0, 1.0, 16 * 16).reshape(16, 16)
        identical = _image_metrics(reference, reference.copy())
        self.assertAlmostEqual(identical["ssim"], 1.0, places=12)
        self.assertTrue(np.isposinf(identical["psnr_db"]))
        self.assertTrue(np.isposinf(identical["snr_db"]))
        self.assertEqual(_json_compatible(identical)["psnr_db"], "Infinity")

        offset = 0.1
        perturbed = _image_metrics(reference, reference + offset)
        self.assertAlmostEqual(perturbed["psnr_db"], 20.0, places=10)
        expected_snr = 10.0 * np.log10(
            np.sum(reference**2) / (reference.size * offset**2)
        )
        self.assertAlmostEqual(perturbed["snr_db"], expected_snr, places=10)
        self.assertLess(perturbed["ssim"], 1.0)

        larger_error = _image_metrics(reference, reference + 2 * offset)
        self.assertLess(larger_error["psnr_db"], perturbed["psnr_db"])
        self.assertLess(larger_error["snr_db"], perturbed["snr_db"])
        with self.assertRaises(TypeError):
            _image_metrics(reference.astype(complex), reference)

    def test_local_ssim_uses_sample_covariance_and_seven_pixel_window(self):
        # On a 7x7 image the cropped valid region contains exactly one window.
        # Its SSIM can be checked directly, independently of uniform_filter.
        reference = np.arange(49, dtype=np.float64).reshape(7, 7) / 48.0
        candidate = reference.copy()
        candidate[3, 3] += 0.2
        actual = _image_metrics(reference, candidate)["ssim"]

        mean_ref = float(np.mean(reference))
        mean_candidate = float(np.mean(candidate))
        variance_ref = float(np.var(reference, ddof=1))
        variance_candidate = float(np.var(candidate, ddof=1))
        covariance = float(np.sum(
            (reference - mean_ref) * (candidate - mean_candidate)
        ) / 48)
        c1, c2 = 0.01**2, 0.03**2
        expected = (
            (2 * mean_ref * mean_candidate + c1) * (2 * covariance + c2)
            / ((mean_ref**2 + mean_candidate**2 + c1)
               * (variance_ref + variance_candidate + c2))
        )
        self.assertAlmostEqual(actual, expected, places=8)
        self.assertLess(actual, 1.0)
        self.assertAlmostEqual(
            _image_metrics(reference, reference.copy())["ssim"], 1.0, places=12
        )

    def test_stage_a_import_does_not_require_scikit_image(self):
        # Run in a fresh interpreter, where a cached module cannot hide an
        # accidental import of skimage on systems without scikit-image.
        script = """
import builtins
original_import = builtins.__import__
def reject_skimage(name, *args, **kwargs):
    if name == 'skimage' or name.startswith('skimage.'):
        raise ModuleNotFoundError('skimage is intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_skimage
import stage_a_validation
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_field_quality_metrics_keep_complex_and_image_domains_distinct(self):
        yy, xx = np.mgrid[:16, :16]
        reference = (1.0 + xx / 16.0) * np.exp(1j * yy / 8.0)
        candidate = 1.25 * reference
        metrics = _field_metrics(reference, candidate, mask_fraction=0.01)
        expected_complex_snr = 10.0 * np.log10(1.0 / 0.25**2)
        self.assertAlmostEqual(metrics["complex_snr_db"], expected_complex_snr,
                               places=10)
        self.assertAlmostEqual(metrics["complex_relative_l2"], 0.25, places=10)
        for prefix in ("amplitude", "intensity"):
            self.assertLess(metrics[f"{prefix}_ssim"], 1.0)
            self.assertTrue(np.isfinite(metrics[f"{prefix}_psnr_db"]))
            self.assertTrue(np.isfinite(metrics[f"{prefix}_snr_db"]))

    def test_frozen_wave_modes_are_symmetric_and_propagating(self):
        design = design_modes(StageAConfig())
        m = np.asarray(design.m)
        kz = np.asarray(design.kz_per_m)
        kr = np.asarray(design.kr_per_m)
        k0 = 2 * np.pi / 532e-9
        kr_center = 2.4048255577 / 50e-6
        q = np.sqrt(k0**2 - kr_center**2)

        np.testing.assert_array_equal(m, np.arange(-7, 8))
        self.assertEqual(len(m), 15)
        np.testing.assert_allclose(kz, q + 2 * np.pi * m / 0.50, rtol=1e-12)
        np.testing.assert_allclose(kr**2 + kz**2, k0**2, rtol=1e-12)
        self.assertTrue(np.all((kz > 0) & (kz <= k0)))
        self.assertGreater(q + 2 * np.pi * 8 / 0.50, k0)
        self.assertAlmostEqual(kr[len(m) // 2] / kr_center, 1.0, places=9)

    def test_unitary_fft_preserves_complex_relative_error(self):
        rng = np.random.default_rng(240)
        reference = rng.normal(size=(19, 23)) + 1j * rng.normal(size=(19, 23))
        recovered = reference + 0.02 * (
            rng.normal(size=reference.shape) + 1j * rng.normal(size=reference.shape)
        )
        fft_ref = np.fft.fft2(reference, norm="ortho")
        fft_rec = np.fft.fft2(recovered, norm="ortho")
        self.assertAlmostEqual(
            _relative_l2(reference, recovered), _relative_l2(fft_ref, fft_rec), places=13
        )
        np.testing.assert_allclose(
            np.fft.ifft2(fft_ref, norm="ortho"), reference, rtol=1e-13, atol=1e-13
        )

    def test_one_real_basis_pair_supports_all_modes_and_complex_core_sum(self):
        rng = np.random.default_rng(241)
        grid_size, rank = 16, 4
        true_basis, _ = np.linalg.qr(rng.normal(size=(grid_size, rank)))
        raw_cores = rng.normal(size=(10, rank, rank)) + 1j * rng.normal(
            size=(10, rank, rank)
        )
        # J0 の FFT と同じく各modeが転置対称な、既知の共通部分空間を作る。
        true_cores = (raw_cores + raw_cores.transpose(0, 2, 1)) / 2
        spectra = np.stack([true_basis @ c @ true_basis.T for c in true_cores])
        np.testing.assert_allclose(spectra, spectra.transpose(0, 2, 1), atol=1e-13)

        config = StageAConfig(grid_size=grid_size, rank=rank,
                              randomized_oversampling=4)
        basis = _randomized_shared_basis(spectra, config)
        cores = np.stack([_project_core(spectrum, basis, basis)
                          for spectrum in spectra])
        np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-12)
        self.assertEqual(cores.shape, (10, rank, rank))
        self.assertTrue(np.isrealobj(basis))
        for original, core in zip(spectra, cores):
            self.assertLess(_relative_l2(original,
                                          _reconstruct_fft(core, basis, basis)), 1e-8)

        weights = rng.normal(size=10) + 1j * rng.normal(size=10)
        direct_fft = np.tensordot(weights, spectra, axes=(0, 0))
        summed_core = np.tensordot(weights, cores, axes=(0, 0))
        proposed_fft = _reconstruct_fft(summed_core, basis, basis)
        individually_reconstructed = np.tensordot(
            weights,
            np.stack([_reconstruct_fft(c, basis, basis) for c in cores]),
            axes=(0, 0),
        )
        np.testing.assert_allclose(proposed_fft, individually_reconstructed, atol=1e-12)
        self.assertLess(_relative_l2(direct_fft, proposed_fft), 1e-8)
        self.assertAlmostEqual(
            _relative_l2(direct_fft, proposed_fft),
            _relative_l2(
                np.fft.ifft2(direct_fft, norm="ortho"),
                np.fft.ifft2(proposed_fft, norm="ortho"),
            ),
            places=12,
        )

    def test_subpixel_core_shift_equals_projection_of_uncompressed_fft(self):
        rng = np.random.default_rng(242)
        grid_size, rank = 16, 4
        pitch = 3.74e-6
        basis, _ = np.linalg.qr(rng.normal(size=(grid_size, rank)))
        raw_core = rng.normal(size=(rank, rank)) + 1j * rng.normal(
            size=(rank, rank)
        )
        core = (raw_core + raw_core.T) / 2
        unshifted_fft = _reconstruct_fft(core, basis, basis)
        dx = 0.37 * pitch
        kx = 2 * np.pi * np.fft.fftfreq(grid_size, pitch)
        phase_x = np.exp(-1j * kx * dx)
        full_shifted_fft = unshifted_fft * phase_x[None, :]

        ax = _x_shift_core_operator(basis, phase_x)
        core_shifted_fft = _reconstruct_fft(core @ ax, basis, basis)
        explicit_projection = (basis @ basis.T) @ full_shifted_fft @ (basis @ basis.T)
        np.testing.assert_allclose(core_shifted_fft, explicit_projection, atol=1e-12)
        self.assertGreater(_relative_l2(full_shifted_fft, core_shifted_fft), 1e-3)
        with self.assertRaises(ValueError):
            _x_shift_core_operator(basis, phase_x[:-1])

        # 基底が全周波数空間を張れば、部分画素移動でも誤差は消える。
        identity = np.eye(grid_size)
        ax_full = _x_shift_core_operator(identity, phase_x)
        np.testing.assert_allclose(
            _reconstruct_fft(unshifted_fft @ ax_full, identity, identity),
            full_shifted_fft,
            atol=1e-12,
        )

    def test_ten_shifted_threads_are_one_projection_of_full_fft_sum(self):
        rng = np.random.default_rng(243)
        grid_size, rank = 14, 4
        pitch = 3.74e-6
        basis, _ = np.linalg.qr(rng.normal(size=(grid_size, rank)))
        cores = rng.normal(size=(10, rank, rank)) + 1j * rng.normal(
            size=(10, rank, rank)
        )
        weights = rng.normal(size=10) + 1j * rng.normal(size=10)
        x_positions = (np.arange(10) - 4.5) * 0.37 * pitch
        weighted_core = np.tensordot(weights, cores, axes=(0, 0))
        kx = 2 * np.pi * np.fft.fftfreq(grid_size, pitch)
        proposed_core = np.zeros((rank, rank), dtype=np.complex128)
        for dx in x_positions:
            proposed_core += weighted_core @ _x_shift_core_operator(
                basis, np.exp(-1j * kx * dx)
            )
        proposed_fft = _reconstruct_fft(proposed_core, basis, basis)
        base_fft = _reconstruct_fft(weighted_core, basis, basis)
        full_fft = sum(
            base_fft * np.exp(-1j * kx * dx)[None, :] for dx in x_positions
        )
        explicit_projection = (basis @ basis.T) @ full_fft @ (basis @ basis.T)
        np.testing.assert_allclose(proposed_fft, explicit_projection, atol=1e-11)

    def test_fourier_shift_wraps_at_numerical_window_boundary(self):
        # J0 は窓の端でも非零。FFT shift は周期境界で折り返す。
        ny, nx, pitch = 15, 17, 4e-6
        x = (np.arange(nx) - nx // 2) * pitch
        y = (np.arange(ny) - ny // 2) * pitch
        original = j0((2.4048255577 / 50e-6) * np.hypot(y[:, None], x[None, :]))
        self.assertGreater(np.max(np.abs(original[:, -1])), 0.1)
        spectrum = np.fft.fft2(original, norm="ortho")
        kx = 2 * np.pi * np.fft.fftfreq(nx, pitch)

        one_pixel = np.fft.ifft2(
            spectrum * np.exp(-1j * kx * pitch)[None, :], norm="ortho"
        )
        np.testing.assert_allclose(one_pixel, np.roll(original, 1, axis=1), atol=1e-13)
        np.testing.assert_allclose(one_pixel[:, 0], original[:, -1], atol=1e-13)

        fractional = 0.31 * pitch
        shifted = np.fft.ifft2(
            spectrum * np.exp(-1j * kx * fractional)[None, :], norm="ortho"
        )
        shifted_one_window = np.fft.ifft2(
            spectrum * np.exp(-1j * kx * (fractional + nx * pitch))[None, :],
            norm="ortho",
        )
        np.testing.assert_allclose(shifted, shifted_one_window, atol=1e-13)

    def test_phase_difference_is_circular_not_raw_subtraction(self):
        reference = np.array([[-1 + 1e-8j, 1 + 0j]], dtype=complex)
        recovered = np.array([[-1 - 1e-8j, 1 + 0j]], dtype=complex)
        raw_difference = np.angle(recovered[0, 0]) - np.angle(reference[0, 0])
        self.assertGreater(np.abs(raw_difference), 6)
        circular_difference = np.angle(recovered * np.conj(reference))
        metrics = _field_metrics(reference, recovered, mask_fraction=0.01)
        self.assertLess(np.max(np.abs(circular_difference)), 3e-8)
        self.assertLess(metrics["bright_phase_rmse_rad"], 3e-8)

        # 8-bit CGH の 0 と 255 は、単純差なら255だが円周上では1階調差。
        reference_codes = np.tile(np.array([0, 255], dtype=np.uint8), (8, 4))
        candidate_codes = 255 - reference_codes
        code_metrics = _cgh_metrics(
            reference_codes,
            candidate_codes,
            np.ones(reference_codes.shape, dtype=complex),
            mask_fraction=0.01,
        )
        self.assertEqual(code_metrics["circular_code_mae_all_pixels"], 1)
        # Raw image metrics see a 255-level jump, whereas the periodic phase
        # difference is only one of 256 levels.
        self.assertLess(code_metrics["raw_8bit_code_ssim_wrap_sensitive"], 1.0)
        self.assertGreater(
            code_metrics["circular_code_psnr_db"],
            code_metrics["raw_8bit_code_psnr_db_wrap_sensitive"] + 20.0,
        )
        self.assertGreater(code_metrics["uniform_mask_field_snr_db"], 20.0)

    def test_dark_candidate_amplitude_is_reported_as_undefined_phase(self):
        reference = np.ones((2, 2), dtype=np.complex128)
        for amplitude in (0.0, 0.005):
            with self.subTest(candidate_amplitude=amplitude):
                candidate = np.full_like(reference, 1j * amplitude)
                metrics = _field_metrics(reference, candidate, mask_fraction=0.01)
                self.assertGreater(metrics["complex_relative_l2"], 0.99)
                self.assertIsNone(metrics["bright_phase_mae_rad"])
                self.assertIsNone(metrics["bright_phase_rmse_rad"])
                self.assertIsNone(metrics["intensity_weighted_phase_mae_rad"])
                self.assertEqual(
                    metrics["phase_excluded_reference_energy_fraction"], 1.0
                )
                self.assertEqual(
                    metrics["bright_reference_dark_candidate_fraction"], 1.0
                )

    def test_small_grid_pipeline_writes_consistent_metrics_and_manifest(self):
        # 全15 mode・10 modeの重み付き和・3本のx shiftを主経路で通す。
        config = StageAConfig(grid_size=16, rank=4,
                              randomized_oversampling=4, thread_count=3,
                              thread_spacing_m=3.74e-6)
        with tempfile.TemporaryDirectory(prefix="stage_a_test_") as scratch:
            output_dir = Path(scratch) / "results"
            summary = run_stage_a(output_dir, config, save_images=False)
            self.assertEqual(summary["metric_schema_version"], METRIC_SCHEMA_VERSION)
            self.assertEqual(summary["mode_design"]["mode_count"], 15)
            self.assertEqual(len(summary["selected_modes"]["mode_indices"]), 10)
            self.assertEqual(len(summary["thread_x_positions_m"]), 3)
            self.assertLess(
                summary["mode_error_max"]["parseval_relative_l2_absolute_difference"],
                1e-12,
            )
            self.assertLess(
                abs(summary["shifted"]["fft"]["complex_relative_l2"]
                    - summary["shifted"]["spatial"]["complex_relative_l2"]),
                1e-12,
            )
            shifted_time = summary["timings_seconds"]["shifted_online_seconds"]
            self.assertEqual(summary["shifted_direct_beam_evaluations"], 30)
            self.assertIn("direct_bessel_recomputation_and_sum_seconds", shifted_time)
            self.assertIn("direct_field_plus_cgh_seconds", shifted_time)
            self.assertIn("spatial_factor_reconstruction_seconds", shifted_time)
            self.assertNotIn("proposed_ifft_seconds", shifted_time)
            for case in ("coaxial", "shifted"):
                benchmark = summary["timings_seconds"][case + "_reconstruction_benchmark"]
                self.assertEqual(len(benchmark["spatial_factor_reconstruction_samples_seconds"]),
                                 config.reconstruction_timing_repeats)
                self.assertLess(benchmark["spatial_factor_vs_fft_ifft_complex_relative_l2"],
                                1e-12)
            with np.load(output_dir / "compressed_library.npz") as library:
                self.assertEqual(library["factor_y_spatial"].shape, (16, 4))
                self.assertEqual(library["factor_x_spatial"].shape, (16, 4))
            self.assertNotIn("direct_uncompressed_fft_generation_seconds", shifted_time)
            self.assertIn("direct_vs_periodic_spatial_complex_relative_l2",
                          summary["shifted"]["periodic_shift_model"])
            for filename in ("compressed_library.npz", "mode_metrics.csv",
                             "summary.json", "timings.json", "manifest.json"):
                self.assertTrue((output_dir / filename).is_file(), filename)
            with (output_dir / "mode_metrics.csv").open(newline="", encoding="utf-8") as stream:
                mode_rows = list(csv.DictReader(stream))
            self.assertEqual(len(mode_rows), 15)
            for prefix in ("fft", "spatial"):
                for metric in ("amplitude_ssim", "amplitude_psnr_db",
                               "intensity_ssim", "intensity_psnr_db",
                               "complex_snr_db"):
                    self.assertIn(f"{prefix}_{metric}", mode_rows[0])
            for result_name in ("coaxial", "shifted"):
                for domain in ("fft", "spatial"):
                    quality = summary[result_name][domain]
                    for metric in ("amplitude_ssim", "amplitude_psnr_db",
                                   "intensity_ssim", "intensity_psnr_db",
                                   "complex_snr_db"):
                        self.assertIn(metric, quality)
                for metric in ("raw_8bit_code_ssim_wrap_sensitive",
                               "raw_8bit_code_psnr_db_wrap_sensitive",
                               "circular_code_psnr_db", "uniform_mask_field_snr_db"):
                    self.assertIn(metric, summary[result_name]["cgh"])
            manifest = json.loads((output_dir / "manifest.json").read_text(
                encoding="utf-8"
            ))
            self.assertIn("summary.json", manifest["files"])
            self.assertIn("manifest.json", manifest["files"])
            with self.assertRaises(FileExistsError):
                run_stage_a(output_dir, config, save_images=False)


if __name__ == "__main__":
    unittest.main()
