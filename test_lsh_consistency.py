"""小規模な整合性テスト。研究計算とは独立。こちらでは未実行。

ユーザーが実行する場合: python -m unittest test_lsh_consistency
"""
import unittest
import numpy as np
from lsh_core import (PhaseOnlyCodec, fit_tucker, phase_metrics, relative_error,
                      spatial_factors, shift_operators, sum_shifted_cores,
                      geometric_reach, effective_output_radius,
                      illumination_aperture)
from thread_cgh import axial_intensity


class ConsistencyTests(unittest.TestCase):
    def test_phase_branch_is_not_physical_error(self):
        a = np.array([[-1+1e-8j, 1+0j]])
        b = np.array([[-1-1e-8j, 1+0j]])
        self.assertGreater(abs(np.angle(a[0, 0])-np.angle(b[0, 0])), 6)
        valid, delta, metrics = phase_metrics(a, b)
        self.assertTrue(valid.all())
        self.assertLess(np.max(abs(delta)), 3e-8)
        self.assertLess(metrics['field_relative_L2'], 3e-8)

    def test_failed_reconstruction_is_not_masked_out(self):
        valid, _, metrics = phase_metrics(np.ones((4, 4)), np.zeros((4, 4)))
        self.assertTrue(valid.all())
        self.assertEqual(metrics['field_relative_L2'], 1)
        self.assertEqual(metrics['reference_bright_reconstruction_dark_fraction'], 1)
        self.assertIsNone(metrics['phase_rmse_rad'])

    def test_parseval_and_coaxial_linear_combination(self):
        rng = np.random.default_rng(9)
        # 既知の小さな空間部分空間を持つ辞書。軸順序とRe/Imを確認。
        uy, _ = np.linalg.qr(rng.normal(size=(16, 3)))
        ux, _ = np.linalg.qr(rng.normal(size=(16, 3)))
        core = rng.normal(size=(4, 3, 3))+1j*rng.normal(size=(4, 3, 3))
        spectra = np.stack([uy @ c @ ux.T for c in core])
        by, bx, compressed = fit_tucker(spectra, 3)
        weights = rng.normal(size=4)+1j*rng.normal(size=4)
        recovered_fft = by @ np.tensordot(weights, compressed, axes=(0, 0)) @ bx.T
        exact_fft = np.tensordot(weights, spectra, axes=(0, 0))
        reference = np.fft.ifft2(exact_fft, norm='ortho')
        recovered = np.fft.ifft2(recovered_fft, norm='ortho')
        self.assertLess(relative_error(reference, recovered), 3e-6)
        self.assertAlmostEqual(relative_error(reference, recovered),
                               relative_error(exact_fft, recovered_fft), places=12)

    def test_codec_respects_four_f_coordinate_scaling(self):
        rng = np.random.default_rng(4)
        field = rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32))
        scale = np.abs(field).max()
        out = PhaseOnlyCodec(field.shape, 2e-6, 8e-6, [8e4], scale)
        equivalent_slm = PhaseOnlyCodec(field.shape, 8e-6, 8e-6, [2e4], scale)
        self.assertTrue(np.array_equal(out.pupil, equivalent_slm.pupil))
        codes = out.encode(field)[1]
        self.assertTrue(np.array_equal(codes, equivalent_slm.encode(field)[1]))
        self.assertTrue(np.allclose(out.replay(codes), equivalent_slm.replay(codes)))
        phase, saturation = out.encode_phase(field)
        self.assertTrue(np.array_equal(codes, out.quantize(phase)))
        self.assertEqual(saturation, out.encode(field)[2])

    def test_shifted_core_projection_matches_explicit_projection(self):
        rng = np.random.default_rng(6)
        uy, _ = np.linalg.qr(rng.normal(size=(8, 3)))
        ux, _ = np.linalg.qr(rng.normal(size=(8, 3)))
        core = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
        ky = 2*np.pi*np.fft.fftfreq(8)
        kx = ky
        sy, sx = 1, -2
        dy = np.exp(-1j*ky*sy)
        dx = np.exp(-1j*kx*sx)
        shifted_full = dy[:, None]*(uy @ core @ ux.T)*dx[None, :]
        ay = uy.T @ (dy[:, None]*uy)
        ax = ux.T @ (dx[:, None]*ux)
        shifted_core = uy @ (ay @ core @ ax) @ ux.T
        explicit_projection = (uy @ uy.T) @ shifted_full @ (ux @ ux.T)
        self.assertTrue(np.allclose(shifted_core, explicit_projection))
        self.assertGreater(relative_error(shifted_full, shifted_core), 1e-6)

    def test_origin_evaluation_odd_grid(self):
        # z=0で離散帯域制限場の原点を評価。奇数グリッドは中央の実サンプルに一致。
        n = 9
        field = np.zeros((n, n), dtype=np.complex128)
        field[n//2, n//2] = 2+1j
        value = axial_intensity(field, 8e-6, 532e-9, [0], padding_factor=3)[0]
        self.assertAlmostEqual(value, 5, places=10)

    def test_subpixel_shifted_core_is_explicit_projection(self):
        rng = np.random.default_rng(11)
        uy, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        ux, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        cores = rng.normal(size=(3, 4, 4))+1j*rng.normal(size=(3, 4, 4))
        coefficients = rng.normal(size=(2, 3))+1j*rng.normal(size=(2, 3))
        pitch = 0.935e-6
        positions = np.array([[0, 0], [3.4*pitch, -1.25*pitch]])
        total = sum_shifted_cores(cores, coefficients, uy, ux, pitch, positions)
        ky = 2*np.pi*np.fft.fftfreq(12, pitch)
        kx = ky
        full = np.zeros((12, 12), complex)
        for weights, (dx, dy) in zip(coefficients, positions):
            centered = uy @ np.tensordot(weights, cores, axes=(0, 0)) @ ux.T
            full += np.exp(-1j*ky*dy)[:, None]*centered*np.exp(-1j*kx*dx)[None, :]
        explicit_projection = (uy@uy.T) @ full @ (ux@ux.T)
        self.assertTrue(np.allclose(uy @ total @ ux.T, explicit_projection))
        ay, ax = shift_operators(uy, ux, pitch, positions[1])
        self.assertTrue(np.allclose(total,
            np.tensordot(coefficients[0], cores, axes=(0, 0)) +
            ay @ np.tensordot(coefficients[1], cores, axes=(0, 0)) @ ax))
        cached = [shift_operators(uy, ux, pitch, position) for position in positions]
        self.assertTrue(np.allclose(total, sum_shifted_cores(
            cores, coefficients, uy, ux, pitch, positions, operators=cached)))

    def test_spatial_basis_absorbs_ifft(self):
        rng = np.random.default_rng(13)
        uy, _ = np.linalg.qr(rng.normal(size=(13, 5)))
        ux, _ = np.linalg.qr(rng.normal(size=(11, 4)))
        core = rng.normal(size=(5, 4))+1j*rng.normal(size=(5, 4))
        vy, vx = spatial_factors(uy, ux)
        direct = vy @ core @ vx.T
        fft_then_ifft = np.fft.ifft2(uy @ core @ ux.T, norm='ortho')
        self.assertLess(relative_error(fft_then_ifft, direct), 1e-13)

    def test_reach_uses_4f_and_reports_ideal_limit(self):
        radius, limits, ideal = effective_output_radius(2048, 3.74e-6, .25)
        self.assertTrue(ideal)
        self.assertAlmostEqual(radius, 2047*3.74e-6*.25/2)
        constrained, _, ideal = effective_output_radius(
            2048, 3.74e-6, .25, laser_diameter=2e-3)
        self.assertTrue(ideal)  # lens/irisがまだ未確定なので到達距離は上限のまま
        self.assertAlmostEqual(constrained, .25e-3)
        self.assertLess(geometric_reach([5e4], 532e-9, constrained)[0],
                        geometric_reach([5e4], 532e-9, radius)[0])
        _, _, all_limits_specified = effective_output_radius(
            2048, 3.74e-6, .25, laser_diameter=2e-3,
            lens_diameter_output=1e-3, iris_diameter_output=1e-3)
        self.assertFalse(all_limits_specified)

    def test_gaussian_illumination_waist_maps_to_output_plane(self):
        n, output_pitch, magnification = 101, 1e-6, .25
        waist_slm = 40e-6
        envelope = illumination_aperture(
            n, output_pitch, magnification, model='gaussian_fixed',
            laser_waist=waist_slm)
        center = n//2
        self.assertAlmostEqual(envelope[center, center], 1)
        self.assertAlmostEqual(envelope[center, center+10], np.exp(-1), places=6)


if __name__ == '__main__':
    unittest.main()
