"""小規模な整合性テスト。研究計算とは独立。こちらでは未実行。

ユーザーが実行する場合: python -m unittest test_lsh_consistency
"""
import unittest
import numpy as np
from lsh_core import PhaseOnlyCodec, fit_tucker, phase_metrics, relative_error
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


if __name__ == '__main__':
    unittest.main()
