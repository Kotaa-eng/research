"""3D係数設計の複素線形写像と、直接J0計算との整合性を確認する。"""
import unittest
from types import SimpleNamespace

import numpy as np

import lsh_multisheet as multisheet
import lsh_multisheet_design as design3d


class JointDesignTests(unittest.TestCase):
    def test_regularized_projection_satisfies_complex_normal_equations(self):
        rng = np.random.default_rng(314)
        b = rng.normal(size=(2, 7, 3)).astype(np.float32)
        phase = np.exp(2j*np.pi*np.arange(8)[:, None]*np.array([-1, 1])[None, :]/8)
        desired = rng.normal(size=(8, 7))+1j*rng.normal(size=(8, 7))
        penalties = np.array([.07, .13])
        projectors = np.stack([np.linalg.solve(matrix.T @ matrix+penalty*np.eye(3), matrix.T)
                              for matrix, penalty in zip(b, penalties)])
        weights = design3d._project(projectors, phase, desired)
        modes = phase.conj().T @ desired/8
        for m in range(2):
            residual = b[m].T @ (b[m] @ weights[m]-modes[m])+penalties[m]*weights[m]
            np.testing.assert_allclose(residual, 0, atol=1e-6)
        # 位相を自由にした目標振幅目的関数でも、正則化写像は初期値から下げる。
        initial = (rng.normal(size=(2, 3))+1j*rng.normal(size=(2, 3))).astype(np.complex64)
        field = design3d._forward(b, phase, initial)
        amplitude = np.abs(desired)
        constrained = amplitude*np.exp(1j*np.angle(field))
        update = design3d._project(projectors, phase, constrained)
        before, _ = design3d._objective(field, amplitude, initial, penalties)
        after, _ = design3d._objective(design3d._forward(b, phase, update), amplitude, update, penalties)
        self.assertLessEqual(after, before+1e-6)

    def test_analytic_volume_at_zero_matches_fresh_direct_field(self):
        axis = (np.arange(16)-8)*.1
        xs, ys = np.array([-.13, .27]), np.array([-.17, .22])
        kr = np.array([2., 3.])
        design = SimpleNamespace(m=np.array([-1, 1]), kr_per_m=kr,
                                 k0_per_m=10., kz_per_m=np.sqrt(100-kr**2))
        coefficients = np.array([[[1+.2j, -.5j], [-.7j, .2]],
                                 [[.3j, .8], [.4, -.1j]]])
        actual = design3d.analytic_volume(axis, axis, np.array([0.]), xs, ys, design, coefficients)[0]
        expected = multisheet.direct_multisheet_field(axis, xs, ys, kr, coefficients, np.ones(2))
        np.testing.assert_allclose(actual, expected, atol=2e-12)


if __name__ == "__main__":
    unittest.main()
