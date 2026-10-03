"""圧縮診断の評価と、失敗時にCGH/伝搬へ進まないことを確認する。

実行する場合: python -m unittest test_lsh_digits123_rank_audit
研究規模の計算はしない。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import lsh_digit1_proposed as proposed
from lsh_digit1_theory import Digit1Config
from lsh_digits123_rank_audit import initial_scores


class RankAuditTests(unittest.TestCase):
    def test_intensity_match_alone_does_not_hide_complex_field_error(self):
        reference = np.ones((4, 4), dtype=complex)
        result = initial_scores(reference, 1j*reference)
        self.assertAlmostEqual(result["intensity_relative_l2"], 0)
        self.assertAlmostEqual(result["power_ratio_vs_direct"], 1)
        self.assertAlmostEqual(result["complex_relative_l2"], np.sqrt(2))

    def test_failed_compression_stops_before_encoding_and_propagation(self):
        config = Digit1Config(grid_size=32, pixel_pitch_m=20e-6,
                              aperture_radius_m=.25e-3, evaluation_x_half_width_m=.2e-3,
                              digit_width_m=.12e-3, thread_count=3, thread_spacing_m=50e-6,
                              coefficient_samples=32, z_samples=3)
        direct = np.ones((32, 32), dtype=complex)
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch)/"failed"
            with patch.object(proposed, "load_or_build_library", return_value=({}, {})), \
                 patch.object(proposed, "truncate_library", side_effect=lambda library, rank: library), \
                 patch.object(proposed.theory, "initial_bessel_field", return_value=direct), \
                 patch.object(proposed, "proposed_field", return_value=(np.zeros_like(direct), {}, np.zeros((1, 1)))), \
                 patch.object(proposed.theory, "encode_and_replay") as encoder, \
                 patch.object(proposed.theory, "propagate") as propagator:
                with self.assertRaisesRegex(ValueError, "Compression precheck failed"):
                    proposed.run_comparison(out, config, rank=1, save_images=False,
                                            run_shift_diagnostics=False, max_initial_complex_error=.01)
                encoder.assert_not_called()
                propagator.assert_not_called()
            report = json.loads((out/"compression_precheck.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertAlmostEqual(report["complex_relative_l2"], 1)
            self.assertFalse((out/"summary.json").exists())

    def test_invalid_parent_rank_fails_before_library_preparation(self):
        with patch.object(proposed, "load_or_build_library") as loader:
            with self.assertRaisesRegex(ValueError, "library_parent_rank"):
                proposed.run_comparison("unused_output", rank=100, library_parent_rank=80)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
