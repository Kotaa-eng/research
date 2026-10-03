"""表示済みでない時間を誤って表示時間と報告しないための小規模テスト。

未実行。SDK・SLMは使わず、フィールド生成は小さい定数場へ置き換える。
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lsh_digit1_slm_timing import DisplayReceipt, run_slm_timing
from lsh_digit1_theory import Digit1Config


class SLMTimingTests(unittest.TestCase):
    @staticmethod
    def config():
        return Digit1Config(grid_size=64, pixel_pitch_m=10e-6,
                            aperture_radius_m=.25e-3, evaluation_x_half_width_m=.2e-3,
                            digit_width_m=.12e-3, thread_count=3, thread_spacing_m=30e-6)

    def run_case(self, folder, **kwargs):
        with patch('lsh_digit1_theory.initial_bessel_field', return_value=np.ones((64, 64), dtype=complex)), \
             patch('lsh_digit1_theory.encode_and_replay', side_effect=AssertionError('validation replay timed')), \
             patch('numpy.fft.fft2', side_effect=AssertionError('full FFT in mask-only timing')):
            return run_slm_timing(folder, self.config(), reference_scale=1,
                                  method='direct', repeats=1, warmup=0, **kwargs)

    def test_disconnected_display_times_remain_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_case(Path(tmp)/'run')
            self.assertFalse(result['hardware_connected'])
            stats = result['statistics_seconds']
            self.assertIsNone(stats['update_to_api_return_seconds'])
            self.assertIsNone(stats['update_to_controller_confirmed_seconds'])
            self.assertIsNone(stats['update_to_optically_confirmed_seconds'])
            self.assertGreaterEqual(stats['update_to_frame_ready_seconds']['median'], 0)

    def test_submission_return_is_not_display_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_case(Path(tmp)/'run', expected_slm_shape=(64, 64),
                                   display_callback=lambda frame: None)
            stats = result['statistics_seconds']
            self.assertIsNotNone(stats['update_to_api_return_seconds'])
            self.assertIsNone(stats['update_to_controller_confirmed_seconds'])
            self.assertIsNone(stats['update_to_optically_confirmed_seconds'])

    def test_controller_confirmation_is_not_optical_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_case(Path(tmp)/'run', expected_slm_shape=(64, 64),
                display_callback=lambda frame: DisplayReceipt('controller_confirmed', 'test fake'))
            stats = result['statistics_seconds']
            self.assertIsNotNone(stats['update_to_controller_confirmed_seconds'])
            self.assertIsNone(stats['update_to_optically_confirmed_seconds'])

    def test_wrong_device_shape_never_reaches_display_callback(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No implicit resizing'):
                self.run_case(Path(tmp)/'run', expected_slm_shape=(32, 64),
                              display_callback=lambda frame: seen.append(frame))
        self.assertEqual(seen, [])


if __name__ == '__main__':
    unittest.main()
