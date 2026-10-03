"""Fixed simulation-only conditions for the Stage A algorithmic validation.

The grid pitch is a numerical sampling interval. It is deliberately not
interpreted as an SLM pixel, a 4f output pitch, or a physical aperture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


J0_FIRST_ZERO = 2.404825557695773


@dataclass(frozen=True)
class StageAConfig:
    wavelength_m: float = 532e-9
    grid_size: int = 2048
    pixel_pitch_m: float = 3.74e-6
    spot_radius_m: float = 50e-6
    design_length_m: float = 0.50
    rank: int = 80
    seed: int = 42
    selected_mode_count: int = 10
    thread_count: int = 10
    thread_spacing_m: float = 150e-6
    phase_mask_fraction: float = 0.01
    randomized_oversampling: int = 16
    randomized_power_iterations: int = 2
    reconstruction_timing_repeats: int = 3

    def __post_init__(self) -> None:
        if self.grid_size < 4 or self.grid_size % 2:
            raise ValueError("grid_size must be even and at least 4")
        if not 1 <= self.rank <= self.grid_size:
            raise ValueError("rank must be between 1 and grid_size")
        if self.selected_mode_count < 1 or self.thread_count < 1:
            raise ValueError("selected_mode_count and thread_count must be positive")
        if any(x <= 0 for x in (self.wavelength_m, self.pixel_pitch_m,
                               self.spot_radius_m, self.design_length_m,
                               self.thread_spacing_m)):
            raise ValueError("all physical lengths and the sampling pitch must be positive")
        if not 0 < self.phase_mask_fraction < 1:
            raise ValueError("phase_mask_fraction must be in (0, 1)")
        if self.randomized_oversampling < 0 or self.randomized_power_iterations < 0:
            raise ValueError("randomized SVD controls must be nonnegative")
        if self.reconstruction_timing_repeats < 1:
            raise ValueError("reconstruction_timing_repeats must be positive")


@dataclass(frozen=True)
class ModeDesign:
    m: np.ndarray
    kz_per_m: np.ndarray
    kr_per_m: np.ndarray
    k0_per_m: float
    q_per_m: float
    kr_center_per_m: float

    @property
    def count(self) -> int:
        return len(self.m)

    @property
    def kz(self) -> np.ndarray:
        """Alias convenient for notebooks and small mathematical tests."""
        return self.kz_per_m

    @property
    def kr(self) -> np.ndarray:
        """Alias convenient for notebooks and small mathematical tests."""
        return self.kr_per_m


def design_modes(config: StageAConfig) -> ModeDesign:
    """Select the largest symmetric propagating Frozen-Wave-style mode set."""
    k0 = 2 * np.pi / config.wavelength_m
    kr_center = J0_FIRST_ZERO / config.spot_radius_m
    if kr_center >= k0:
        raise ValueError("spot radius produces kr_center >= k0")
    q = float(np.sqrt((k0 - kr_center) * (k0 + kr_center)))
    m_max = int(np.floor(min(k0 - q, q) * config.design_length_m / (2 * np.pi)))
    # Handle a floating-point boundary at kz=0 or kz=k0 conservatively.
    while m_max > 0 and (q - 2 * np.pi * m_max / config.design_length_m <= 0 or
                         q + 2 * np.pi * m_max / config.design_length_m > k0):
        m_max -= 1
    m = np.arange(-m_max, m_max + 1, dtype=np.int32)
    kz = q + 2 * np.pi * m / config.design_length_m
    kr = np.sqrt(np.maximum(0.0, (k0 - kz) * (k0 + kz)))
    if len(m) < config.selected_mode_count:
        raise ValueError(f"only {len(m)} propagating modes, fewer than selected_mode_count")
    if float(np.max(kr)) >= np.pi / config.pixel_pitch_m:
        raise ValueError("kr exceeds the numerical grid's Nyquist wavenumber")
    return ModeDesign(m=m, kz_per_m=kz, kr_per_m=kr, k0_per_m=float(k0),
                      q_per_m=q, kr_center_per_m=float(kr_center))
