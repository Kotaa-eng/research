"""Stage A: numerical validation of shared-basis FFT/Tucker core summation.

Run ``python stage_a_validation.py --output-dir stage_a_outputs``. This module
uses an unwindowed Bessel J0 field on a *simulation* grid. The direct shifted
baseline regenerates each displaced beam in real space; the proposed Fourier
shift is periodic on that grid. No illumination, physical aperture, SLM, 4f system,
or propagation model is present. All online timers exclude dictionary fitting
and figure/file writing. The phase-only CGH is an explicitly ideal numerical
angle-and-8-bit codec, not a model of optical replay.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from scipy.special import j0

from stage_a_parameters import J0_FIRST_ZERO, ModeDesign, StageAConfig, design_modes

METRIC_SCHEMA_VERSION = 4


def _radius_grid(config: StageAConfig) -> np.ndarray:
    axis = (np.arange(config.grid_size, dtype=np.float64)
            - (config.grid_size - 1) / 2) * config.pixel_pitch_m
    return np.hypot(axis[:, None], axis[None, :])


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    difference = np.linalg.norm(candidate - reference)
    return float(difference / denominator) if denominator else (0.0 if not difference else float("inf"))


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    """Power SNR = 10 log10(sum |reference|² / sum |error|²)."""
    signal_norm = float(np.linalg.norm(reference))
    error_norm = float(np.linalg.norm(candidate - reference))
    if signal_norm == 0:
        return None
    if error_norm == 0:
        return float("inf")
    return float(20 * np.log10(signal_norm / error_norm))


def _ssim_gray(reference: np.ndarray, candidate: np.ndarray,
               window: int) -> float:
    """Grayscale SSIM with a uniform local window and sample covariance.

    Inputs are real images scaled by the same reference peak, so data_range=1.
    The local SSIM map is averaged after excluding the half-window border.
    """
    mean_ref = uniform_filter(reference, size=window, mode="reflect")
    mean_cand = uniform_filter(candidate, size=window, mode="reflect")
    mean_ref_sq = mean_ref * mean_ref
    mean_cand_sq = mean_cand * mean_cand
    mean_product = mean_ref * mean_cand
    covariance_scale = window**2 / (window**2 - 1)
    var_ref = covariance_scale * (
        uniform_filter(reference * reference, size=window, mode="reflect")
        - mean_ref_sq)
    var_cand = covariance_scale * (
        uniform_filter(candidate * candidate, size=window, mode="reflect")
        - mean_cand_sq)
    covariance = covariance_scale * (
        uniform_filter(reference * candidate, size=window, mode="reflect")
        - mean_product)
    c1, c2 = 0.01**2, 0.03**2
    similarity = ((2 * mean_product + c1) * (2 * covariance + c2)
                  / ((mean_ref_sq + mean_cand_sq + c1)
                     * (var_ref + var_cand + c2)))
    border = window // 2
    return float(np.mean(similarity[border:-border, border:-border]))


def _image_metrics(reference: np.ndarray, candidate: np.ndarray,
                   *, peak_value: float | None = None) -> dict:
    """Full-grid grayscale SSIM, reference-peak PSNR, and power SNR.

    Both images share the reference peak scale; the candidate is never
    independently normalized. SSIM uses a 7x7 window (smaller on test grids).
    """
    if np.iscomplexobj(reference) or np.iscomplexobj(candidate):
        raise TypeError("SSIM and PSNR require real grayscale images")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("image metrics require two real 2D arrays of equal shape")
    peak = float(np.max(reference) if peak_value is None else peak_value)
    error_norm = float(np.linalg.norm(candidate - reference))
    if peak <= 0:
        return {"ssim": 1.0 if error_norm == 0 else None,
                "psnr_db": None, "snr_db": _snr_db(reference, candidate)}
    psnr = (float("inf") if error_norm == 0 else
            float(20 * np.log10(peak * np.sqrt(reference.size) / error_norm)))
    if error_norm == 0:
        ssim = 1.0
    else:
        side = min(reference.shape)
        window = min(7, side if side % 2 else side - 1)
        ssim = (_ssim_gray(reference / peak, candidate / peak, window)
                if window >= 3 else None)
    return {"ssim": ssim, "psnr_db": psnr,
            "snr_db": _snr_db(reference, candidate)}


def _field_metrics(reference: np.ndarray, candidate: np.ndarray,
                   mask_fraction: float) -> dict:
    """Complex, amplitude, intensity and circular phase errors.

    The phase threshold is fixed by the reference peak. Phase statistics use
    pixels bright in both fields; lost bright-reference pixels are counted.
    """
    reference_amp = np.abs(reference)
    candidate_amp = np.abs(candidate)
    threshold = mask_fraction * float(reference_amp.max())
    mask = reference_amp > threshold
    defined = mask & (candidate_amp > threshold)
    phase_difference = np.angle(candidate * np.conj(reference))
    reference_intensity = reference_amp**2
    candidate_intensity = candidate_amp**2
    amplitude_image_metrics = _image_metrics(reference_amp, candidate_amp)
    intensity_image_metrics = _image_metrics(reference_intensity, candidate_intensity)
    reference_energy = float(reference_intensity.sum())
    weighted_denominator = float(reference_intensity[defined].sum())
    undefined_phase = mask & ~defined
    return {
        "complex_relative_l2": _relative_l2(reference, candidate),
        "complex_snr_db": _snr_db(reference, candidate),
        "amplitude_relative_l2": _relative_l2(reference_amp, candidate_amp),
        "amplitude_ssim": amplitude_image_metrics["ssim"],
        "amplitude_psnr_db": amplitude_image_metrics["psnr_db"],
        "amplitude_snr_db": amplitude_image_metrics["snr_db"],
        "intensity_relative_l2": _relative_l2(reference_intensity, candidate_intensity),
        "intensity_ssim": intensity_image_metrics["ssim"],
        "intensity_psnr_db": intensity_image_metrics["psnr_db"],
        "intensity_snr_db": intensity_image_metrics["snr_db"],
        "bright_phase_mae_rad": float(np.mean(np.abs(phase_difference[defined]))) if defined.any() else None,
        "bright_phase_rmse_rad": float(np.sqrt(np.mean(phase_difference[defined]**2))) if defined.any() else None,
        "intensity_weighted_phase_mae_rad": (
            float(np.sum(reference_intensity[defined] * np.abs(phase_difference[defined]))
                  / weighted_denominator) if weighted_denominator else None
        ),
        "phase_excluded_reference_energy_fraction": (
            float(reference_intensity[undefined_phase].sum()
                  / reference_energy) if reference_energy else None
        ),
        "phase_mask_pixel_fraction": float(mask.mean()),
        "bright_reference_dark_candidate_fraction": (
            float(np.mean(candidate_amp[mask] <= threshold))
            if mask.any() else None
        ),
    }


def _save_scalar(path: Path, values: np.ndarray, *, cmap: str = "gray",
                 vmin: float | None = None, vmax: float | None = None,
                 fft_shift: bool = False, mask: np.ndarray | None = None) -> None:
    """Write one image pixel per grid sample, without resizing/cropping."""
    data = np.fft.fftshift(values) if fft_shift else values
    if mask is not None:
        valid = np.fft.fftshift(mask) if fft_shift else mask
        data = np.ma.array(data, mask=~valid)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad((0.5, 0.5, 0.5, 1.0))
    plt.imsave(path, data, cmap=color_map, vmin=vmin, vmax=vmax, origin="lower")


def _save_complex_images(directory: Path, prefix: str, field: np.ndarray,
                         *, reference: np.ndarray | None = None,
                         fft_domain: bool = False,
                         mask_fraction: float = 0.01) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    amplitude = np.abs(field)
    reference_amplitude = np.abs(reference) if reference is not None else amplitude
    scale = max(float(reference_amplitude.max()), np.finfo(float).tiny)
    mask = ((reference_amplitude > mask_fraction * scale)
            & (amplitude > mask_fraction * scale))
    created = []
    amplitude_path = directory / f"{prefix}_amplitude.png"
    _save_scalar(amplitude_path, amplitude / scale, vmin=0, vmax=1,
                 fft_shift=fft_domain)
    created.append(str(amplitude_path))
    phase_path = directory / f"{prefix}_phase_masked.png"
    _save_scalar(phase_path, np.angle(field), cmap="twilight", vmin=-np.pi,
                 vmax=np.pi, fft_shift=fft_domain, mask=mask)
    created.append(str(phase_path))
    if fft_domain:
        db_path = directory / f"{prefix}_amplitude_db.png"
        db = 20 * np.log10(np.maximum(amplitude / scale, 1e-4))
        _save_scalar(db_path, db, vmin=-80, vmax=0, fft_shift=True)
        created.append(str(db_path))
    if reference is not None:
        difference_path = directory / f"{prefix}_complex_abs_error.png"
        _save_scalar(difference_path, np.abs(field - reference) / scale,
                     vmin=0, fft_shift=fft_domain)
        created.append(str(difference_path))
        phase_difference_path = directory / f"{prefix}_circular_phase_error.png"
        circular_difference = np.angle(field * np.conj(reference))
        _save_scalar(phase_difference_path, circular_difference,
                     cmap="twilight", vmin=-np.pi, vmax=np.pi,
                     fft_shift=fft_domain, mask=mask)
        created.append(str(phase_difference_path))
    return created


def _randomized_shared_basis(spectra: np.ndarray, config: StageAConfig) -> np.ndarray:
    """Streaming randomized HOSVD basis for the stacked Re/Im unfoldings.

    For J0 on a square centered grid, every spectrum is transpose-symmetric.
    Therefore the ky and kx mode unfoldings have the same Gram matrix, and
    one real orthonormal factor can be used for both axes. The complex mode
    index and Re/Im axes are *not* compressed.
    """
    mode_count, grid_size, _ = spectra.shape
    sketch_size = min(grid_size, config.rank + config.randomized_oversampling)
    rng = np.random.default_rng(config.seed + 101)
    sketch = np.zeros((grid_size, sketch_size), dtype=np.float64)
    for mode in range(mode_count):
        spectrum = spectra[mode]
        for component in (spectrum.real, spectrum.imag):
            omega = rng.standard_normal((grid_size, sketch_size), dtype=np.float32)
            sketch += component @ omega
    q, _ = np.linalg.qr(sketch, mode="reduced")
    del sketch
    for _ in range(config.randomized_power_iterations):
        projected = np.zeros((grid_size, sketch_size), dtype=np.float64)
        for mode in range(mode_count):
            spectrum = spectra[mode]
            for component in (spectrum.real, spectrum.imag):
                projected += component @ (component.T @ q)
        q, _ = np.linalg.qr(projected, mode="reduced")
        del projected
    small_gram = np.zeros((sketch_size, sketch_size), dtype=np.float64)
    for mode in range(mode_count):
        spectrum = spectra[mode]
        for component in (spectrum.real, spectrum.imag):
            compressed = component.T @ q
            small_gram += compressed.T @ compressed
    eigenvalues, eigenvectors = np.linalg.eigh(small_gram)
    selection = np.argsort(eigenvalues)[-config.rank:][::-1]
    basis = q @ eigenvectors[:, selection]
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return basis


def _project_core(spectrum: np.ndarray, uy: np.ndarray,
                  ux: np.ndarray) -> np.ndarray:
    return uy.T @ (spectrum @ ux)


def _reconstruct_fft(core: np.ndarray, uy: np.ndarray,
                     ux: np.ndarray) -> np.ndarray:
    return (uy @ core) @ ux.T


def _spatial_factors(uy: np.ndarray, ux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offline: inverse-transform each factor column along its spatial axis."""
    return (np.fft.ifft(uy, axis=0, norm="ortho"),
            np.fft.ifft(ux, axis=0, norm="ortho"))


def _reconstruct_spatial(core: np.ndarray, vy: np.ndarray,
                         vx: np.ndarray) -> np.ndarray:
    """IFFT2(Uy C Ux.T) = Vy C Vx.T; use transpose, not conjugate transpose."""
    return (vy @ core) @ vx.T


def _benchmark_reconstruction(core: np.ndarray, uy: np.ndarray, ux: np.ndarray,
                              vy: np.ndarray, vx: np.ndarray, repeats: int) -> dict:
    """Diagnostic comparison outside online timers, with warmup and repeated runs.

    Includes the full FFT reconstruction in the legacy time and both matrix
    products in the spatial-factor time; excludes factor IFFTs in both cases.
    """
    legacy = np.fft.ifft2(_reconstruct_fft(core, uy, ux), norm="ortho")
    spatial = _reconstruct_spatial(core, vy, vx)
    equivalence_error = _relative_l2(legacy, spatial)
    del legacy, spatial
    legacy_times, spatial_times = [], []
    for repeat in range(repeats):
        # Alternate order so one method is not always measured first.
        for method in (("legacy", "spatial") if repeat % 2 == 0
                       else ("spatial", "legacy")):
            tic = time.perf_counter()
            if method == "legacy":
                field = np.fft.ifft2(_reconstruct_fft(core, uy, ux), norm="ortho")
            else:
                field = _reconstruct_spatial(core, vy, vx)
            elapsed = time.perf_counter() - tic
            (legacy_times if method == "legacy" else spatial_times).append(elapsed)
            del field
    legacy_median = float(np.median(legacy_times))
    spatial_median = float(np.median(spatial_times))
    return {
        "repeats": repeats,
        "warmup_runs_per_method": 1,
        "legacy_fft_reconstruction_plus_ifft_samples_seconds": legacy_times,
        "spatial_factor_reconstruction_samples_seconds": spatial_times,
        "legacy_median_seconds": legacy_median,
        "spatial_factor_median_seconds": spatial_median,
        "reconstruction_speedup": legacy_median / spatial_median,
        "spatial_factor_vs_fft_ifft_complex_relative_l2": equivalence_error,
    }


def _x_shift_core_operator(ux: np.ndarray, phase_x: np.ndarray) -> np.ndarray:
    """Project a full Fourier x-translation onto the shared real x basis.

    With F = Uy C Ux.T and F_shift = F Dx, the shifted core is C Ax,
    Ax = Ux.T Dx Ux. This is generally an approximation because Dx Ux can
    leave the column space of Ux.
    """
    if ux.ndim != 2 or phase_x.ndim != 1 or len(phase_x) != ux.shape[0]:
        raise ValueError("ux must be [kx, rank] and phase_x must be [kx]")
    return ux.T @ (phase_x[:, None] * ux)


def _save_factor_figures(output_dir: Path, uy: np.ndarray, ux: np.ndarray,
                         cores: np.ndarray, mode_design: ModeDesign) -> None:
    common_scale = max(float(np.max(np.abs(uy))), float(np.max(np.abs(ux))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for ax, factor, name in zip(axes, (uy, ux), ("Uy (ky)", "Ux (kx)")):
        img = ax.imshow(factor, origin="lower", aspect="auto", cmap="coolwarm",
                        vmin=-common_scale, vmax=common_scale)
        ax.set(title=f"{name}: grid sample x rank", xlabel="rank column",
               ylabel="unshifted FFT sample")
        fig.colorbar(img, ax=ax, fraction=0.045)
    fig.savefig(output_dir / "factor_heatmaps.png", dpi=140)
    plt.close(fig)

    columns = 5
    rows = int(np.ceil(len(cores) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3 * rows),
                             constrained_layout=True)
    common_scale = max(float(np.max(np.abs(cores))), np.finfo(float).tiny)
    for ax, mode, core in zip(np.ravel(axes), mode_design.m, cores):
        ax.imshow(np.abs(core) / common_scale, origin="lower", cmap="gray",
                  vmin=0, vmax=1)
        ax.set(title=f"m={mode:+d}", xlabel="rank x", ylabel="rank y")
    for ax in np.ravel(axes)[len(cores):]:
        ax.axis("off")
    fig.savefig(output_dir / "core_magnitude_all_modes.png", dpi=140)
    plt.close(fig)


def _save_mode_overview(output_dir: Path,
                        samples: list[tuple[int, np.ndarray]],
                        mode_design: ModeDesign) -> None:
    """A compact selected-mode guide; individual mode PNGs retain full N x N."""
    fig, axes = plt.subplots(2, len(samples), figsize=(3.3 * len(samples), 6.5),
                             squeeze=False, constrained_layout=True)
    for column, (index, beam) in enumerate(samples):
        scale = max(float(np.abs(beam).max()), np.finfo(float).tiny)
        phase = np.ma.array(np.angle(beam.astype(np.complex128)),
                            mask=np.abs(beam) <= 0.01 * scale)
        axes[0, column].imshow(np.abs(beam) / scale, origin="lower", cmap="gray",
                               vmin=0, vmax=1, interpolation="nearest")
        axes[0, column].set_title(f"m={int(mode_design.m[index]):+d} |J0|")
        axes[1, column].imshow(phase, origin="lower", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        axes[1, column].set_title("phase [rad], bright pixels")
        for row in range(2):
            axes[row, column].set(xlabel="x (display sample)",
                                   ylabel="y (display sample)")
    fig.suptitle("Representative theoretical Bessel modes (display downsample only)")
    fig.savefig(output_dir / "representative_bessel_modes.png", dpi=140)
    plt.close(fig)


def _save_representative_mode_comparison(output_dir: Path, mode: int,
                                         kr_per_m: float, pixel_pitch_m: float,
                                         spatial_ref: np.ndarray,
                                         spatial_prop: np.ndarray,
                                         fft_ref: np.ndarray,
                                         fft_prop: np.ndarray,
                                         mask_fraction: float) -> None:
    """Show the ring at a readable zoom while raw PNGs retain all FFT samples."""
    n = spatial_ref.shape[0]
    ring_radius_samples = kr_per_m * n * pixel_pitch_m / (2 * np.pi)
    half_width = min(n // 2, max(8, int(np.ceil(1.25 * ring_radius_samples))))
    center = n // 2
    crop = np.s_[center - half_width:center + half_width,
                 center - half_width:center + half_width]
    display_stride = max(1, n // 512)
    space_scale = max(float(np.abs(spatial_ref).max()), np.finfo(float).tiny)
    fft_scale = max(float(np.abs(fft_ref).max()), np.finfo(float).tiny)
    space_mask = ((np.abs(spatial_ref) > mask_fraction * space_scale)
                  & (np.abs(spatial_prop) > mask_fraction * space_scale))
    fft_mask = ((np.abs(fft_ref) > mask_fraction * fft_scale)
                & (np.abs(fft_prop) > mask_fraction * fft_scale))
    panels = [
        (np.abs(spatial_ref)[::display_stride, ::display_stride] / space_scale,
         "Theory |E|", "gray", 0, 1),
        (np.abs(spatial_prop)[::display_stride, ::display_stride] / space_scale,
         "Reconstructed |E|", "gray", 0, 1),
        (np.abs(spatial_prop - spatial_ref)[::display_stride, ::display_stride] / space_scale,
         "Spatial complex absolute error", "gray", 0, None),
        (np.ma.array(np.angle(spatial_prop * np.conj(spatial_ref)),
                     mask=~space_mask)[::display_stride, ::display_stride],
         "Spatial circular phase error", "twilight", -np.pi, np.pi),
        (np.abs(np.fft.fftshift(fft_ref))[crop] / fft_scale,
         "Theory |FFT| (zoom)", "gray", 0, 1),
        (np.abs(np.fft.fftshift(fft_prop))[crop] / fft_scale,
         "Reconstructed |FFT| (zoom)", "gray", 0, 1),
        (np.abs(np.fft.fftshift(fft_prop - fft_ref))[crop] / fft_scale,
         "FFT complex absolute error (zoom)", "gray", 0, None),
        (np.ma.array(np.fft.fftshift(np.angle(fft_prop * np.conj(fft_ref))),
                     mask=~np.fft.fftshift(fft_mask))[crop],
         "FFT circular phase error (zoom)", "twilight", -np.pi, np.pi),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(f"Representative mode m={mode:+d}; full {n} x {n} PNGs saved separately")
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.savefig(output_dir / "representative_mode_comparison.png", dpi=140)
    plt.close(fig)


def _save_case_figure(directory: Path, name: str, field_ref: np.ndarray,
                      field_prop: np.ndarray, fft_ref: np.ndarray,
                      fft_prop: np.ndarray, codes_ref: np.ndarray,
                      codes_prop: np.ndarray, mask_fraction: float) -> None:
    """Compact labeled overview; full-grid images are saved separately."""
    stride = max(1, field_ref.shape[0] // 512)
    amp_scale = max(float(np.abs(field_ref).max()), np.finfo(float).tiny)
    fft_scale = max(float(np.abs(fft_ref).max()), np.finfo(float).tiny)
    field_mask = ((np.abs(field_ref) > mask_fraction * amp_scale)
                  & (np.abs(field_prop) > mask_fraction * amp_scale))
    fft_mask = ((np.abs(fft_ref) > mask_fraction * fft_scale)
                & (np.abs(fft_prop) > mask_fraction * fft_scale))
    code_diff = ((codes_prop.astype(np.int16) - codes_ref.astype(np.int16) + 128) % 256) - 128
    panels = [
        (np.abs(field_ref) / amp_scale, "Reference |E|", "gray", 0, 1),
        (np.abs(field_prop) / amp_scale, "Proposed |E|", "gray", 0, 1),
        (np.abs(field_prop - field_ref) / amp_scale, "|E proposed - reference|", "gray", 0, None),
        (np.angle(field_prop * np.conj(field_ref)), "Circular phase error [rad]", "twilight", -np.pi, np.pi),
        (np.abs(np.fft.fftshift(fft_ref)) / fft_scale, "Reference |FFT|", "gray", 0, 1),
        (np.abs(np.fft.fftshift(fft_prop)) / fft_scale, "Proposed |FFT|", "gray", 0, 1),
        (np.abs(np.fft.fftshift(fft_prop - fft_ref)) / fft_scale, "FFT absolute error", "gray", 0, None),
        (np.fft.fftshift(np.angle(fft_prop * np.conj(fft_ref))), "FFT circular phase error", "twilight", -np.pi, np.pi),
        (codes_ref, "Ideal 8-bit CGH reference", "gray", 0, 255),
        (codes_prop, "Ideal 8-bit CGH proposed", "gray", 0, 255),
        (code_diff, "Circular CGH code difference", "coolwarm", -128, 127),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    fig.suptitle(f"{name}: full-grid numerical comparison")
    for index, (data, title, cmap, vmin, vmax) in enumerate(panels):
        ax = axes.ravel()[index]
        if index == 3:
            data = np.ma.array(data, mask=~field_mask)
        if index == 7:
            data = np.ma.array(data, mask=~np.fft.fftshift(fft_mask))
        img = ax.imshow(data[::stride, ::stride], cmap=cmap, origin="lower",
                        vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set(title=title, xlabel="x sample", ylabel="y sample")
        fig.colorbar(img, ax=ax, fraction=0.044)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 1,
        "FFT shift is periodic on the simulation grid.\n"
        "Phase masks require both amplitudes > 1% of reference peak.\n"
        "CGH = quantized angle(E), no optical replay model.",
        va="top", family="monospace")
    fig.savefig(directory / f"{name}_comparison.png", dpi=130)
    plt.close(fig)


def _quantize_ideal_phase_only(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-amplitude numerical mask: phase=arg(E), 256 cyclic levels."""
    phase = np.mod(np.angle(field), 2 * np.pi)
    codes = (np.rint(phase * (256 / (2 * np.pi))).astype(np.uint16) % 256).astype(np.uint8)
    return phase, codes


def _cgh_metrics(reference_codes: np.ndarray, candidate_codes: np.ndarray,
                 reference_field: np.ndarray, mask_fraction: float) -> dict:
    difference = ((candidate_codes.astype(np.int16) - reference_codes.astype(np.int16) + 128) % 256) - 128
    bright_mask = np.abs(reference_field) > mask_fraction * float(np.abs(reference_field).max())
    phase_difference = difference.astype(np.float64) * (2 * np.pi / 256)
    raw_image_metrics = _image_metrics(reference_codes, candidate_codes,
                                       peak_value=255.0)
    circular_code_rms = float(np.sqrt(np.mean(difference.astype(np.float64)**2)))
    uniform_mask_relative_l2 = float(np.sqrt(max(
        0.0, float(np.mean(2 - 2 * np.cos(phase_difference))))))
    return {
        "different_code_fraction_all_pixels": float(np.mean(difference != 0)),
        "circular_code_mae_all_pixels": float(np.mean(np.abs(difference))),
        "circular_phase_mae_all_pixels_rad": float(np.mean(np.abs(phase_difference))),
        "circular_phase_mae_bright_pixels_rad": (
            float(np.mean(np.abs(phase_difference[bright_mask]))) if bright_mask.any() else None
        ),
        "raw_8bit_code_ssim_wrap_sensitive": raw_image_metrics["ssim"],
        "raw_8bit_code_psnr_db_wrap_sensitive": raw_image_metrics["psnr_db"],
        "circular_code_psnr_db": (
            float("inf") if circular_code_rms == 0 else
            float(20 * np.log10(128.0 / circular_code_rms))
        ),
        "uniform_mask_field_relative_l2": uniform_mask_relative_l2,
        "uniform_mask_field_snr_db": (
            float("inf") if uniform_mask_relative_l2 == 0 else
            float(-20 * np.log10(uniform_mask_relative_l2))
        ),
        "replay_intensity": None,
        "replay_note": "No optical replay model: phase-only mask discards target amplitude.",
    }


def _save_case_images(directory: Path, name: str, field_ref: np.ndarray,
                      field_prop: np.ndarray, fft_ref: np.ndarray,
                      fft_prop: np.ndarray, codes_ref: np.ndarray,
                      codes_prop: np.ndarray, mask_fraction: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _save_complex_images(directory, "reference_spatial", field_ref,
                         mask_fraction=mask_fraction)
    _save_complex_images(directory, "proposed_spatial", field_prop,
                         reference=field_ref, mask_fraction=mask_fraction)
    _save_complex_images(directory, "reference_fft", fft_ref,
                         fft_domain=True, mask_fraction=mask_fraction)
    _save_complex_images(directory, "proposed_fft", fft_prop,
                         reference=fft_ref, fft_domain=True,
                         mask_fraction=mask_fraction)
    Image.fromarray(codes_ref).save(directory / "reference_cgh_8bit.png")
    Image.fromarray(codes_prop).save(directory / "proposed_cgh_8bit.png")
    code_difference = ((codes_prop.astype(np.int16) - codes_ref.astype(np.int16) + 128) % 256) - 128
    _save_scalar(directory / "cgh_circular_code_difference.png", code_difference,
                 cmap="coolwarm", vmin=-128, vmax=127)
    phase_mask = ((np.abs(field_ref) > mask_fraction * float(np.abs(field_ref).max()))
                  & (np.abs(field_prop) > mask_fraction * float(np.abs(field_ref).max())))
    _save_scalar(directory / "unquantized_field_phase_difference_rad.png",
                 np.angle(field_prop * np.conj(field_ref)), cmap="twilight",
                 vmin=-np.pi, vmax=np.pi, mask=phase_mask)
    _save_scalar(directory / "cgh_8bit_circular_phase_difference_rad.png",
                 code_difference.astype(np.float64) * (2 * np.pi / 256),
                 cmap="twilight", vmin=-np.pi, vmax=np.pi)
    _save_case_figure(directory, name, field_ref, field_prop, fft_ref, fft_prop,
                      codes_ref, codes_prop, mask_fraction)


def _save_shift_mechanism(directory: Path, config: StageAConfig,
                          core: np.ndarray, ux: np.ndarray,
                          centered_fft: np.ndarray, exact_shifted_fft: np.ndarray,
                          projected_shifted_fft: np.ndarray, x_shift_m: float,
                          ax_operator: np.ndarray) -> None:
    """Show explicitly where the common-basis projection loses information."""
    phase_x = np.exp(-1j * 2 * np.pi
                     * np.fft.fftfreq(config.grid_size, config.pixel_pitch_m)
                     * x_shift_m)
    scale = max(float(np.abs(centered_fft).max()), np.finfo(float).tiny)
    core_scale = max(float(np.abs(core).max()), np.finfo(float).tiny)
    shifted_core = core @ ax_operator
    stride = max(1, config.grid_size // 512)
    panels = [
        (np.fft.fftshift(np.angle(phase_x))[None, :], "angle(Dx), shift phase [rad]", "twilight"),
        (np.abs(np.fft.fftshift(centered_fft)) / scale, "Centered full FFT amplitude", "gray"),
        (np.abs(np.fft.fftshift(exact_shifted_fft)) / scale, "Exact shifted full FFT amplitude", "gray"),
        (np.abs(np.fft.fftshift(exact_shifted_fft - projected_shifted_fft)) / scale,
         "Total FFT error (Tucker + projection)", "gray"),
        (np.abs(core) / core_scale, "Weighted core H", "gray"),
        (np.abs(ax_operator), "Ax = Ux.T Dx Ux", "gray"),
        (np.abs(shifted_core) / core_scale, "Shifted core H Ax", "gray"),
        (np.abs(np.fft.fftshift(projected_shifted_fft)) / scale,
         "Uy (H Ax) Ux.T", "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    fig.suptitle("Fourier shift and shared-basis core projection (one thread)")
    ring_radius_samples = ((J0_FIRST_ZERO / config.spot_radius_m)
                           * config.grid_size * config.pixel_pitch_m / (2 * np.pi))
    half_width = min(config.grid_size // 2,
                     max(8, int(np.ceil(1.75 * ring_radius_samples))))
    center = config.grid_size // 2
    for index, (ax, (data, title, cmap)) in enumerate(zip(axes.ravel(), panels)):
        if index in (1, 2, 3, 7):
            data = data[center - half_width:center + half_width,
                        center - half_width:center + half_width]
            title += " (FFT zoom)"
        elif data.ndim == 2 and data.shape[0] > 1:
            data = data[::stride, ::stride]
        ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                  interpolation="nearest")
        ax.set_title(title)
    fig.savefig(directory / "core_shift_mechanism.png", dpi=140)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_compatible(value):
    """Keep JSON standards-compliant when exact agreement yields infinite dB."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_json_compatible(data), ensure_ascii=False,
                               indent=2, allow_nan=False),
                    encoding="utf-8")


def run_stage_a(output_dir: str | Path, config: StageAConfig | None = None,
                *, save_images: bool = True) -> dict:
    """Run the complete Stage A numerical study and return the summary.

    A temporary on-disk array holds the 15 full, uncompressed complex128 FFTs
    (1.01 GB at N=2048). The entire Re/Im tensor is never duplicated in RAM.
    The temporary file is deleted after a successful run. On Windows, an
    interrupted run may leave a mapped temporary file if it remains open.
    """
    config = StageAConfig() if config is None else config
    design = design_modes(config)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Stage A output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Stage A output directory is not empty: {output_dir}. "
            "Choose a new --output-dir so old results cannot enter the new manifest."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_dir = output_dir / "modes"
    mode_dir.mkdir(exist_ok=True)
    offline_begin = time.perf_counter()
    radius = _radius_grid(config)
    timings: dict[str, object] = {
        "radius_grid_seconds": time.perf_counter() - offline_begin,
        "per_mode": [],
    }
    mode_rows: list[dict] = []
    overview_indices = set(np.linspace(0, design.count - 1,
                                       min(5, design.count), dtype=int).tolist())
    overview_samples: list[tuple[int, np.ndarray]] = []

    with tempfile.TemporaryDirectory(prefix="stage_a_fft_cache_",
                                     ignore_cleanup_errors=True) as scratch:
        cache_path = Path(scratch) / "full_uncompressed_spectra.npy"
        spectra = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.complex128,
                                            shape=(design.count, config.grid_size,
                                                   config.grid_size))
        for index, (m, kr) in enumerate(zip(design.m, design.kr_per_m)):
            tic = time.perf_counter()
            beam = j0(kr * radius)
            generation_seconds = time.perf_counter() - tic
            tic = time.perf_counter()
            spectra[index] = np.fft.fft2(beam, norm="ortho")
            fft_seconds = time.perf_counter() - tic
            timings["per_mode"].append({"m": int(m),
                                        "bessel_generation_seconds": generation_seconds,
                                        "fft_seconds": fft_seconds})
            del beam
            print(f"Stage A: cached uncompressed FFT {index + 1}/{design.count}, m={m:+d}",
                  flush=True)
        spectra.flush()
        first = spectra[0]
        symmetry_error = _relative_l2(first, first.T)
        del first
        if symmetry_error > 1e-10:
            raise RuntimeError("J0 FFT is not transpose-symmetric; shared Uy=Ux assumption failed")
        tic = time.perf_counter()
        uy = _randomized_shared_basis(spectra, config)
        ux = uy.copy()
        timings["shared_basis_fit_seconds"] = time.perf_counter() - tic
        cores = np.empty((design.count, config.rank, config.rank), dtype=np.complex128)
        tic = time.perf_counter()
        for index in range(design.count):
            cores[index] = _project_core(spectra[index], uy, ux)
        timings["core_projection_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        vy, vx = _spatial_factors(uy, ux)
        timings["spatial_factor_ifft_seconds"] = time.perf_counter() - tic
        timings["offline_dictionary_seconds"] = time.perf_counter() - offline_begin

        library_path = output_dir / "compressed_library.npz"
        np.savez_compressed(library_path,
                            factor_ky=uy, factor_kx=ux,
                            factor_y_spatial=vy, factor_x_spatial=vx,
                            core_complex=cores,
                            core_reim=np.stack((cores.real, cores.imag)),
                            m=design.m, kz_per_m=design.kz_per_m,
                            kr_per_m=design.kr_per_m,
                            config_json=json.dumps(asdict(config), sort_keys=True))
        if save_images:
            _save_factor_figures(output_dir, uy, ux, cores, design)

        for index, (m, kr) in enumerate(zip(design.m, design.kr_per_m)):
            full_fft = spectra[index]
            reconstructed_fft = _reconstruct_fft(cores[index], uy, ux)
            # The IFFT of the full spectrum is the exact matched numerical
            # reference for the Parseval check; direct J0 is compared too.
            spatial_reference = np.fft.ifft2(full_fft, norm="ortho")
            spatial_reconstructed = np.fft.ifft2(reconstructed_fft, norm="ortho")
            theoretical_beam = j0(kr * radius)
            if save_images and index in overview_indices:
                display_stride = max(1, config.grid_size // 512)
                overview_samples.append((index,
                                         theoretical_beam[::display_stride,
                                                          ::display_stride].copy()))
            fft_metrics = _field_metrics(full_fft, reconstructed_fft,
                                         config.phase_mask_fraction)
            spatial_metrics = _field_metrics(spatial_reference, spatial_reconstructed,
                                             config.phase_mask_fraction)
            row = {"mode_index": index, "m": int(m), "kz_per_m": float(design.kz_per_m[index]),
                   "kr_per_m": float(kr),
                   "first_zero_radius_um": (
                       float(J0_FIRST_ZERO / kr * 1e6) if kr > 0 else None),
                   "theory_vs_fft_ifft_relative_l2": _relative_l2(theoretical_beam,
                                                                    spatial_reference),
                   "fft_complex_relative_l2": fft_metrics["complex_relative_l2"],
                   "fft_complex_snr_db": fft_metrics["complex_snr_db"],
                   "fft_amplitude_relative_l2": fft_metrics["amplitude_relative_l2"],
                   "fft_amplitude_ssim": fft_metrics["amplitude_ssim"],
                   "fft_amplitude_psnr_db": fft_metrics["amplitude_psnr_db"],
                   "fft_amplitude_snr_db": fft_metrics["amplitude_snr_db"],
                   "fft_intensity_relative_l2": fft_metrics["intensity_relative_l2"],
                   "fft_intensity_ssim": fft_metrics["intensity_ssim"],
                   "fft_intensity_psnr_db": fft_metrics["intensity_psnr_db"],
                   "fft_intensity_snr_db": fft_metrics["intensity_snr_db"],
                   "fft_bright_phase_mae_rad": fft_metrics["bright_phase_mae_rad"],
                   "fft_bright_phase_rmse_rad": fft_metrics["bright_phase_rmse_rad"],
                   "fft_intensity_weighted_phase_mae_rad": fft_metrics["intensity_weighted_phase_mae_rad"],
                   "fft_phase_excluded_reference_energy_fraction": fft_metrics["phase_excluded_reference_energy_fraction"],
                   "fft_bright_reference_dark_candidate_fraction": fft_metrics["bright_reference_dark_candidate_fraction"],
                   "spatial_complex_relative_l2": spatial_metrics["complex_relative_l2"],
                   "spatial_complex_snr_db": spatial_metrics["complex_snr_db"],
                   "spatial_amplitude_relative_l2": spatial_metrics["amplitude_relative_l2"],
                   "spatial_amplitude_ssim": spatial_metrics["amplitude_ssim"],
                   "spatial_amplitude_psnr_db": spatial_metrics["amplitude_psnr_db"],
                   "spatial_amplitude_snr_db": spatial_metrics["amplitude_snr_db"],
                   "spatial_intensity_relative_l2": spatial_metrics["intensity_relative_l2"],
                   "spatial_intensity_ssim": spatial_metrics["intensity_ssim"],
                   "spatial_intensity_psnr_db": spatial_metrics["intensity_psnr_db"],
                   "spatial_intensity_snr_db": spatial_metrics["intensity_snr_db"],
                   "spatial_bright_phase_mae_rad": spatial_metrics["bright_phase_mae_rad"],
                   "spatial_bright_phase_rmse_rad": spatial_metrics["bright_phase_rmse_rad"],
                   "spatial_intensity_weighted_phase_mae_rad": spatial_metrics["intensity_weighted_phase_mae_rad"],
                   "spatial_phase_excluded_reference_energy_fraction": spatial_metrics["phase_excluded_reference_energy_fraction"],
                   "spatial_bright_reference_dark_candidate_fraction": spatial_metrics["bright_reference_dark_candidate_fraction"],
                   "parseval_relative_l2_absolute_difference": abs(
                       fft_metrics["complex_relative_l2"]
                       - spatial_metrics["complex_relative_l2"])}
            mode_rows.append(row)
            if save_images:
                target = mode_dir / f"mode_{index:02d}_m{int(m):+d}"
                target.mkdir(exist_ok=True)
                _save_complex_images(target, "theory_spatial", theoretical_beam,
                                     mask_fraction=config.phase_mask_fraction)
                _save_complex_images(target, "theory_fft", full_fft, fft_domain=True,
                                     mask_fraction=config.phase_mask_fraction)
                _save_complex_images(target, "reconstructed_fft", reconstructed_fft,
                                     reference=full_fft, fft_domain=True,
                                     mask_fraction=config.phase_mask_fraction)
                _save_complex_images(target, "reconstructed_spatial", spatial_reconstructed,
                                     reference=spatial_reference,
                                     mask_fraction=config.phase_mask_fraction)
                if int(m) == 0:
                    _save_representative_mode_comparison(
                        output_dir, int(m), float(kr), config.pixel_pitch_m,
                        spatial_reference, spatial_reconstructed,
                        full_fft, reconstructed_fft, config.phase_mask_fraction)
            print(f"Stage A: validated mode {index + 1}/{design.count}, m={m:+d}",
                  flush=True)
            del full_fft, reconstructed_fft, spatial_reference, spatial_reconstructed, theoretical_beam

        if save_images:
            _save_mode_overview(output_dir, overview_samples, design)

        _write_csv(output_dir / "mode_metrics.csv", mode_rows)
        _write_json(output_dir / "mode_metrics.json", {"rows": mode_rows})

        rng = np.random.default_rng(config.seed)
        selected = rng.choice(design.count, size=config.selected_mode_count,
                              replace=False)
        coefficients = (rng.standard_normal(config.selected_mode_count)
                        + 1j * rng.standard_normal(config.selected_mode_count))
        coefficients = coefficients / np.linalg.norm(coefficients)
        selection = {
            "seed": config.seed,
            "mode_indices": [int(i) for i in selected],
            "m": [int(design.m[i]) for i in selected],
            "coefficient_real": coefficients.real.tolist(),
            "coefficient_imag": coefficients.imag.tolist(),
        }
        _write_json(output_dir / "selected_modes_and_weights.json", selection)

        coax_timings = {}
        tic = time.perf_counter()
        coax_reference_field = np.zeros(radius.shape, dtype=np.complex128)
        for index, coefficient in zip(selected, coefficients):
            coax_reference_field += coefficient * j0(design.kr_per_m[index] * radius)
        coax_timings["direct_bessel_recomputation_and_sum_seconds"] = time.perf_counter() - tic

        tic = time.perf_counter()
        coax_reference_fft = np.zeros(radius.shape, dtype=np.complex128)
        for index, coefficient in zip(selected, coefficients):
            coax_reference_fft += coefficient * spectra[index]
        coax_timings["uncompressed_fft_weighted_sum_seconds"] = time.perf_counter() - tic

        tic = time.perf_counter()
        weighted_core = np.tensordot(coefficients, cores[selected], axes=(0, 0))
        coax_timings["core_weighted_sum_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        coax_proposed_field = _reconstruct_spatial(weighted_core, vy, vx)
        coax_timings["spatial_factor_reconstruction_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        _, coax_reference_codes = _quantize_ideal_phase_only(coax_reference_field)
        coax_timings["reference_cgh_generation_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        _, coax_proposed_codes = _quantize_ideal_phase_only(coax_proposed_field)
        coax_timings["proposed_cgh_generation_seconds"] = time.perf_counter() - tic
        coax_timings["direct_field_plus_cgh_seconds"] = (
            coax_timings["direct_bessel_recomputation_and_sum_seconds"]
            + coax_timings["reference_cgh_generation_seconds"])
        coax_timings["proposed_field_generation_seconds"] = sum(
            coax_timings[key] for key in ("core_weighted_sum_seconds",
                                           "spatial_factor_reconstruction_seconds"))
        coax_timings["proposed_field_plus_cgh_seconds"] = (
            coax_timings["proposed_field_generation_seconds"]
            + coax_timings["proposed_cgh_generation_seconds"])
        # Full spectra and legacy IFFTs are only used for diagnostics, not online.
        coax_proposed_fft = _reconstruct_fft(weighted_core, uy, ux)
        timings["coaxial_reconstruction_benchmark"] = _benchmark_reconstruction(
            weighted_core, uy, ux, vy, vx, config.reconstruction_timing_repeats)
        coax_metrics = {
            "spatial": _field_metrics(coax_reference_field, coax_proposed_field,
                                      config.phase_mask_fraction),
            "fft": _field_metrics(coax_reference_fft, coax_proposed_fft,
                                  config.phase_mask_fraction),
            "direct_field_vs_ifft_full_fft_relative_l2": _relative_l2(
                coax_reference_field,
                np.fft.ifft2(coax_reference_fft, norm="ortho")),
            "cgh": _cgh_metrics(coax_reference_codes, coax_proposed_codes,
                                coax_reference_field, config.phase_mask_fraction),
        }
        coax_directory = output_dir / "coaxial"
        if save_images:
            _save_case_images(coax_directory, "coaxial", coax_reference_field,
                              coax_proposed_field, coax_reference_fft,
                              coax_proposed_fft, coax_reference_codes,
                              coax_proposed_codes, config.phase_mask_fraction)
        timings["coaxial_online_seconds"] = coax_timings

        shifted_timings = {}
        thread_positions = ((np.arange(config.thread_count, dtype=np.float64)
                             - (config.thread_count - 1) / 2)
                            * config.thread_spacing_m)
        kx = 2 * np.pi * np.fft.fftfreq(config.grid_size, config.pixel_pitch_m)
        # Online direct baseline: regenerate every displaced Bessel beam.
        # Neither the uncompressed FFT cache nor the shared Tucker library is used.
        tic = time.perf_counter()
        axis = (np.arange(config.grid_size, dtype=np.float64)
                - (config.grid_size - 1) / 2) * config.pixel_pitch_m
        shifted_reference_field = np.zeros(radius.shape, dtype=np.complex128)
        for displacement_m in thread_positions:
            shifted_radius = np.hypot(axis[:, None], axis[None, :] - displacement_m)
            for index, coefficient in zip(selected, coefficients):
                shifted_reference_field += coefficient * j0(
                    design.kr_per_m[index] * shifted_radius)
            del shifted_radius
        shifted_timings["direct_bessel_recomputation_and_sum_seconds"] = (
            time.perf_counter() - tic)

        tic = time.perf_counter()
        shifted_weighted_core = np.tensordot(coefficients, cores[selected], axes=(0, 0))
        shifted_timings["core_weighted_sum_seconds"] = time.perf_counter() - tic
        shifted_core_total = np.zeros_like(shifted_weighted_core)
        operator_seconds = 0.0
        accumulation_seconds = 0.0
        example_operator = None
        for displacement_m in thread_positions:
            tic = time.perf_counter()
            phase_x = np.exp(-1j * kx * displacement_m)
            ax_operator = _x_shift_core_operator(ux, phase_x)
            operator_seconds += time.perf_counter() - tic
            tic = time.perf_counter()
            shifted_core_total += shifted_weighted_core @ ax_operator
            accumulation_seconds += time.perf_counter() - tic
            if example_operator is None:
                example_operator = ax_operator.copy()
        shifted_timings["shift_operator_construction_seconds"] = operator_seconds
        shifted_timings["shifted_core_accumulation_seconds"] = accumulation_seconds
        tic = time.perf_counter()
        shifted_proposed_field = _reconstruct_spatial(shifted_core_total, vy, vx)
        shifted_timings["spatial_factor_reconstruction_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        _, shifted_reference_codes = _quantize_ideal_phase_only(shifted_reference_field)
        shifted_timings["reference_cgh_generation_seconds"] = time.perf_counter() - tic
        tic = time.perf_counter()
        _, shifted_proposed_codes = _quantize_ideal_phase_only(shifted_proposed_field)
        shifted_timings["proposed_cgh_generation_seconds"] = time.perf_counter() - tic
        shifted_timings["direct_field_plus_cgh_seconds"] = sum(
            shifted_timings[key] for key in ("direct_bessel_recomputation_and_sum_seconds",
                                             "reference_cgh_generation_seconds"))
        shifted_timings["proposed_field_generation_seconds"] = sum(
            shifted_timings[key] for key in ("core_weighted_sum_seconds",
                                             "shift_operator_construction_seconds",
                                             "shifted_core_accumulation_seconds",
                                             "spatial_factor_reconstruction_seconds"))
        shifted_timings["proposed_field_plus_cgh_seconds"] = (
            shifted_timings["proposed_field_generation_seconds"]
            + shifted_timings["proposed_cgh_generation_seconds"])

        # Diagnostics are outside the online timers. The real-space direct field
        # and a periodic Fourier shift are different near the simulation boundary.
        shifted_proposed_fft = _reconstruct_fft(shifted_core_total, uy, ux)
        timings["shifted_reconstruction_benchmark"] = _benchmark_reconstruction(
            shifted_core_total, uy, ux, vy, vx, config.reconstruction_timing_repeats)
        shifted_reference_fft = np.fft.fft2(shifted_reference_field, norm="ortho")
        periodic_reference_fft = np.zeros_like(shifted_reference_fft)
        fresh_coax_fft = np.fft.fft2(coax_reference_field, norm="ortho")
        shifted_compressed_without_reprojection = np.zeros_like(shifted_reference_fft)
        for displacement_m in thread_positions:
            phase_x = np.exp(-1j * kx * displacement_m)[None, :]
            periodic_reference_fft += fresh_coax_fft * phase_x
            shifted_compressed_without_reprojection += coax_proposed_fft * phase_x
        periodic_reference_field = np.fft.ifft2(periodic_reference_fft, norm="ortho")
        shifted_metrics = {
            "spatial": _field_metrics(shifted_reference_field, shifted_proposed_field,
                                      config.phase_mask_fraction),
            "fft": _field_metrics(shifted_reference_fft, shifted_proposed_fft,
                                  config.phase_mask_fraction),
            "periodic_shift_model": {
                "direct_vs_periodic_spatial_complex_relative_l2": _relative_l2(
                    shifted_reference_field, periodic_reference_field),
                "periodic_vs_proposed_spatial_complex_relative_l2": _relative_l2(
                    periodic_reference_field, shifted_proposed_field),
            },
            "tucker_only_fft_complex_relative_l2": _relative_l2(
                periodic_reference_fft, shifted_compressed_without_reprojection),
            "extra_shift_reprojection_fft_complex_relative_l2": _relative_l2(
                shifted_compressed_without_reprojection, shifted_proposed_fft),
            "cgh": _cgh_metrics(shifted_reference_codes, shifted_proposed_codes,
                                shifted_reference_field, config.phase_mask_fraction),
        }
        shifted_directory = output_dir / "shifted"
        if save_images:
            _save_case_images(shifted_directory, "shifted", shifted_reference_field,
                              shifted_proposed_field, shifted_reference_fft,
                              shifted_proposed_fft, shifted_reference_codes,
                              shifted_proposed_codes, config.phase_mask_fraction)
            example_displacement = float(thread_positions[0])
            example_phase = np.exp(-1j * kx * example_displacement)
            example_exact = fresh_coax_fft * example_phase[None, :]
            example_projected = _reconstruct_fft(
                shifted_weighted_core @ example_operator, uy, ux)
            _save_shift_mechanism(shifted_directory, config, shifted_weighted_core,
                                  ux, fresh_coax_fft, example_exact,
                                  example_projected, example_displacement,
                                  example_operator)
        timings["shifted_online_seconds"] = shifted_timings

        summary = {
            "stage": "A_algorithmic_validation_only",
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "config": asdict(config),
            "simulation_window_width_m": config.grid_size * config.pixel_pitch_m,
            "mode_design": {
                "bessel_order": 0,
                "mode_count": design.count,
                "m": design.m.tolist(),
                "kz_per_m": design.kz_per_m.tolist(),
                "kr_per_m": design.kr_per_m.tolist(),
                "k0_per_m": design.k0_per_m,
                "q_per_m": design.q_per_m,
                "kr_center_per_m": design.kr_center_per_m,
                "first_zero_radius_center_m": config.spot_radius_m,
            },
            "selected_modes": selection,
            "thread_x_positions_m": thread_positions.tolist(),
            "shifted_direct_beam_evaluations": int(
                config.selected_mode_count * config.thread_count),
            "factor_rank": config.rank,
            "decomposition_method": (
                "Streaming randomized HOSVD on stacked real/imag FFT unfoldings; "
                "same real orthonormal factor on ky and kx because the square-grid J0 FFTs "
                "are transpose-symmetric; complex cores are exact projections into that basis"
            ),
            "fft_transpose_symmetry_relative_l2": symmetry_error,
            "mode_error_max": {
                "fft_complex_relative_l2": max(row["fft_complex_relative_l2"] for row in mode_rows),
                "spatial_complex_relative_l2": max(row["spatial_complex_relative_l2"] for row in mode_rows),
                "parseval_relative_l2_absolute_difference": max(
                    row["parseval_relative_l2_absolute_difference"] for row in mode_rows),
            },
            "mode_quality_worst": {
                "spatial_amplitude_ssim": min(row["spatial_amplitude_ssim"] for row in mode_rows),
                "spatial_amplitude_psnr_db": min(row["spatial_amplitude_psnr_db"] for row in mode_rows),
                "spatial_intensity_ssim": min(row["spatial_intensity_ssim"] for row in mode_rows),
                "spatial_intensity_psnr_db": min(row["spatial_intensity_psnr_db"] for row in mode_rows),
                "spatial_complex_snr_db": min(row["spatial_complex_snr_db"] for row in mode_rows),
            },
            "coaxial": coax_metrics,
            "shifted": shifted_metrics,
            "timings_seconds": timings,
            "reference_definition": "Regenerate and sum every spatially displaced J0 beam for the direct online baseline; FFT of the result is diagnostic only",
            "cgh_definition": "Ideal numerical angle(E) with cyclic 8-bit quantization; no amplitude retrieval or optical replay",
            "quality_metric_definitions": {
                "ssim": "Full-grid grayscale SSIM on amplitude or intensity, shared reference-peak scale, 7x7 window",
                "psnr_db": "20 log10(reference peak / image RMSE), amplitude or intensity; no independent normalization",
                "snr_db": "10 log10(sum |reference|^2 / sum |candidate-reference|^2); complex SNR uses complex fields",
                "infinite_db_json": "Infinity string means exactly zero error; unavailable metrics are null",
                "cgh": "Raw 8-bit SSIM/PSNR are wrap-sensitive; circular code PSNR uses a 128-level peak; replay is not modeled",
            },
            "boundary_definition": "Direct J0 displacement is evaluated in real space; proposed FFT shift is periodic on the numerical simulation window. Their model mismatch is reported separately",
            "simulation_window_note": (
                "Sampling only covers a finite square, so the mathematical infinite Bessel field is numerically truncated; "
                "the square is not treated as a measured optical aperture."
            ),
            "offline_timing_scope": "Radius grid, Bessel generation, FFT cache, randomized shared basis, core projection, factor-column IFFTs; excludes library and image saving",
            "online_timing_scope": (
                "Single perf_counter observations; direct shifted timing includes fresh displaced "
                "radius grids and all beam evaluations; proposed factors and cores are already in memory. "
                "Excludes dictionary fit, factor-column IFFTs, reconstruction benchmarks, "
                "diagnostic FFTs/metrics and file/figure saving."
            ),
            "timing_baselines": {
                "coaxial_direct": "Re-evaluate selected J0 modes using the precomputed radius grid and sum fields",
                "shifted_direct": "Regenerate selected spatial J0 beams at each thread position, sum all fields, encode CGH; no cached FFT or IFFT",
                "proposed": "Sum stored cores, project each Fourier shift into common basis, reconstruct spatial field with pre-IFFT factors Vy C Vx.T, encode CGH; no online FFT/IFFT",
            },
            "full_fft_cache_bytes": int(design.count * config.grid_size**2 * np.dtype(np.complex128).itemsize),
            "compressed_library_bytes": library_path.stat().st_size,
            "storage_ratio_full_fft_to_library": float(
                design.count * config.grid_size**2 * np.dtype(np.complex128).itemsize
                / library_path.stat().st_size),
            "physical_scope": {
                "SLM_and_4f": "not_run",
                "illumination_and_physical_aperture": "not_run",
                "propagation_to_50_cm": "not_run",
                "experimental_replay": "not_run",
            },
        }
        _write_json(output_dir / "timings.json", {"timings_seconds": timings})
        _write_json(output_dir / "summary.json", summary)
        # Avoid deleting a mapped file on Windows while a view is alive.
        del spectra, radius, coax_reference_fft, coax_proposed_fft
        del shifted_reference_fft, shifted_proposed_fft
        gc.collect()

    manifest = {
        "output_dir": str(output_dir),
        "mode_count": design.count,
        "rank": config.rank,
        "files": sorted({"manifest.json"} | {
            str(path.relative_to(output_dir)).replace("\\", "/")
            for path in output_dir.rglob("*") if path.is_file()
        }),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(f"Stage A complete: {output_dir / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="stage_a_outputs")
    parser.add_argument("--grid-size", type=int, default=2048,
                        help="Development override; production Stage A uses 2048")
    parser.add_argument("--rank", type=int, default=64,
                        help="Development override; production Stage A uses 64")
    parser.add_argument("--thread-count", type=int, default=10,
                        help="Number of equally spaced x-shifted threads; default 10")
    parser.add_argument("--reconstruction-repeats", type=int, default=3,
                        help="Diagnostic reconstruction repeats per method after warmup")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip PNG generation for a quick numerical diagnostic")
    args = parser.parse_args()
    config = replace(StageAConfig(), grid_size=args.grid_size, rank=args.rank,
                     thread_count=args.thread_count,
                     reconstruction_timing_repeats=args.reconstruction_repeats)
    run_stage_a(args.output_dir, config, save_images=not args.no_images)


if __name__ == "__main__":
    main()
