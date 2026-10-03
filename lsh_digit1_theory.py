"""Theory-only single light sheet: a digit 1 in the longitudinal x-z plane.

No compressed library is loaded. All Bessel modes are generated analytically.
The camera model measures |E(x,y,z)|^2; stacking its y=0 rows gives the sheet.
The phase CGH uses Arrizon amplitude encoding and an ideal first-order filter,
not the angle-only masks used by Stage A. All pitches refer to a virtual 1:1
4f system, not to a calibrated physical SLM. See LSH数字1_理論検証.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PolygonPath
from PIL import Image
from scipy import fft
from scipy.special import j0, j1

from stage_a_parameters import ModeDesign, StageAConfig, design_modes
from stage_a_validation import _image_metrics, _relative_l2

THEORY_SCHEMA_VERSION = 1
SOURCES = {
    "LSH": "https://doi.org/10.1038/s41566-023-01188-y",
    "LSH_supplement": "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41566-023-01188-y/MediaObjects/41566_2023_1188_MOESM1_ESM.pdf",
    "frozen_waves": "https://arxiv.org/abs/physics/0407128",
    "Arrizon": "https://doi.org/10.1364/JOSAA.24.003500",
    "angular_spectrum_sampling": "https://doi.org/10.1364/OE.17.019662",
}


@dataclass(frozen=True)
class Digit1Config:
    # 数値座標は理想4f出力面。z=0以降を自由空間で伝搬させる。
    wavelength_m: float = 532e-9
    grid_size: int = 2048
    pixel_pitch_m: float = 3.74e-6
    spot_radius_m: float = 50e-6  # 中心モードJ0の第1零点半径
    design_length_m: float = 0.50
    thread_count: int = 21
    thread_spacing_m: float = 75e-6
    digit_width_m: float = 1.2e-3
    digit_z_start_m: float = 0.075
    digit_z_end_m: float = 0.425
    evaluation_x_half_width_m: float = 0.9e-3
    coefficient_samples: int = 2048
    z_samples: int = 101
    aperture_radius_m: float = 3.5e-3  # 有限開口。位相0で代用しない。
    illumination_waist_m: float | None = None  # Noneは開口内の一様入射光
    padding_factor: int = 2
    angular_bandlimit: bool = True
    include_cgh_replay: bool = True
    carrier_period_pixels: int = 8
    encoding_coupling: float = 0.58
    filter_radius_factor: float = 1.2
    sensor_z_m: tuple[float, ...] = (0.0, 0.10, 0.25, 0.40, 0.50)
    check_padding: bool = False
    padding_check_factor: int = 3

    def __post_init__(self) -> None:
        if self.carrier_period_pixels < 3 or self.padding_factor < 1:
            raise ValueError("carrier period must be >=3 and padding_factor >=1")
        if (self.grid_size < 16 or self.grid_size % 2
                or self.grid_size % self.carrier_period_pixels):
            raise ValueError("grid_size must be >=16 and divisible by carrier_period_pixels")
        if self.thread_count < 1 or self.z_samples < 3 or self.coefficient_samples < 16:
            raise ValueError("thread_count, z_samples and coefficient_samples are too small")
        if any(v <= 0 for v in (self.wavelength_m, self.pixel_pitch_m,
                               self.spot_radius_m, self.design_length_m,
                               self.thread_spacing_m, self.digit_width_m,
                               self.evaluation_x_half_width_m, self.aperture_radius_m)):
            raise ValueError("lengths must be positive")
        if not 0 < self.digit_z_start_m < self.digit_z_end_m < self.design_length_m:
            raise ValueError("digit depth must lie strictly inside (0, design_length_m)")
        half_width = self.grid_size * self.pixel_pitch_m / 2
        if self.aperture_radius_m >= half_width:
            raise ValueError("aperture_radius_m must fit inside the numerical window")
        if self.evaluation_x_half_width_m >= half_width:
            raise ValueError("evaluation ROI must fit inside the numerical window")
        if self.digit_width_m / 2 > self.evaluation_x_half_width_m:
            raise ValueError("digit must fit inside the evaluation ROI")
        if ((self.thread_count - 1) * self.thread_spacing_m / 2
                >= self.aperture_radius_m):
            raise ValueError("thread centres must fit inside the aperture")
        if self.illumination_waist_m is not None and self.illumination_waist_m <= 0:
            raise ValueError("illumination_waist_m must be positive or None")
        if not 0 < self.encoding_coupling <= 0.58 or self.filter_radius_factor <= 0:
            raise ValueError("encoding coupling must be in (0,.58] and filter factor positive")
        if not self.sensor_z_m or any(z < 0 or z > self.design_length_m for z in self.sensor_z_m):
            raise ValueError("sensor depths must lie inside [0, design_length_m]")
        if self.check_padding and self.padding_check_factor <= self.padding_factor:
            raise ValueError("padding_check_factor must exceed padding_factor")


def mode_design(config: Digit1Config) -> ModeDesign:
    """Same dispersion relation as Stage A, with ALL propagating modes retained."""
    return design_modes(StageAConfig(
        wavelength_m=config.wavelength_m, grid_size=config.grid_size,
        pixel_pitch_m=config.pixel_pitch_m, spot_radius_m=config.spot_radius_m,
        design_length_m=config.design_length_m, rank=1, selected_mode_count=1))


def digit_target(x_m: np.ndarray, z_m: np.ndarray, config: Digit1Config) -> np.ndarray:
    """[z,x] binary intensity target, drawn as a stem, upper flag and lower foot."""
    # 複製画像・フォント依存を避け、数字1の輪郭を正規化座標で定義する。
    vertices = np.array([
        [.10, .00], [.90, .00], [.90, .12], [.62, .12], [.62, 1.00],
        [.43, 1.00], [.10, .73], [.24, .60], [.42, .76], [.42, .12],
        [.10, .12], [.10, .00],
    ])
    u, v = np.meshgrid(np.asarray(x_m) / config.digit_width_m + 0.5,
                       (np.asarray(z_m) - config.digit_z_start_m)
                       / (config.digit_z_end_m - config.digit_z_start_m))
    return PolygonPath(vertices, closed=True).contains_points(
        np.column_stack((u.ravel(), v.ravel()))).reshape(u.shape).astype(np.float64)


def fourier_coefficients(amplitude: np.ndarray, z_midpoints: np.ndarray,
                         modes: np.ndarray, length_m: float) -> np.ndarray:
    """A[m,q] = (1/L) integral F_q(z) exp(-2 pi i m z/L) dz.

    F_q is AMPLITUDE: use sqrt(target intensity), not intensity itself.
    The uniform midpoint quadrature has weight 1/number_of_samples.
    """
    expected = (np.arange(len(z_midpoints)) + 0.5) * length_m / len(z_midpoints)
    if amplitude.ndim != 2 or amplitude.shape[0] != len(z_midpoints):
        raise ValueError("amplitude must have shape [z_samples, threads]")
    if not np.allclose(z_midpoints, expected, rtol=1e-12, atol=1e-14):
        raise ValueError("coefficient integration requires uniform midpoint samples")
    return (np.exp(-2j * np.pi * modes[:, None] * z_midpoints[None, :] / length_m)
            @ amplitude) / len(z_midpoints)


def initial_bessel_field(axis: np.ndarray, positions: np.ndarray,
                         kr: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """各Threadで解析J0を直接生成・加算。FFTシフトや圧縮データを使わない。"""
    field = np.zeros((len(axis), len(axis)), dtype=np.complex128)
    for q, position in enumerate(positions):
        if not np.any(coefficients[:, q]):
            continue
        radius = np.hypot(axis[:, None], axis[None, :] - position)
        for radial_k, coefficient in zip(kr, coefficients[:, q]):
            field += coefficient * j0(radial_k * radius)
        print(f"Initial field: thread {q + 1}/{len(positions)}", flush=True)
    return field


def analytic_sheet(x: np.ndarray, z: np.ndarray, positions: np.ndarray,
                    design: ModeDesign, coefficients: np.ndarray) -> np.ndarray:
    """Infinite-aperture coherent Bessel sum at y=0, removing exp(i k0 z)."""
    # 安定なkz-k0。共通キャリアを取り除いても強度は変化しない。
    delta_kz = -design.kr_per_m**2 / (design.kz_per_m + design.k0_per_m)
    phase = np.exp(1j * z[:, None] * delta_kz[None, :])
    field = np.zeros((len(z), len(x)), dtype=np.complex128)
    for q, position in enumerate(positions):
        radial = j0(design.kr_per_m[:, None] * np.abs(x[None, :] - position))
        field += (phase * coefficients[:, q][None, :]) @ radial
    return field


class AngularSpectrum:
    """Padded scalar angular spectrum with explicit transfer-sampling guards.

    Coordinates have an actual y=0 pixel at N//2. Spectra use ifftshift(field)
    so y=0 is index 0 after inverse FFT. The exact central sensor row can be
    obtained by summing ky first and doing only a 1D IFFT along x.
    """
    def __init__(self, n: int, pitch: float, wavelength: float, padding: int,
                 bandlimit: bool = True):
        self.n, self.p = n, n * padding
        self.pad = (self.p - n) // 2
        self.bandlimit = bandlimit
        self.width = self.p * pitch
        self.k_axis = 2 * np.pi * fft.fftfreq(self.p, d=pitch)
        k0 = 2 * np.pi / wavelength
        transverse2 = self.k_axis[:, None]**2 + self.k_axis[None, :]**2
        self.kz = np.sqrt(np.maximum(k0**2 - transverse2, 0))
        self.delta_kz = -transverse2 / (self.kz + k0)

    def spectrum(self, field: np.ndarray) -> np.ndarray:
        if field.shape != (self.n, self.n):
            raise ValueError("initial field shape does not match the propagation grid")
        padded = np.pad(field.astype(np.complex64), self.pad)
        return fft.fft2(fft.ifftshift(padded), norm="ortho", workers=1)

    def transfer(self, z: float) -> tuple[np.ndarray, np.ndarray]:
        if z < 0:
            raise ValueError("only forward propagation is supported")
        phase = self.delta_kz * z
        h = np.empty(phase.shape, dtype=np.complex64)
        np.cos(phase, out=h.real)
        np.sin(phase, out=h.imag)
        valid = self.kz > 0  # evanescent components are excluded
        if self.bandlimit and z > 0:
            # Nyquist guard on the sampled transfer phase along both axes:
            # |d(kz z)/dkx| Delta_kx <= pi, and likewise for ky.
            required_kz = 2 * z * np.abs(self.k_axis) / self.width
            valid &= self.kz >= required_kz[None, :]
            valid &= self.kz >= required_kz[:, None]
        h[~valid] = 0
        return h, valid

    def row(self, spectrum: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        reduced = np.einsum("ij,ij->j", spectrum, transfer,
                            dtype=np.complex128, optimize=False) / np.sqrt(self.p)
        row = fft.fftshift(fft.ifft(reduced, norm="ortho", workers=1))
        return row[self.pad:self.pad + self.n]

    def frame(self, spectrum: np.ndarray, transfer: np.ndarray) -> np.ndarray:
        field = fft.fftshift(fft.ifft2(spectrum * transfer, norm="ortho", workers=1))
        return field[self.pad:self.pad + self.n, self.pad:self.pad + self.n]


def encode_phase_mask(field: np.ndarray, config: Digit1Config) -> np.ndarray:
    """振幅を符号化した8-bit位相マスクだけを生成する。4f/伝搬は計算しない。"""
    if field.shape != (config.grid_size, config.grid_size):
        raise ValueError("field shape differs from config.grid_size")
    yy, xx = np.indices(field.shape)
    carrier = 2 * np.pi * (xx + yy) / config.carrier_period_pixels
    a = np.abs(field)
    modulation_axis = np.linspace(0, 1.84118, 65536)
    modulation = np.interp(config.encoding_coupling * np.clip(a, 0, 1),
                           j1(modulation_axis), modulation_axis)
    phase = np.mod(modulation * np.sin(np.angle(field) + carrier), 2 * np.pi)
    return (np.rint(phase * 256 / (2 * np.pi)).astype(np.uint16) % 256).astype(np.uint8)


def encode_and_replay(field: np.ndarray, illumination: np.ndarray,
                      config: Digit1Config, kr_max: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Arrizon sinusoidal phase encoding, cyclic 8-bit LUT, ideal +1 order.

    field has peak <=1 BEFORE illumination. J1(f(a))=coupling*a.
    Returned field divides out ONLY the known constant coupling; no fitted
    spatial correction is applied. Raw sensor intensity is coupling^2 times
    the intensity propagated from this gain-corrected field.
    """
    n = config.grid_size
    yy, xx = np.indices(field.shape)
    carrier = 2 * np.pi * (xx + yy) / config.carrier_period_pixels
    a = np.abs(field)
    codes = encode_phase_mask(field, config)
    transmittance = illumination * np.exp(2j * np.pi * codes.astype(np.float64) / 256)
    k_axis = 2 * np.pi * fft.fftfreq(n, d=config.pixel_pitch_m)
    centre = 2 * np.pi / (config.carrier_period_pixels * config.pixel_pitch_m)
    radius = config.filter_radius_factor * kr_max
    if radius >= min(centre, np.pi / config.pixel_pitch_m - centre):
        raise ValueError("carrier/filter orders overlap the origin or FFT sampling boundary")
    pupil = ((k_axis[:, None] - centre)**2 + (k_axis[None, :] - centre)**2 <= radius**2)
    spectrum = fft.fft2(transmittance, norm="ortho", workers=1)
    filtered = fft.ifft2(spectrum * pupil, norm="ortho", workers=1)
    replay = filtered * np.exp(-1j * carrier) / config.encoding_coupling
    desired = illumination * field
    raw_power = float(np.sum(np.abs(filtered)**2))
    incident_power = float(np.sum(illumination**2))
    return codes, replay, {
        "definition": f"Theta=f(a) sin(phi+carrier); J1(f)={config.encoding_coupling}*a; 8-bit; ideal +1 order filter",
        "amplitude_coupling": config.encoding_coupling,
        "sensor_intensity_gain_correction": 1 / config.encoding_coupling**2,
        "first_order_power_fraction_of_incident": raw_power / incident_power,
        "initial_plane_complex_relative_l2": _relative_l2(desired, replay),
        "filter_radius_rad_per_m": float(radius),
        "amplitude_saturation_fraction": float(np.mean(a > 1 + 1e-12)),
        "physical_4f_magnification": "virtual_1_to_1; physical SLM mapping not calibrated",
    }


def propagate(fields: dict[str, np.ndarray], z: np.ndarray, config: Digit1Config,
              snapshot_indices: set[int]) -> tuple[dict, dict, dict]:
    plan = AngularSpectrum(config.grid_size, config.pixel_pitch_m, config.wavelength_m,
                           config.padding_factor, config.angular_bandlimit)
    spectra = {name: plan.spectrum(field) for name, field in fields.items()}
    rows = {name: np.empty((len(z), config.grid_size), dtype=np.complex128) for name in fields}
    frames = {name: {} for name in fields}
    sampling = {name: [] for name in fields}
    for index, depth in enumerate(z):
        transfer, valid = plan.transfer(float(depth))
        for name, spectrum in spectra.items():
            rows[name][index] = plan.row(spectrum, transfer)
            if index in snapshot_indices:
                frame = plan.frame(spectrum, transfer)
                frames[name][index] = (np.abs(frame)**2).astype(np.float32)
                power = np.abs(spectrum)**2
                lost = float(np.sum(power, where=~valid, dtype=np.float64)
                             / np.sum(power, dtype=np.float64))
                sampling[name].append({"z_m": float(depth),
                                       "discarded_spectral_power_fraction": lost})
        if index % max(1, len(z) // 20) == 0 or index == len(z) - 1:
            print(f"Sensor propagation: {index + 1}/{len(z)}, z={depth * 1000:.1f} mm", flush=True)
    return rows, frames, sampling


def _padding_check(fields: dict, rows: dict, z: np.ndarray, roi: np.ndarray,
                   config: Digit1Config, indices: set[int]) -> dict:
    if not config.check_padding:
        return {"status": "not_checked", "padding_factor": config.padding_factor,
                "how_to_check": "Set check_padding=True; compares central rows at saved sensor depths"}
    larger = replace(config, padding_factor=config.padding_check_factor, check_padding=False)
    selected = sorted(indices)
    extended, _, _ = propagate(fields, z[selected], larger, set())
    errors = {name: [{"z_m": float(z[index]), "row_complex_relative_l2":
                     _relative_l2(extended[name][j, roi], rows[name][index, roi])}
                    for j, index in enumerate(selected)] for name in fields}
    return {"status": "measured_at_sensor_depths_only", "baseline_padding": config.padding_factor,
            "larger_padding": config.padding_check_factor, "errors": errors,
            "note": "Not a convergence proof at all depths; inspect the recorded differences"}


def _write_json(path: Path, value: dict) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def _save_images(out: Path, x: np.ndarray, z: np.ndarray, roi: np.ndarray,
                 sheets: dict, frames: dict, codes: np.ndarray | None,
                 field0: np.ndarray, positions: np.ndarray,
                 target_threads: np.ndarray, isolated: np.ndarray) -> None:
    extent = [x[roi][0] * 1000, x[roi][-1] * 1000, z[0] * 1000, z[-1] * 1000]
    shared_peak = max(float(sheets["target"].max()), float(sheets["analytic_infinite"].max()))
    titles = {
        "target": "Digit 1 target",
        "analytic_infinite": "Analytic Bessel sum (infinite aperture)",
        "finite_complex": "Finite complex field: angular spectrum",
        "cgh_replay": "8-bit amplitude-aware CGH + ideal 4f (gain corrected)",
    }
    fig, axes = plt.subplots(1, len(sheets), figsize=(5 * len(sheets), 6), constrained_layout=True)
    for ax, (name, intensity) in zip(np.atleast_1d(axes), sheets.items()):
        img = ax.imshow(intensity[:, roi], origin="lower", extent=extent, aspect="auto",
                        cmap="gray", vmin=0, vmax=shared_peak)
        ax.set(title=titles[name], xlabel="x [mm]", ylabel="z [mm]")
        fig.colorbar(img, ax=ax, label="Intensity, common scale")
        single, single_ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
        single_ax.imshow(intensity[:, roi], origin="lower", extent=extent, aspect="auto",
                         cmap="gray", vmin=0, vmax=shared_peak)
        single_ax.set(title=titles[name], xlabel="x [mm]", ylabel="z [mm]")
        single.savefig(out / f"sheet_{name}_xz.png", dpi=160)
        plt.close(single)
    fig.savefig(out / "digit1_sheet_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    thread_extent = [positions[0] * 1000, positions[-1] * 1000, z[0] * 1000, z[-1] * 1000]
    for ax, intensity, title in zip(axes, (target_threads, isolated),
                                    ("Target on each thread", "Truncated FW series, isolated threads")):
        ax.imshow(intensity, origin="lower", extent=thread_extent, aspect="auto",
                  cmap="gray", vmin=0, vmax=max(float(target_threads.max()), float(isolated.max())))
        ax.set(title=title, xlabel="Thread x [mm]", ylabel="z [mm]")
    fig.savefig(out / "thread_profile_design.png", dpi=160)
    plt.close(fig)
    if codes is not None:
        Image.fromarray(codes).save(out / "digit1_cgh_8bit.png")
    plt.imsave(out / "initial_complex_amplitude.png", np.abs(field0), cmap="gray", vmin=0, vmax=1)
    plt.imsave(out / "initial_complex_phase.png", np.angle(field0), cmap="twilight",
               vmin=-np.pi, vmax=np.pi)
    sensor = out / "sensor_xy"
    sensor.mkdir()
    frame_peak = max(float(image.max()) for case in frames.values() for image in case.values())
    xy_extent = [x[0] * 1000, x[-1] * 1000, x[0] * 1000, x[-1] * 1000]
    for case, case_frames in frames.items():
        for index, intensity in case_frames.items():
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            img = ax.imshow(intensity, origin="lower", extent=xy_extent, cmap="gray",
                            vmin=0, vmax=frame_peak)
            ax.axhline(0, color="red", linewidth=0.5)
            ax.set(title=f"{case}: sensor x-y at z={z[index] * 1000:.1f} mm",
                   xlabel="x [mm]", ylabel="y [mm]")
            fig.colorbar(img, ax=ax, label="Intensity (CGH: known gain corrected)")
            fig.savefig(sensor / f"{case}_z{z[index] * 1000:06.1f}mm.png", dpi=160)
            plt.close(fig)


def run_digit1(output_dir: str | Path, config: Digit1Config | None = None,
               save_images: bool = True) -> dict:
    """Run only when explicitly invoked by the user, not on import."""
    config = config or Digit1Config()
    design = mode_design(config)
    if config.coefficient_samples < design.count:
        raise ValueError("coefficient_samples must be at least the Bessel mode count")
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}; choose a new output_dir")
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    x = (np.arange(config.grid_size) - config.grid_size // 2) * config.pixel_pitch_m
    z = np.linspace(0, config.design_length_m, config.z_samples)
    positions = (np.arange(config.thread_count) - (config.thread_count - 1) / 2) * config.thread_spacing_m
    z_mid = (np.arange(config.coefficient_samples) + 0.5) * config.design_length_m / config.coefficient_samples
    target_on_threads = digit_target(positions, z_mid, config)
    coefficients = fourier_coefficients(np.sqrt(target_on_threads), z_mid,
                                        design.m, config.design_length_m)
    if not np.any(coefficients):
        raise ValueError("No thread intersects the digit; increase thread_count or reduce spacing")
    field = initial_bessel_field(x, positions, design.kr_per_m, coefficients)
    radius = np.hypot(x[:, None], x[None, :])
    illumination = (radius <= config.aperture_radius_m).astype(np.float64)
    if config.illumination_waist_m is not None:
        illumination *= np.exp(-(radius / config.illumination_waist_m)**2)
    scale = float(np.max(np.abs(field)[illumination > 0]))
    # Saturation outside the illuminated pupil is irrelevant; set it to zero.
    field /= scale
    analytic = analytic_sheet(x, z, positions, design, coefficients) / scale
    encoding_field = np.where(illumination > 0, field, 0)
    finite = (encoding_field * illumination).astype(np.complex64)
    fields = {"finite_complex": finite}
    codes, encoding = None, {"status": "not_run"}
    if config.include_cgh_replay:
        codes, replay, encoding = encode_and_replay(
            encoding_field, illumination, config, float(design.kr_per_m.max()))
        fields["cgh_replay"] = replay.astype(np.complex64)
    snapshot_indices = {int(np.argmin(np.abs(z - depth))) for depth in config.sensor_z_m}
    tic = time.perf_counter()
    rows, frames, sampling = propagate(fields, z, config, snapshot_indices)
    propagation_seconds = time.perf_counter() - tic
    roi = np.abs(x) <= config.evaluation_x_half_width_m
    target = digit_target(x, z, config) / scale**2
    isolated = np.abs(np.exp(2j * np.pi * z[:, None] * design.m[None, :]
                            / config.design_length_m) @ coefficients)**2 / scale**2
    target_threads = digit_target(positions, z, config) / scale**2
    sheets = {"target": target, "analytic_infinite": np.abs(analytic)**2}
    sheets.update({name: np.abs(row)**2 for name, row in rows.items()})
    metrics = {
        "analytic_vs_initial_y0_complex_relative_l2": _relative_l2(
            analytic[0, roi], field[config.grid_size // 2, roi]),
        "isolated_thread_profiles_vs_target": _image_metrics(target_threads, isolated),
        "analytic_coherent_vs_target": _image_metrics(target[:, roi], sheets["analytic_infinite"][:, roi]),
        "finite_aperture_vs_target": _image_metrics(target[:, roi], sheets["finite_complex"][:, roi]),
        "finite_vs_infinite": {
            "complex_relative_l2": _relative_l2(analytic[:, roi], rows["finite_complex"][:, roi]),
            **_image_metrics(sheets["analytic_infinite"][:, roi], sheets["finite_complex"][:, roi]),
        },
    }
    if config.include_cgh_replay:
        metrics["cgh_replay_vs_target"] = _image_metrics(target[:, roi], sheets["cgh_replay"][:, roi])
        metrics["cgh_replay_vs_finite_complex"] = {
            "complex_relative_l2": _relative_l2(rows["finite_complex"][:, roi], rows["cgh_replay"][:, roi]),
            **_image_metrics(sheets["finite_complex"][:, roi], sheets["cgh_replay"][:, roi]),
        }
    padding = _padding_check(fields, rows, z, roi, config, snapshot_indices)
    np.savez_compressed(out / "digit1_design.npz", x_m=x, z_m=z, thread_x_m=positions,
                        m=design.m, kr_per_m=design.kr_per_m, kz_per_m=design.kz_per_m,
                        coefficients=coefficients, target_thread_intensity=target_threads,
                        isolated_thread_intensity=isolated)
    np.savez_compressed(out / "digit1_initial_fields.npz", **fields,
                        illumination=illumination.astype(np.float32))
    np.savez_compressed(out / "digit1_sheet_data.npz", x_m=x, z_m=z, roi=roi,
                        **{name + "_intensity": val for name, val in sheets.items()},
                        **{name + "_complex": val for name, val in rows.items()},
                        **({"cgh_sensor_raw_intensity": sheets["cgh_replay"]
                            * config.encoding_coupling**2} if config.include_cgh_replay else {}))
    np.savez_compressed(out / "sensor_frames.npz", x_m=x, z_m=z,
                        **{f"{case}_index{index:03d}": image
                           for case, case_frames in frames.items()
                           for index, image in case_frames.items()},
                        **({f"cgh_raw_index{index:03d}": image * config.encoding_coupling**2
                            for index, image in frames["cgh_replay"].items()}
                           if config.include_cgh_replay else {}))
    if save_images:
        _save_images(out, x, z, roi, sheets, frames, codes, finite,
                     positions, target_threads, isolated)
    summary = {
        "study": "theory_only_digit1_single_xz_sheet", "schema_version": THEORY_SCHEMA_VERSION,
        "config": asdict(config), "mode_count_per_thread": design.count,
        "active_thread_count": int(np.count_nonzero(np.any(np.abs(coefficients) > 0, axis=0))),
        "m": design.m.tolist(), "kr_per_m": design.kr_per_m.tolist(),
        "thread_x_m": positions.tolist(), "metrics": metrics, "encoding": encoding,
        "angular_sampling_at_sensor_depths": sampling, "padding_validation": padding,
        "propagation_seconds": propagation_seconds, "total_seconds": time.perf_counter() - begin,
        "notes": {
            "sheet": "Single y=0 x-z sheet; each camera frame is x-y, not the full digit",
            "analytic": "Infinite-aperture coherent sum; includes interference between threads",
            "finite": "Scalar angular spectrum of a freshly generated finite-aperture field",
            "camera": "Ideal intensity readout; no shot noise, pixel integration, LUT calibration or aberrations",
            "coordinates": "z=0 is the virtual 1:1 4f output plane; pitch is numerical/output-plane sampling",
            "coordinate_origin": "An actual y=0 pixel at N//2; differs by half a pixel from Stage A; match dictionary coordinates before a later compressed comparison",
            "SLM": f"CGH is a virtual {config.grid_size}-square phase mask; physical 1920x1200 / 8um / 1:4 mapping is not applied",
            "gain": "CGH replay comparisons divide the known constant encoding coupling only; raw intensities are also saved",
            "normalization": "All fields share the same initial reference scale; no independent image peak normalization",
            "padding": "Transfer phase sampling is guarded; finite padding convergence requires its own check",
            "compression": "No Tucker compression, saved cores, or FFT-shift operator used",
        }, "sources": SOURCES,
    }
    _write_json(out / "summary.json", summary)
    _write_json(out / "manifest.json", {"files": sorted(
        {"manifest.json"} | {str(p.relative_to(out)).replace("\\", "/")
                              for p in out.rglob("*") if p.is_file()})})
    print(f"Saved theory-only digit 1 to: {out.resolve()}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="lsh_digit1_theory_outputs_v1")
    parser.add_argument("--z-samples", type=int, default=101)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--check-padding", action="store_true")
    parser.add_argument("--no-cgh", action="store_true", help="Only analytic and finite complex-field propagation")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    config = Digit1Config(z_samples=args.z_samples, padding_factor=args.padding,
                          check_padding=args.check_padding, include_cgh_replay=not args.no_cgh)
    run_digit1(args.output_dir, config, save_images=not args.no_images)


if __name__ == "__main__":
    main()
