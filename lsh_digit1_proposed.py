"""数字1: 直接生成と共有FFTコアの加算・シフトを同じ光学条件で比較する。

数値計算は run_comparison() またはCLIを明示的に呼んだ場合だけ行う。
辞書は理論版と同じ整数画素中心で作る。旧Stage A辞書は読み込まない。
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import j0

import lsh_digit1_theory as theory
import matplotlib.pyplot as plt
from stage_a_parameters import ModeDesign, StageAConfig
from stage_a_validation import (
    _cgh_metrics, _field_metrics, _image_metrics, _project_core,
    _randomized_shared_basis, _reconstruct_spatial, _relative_l2,
    _save_factor_figures, _spatial_factors, _x_shift_core_operator,
)

SCHEMA_VERSION = 1


def config_from_theory_results(directory: str | Path) -> theory.Digit1Config:
    """保存済み理論結果の設定を読むだけ。画像や計算結果は再利用しない。"""
    data = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
    if data.get("study") != "theory_only_digit1_single_xz_sheet":
        raise ValueError("Not a digit1 theory result directory")
    values = dict(data["config"])
    values["sensor_z_m"] = tuple(values["sensor_z_m"])
    return theory.Digit1Config(**values)


def mode_design_for_spacing(config: theory.Digit1Config,
                            mode_spacing_factor: int = 1) -> ModeDesign:
    """kz間隔だけを1/factorにし、元のkr範囲・中心半径を保つ。

    kz=Q+2pi*m/(factor*L)。mの範囲もfactor倍にする。
    因子1/2/4なら15/29/57モード（標準条件）。krは非等間隔。
    伝搬範囲Lは変更しない。係数積分の周期のみfactor*Lになる。
    """
    if (isinstance(mode_spacing_factor, bool)
            or not isinstance(mode_spacing_factor, (int, np.integer))
            or mode_spacing_factor < 1):
        raise ValueError("mode_spacing_factor must be a positive integer")
    base = theory.mode_design(config)
    if mode_spacing_factor == 1:
        return base
    fine = theory.mode_design(replace(config, design_length_m=
                                     config.design_length_m*mode_spacing_factor))
    selected = np.abs(fine.m) <= int(np.max(np.abs(base.m)))*mode_spacing_factor
    return ModeDesign(m=fine.m[selected], kz_per_m=fine.kz_per_m[selected],
                      kr_per_m=fine.kr_per_m[selected], k0_per_m=fine.k0_per_m,
                      q_per_m=fine.q_per_m, kr_center_per_m=fine.kr_center_per_m)


def design_coefficients(config: theory.Digit1Config, positions: np.ndarray,
                        mode_spacing_factor: int = 1) -> tuple[ModeDesign, np.ndarray]:
    """目標は元の0〜Lだけ。周期を伸ばした部分は強度0として係数を再設計。

    周期を伸ばしても積分のzサンプリング幅は元と同じに保つ。
    """
    design = mode_design_for_spacing(config, mode_spacing_factor)
    period = config.design_length_m*mode_spacing_factor
    count = config.coefficient_samples*mode_spacing_factor
    if count < design.count:
        raise ValueError("coefficient quadrature samples must be >= mode count")
    zm = (np.arange(count)+.5)*period/count
    weights = theory.fourier_coefficients(np.sqrt(theory.digit_target(positions, zm, config)),
                                         zm, design.m, period)
    return design, weights


def library_spec(config: theory.Digit1Config, rank: int, seed: int,
                 mode_spacing_factor: int = 1,
                 basis_strategy: str = "shift_aware") -> dict:
    mode_design_for_spacing(config, mode_spacing_factor)  # 値の検証
    spec = {
        "schema_version": SCHEMA_VERSION, "rank": rank, "seed": seed,
        "grid_size": config.grid_size, "pixel_pitch_m": config.pixel_pitch_m,
        "wavelength_m": config.wavelength_m, "spot_radius_m": config.spot_radius_m,
        "design_length_m": config.design_length_m,
        "coordinates": "(arange(N)-N//2)*pitch; ordinary unshifted fft2",
        "dictionary_field": "unwindowed J0; aperture applied after thread summation",
        "oversampling": 16, "power_iterations": 2,
    }
    # 標準辞書の既存メタデータはそのまま。密なモード辞書は区別する。
    if mode_spacing_factor != 1:
        spec.update(mode_spacing_factor=mode_spacing_factor,
                    coefficient_period_m=config.design_length_m*mode_spacing_factor,
                    mode_range="same kr endpoints as spacing_factor=1")
    if basis_strategy not in ("center_only", "shift_aware"):
        raise ValueError("basis_strategy must be center_only or shift_aware")
    if basis_strategy == "shift_aware":
        positions = (np.arange(config.thread_count)-(config.thread_count-1)/2)*config.thread_spacing_m
        spec.update(basis_strategy=basis_strategy,
                    training_x_shifts_m=np.unique(np.append(positions, 0)).tolist(),
                    training_method="real/imag columns of Dx Ux_center sqrt(sum Cm.H Cm), separate Uy/Ux")
    return spec


def _shift_aware_x_basis(center_factor: np.ndarray, center_cores: np.ndarray,
                         shifts: np.ndarray, pitch: float,
                         config: StageAConfig) -> np.ndarray:
    """オフライン: 移動による奇成分を含めて実直交Uxを学習する。

    中心場のx共分散G=sum(Re Cm.T Re Cm+Im Cm.T Im Cm)を用い、
    各位置でB=diag(exp(-ikx*s)) Ux_center sqrt(G)を作る。
    Re B/Im Bを列学習データにするので、中心のみの偶基底へ制限されない。
    中心圧縮の近似を介した学習であり、全シフトの厳密表現は保証しない。
    NxNのシフト済みビームFFTを追加保存する必要はない。
    """
    n, r = center_factor.shape
    gram = sum((c.real.T @ c.real+c.imag.T @ c.imag for c in center_cores),
               np.zeros((r, r), dtype=float))
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    root = vectors*np.sqrt(np.maximum(values, 0))[None, :]
    weighted = center_factor @ root
    kx = 2*np.pi*np.fft.fftfreq(n, d=pitch)
    def components():
        for shift in shifts:
            b = np.exp(-1j*kx*shift)[:, None]*weighted
            yield b.real
            yield b.imag
    width = min(n, config.rank+config.randomized_oversampling)
    rng = np.random.default_rng(config.seed+202)
    sketch = np.zeros((n, width))
    for component in components():
        sketch += component @ rng.standard_normal((r, width))
    q, _ = np.linalg.qr(sketch, mode="reduced")
    for _ in range(config.randomized_power_iterations):
        accumulated = np.zeros_like(q)
        for component in components():
            accumulated += component @ (component.T @ q)
        q, _ = np.linalg.qr(accumulated, mode="reduced")
    small_gram = np.zeros((width, width))
    for component in components():
        projected = component.T @ q
        small_gram += projected.T @ projected
    values, vectors = np.linalg.eigh((small_gram+small_gram.T)/2)
    selected = np.argsort(values)[-config.rank:][::-1]
    basis, _ = np.linalg.qr(q @ vectors[:, selected], mode="reduced")
    return basis


def _basis_parity(ux: np.ndarray) -> dict:
    """FFTの鏡映indexは(-j)%N。fliplrでは半画素分ずれる。"""
    reflected = ux[(-np.arange(len(ux))) % len(ux)]
    energy = float(np.sum(ux**2))
    return {"even_column_energy_fraction": float(np.sum(((ux+reflected)/2)**2)/energy),
            "odd_column_energy_fraction": float(np.sum(((ux-reflected)/2)**2)/energy)}


def load_or_build_library(path: str | Path, config: theory.Digit1Config,
                          rank: int, seed: int = 42,
                          mode_spacing_factor: int = 1,
                          basis_strategy: str = "shift_aware") -> tuple[dict, dict]:
    """オフライン: 全モードFFT→共有実直交因子→複素コア→因子列IFFT。

    全FFTは一時memmapだけに保持し、保存するのは圧縮辞書のみ。
    同じ仕様の辞書があれば再利用し、不一致なら上書きせずエラーにする。
    """
    if not 1 <= rank <= config.grid_size:
        raise ValueError("rank must be between 1 and grid_size")
    path = Path(path)
    spec = library_spec(config, rank, seed, mode_spacing_factor, basis_strategy)
    design = mode_design_for_spacing(config, mode_spacing_factor)
    required = {"uy", "ux", "vy", "vx", "cores", "m", "kr", "spec_json"}
    if path.exists():
        tic = time.perf_counter()
        with np.load(path, allow_pickle=False) as saved:
            if not required.issubset(saved.files):
                raise ValueError("Incompatible library: use a new digit1 library path")
            if json.loads(str(saved["spec_json"].item())) != spec:
                raise ValueError("Library settings differ: choose another library_path")
            library = {key: saved[key].copy() for key in required - {"spec_json"}}
        n, r, nm = config.grid_size, rank, design.count
        if (any(library[k].shape != (n, r) for k in ("uy", "ux", "vy", "vx"))
                or library["cores"].shape != (nm, r, r)
                or not np.array_equal(library["m"], design.m)
                or not np.allclose(library["kr"], design.kr_per_m, rtol=1e-12, atol=0)):
            raise ValueError("Invalid library dimensions or modes")
        return library, {"status": "loaded", "load_seconds": time.perf_counter() - tic,
                         "build_seconds": None, "spec": spec,
                         "x_basis_parity": _basis_parity(library["ux"])}

    path.parent.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    n = config.grid_size
    x = (np.arange(n) - n // 2) * config.pixel_pitch_m
    radius = np.hypot(x[:, None], x[None, :])
    fit_config = StageAConfig(
        grid_size=n, rank=rank, seed=seed, selected_mode_count=1,
        randomized_oversampling=spec["oversampling"],
        randomized_power_iterations=spec["power_iterations"])
    with tempfile.TemporaryDirectory(prefix="digit1_fft_", dir=path.parent) as tmp:
        spectra = np.memmap(Path(tmp) / "spectra.dat", mode="w+",
                            dtype=np.complex128, shape=(design.count, n, n))
        try:
            for i, kr in enumerate(design.kr_per_m):
                spectra[i] = np.fft.fft2(j0(kr * radius), norm="ortho")
                print(f"Offline dictionary: mode {i+1}/{design.count}", flush=True)
            spectra.flush()
            uy = _randomized_shared_basis(spectra, fit_config)
            ux = uy.copy()
            center_cores = np.stack([_project_core(spectra[i], uy, ux)
                                     for i in range(design.count)])
            center_parity = _basis_parity(ux)
            if basis_strategy == "shift_aware":
                print("Offline: fit x basis including signed translations", flush=True)
                ux = _shift_aware_x_basis(ux, center_cores,
                    np.asarray(spec["training_x_shifts_m"]), config.pixel_pitch_m, fit_config)
                cores = np.stack([_project_core(spectra[i], uy, ux)
                                  for i in range(design.count)])
            else:
                cores = center_cores
        finally:
            # Windowsではmemmapを閉じてから一時ディレクトリを削除する。
            spectra._mmap.close()
            del spectra
    vy, vx = _spatial_factors(uy, ux)
    build_seconds = time.perf_counter() - tic
    library = {"uy": uy, "ux": ux, "vy": vy, "vx": vx,
               "cores": cores, "m": design.m, "kr": design.kr_per_m}
    np.savez_compressed(path, **library, spec_json=json.dumps(spec, sort_keys=True))
    return library, {"status": "built", "build_seconds": build_seconds,
                     "load_seconds": None, "spec": spec,
                     "center_x_basis_parity": center_parity,
                     "x_basis_parity": _basis_parity(ux)}


def proposed_field(library: dict, coefficients: np.ndarray,
                   positions: np.ndarray, pitch: float) -> tuple[np.ndarray, dict, np.ndarray]:
    """オンライン: Thread別重み付きコア→射影シフト→全コア加算→1回復元。

    H_q=sum_m A[m,q] C_m; Ax_q=Ux.T diag(exp(-ikx*xq)) Ux;
    H_total=sum_q H_q Ax_q; E=Vy H_total Vx.T。
    因子列IFFTはオフライン済みなので、ここで全画素IFFTは実行しない。
    """
    cores, ux = library["cores"], library["ux"]
    if coefficients.shape != (len(cores), len(positions)):
        raise ValueError("Coefficient and library mode counts differ")
    active = np.flatnonzero(np.any(coefficients != 0, axis=0))
    timings = {"thread_core_weighting_seconds": 0.0,
               "shift_operator_seconds": 0.0, "shifted_core_sum_seconds": 0.0}
    tic_total = time.perf_counter()
    kx = 2 * np.pi * np.fft.fftfreq(ux.shape[0], d=pitch)
    total = np.zeros(cores.shape[1:], dtype=np.complex128)
    for q in active:
        tic = time.perf_counter()
        weighted = np.tensordot(coefficients[:, q], cores, axes=(0, 0))
        timings["thread_core_weighting_seconds"] += time.perf_counter() - tic
        tic = time.perf_counter()
        ax = _x_shift_core_operator(ux, np.exp(-1j * kx * positions[q]))
        timings["shift_operator_seconds"] += time.perf_counter() - tic
        tic = time.perf_counter()
        total += weighted @ ax
        timings["shifted_core_sum_seconds"] += time.perf_counter() - tic
    tic = time.perf_counter()
    field = _reconstruct_spatial(total, library["vy"], library["vx"])
    timings["spatial_factor_reconstruction_seconds"] = time.perf_counter() - tic
    timings["field_generation_seconds"] = time.perf_counter() - tic_total
    return field, timings, total


def shift_diagnostics(library: dict, coefficients: np.ndarray, positions: np.ndarray,
                      pitch: float, direct: np.ndarray, proposed: np.ndarray,
                      illumination: np.ndarray) -> tuple[dict, np.ndarray]:
    """オンライン時間外で周期境界・辞書圧縮・シフト射影の差を切り分ける。

    exact_periodic: 理論J0の有限配列をFFTで厳密に周期シフト。
    compressed_periodic: コア復元FFTを同じようにシフトし、再射影しない。
    analytic_direct: ずらした座標でJ0を直接計算（配列の折り返しなし）。
    """
    n = direct.shape[0]
    kx = 2 * np.pi * np.fft.fftfreq(n, d=pitch)
    x = (np.arange(n) - n // 2) * pitch
    radius = np.hypot(x[:, None], x[None, :])
    phases = np.exp(-1j * positions[:, None] * kx[None, :])
    exact_spectrum = np.zeros((n, n), dtype=np.complex128)
    compressed_spectrum = np.zeros_like(exact_spectrum)
    # 各モードを1回だけ計算。sum_q A[m,q] D_qでThreadをまとめる。
    for m, kr in enumerate(library["kr"]):
        weight_x = coefficients[m] @ phases
        exact_spectrum += np.fft.fft2(j0(kr * radius), norm="ortho") * weight_x[None, :]
        compressed_spectrum += ((library["uy"] @ library["cores"][m])
                                @ library["ux"].T) * weight_x[None, :]
    exact = np.fft.ifft2(exact_spectrum, norm="ortho")
    compressed = np.fft.ifft2(compressed_spectrum, norm="ortho")
    def error(a, b):
        return {"full_window_complex_relative_l2": _relative_l2(a, b),
                "after_aperture_complex_relative_l2": _relative_l2(a * illumination,
                                                                    b * illumination)}
    return {
        "analytic_direct_vs_exact_periodic": error(direct, exact),
        "exact_periodic_vs_compressed_periodic": error(exact, compressed),
        "compressed_periodic_vs_projected_proposed": error(compressed, proposed),
        "analytic_direct_vs_proposed": error(direct, proposed),
        "note": "Different references: errors are not additive. No diagnostics in online timing.",
    }, exact


def _plot_comparison(out: Path, x: np.ndarray, z: np.ndarray, roi: np.ndarray,
                     images: dict, filename: str, xy: bool = False) -> None:
    # 全ケース同一スケール。差分だけ専用スケールを明示する。
    names = list(images)
    a, b = images[names[0]], images[names[1]]
    difference = np.abs(b - a)
    extent = ([x[0]*1000, x[-1]*1000, x[0]*1000, x[-1]*1000] if xy else
              [x[roi][0]*1000, x[roi][-1]*1000, z[0]*1000, z[-1]*1000])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, data, title, vmax in zip(axes, (a, b, difference),
                                   (*names, "Absolute intensity difference (own scale)"),
                                   (max(a.max(), b.max()), max(a.max(), b.max()), difference.max())):
        im = ax.imshow(data if xy else data[:, roi], origin="lower", extent=extent,
                       aspect="equal" if xy else "auto", cmap="gray", vmin=0,
                       vmax=max(float(vmax), np.finfo(float).tiny))
        ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]" if xy else "z [mm]")
        if xy:
            ax.axhline(0, color="red", linewidth=.5)
        fig.colorbar(im, ax=ax)
    fig.savefig(out / filename, dpi=160)
    plt.close(fig)


def _save_initial_comparison(out: Path, direct: np.ndarray, proposed: np.ndarray,
                             codes: dict) -> None:
    """位相は基準・復元の両方が明るい画素で表示し、円周差を用いる。"""
    a, b = np.abs(direct), np.abs(proposed)
    mask = (a > .01*a.max()) & (b > .01*a.max())
    values = (a, b, np.abs(b-a), np.where(mask, np.angle(direct), np.nan),
              np.where(mask, np.angle(proposed), np.nan),
              np.where(mask, np.angle(proposed*np.conj(direct)), np.nan))
    titles = ("Direct amplitude", "Proposed amplitude", "Amplitude difference (own scale)",
              "Direct phase (bright mask)", "Proposed phase (bright mask)", "Circular phase error [rad]")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    peak = max(float(a.max()), float(b.max()))
    for i, (ax, value, title) in enumerate(zip(axes.ravel(), values, titles)):
        im = ax.imshow(value, origin="lower", cmap="gray",
                       vmin=-np.pi if i >= 3 else 0,
                       vmax=np.pi if i >= 3 else (peak if i < 2 else max(float(value.max()), 1e-15)))
        ax.set(title=title, xlabel="x sample", ylabel="y sample")
        fig.colorbar(im, ax=ax)
    fig.savefig(out/"initial_field_comparison.png", dpi=150)
    plt.close(fig)
    difference = ((codes["proposed"].astype(np.int16)-codes["direct"].astype(np.int16)+128)%256)-128
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, data, title in zip(axes, (codes["direct"], codes["proposed"], difference),
                              ("Direct 8-bit CGH", "Proposed 8-bit CGH", "Circular code difference")):
        is_diff = title.startswith("Circular")
        im = ax.imshow(data, origin="lower", cmap="coolwarm" if is_diff else "gray",
                       vmin=-128 if is_diff else 0, vmax=128 if is_diff else 255)
        ax.set(title=title, xlabel="x sample", ylabel="y sample")
        fig.colorbar(im, ax=ax)
    fig.savefig(out/"cgh_mask_comparison.png", dpi=150)
    plt.close(fig)


def run_comparison(output_dir: str | Path, config: theory.Digit1Config | None = None,
                   rank: int = 100, library_path: str | Path | None = None,
                   seed: int = 42, save_images: bool = True,
                   run_shift_diagnostics: bool = True,
                   mode_spacing_factor: int = 1,
                   basis_strategy: str = "shift_aware") -> dict:
    config = config or theory.Digit1Config()
    if not config.include_cgh_replay:
        raise ValueError("This comparison needs include_cgh_replay=True")
    design = mode_design_for_spacing(config, mode_spacing_factor)
    if config.coefficient_samples*mode_spacing_factor < design.count:
        raise ValueError("coefficient quadrature samples must be >= mode count")
    if not 1 <= rank <= config.grid_size:
        raise ValueError("rank must be between 1 and grid_size")
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output not empty: {out}. Choose a new output_dir.")
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if mode_spacing_factor == 1 else f"_spacing{mode_spacing_factor}"
    suffix += "_shiftaware" if basis_strategy == "shift_aware" else ""
    library_path = Path(library_path or out.parent / f"lsh_digit1_library_rank{rank}{suffix}.npz")
    library, offline = load_or_build_library(library_path, config, rank, seed,
                                             mode_spacing_factor, basis_strategy)
    if save_images:
        factors = out / "factors"
        factors.mkdir()
        _save_factor_figures(factors, library["uy"], library["ux"], library["cores"], design)

    n = config.grid_size
    x = (np.arange(n) - n//2) * config.pixel_pitch_m
    z = np.linspace(0, config.design_length_m, config.z_samples)
    positions = (np.arange(config.thread_count) - (config.thread_count-1)/2) * config.thread_spacing_m
    _, weights = design_coefficients(config, positions, mode_spacing_factor)
    if not np.any(weights):
        raise ValueError("No thread intersects the target")
    illumination = (np.hypot(x[:, None], x[None, :]) <= config.aperture_radius_m).astype(float)
    if config.illumination_waist_m is not None:
        illumination *= np.exp(-(np.hypot(x[:, None], x[None, :])/config.illumination_waist_m)**2)

    # 直接法は毎回J0から計算し、保存FFT・辞書を使わない。
    tic = time.perf_counter()
    direct = theory.initial_bessel_field(x, positions, design.kr_per_m, weights)
    direct_seconds = time.perf_counter()-tic
    proposed, proposed_times, total_core = proposed_field(library, weights, positions, config.pixel_pitch_m)
    scale = float(np.max(np.abs(direct)[illumination > 0]))
    if scale <= 0:
        raise ValueError("Zero reference field")
    diagnostics = {"status": "not_run"}
    if run_shift_diagnostics:
        diagnostics, exact_periodic = shift_diagnostics(
            library, weights, positions, config.pixel_pitch_m, direct, proposed, illumination)
        del exact_periodic
    # 両手法の正規化は必ず直接法の同じ定数を使用する。
    direct /= scale
    proposed /= scale
    fields, codes, encodings, cgh_times = {}, {}, {}, {}
    for name, field in (("direct", direct), ("proposed", proposed)):
        encoding_field = np.where(illumination > 0, field, 0)
        fields[name+"_complex"] = (encoding_field * illumination).astype(np.complex64)
        tic = time.perf_counter()
        codes[name], replay, encodings[name] = theory.encode_and_replay(
            encoding_field, illumination, config, float(design.kr_per_m.max()))
        cgh_times[name] = time.perf_counter()-tic
        fields[name+"_cgh"] = replay.astype(np.complex64)
    snapshots = {int(np.argmin(np.abs(z-depth))) for depth in config.sensor_z_m}
    tic = time.perf_counter()
    rows, frames, sampling = theory.propagate(fields, z, config, snapshots)
    propagation_seconds = time.perf_counter()-tic
    roi = np.abs(x) <= config.evaluation_x_half_width_m
    target = theory.digit_target(x, z, config)/scale**2
    sheets = {name: np.abs(row)**2 for name, row in rows.items()}
    mirror = (-np.arange(n)) % n
    kx = 2*np.pi*np.fft.fftfreq(n, d=config.pixel_pitch_m)
    max_shift = float(np.max(np.abs(positions)))
    a_plus = _x_shift_core_operator(library["ux"], np.exp(-1j*kx*max_shift))
    a_minus = _x_shift_core_operator(library["ux"], np.exp(1j*kx*max_shift))
    symmetry = {
        "reflection_indices": "(-arange(N)) % N; not fliplr",
        "x_basis_parity": _basis_parity(library["ux"]),
        "signed_operator_test_shift_m": max_shift,
        "plus_vs_minus_operator_relative_l2": _relative_l2(a_plus, a_minus),
        "initial_field_vs_x_reflection_complex_l2": {
            name: _relative_l2(field, field[:, mirror]) for name, field in fields.items()},
        "sheet_vs_x_reflection_intensity_l2": {
            name: _relative_l2(image[:, roi], image[:, mirror][:, roi])
            for name, image in sheets.items()},
        "note": "Nonzero asymmetry alone is not success; also inspect proposed-vs-direct metrics.",
    }
    metrics = {
        "initial_complex_proposed_vs_direct": _field_metrics(fields["direct_complex"], fields["proposed_complex"], .01),
        "initial_cgh_replay_proposed_vs_direct": _field_metrics(fields["direct_cgh"], fields["proposed_cgh"], .01),
        "cgh_mask_proposed_vs_direct": _cgh_metrics(codes["direct"], codes["proposed"],
                                                    fields["direct_complex"], .01),
    }
    # Stage A共通関数の旧angle-only説明は、この振幅符号化実験には適用しない。
    metrics["cgh_mask_proposed_vs_direct"].pop("replay_intensity", None)
    metrics["cgh_mask_proposed_vs_direct"]["replay_note"] = (
        "Amplitude-aware mask metrics only. Actual optical replay is evaluated separately below.")
    for kind in ("complex", "cgh"):
        ref, rec = "direct_"+kind, "proposed_"+kind
        metrics[kind+"_sheet_proposed_vs_direct"] = {
            "complex_relative_l2": _relative_l2(rows[ref][:, roi], rows[rec][:, roi]),
            "intensity_relative_l2": _relative_l2(sheets[ref][:, roi], sheets[rec][:, roi]),
            **_image_metrics(sheets[ref][:, roi], sheets[rec][:, roi])}
        for name in (ref, rec):
            metrics[name+"_sheet_vs_target"] = {
                "intensity_relative_l2": _relative_l2(target[:, roi], sheets[name][:, roi]),
                **_image_metrics(target[:, roi], sheets[name][:, roi])}
    metrics["sensor_xy_proposed_vs_direct"] = {
        kind: [{"z_m": float(z[i]), **_image_metrics(frames["direct_"+kind][i], frames["proposed_"+kind][i])}
               for i in sorted(snapshots)] for kind in ("complex", "cgh")}
    padding = theory._padding_check(fields, rows, z, roi, config, snapshots)
    np.savez_compressed(out/"comparison_data.npz", x_m=x, z_m=z, target_intensity=target,
                        thread_x_m=positions, coefficients=weights, total_core=total_core,
                        **{name+"_row": row for name, row in rows.items()}, **sheets)
    np.savez_compressed(out/"initial_fields.npz", **fields)
    np.savez_compressed(out/"sensor_frames.npz",
                        **{f"{name}_index{i:03d}": image for name, case in frames.items() for i, image in case.items()},
                        **{f"{name}_raw_index{i:03d}": image*config.encoding_coupling**2
                           for name, case in frames.items() if name.endswith("_cgh") for i, image in case.items()})
    if save_images:
        for name, code in codes.items():
            Image.fromarray(code).save(out/f"{name}_cgh_8bit.png")
        _save_initial_comparison(out, fields["direct_complex"], fields["proposed_complex"], codes)
        for kind in ("complex", "cgh"):
            _plot_comparison(out, x, z, roi, {"Direct "+kind: sheets["direct_"+kind],
                             "Proposed "+kind: sheets["proposed_"+kind]}, f"{kind}_sheet_comparison.png")
        # 目標と両手法を並べて「直接法に一致」と「目標に一致」を区別する。
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        peak = max(float(target.max()), float(sheets["direct_cgh"].max()), float(sheets["proposed_cgh"].max()))
        for ax, image, title in zip(axes, (target, sheets["direct_cgh"], sheets["proposed_cgh"]),
                                   ("Digit 1 target", "Direct CGH replay", "Proposed CGH replay")):
            im = ax.imshow(image[:, roi], origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=peak,
                           extent=[x[roi][0]*1000, x[roi][-1]*1000, z[0]*1000, z[-1]*1000])
            ax.set(title=title, xlabel="x [mm]", ylabel="z [mm]")
            fig.colorbar(im, ax=ax, label="Intensity, common scale")
        fig.savefig(out/"target_direct_proposed.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
        antisym = {name: sheets[name]-sheets[name][:, mirror]
                   for name in ("direct_cgh", "proposed_cgh")}
        limit = max(max(float(np.max(np.abs(v[:, roi]))), 1e-15) for v in antisym.values())
        for ax, (name, image) in zip(axes, antisym.items()):
            im = ax.imshow(image[:, roi], origin="lower", aspect="auto", cmap="coolwarm",
                           vmin=-limit, vmax=limit,
                           extent=[x[roi][0]*1000, x[roi][-1]*1000, z[0]*1000, z[-1]*1000])
            ax.set(title=name+": I(x,z)-I(-x,z)", xlabel="x [mm]", ylabel="z [mm]")
            fig.colorbar(im, ax=ax, label="Antisymmetric intensity, common scale")
        fig.savefig(out/"asymmetry_comparison.png", dpi=160)
        plt.close(fig)
        sensor = out/"sensor_xy"
        sensor.mkdir()
        for i in sorted(snapshots):
            for kind in ("complex", "cgh"):
                _plot_comparison(sensor, x, z, roi,
                    {f"Direct {kind}, z={z[i]*1000:.1f}mm": frames["direct_"+kind][i],
                     f"Proposed {kind}, z={z[i]*1000:.1f}mm": frames["proposed_"+kind][i]},
                    f"{kind}_z{z[i]*1000:06.1f}mm.png", xy=True)
    summary = {
        "study": "digit1_direct_vs_projected_shared_core", "schema_version": SCHEMA_VERSION,
        "config": asdict(config), "rank": rank, "seed": seed,
        "mode_spacing_factor": mode_spacing_factor,
        "basis_strategy": basis_strategy,
        "coefficient_period_m": config.design_length_m*mode_spacing_factor,
        "kz_spacing_per_m": 2*np.pi/(config.design_length_m*mode_spacing_factor),
        "m": design.m.tolist(), "kr_per_m": design.kr_per_m.tolist(),
        "library_path": str(library_path.resolve()), "offline": offline,
        "modes_per_thread": design.count,
        "active_thread_count": int(np.count_nonzero(np.any(weights != 0, axis=0))),
        "nonzero_bessel_term_count": int(np.count_nonzero(weights)),
        "reference_scale": scale, "encoding": encodings, "metrics": metrics,
        "shift_diagnostics": diagnostics, "symmetry_diagnostics": symmetry,
        "propagation_sampling": sampling, "padding_check": padding,
        "timings_seconds": {"direct_field_generation": direct_seconds,
            "proposed_field_generation": proposed_times,
            "cgh_encode_and_ideal4f_replay": cgh_times, "all_four_propagation": propagation_seconds},
        "timing_scope": "Single runs, not a repeated benchmark. Field timers exclude offline, shared weights, aperture, normalization, diagnostics, saving and propagation. CGH timer includes ideal4f replay, not just mask generation.",
        "interpretation": "Check proposed-vs-direct AND each method-vs-target. Rank cannot repair the target design or finite-window periodic boundary mismatch. Ideal virtual 1:1 optics, not experimental sensor data.",
        "online_algorithm": "Hq=sum_m A[m,q] Cm; Htotal=sum_q Hq (Ux.T Dxq Ux); E=Vy Htotal Vx.T",
    }
    theory._write_json(out/"summary.json", summary)
    theory._write_json(out/"manifest.json", {"study": summary["study"], "schema_version": SCHEMA_VERSION,
        "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Default: lsh_digit1_proposed_rank{rank}_v1")
    parser.add_argument("--rank", type=int, default=100)
    parser.add_argument("--mode-spacing-factor", type=int, default=1,
                        help="1=original kz spacing; 2=half; 4=quarter; same kr endpoints")
    parser.add_argument("--basis-strategy", choices=("center_only", "shift_aware"), default="shift_aware")
    parser.add_argument("--library-path")
    parser.add_argument("--theory-results", help="Reuse the saved theory configuration only")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-shift-diagnostics", action="store_true")
    args = parser.parse_args()
    config = config_from_theory_results(args.theory_results) if args.theory_results else theory.Digit1Config()
    suffix = "" if args.mode_spacing_factor == 1 else f"_spacing{args.mode_spacing_factor}"
    suffix += "_shiftaware_v2" if args.basis_strategy == "shift_aware" else "_v1"
    output_dir = args.output_dir or f"lsh_digit1_proposed_rank{args.rank}{suffix}"
    summary = run_comparison(output_dir, config, args.rank, args.library_path,
                             save_images=not args.no_images,
                             run_shift_diagnostics=not args.no_shift_diagnostics,
                             mode_spacing_factor=args.mode_spacing_factor,
                             basis_strategy=args.basis_strategy)
    print("Saved to", Path(output_dir).resolve())
    print(summary["metrics"]["cgh_sheet_proposed_vs_direct"])


if __name__ == "__main__":
    main()
