"""同じ123シートをy方向へ並べ、直接J0生成と2軸コアシフトを検証する。

数値計算はrun_multisheet()またはCLIを明示的に呼んだときだけ実行する。
既存結果から読むのは設定・Thread座標・重み係数・圧縮辞書であり、
保存済み直接場や再生画像を新しい場の生成に使用しない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import j0

import lsh_digit1_proposed as proposed
import lsh_digit1_theory as theory
from stage_a_parameters import StageAConfig
from stage_a_validation import (
    _field_metrics, _image_metrics, _project_core, _reconstruct_spatial,
    _relative_l2, _save_factor_figures, _spatial_factors, _x_shift_core_operator,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MultiSheetConfig:
    rank: int = 450
    parent_rank: int = 512
    sheet_count: int = 5
    sheet_spacing_m: float = 75e-6
    sheet_phase_step_rad: float = 0.0
    z_samples: int = 101
    padding_factor: int | None = None  # Noneなら元の条件。3等を指定して収束を比較できる。
    evaluation_y_half_width_m: float = 0.5e-3
    # Noneなら使用するy位置を事前学習する。別の座標を指定すれば未学習位置を評価できる。
    training_y_shifts_m: tuple[float, ...] | None = None
    # 直接法も毎回J0を計算するため、初回は両手法1回・ウォームアップなし。
    timing_repeats: int = 1
    timing_warmups: int = 0
    max_initial_complex_error: float = 0.01
    max_initial_intensity_error: float = 0.03
    scatter_threshold_fraction: float = 0.20
    scatter_max_points: int = 12000

    def __post_init__(self) -> None:
        integers = (self.rank, self.parent_rank, self.sheet_count, self.z_samples,
                    self.timing_repeats, self.timing_warmups, self.scatter_max_points)
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in integers):
            raise ValueError("ranks, counts and repeat settings must be integers")
        if not 1 <= self.rank <= self.parent_rank:
            raise ValueError("require 1 <= rank <= parent_rank")
        if self.sheet_count < 1 or self.z_samples < 3 or self.scatter_max_points < 1:
            raise ValueError("invalid sheet/z/scatter count")
        if self.timing_repeats < 1 or self.timing_warmups < 0:
            raise ValueError("invalid timing repeats/warmups")
        if (self.padding_factor is not None
                and (isinstance(self.padding_factor, bool)
                     or not isinstance(self.padding_factor, (int, np.integer))
                     or self.padding_factor < 1)):
            raise ValueError("padding_factor must be a positive integer or None")
        if (not np.isfinite(self.sheet_phase_step_rad)
                or not np.isfinite(self.sheet_spacing_m) or self.sheet_spacing_m <= 0
                or not np.isfinite(self.evaluation_y_half_width_m)
                or self.evaluation_y_half_width_m <= 0):
            raise ValueError("invalid sheet phase, spacing or evaluation width")
        for value in (self.max_initial_complex_error, self.max_initial_intensity_error,
                      self.scatter_threshold_fraction):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError("error limits and scatter threshold must be in (0,1]")
        if self.training_y_shifts_m is not None:
            if not self.training_y_shifts_m or not np.all(np.isfinite(self.training_y_shifts_m)):
                raise ValueError("training_y_shifts_m must contain finite coordinates")

    def y_positions(self) -> np.ndarray:
        return (np.arange(self.sheet_count)-(self.sheet_count-1)/2)*self.sheet_spacing_m

    def sheet_weights(self) -> np.ndarray:
        # 各シートは同じ振幅。位相差0なら同位相のコヒーレント重ね合わせ。
        return np.exp(1j*np.arange(self.sheet_count)*self.sheet_phase_step_rad)


def load_or_build_xy_library(path: Path, parent: dict, parent_spec: dict,
                             config: theory.Digit1Config, stack: MultiSheetConfig,
                             seed: int) -> tuple[dict, dict]:
    """x基底を継承し、y移動を含むUyを別途学習。中心J0のFFTを再射影する。

    y学習ではC_m.Tを使い、既存のx共分散処理をy側に適用する。
    学習に目標画像・その係数・最終再生像は使用しない。
    新しいコアは圧縮済みコアの変換で済ませず、解析J0のFFTから作る。
    """
    actual_y = stack.y_positions()
    training_y = np.unique(np.append(
        actual_y if stack.training_y_shifts_m is None else stack.training_y_shifts_m, 0.0))
    spec = {"schema_version": SCHEMA_VERSION, "parent_spec": parent_spec,
            "parent_rank": stack.parent_rank, "training_y_shifts_m": training_y.tolist(),
            "method": "shift_aware_y_from_parent; fresh unwindowed J0 core projection",
            "y_seed": seed+303}
    required = {"uy", "ux", "vy", "vx", "cores", "m", "kr", "spec_json"}
    if path.exists():
        tic = time.perf_counter()
        with np.load(path, allow_pickle=False) as saved:
            if not required.issubset(saved.files) or json.loads(str(saved["spec_json"].item())) != spec:
                raise ValueError("Multi-sheet dictionary settings differ; use a new library path")
            library = {key: saved[key].copy() for key in required-{"spec_json"}}
        n, r = config.grid_size, stack.parent_rank
        if (any(library[key].shape != (n, r) for key in ("uy", "ux", "vy", "vx"))
                or library["cores"].shape != (len(parent["kr"]), r, r)
                or not np.array_equal(library["m"], parent["m"])
                or not np.allclose(library["kr"], parent["kr"], rtol=1e-12, atol=0)):
            raise ValueError("Invalid multi-sheet dictionary arrays")
        return library, {"status": "loaded", "load_seconds": time.perf_counter()-tic,
                         "spec": spec, "path": str(path.resolve())}
    tic = time.perf_counter()
    fit = StageAConfig(grid_size=config.grid_size, rank=stack.parent_rank, seed=seed+303,
                       selected_mode_count=1, randomized_oversampling=16,
                       randomized_power_iterations=2)
    uy = proposed._shift_aware_x_basis(parent["uy"], parent["cores"].transpose(0, 2, 1),
                                      training_y, config.pixel_pitch_m, fit)
    ux = parent["ux"].copy()
    axis = (np.arange(config.grid_size)-config.grid_size//2)*config.pixel_pitch_m
    radius = np.hypot(axis[:, None], axis[None, :])
    cores = np.empty_like(parent["cores"])
    for index, kr in enumerate(parent["kr"]):
        spectrum = np.fft.fft2(j0(kr*radius), norm="ortho")
        cores[index] = _project_core(spectrum, uy, ux)
        print(f"Offline xy dictionary: mode {index+1}/{len(cores)}", flush=True)
    vy, vx = _spatial_factors(uy, ux)
    library = {"uy": uy, "ux": ux, "vy": vy, "vx": vx, "cores": cores,
               "kr": parent["kr"].copy(), "m": parent["m"].copy()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **library, spec_json=json.dumps(spec, sort_keys=True))
    return library, {"status": "built", "build_and_save_seconds": time.perf_counter()-tic,
                     "spec": spec, "path": str(path.resolve())}


def _sheet_coefficients(coefficients: np.ndarray, mode_count: int,
                        thread_count: int, sheet_count: int) -> np.ndarray:
    """共通係数[m,q]と、シート別の係数[m,s,q]の両方を扱う。"""
    coefficients = np.asarray(coefficients)
    if coefficients.shape == (mode_count, thread_count):
        return np.broadcast_to(coefficients[:, None, :],
                               (mode_count, sheet_count, thread_count))
    if coefficients.shape == (mode_count, sheet_count, thread_count):
        return coefficients
    raise ValueError("coefficients must have shape [mode,thread] or [mode,sheet,thread]")


def direct_multisheet_field(axis: np.ndarray, thread_x: np.ndarray, sheet_y: np.ndarray,
                           kr: np.ndarray, coefficients: np.ndarray,
                           sheet_weights: np.ndarray) -> np.ndarray:
    """直接法：全シートの各(xq,ys)でJ0を都度生成し、複素場として加算する。"""
    weights = _sheet_coefficients(coefficients, len(kr), len(thread_x), len(sheet_y))
    if sheet_weights.shape != sheet_y.shape:
        raise ValueError("coefficient/sheet dimensions differ")
    field = np.zeros((len(axis), len(axis)), dtype=np.complex128)
    for index, (y, gain) in enumerate(zip(sheet_y, sheet_weights)):
        for q, x in enumerate(thread_x):
            if not np.any(weights[:, index, q]):
                continue
            radius = np.hypot(axis[:, None]-y, axis[None, :]-x)
            for mode, coefficient in zip(kr, weights[:, index, q]):
                if coefficient != 0:
                    field += (gain*coefficient)*j0(mode*radius)
        print(f"Direct J0 field: sheet {index+1}/{len(sheet_y)}", flush=True)
    return field


def proposed_multisheet_field(library: dict, coefficients: np.ndarray,
                             thread_x: np.ndarray, sheet_y: np.ndarray,
                             sheet_weights: np.ndarray, pitch: float) -> tuple[np.ndarray, dict]:
    """H_s=sum_q (sum_m a_msq C_m) Axq; H_total=sum_s gs Ays H_s。

    共通係数[m,q]のときだけH_sを一度作って再利用する。
    3D設計でシート別係数[m,s,q]を用いた場合は各H_sを別々に作る。
    Ax/Ayはオンライン計算。最後の共通因子による復元は一度だけ。
    """
    cores, ux, uy = library["cores"], library["ux"], library["uy"]
    weights = _sheet_coefficients(coefficients, len(cores), len(thread_x), len(sheet_y))
    if sheet_weights.shape != sheet_y.shape:
        raise ValueError("coefficient/sheet dimensions differ")
    shared = np.asarray(coefficients).ndim == 2
    tic_total = time.perf_counter()
    kx = 2*np.pi*np.fft.fftfreq(len(ux), d=pitch)
    ky = 2*np.pi*np.fft.fftfreq(len(uy), d=pitch)
    core_matrix = cores.reshape(len(cores), -1)
    sheet_cores = np.zeros((1 if shared else len(sheet_y), *cores.shape[1:]), dtype=np.complex128)
    times = {"thread_core_weighting_seconds": 0.0, "x_shift_operator_seconds": 0.0,
             "x_shifted_core_sum_seconds": 0.0, "y_shift_operator_seconds": 0.0,
             "y_shifted_core_sum_seconds": 0.0}
    for q in np.flatnonzero(np.any(weights != 0, axis=(0, 1))):
        tic = time.perf_counter()
        ax = _x_shift_core_operator(ux, np.exp(-1j*kx*thread_x[q]))
        times["x_shift_operator_seconds"] += time.perf_counter()-tic
        for s in range(len(sheet_cores)):
            tic = time.perf_counter()
            weighted = (weights[:, s, q] @ core_matrix).reshape(cores.shape[1:])
            times["thread_core_weighting_seconds"] += time.perf_counter()-tic
            tic = time.perf_counter()
            sheet_cores[s] += weighted @ ax
            times["x_shifted_core_sum_seconds"] += time.perf_counter()-tic
    total = np.zeros(cores.shape[1:], dtype=np.complex128)
    for s, (y, gain) in enumerate(zip(sheet_y, sheet_weights)):
        tic = time.perf_counter()
        ay = _x_shift_core_operator(uy, np.exp(-1j*ky*y))
        times["y_shift_operator_seconds"] += time.perf_counter()-tic
        tic = time.perf_counter()
        # y移動はコアの左、x移動は右。共役転置ではなく実基底の転置。
        total += gain*(ay @ sheet_cores[0 if shared else s])
        times["y_shifted_core_sum_seconds"] += time.perf_counter()-tic
    tic = time.perf_counter()
    field = _reconstruct_spatial(total, library["vy"], library["vx"])
    times["spatial_factor_reconstruction_seconds"] = time.perf_counter()-tic
    times["field_generation_seconds"] = time.perf_counter()-tic_total
    return field, times


def _measure_field(function, stack: MultiSheetConfig) -> tuple[np.ndarray, dict]:
    for _ in range(stack.timing_warmups):
        warm, _ = function()
        del warm
    samples = []
    field = None
    for _ in range(stack.timing_repeats):
        if field is not None:
            del field
        field, sample = function()
        samples.append(sample)
    totals = [s["field_generation_seconds"] for s in samples]
    return field, {"samples": samples, "median_seconds": float(np.median(totals)),
                   "min_seconds": min(totals), "max_seconds": max(totals),
                   "repeats": stack.timing_repeats, "warmups": stack.timing_warmups}


def _voxel_scores(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref_i, rec_i = np.abs(reference)**2, np.abs(candidate)**2
    delta = float(np.linalg.norm(rec_i-ref_i))
    peak = float(ref_i.max())
    power = float(ref_i.sum(dtype=np.float64))
    return {"complex_relative_l2": _relative_l2(reference, candidate),
            "intensity_relative_l2": _relative_l2(ref_i, rec_i),
            "intensity_psnr_db": (float("inf") if delta == 0 else
                                   float(20*np.log10(peak*np.sqrt(ref_i.size)/delta)) if peak else None),
            "intensity_sum_ratio": float(rec_i.sum(dtype=np.float64)/power) if power else None,
            "ssim_note": "SSIM is reported per 2D sheet/sensor plane; no flattened 3D SSIM."}


def _save_sensor(out: Path, frames: dict, axis: np.ndarray, depth: float) -> dict:
    metrics = {}
    extent = [axis[0]*1e3, axis[-1]*1e3, axis[0]*1e3, axis[-1]*1e3]
    for kind in ("complex", "cgh"):
        ref, rec = frames["direct_"+kind], frames["proposed_"+kind]
        metrics[kind] = _field_metrics(ref, rec, .01)
        a, b = np.abs(ref)**2, np.abs(rec)**2
        peak = max(float(a.max()), float(b.max()), np.finfo(float).tiny)
        fig, axes = theory.plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, data, title, vmax in zip(axes, (a, b, np.abs(b-a)),
                ("Direct sensor intensity", "Proposed sensor intensity", "Absolute difference (own scale)"),
                (peak, peak, None)):
            im = ax.imshow(data, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=vmax)
            ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{kind}: full x-y sensor at z={depth*1000:.1f} mm")
        fig.savefig(out/f"sensor_{kind}_z{depth*1000:07.2f}mm.png", dpi=150)
        theory.plt.close(fig)
    return {"z_m": depth, "scope": "full original N x N sensor window", **metrics}


def propagate_volume(fields: dict, axis: np.ndarray, z: np.ndarray,
                     x_indices: np.ndarray, y_indices: np.ndarray,
                     config: theory.Digit1Config, out: Path) -> tuple[dict, dict]:
    """全x-y面をASM伝搬し、[z,y,x]のROI複素値と代表深度の全センサー面を記録。

    旧row()のy=0専用縮約は使わない。1層の伝搬像を複製して3D化しない。
    メモリ量を抑えるため、全N x N x Nzの場ではなくROIと代表面を保存する。
    """
    tic = time.perf_counter()
    plan = theory.AngularSpectrum(config.grid_size, config.pixel_pitch_m,
                                  config.wavelength_m, config.padding_factor, config.angular_bandlimit)
    spectra = {name: plan.spectrum(field) for name, field in fields.items()}
    volumes = {name: np.empty((len(z), len(y_indices), len(x_indices)), dtype=np.complex64)
               for name in fields}
    snapshots = {int(np.argmin(np.abs(z-depth))) for depth in config.sensor_z_m}
    sensor_metrics, bandlimit_metrics = [], []
    roi = np.ix_(y_indices, x_indices)
    for index, depth in enumerate(z):
        transfer, valid = plan.transfer(float(depth))
        frames = {}
        lost = {}
        for name, spectrum in spectra.items():
            frame = plan.frame(spectrum, transfer)
            volumes[name][index] = frame[roi]
            if index in snapshots:
                frames[name] = frame
                power = np.abs(spectrum)**2
                denominator = float(power.sum(dtype=np.float64))
                lost[name] = (float(np.sum(power, where=~valid, dtype=np.float64)/denominator)
                              if denominator else None)
        if index in snapshots:
            sensor_metrics.append(_save_sensor(out, frames, axis, float(depth)))
            bandlimit_metrics.append({"z_m": float(depth), "discarded_spectral_power_fraction": lost})
        print(f"Full x-y propagation: {index+1}/{len(z)}, z={depth*1000:.1f} mm", flush=True)
    return volumes, {"four_field_propagation_and_sensor_figures_seconds": time.perf_counter()-tic,
                     "full_sensor_metrics": sensor_metrics, "bandlimit": bandlimit_metrics,
                     "padding_convergence": "not checked; repeat with larger padding_factor",
                     "array_order": "[z,y,x]", "saved_volume_scope": "explicit x/y ROI"}


def _profile_stats(intensity: np.ndarray, y: np.ndarray, slab: np.ndarray) -> dict:
    profile = intensity.sum(axis=(0, 2), dtype=np.float64)
    active = profile >= .5*float(profile.max()) if profile.max() > 0 else np.zeros(len(y), bool)
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)))
    intervals = [[float(y[a]*1e6), float(y[b-1]*1e6)] for a, b in zip(edges[::2], edges[1::2])]
    total = float(profile.sum())
    inside = profile[slab]
    return {"half_max_support_intervals_um": intervals,
            "half_max_support_span_um": float((y[active][-1]-y[active][0])*1e6) if active.any() else None,
            "out_of_slab_power_fraction_in_saved_roi": float(profile[~slab].sum()/total) if total else None,
            "integrated_y_profile_cv_inside_slab": float(inside.std()/inside.mean()) if inside.size and inside.mean() else None,
            "note": "Integrated over saved x/z ROI. Multiple peaks are not a single FWHM."}


def _save_volume_figures(out: Path, volumes: dict, target: np.ndarray,
                         x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         sheet_y: np.ndarray, slab: np.ndarray, stack: MultiSheetConfig) -> None:
    images = {name: np.abs(value)**2 for name, value in volumes.items()}
    ref, rec = images["direct_cgh"], images["proposed_cgh"]
    peak = max(float(target.max()), float(ref.max()), float(rec.max()), np.finfo(float).tiny)
    # 最大値投影は形状の概要だけ。評価値は投影前の全ROIボクセルから計算する。
    fig, axes = theory.plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    transforms = ((lambda v: v.max(axis=1).T, [z[0]*1e3, z[-1]*1e3, x[0]*1e3, x[-1]*1e3], "z [mm]", "x [mm]", "max over y"),
                  (lambda v: v.max(axis=2).T, [z[0]*1e3, z[-1]*1e3, y[0]*1e6, y[-1]*1e6], "z [mm]", "y [um]", "max over x"),
                  (lambda v: v.max(axis=0), [x[0]*1e3, x[-1]*1e3, y[0]*1e6, y[-1]*1e6], "x [mm]", "y [um]", "max over z"))
    for row, (transform, extent, xlabel, ylabel, label) in enumerate(transforms):
        for col, (value, title) in enumerate(((target, "Desired extruded 123"), (ref, "Direct CGH replay"), (rec, "Proposed CGH replay"))):
            ax = axes[row, col]
            artist = ax.imshow(transform(value), origin="lower", aspect="auto", extent=extent,
                               cmap="gray", vmin=0, vmax=peak)
            ax.set(title=f"{title}: {label}", xlabel=xlabel, ylabel=ylabel)
            fig.colorbar(artist, ax=ax, label="Intensity, shared scale")
    fig.savefig(out/"volume_projections.png", dpi=160)
    theory.plt.close(fig)
    fig, axes = theory.plt.subplots(len(sheet_y), 4, figsize=(18, 3.2*len(sheet_y)),
                                   squeeze=False, constrained_layout=True)
    extent = [z[0]*1e3, z[-1]*1e3, x[0]*1e3, x[-1]*1e3]
    for row, requested in enumerate(sheet_y):
        index = int(np.argmin(np.abs(y-requested)))
        values = (target[:, index].T, ref[:, index].T, rec[:, index].T,
                  np.abs(rec[:, index]-ref[:, index]).T)
        for col, (value, title) in enumerate(zip(values, ("Target", "Direct CGH replay", "Proposed CGH replay", "Absolute difference (own scale)"))):
            artist = axes[row, col].imshow(value, origin="lower", extent=extent, aspect="auto",
                                          cmap="gray", vmin=0, vmax=peak if col < 3 else None)
            axes[row, col].set(title=f"{title}\ny={y[index]*1e6:.1f} um (requested {requested*1e6:.1f})",
                              xlabel="z [mm]", ylabel="x [mm]")
            fig.colorbar(artist, ax=axes[row, col])
    fig.savefig(out/"sheets_at_each_y.png", dpi=150)
    theory.plt.close(fig)
    fig, ax = theory.plt.subplots(figsize=(10, 5), constrained_layout=True)
    profiles = {name: value.sum(axis=(0, 2), dtype=np.float64) for name, value in
                (("Desired slab", target), ("Direct CGH replay", ref), ("Proposed CGH replay", rec))}
    scale = max(float(profiles["Direct CGH replay"].max()), np.finfo(float).tiny)
    for name, profile in profiles.items():
        ax.plot(y*1e6, profile/scale, label=name)
    for position in sheet_y:
        ax.axvline(position*1e6, color="gray", alpha=.3, linestyle="--")
    ax.set(xlabel="y [um]", ylabel="Integrated x/z intensity / direct peak",
           title="Thickness profile: one common scale, no individual brightness fitting")
    ax.legend()
    fig.savefig(out/"thickness_profile.png", dpi=160)
    theory.plt.close(fig)
    # 点群は表示だけを間引く。最大値投影や閾値を定量評価には使用しない。
    fig = theory.plt.figure(figsize=(18, 6), constrained_layout=True)
    for col, (value, title) in enumerate(((target, "Target slab"), (ref, "Direct CGH replay"), (rec, "Proposed CGH replay")), 1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        # 目標の幾何形状に再生強度の閾値を掛けると、目標だけ空白になり得る。
        indices = np.argwhere(value > 0 if col == 1 else value >= stack.scatter_threshold_fraction*peak)
        if len(indices) > stack.scatter_max_points:
            indices = indices[np.linspace(0, len(indices)-1, stack.scatter_max_points, dtype=int)]
        if len(indices):
            iz, iy, ix = indices.T
            ax.scatter(z[iz]*1e3, x[ix]*1e3, y[iy]*1e6, s=2,
                       c=value[iz, iy, ix], cmap="gray", vmin=0, vmax=peak, alpha=.35)
        ax.set(xlabel="z [mm]", ylabel="x [mm]", zlabel="y [um]", title=title)
        ax.set_xlim(z[0]*1e3, z[-1]*1e3)
        ax.set_ylim(x[0]*1e3, x[-1]*1e3)
        ax.set_zlim(y[0]*1e6, y[-1]*1e6)
        ax.set_box_aspect((2, 1, 1))
    fig.suptitle(f"Target geometry; replay points above shared {100*stack.scatter_threshold_fraction:.0f}% threshold; axes stretched")
    fig.savefig(out/"volume_3d.png", dpi=160)
    theory.plt.close(fig)
    # 黒い立方体背景・白い外表面。評価値や入力光場は変更しない。
    from lsh_multisheet_3d import save_3d_surfaces
    save_3d_surfaces(out/"3d_surface", volumes, target, x, y, z,
                     threshold_fraction=stack.scatter_threshold_fraction)


def run_multisheet(reference_dir: str | Path, output_dir: str | Path,
                   stack: MultiSheetConfig | None = None,
                   coefficient_path: str | Path | None = None) -> dict:
    stack = stack or MultiSheetConfig()
    root, out = Path(reference_dir), Path(output_dir)
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if source.get("study") != "digit1_direct_vs_projected_shared_core" or source.get("basis_strategy") != "shift_aware":
        raise ValueError("Use the saved shift_aware direct/proposed comparison as the coefficient source")
    base_config = theory.Digit1Config(**source["config"])
    config = replace(base_config, z_samples=stack.z_samples,
                     padding_factor=base_config.padding_factor if stack.padding_factor is None else stack.padding_factor)
    if config.target_digits != "123" or not config.include_cgh_replay:
        raise ValueError("This initial study repeats the same 123 sheet and requires CGH replay")
    if stack.parent_rank > config.grid_size or source.get("library_parent_rank", source["rank"]) != stack.parent_rank:
        raise ValueError("parent_rank must match the source dictionary rank and fit the grid")
    with np.load(root/"comparison_data.npz", allow_pickle=False) as saved:
        # 保存済みdirect場/伝搬画像は読み出さない。
        coefficients, thread_x = saved["coefficients"].copy(), saved["thread_x_m"].copy()
    design = proposed.mode_design_for_spacing(config, source.get("mode_spacing_factor", 1))
    if (coefficients.shape != (len(design.m), config.thread_count)
            or not np.all(np.isfinite(coefficients)) or not np.any(coefficients)):
        raise ValueError("source coefficients must match the modes and contain finite nonzero values")
    intensity_bandwidth = float(np.ptp(design.kz_per_m)/(2*np.pi))
    dz = config.design_length_m/(config.z_samples-1)
    if intensity_bandwidth > 0 and 2*intensity_bandwidth*dz > 1+1e-10:
        raise ValueError("z_samples is too small for the analytic Bessel intensity bandwidth; increase it before running")
    expected_x = (np.arange(config.thread_count)-(config.thread_count-1)/2)*config.thread_spacing_m
    if thread_x.shape != expected_x.shape or not np.allclose(thread_x, expected_x, rtol=1e-12, atol=1e-15):
        raise ValueError("source Thread positions differ from the configuration")
    sheet_y, gains = stack.y_positions(), stack.sheet_weights()
    coefficient_source = "saved single-sheet coefficients, copied to all y sheets; no 3D refit"
    if coefficient_path is not None:
        # 設計した係数のみを両手法へ同じ条件で渡す。設計時の再生場は使わない。
        with np.load(coefficient_path, allow_pickle=False) as fitted:
            required = {"coefficients", "thread_x_m", "sheet_y_m", "kr_per_m", "m", "config_json"}
            if not required.issubset(fitted.files):
                raise ValueError("Incomplete 3D coefficient file")
            if json.loads(str(fitted["config_json"].item())) != theory.config_payload(base_config):
                raise ValueError("3D coefficient source settings differ")
            for key, expected in (("thread_x_m", thread_x), ("sheet_y_m", sheet_y),
                                  ("kr_per_m", design.kr_per_m), ("m", design.m)):
                value = fitted[key]
                if value.shape != expected.shape or not np.allclose(value, expected, rtol=1e-12, atol=1e-15):
                    raise ValueError(f"3D coefficient coordinates/modes differ: {key}")
            coefficients = fitted["coefficients"].copy()
        if (coefficients.shape != (len(design.m), len(sheet_y), len(thread_x))
                or not np.all(np.isfinite(coefficients)) or not np.any(coefficients)):
            raise ValueError("Invalid 3D coefficients")
        gains = np.ones(len(sheet_y), dtype=complex)  # 位相と振幅は係数に含まれる。
        coefficient_source = f"joint 3D amplitude design: {Path(coefficient_path).resolve()}"
    slab_min, slab_max = float(sheet_y.min()-stack.sheet_spacing_m/2), float(sheet_y.max()+stack.sheet_spacing_m/2)
    if (max(abs(slab_min), abs(slab_max)) >= stack.evaluation_y_half_width_m
            or stack.evaluation_y_half_width_m >= config.grid_size*config.pixel_pitch_m/2
            or np.hypot(np.abs(thread_x).max(), np.abs(sheet_y).max()) >= config.aperture_radius_m):
        raise ValueError("sheet centres/slab/ROI must fit within the aperture and numerical window")
    if stack.training_y_shifts_m is not None and max(abs(v) for v in stack.training_y_shifts_m) >= config.grid_size*config.pixel_pitch_m/2:
        raise ValueError("training shifts must fit inside the numerical window")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    settings = {"schema_version": SCHEMA_VERSION, "reference_dir": str(root.resolve()),
                "config": theory.config_payload(config), "stack": asdict(stack),
                "coefficient_source": coefficient_source}
    theory._write_json(out/"settings.json", settings)
    parent_path = Path(source["library_path"])
    if not parent_path.is_file():
        parent_path = root.parent/parent_path.name
    parent, parent_info = proposed.load_or_build_library(parent_path, config, stack.parent_rank,
        source["seed"], source.get("mode_spacing_factor", 1), "shift_aware")
    if coefficients.shape[0] != len(parent["kr"]):
        raise ValueError("dictionary/coefficients have different mode counts")
    training_y = sorted(set((sheet_y.tolist() if stack.training_y_shifts_m is None
                             else list(stack.training_y_shifts_m))+[0.0]))
    signature = hashlib.sha256(json.dumps({"parent_spec": parent_info["spec"],
                                         "y": training_y}, sort_keys=True).encode()).hexdigest()[:16]
    path = out.parent/f"lsh_multisheet_library_rank{stack.parent_rank}_{signature}.npz"
    library, offline = load_or_build_xy_library(path, parent, parent_info["spec"], config, stack, source["seed"])
    del parent
    tic = time.perf_counter()
    library = proposed.truncate_library(library, stack.rank)
    offline["one_time_packing_seconds"] = time.perf_counter()-tic
    offline["parent_dictionary"] = parent_info
    offline["trained_at_used_y_positions"] = [bool(np.any(np.isclose(y, training_y, rtol=0, atol=1e-15))) for y in sheet_y]
    factors = out/"factors"
    factors.mkdir()
    _save_factor_figures(factors, library["uy"], library["ux"], library["cores"], design)
    axis = (np.arange(config.grid_size)-config.grid_size//2)*config.pixel_pitch_m
    illumination = (np.hypot(axis[:, None], axis[None, :]) <= config.aperture_radius_m).astype(float)
    if config.illumination_waist_m is not None:
        illumination *= np.exp(-(np.hypot(axis[:, None], axis[None, :])/config.illumination_waist_m)**2)
    def direct_call():
        tic = time.perf_counter()
        field = direct_multisheet_field(axis, thread_x, sheet_y, library["kr"], coefficients, gains)
        return field, {"field_generation_seconds": time.perf_counter()-tic}
    direct, direct_time = _measure_field(direct_call, stack)
    candidate, proposed_time = _measure_field(lambda: proposed_multisheet_field(
        library, coefficients, thread_x, sheet_y, gains, config.pixel_pitch_m), stack)
    # 比較用に同じ定数を適用する。空間的補正・位相合わせ・個別輝度フィットはしない。
    scale = float(np.max(np.abs(direct)[illumination > 0]))
    if scale <= 0:
        raise ValueError("Zero direct field")
    direct /= scale
    candidate /= scale
    initial = _field_metrics(direct*illumination, candidate*illumination, .01)
    passed = (initial["complex_relative_l2"] <= stack.max_initial_complex_error
              and initial["intensity_relative_l2"] <= stack.max_initial_intensity_error)
    common = {"settings": settings, "offline": offline, "rank": stack.rank,
              "sheet_y_m": sheet_y.tolist(), "training_y_shifts_m": training_y,
              "centre_span_um": float(np.ptp(sheet_y)*1e6),
              "desired_slab_thickness_um": float((slab_max-slab_min)*1e6),
              "nominal_beam_terms": int(len(library["kr"])*len(thread_x)*len(sheet_y)),
              "nonzero_beam_terms": int(np.count_nonzero(coefficients)*(len(sheet_y) if coefficients.ndim == 2 else 1)),
              "axial_sampling": {"dz_m": dz, "analytic_intensity_bandwidth_cycles_per_m": intensity_bandwidth,
                                 "nyquist_checked": True,
                                 "note": "Necessary analytic-mode bandwidth check only; finite-aperture/axial convergence not established."},
              "normalization": {"shared_scale": scale, "source": "direct-field aperture maximum for comparison only",
                                "independent_brightness_or_phase_fit": False},
              "initial_field_metrics": initial, "initial_check_passed": passed,
              "coefficient_shape": list(coefficients.shape),
              "shared_sheet_core_shortcut": coefficients.ndim == 2,
              "timing": {"direct_field": direct_time, "proposed_field": proposed_time,
                         "note": "Both regenerate fields with identical repeats/warmups. Excludes dictionary preparation/loading, coefficient design, aperture, normalization, CGH and propagation. Default single-run exploratory timing; not actual SLM display latency."},
              "scope": "Coherent sum of parallel 123 sheet threads; one CGH; virtual 1:1 ideal 4f; physical SLM/1:4 mapping not applied. Coefficient source and shared-core shortcut recorded separately. Known training coordinates are explicitly recorded; no unseen-shift guarantee."}
    theory._write_json(out/"initial_summary.json", common)
    print(f"Initial field error={initial['complex_relative_l2']:.3%}; check passed={passed}", flush=True)
    print("Continuing diagnostic propagation even if the initial check failed.", flush=True)
    fields, codes, encodings, encoding_times = {}, {}, {}, {}
    for name, field in (("direct", direct), ("proposed", candidate)):
        encoding_field = np.where(illumination > 0, field, 0)
        fields[name+"_complex"] = (encoding_field*illumination).astype(np.complex64)
        tic = time.perf_counter()
        codes[name], replay, encodings[name] = theory.encode_and_replay(
            encoding_field, illumination, config, float(library["kr"].max()))
        encoding_times[name] = time.perf_counter()-tic
        fields[name+"_cgh"] = replay.astype(np.complex64)
        Image.fromarray(codes[name]).save(out/f"{name}_cgh_8bit.png")
    proposed._save_initial_comparison(out, fields["direct_complex"], fields["proposed_complex"], codes)
    np.savez_compressed(out/"initial_fields.npz", **fields, coefficients=coefficients,
                        thread_x_m=thread_x, sheet_y_m=sheet_y, sheet_weights=gains)
    del direct, candidate, library, encoding_field, replay
    z = np.linspace(0, config.design_length_m, config.z_samples)
    ix = np.flatnonzero(np.abs(axis) <= config.evaluation_x_half_width_m)
    iy = np.flatnonzero(np.abs(axis) <= stack.evaluation_y_half_width_m)
    x_roi, y_roi = axis[ix], axis[iy]
    volumes, propagation = propagate_volume(fields, axis, z, ix, iy, config, out)
    del fields
    slab = (y_roi >= slab_min) & (y_roi <= slab_max)
    # 目標は同じ123を各yセルへ押し出した一定強度の板。これは達成済み像ではない。
    target = (theory.digit_target(x_roi, z, config)[:, None, :]*slab[None, :, None]/scale**2).astype(np.float32)
    volume_metrics = {kind: _voxel_scores(volumes["direct_"+kind], volumes["proposed_"+kind])
                      for kind in ("complex", "cgh")}
    per_sheet = []
    for y in sheet_y:
        index = int(np.argmin(np.abs(y_roi-y)))
        ref, rec = volumes["direct_cgh"][:, index], volumes["proposed_cgh"][:, index]
        ref_i, rec_i, desired = np.abs(ref)**2, np.abs(rec)**2, target[:, index]
        per_sheet.append({"requested_y_m": float(y), "sampled_y_m": float(y_roi[index]),
            "proposed_vs_direct": {"complex_relative_l2": _relative_l2(ref, rec),
                "intensity_relative_l2": _relative_l2(ref_i, rec_i), **_image_metrics(ref_i, rec_i)},
            "direct_vs_target": {"intensity_relative_l2": _relative_l2(desired, ref_i), **_image_metrics(desired, ref_i)},
            "proposed_vs_target": {"intensity_relative_l2": _relative_l2(desired, rec_i), **_image_metrics(desired, rec_i)}})
    direct_i, proposed_i = np.abs(volumes["direct_cgh"])**2, np.abs(volumes["proposed_cgh"])**2
    target_metrics = {}
    for method in ("direct", "proposed"):
        paths = {}
        for kind in ("complex", "cgh"):
            value = np.abs(volumes[method+"_"+kind])**2
            total = float(value.sum(dtype=np.float64))
            paths[kind] = {"intensity_relative_l2": _relative_l2(target, value),
                           "off_target_power_fraction_in_saved_roi":
                               float(value[target == 0].sum(dtype=np.float64)/total) if total else None,
                           "thickness": _profile_stats(value, y_roi, slab)}
        # 既存ノートブックで読むキーも残し、CGH化前後の経路を追加する。
        target_metrics[method] = {"intensity_relative_l2": paths["cgh"]["intensity_relative_l2"], **paths}
    thickness = {name: _profile_stats(value, y_roi, slab) for name, value in
                 (("direct_cgh", direct_i), ("proposed_cgh", proposed_i))}
    np.savez_compressed(out/"volume_data.npz", x_m=x_roi, y_m=y_roi, z_m=z,
                        target_intensity=target, **volumes)
    _save_volume_figures(out, volumes, target, x_roi, y_roi, z, sheet_y, slab, stack)
    summary = {**common, "study": ("joint_designed_123_sheets_xy_projected_core" if coefficients.ndim == 3
                                   else "repeated_123_sheets_xy_projected_core"),
               "schema_version": SCHEMA_VERSION, "encoding": encodings,
               "cgh_encoding_and_ideal4f_seconds": encoding_times,
               "propagation": propagation, "volume_proposed_vs_direct": volume_metrics,
               "volume_vs_desired_slab": target_metrics, "per_sheet_cgh": per_sheet,
               "thickness": thickness,
               "volume_roi": {"x_m": [float(x_roi[0]), float(x_roi[-1])],
                              "y_m": [float(y_roi[0]), float(y_roi[-1])],
                              "z_m": [float(z[0]), float(z[-1])], "shape_zyx": list(target.shape)},
               "interpretation": "Initial acceptance does not ensure uniform thickness or target fidelity. Inspect direct-vs-target separately from proposed-vs-direct. Coherent interference between sheets is included; intensities are never added as if sheets were incoherent. MIP/scatter plots are only visualizations; voxel errors are unthresholded."}
    theory._write_json(out/"summary.json", summary)
    theory._write_json(out/"manifest.json", {"files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())})
    print(f"Saved multi-sheet study to: {out.resolve()}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default="lsh_digits123_rank_verified_outputs_rank512_parent512_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=450)
    parser.add_argument("--sheet-count", type=int, default=5)
    parser.add_argument("--sheet-spacing-um", type=float, default=75)
    parser.add_argument("--z-samples", type=int, default=101)
    parser.add_argument("--coefficient-path", help="optional joint 3D coefficient_design.npz")
    args = parser.parse_args()
    run_multisheet(args.reference_dir, args.output_dir, MultiSheetConfig(
        rank=args.rank, sheet_count=args.sheet_count, sheet_spacing_m=args.sheet_spacing_um*1e-6,
        z_samples=args.z_samples), coefficient_path=args.coefficient_path)


if __name__ == "__main__":
    main()
