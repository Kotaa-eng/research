"""123の直接設計を比較する。importでは計算しない。

解析シートで候補を絞り、その候補を別途有限開口・CGH伝搬で検証する。
解析的な良さは実機や有限開口での良さを保証しない。
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import lsh_digit1_theory as theory
from stage_a_validation import _image_metrics, _relative_l2

SCREEN_SCHEMA_VERSION = 1


def design_candidates(base: theory.Digit1Config | None = None) -> dict:
    """帯域・Thread密度・干渉を考慮した係数設計を順に変える。

    横方向のThread配置範囲は維持する。広帯域候補では有限開口を広げる。
    7µmは仮想4f出力面のサンプリングであり、実機SLMへの倍率校正ではない。
    """
    base = base or theory.Digit1Config(target_digits="123", digit_axes="zx")
    base = replace(base, coefficient_design="independent_threads")
    dense_spacing = base.thread_spacing_m/2
    dense = replace(base, thread_spacing_m=dense_spacing,
                    thread_count=2*(base.thread_count-1)+1)
    # 半径を小さくしてQを変え、単なるkrの密補間ではなく帯域を広げる。
    wide = replace(base, spot_radius_m=30e-6)
    design = theory.mode_design(wide)
    max_shift = (base.thread_count-1)*base.thread_spacing_m/2
    cone_tangent = float(np.max(design.kr_per_m/design.kz_per_m))
    # 全モードの円錐重なりを設計端Lまで維持するための幾何学的目安。
    needed = max_shift + base.design_length_m*cone_tangent
    aperture = max(base.aperture_radius_m, float(np.ceil(needed*1.05/1e-4)*1e-4))
    pitch = max(base.pixel_pitch_m, 7e-6)
    # 開口＋最大シフトを数値窓に収め、FFT周期端の混入を抑える余白を取る。
    minimum_n = int(np.ceil(2*(aperture+max_shift+2*pitch)/pitch))
    n = max(base.grid_size, int(np.ceil(minimum_n/256))*256)
    wide = replace(wide, grid_size=n, pixel_pitch_m=pitch,
                   aperture_radius_m=aperture, carrier_period_pixels=4)
    wide_dense = replace(wide, thread_spacing_m=dense_spacing,
                         thread_count=dense.thread_count)
    coupled = replace(wide_dense, coefficient_design="coupled_least_squares")
    return {
        "baseline": base,
        "finer_threads": dense,
        "wider_kr": wide,
        "wider_kr_finer_threads": wide_dense,
        "wide_dense_coupled": coupled,
    }


def design_report(config: theory.Digit1Config) -> dict:
    """実行前に波数・開口・サンプリングの整合性を確認する。"""
    modes = theory.mode_design(config)
    shift = (config.thread_count-1)*config.thread_spacing_m/2
    overlap = float(np.min((config.aperture_radius_m-shift)
                           * modes.kz_per_m/np.maximum(modes.kr_per_m, 1e-30)))
    radius = config.filter_radius_factor*float(modes.kr_per_m.max())
    carrier = 2*np.pi/(config.carrier_period_pixels*config.pixel_pitch_m)
    nyquist = np.pi/config.pixel_pitch_m
    if radius >= min(carrier, nyquist-carrier):
        raise ValueError("CGH carrier/filter are incompatible with the new kr band")
    return {
        "modes_per_thread": modes.count,
        "thread_count": config.thread_count,
        "thread_spacing_um": config.thread_spacing_m*1e6,
        "central_first_zero_radius_um": config.spot_radius_m*1e6,
        "kr_min_per_m": float(modes.kr_per_m.min()),
        "kr_max_per_m": float(modes.kr_per_m.max()),
        "kz_bandwidth_per_m": float(np.ptp(modes.kz_per_m)),
        "aperture_radius_mm": config.aperture_radius_m*1000,
        "grid_size": config.grid_size,
        "output_plane_pitch_um": config.pixel_pitch_m*1e6,
        "geometric_all_mode_overlap_estimate_mm": overlap*1000,
        "design_end_mm": config.design_length_m*1000,
        "coefficient_design": config.coefficient_design,
        "offline_spectra_temporary_gib": modes.count*config.grid_size**2*16/1024**3,
        "overlap_note": "geometric estimate only; check finite-aperture propagation and padding",
    }


def analytic_screen(output_dir: str | Path, candidates: dict,
                    x_samples: int = 301, z_samples: int = 251) -> dict:
    """2D初期場・FFT・圧縮なしで候補の解析的な強度再現性を比較する。

    個別ピーク正規化や最適ゲイン合わせをしない。黒い背景を含む誤差の
    ほか、目標が明るい領域のみの誤差も記録する。
    """
    if not candidates or x_samples < 3 or z_samples < 3:
        raise ValueError("provide candidates and at least 3 samples per axis")
    out = Path(output_dir)
    expected = {name: theory.config_payload(config) for name, config in candidates.items()}
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        summary_path = out/"summary.json"
        if not summary_path.is_file() or not (out/"manifest.json").is_file():
            raise FileExistsError(f"Incomplete output: {out}; choose a new output_dir")
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if (saved.get("schema_version") != SCREEN_SCHEMA_VERSION
                or saved.get("configs") != expected
                or saved.get("sampling") != {"x_samples": x_samples, "z_samples": z_samples}):
            raise ValueError(f"Screen settings differ: {out}; choose a new output_dir")
        return saved
    # 同じ目標・評価窓を全候補で使う。物理範囲の違う画像を比較しない。
    first = next(iter(candidates.values()))
    target_keys = ("target_digits", "digit_axes", "digit_gap_fraction", "digit_width_m",
                   "digit_z_start_m", "digit_z_end_m", "design_length_m",
                   "evaluation_x_half_width_m")
    if any(any(getattr(config, key) != getattr(first, key) for key in target_keys)
           for config in candidates.values()):
        raise ValueError("All candidates must share the target and evaluation coordinates")
    out.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-first.evaluation_x_half_width_m, first.evaluation_x_half_width_m, x_samples)
    z = np.linspace(0, first.design_length_m, z_samples)
    target = theory.digit_target(x, z, first)
    foreground = target > 0
    rows, images = [], {}
    for name, config in candidates.items():
        start = time.perf_counter()
        design = theory.mode_design(config)
        positions = ((np.arange(config.thread_count)-(config.thread_count-1)/2)
                     * config.thread_spacing_m)
        weights = theory.target_coefficients(config, positions, design)
        coefficient_seconds = time.perf_counter()-start
        field = theory.analytic_sheet(x, z, positions, design, weights)
        intensity = np.abs(field)**2
        images[name] = intensity
        row = {"case": name, **design_report(config),
               "intensity_relative_l2": _relative_l2(target, intensity),
               "foreground_intensity_relative_l2": _relative_l2(target[foreground], intensity[foreground]),
               **_image_metrics(target, intensity),
               "background_mean_intensity": float(np.mean(intensity[~foreground])),
               "coefficient_l2": float(np.linalg.norm(weights)),
               "coefficient_design_seconds": coefficient_seconds,
               "screen_seconds": time.perf_counter()-start}
        rows.append(row)
        print(f"Analytic screen {name}: target intensity error={row['intensity_relative_l2']:.3%}", flush=True)
    best = min(rows, key=lambda row: row["intensity_relative_l2"])["case"]
    summary = {"schema_version": SCREEN_SCHEMA_VERSION, "study": "digits123_analytic_design_screen",
               "configs": expected, "sampling": {"x_samples": x_samples, "z_samples": z_samples},
               "cases": rows, "best_analytic_case": best,
               "selection": "minimum intensity relative L2, WITHOUT fitted brightness gain",
               "scope": "infinite-aperture y=0 analytic screening only; confirm finite complex and CGH replay next"}
    np.savez_compressed(out/"screen_data.npz", x_m=x, z_m=z, target=target, **images)
    theory._write_json(out/"summary.json", summary)
    with (out/"design_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = theory.plt.subplots(len(rows), 2, figsize=(14, 3.5*len(rows)),
                                    constrained_layout=True, squeeze=False)
    roi = np.ones(len(x), dtype=bool)
    for index, row in enumerate(rows):
        name = row["case"]
        peak = max(1., float(images[name].max()))
        for ax, value, title in zip(axes[index], (target, images[name]),
                                   ("Digit 123 target", name)):
            display, extent, xlabel, ylabel = theory.sheet_display(value, x, z, roi, first.digit_axes)
            im = ax.imshow(display, origin="lower", extent=extent, aspect="auto", cmap="gray",
                           vmin=0, vmax=peak)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            fig.colorbar(im, ax=ax, label="Intensity (same scale within this row)")
    fig.savefig(out/"analytic_design_comparison.png", dpi=130)
    theory.plt.close(fig)
    theory._write_json(out/"manifest.json", {"files": sorted(
        {"manifest.json"} | {p.name for p in out.iterdir() if p.is_file()})})
    return summary


def saved_replay_scores(result_dir: str | Path, x_samples: int = 201,
                        z_samples: int = 101) -> list[dict]:
    """保存済みの再生強度を共通座標へ補間して評価。伝搬は再実行しない。

    既知の場の正規化定数のみを戻し、目標強度1の単位で比較する。
    最適ゲイン合わせは行わない。実験での同じ入射パワーの比較ではない。
    """
    folder = Path(result_dir)
    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if summary.get("study") != "digit1_direct_vs_projected_shared_core":
        raise ValueError("Expected direct/proposed CGH comparison results")
    config = theory.Digit1Config(**summary["config"])
    if x_samples < 3 or z_samples < 3:
        raise ValueError("need >=3 common samples per axis")
    x_common = np.linspace(-config.evaluation_x_half_width_m,
                           config.evaluation_x_half_width_m, x_samples)
    z_common = np.linspace(0, config.design_length_m, z_samples)
    target = theory.digit_target(x_common, z_common, config)
    foreground = target > 0
    records = []
    with np.load(folder/"comparison_data.npz", allow_pickle=False) as saved:
        x_saved, z_saved = saved["x_m"], saved["z_m"]
        for method in ("direct", "proposed"):
            image = saved[method+"_cgh"] * summary["reference_scale"]**2
            along_x = np.stack([np.interp(x_common, x_saved, row) for row in image])
            shared = np.column_stack([np.interp(z_common, z_saved, column)
                                      for column in along_x.T])
            records.append({"result_dir": str(folder.resolve()), "method": method,
                "intensity_relative_l2": _relative_l2(target, shared),
                "foreground_intensity_relative_l2": _relative_l2(target[foreground], shared[foreground]),
                **_image_metrics(target, shared),
                "background_mean_intensity": float(np.mean(shared[~foreground])),
                "first_order_power_fraction_of_incident": summary["encoding"][method]["first_order_power_fraction_of_incident"],
                "common_x_samples": x_samples, "common_z_samples": z_samples,
                "sampling": "linear interpolation of saved intensity, no new propagation",
                "gain": "known reference_scale squared restored; no fitted gain"})
    return records
