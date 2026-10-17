"""旧方式（同軸コア加算のみ）と現行のシフトコア加算法のオンライン時間。

保存済みの同じrank80表現を使用し、辞書を新規学習しない簡易比較。
旧方式はFFTを呼ばず、座標を反映済みの空間因子でThreadを復元して画像加算。
固定座標の事前準備を両方式に認め、提案法もシフト行列を事前準備する。
提案法の行列をオンライン計算する場合も参考として併記する。
旧方式の元の空間Tucker学習を再現するオフライン比較ではない。
CGH・伝搬・保存画像の生成はしない。実行はrun_benchmarkまたはCLIのみ。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import lsh_digit1_proposed as proposed
from stage_a_validation import _image_metrics, _x_shift_core_operator


def image_quality(direct: np.ndarray, fields: dict) -> dict:
    """CGH化前の全2048²画像。共通尺度の振幅・強度を別々に評価する。"""
    pairs = {"legacy_vs_direct": (direct, fields["legacy"]),
             "proposed_vs_direct": (direct, fields["proposed"]),
             "proposed_vs_legacy": (fields["legacy"], fields["proposed"])}
    if "proposed_online" in fields:
        pairs["proposed_online_vs_direct"] = (direct, fields["proposed_online"])
    metrics = {}
    for name, (reference, candidate) in pairs.items():
        reference_amp, candidate_amp = np.abs(reference), np.abs(candidate)
        metrics[name] = {"amplitude": _image_metrics(reference_amp, candidate_amp),
                         "intensity": _image_metrics(reference_amp**2, candidate_amp**2)}
    return metrics


def save_quality(directory: Path, metrics: dict) -> None:
    with (directory/"quality_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "image", "ssim", "psnr_db", "snr_db"])
        writer.writeheader()
        writer.writerows({"comparison": name, "image": image, **score}
                         for name, values in metrics.items() for image, score in values.items())
    print("\nCGH化前の強度画像（全画素）: 比較 | SSIM | PSNR[dB]")
    for name, values in metrics.items():
        score = values["intensity"]
        print(f"{name} | {score['ssim']:.6f} | {score['psnr_db']:.3f}")


def shift_x_linear(field: np.ndarray, displacement_pixels: float) -> np.ndarray:
    """FFTなしの実空間x移動。範囲外は0、非整数画素は線形補間。

    E_shift(x)=E(x-dx)。np.rollによる整数丸めを使わない。
    """
    if displacement_pixels == 0:
        return field
    source = np.arange(field.shape[1])-displacement_pixels
    valid = (source >= 0) & (source <= field.shape[1]-1)
    lower = np.floor(source[valid]).astype(int)
    upper = np.minimum(lower+1, field.shape[1]-1)
    fraction = source[valid]-lower
    shifted = np.zeros_like(field)
    shifted[:, valid] = ((1-fraction)[None, :]*field[:, lower]
                         + fraction[None, :]*field[:, upper])
    return shifted


def prepare_placed_factors(library: dict, positions: np.ndarray, pitch: float) -> np.ndarray:
    """従来法の準備：各座標を空間因子Vxの行に反映する。FFTを呼ばない。

    線形移動Tについて T(Vy H Vx.T)=Vy H (T Vx).T。
    復元後の画像補間と同じ結果を、座標入り因子から直接復元できる。
    """
    return np.stack([shift_x_linear(library["vx"].T, float(x/pitch)).T for x in positions])


def legacy_field(library: dict, coefficients: np.ndarray,
                 positions: np.ndarray, pitch: float,
                 placed_factors: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """各Threadを同軸コア加算→座標入り因子で復元→フル画像加算。"""
    if placed_factors is None:
        # 準備はオンラインタイマーの前。run_benchmarkでは一度だけ用意する。
        placed_factors = prepare_placed_factors(library, positions, pitch)
    tic_total = perf_counter()
    matrix = library["cores"].reshape(len(library["cores"]), -1)
    field = np.zeros((len(library["vy"]), len(library["vx"])), dtype=np.complex128)
    times = {"core_weighting_seconds": 0.0, "reconstruction_seconds": 0.0,
             "shift_seconds": 0.0, "summation_seconds": 0.0}
    for q in np.flatnonzero(np.any(coefficients != 0, axis=0)):
        tic = perf_counter()
        core = (coefficients[:, q] @ matrix).reshape(library["cores"].shape[1:])
        times["core_weighting_seconds"] += perf_counter()-tic
        tic = perf_counter()
        thread = (library["vy"] @ core) @ placed_factors[q].T
        times["reconstruction_seconds"] += perf_counter()-tic
        tic = perf_counter()
        field += thread
        times["summation_seconds"] += perf_counter()-tic
    times["field_generation_seconds"] = perf_counter()-tic_total
    return field, times


def proposed_cached_field(library: dict, coefficients: np.ndarray,
                           operators: np.ndarray) -> tuple[np.ndarray, dict]:
    """同じ固定座標についてAxを事前準備した提案法。最後の復元は一度。"""
    tic_total = perf_counter()
    matrix = library["cores"].reshape(len(library["cores"]), -1)
    core_sum = np.zeros(library["cores"].shape[1:], dtype=np.complex128)
    times = {"core_weighting_seconds": 0.0, "reconstruction_seconds": 0.0,
             "shift_seconds": 0.0, "summation_seconds": 0.0}
    for q in np.flatnonzero(np.any(coefficients != 0, axis=0)):
        tic = perf_counter()
        core = (coefficients[:, q] @ matrix).reshape(core_sum.shape)
        times["core_weighting_seconds"] += perf_counter()-tic
        tic = perf_counter()
        core_sum += core @ operators[q]
        times["summation_seconds"] += perf_counter()-tic
    tic = perf_counter()
    field = (library["vy"] @ core_sum) @ library["vx"].T
    times["reconstruction_seconds"] = perf_counter()-tic
    times["field_generation_seconds"] = perf_counter()-tic_total
    return field, times


def run_benchmark(reference_dir: str | Path = "lsh_digits123_rank_verified_outputs_rank512_parent512_v1",
                  output_dir: str | Path = "legacy_vs_proposed_rank80_outputs",
                  rank: int = 80, repeats: int = 3, warmups: int = 0) -> dict:
    """準備後に同じ入力を交互に測定。画質評価はタイマー外。"""
    if (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
            or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1
            or isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0):
        raise ValueError("rank/repeats must be positive integers; warmups >= 0")
    root, out = Path(reference_dir), Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new empty output directory: {out}")
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if source.get("study") != "digit1_direct_vs_projected_shared_core":
        raise ValueError("Use a saved single-sheet direct/proposed comparison")
    config = source["config"]
    preparation_start = perf_counter()
    with np.load(root/"comparison_data.npz", allow_pickle=False) as saved:
        coefficients, positions = saved["coefficients"].copy(), saved["thread_x_m"].copy()
    path = Path(source["library_path"])
    if not path.is_file():
        path = root.parent/path.name
    with np.load(path, allow_pickle=False) as saved:
        parent = {key: saved[key].copy() for key in ("uy", "ux", "vy", "vx", "cores", "m", "kr")}
    if rank > parent["cores"].shape[1]:
        raise ValueError("rank exceeds the saved parent rank")
    if (parent["cores"].shape[0] != len(source["kr_per_m"])
            or not np.allclose(parent["kr"], source["kr_per_m"], rtol=1e-12, atol=0)
            or coefficients.shape != (len(parent["cores"]), len(positions))
            or any(parent[key].shape[0] != config["grid_size"] for key in ("uy", "ux", "vy", "vx"))
            or not np.all(np.isfinite(coefficients))):
        raise ValueError("Dictionary/input dimensions or modes differ")
    library = proposed.truncate_library(parent, rank)
    del parent
    preparation_seconds = perf_counter()-preparation_start
    pitch = config["pixel_pitch_m"]
    tic = perf_counter()
    placed = prepare_placed_factors(library, positions, pitch)
    placed_preparation_seconds = perf_counter()-tic
    tic = perf_counter()
    kx = 2*np.pi*np.fft.fftfreq(len(library["ux"]), d=pitch)
    operators = np.stack([_x_shift_core_operator(library["ux"], np.exp(-1j*kx*x)) for x in positions])
    operator_preparation_seconds = perf_counter()-tic
    methods = ("legacy", "proposed", "proposed_online")

    def generate(method: str) -> tuple[np.ndarray, dict]:
        if method == "legacy":
            return legacy_field(library, coefficients, positions, pitch, placed)
        if method == "proposed":
            return proposed_cached_field(library, coefficients, operators)
        field, parts, _ = proposed.proposed_field(library, coefficients, positions, pitch)
        return field, {"core_weighting_seconds": parts["thread_core_weighting_seconds"],
                       "reconstruction_seconds": parts["spatial_factor_reconstruction_seconds"],
                       "shift_seconds": parts["shift_operator_seconds"],
                       "summation_seconds": parts["shifted_core_sum_seconds"],
                       "field_generation_seconds": parts["field_generation_seconds"]}

    for _ in range(warmups):
        for method in methods:
            field, _ = generate(method)
            del field
    samples = {method: [] for method in methods}
    fields = {}
    for repeat in range(repeats):
        order = methods[repeat % len(methods):]+methods[:repeat % len(methods)]
        for method in order:
            # 同じ方式の前の出力を解放。最後の1回だけ比較用の場を保持する。
            fields.pop(method, None)
            field, parts = generate(method)
            samples[method].append(parts)
            if repeat == repeats-1:
                fields[method] = field
            else:
                del field
            print(f"{method} {repeat+1}/{repeats}: {parts['field_generation_seconds']:.4f} s", flush=True)
    statistics = {method: {key: median(s[key] for s in records) for key in records[0]}
                  for method, records in samples.items()}
    # タイマー外で同じ開口・照明・尺度を適用。保存直接場は評価だけに使う。
    with np.load(root/"initial_fields.npz", allow_pickle=False) as saved:
        direct = saved["direct_complex"].copy()
    axis = (np.arange(config["grid_size"])-config["grid_size"]//2)*pitch
    radius = np.hypot(axis[:, None], axis[None, :])
    illumination = (radius <= config["aperture_radius_m"]).astype(float)
    if config["illumination_waist_m"] is not None:
        illumination *= np.exp(-(radius/config["illumination_waist_m"])**2)
    for method in fields:
        fields[method] *= illumination/source["reference_scale"]
    norm = float(np.linalg.norm(direct))
    errors = {method: float(np.linalg.norm(field-direct)/norm) for method, field in fields.items()}
    old_norm = float(np.linalg.norm(fields["legacy"]))
    inter_method_error = (float(np.linalg.norm(fields["proposed"]-fields["legacy"])/old_norm)
                          if old_norm else None)
    quality = image_quality(direct, fields)  # 評価は場生成タイマー外。
    data = {"study": "legacy_coaxial_restore_placed_factors_vs_projected_core_shift",
            "implementation_version": 2,
            "rank": rank, "grid_size": config["grid_size"], "target_digits": config["target_digits"],
            "modes_per_thread": len(coefficients), "thread_count": len(positions),
            "active_thread_count": int(np.count_nonzero(np.any(coefficients != 0, axis=0))),
            "nonzero_beam_terms": int(np.count_nonzero(coefficients)),
            "thread_spacing_m": config["thread_spacing_m"], "spot_radius_m": config["spot_radius_m"],
            "repeats": repeats, "warmups": warmups, "reference_dir": str(root.resolve()),
            "library_path": str(path.resolve()), "shared_library_for_both_methods": True,
            "dictionary_load_and_packing_seconds": preparation_seconds,
            "coordinate_preparation_seconds": {"legacy_placed_factors": placed_preparation_seconds,
                                               "proposed_shift_operators": operator_preparation_seconds},
            "coordinate_cache_mib": {"legacy_placed_factors": placed.nbytes/1024**2,
                                     "proposed_shift_operators": operators.nbytes/1024**2},
            "statistics_seconds": statistics, "samples_seconds": samples,
            "stage_definitions": "shift_seconds is online coordinate/factor or operator preparation; zero for both prepared methods. Proposed summation includes core translation matrix multiplication and addition. Legacy summation is full-image addition.",
            "speedup_legacy_over_proposed": statistics["legacy"]["field_generation_seconds"]/
                                             statistics["proposed"]["field_generation_seconds"],
            "speedup_legacy_over_proposed_online": statistics["legacy"]["field_generation_seconds"]/
                                                    statistics["proposed_online"]["field_generation_seconds"],
            "complex_relative_l2_vs_saved_direct": errors,
            "complex_relative_l2_proposed_vs_legacy": inter_method_error,
            "image_metrics": quality,
            "image_metrics_scope": "Full-grid pre-CGH amplitude/intensity; shared scale, reference peak PSNR, 7x7 SSIM. Not CGH replay or target-digit quality.",
            "dictionary_array_mib": sum(library[key].nbytes for key in ("uy", "ux", "vy", "vx", "cores"))/1024**2,
            "scope": "Field generation only. Fixed coordinates: legacy spatial factors and proposed Ax operators prepared once outside online timers. Legacy reconstructs each placed Thread and sums full images; no online spatial shift. proposed_online separately builds Ax online. Common saved rank-r representation; not the original spatial Tucker offline fit. Legacy coordinate preparation uses linear interpolation, proposed Fourier projection; inspect errors. No CGH/propagation.",
            "quality_note": "Rank80 is a timing candidate, not a validated acceptable rank for this 123 input. Inspect reported field errors; speed alone does not establish reconstruction quality."}
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (out/"timing_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "rank", *statistics["legacy"]])
        writer.writeheader()
        writer.writerows({"method": method, "rank": rank, **parts} for method, parts in statistics.items())
    print("\n方式 | 場生成[s] | コア重み付け | 座標準備(オンライン) | 加算(コア移動を含む) | 復元", flush=True)
    for method, t in statistics.items():
        print(f"{method} | {t['field_generation_seconds']:.4f} | {t['core_weighting_seconds']:.4f} | "
              f"{t['shift_seconds']:.4f} | {t['summation_seconds']:.4f} | {t['reconstruction_seconds']:.4f}")
    print(f"従来法/提案法の時間比: {data['speedup_legacy_over_proposed']:.2f}")
    print("直接場への複素場誤差[%]:", {method: 100*error for method, error in errors.items()})
    save_quality(out, quality)
    print("保存先:", out.resolve())
    return data


def evaluate_saved_run(result_dir: str | Path) -> dict:
    """旧結果に画質だけ追記する。両方式を1回再構成し、元の時間記録は変更しない。

    元の実行は場を保存していなかったのでSSIM/PSNRはL2値から復元できない。
    J0の直接生成・CGH化・伝搬・3回のベンチマークは行わない。
    """
    out = Path(result_dir)
    record = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    if record.get("study") not in ("legacy_coaxial_restore_spatial_shift_vs_projected_core_shift",
                                   "legacy_coaxial_restore_placed_factors_vs_projected_core_shift"):
        raise ValueError("Use a legacy/proposed benchmark result directory")
    cached = out/"quality_metrics.json"
    if "image_metrics" in record:
        metrics = record["image_metrics"]
        save_quality(out, metrics)
        return metrics
    if cached.is_file():
        metrics = json.loads(cached.read_text(encoding="utf-8"))["image_metrics"]
        save_quality(out, metrics)
        return metrics
    root = Path(record["reference_dir"])
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    config = source["config"]
    if (config["grid_size"] != record["grid_size"]
            or config["target_digits"] != record["target_digits"]
            or config["thread_spacing_m"] != record["thread_spacing_m"]
            or config["spot_radius_m"] != record["spot_radius_m"]):
        raise ValueError("Reference configuration changed; do not attach metrics to old timings")
    with np.load(root/"comparison_data.npz", allow_pickle=False) as saved:
        coefficients, positions = saved["coefficients"].copy(), saved["thread_x_m"].copy()
    with np.load(record["library_path"], allow_pickle=False) as saved:
        parent = {key: saved[key].copy() for key in ("uy", "ux", "vy", "vx", "cores", "m", "kr")}
    if (coefficients.shape != (record["modes_per_thread"], record["thread_count"])
            or not np.allclose(parent["kr"], source["kr_per_m"], rtol=1e-12, atol=0)):
        raise ValueError("Saved weights/dictionary differ from the benchmark")
    library = proposed.truncate_library(parent, record["rank"])
    del parent
    pitch = config["pixel_pitch_m"]
    print("画質追加: 従来法と提案法を1回ずつ再構成します。元の時間表は変更しません。", flush=True)
    old, _ = legacy_field(library, coefficients, positions, pitch)
    new, _, _ = proposed.proposed_field(library, coefficients, positions, pitch)
    fields = {"legacy": old, "proposed": new}
    del library
    with np.load(root/"initial_fields.npz", allow_pickle=False) as saved:
        direct = saved["direct_complex"].copy()
    axis = (np.arange(config["grid_size"])-config["grid_size"]//2)*pitch
    radius = np.hypot(axis[:, None], axis[None, :])
    illumination = (radius <= config["aperture_radius_m"]).astype(float)
    if config["illumination_waist_m"] is not None:
        illumination *= np.exp(-(radius/config["illumination_waist_m"])**2)
    norm = float(np.linalg.norm(direct))
    for name, field in fields.items():
        field *= illumination/source["reference_scale"]
        error = float(np.linalg.norm(field-direct)/norm)
        if not np.isclose(error, record["complex_relative_l2_vs_saved_direct"][name], rtol=1e-6, atol=1e-9):
            raise ValueError("Regenerated field errors differ; inputs changed, use a new benchmark")
    metrics = image_quality(direct, fields)
    payload = {"schema_version": 1, "rank": record["rank"], "image_metrics": metrics,
               "fields_regenerated_once": True, "original_timing_records_changed": False,
               "scope": "Pre-CGH full-grid amplitude/intensity; no independent normalization. Saved direct field is the reference. Not propagated CGH replay or target-digit quality."}
    cached.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    save_quality(out, metrics)
    print("画質の保存先:", cached.resolve())
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="legacy_vs_proposed_rank80_outputs")
    parser.add_argument("--reference-dir", default="lsh_digits123_rank_verified_outputs_rank512_parent512_v1")
    parser.add_argument("--rank", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    args = parser.parse_args()
    run_benchmark(args.reference_dir, args.output_dir, args.rank, args.repeats, args.warmups)
