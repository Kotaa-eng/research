"""保存された直接場を基準に、CGH/伝搬の前に共有基底のランクを検証する。

研究計算はrun_rank_audit()をユーザーが呼んだ場合だけ実行する。
既存の直接場は誤差診断専用。直接法の計算時間をキャッシュで短縮しない。
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import lsh_digit1_theory as theory
import lsh_digit1_proposed as proposed
from stage_a_validation import _relative_l2

SCHEMA_VERSION = 2


def initial_scores(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("two initial fields with the same 2D shape are required")
    ref_i, rec_i = np.abs(reference)**2, np.abs(candidate)**2
    power = float(np.sum(ref_i, dtype=np.float64))
    if power <= 0:
        raise ValueError("zero reference power")
    return {"complex_relative_l2": _relative_l2(reference, candidate),
            "intensity_relative_l2": _relative_l2(ref_i, rec_i),
            "power_ratio_vs_direct": float(np.sum(rec_i, dtype=np.float64)/power)}


def run_rank_audit(reference_dir: str | Path, output_dir: str | Path,
                   ranks: tuple[int, ...] = (100, 160, 256, 384, 512),
                   max_complex_error: float = .01,
                   max_intensity_error: float = .03, *,
                   timing_repeats: int = 3, timing_warmups: int = 1) -> dict:
    """最大ランクで一度学習し、同じ基底の列数だけを変えて測る。

    小さいランクで独立に学習した辞書とは同一ではない。合格した列の
    部分集合を最終比較でもそのまま使うため、親ランクと辞書を記録する。
    配列準備を別計測し、全ランクを同じウォームアップ・反復回数で測る。
    field_secondsはオンライン場生成時間の中央値。
    """
    root, out = Path(reference_dir), Path(output_dir)
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if source.get("study") != "digit1_direct_vs_projected_shared_core":
        raise ValueError("use a saved direct/proposed comparison directory")
    config = theory.Digit1Config(**source["config"])
    if source.get("basis_strategy") != "shift_aware":
        raise ValueError("this audit expects a shift_aware reference comparison")
    if any(isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) for rank in ranks):
        raise ValueError("candidate ranks must be integers")
    ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not ranks or not 1 <= ranks[0] <= ranks[-1] <= config.grid_size:
        raise ValueError("candidate ranks must fit inside grid_size")
    if not (0 < max_complex_error <= 1 and 0 < max_intensity_error <= 1):
        raise ValueError("error limits must be inside (0,1]")
    if (isinstance(timing_repeats, bool) or not isinstance(timing_repeats, (int, np.integer))
            or timing_repeats < 1):
        raise ValueError("timing_repeats must be a positive integer")
    if (isinstance(timing_warmups, bool) or not isinstance(timing_warmups, (int, np.integer))
            or timing_warmups < 0):
        raise ValueError("timing_warmups must be a nonnegative integer")
    timing_repeats, timing_warmups = int(timing_repeats), int(timing_warmups)
    paths = (root/"summary.json", root/"comparison_data.npz", root/"initial_fields.npz")
    signatures = {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                  for path in paths}
    expected = {"schema_version": SCHEMA_VERSION, "reference_dir": str(root.resolve()),
                "source_signatures": signatures, "config": theory.config_payload(config),
                "ranks": list(ranks), "max_complex_error": max_complex_error,
                "max_intensity_error": max_intensity_error,
                "online_implementation": proposed.ONLINE_IMPLEMENTATION_VERSION,
                "timing_repeats": timing_repeats, "timing_warmups": timing_warmups}
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        if not (out/"summary.json").is_file() or not (out/"manifest.json").is_file():
            raise FileExistsError(f"Incomplete audit: {out}; use a new output_dir")
        saved = json.loads((out/"summary.json").read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Audit settings differ: {out}; use a new output_dir")
        return saved
    with np.load(root/"comparison_data.npz", allow_pickle=False) as data:
        positions, weights = data["thread_x_m"].copy(), data["coefficients"].copy()
    with np.load(root/"initial_fields.npz", allow_pickle=False) as data:
        direct = data["direct_complex"].copy()
        old = data["proposed_complex"].copy()
    n = config.grid_size
    if direct.shape != (n, n) or old.shape != direct.shape:
        raise ValueError("saved initial-field shape differs from config")
    x = (np.arange(n)-n//2)*config.pixel_pitch_m
    illumination = (np.hypot(x[:, None], x[None, :]) <= config.aperture_radius_m).astype(float)
    if config.illumination_waist_m is not None:
        illumination *= np.exp(-(np.hypot(x[:, None], x[None, :])/config.illumination_waist_m)**2)
    scale = float(source["reference_scale"])
    if scale <= 0:
        raise ValueError("invalid reference scale")
    old_row = {"rank": source["rank"], "basis": "saved_original_basis",
               **initial_scores(direct, old), "field_seconds": None}
    del old
    spacing = source.get("mode_spacing_factor", 1)
    seed = source["seed"]
    parent_rank = ranks[-1]
    spec = proposed.library_spec(config, parent_rank, seed, spacing, "shift_aware")
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    library_path = out.parent/f"lsh_digits123_design_library_rank{parent_rank}_{signature}.npz"
    # 前処理の失敗で空でない出力フォルダが残らないよう、辞書を先に準備する。
    library, offline = proposed.load_or_build_library(library_path, config, parent_rank,
                                                      seed, spacing, "shift_aware")
    if weights.shape != (len(library["m"]), len(positions)):
        raise ValueError("saved coefficients and dictionary modes differ")
    design = proposed.mode_design_for_spacing(config, spacing)
    if not np.allclose(library["kr"], design.kr_per_m, rtol=1e-12, atol=0):
        raise ValueError("dictionary kr differs from the saved design")
    expected_positions = (np.arange(config.thread_count)-(config.thread_count-1)/2)*config.thread_spacing_m
    if not np.allclose(positions, expected_positions, rtol=1e-12, atol=1e-15):
        raise ValueError("saved thread placement differs from config")
    rows = []
    timing_details = []
    for rank in ranks:
        source_core = library["cores"][:, :rank, :rank]
        tic = time.perf_counter()
        trial_library = proposed.truncate_library(library, rank)
        packing_seconds = time.perf_counter()-tic
        layout = {"slice_c_contiguous": bool(source_core.flags.c_contiguous),
                  "slice_strides_bytes": list(source_core.strides),
                  "packed_c_contiguous": bool(trial_library["cores"].flags.c_contiguous),
                  "packed_strides_bytes": list(trial_library["cores"].strides)}
        for _ in range(timing_warmups):
            warm_field, _, warm_core = proposed.proposed_field(
                trial_library, weights, positions, config.pixel_pitch_m)
            del warm_field, warm_core
        samples = []
        for repeat in range(timing_repeats):
            candidate, timings, total_core = proposed.proposed_field(
                trial_library, weights, positions, config.pixel_pitch_m)
            samples.append(timings)
            del total_core
            if repeat < timing_repeats-1:
                del candidate
        medians = {key: float(np.median([sample[key] for sample in samples]))
                   for key in samples[0]}
        candidate *= illumination/scale
        scores = initial_scores(direct, candidate)
        row = {"rank": rank, "basis": f"prefix_of_rank{parent_rank}", **scores,
               "field_seconds": medians["field_generation_seconds"],
               "field_seconds_min": min(sample["field_generation_seconds"] for sample in samples),
               "field_seconds_max": max(sample["field_generation_seconds"] for sample in samples),
               "library_packing_seconds": packing_seconds,
               **{key: value for key, value in medians.items() if key != "field_generation_seconds"}}
        rows.append(row)
        timing_details.append({"rank": rank, "library_packing_seconds": packing_seconds,
                               "core_layout": layout, "median_seconds": medians,
                               "samples_seconds": samples})
        print(f"Rank {rank}: complex error={scores['complex_relative_l2']:.3%}, "
              f"intensity error={scores['intensity_relative_l2']:.3%}, "
              f"power/direct={scores['power_ratio_vs_direct']:.4f}, "
              f"field median={row['field_seconds']:.4f}s ({timing_repeats} repeats)", flush=True)
        del candidate, trial_library
    passing = [row for row in rows if row["complex_relative_l2"] <= max_complex_error
               and row["intensity_relative_l2"] <= max_intensity_error]
    accepted = passing[0]["rank"] if passing else None
    recommendation = None if accepted is None else {
        "rank": accepted, "library_parent_rank": parent_rank,
        "library_path": str(library_path.resolve()),
        "seed": seed, "mode_spacing_factor": spacing,
        "basis_strategy": "shift_aware", "max_initial_complex_error": max_complex_error}
    result = {**expected, "study": "digits123_shared_basis_initial_rank_audit",
              "original_rank_result": old_row, "cases": rows,
              "timing_details": timing_details,
              "accepted_rank": accepted, "recommendation": recommendation, "offline": offline,
              "scope": "initial finite-aperture complex field only; no CGH or propagation",
              "basis_note": "Trials share the parent basis. Reuse this exact parent library and prefix in final comparison.",
              "timing_note": "Same warmups/repeats for every rank; field_seconds and component timings are medians. Component medians need not sum to total median. Packing is a separately reported one-time dictionary preparation; it is excluded from online timing. Saved direct fields are NOT a fresh direct-method timing benchmark.",
              "next_step": "Validate CGH replay at the accepted rank. If none passes, increase maximum rank or revise basis training; do not assert reconstruction success."}
    out.mkdir(parents=True, exist_ok=True)
    theory._write_json(out/"summary.json", result)
    with (out/"rank_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([old_row, *rows])
    fig, axes = theory.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for key, label, limit in (("complex_relative_l2", "Complex field", max_complex_error),
                               ("intensity_relative_l2", "Intensity", max_intensity_error)):
        axes[0].semilogy(ranks, [max(row[key], 1e-12) for row in rows], "o-", label=label)
        axes[0].axhline(limit, linestyle="--", label=label+" limit")
    axes[0].scatter([old_row["rank"]], [max(old_row["complex_relative_l2"], 1e-12)],
                    color="red", marker="x", label="Original saved basis: complex")
    axes[0].set(xlabel="Spatial rank", ylabel="Relative L2 error (fraction)")
    axes[0].legend()
    axes[1].plot(ranks, [row["power_ratio_vs_direct"] for row in rows], "o-")
    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].set(xlabel="Spatial rank", ylabel="Initial power / direct power")
    fig.savefig(out/"rank_initial_error.png", dpi=150)
    theory.plt.close(fig)
    fig, ax = theory.plt.subplots(figsize=(8, 4), constrained_layout=True)
    middle = np.array([row["field_seconds"] for row in rows])
    lower = np.array([row["field_seconds_min"] for row in rows])
    upper = np.array([row["field_seconds_max"] for row in rows])
    ax.errorbar(ranks, middle, yerr=np.stack((middle-lower, upper-middle)), fmt="o-",
                capsize=3, label=f"Median, {timing_repeats} repeats; bars=min/max")
    ax.set(xlabel="Spatial rank", ylabel="Online field generation [s]")
    ax.legend()
    fig.savefig(out/"rank_field_timing.png", dpi=150)
    theory.plt.close(fig)
    theory._write_json(out/"manifest.json", {"files": sorted(
        {"manifest.json"} | {path.name for path in out.iterdir() if path.is_file()})})
    return result
