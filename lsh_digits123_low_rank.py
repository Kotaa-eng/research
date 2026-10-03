"""低ランクの失敗も含め、保存済み直接像とCGH再生を比較する診断。

直接場・係数・直接伝搬像を保存結果から読む。新しい直接法の時間は
測らない。既存の採用候補用の品質ガードは変更しない。
研究計算はrun_low_rank_study()をユーザーが呼んだ場合だけ行う。
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

import lsh_digit1_theory as theory
import lsh_digit1_proposed as proposed
import lsh_digits123_rank_audit as rank_audit
from stage_a_validation import _field_metrics, _image_metrics, _relative_l2

SCHEMA_VERSION = 1


def _sheet_metrics(reference_row: np.ndarray, candidate_row: np.ndarray,
                   roi: np.ndarray) -> dict:
    ref, rec = reference_row[:, roi], candidate_row[:, roi]
    ref_i, rec_i = np.abs(ref)**2, np.abs(rec)**2
    return {"complex_relative_l2": _relative_l2(ref, rec),
            "intensity_relative_l2": _relative_l2(ref_i, rec_i),
            **_image_metrics(ref_i, rec_i)}


def _overview(path: Path, images: dict, x: np.ndarray, z: np.ndarray,
              roi: np.ndarray, config: theory.Digit1Config,
              *, masks: bool = False, difference: bool = False) -> None:
    count = len(images)
    columns = min(3, count)
    rows = (count+columns-1)//columns
    fig, axes = theory.plt.subplots(rows, columns, figsize=(5*columns, 5*rows),
                                    squeeze=False, constrained_layout=True)
    peak = 255 if masks else max(np.finfo(float).tiny,
                                max(float(image[:, roi].max()) for image in images.values()))
    for ax, (title, image) in zip(axes.flat, images.items()):
        if masks:
            artist = ax.imshow(image, cmap="gray", vmin=0, vmax=255,
                               interpolation="nearest")
            ax.set(title=title, xlabel="x pixel", ylabel="y pixel")
        else:
            values, extent, xlabel, ylabel = theory.sheet_display(
                image, x, z, roi, config.digit_axes)
            artist = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                               cmap="gray", vmin=0, vmax=peak)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        label = ("8-bit phase code" if masks else "Absolute intensity difference, common scale"
                 if difference else "Intensity, common scale")
        fig.colorbar(artist, ax=ax, label=label)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.savefig(path, dpi=160)
    theory.plt.close(fig)


def run_low_rank_study(reference_dir: str | Path, output_dir: str | Path,
                       ranks: tuple[int, ...] = (50, 100, 200, 300, 450, 512), *,
                       max_complex_error: float = .01,
                       max_intensity_error: float = .03,
                       timing_repeats: int = 3, timing_warmups: int = 1,
                       reuse_dir: str | Path | None = None) -> dict:
    """全候補をCGH・伝搬まで診断する。初期場が不合格でも止めない。

    正規化は保存済み直接法の定数で固定。低ランクの光量低下を隠さない。
    全ランクに同じ親基底・係数・開口・エンコード・伝搬条件を使う。
    reuse_dirの物理条件・元データ・実装が一致する場合、完成した伝搬結果を
    新しい出力へコピーする。場生成時間は全候補で改めて測る。
    """
    root, out = Path(reference_dir), Path(output_dir)
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if source.get("study") != "digit1_direct_vs_projected_shared_core":
        raise ValueError("use a saved direct/proposed comparison directory")
    config = theory.Digit1Config(**source["config"])
    if not config.include_cgh_replay or source.get("basis_strategy") != "shift_aware":
        raise ValueError("this study needs CGH replay and a shift_aware reference")
    parent_rank = source.get("library_parent_rank", source["rank"])
    if any(isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) for rank in ranks):
        raise ValueError("candidate ranks must be integers")
    ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not ranks or ranks[0] < 1 or ranks[-1] != parent_rank:
        raise ValueError(f"include parent rank {parent_rank} as the largest candidate")
    paths = [root/name for name in ("summary.json", "comparison_data.npz",
                                   "initial_fields.npz", "direct_cgh_8bit.png")]
    paths.append(Path(source["library_path"]))
    expected = {"schema_version": SCHEMA_VERSION, "reference_dir": str(root.resolve()),
                "source_signatures": {p.name: {"size": p.stat().st_size,
                                               "mtime_ns": p.stat().st_mtime_ns} for p in paths},
                "config": theory.config_payload(config), "ranks": list(ranks),
                "library_parent_rank": parent_rank,
                "online_implementation": proposed.ONLINE_IMPLEMENTATION_VERSION,
                "max_complex_error": max_complex_error,
                "max_intensity_error": max_intensity_error,
                "timing_repeats": timing_repeats, "timing_warmups": timing_warmups,
                "reuse_dir": str(Path(reuse_dir).resolve()) if reuse_dir is not None else None}
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        settings = out/"settings.json"
        if not settings.is_file() or json.loads(settings.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Study settings differ: {out}; use a new output_dir")
        if (out/"summary.json").is_file() and (out/"manifest.json").is_file():
            manifest = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
            if all((out/name).is_file() for name in manifest["files"]):
                return json.loads((out/"summary.json").read_text(encoding="utf-8"))

    reuse = Path(reuse_dir) if reuse_dir is not None else None
    reusable_ranks = set()
    if reuse is not None:
        if reuse.resolve() == out.resolve():
            raise ValueError("reuse_dir must differ from output_dir")
        saved_settings = json.loads((reuse/"settings.json").read_text(encoding="utf-8"))
        # 候補の追加だけを許す。基底・kr・係数・実装・計測条件の変更は拒否。
        mismatches = [key for key, value in expected.items()
                      if key not in {"ranks", "reuse_dir"} and saved_settings.get(key) != value]
        if mismatches:
            raise ValueError(f"Reuse settings differ in {mismatches}: {reuse}")
        reusable_ranks = set(saved_settings["ranks"])

    # auditを研究フォルダの隣に置き、既存の親辞書を再利用する。
    audit_dir = out.parent/(out.name+"_initial")
    audit = rank_audit.run_rank_audit(root, audit_dir, ranks,
        max_complex_error=max_complex_error, max_intensity_error=max_intensity_error,
        timing_repeats=timing_repeats, timing_warmups=timing_warmups)
    initial_by_rank = {row["rank"]: row for row in audit["cases"]}
    library, offline = proposed.load_or_build_library(source["library_path"], config,
        parent_rank, source["seed"], source.get("mode_spacing_factor", 1), "shift_aware")
    with np.load(root/"comparison_data.npz", allow_pickle=False) as data:
        x, z = data["x_m"].copy(), data["z_m"].copy()
        target = data["target_intensity"].copy()
        weights, positions = data["coefficients"].copy(), data["thread_x_m"].copy()
        reference_rows = {kind: data["direct_"+kind+"_row"].copy() for kind in ("complex", "cgh")}
    with np.load(root/"initial_fields.npz", allow_pickle=False) as data:
        direct_field = data["direct_complex"].copy()
    with Image.open(root/"direct_cgh_8bit.png") as image:
        direct_codes = np.array(image)
    radius = np.hypot(x[:, None], x[None, :])
    illumination = (radius <= config.aperture_radius_m).astype(float)
    if config.illumination_waist_m is not None:
        illumination *= np.exp(-(radius/config.illumination_waist_m)**2)
    scale = float(source["reference_scale"])
    roi = np.abs(x) <= config.evaluation_x_half_width_m
    reference_i = np.abs(reference_rows["cgh"])**2
    replay_images = {f"Digit {config.target_digits} target": target,
                     "Direct CGH replay intensity": reference_i}
    mask_images = {"Direct 8-bit CGH phase mask": direct_codes}
    results = []
    out.mkdir(parents=True, exist_ok=True)
    theory._write_json(out/"settings.json", expected)
    for rank in ranks:
        case_dir = out/f"rank{rank}"
        case_dir.mkdir(exist_ok=True)
        case_json, case_npz = case_dir/"summary.json", case_dir/"sheet_data.npz"
        case_files = (case_json, case_npz, case_dir/"proposed_cgh_8bit.png",
                      case_dir/"initial_field_comparison.png", case_dir/"cgh_sheet_comparison.png",
                      case_dir/"cgh_mask_comparison.png")
        if not all(path.is_file() for path in case_files) and rank in reusable_ranks:
            previous = reuse/f"rank{rank}"
            previous_files = [previous/path.name for path in case_files]
            if all(path.is_file() for path in previous_files):
                previous_case = json.loads((previous/"summary.json").read_text(encoding="utf-8"))
                if (previous_case.get("rank") != rank
                        or previous_case.get("library_parent_rank") != parent_rank
                        or previous_case.get("online_implementation") != proposed.ONLINE_IMPLEMENTATION_VERSION):
                    raise ValueError(f"Reuse case settings differ: {previous}")
                for old, new in zip(previous_files, case_files):
                    shutil.copy2(old, new)
                previous_case["replay_reused_from"] = str(previous.resolve())
                theory._write_json(case_json, previous_case)
                print(f"Rank {rank}: reusing saved CGH/propagation from {previous}", flush=True)
        if all(path.is_file() for path in case_files):
            case = json.loads(case_json.read_text(encoding="utf-8"))
            if (case.get("rank") != rank or case.get("library_parent_rank") != parent_rank
                    or case.get("online_implementation") != proposed.ONLINE_IMPLEMENTATION_VERSION):
                raise ValueError(f"Case settings differ: {case_dir}")
            with np.load(case_npz, allow_pickle=False) as data:
                replay_i = data["cgh_intensity"].copy()
            with Image.open(case_dir/"proposed_cgh_8bit.png") as image:
                codes = np.array(image)
        else:
            print(f"Rank {rank}: diagnostic CGH/propagation, including failed initial checks", flush=True)
            trial = proposed.truncate_library(library, rank)
            field, one_run_timing, _ = proposed.proposed_field(trial, weights, positions, config.pixel_pitch_m)
            del trial
            field /= scale
            encoding_field = np.where(illumination > 0, field, 0)
            finite_field = (encoding_field*illumination).astype(np.complex64)
            tic = time.perf_counter()
            codes, replay, encoding = theory.encode_and_replay(
                encoding_field, illumination, config, float(library["kr"].max()))
            encoding_seconds = time.perf_counter()-tic
            del field, encoding_field
            tic = time.perf_counter()
            rows, _, _ = theory.propagate({"complex": finite_field,
                "cgh": replay.astype(np.complex64)}, z, config, set())
            propagation_seconds = time.perf_counter()-tic
            replay_i = np.abs(rows["cgh"])**2
            initial = initial_by_rank[rank]
            passed = (initial["complex_relative_l2"] <= max_complex_error
                      and initial["intensity_relative_l2"] <= max_intensity_error)
            case = {"rank": rank, "library_parent_rank": parent_rank,
                    "online_implementation": proposed.ONLINE_IMPLEMENTATION_VERSION,
                    "diagnostic_only": True, "initial_check_passed": passed,
                    "initial": initial,
                    "initial_field_metrics": _field_metrics(direct_field, finite_field, .01),
                    "complex_sheet_proposed_vs_direct": _sheet_metrics(reference_rows["complex"], rows["complex"], roi),
                    "cgh_sheet_proposed_vs_direct": _sheet_metrics(reference_rows["cgh"], rows["cgh"], roi),
                    "proposed_cgh_sheet_vs_target": {"intensity_relative_l2": _relative_l2(target[:, roi], replay_i[:, roi]),
                        **_image_metrics(target[:, roi], replay_i[:, roi])},
                    "encoding": encoding,
                    "additional_field_generation_single_run": one_run_timing,
                    "cgh_encode_and_ideal4f_replay_seconds": encoding_seconds,
                    "two_field_propagation_seconds": propagation_seconds}
            np.savez_compressed(case_npz, complex_row=rows["complex"], cgh_row=rows["cgh"],
                                cgh_intensity=replay_i)
            Image.fromarray(codes).save(case_dir/"proposed_cgh_8bit.png")
            proposed._save_initial_comparison(case_dir, direct_field, finite_field,
                                             {"direct": direct_codes, "proposed": codes})
            proposed._plot_comparison(case_dir, x, z, roi,
                {"Direct CGH replay intensity": reference_i,
                 f"Rank {rank} CGH replay intensity (diagnostic)": replay_i},
                "cgh_sheet_comparison.png", digit_axes=config.digit_axes)
            del finite_field, replay, rows
        # 再利用したCGH・伝搬の旧時間と、今回の場生成の中央値を区別する。
        case["initial"] = initial_by_rank[rank]
        case["initial_check_passed"] = (
            case["initial"]["complex_relative_l2"] <= max_complex_error
            and case["initial"]["intensity_relative_l2"] <= max_intensity_error)
        theory._write_json(case_json, case)
        status = "initial check passed" if case["initial_check_passed"] else "initial check FAILED"
        replay_images[f"Rank {rank} CGH replay intensity\n{status}"] = replay_i
        mask_images[f"Rank {rank} 8-bit CGH phase mask"] = codes
        metric, initial = case["cgh_sheet_proposed_vs_direct"], case["initial"]
        results.append({"rank": rank, "initial_check_passed": case["initial_check_passed"],
            "initial_complex_relative_l2": initial["complex_relative_l2"],
            "initial_intensity_relative_l2": initial["intensity_relative_l2"],
            "initial_power_ratio_vs_direct": initial["power_ratio_vs_direct"],
            "cgh_replay_complex_relative_l2": metric["complex_relative_l2"],
            "cgh_replay_intensity_relative_l2": metric["intensity_relative_l2"],
            "cgh_replay_ssim": metric["ssim"], "cgh_replay_psnr_db": metric["psnr_db"],
            "target_intensity_relative_l2": case["proposed_cgh_sheet_vs_target"]["intensity_relative_l2"],
            "field_seconds_median": initial["field_seconds"],
            "field_seconds_min": initial["field_seconds_min"],
            "field_seconds_max": initial["field_seconds_max"],
            "cgh_encode_and_ideal4f_replay_seconds": case["cgh_encode_and_ideal4f_replay_seconds"],
            "propagation_seconds": case["two_field_propagation_seconds"],
            "replay_reused_from": case.get("replay_reused_from", "")})
    _overview(out/"cgh_replay_all_ranks.png", replay_images, x, z, roi, config)
    _overview(out/"cgh_masks_all_ranks.png", mask_images, x, z, roi, config, masks=True)
    differences = {title: np.abs(image-reference_i) for title, image in list(replay_images.items())[2:]}
    _overview(out/"cgh_replay_absolute_differences.png", differences, x, z, roi, config,
              difference=True)
    with (out/"rank_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {**expected, "study": "digits123_low_rank_diagnostic_cgh_replay",
               "diagnostic_only": True, "initial_audit_dir": str(audit_dir.resolve()),
               "library_path": source["library_path"], "offline": offline,
               "cases": results, "direct_vs_target": source["metrics"]["direct_cgh_sheet_vs_target"],
               "reused_replay_ranks": [row["rank"] for row in results if row["replay_reused_from"]],
               "reference_scale": scale,
               "scope": "Saved direct field, coefficients and replay rows are the reference; no new direct timing benchmark. All ranks propagate despite failed initial checks. Shared scale, no independent brightness fitting; no 2D sensor snapshots.",
               "timing_scope": "All field medians are from this study's initial audit and exclude packing, aperture, normalization, CGH and propagation. CGH timings include ideal4f replay. Cases with replay_reused_from retain the previous encoding/propagation times, not fresh measurements. Direct generation and direct propagation are reused, not timed here."}
    theory._write_json(out/"summary.json", summary)
    theory._write_json(out/"manifest.json", {"files": sorted(
        {"manifest.json"} | {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()})})
    return summary
