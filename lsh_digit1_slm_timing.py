"""CGH更新開始→8-bitマスク→SLMフレーム準備→表示APIまでの計測。

4f再生・光学伝搬・画像保存はオンライン時間から除く。
実機未接続ならパターン準備までのみ記録し、表示時間はnullとする。
表示完了は専用コールバックが待機・確認できた場合にのみ記録する。
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import lsh_digit1_proposed as proposed
import lsh_digit1_theory as theory


@dataclass(frozen=True)
class DisplayReceipt:
    # submitted=転送要求が戻っただけ。表示済みとは扱わない。
    # controller_confirmed=SDK等で当該フレームの表示完了を確認。
    # optically_confirmed=センサー等で光学的な表示・整定を確認。
    stage: str = "submitted"
    detail: str = ""


def summarize_samples(samples: list[dict]) -> dict:
    stats = {}
    for key in samples[0]:
        if not key.endswith("_seconds"):
            continue
        values = [s[key] for s in samples if s[key] is not None]
        stats[key] = ({"count": len(values), "median": float(np.median(values)),
                       "min": float(np.min(values)), "max": float(np.max(values)),
                       "p95": float(np.percentile(values, 95))} if values else None)
    return stats


def run_slm_timing(output_dir: str | Path, config: theory.Digit1Config,
                   reference_scale: float, rank: int = 100,
                   library_path: str | Path | None = None,
                   method: str = "proposed", repeats: int = 5, warmup: int = 1,
                   include_coefficient_design: bool = True,
                   display_callback: Callable[[np.ndarray], DisplayReceipt | None] | None = None,
                   expected_slm_shape: tuple[int, int] | None = None,
                   frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                   mode_spacing_factor: int = 1, seed: int = 42) -> dict:
    """表示コールバックはユーザーのSDK等へ接続する。自動接続はしない。

    display_callbackは同じ時計で測る同期関数。戻るまでの待機も計測する。
    Noneを返した場合はsubmitted扱い。確認不能な表示/光学時間は報告しない。
    実機には必ずexpected_slm_shape=(高さ,幅)を指定し、合致を検査する。
    frame_transformは校正済み座標変換・LUT等を実装する任意の関数。
    自動リサイズはしない。初期化・辞書読込は開始時刻の前。
    """
    if method not in ("direct", "proposed"):
        raise ValueError("method must be direct or proposed")
    if repeats < 1 or warmup < 0 or not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("invalid repeats/warmup/reference_scale")
    if display_callback is not None and expected_slm_shape is None:
        raise ValueError("Specify actual expected_slm_shape=(height,width) before displaying")
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(str(out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output is not empty: {out}; choose a new output_dir")
    out.mkdir(parents=True, exist_ok=True)
    prep_start = time.perf_counter()
    positions = (np.arange(config.thread_count)-(config.thread_count-1)/2)*config.thread_spacing_m
    design, fixed_weights = proposed.design_coefficients(config, positions, mode_spacing_factor)
    n = config.grid_size
    x = (np.arange(n)-n//2)*config.pixel_pitch_m
    pupil = np.hypot(x[:, None], x[None, :]) <= config.aperture_radius_m
    library, offline = None, {"status": "not_needed"}
    if method == "proposed":
        suffix = "" if mode_spacing_factor == 1 else f"_spacing{mode_spacing_factor}"
        path = library_path or out.parent/f"lsh_digit1_library_rank{rank}{suffix}_shiftaware.npz"
        library, offline = proposed.load_or_build_library(path, config, rank, seed,
                                                          mode_spacing_factor, "shift_aware")
    preparation_seconds = time.perf_counter()-prep_start

    def one_sample(index: int):
        start = time.perf_counter_ns()
        if include_coefficient_design:
            _, weights = proposed.design_coefficients(config, positions, mode_spacing_factor)
        else:
            weights = fixed_weights
        designed = time.perf_counter_ns()
        if method == "proposed":
            field, _, _ = proposed.proposed_field(library, weights, positions, config.pixel_pitch_m)
        else:
            field = theory.initial_bessel_field(x, positions, design.kr_per_m, weights)
        generated = time.perf_counter_ns()
        encoding_field = np.where(pupil, field/reference_scale, 0)
        normalized = time.perf_counter_ns()
        codes = theory.encode_phase_mask(encoding_field, config)
        encoded = time.perf_counter_ns()
        frame = frame_transform(codes) if frame_transform is not None else codes
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.dtype != np.uint8:
            raise ValueError("SLM frame must be a 2D uint8 array; implement frame_transform for the SDK")
        if expected_slm_shape is not None and frame.shape != tuple(expected_slm_shape):
            raise ValueError(f"Frame {frame.shape} != SLM {expected_slm_shape}. No implicit resizing is applied.")
        frame = np.ascontiguousarray(frame)
        ready = time.perf_counter_ns()
        receipt = None
        returned = None
        if display_callback is not None:
            receipt = display_callback(frame)
            returned = time.perf_counter_ns()
            if receipt is None:
                receipt = DisplayReceipt()
            if not isinstance(receipt, DisplayReceipt) or receipt.stage not in (
                    "submitted", "controller_confirmed", "optically_confirmed"):
                raise TypeError("display_callback must return DisplayReceipt or None")
        elapsed = lambda a, b: (b-a)*1e-9
        row = {
            "sample": index,
            "coefficient_design_seconds": elapsed(start, designed),
            "field_generation_seconds": elapsed(designed, generated),
            "normalization_and_pupil_seconds": elapsed(generated, normalized),
            "phase_mask_encoding_seconds": elapsed(normalized, encoded),
            "slm_frame_preparation_seconds": elapsed(encoded, ready),
            "update_to_frame_ready_seconds": elapsed(start, ready),
            "display_callback_seconds": elapsed(ready, returned) if returned is not None else None,
            "update_to_api_return_seconds": elapsed(start, returned) if returned is not None else None,
            "update_to_controller_confirmed_seconds": (
                elapsed(start, returned) if receipt is not None and receipt.stage == "controller_confirmed" else None),
            "update_to_optically_confirmed_seconds": (
                elapsed(start, returned) if receipt is not None and receipt.stage == "optically_confirmed" else None),
            "confirmation_stage": receipt.stage if receipt is not None else "not_connected",
            "confirmation_detail": receipt.detail if receipt is not None else "",
        }
        return row, codes, frame

    # 実機接続時はウォームアップも表示する。未接続時はソフトウェアのみ。
    for index in range(warmup):
        one_sample(-index-1)
    samples = []
    last_codes, last_frame = None, None
    for index in range(repeats):
        row, last_codes, last_frame = one_sample(index)
        samples.append(row)
    Image.fromarray(last_codes).save(out/"last_cgh_8bit.png")
    Image.fromarray(last_frame).save(out/"last_slm_frame_8bit.png")
    with (out/"timing_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    summary = {
        "study": "digit1_update_to_slm_display_timing", "schema_version": 1,
        "config": asdict(config), "method": method, "rank": rank if method == "proposed" else None,
        "mode_spacing_factor": mode_spacing_factor, "modes_per_thread": design.count,
        "active_thread_count": int(np.count_nonzero(np.any(fixed_weights != 0, axis=0))),
        "reference_scale": reference_scale, "repeats": repeats, "warmup": warmup,
        "include_coefficient_design": include_coefficient_design,
        "hardware_connected": display_callback is not None, "expected_slm_shape": expected_slm_shape,
        "last_frame_shape": list(last_frame.shape), "frame_transform_used": frame_transform is not None,
        "preparation_seconds": preparation_seconds, "offline": offline,
        "statistics_seconds": summarize_samples(samples), "samples": samples,
        "scope": "From target/weight update processing (or weights-ready when include_coefficient_design=False) to frame-ready / blocking callback return. Excludes device initialization, dictionary preparation, validation 4f replay, propagation and saving. No fake display or settling wait.",
        "confirmation": "API return alone means submitted. Controller/optical time is populated only when callback explicitly confirms that stage. All trials repeat the same digit1: device deduplication can affect latency.",
        "physical_mapping": "Current mask uses virtual 1:1 coordinates. Hardware geometry, phase calibration and illumination must be handled separately in frame_transform/SDK; no automatic physical mapping.",
    }
    theory._write_json(out/"summary.json", summary)
    return summary
