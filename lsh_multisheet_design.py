"""複数シートの直接法を、圧縮やCGH化の前に設計・評価する。

1枚用係数のコピーと、全[z,y,x]の目標振幅を反復的に合わせた係数を比較。
これは論文の非反復設計そのものではなく、固定Bessel辞書上の追加設計。
有限開口、8-bit CGH、4f、実機の最適化は含めない。改善は実行結果で判断する。
数値計算は run_joint_design() またはCLIを明示的に呼んだ場合だけ実行。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.special import j0

import lsh_digit1_proposed as proposed
import lsh_digit1_theory as theory
from lsh_multisheet import MultiSheetConfig, _profile_stats, _sheet_coefficients
from stage_a_validation import _image_metrics, _relative_l2


@dataclass(frozen=True)
class JointDesignConfig:
    # 最大krに対する強度の横方向Nyquist条件も検査する。
    fit_x_samples: int = 145
    fit_y_samples: int = 81
    fit_z_samples: int = 128  # 元のLあたり。積分周期を伸ばす場合は自動で増やす。
    iterations: int = 30
    ridge_relative: float = 1e-4  # 各mでlambda=relative*s_max(B_m)^2。
    max_design_memory_mib: float = 2048

    def __post_init__(self) -> None:
        for value in (self.fit_x_samples, self.fit_y_samples, self.fit_z_samples, self.iterations):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("fit samples/iterations must be integers")
        if min(self.fit_x_samples, self.fit_y_samples, self.fit_z_samples) < 3 or self.iterations < 1:
            raise ValueError("fit counts are too small")
        if not np.isfinite(self.ridge_relative) or not 0 < self.ridge_relative <= 1:
            raise ValueError("ridge_relative must be in (0,1]")
        if not np.isfinite(self.max_design_memory_mib) or self.max_design_memory_mib <= 0:
            raise ValueError("invalid memory limit")


def _target(x: np.ndarray, y: np.ndarray, z: np.ndarray,
            config: theory.Digit1Config, stack: MultiSheetConfig) -> tuple[np.ndarray, np.ndarray]:
    sheets = stack.y_positions()
    inside = (y >= sheets.min()-stack.sheet_spacing_m/2) & (y <= sheets.max()+stack.sheet_spacing_m/2)
    return theory.digit_target(x, z, config)[:, None, :]*inside[None, :, None], inside


def _forward(basis: np.ndarray, phase: np.ndarray, weights: np.ndarray) -> np.ndarray:
    transverse = np.empty(basis.shape[:2], dtype=np.complex64)
    for m, b in enumerate(basis):
        # 実行列Bを複素行列へ丸ごとコピーせず、実部・虚部を別々に積算。
        transverse[m] = b @ weights[m].real + 1j*(b @ weights[m].imag)
    return phase @ transverse


def _project(projectors: np.ndarray, phase: np.ndarray, desired: np.ndarray) -> np.ndarray:
    modes = (phase.conj().T @ desired)/len(phase)
    weights = np.empty((len(projectors), projectors.shape[1]), dtype=np.complex64)
    for m, p in enumerate(projectors):
        weights[m] = p @ modes[m].real + 1j*(p @ modes[m].imag)
    return weights


def _objective(field: np.ndarray, amplitude: np.ndarray,
               coefficients: np.ndarray, penalties: np.ndarray) -> tuple[float, float]:
    delta = np.abs(field)-amplitude
    amplitude_mse = float(np.mean(delta.astype(np.float64)**2))
    regularization = float(np.sum(penalties[:, None]*np.abs(coefficients.astype(np.complex128))**2)
                           /amplitude.shape[1])
    return .5*(amplitude_mse+regularization), amplitude_mse


def _fit_basis(x: np.ndarray, y: np.ndarray, thread_x: np.ndarray,
               sheet_y: np.ndarray, kr: np.ndarray,
               settings: JointDesignConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    xx, yy = np.meshgrid(x, y)
    centres_x = np.tile(thread_x, len(sheet_y))
    centres_y = np.repeat(sheet_y, len(thread_x))
    p, j, m = xx.size, len(centres_x), len(kr)
    # Bと正則化逆写像はfloat32。SVD・正則化の作成はfloat64。
    estimated = (8*m*p*j + 40*p*j + 32*settings.fit_z_samples*p)/(1024**2)
    if estimated > settings.max_design_memory_mib:
        raise ValueError(f"Estimated design workspace {estimated:.0f} MiB exceeds the configured limit")
    radius = np.hypot(xx.ravel()[:, None]-centres_x,
                      yy.ravel()[:, None]-centres_y)
    basis = np.empty((m, p, j), dtype=np.float32)
    projectors = np.empty((m, j, p), dtype=np.float32)
    penalties = np.empty(m, dtype=np.float64)
    diagnostics = []
    for index, radial_k in enumerate(kr):
        b = j0(radial_k*radius)
        u, singular, vh = svd(b, full_matrices=False, check_finite=False)
        penalty = settings.ridge_relative*singular[0]**2
        basis[index] = b
        projectors[index] = (vh.T*(singular/(singular**2+penalty))) @ u.T
        penalties[index] = penalty
        diagnostics.append({"mode_index": index, "kr_per_m": float(radial_k),
                            "largest_singular_value": float(singular[0]),
                            "smallest_singular_value": float(singular[-1]),
                            "regularization_lambda": float(penalty)})
        print(f"3D transverse fit: mode {index+1}/{m}", flush=True)
    return basis, projectors, penalties, diagnostics


def fit_joint_coefficients(config: theory.Digit1Config, stack: MultiSheetConfig,
                           design, thread_x: np.ndarray, copied: np.ndarray,
                           settings: JointDesignConfig,
                           mode_spacing_factor: int = 1) -> tuple[np.ndarray, list[dict], dict]:
    """自由な目標位相と、正則化Bessel係数への写像を交互に更新する。

    E(z,y,x)=sum_m exp(i 2pi*m*z/period) sum_sq a_msq J0(kr_m*r_sq)。
    位相更新: D=sqrt(I_target)*exp(i angle(E))。
    係数更新: 各mで正則化最小二乗 B_m a_m≈Fourier_z(D)_m。
    共通の縦キャリアは強度に影響しないので除く。強度を独立に足さない。
    """
    x = np.linspace(-config.evaluation_x_half_width_m, config.evaluation_x_half_width_m,
                    settings.fit_x_samples)
    y = np.linspace(-stack.evaluation_y_half_width_m, stack.evaluation_y_half_width_m,
                    settings.fit_y_samples)
    period = config.design_length_m*mode_spacing_factor
    nz = settings.fit_z_samples*mode_spacing_factor
    z = (np.arange(nz)+.5)*period/nz
    if len(design.m) > nz or 2*np.ptp(design.m) >= nz:
        raise ValueError("Increase fit_z_samples to sample analytic intensity bandwidth")
    max_step = np.pi/(2*float(design.kr_per_m.max()))
    if max(float(np.diff(x).max()), float(np.diff(y).max())) > max_step:
        raise ValueError(f"Increase x/y fit samples: intensity Nyquist step must be <= {max_step*1e6:.2f} um")
    phase = np.exp(2j*np.pi*z[:, None]*design.m[None, :]/period).astype(np.complex64)
    expected_kz = design.q_per_m+2*np.pi*design.m/period
    if not np.allclose(design.kz_per_m, expected_kz, rtol=0, atol=1e-6):
        raise ValueError("Design modes do not match the assumed Fourier period")
    target, _ = _target(x, y, z, config, stack)
    amplitude = np.sqrt(target).reshape(nz, -1).astype(np.float32)
    if not amplitude.any():
        raise ValueError("No illuminated target points in design grid")
    basis, projectors, penalties, diagnostics = _fit_basis(x, y, thread_x, stack.y_positions(),
                                                          design.kr_per_m, settings)
    weights = np.asarray(copied.reshape(len(design.m), -1), dtype=np.complex64).copy()
    field = _forward(basis, phase, weights)
    loss, mse = _objective(field, amplitude, weights, penalties)
    history = [{"iteration": 0, "objective": loss, "amplitude_mse": mse, "accepted_step": 0.0}]
    for iteration in range(1, settings.iterations+1):
        desired = amplitude*np.exp(1j*np.angle(field))
        trial = _project(projectors, phase, desired)
        step = 1.0
        accepted = False
        # 浮動小数点誤差も含めて、記録する目的関数を悪化させる更新を採用しない。
        for _ in range(12):
            next_weights = weights+step*(trial-weights)
            next_field = _forward(basis, phase, next_weights)
            next_loss, next_mse = _objective(next_field, amplitude, next_weights, penalties)
            if next_loss < loss:
                weights, field, loss, mse = next_weights, next_field, next_loss, next_mse
                accepted = True
                break
            step *= .5
        if not accepted:
            print("Stopped: no decreasing update; do not assume the target is achievable.", flush=True)
            break
        history.append({"iteration": iteration, "objective": loss,
                        "amplitude_mse": mse, "accepted_step": step})
        print(f"3D design {iteration}/{settings.iterations}: amplitude MSE={mse:.6g}", flush=True)
    result = weights.reshape(len(design.m), len(stack.y_positions()), len(thread_x)).astype(np.complex128)
    return result, history, {"fit_grid_shape_zyx": list(target.shape), "period_m": period,
                             "maximum_allowed_intensity_sampling_step_um": max_step*1e6,
                             "svd": diagnostics,
                             "scope": "regularized analytic amplitude fit only; no finite aperture, CGH or empirical gain correction"}


def analytic_volume(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    thread_x: np.ndarray, sheet_y: np.ndarray,
                    design, coefficients: np.ndarray) -> np.ndarray:
    """設計用行列を再利用せず、別の評価格子で解析J0を直接計算する。"""
    weights = _sheet_coefficients(coefficients, len(design.m), len(thread_x), len(sheet_y))
    xx, yy = np.meshgrid(x, y)
    cx, cy = np.tile(thread_x, len(sheet_y)), np.repeat(sheet_y, len(thread_x))
    radius = np.hypot(xx.ravel()[:, None]-cx, yy.ravel()[:, None]-cy)
    delta_kz = -design.kr_per_m**2/(design.kz_per_m+design.k0_per_m)
    phase = np.exp(1j*z[:, None]*delta_kz[None, :])
    result = np.zeros((len(z), radius.shape[0]), dtype=np.complex128)
    for index, kr in enumerate(design.kr_per_m):
        transverse = j0(kr*radius) @ weights[index].ravel()
        result += phase[:, index, None]*transverse[None, :]
    return result.reshape(len(z), len(y), len(x))


def _scores(target: np.ndarray, field: np.ndarray, y: np.ndarray, slab: np.ndarray) -> dict:
    intensity = np.abs(field)**2
    on, off = intensity[target > 0], intensity[target == 0]
    total = float(intensity.sum())
    denominator = float(np.linalg.norm(target)*np.linalg.norm(intensity))
    return {"intensity_relative_l2": _relative_l2(target, intensity),
            "intensity_cosine_similarity_to_target":
                float(np.sum(target*intensity)/denominator) if denominator else 0.0,
            "intensity_sum_ratio_to_target": float(total/target.sum()),
            "off_target_power_fraction_in_roi": float(off.sum()/total) if total else None,
            "mean_on_target_intensity": float(on.mean()),
            "mean_off_target_intensity": float(off.mean()),
            "on_target_intensity_cv": float(on.std()/on.mean()) if on.mean() else None,
            "thickness": _profile_stats(intensity, y, slab),
            "per_sheet": [{"y_um": float(y[i]*1e6),
                           "intensity_relative_l2": _relative_l2(target[:, i], intensity[:, i]),
                           **_image_metrics(target[:, i], intensity[:, i])}
                          for i in range(len(y)) if slab[i]]}


def _save_figures(out: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  target: np.ndarray, fields: dict, sheet_y: np.ndarray,
                  history: list[dict]) -> None:
    intensity = {name: np.abs(field)**2 for name, field in fields.items()}
    values = [("Target intensity", target), ("Copied single-sheet coefficients", intensity["copied"]),
              ("Joint 3D coefficient design", intensity["joint"])]
    extent = [z[0]*1e3, z[-1]*1e3, x[0]*1e3, x[-1]*1e3]
    # 生の共通強度と、形を見るための共通の対数スケールを別々に保存する。
    peak = max(float(value.max()) for _, value in values)
    for logarithmic in (False, True):
        fig, axes = theory.plt.subplots(len(sheet_y), 3, figsize=(15, 3*len(sheet_y)),
                                       squeeze=False, constrained_layout=True)
        for row, position in enumerate(sheet_y):
            iy = int(np.argmin(np.abs(y-position)))
            for col, (title, value) in enumerate(values):
                data = value[:, iy].T
                if logarithmic:
                    data = 10*np.log10(np.maximum(data, peak*1e-6)/peak)
                im = axes[row, col].imshow(data, origin="lower", aspect="auto", extent=extent,
                    cmap="gray", vmin=-40 if logarithmic else 0, vmax=0 if logarithmic else peak)
                axes[row, col].set(title=f"{title}; y={y[iy]*1e6:.1f} um",
                                   xlabel="z [mm]", ylabel="x [mm]")
                fig.colorbar(im, ax=axes[row, col], label="Intensity [dB, shared reference]" if logarithmic else "Intensity [shared scale]")
        fig.suptitle("Analytic infinite-aperture model: NOT a CGH replay; no individual brightness fitting")
        fig.savefig(out/("analytic_sheets_db.png" if logarithmic else "analytic_sheets.png"), dpi=140)
        theory.plt.close(fig)
    fig, axes = theory.plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for label, value in values:
        axes[0].plot(y*1e6, value.sum(axis=(0, 2)), label=label)
    axes[0].set(xlabel="y [um]", ylabel="Integrated x/z intensity", title="Raw thickness profile")
    axes[0].legend()
    axes[1].plot([h["iteration"] for h in history], [h["objective"] for h in history])
    axes[1].set(xlabel="Iteration", ylabel="Regularized amplitude objective", title="Design-grid optimization only")
    fig.savefig(out/"design_diagnostics.png", dpi=160)
    theory.plt.close(fig)
    # 係数設計前後の比較をDirect/Proposedと誤記しない。3D表示は後段のCGH検証で行う。


def run_joint_design(reference_dir: str | Path, output_dir: str | Path,
                     stack: MultiSheetConfig | None = None,
                     settings: JointDesignConfig | None = None) -> dict:
    stack, settings = stack or MultiSheetConfig(), settings or JointDesignConfig()
    root, out = Path(reference_dir), Path(output_dir)
    source = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if source.get("study") != "digit1_direct_vs_projected_shared_core":
        raise ValueError("Use the saved single-sheet direct/proposed comparison as source")
    config = theory.Digit1Config(**source["config"])
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Choose a new empty directory: {out}")
    with np.load(root/"comparison_data.npz", allow_pickle=False) as saved:
        original, thread_x = saved["coefficients"].copy(), saved["thread_x_m"].copy()
    factor = source.get("mode_spacing_factor", 1)
    design = proposed.mode_design_for_spacing(config, factor)
    sheet_y = stack.y_positions()
    if original.shape != (len(design.m), len(thread_x)) or not np.all(np.isfinite(original)):
        raise ValueError("Source coefficients do not match the modes")
    if (np.max(np.abs(sheet_y))+stack.sheet_spacing_m/2 >= stack.evaluation_y_half_width_m
            or stack.evaluation_y_half_width_m >= config.grid_size*config.pixel_pitch_m/2):
        raise ValueError("Desired slab/ROI does not fit the numerical window")
    dz = config.design_length_m/(stack.z_samples-1)
    if 2*np.ptp(design.kz_per_m)/(2*np.pi)*dz > 1+1e-10:
        raise ValueError("Increase evaluation z_samples for the intensity bandwidth")
    copied = np.broadcast_to(original[:, None, :], (len(design.m), len(sheet_y), len(thread_x)))
    copied = copied*stack.sheet_weights()[None, :, None]
    out.mkdir(parents=True, exist_ok=True)
    theory._write_json(out/"settings.json", {"reference_dir": str(root.resolve()),
        "config": theory.config_payload(config), "stack": asdict(stack), "design": asdict(settings)})
    tic = time.perf_counter()
    fitted, history, fit_info = fit_joint_coefficients(config, stack, design, thread_x, copied, settings, factor)
    design_seconds = time.perf_counter()-tic
    # この係数ファイルだけを後段に渡す。評価画像から場を作り直す処理はない。
    np.savez_compressed(out/"coefficient_design.npz", coefficients=fitted,
        thread_x_m=thread_x, sheet_y_m=sheet_y, m=design.m, kr_per_m=design.kr_per_m,
        config_json=json.dumps(theory.config_payload(config), sort_keys=True))
    with (out/"iterations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # 7umの既存格子・端点を含むz格子で評価。設計用格子とは別。
    axis = (np.arange(config.grid_size)-config.grid_size//2)*config.pixel_pitch_m
    x = axis[np.abs(axis) <= config.evaluation_x_half_width_m]
    y = axis[np.abs(axis) <= stack.evaluation_y_half_width_m]
    z = np.linspace(0, config.design_length_m, stack.z_samples)
    target, slab = _target(x, y, z, config, stack)
    fields = {"copied": analytic_volume(x, y, z, thread_x, sheet_y, design, copied),
              "joint": analytic_volume(x, y, z, thread_x, sheet_y, design, fitted)}
    scores = {name: _scores(target, field, y, slab) for name, field in fields.items()}
    improved = scores["joint"]["intensity_relative_l2"] < scores["copied"]["intensity_relative_l2"]
    shape_improved = (scores["joint"]["intensity_cosine_similarity_to_target"]
                      > scores["copied"]["intensity_cosine_similarity_to_target"])
    summary = {"study": "analytic_joint_3d_coefficient_design", "schema_version": 1,
        "reference_dir": str(root.resolve()), "config": theory.config_payload(config),
        "stack": asdict(stack), "design": asdict(settings), "fit": fit_info,
        "coefficient_design_seconds": design_seconds, "evaluation": scores,
        "evaluation_improved_intensity_l2": improved,
        "evaluation_improved_shape_cosine": shape_improved,
        "evaluation_grid_shape_zyx": list(target.shape),
        "coefficient_path": str((out/"coefficient_design.npz").resolve()),
        "scope": "No Tucker compression, aperture, CGH, 4f or angular-spectrum propagation; no brightness/phase fitting to evaluation images. Coefficients differ by sheet. Analytic improvement does not establish finite-CGH quality or convergence."}
    theory._write_json(out/"design_summary.json", summary)
    np.savez_compressed(out/"analytic_volume_data.npz", target_intensity=target,
                        x_m=x, y_m=y, z_m=z, copied=fields["copied"].astype(np.complex64),
                        joint=fields["joint"].astype(np.complex64))
    _save_figures(out, x, y, z, target, fields, sheet_y, history)
    theory._write_json(out/"manifest.json", {"files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())})
    print(f"Saved analytic design check to: {out.resolve()}", flush=True)
    print("Evaluate the new direct baseline before running full CGH/Tucker propagation.", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default="lsh_digits123_rank_verified_outputs_rank512_parent512_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    run_joint_design(args.reference_dir, args.output_dir, settings=JointDesignConfig(iterations=args.iterations))


if __name__ == "__main__":
    main()
