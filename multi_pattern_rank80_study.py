"""異なる軸方向パターンを持つ複数Threadを移動・合成し、CGHまで検証する。

保存済みランク80辞書を使用。import時には計算を行わない。
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shifted_rank80_study import run_shifted_rank80
from lsh_core import relative_error
from thread_cgh import axial_intensity


def frozen_wave_weights(mode_index, start_fraction, end_fraction):
    """0..Lに対する矩形目標のFourier係数。全Threadで同じkr辞書を使う。"""
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError('明区間は0～1の範囲で開始<終了としてください。')
    width = end_fraction-start_fraction
    center = (start_fraction+end_fraction)/2
    return width*np.sinc(mode_index*width)*np.exp(-2j*np.pi*mode_index*center)


def separated_random_offsets(count, max_shift_px, min_distance_px, seed):
    """原点を1本含めて、中心間隔を確保した出力面画素の整数座標を作る。"""
    if count < 2 or max_shift_px < 1 or min_distance_px < 0:
        raise ValueError('Thread数・移動範囲・最小間隔を見直してください。')
    rng = np.random.default_rng(seed)
    offsets = [np.array([0, 0], dtype=int)]
    for _ in range(1, count):
        for _attempt in range(10000):
            candidate = rng.integers(-max_shift_px, max_shift_px+1, size=2)
            if all(np.linalg.norm(candidate-prev) >= min_distance_px
                   for prev in offsets):
                offsets.append(candidate)
                break
        else:
            raise ValueError('指定した移動範囲では中心間隔を満たせません。')
    return np.asarray(offsets)


def run_multi_pattern_cgh(library_path, output_dir, *, mode_index, kr,
                          saved_reference_weights, pixel_pitch, slm_pixel_pitch,
                          wavelength, design_length, edge_fraction,
                          expected_config_id,
                          windows=((0.08, 0.24), (0.30, 0.46),
                                   (0.53, 0.69), (0.76, 0.92)),
                          max_shift_px=80, min_distance_px=45, seed=42,
                          validate_axial=True, axial_samples=21,
                          propagation_padding=2):
    """複数の異なるFWパターンの複素場・CGH・数値再生を3方式で比較。"""
    mode_index = np.asarray(mode_index)
    kr = np.asarray(kr)
    if mode_index.shape != kr.shape:
        raise ValueError('モード番号とkrの数が一致しません。')
    if design_length <= 0:
        raise ValueError('design_lengthは正にしてください。')
    windows = tuple(tuple(float(v) for v in pair) for pair in windows)
    if len(windows) < 2:
        raise ValueError('異なる2個以上の目標パターンを設定してください。')
    if len(set(windows)) != len(windows):
        raise ValueError('目標明区間が重複しています。異なるパターンを指定してください。')
    weights = np.stack([frozen_wave_weights(mode_index, start, end)
                        for start, end in windows])
    offsets = separated_random_offsets(len(windows), max_shift_px,
                                       min_distance_px, seed)
    out = Path(output_dir)
    result = run_shifted_rank80(
        library_path, out, kr=kr, weights=saved_reference_weights,
        pixel_pitch=pixel_pitch, slm_pixel_pitch=slm_pixel_pitch,
        wavelength=wavelength, edge_fraction=edge_fraction,
        thread_shifts_px=offsets, expected_config_id=expected_config_id,
        thread_weights=weights,
        pattern_labels=[f'{start:.2f}L–{end:.2f}L' for start, end in windows],
    )
    with (out/'thread_patterns.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['thread', 'shift_x_px', 'shift_y_px', 'start_z_m', 'end_z_m'])
        writer.writerows((i, int(offsets[i, 0]), int(offsets[i, 1]),
                          windows[i][0]*design_length, windows[i][1]*design_length)
                         for i in range(len(windows)))
    z = np.linspace(0, design_length, 501)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ((start, end), (sx, sy)) in enumerate(zip(windows, offsets)):
        target = ((z >= start*design_length)&(z <= end*design_length)).astype(float)
        ax.plot(z*1000, target+i*1.25,
                label=f'Thread {i}: ({sx:+d},{sy:+d}) px; {start:.2f}–{end:.2f} L')
    ax.set(xlabel='z [mm, from 4f output plane]',
           ylabel='Target window (vertical offsets for display)',
           title='Distinct axial target patterns for shifted Threads')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'thread_target_windows.png', dpi=150)
    plt.close(fig)
    if validate_axial:
        # 同じ4f出力座標でCGH再生場を伝搬。圧縮誤差と目標形状との差は別々に報告。
        z_samples = np.linspace(0, design_length, axial_samples)
        with np.load(out/'comparison_fields.npz') as stored:
            replay_fields = {key: stored[key] for key in
                             ('direct_replay', 'periodic_replay', 'rank80_replay')}
        axial_rows = []
        axial_metrics = []
        fig, axes = plt.subplots(len(windows), 1,
                                 figsize=(10, 2.8*len(windows)), sharex=True,
                                 constrained_layout=True)
        for thread, ((start, end), (sx, sy)) in enumerate(zip(windows, offsets)):
            curves = {}
            for method, field in replay_fields.items():
                curves[method] = axial_intensity(
                    field, pixel_pitch, wavelength, z_samples,
                    padding_factor=propagation_padding,
                    x_target=sx*pixel_pitch, y_target=sy*pixel_pitch)
                print(f'Axial check: thread {thread+1}/{len(windows)}, {method}', flush=True)
            axial_metrics.append({
                'thread': thread,
                'rank80_vs_periodic_replay_axial_error': relative_error(
                    curves['periodic_replay'], curves['rank80_replay']),
                'rank80_vs_direct_replay_axial_error': relative_error(
                    curves['direct_replay'], curves['rank80_replay']),
                'periodic_vs_direct_replay_axial_error': relative_error(
                    curves['direct_replay'], curves['periodic_replay']),
            })
            peak = max(float(curves['direct_replay'].max()), 1e-30)
            target = ((z_samples >= start*design_length)
                      & (z_samples <= end*design_length)).astype(float)
            axial_metrics[-1]['direct_replay_vs_target_shape_error'] = relative_error(
                target, curves['direct_replay']/peak)
            axial_metrics[-1]['rank80_replay_vs_target_shape_error'] = relative_error(
                target, curves['rank80_replay']/peak)
            ax = axes[thread]
            ax.plot(z_samples*1000, target, ':', color='black', label='Target')
            for method, style in [('direct_replay', '-'),
                                  ('periodic_replay', '--'), ('rank80_replay', '-.')]:
                ax.plot(z_samples*1000, curves[method]/peak, style, label=method)
            ax.set(ylabel=f'Thread {thread}\n/ direct peak', ylim=(-0.05, None))
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
            for j, z in enumerate(z_samples):
                axial_rows.append({'thread': thread, 'z_m': float(z),
                    'target': float(target[j]),
                    **{key: float(curve[j]) for key, curve in curves.items()}})
        axes[-1].set_xlabel('z [mm, from 4f output plane]')
        fig.savefig(out/'cgh_axial_validation.png', dpi=150)
        plt.close(fig)
        with (out/'cgh_axial_validation.csv').open('w', newline='',
                                                  encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(axial_rows[0]))
            writer.writeheader()
            writer.writerows(axial_rows)
        result['axial_cgh_replay_errors'] = axial_metrics
        result['axial_note'] = ('2x padded angular-spectrum diagnostic at sampled z values. '
                                'Target mismatch and propagation convergence require separate checks.')
        (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
