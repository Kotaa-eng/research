"""保存済みランク80辞書で複数Threadをずらして合成する。

実行入口はrun_shifted_rank80。import時には計算しない。
座標ずれは出力面の整数画素。各Thread内では全Besselモードが同じ位置。
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import j0

from lsh_core import PhaseOnlyCodec, relative_error, phase_metrics


def _aperture(n, pitch, edge_fraction):
    x = (np.arange(n)-(n-1)/2)*pitch
    half_width = (n-1)*pitch/2
    if edge_fraction:
        start = half_width*(1-edge_fraction)
        u = np.clip((np.abs(x)-start)/(half_width-start), 0, 1)
        edge = .5*(1+np.cos(np.pi*u))
    else:
        edge = np.ones_like(x)
    return x, (edge[:, None]*edge[None, :]).astype(np.float32)


def run_shifted_rank80(library_path, output_dir, *, kr, weights,
                       pixel_pitch, slm_pixel_pitch, wavelength,
                       edge_fraction=0.10, thread_shifts_px=None, thread_count=3, max_shift_px=16,
                       seed=42, expected_config_id=None, thread_weights=None,
                       pattern_labels=None):
    """理論直接生成・非圧縮FFT移動・ランク80移動コアの3段階で誤差を分離。

    FFTの移動位相は周期シフトであり、固定開口を持つ直接生成と厳密には違う。
    コア内での移動はUy.T Dy Uy @ C @ Ux.T Dx Uxという再射影近似。
    移動後も厳密な共通低ランク表現になるとは仮定しない。
    """
    library_path = Path(library_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(library_path) as saved:
        required = {'factor_ky', 'factor_kx', 'core_complex', 'kr', 'weights',
                    'pixel_pitch', 'slm_pixel_pitch', 'wavelength', 'rank', 'config_id'}
        if not required.issubset(saved.files):
            raise ValueError('辞書の必須項目が不足しています。現在のランク調査セルでrank80を再作成してください。')
        uy = saved['factor_ky'].astype(np.float64)
        ux = saved['factor_kx'].astype(np.float64)
        cores = saved['core_complex'].astype(np.complex128)
        saved_kr = saved['kr']
        saved_weights = saved['weights']
        saved_pitch = float(saved['pixel_pitch'])
        saved_slm_pitch = float(saved['slm_pixel_pitch'])
        saved_wavelength = float(saved['wavelength'])
        saved_rank = int(saved['rank'])
        saved_id = str(saved['config_id'].item())
    if saved_rank != 80 or uy.shape[1] != 80 or ux.shape[1] != 80:
        raise ValueError('ランク80の圧縮辞書を指定してください。')
    if not saved_id or (expected_config_id is not None and saved_id != expected_config_id):
        raise ValueError('圧縮辞書と現在の物理設定が違います。同じ設定でランク調査を再実行してください。')
    kr = np.asarray(kr, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.complex128)
    if (not np.array_equal(saved_kr, kr) or not np.array_equal(saved_weights, weights)
            or saved_pitch != pixel_pitch or saved_slm_pitch != slm_pixel_pitch
            or saved_wavelength != wavelength):
        raise ValueError('kr・重み・画素ピッチ・波長が保存辞書と一致しません。')
    count, rank, rank_x = cores.shape
    n = uy.shape[0]
    if (count != len(kr) or rank != rank_x or ux.shape != (n, 80)
            or not 0 <= edge_fraction < 1):
        raise ValueError('辞書・因子・開口設定の形状が不正です。')
    if thread_shifts_px is None:
        generated_shifts = True
        if thread_count < 2 or max_shift_px < 1:
            raise ValueError('移動検証には2本以上のThreadと正の最大移動量が必要です。')
        if thread_count > (2*max_shift_px+1)**2:
            raise ValueError('移動範囲に対してThread数が多すぎます。')
        rng = np.random.default_rng(seed)
        thread_shifts_px = rng.integers(-max_shift_px, max_shift_px+1,
                                        size=(thread_count, 2))
        thread_shifts_px[0] = [0, 0]
        for thread in range(1, thread_count):
            while any(np.array_equal(thread_shifts_px[thread], earlier)
                      for earlier in thread_shifts_px[:thread]):
                thread_shifts_px[thread] = rng.integers(
                    -max_shift_px, max_shift_px+1, size=2)
    else:
        generated_shifts = False
    thread_shifts_px = np.asarray(thread_shifts_px)
    if (thread_shifts_px.ndim != 2 or thread_shifts_px.shape[1] != 2
            or len(thread_shifts_px) < 2
            or not np.all(thread_shifts_px == np.rint(thread_shifts_px))):
        raise ValueError('thread_shifts_pxは[Thread数,2]の整数画素指定です。列は[x,y]。')
    thread_shifts_px = thread_shifts_px.astype(int)
    if not np.any(thread_shifts_px):
        raise ValueError('少なくとも1本のThreadを原点から移動してください。')
    if thread_weights is None:
        thread_weights = np.broadcast_to(weights, (len(thread_shifts_px), count))
    thread_weights = np.asarray(thread_weights, dtype=np.complex128)
    if thread_weights.shape != (len(thread_shifts_px), count):
        raise ValueError('thread_weightsは[Thread数,モード数]で指定してください。')
    if pattern_labels is not None and len(pattern_labels) != len(thread_shifts_px):
        raise ValueError('pattern_labelsの数がThread数と一致しません。')
    x, aperture = _aperture(n, pixel_pitch, edge_fraction)
    xx, yy = np.meshgrid(x, x)
    kx = 2*np.pi*np.fft.fftfreq(n, pixel_pitch)
    ky = kx
    # Thread内のモードを先にコア加算し、位置ずれごとに一度だけ再射影する。
    physical_direct = np.zeros((n, n), dtype=np.complex128)
    periodic_fft = np.zeros((n, n), dtype=np.complex128)
    shifted_core_sum = np.zeros((80, 80), dtype=np.complex128)
    beam_rows = []
    factor_shift_x = factor_shift_y = None
    for thread_id, (sx, sy) in enumerate(thread_shifts_px):
        thread_mode_weights = thread_weights[thread_id]
        centered_thread_core = np.tensordot(thread_mode_weights, cores, axes=(0, 0))
        phase_y = np.exp(-1j*ky*sy*pixel_pitch)
        phase_x = np.exp(-1j*kx*sx*pixel_pitch)
        ay = uy.T @ (phase_y[:, None]*uy)
        ax = ux.T @ (phase_x[:, None]*ux)
        shifted_core_sum += ay @ centered_thread_core @ ax
        if factor_shift_y is None and (sx != 0 or sy != 0):
            factor_shift_y, factor_shift_x = ay, ax
        for i, (wave_number, weight) in enumerate(zip(kr, thread_mode_weights)):
            centered = (j0(wave_number*np.hypot(xx, yy))*aperture).astype(np.complex64)
            shifted_direct = (j0(wave_number*np.hypot(xx-sx*pixel_pitch,
                                                      yy-sy*pixel_pitch))*aperture).astype(np.complex64)
            physical_direct += weight*shifted_direct
            centered_fft = np.fft.fft2(centered, norm='ortho')
            shifted_full_fft = (phase_y[:, None]*centered_fft)*phase_x[None, :]
            periodic_fft += weight*shifted_full_fft
            # 元の共通基底への直交再射影。移動位相はコアの単なる定数倍ではない。
            projected_core = ay @ cores[i] @ ax
            shifted_rank_fft = uy @ projected_core @ ux.T
            periodic_field = np.fft.ifft2(shifted_full_fft, norm='ortho')
            beam_rows.append({
                'thread': thread_id, 'mode': i, 'kr_per_m': float(wave_number),
                'shift_x_px': int(sx), 'shift_y_px': int(sy),
                'centered_rank80_fft_error': relative_error(centered_fft, uy @ cores[i] @ ux.T),
                'shifted_rank80_fft_error': relative_error(shifted_full_fft, shifted_rank_fft),
                'fixed_aperture_vs_periodic_shift_error': relative_error(shifted_direct, periodic_field),
            })
            print(f'Thread {thread_id+1}/{len(thread_shifts_px)}, mode {i+1}/{count}, shift=({sx:+d},{sy:+d}) pixels', flush=True)
    periodic_field = np.fft.ifft2(periodic_fft, norm='ortho')
    proposed_fft = uy @ shifted_core_sum @ ux.T
    proposed_field = np.fft.ifft2(proposed_fft, norm='ortho')
    # 1つの図の中では同じ振幅・強度スケール。圧縮誤差と開口移動誤差を別評価。
    scale = float(np.abs(physical_direct).max())
    codec = PhaseOnlyCodec((n, n), pixel_pitch, slm_pixel_pitch, kr, scale)
    _, physical_codes, physical_clipped = codec.encode(physical_direct)
    _, periodic_codes, periodic_clipped = codec.encode(periodic_field)
    _, proposed_codes, proposed_clipped = codec.encode(proposed_field)
    Image.fromarray(physical_codes).save(out/'cgh_fixed_aperture_direct.png')
    Image.fromarray(periodic_codes).save(out/'cgh_uncompressed_periodic_shift.png')
    Image.fromarray(proposed_codes).save(out/'cgh_rank80_shifted_core.png')
    physical_replay = codec.replay(physical_codes)
    periodic_replay = codec.replay(periodic_codes)
    proposed_replay = codec.replay(proposed_codes)
    def circular_code_mae(first, second):
        delta = (first.astype(np.int16)-second.astype(np.int16)+128) % 256-128
        return float(np.mean(np.abs(delta)))
    phase_info = phase_metrics(periodic_field, proposed_field)[2]
    report = {
        'rank': 80, 'config_id': saved_id, 'mode_count': count,
        'shift_units': 'output-plane pixels [x,y] per Thread', 'thread_count': len(thread_shifts_px), 'output_pixel_pitch_m': pixel_pitch,
        'shift_seed': seed if generated_shifts else None,
        'pattern_labels': None if pattern_labels is None else list(pattern_labels),
        'rank80_vs_uncompressed_periodic_field_error': relative_error(periodic_field, proposed_field),
        'rank80_vs_uncompressed_periodic_fft_error': relative_error(periodic_fft, proposed_fft),
        'fixed_aperture_vs_periodic_shift_field_error': relative_error(physical_direct, periodic_field),
        'rank80_vs_fixed_aperture_direct_field_error': relative_error(physical_direct, proposed_field),
        'rank80_vs_periodic_cgh_replay_error': relative_error(periodic_replay, proposed_replay),
        'fixed_aperture_vs_periodic_cgh_replay_error': relative_error(physical_replay, periodic_replay),
        'rank80_vs_fixed_aperture_cgh_replay_error': relative_error(physical_replay, proposed_replay),
        'rank80_vs_periodic_cgh_code_mae': circular_code_mae(periodic_codes, proposed_codes),
        'rank80_vs_fixed_aperture_cgh_code_mae': circular_code_mae(physical_codes, proposed_codes),
        'cgh_saturation': {'direct': physical_clipped, 'periodic': periodic_clipped,
                           'rank80': proposed_clipped},
        'phase_metrics_against_periodic_reference': phase_info,
        'warning': 'Integer FFT shifts are periodic. Core translation is projection into the original basis. '
                   'Both differ from shifted J0 times the fixed square aperture. '
                   'Each Thread shifts all its Bessel modes together. This is a multi-Thread field synthesis check, not a 50 cm reach or timing result.',
    }
    with (out/'mode_errors.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(beam_rows[0]))
        writer.writeheader()
        writer.writerows(beam_rows)
    with (out/'shifts_pixels.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['thread', 'shift_x_px', 'shift_y_px'])
        writer.writerows((i, *map(int, pair)) for i, pair in enumerate(thread_shifts_px))
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(out/'comparison_fields.npz', direct=physical_direct,
                        periodic=periodic_field, rank80=proposed_field,
                        direct_replay=physical_replay,
                        periodic_replay=periodic_replay,
                        rank80_replay=proposed_replay,
                        thread_shifts_px=thread_shifts_px,
                        thread_weights=thread_weights)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fields = [physical_direct, periodic_field, proposed_field]
    titles = ['Direct J0 × fixed aperture', 'Uncompressed FFT periodic shift',
              'Rank80 shifted core projection']
    vmax = max(float(np.abs(f).max()**2) for f in fields)
    for j, (field, title) in enumerate(zip(fields, titles)):
        axes[0, j].imshow(np.abs(field)**2, cmap='gray', origin='lower',
                          vmin=0, vmax=vmax)
        axes[0, j].set_title(title)
    axes[1, 0].imshow(np.abs(physical_direct-periodic_field), cmap='gray', origin='lower')
    axes[1, 0].set_title('Physical vs periodic: amplitude difference')
    axes[1, 1].imshow(np.abs(periodic_field-proposed_field), cmap='gray', origin='lower')
    axes[1, 1].set_title('Periodic vs rank80: amplitude difference')
    valid, delta, _ = phase_metrics(periodic_field, proposed_field)
    axes[1, 2].imshow(np.ma.array(delta, mask=~valid), cmap='twilight',
                      origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('Rank80 phase error [rad], reference bright pixels')
    fig.savefig(out/'field_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for j, (codes, title) in enumerate(zip(
            [physical_codes, periodic_codes, proposed_codes],
            ['Direct fixed-aperture CGH', 'Uncompressed periodic CGH',
             'Rank80 shifted-core CGH'])):
        axes[0, j].imshow(codes, cmap='gray', vmin=0, vmax=255,
                          interpolation='nearest')
        axes[0, j].set_title(title)
    periodic_code_diff = np.abs((periodic_codes.astype(np.int16)
                                 - proposed_codes.astype(np.int16)+128) % 256-128)
    physical_code_diff = np.abs((physical_codes.astype(np.int16)
                                 - proposed_codes.astype(np.int16)+128) % 256-128)
    axes[1, 0].imshow(physical_code_diff, cmap='gray')
    axes[1, 0].set_title('CGH code difference: direct vs rank80')
    axes[1, 1].imshow(periodic_code_diff, cmap='gray')
    axes[1, 1].set_title('CGH code difference: periodic vs rank80')
    axes[1, 2].imshow(np.abs(np.abs(physical_replay)**2
                             - np.abs(proposed_replay)**2), cmap='gray')
    axes[1, 2].set_title('Replay intensity difference: direct vs rank80')
    fig.savefig(out/'cgh_validation.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    plots = [(uy, 'U_ky'), (factor_shift_y.real, 'Re(A_y)'),
             (factor_shift_y.imag, 'Im(A_y)'), (ux, 'U_kx'),
             (factor_shift_x.real, 'Re(A_x)'), (factor_shift_x.imag, 'Im(A_x)')]
    for ax, (matrix, title) in zip(axes.flat, plots):
        im = ax.imshow(matrix, cmap='gray', aspect='auto')
        ax.set(title=title, xlabel='column', ylabel='row')
        fig.colorbar(im, ax=ax)
    fig.savefig(out/'factor_matrices.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report
