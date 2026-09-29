"""4f出力面 z=0 から50 cmを対象とする半径別Tucker復元調査。

このファイルをimportしただけでは計算しない。ノートブックから実行する。
幾何学的な有限開口の到達距離は目安であり、実機の保証ではない。
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0
from lsh_core import fit_tucker
from tensorly import unfold


J0_FIRST_ZERO = 2.404825557695773


def design_modes(r0_output, wavelength, length, output_half_aperture,
                 edge_fraction, output_pitch, magnification):
    """出力面のFW全モードを返す。到達条件は診断のみで、モードを黙って除外しない。"""
    if min(r0_output, wavelength, length, output_half_aperture,
           output_pitch, magnification) <= 0 or not 0 <= edge_fraction < 1:
        raise ValueError('半径・波長・長さ・ピッチ・倍率は正、edge_fractionは[0,1)にしてください。')
    k = 2*np.pi/wavelength
    kr0 = J0_FIRST_ZERO/r0_output
    if kr0 >= k:
        raise ValueError('中心モードが伝搬波になりません。')
    q = np.sqrt(k*k-kr0*kr0)
    max_m = int(np.floor(min(k-q, q)*length/(2*np.pi)))
    m = np.arange(-max_m, max_m+1, dtype=int)
    kz = q+2*np.pi*m/length
    kr = np.sqrt(np.maximum(0, (k-kz)*(k+kz)))
    # 外側10%のテーパを除く、内側に完全に残る円形領域を保守的に採用。
    usable_radius = output_half_aperture*(1-edge_fraction)
    reach = usable_radius*kz/kr
    sampled = kr < np.pi/output_pitch
    # 4fの横方向周波数倍率。SLMではkr_slm=M*kr_output。
    slm_sampled = magnification*kr < np.pi/(output_pitch/magnification)
    ok = sampled & slm_sampled
    if not np.all(ok):
        raise ValueError('全モードをサンプリングできません。ピッチを見直してください。')
    return {'m': m[ok], 'kr': kr[ok], 'kz': kz[ok],
            'all_m': m, 'all_kr': kr, 'all_reach': reach,
            'q': q, 'kr0': kr0, 'center_reach': usable_radius*q/kr0,
            'usable_radius': usable_radius}


def study_radii(r0_output_um=(30, 60, 120, 150), ranks=(32, 64, 128, 192, 256, 300),
                n=1024, slm_pitch=8e-6, magnification=0.25,
                wavelength=532e-9, length=0.50, edge_fraction=0.10,
                output_dir='radius_tucker_outputs', bright_fraction=(0.15, 0.65)):
    """半径ごとにFFTテンソルを作り、空間2方向のTucker近似誤差を保存。

    r0は4f出力面における中心J0モードの第1零点半径。全モード同半径ではない。
    圧縮比・誤差は各半径の辞書全体および最悪モードで評価する。
    """
    ranks = sorted(set(int(v) for v in ranks))
    start_fraction, end_fraction = bright_fraction
    if not 0 <= start_fraction < end_fraction <= 1:
        raise ValueError('目標明区間は0～1の昇順の範囲で設定してください。')
    if not ranks or ranks[0] < 1 or ranks[-1] > n:
        raise ValueError('ランクは1～画素数の範囲にしてください。')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_pitch = magnification*slm_pitch
    half_aperture = (n-1)*output_pitch/2
    x = (np.arange(n)-(n-1)/2)*output_pitch
    xx, yy = np.meshgrid(x, x)
    rr = np.hypot(xx, yy)
    start = half_aperture*(1-edge_fraction)
    u = np.clip((np.abs(x)-start)/(half_aperture-start), 0, 1) if edge_fraction else np.zeros_like(x)
    taper = .5*(1+np.cos(np.pi*u))
    aperture = (taper[:, None]*taper[None, :]).astype(np.float32)
    rows = []
    for radius_um in r0_output_um:
        radius = float(radius_um)*1e-6
        d = design_modes(radius, wavelength, length, half_aperture,
                         edge_fraction, output_pitch, magnification)
        print(f'r0(out)={radius_um:g} μm; r0(SLM)={radius_um/magnification:g} μm; '
              f'center reach≈{d["center_reach"]:.3f} m; '
              f'sampled modes={len(d["m"])} / {len(d["all_m"])}', flush=True)
        beams = np.stack([(j0(float(kr)*rr)*aperture).astype(np.complex64)
                          for kr in d['kr']])
        spectra = np.fft.fft2(beams, axes=(-2, -1), norm='ortho')
        tensor = np.stack((spectra.real, spectra.imag)).astype(np.float32)
        uy, ux, cores = fit_tucker(spectra, ranks[-1], tensor)
        del tensor
        beam_norm2 = np.array([np.vdot(s.astype(np.complex128),
                                      s.astype(np.complex128)).real for s in spectra])
        # 主検証から目標明区間も受け取る。半径による変更はモード数・krのみ。
        width = end_fraction-start_fraction
        center = (end_fraction+start_fraction)/2
        weights = width*np.sinc(d['m']*width)*np.exp(-2j*np.pi*d['m']*center)
        exact_thread_fft = np.tensordot(weights, spectra, axes=(0, 0))
        thread_norm = np.linalg.norm(exact_thread_fft)
        radius_folder = out / f'r0_output_{float(radius_um):g}um'
        radius_folder.mkdir(exist_ok=True)
        for rank in ranks:
            compressed_cores = cores[:, :rank, :rank].astype(np.complex64)
            saved_uy = uy[:, :rank].astype(np.float32)
            saved_ux = ux[:, :rank].astype(np.float32)
            # 保存精度で復元し、丸めを含む最悪ビーム誤差を実測。
            errors = []
            total_error2 = 0.0
            for i, spectrum in enumerate(spectra):
                reconstructed = saved_uy @ compressed_cores[i] @ saved_ux.T
                error2 = float(np.vdot(spectrum-reconstructed,
                                       spectrum-reconstructed).real)
                errors.append(np.sqrt(error2/beam_norm2[i]))
                total_error2 += error2
            thread_core = np.tensordot(weights, compressed_cores, axes=(0, 0))
            restored_thread_fft = saved_uy @ thread_core @ saved_ux.T
            thread_error = np.linalg.norm(exact_thread_fft-restored_thread_fft)/thread_norm
            rows.append({'r0_output_um': radius_um, 'rank': rank, 'status': '評価済み',
                         'center_reach_m': d['center_reach'], 'modes': len(d['m']),
                         'all_modes_reach_length': bool(np.all(d['all_reach'] >= length)),
                         'worst_beam_error': max(errors),
                         'library_error': np.sqrt(total_error2/beam_norm2.sum()),
                         'thread_field_error': float(thread_error)})
            print(f'  rank={rank}: worst={max(errors):.3%}, '
                  f'library={rows[-1]["library_error"]:.3%}, '
                  f'Thread={thread_error:.3%}', flush=True)
        # ビームごとの実データ・係数は巨大になり得るので保存しない。
        fig, ax = plt.subplots(figsize=(7, 4))
        subset = [row for row in rows if row['r0_output_um'] == radius_um
                  and row['status'] == '評価済み']
        for key, label in [('worst_beam_error', 'Worst beam'),
                           ('library_error', 'Whole library'),
                           ('thread_field_error', 'Coaxial Thread')]:
            ax.semilogy([r['rank'] for r in subset],
                        [max(r[key], 1e-12) for r in subset], marker='o', label=label)
        ax.set(xlabel='Tucker spatial rank', ylabel='Relative complex-field L2 error',
               title=f'r0(out)={radius_um:g} μm; 0–{length:.2f} m')
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(radius_folder/'rank_errors.png', dpi=150)
        plt.close(fig)
    with (out/'radius_rank_errors.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['r0_output_um', 'rank', 'status',
                                'center_reach_m', 'modes', 'all_modes_reach_length', 'worst_beam_error',
                                'library_error', 'thread_field_error'])
        writer.writeheader()
        writer.writerows(rows)
    return rows
