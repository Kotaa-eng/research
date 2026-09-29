"""全Besselモードの理論値と圧縮復元値を比較する画像を保存・表示します。

インポート時に計算は行いません。ノートブックからexport_all_modesを呼びます。
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from lsh_core import phase_metrics, relative_error


def export_all_modes(bessel_centered, bessel_fft, core_complex,
                     factor_ky, factor_kx, kr, mode_index, pixel_pitch,
                     output_dir, show=True):
    """各モードを一つずつ復元し、比較図と全画素グレースケールPNGを出力。

    空間振幅・FFT振幅はモードごとに理論/圧縮共通の線形スケールです。
    FFT振幅の対数表示も別途保存します。位相は±πが同色の循環色で表示。
    マスクは理論振幅のみで決めます。復元値が暗くなる失敗を隠しません。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count, ny, nx = bessel_centered.shape
    if (bessel_fft.shape != (count, ny, nx)
            or len(kr) != count or len(mode_index) != count
            or core_complex.shape[0] != count
            or factor_ky.shape != (ny, core_complex.shape[1])
            or factor_kx.shape != (nx, core_complex.shape[2])):
        raise ValueError('ビーム辞書・コア・共通因子の形状が一致しません。')
    spatial_extent = np.array([-nx/2, nx/2, -ny/2, ny/2])*pixel_pitch*1e3
    fx = np.fft.fftshift(np.fft.fftfreq(nx, pixel_pitch))/1e3
    fy = np.fft.fftshift(np.fft.fftfreq(ny, pixel_pitch))/1e3
    dfx, dfy = 1/(nx*pixel_pitch)/1e3, 1/(ny*pixel_pitch)/1e3
    fft_extent = [fx[0]-dfx/2, fx[-1]+dfx/2,
                  fy[0]-dfy/2, fy[-1]+dfy/2]
    for i in range(count):
        # 合成Threadではなく、保存された各ビームのコアを個別に復元します。
        reconstructed_fft = factor_ky @ core_complex[i] @ factor_kx.T
        reconstructed = np.fft.ifft2(reconstructed_fft, norm='ortho')
        fields = [bessel_centered[i], reconstructed]
        spectra = [np.fft.fftshift(bessel_fft[i]),
                   np.fft.fftshift(reconstructed_fft)]
        amplitudes = [np.abs(v) for v in fields]
        fft_amplitudes = [np.abs(v) for v in spectra]
        amplitude_scale = max(float(v.max()) for v in amplitudes)
        fft_scale = max(float(v.max()) for v in fft_amplitudes)
        fft_component_scale = max(float(np.max(np.abs(component)))
                                  for spectrum in spectra
                                  for component in (spectrum.real, spectrum.imag))
        amplitude_scale = max(amplitude_scale, np.finfo(float).tiny)
        fft_scale = max(fft_scale, np.finfo(float).tiny)
        fft_component_scale = max(fft_component_scale, np.finfo(float).tiny)
        folder = out / f'mode_{i:02d}_m{int(mode_index[i]):+d}'
        folder.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 5, figsize=(21, 8))
        titles = ['Amplitude', 'Phase [rad]', 'FFT amplitude',
                  'FFT amplitude [dB]', 'FFT phase [rad]']
        # 真の零点付近では微小な複素誤差でも位相が任意に飛ぶ。
        # 理論・復元の両方に同じ有効画素マスクを適用する。
        valid, phase_delta, detailed_metrics = phase_metrics(fields[0], fields[1])
        spatial_mask = ~valid
        fft_mask = fft_amplitudes[0] <= 0.001*fft_amplitudes[0].max()
        phase_rmse = detailed_metrics['phase_rmse_rad']
        phase_rmse = float('nan') if phase_rmse is None else phase_rmse
        fft_roundtrip_error = (np.linalg.norm(np.fft.fft2(fields[1], norm='ortho')
                                              - reconstructed_fft)
                               / np.linalg.norm(reconstructed_fft))
        detailed_metrics['fft_complex_relative_L2'] = relative_error(
            bessel_fft[i], reconstructed_fft)
        detailed_metrics['parseval_absolute_gap'] = abs(
            detailed_metrics['fft_complex_relative_L2']-detailed_metrics['field_relative_L2'])
        detailed_metrics['reference_fft_consistency'] = relative_error(
            np.fft.fft2(fields[0], norm='ortho'), bessel_fft[i])
        detailed_metrics['ifft_fft_roundtrip_relative_L2'] = float(fft_roundtrip_error)
        # ±πの境界を横切っただけの画素を数える。小振幅でなくても起こる。
        raw_difference = np.angle(fields[1])-np.angle(fields[0])
        detailed_metrics['branch_wrap_fraction_on_reference_bright_pixels'] = (
            float(np.mean((np.abs(raw_difference[valid]) > np.pi)
                          & (np.abs(phase_delta[valid]) < 1e-3))) if valid.any() else None)
        for row, label in enumerate(['theory', 'compressed']):
            phase = np.angle(fields[row])
            fft_phase = np.angle(spectra[row])
            db = 20*np.log10(np.maximum(fft_amplitudes[row]/fft_scale, 1e-6))
            panels = [amplitudes[row], np.ma.array(phase, mask=spatial_mask),
                      fft_amplitudes[row], db,
                      np.ma.array(fft_phase, mask=fft_mask)]
            limits = [(0, amplitude_scale), (-np.pi, np.pi),
                      (0, fft_scale), (-80, 0), (-np.pi, np.pi)]
            for col, (panel, (lo, hi)) in enumerate(zip(panels, limits)):
                ax = axes[row, col]
                im = ax.imshow(panel, origin='lower', cmap='twilight' if col in (1, 4) else 'gray',
                               extent=spatial_extent if col < 2 else fft_extent,
                               vmin=lo, vmax=hi, interpolation='nearest')
                ax.set_title(f'{label}: {titles[col]}')
                ax.set_xlabel('x [mm]' if col < 2 else 'fx [cycles/mm]')
                ax.set_ylabel('y [mm]' if col < 2 else 'fy [cycles/mm]')
                fig.colorbar(im, ax=ax, shrink=0.7)
            # 位相の白黒版はcos/sinも保存。±πが黒白に分離する誤解を避ける。
            images = {
                'amplitude': amplitudes[row]/amplitude_scale,
                'phase_wrapped_gray': np.where(spatial_mask, 0.5, (phase+np.pi)/(2*np.pi)),
                'fft_amplitude': fft_amplitudes[row]/fft_scale,
                'fft_intensity': (fft_amplitudes[row]/fft_scale)**2,
                'fft_amplitude_db': (np.clip(db, -80, 0)+80)/80,
                'fft_phase_wrapped_gray': np.where(fft_mask, 0.5, (fft_phase+np.pi)/(2*np.pi)),
                # Full FFT grid: no crop or interpolation. Middle gray is zero;
                # real and imaginary parts share one scale for both methods.
                'fft_real': 0.5 + 0.5*spectra[row].real/fft_component_scale,
                'fft_imag': 0.5 + 0.5*spectra[row].imag/fft_component_scale,
                'phase_cos': np.where(spatial_mask, 0.5, (np.cos(phase)+1)/2),
                'phase_sin': np.where(spatial_mask, 0.5, (np.sin(phase)+1)/2),
            }
            component_images = {}
            for name, pixels in images.items():
                component_images[name] = Image.fromarray(
                    np.rint(np.clip(pixels, 0, 1)*255).astype(np.uint8))
                component_images[name].save(folder / f'{label}_{name}.png')
            # Four full-resolution 1024x1024 panels (for a 1024x1024 input).
            # Order: amplitude, phase, real part, imaginary part.
            montage = Image.new('L', (2*nx, 2*ny))
            for name, position in [('fft_amplitude', (0, 0)),
                                   ('fft_phase_wrapped_gray', (nx, 0)),
                                   ('fft_real', (0, ny)),
                                   ('fft_imag', (nx, ny))]:
                montage.paste(component_images[name], position)
            montage.save(folder / f'{label}_fft_complex_components_full.png')
            for name, values, mask_values in [('phase', phase, spatial_mask),
                                              ('fft_phase', fft_phase, fft_mask)]:
                rgba = plt.get_cmap('twilight')((values+np.pi)/(2*np.pi))
                rgba[mask_values] = [0.5, 0.5, 0.5, 1]
                Image.fromarray(np.rint(rgba*255).astype(np.uint8)).save(
                    folder/f'{label}_{name}_cyclic.png')
                Image.fromarray(np.rint(rgba*255).astype(np.uint8)).save(
                    folder/f'{label}_{name}.png')
        error = np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[0])
        fig.suptitle(f'l=0, m={mode_index[i]}, kr={kr[i]:.3f} /m, '
                     f'rank={core_complex.shape[1]}, field error={error:.2e}, '
                     f'bright-phase RMSE={phase_rmse:.2e} rad')
        Image.fromarray((~spatial_mask).astype(np.uint8)*255).save(folder/'phase_valid_mask.png')
        phase_difference_image = np.where(spatial_mask, 0.5,
                                          (phase_delta+np.pi)/(2*np.pi))
        Image.fromarray(np.rint(np.clip(phase_difference_image, 0, 1)*255).astype(np.uint8)
                        ).save(folder/'phase_error_masked.png')
        phase_fig, phase_ax = plt.subplots(figsize=(5, 4))
        phase_im = phase_ax.imshow(np.ma.array(phase_delta, mask=spatial_mask),
                                  origin='lower', extent=spatial_extent,
                                  cmap='twilight', vmin=-np.pi, vmax=np.pi)
        phase_ax.set(title='Phase error: arg(Erec conj(Etheory))',
                     xlabel='x [mm]', ylabel='y [mm]')
        phase_fig.colorbar(phase_im, ax=phase_ax, label='rad')
        phase_fig.tight_layout()
        phase_fig.savefig(folder/'phase_error_masked_plot.png', dpi=140)
        plt.close(phase_fig)
        (folder/'numerical_checks.txt').write_text(
            json.dumps(detailed_metrics, indent=2, allow_nan=False), encoding='utf-8')
        fig.tight_layout()
        fig.savefig(folder / 'comparison.png', dpi=140)
        # 全周波数範囲ではリングが中心の数画素に縮んで見えるため、
        # 予測リング半径 kr/(2π) を基準に同じ倍率の拡大図を別途出力する。
        ring_frequency = float(kr[i])/(2*np.pi*1e3)  # cycles/mm
        # Use frequency units for the FFT panels. The ring occupies a small
        # fraction of the full Nyquist range because kr/(2*pi) is much lower
        # than 1/(2*pixel_pitch). Frame the ring with a 12% margin so that
        # its displayed diameter nearly fills each panel, without rescaling
        # or changing the FFT data itself.
        zoom_halfwidth = max(1.12*ring_frequency, 4*max(dfx, dfy))
        for row in range(2):
            for col in (2, 3, 4):
                axes[row, col].set_xlim(-zoom_halfwidth, zoom_halfwidth)
                axes[row, col].set_ylim(-zoom_halfwidth, zoom_halfwidth)
                axes[row, col].add_patch(plt.Circle((0, 0), ring_frequency,
                    fill=False, color='tab:red', linestyle='--', linewidth=1))
        fig.savefig(folder / 'comparison_fft_zoom.png', dpi=140)
        if show:
            plt.show()
        plt.close(fig)
        print(f'[{i+1}/{count}] {folder}')
    (out / '画像の見方.txt').write_text(
        'z=0の各モードを比較しています。Threadの加算結果やCGHではありません。\n'
        'FFTはfftshiftでゼロ周波数を中央に配置。振幅は線形と対数の両方を保存。\n'
        '振幅スケールは各モードの理論/圧縮で共通。異なるモード間は別スケールです。\n'
        '比較図と*_cyclic.pngの位相は±πが同色の循環色。従来の白黒位相は境界で白黒反転します。\n'
        '白黒で比較する場合はphase_cosとphase_sinを使ってください。値の丸め補正はしていません。\n'
        '位相マスクは理論振幅1%、FFT位相マスクは理論FFT振幅0.1%。復元値ではマスクしません。\n'
        'phase_valid_mask.pngは白が有効画素。numerical_checks.txtに誤差とFFT往復検査を記録。\n'
        'phase_error_masked_plot.pngはarg(Erec*conj(Etheory))を共通1%マスクで表示。\n'
        'FFT対数画像は-80～0 dBを白黒表示。\n'
        'comparison_fft_zoom.pngは予測リング半径kr/(2π)付近を拡大し、赤破線で示します。\n'
        'PNGは8bit表示用であり、誤差の数値評価用データではありません。\n',
        encoding='utf-8')
    return out
