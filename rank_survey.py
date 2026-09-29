# 日本語説明：候補ランクごとの復元誤差・CGH再生誤差・容量を調べ、設定閾値を満たす最小の調査済みランクを報告します。最大ランクの基底の先頭列を共用します。
"""Survey Tucker spatial ranks without executing anything on import.
Run from the notebook after the centered library and FFT tensor are available.
One shared basis is fitted at the largest candidate rank; smaller ranks use its
leading columns. This is a nested-basis survey, not independent SVD fits.
"""
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from lsh_core import PhaseOnlyCodec, fit_tucker
from PIL import Image
from thread_cgh import axial_intensity


def _error(reference, value):
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(value-reference)/norm) if norm else float('inf')


def survey_ranks(fft_tensor, bessel_fft, weights, kr, pixel_pitch, wavelength,
                 design_length, ranks, thresholds, output_dir='rank_survey',
                 evaluate_axial=False, axial_samples=41, slm_pixel_pitch=None,
                 padding_factor=2, config_id=None):
    ranks = sorted(set(int(r) for r in ranks))
    count, ny, nx = bessel_fft.shape
    if ny != nx or not ranks or ranks[0] < 1 or ranks[-1] > min(ny, nx):
        raise ValueError('Use positive ranks no larger than either spatial dimension.')
    if fft_tensor.shape != (2, count, ny, nx) or len(weights) != count:
        raise ValueError('Library, FFT tensor and weights must have matching shapes.')
    required = {'beam_max_error', 'thread_field_error', 'thread_intensity_error',
                'cgh_replay_field_error'}
    if evaluate_axial:
        required.add('axial_replay_intensity_error')
    if set(thresholds) != required or any(v <= 0 for v in thresholds.values()):
        raise ValueError(f'Threshold keys must be {sorted(required)}, with positive values.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(weights, dtype=np.complex128)
    maximum = ranks[-1]
    setup_start = perf_counter()
    print(f'Fitting common spatial bases once at maximum rank {maximum}...', flush=True)
    uy_saved, ux_saved, cores_saved = fit_tucker(bessel_fft, maximum, fft_tensor)
    norm2 = np.asarray([np.vdot(s, s).real for s in bessel_fft], dtype=np.float64)
    setup_seconds = perf_counter()-setup_start
    if np.any(norm2 == 0):
        raise ValueError('Zero-energy beam in the library.')
    reference = np.fft.ifft2(np.tensordot(weights, bessel_fft, axes=(0, 0)), norm='ortho')
    scale = float(np.abs(reference).max())
    if not scale:
        raise ValueError('Thread has zero amplitude; use non-cancelling weights.')
    if slm_pixel_pitch is None:
        slm_pixel_pitch = pixel_pitch
    codec = PhaseOnlyCodec(reference.shape, pixel_pitch, slm_pixel_pitch, kr, scale)
    def encode(field):
        _, codes, saturation = codec.encode(field)
        return codes, saturation
    replay = codec.replay
    reference_codes, _ = encode(reference)
    reference_replay = replay(reference_codes)
    Image.fromarray(reference_codes).save(out/'cgh_reference.png')
    reference_intensity = np.abs(reference)**2
    z = np.linspace(0, design_length, axial_samples)
    reference_axial = (axial_intensity(reference_replay, pixel_pitch, wavelength, z, padding_factor)
                       if evaluate_axial else None)
    rows = []
    beam_rows = []
    beam_errors = []
    # 全モードを保存精度の因子・コアから復元して照合する。
    for rank in ranks:
        # エネルギー差の推定ではなく、保存精度で全モードを直接照合。
        errors = np.asarray([
            _error(spectrum, uy_saved[:, :rank] @ cores_saved[i, :rank, :rank]
                   @ ux_saved[:, :rank].T)
            for i, spectrum in enumerate(bessel_fft)])
        worst = int(np.argmax(errors))
        start = perf_counter()
        core_sum = np.tensordot(weights, cores_saved[:, :rank, :rank], axes=(0, 0))
        field = np.fft.ifft2(uy_saved[:, :rank] @ core_sum @ ux_saved[:, :rank].T, norm='ortho')
        synthesis_seconds = perf_counter()-start
        codes, saturation = encode(field)
        reconstructed = replay(codes)
        code_diff = (codes.astype(int)-reference_codes.astype(int)+128)%256-128
        storage = count*rank*rank*8+(ny+nx)*rank*4  # intended complex64/float32 storage
        row = {
            'rank': rank,
            'beam_max_error': float(np.max(errors)),
            'beam_mean_error': float(np.mean(errors)),
            'worst_beam_index': worst,
            'worst_beam_kr_per_m': float(kr[worst]),
            'thread_field_error': _error(reference, field),
            'thread_intensity_error': _error(reference_intensity, np.abs(field)**2),
            'cgh_replay_field_error': _error(reference_replay, reconstructed),
            'cgh_code_circular_mae': float(np.mean(np.abs(code_diff))),
            'amplitude_clipped_pixel_fraction': saturation,
            'compressed_storage_MiB': storage/2**20,
            'full_to_compressed_ratio': (count*ny*nx*8)/storage,
            'synthesis_single_seconds': synthesis_seconds,
        }
        if evaluate_axial:
            row['axial_replay_intensity_error'] = _error(
                reference_axial, axial_intensity(reconstructed, pixel_pitch, wavelength, z, padding_factor))
        row['pass'] = all(row[key] <= limit for key, limit in thresholds.items())
        rows.append(row)
        beam_errors.append(errors)
        for index, value in enumerate(errors):
            beam_rows.append({'rank':rank, 'beam_index':index,
                              'kr_per_m':float(kr[index]), 'relative_field_error':float(value)})
        Image.fromarray(codes).save(out/f'cgh_rank_{rank}.png')
        np.savez_compressed(out/f'compressed_rank_{rank}.npz',
                            factor_ky=uy_saved[:, :rank],
                            factor_kx=ux_saved[:, :rank],
                            core_complex=cores_saved[:, :rank, :rank],
                            kr=kr, weights=weights, pixel_pitch=pixel_pitch,
                            slm_pixel_pitch=slm_pixel_pitch,
                            wavelength=wavelength, rank=rank,
                            config_id='' if config_id is None else config_id)
        # Reload the exported representation and check the actual persisted field.
        with np.load(out/f'compressed_rank_{rank}.npz') as saved:
            saved_sum = np.tensordot(saved['weights'], saved['core_complex'], axes=(0, 0))
            saved_field = np.fft.ifft2(saved['factor_ky'] @ saved_sum @ saved['factor_kx'].T,
                                       norm='ortho')
        row['persisted_thread_field_error'] = _error(reference, saved_field)
        row['pass'] = row['pass'] and row['persisted_thread_field_error'] <= thresholds['thread_field_error']
        print(f'rank={rank:4d}: worst beam={row["beam_max_error"]:.3%}, '
              f'Thread={row["thread_field_error"]:.3%}, '
              f'CGH replay={row["cgh_replay_field_error"]:.3%}, pass={row["pass"]}', flush=True)
    passing = [row['rank'] for row in rows if row['pass']]
    selected = min(passing) if passing else None
    # Report separately whether all tested larger ranks also pass; quantized CGH
    # errors need not decrease monotonically with rank.
    stable = [row['rank'] for i, row in enumerate(rows)
              if all(other['pass'] for other in rows[i:])]
    report = {'minimum_tested_passing_rank':selected,
              'minimum_rank_with_all_tested_larger_ranks_passing':min(stable) if stable else None,
              'thresholds':thresholds, 'rows':rows, 'basis_fit_seconds':setup_seconds,
              'reference_encoding_filter_error':_error(reference, reference_replay),
              'scope':'Centered modes and this coaxial Thread on the current finite aperture. '
                      'CGH metrics compare compression against the same uncompressed encoding. '
                      'No guarantee for translated threads, other coefficients, target fidelity or a physical SLM.',
              'method':'Single maximum-rank randomized SVD, nested subspaces. '
                       'Uncompressed finite-aperture library is the reference. '
                       'Every beam is reconstructed and compared at stored precision.',
              'storage_precision':'Exported cores complex64 and bases float32; Thread and CGH use these same rounded arrays, '
                                  'persisted Thread field error is separately checked.'}
    (out/'rank_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for filename, data in [('rank_results.csv',rows),('beam_errors.csv',beam_rows)]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for key,label in [('beam_max_error','Worst beam'),('thread_field_error','Thread field'),
                      ('thread_intensity_error','Thread intensity'),('cgh_replay_field_error','CGH replay')]:
        axes[0].semilogy(ranks,[max(row[key]*100,1e-10) for row in rows],'-o',label=label)
        axes[0].axhline(thresholds[key]*100,color='0.7',ls=':',lw=.8)
    if evaluate_axial:
        axes[0].semilogy(ranks,[max(row['axial_replay_intensity_error']*100,1e-10) for row in rows],'-o',label='Axial replay intensity')
    axes[0].set(xlabel='Spatial rank (same in x and y)',ylabel='Relative error [%]',title='Candidate rank vs reconstruction error')
    axes[0].legend();axes[0].grid(alpha=.3)
    im=axes[1].imshow(np.asarray(beam_errors)*100,aspect='auto',origin='lower',cmap='magma')
    axes[1].set_yticks(range(len(ranks)),labels=ranks)
    axes[1].set(xlabel='Bessel mode index',ylabel='Rank',title='Per-mode field error [%]')
    fig.colorbar(im,ax=axes[1])
    fig.savefig(out/'rank_error_summary.png',dpi=160)
    plt.show()
    print('Minimum tested rank satisfying every criterion:', selected)
    if selected is None:
        print('No passing candidate. Extend SURVEY_RANKS; do not infer a threshold from the current sweep.')
    else:
        print('Refine the ranks between the previous failing candidate and this candidate for a finer threshold.')
    print('Reference CGH encoding/filter error:', report['reference_encoding_filter_error'])
    return report
