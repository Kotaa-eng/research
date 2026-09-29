"""保存済み辞書をメモリへ読んだ後のオンライン処理を段階別に計測する。

import時には計算しない。図・ディスク読込・Tucker分解・演算子の保存は計測外。
direct j0は固定照明、その他のshiftは周期的帯域制限移動なので、
速度と精度を同一の物理基準と誤認しないこと。
cached_spatialは連続位置に対応させるため、保存済み空間場をオンラインで
FFTしてから周期移動する。整数画素専用のnp.rollとは異なる比較である。
"""
from time import perf_counter

import numpy as np
from scipy.special import j0

from lsh_core import (relative_error, shift_operators, spatial_factors)


def benchmark_online(uy, ux, cores, cached_fields, cached_fft, kr, coefficients,
                     positions_xy_m, output_pitch, aperture, codec,
                     repeats=3, warmups=1, cached_operators=None):
    """5方式を同じ係数・位置・照明・CGH処理で比較。戻り値は秒と誤差の辞書。"""
    coefficients = np.asarray(coefficients)
    positions = np.asarray(positions_xy_m, dtype=float)
    count, ny, nx = cached_fft.shape
    if (cached_fields.shape != cached_fft.shape or aperture.shape != (ny, nx)
            or len(kr) != count or cores.shape[0] != count
            or coefficients.shape != (len(positions), count)
            or positions.shape[1:] != (2,)):
        raise ValueError('辞書、係数、位置、開口の形状を一致させてください。')
    vy, vx = spatial_factors(uy, ux)  # オフライン準備
    x = (np.arange(nx)-(nx-1)/2)*output_pitch
    y = (np.arange(ny)-(ny-1)/2)*output_pitch
    xx, yy = np.meshgrid(x, y)
    kx = 2*np.pi*np.fft.fftfreq(nx, output_pitch)
    ky = 2*np.pi*np.fft.fftfreq(ny, output_pitch)

    def timed(method):
        stages = {key: 0.0 for key in (
            'coefficient_core_weighted_sum', 'shift_operator_construction',
            'core_shift', 'core_summation', 'full_field_reconstruction',
            'ifft', 'phase_only_cgh_encoding', 'quantization')}
        start_total = perf_counter()
        if method.startswith('compressed'):
            total_core = np.zeros(cores.shape[1:], complex)
            for q, (weights, displacement) in enumerate(zip(coefficients, positions)):
                t = perf_counter()
                hq = np.tensordot(weights, cores, axes=(0, 0))
                stages['coefficient_core_weighted_sum'] += perf_counter()-t
                t = perf_counter()
                ay, ax = (shift_operators(uy, ux, output_pitch, displacement)
                          if cached_operators is None else cached_operators[q])
                stages['shift_operator_construction'] += perf_counter()-t
                t = perf_counter()
                shifted_core = ay @ hq @ ax
                stages['core_shift'] += perf_counter()-t
                t = perf_counter()
                total_core += shifted_core
                stages['core_summation'] += perf_counter()-t
            t = perf_counter()
            if method == 'compressed_spatial_basis':
                field = vy @ total_core @ vx.T
            else:
                full_fft = uy @ total_core @ ux.T
            stages['full_field_reconstruction'] = perf_counter()-t
            if method == 'compressed_fft_ifft':
                t = perf_counter()
                field = np.fft.ifft2(full_fft, norm='ortho')
                stages['ifft'] = perf_counter()-t
        elif method == 'direct_j0_fixed_illumination':
            t = perf_counter()
            field = np.zeros((ny, nx), complex)
            for weights, (dx, dy) in zip(coefficients, positions):
                radius = np.hypot(xx-dx, yy-dy)
                for weight, wave_number in zip(weights, kr):
                    field += weight*j0(wave_number*radius)*aperture
            stages['full_field_reconstruction'] = perf_counter()-t
        else:
            t = perf_counter()
            summed_fft = np.zeros((ny, nx), complex)
            for weights, (dx, dy) in zip(coefficients, positions):
                dy_phase = np.exp(-1j*ky*dy)[:, None]
                dx_phase = np.exp(-1j*kx*dx)[None, :]
                for mode, weight in enumerate(weights):
                    spectrum = (np.fft.fft2(cached_fields[mode], norm='ortho')
                                if method == 'cached_spatial' else cached_fft[mode])
                    summed_fft += weight*dy_phase*spectrum*dx_phase
            stages['full_field_reconstruction'] = perf_counter()-t
            t = perf_counter()
            field = np.fft.ifft2(summed_fft, norm='ortho')
            stages['ifft'] = perf_counter()-t
        t = perf_counter()
        phase, saturation = codec.encode_phase(field)
        stages['phase_only_cgh_encoding'] = perf_counter()-t
        t = perf_counter()
        codes = codec.quantize(phase)
        stages['quantization'] = perf_counter()-t
        return field, codes, saturation, stages, perf_counter()-start_total

    methods = ('direct_j0_fixed_illumination', 'cached_spatial', 'cached_full_fft',
               'compressed_fft_ifft', 'compressed_spatial_basis')
    for _ in range(warmups):
        for method in methods:
            timed(method)
    samples = {method: [] for method in methods}
    final = {}
    for _ in range(repeats):
        for method in methods:
            field, codes, saturation, stages, total = timed(method)
            samples[method].append({**stages, 'total_online': total})
            final[method] = (field, codes, saturation)
    reference = final['cached_full_fft'][0]
    reference_codes = final['cached_full_fft'][1]
    reference_replay = codec.replay(reference_codes)  # 評価のみ、オンライン時間から除外
    replayed = {method: codec.replay(final[method][1]) for method in methods}
    return {
        method: {
            'median_seconds': {key: float(np.median([sample[key] for sample in samples[method]]))
                               for key in samples[method][0]},
            'field_error_vs_periodic_cached_fft': relative_error(reference, final[method][0]),
            'field_error_vs_direct_fixed_illumination': relative_error(
                final['direct_j0_fixed_illumination'][0], final[method][0]),
            'saturation_fraction': final[method][2],
            'cgh_code_circular_mae_8bit': float(np.mean(np.abs(
                (final[method][1].astype(np.int16)-reference_codes.astype(np.int16)
                 +128) % 256-128))),
            'filtered_first_order_replay_field_error_vs_periodic': relative_error(
                reference_replay, replayed[method]),
            'filtered_first_order_replay_intensity_error_vs_periodic': relative_error(
                np.abs(reference_replay)**2,
                np.abs(replayed[method])**2),
        } for method in methods
    }
