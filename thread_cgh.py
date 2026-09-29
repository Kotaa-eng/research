# 日本語説明：理論場と圧縮復元場から位相のみのCGHを作り、回折次数の抽出による再生場と軸方向強度を比較する補助ファイルです。test.ipynbのCGH比較セルから呼び出します。
"""Phase-only encoding and crop assessment for the coaxial LSH test.
Arrizon sinusoidal encoding: J1(f(a)) = 0.58*a.
Carrier is inside the encoded phase argument, separating diffraction orders.
No SLM LUT/calibration or physical device control is performed.
"""
from pathlib import Path
import json
import numpy as np
from lsh_core import PhaseOnlyCodec, phase_metrics
from PIL import Image
import matplotlib.pyplot as plt


def relative_error(ref, value):
    return float(np.linalg.norm(ref-value) / np.linalg.norm(ref))


def axial_intensity(field, pixel_pitch, wavelength, z_values, padding_factor=2,
                    x_target=0.0, y_target=0.0):
    """On-axis angular spectrum. Padding convergence is not assumed."""
    n = field.shape[0]
    p = int(padding_factor)*n
    if p < 2*n or (p-n) % 2:
        raise ValueError('padding_factorは2以上、同一中心のグリッドにしてください。')
    pad = (p-n)//2
    freq = 2*np.pi*np.fft.fftfreq(p, pixel_pitch)
    k = 2*np.pi/wavelength
    transverse2 = freq[:, None]**2 + freq[None, :]**2
    delta_kz = -transverse2/(np.sqrt((k*k-transverse2).astype(np.complex128))+k)
    spectrum = np.fft.fft2(np.pad(field, ((pad, pad), (pad, pad))), norm='ortho')
    x_min = -(p-1)*pixel_pitch/2
    origin_y = np.exp(-1j*freq*(x_min-y_target))
    origin_x = np.exp(-1j*freq*(x_min-x_target))
    spectrum *= origin_y[:, None]*origin_x[None, :]
    return np.asarray([abs(np.sum(spectrum*np.exp(1j*delta_kz*z))/p)**2
                       for z in z_values])


def axial_rayleigh_sommerfeld(field, pixel_pitch, wavelength, z_values):
    """独立検査：有限開口のRS第1積分を軸上で直接求積。

    FFTの周期複製を持たない。核の位相変化が1画素でπを超えるzはNaNとし、
    この離散積分だけで検証できない領域を明示する。z=0も検証対象外。
    """
    n = field.shape[0]
    x = (np.arange(n)-(n-1)/2)*pixel_pitch
    rho2 = x[:, None]**2+x[None, :]**2
    k = 2*np.pi/wavelength
    values = []
    for z in z_values:
        if z <= 0 or k*max(abs(x))*pixel_pitch/np.sqrt(z*z+max(abs(x))**2) >= np.pi:
            values.append(np.nan)
            continue
        radius = np.sqrt(rho2+z*z)
        phase_distance = rho2/(radius+z)  # radius-zを安定に計算
        kernel = z*(1/radius-1j*k)/(2*np.pi*radius**2)*np.exp(1j*k*phase_distance)
        values.append(abs(np.sum(field*kernel)*pixel_pitch**2)**2)
    return np.asarray(values)


def run_comparison(theory, proposed, pixel_pitch, wavelength, kz, kr, weights,
                   design_length, output_dir='thread_outputs_v2',
                   run_crop_study=False, slm_pixel_pitch=None, padding_factor=2,
                   check_propagation=True):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    n = theory.shape[0]
    assert theory.shape == proposed.shape == (n, n)
    # 全手法で理論場由来の同一スケールと符号化処理を使用。
    scale = float(np.abs(theory).max())
    if slm_pixel_pitch is None:
        slm_pixel_pitch = pixel_pitch
    codec = PhaseOnlyCodec(theory.shape, pixel_pitch, slm_pixel_pitch, kr, scale)
    phase_t, codes_t, clipped_t = codec.encode(theory)
    phase_p, codes_p, clipped_p = codec.encode(proposed)
    replay = codec.replay
    filter_radius = codec.radius
    # Raw PNGs have exactly n x n pixels, no axes, no resizing.
    Image.fromarray(codes_t).save(out/'cgh_theory.png')
    Image.fromarray(codes_p).save(out/'cgh_compressed.png')
    circular_delta = np.angle(np.exp(1j*(phase_p-phase_t)))
    delta8 = np.abs((codes_p.astype(int)-codes_t.astype(int)+128)%256-128)
    replay_t, replay_p = replay(codes_t), replay(codes_p)
    metrics = {
        'relative_complex_field_error': relative_error(theory, proposed),
        'relative_intensity_error': relative_error(np.abs(theory)**2,np.abs(proposed)**2),
        'cgh_circular_phase_rmse_rad': float(np.sqrt(np.mean(circular_delta**2))),
        'cgh_8bit_circular_code_mae': float(np.mean(delta8)),
        'replayed_field_compression_error': relative_error(replay_t,replay_p),
        'encoding_filter_error_theory': relative_error(theory,replay_t),
        'encoding_filter_error_compressed': relative_error(proposed,replay_p),
        'common_amplitude_scale':scale,
        'amplitude_clipped_pixel_fraction':clipped_p,
        'carrier_period_pixels':4,
        'filter_radius_per_m':filter_radius,
        'slm_pixel_pitch_m':slm_pixel_pitch,
        'output_pixel_pitch_m':pixel_pitch,
        'note':'Simulation only; square ROI, ideal uniform illumination, 8-bit linear phase, diagonal carrier, circular first-order filter. Not a calibrated SLM-ready file.'
    }
    metrics['thread_phase_metrics'] = phase_metrics(theory, proposed)[2]
    fig, axes = plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,codes,title in zip(axes[0,:2],[codes_t,codes_p],['Theory CGH (8 bit)','Compressed CGH (8 bit)']):
        ax.imshow(codes,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(title)
    im=axes[0,2].imshow(np.abs(circular_delta),cmap='gray',vmin=0)
    axes[0,2].set_title('Circular CGH phase difference [rad]')
    fig.colorbar(im,ax=axes[0,2],shrink=.8)
    c=n//2; width=64; crop=np.s_[c-width:c+width,c-width:c+width]
    axes[1,0].imshow(codes_t[crop],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    axes[1,0].set_title('Theory: central 128 x 128 pixels')
    axes[1,1].imshow(codes_p[crop],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    axes[1,1].set_title('Compressed: central 128 x 128 pixels')
    replay_intensity_diff=np.abs(np.abs(replay_t)**2-np.abs(replay_p)**2)
    im=axes[1,2].imshow(replay_intensity_diff,cmap='gray')
    axes[1,2].set_title('Intensity difference after order filtering')
    fig.colorbar(im,ax=axes[1,2],shrink=.8)
    for ax in axes.flat: ax.set_xlabel('x pixel');ax.set_ylabel('y pixel')
    fig.suptitle('One coaxial thread: identical encoding, normalization and filter')
    fig.savefig(out/'cgh_comparison.png',dpi=150)
    plt.show()

    # Use the same amplitude scale for all curves; do not independently normalize.
    z_values = np.linspace(0, design_length, 81)
    ideal = np.abs(np.exp(1j*np.outer(z_values, kz-kz[len(kz)//2])) @ weights)**2
    reference_peak = max(float(ideal.max()), 1e-30)
    axial_curves = {
        'Theory, finite aperture': axial_intensity(theory, pixel_pitch, wavelength, z_values, padding_factor),
        'Compressed, finite aperture': axial_intensity(proposed, pixel_pitch, wavelength, z_values, padding_factor),
        'Theory CGH replay': axial_intensity(replay_t, pixel_pitch, wavelength, z_values, padding_factor),
        'Compressed CGH replay': axial_intensity(replay_p, pixel_pitch, wavelength, z_values, padding_factor),
    }
    metrics['axial_compression_intensity_error'] = relative_error(
        axial_curves['Theory, finite aperture'], axial_curves['Compressed, finite aperture'])
    metrics['axial_cgh_replay_compression_error'] = relative_error(
        axial_curves['Theory CGH replay'], axial_curves['Compressed CGH replay'])
    metrics['axial_aperture_error_vs_ideal'] = relative_error(
        ideal, axial_curves['Theory, finite aperture'])
    metrics['propagation_padding_factor'] = padding_factor
    metrics['propagation_validation'] = 'not_checked'
    rs_curve = np.full_like(z_values, np.nan)
    if check_propagation:
        rs_curve = axial_rayleigh_sommerfeld(theory, pixel_pitch, wavelength, z_values)
        valid = np.isfinite(rs_curve)
        if np.any(valid):
            discrepancy = relative_error(rs_curve[valid], axial_curves['Theory, finite aperture'][valid])
            metrics['propagation_AS_vs_RS_relative_error'] = discrepancy
            metrics['propagation_validation'] = (
                'agreement_within_1pct_on_checked_samples' if discrepancy <= .01
                else 'NOT_CONVERGED: increase padding and check sampling')
            metrics['propagation_checked_z_min_m'] = float(z_values[valid].min())
        metrics['propagation_unchecked_sample_count'] = int((~valid).sum())
    metrics['scope'] = ('One coaxial Thread, same finite aperture. Compression accuracy, '
                        'target fidelity and propagation validation are separate. '
                        'This does not validate a multi-thread LSH or 50 cm reach.')
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(z_values*1000, ideal/reference_peak, 'k--', label='Ideal Bessel sum')
    for name, curve in axial_curves.items():
        ax.plot(z_values*1000, curve/reference_peak, label=name)
    if check_propagation:
        ax.plot(z_values*1000, rs_curve/reference_peak, 'o', ms=3,
                markerfacecolor='none', label='Finite aperture: independent RS check')
    ax.set(xlabel='z [mm, from 4f output plane]',
           ylabel='Intensity / ideal peak (common scale)',
           title='Coaxial thread: ideal vs finite aperture and compression')
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'thread_axial_comparison.png', dpi=150)
    plt.show()
    np.savez_compressed(out/'thread_axial_profiles.npz', z=z_values, ideal=ideal,
                        theory=axial_curves['Theory, finite aperture'],
                        compressed=axial_curves['Compressed, finite aperture'],
                        replay_theory=axial_curves['Theory CGH replay'],
                        replay_compressed=axial_curves['Compressed CGH replay'],
                        finite_aperture_rs=rs_curve)
    if not run_crop_study:
        metrics['crop_study'] = 'Disabled; enable RUN_CROP_STUDY in the notebook.'
        (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(metrics, indent=2))
        return metrics

    # Crop the same physical field, embed at the same location; do not rescale.
    # Padded propagation reduces wraparound relative to a 1024-only propagation.
    # This is a convergence check relative to current 1024 reference, NOT proof
    # of convergence to an infinite-aperture or paper Bessel-Gauss field.
    sizes=[s for s in [128,256,512,768,1024] if s<=n]
    p=2*n; pad=(p-n)//2
    fpad=2*np.pi*np.fft.fftfreq(p,pixel_pitch)
    k0=2*np.pi/wavelength
    kperp2=fpad[:,None]**2+fpad[None,:]**2
    # Remove common carrier exp(i*k0*z) and use a stable difference.
    kz_minus_k=-kperp2/(np.sqrt(k0*k0-kperp2)+k0)
    z_values=np.linspace(0,design_length,41)
    center_samples=[]; crop_rows=[]
    retained=[]
    for size in sizes:
        cut=(n-size)//2
        cropped=np.zeros_like(theory)
        cropped[cut:cut+size,cut:cut+size]=theory[cut:cut+size,cut:cut+size]
        retained.append(float(np.sum(np.abs(cropped)**2, dtype=np.float64)
                              / np.sum(np.abs(theory)**2, dtype=np.float64)))
        padded=np.pad(cropped,((pad,pad),(pad,pad)))
        spec=np.fft.fft2(padded,norm='ortho')
        # Evaluate exactly x=y=0 for a half-pixel centered coordinate grid.
        x0=-(p-1)*pixel_pitch/2
        origin_phase=np.exp(-1j*fpad*x0)
        origin_spectrum=spec*origin_phase[:,None]*origin_phase[None,:]
        axial=[]
        for z in z_values:
            axial.append(np.sum(origin_spectrum*np.exp(1j*kz_minus_k*z))/p)
        center_samples.append(np.abs(axial)**2)
        print(f'Crop propagation {size} x {size} complete',flush=True)
    ref_axial=center_samples[-1]
    peak=float(ref_axial.max())
    for size,energy,axial in zip(sizes,retained,center_samples):
        crop_rows.append({'size':size,'width_mm':size*pixel_pitch*1e3,
                          'retained_initial_energy':energy,
                          'initial_field_crop_error':float(np.sqrt(max(0,1-energy))),
                          'axial_intensity_error_vs_1024':relative_error(ref_axial,np.asarray(axial))})
    ideal=np.abs(np.exp(1j*np.outer(z_values,kz-kz[len(kz)//2]))@weights)**2
    ideal/=max(ideal.max(),1e-30)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for size,axial in zip(sizes,center_samples):
        axes[0].plot(z_values*1000,np.asarray(axial)/peak,label=f'{size} px')
    axes[0].set(xlabel='z [mm, from 4f output plane]',ylabel='On-axis intensity / reference peak',title='Same pixel pitch; different retained aperture')
    axes[0].legend();axes[0].grid(alpha=.3)
    target=((z_values>=.15*design_length)&(z_values<=.65*design_length)).astype(float)
    axes[1].plot(z_values*1000,target,':',label='Requested bright interval')
    axes[1].plot(z_values*1000,ideal,label='Ideal infinite Bessel sum')
    axes[1].plot(z_values*1000,ref_axial/peak,label='Current tapered 1024 field')
    axes[1].set(xlabel='Design z [mm]',ylabel='Separately peak-normalized intensity',title='Finite aperture/taper changes the axial profile')
    axes[1].legend();axes[1].grid(alpha=.3)
    fig.savefig(out/'crop_comparison.png',dpi=150)
    plt.show()
    metrics['crop_tests']=crop_rows
    metrics['crop_propagation']='41 axial positions, 2x padded angular spectrum; output pixel pitch unchanged; relative to existing full field only; no SLM encoding in crop study'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    return metrics
