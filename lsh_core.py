"""本検証・ランク調査・半径調査で共用する数値処理。import時は計算しない。"""
import hashlib
import json

import numpy as np
from scipy.special import j1
from sklearn.utils.extmath import randomized_svd
from tensorly import unfold


def illumination_aperture(n, output_pitch, magnification, edge_fraction=0.10,
                          model='square', laser_diameter=None, laser_waist=None):
    """固定開口/固定Gaussian照明。レーザー寸法はSLM面、戻り値は出力面。"""
    if laser_diameter is not None and laser_waist is not None:
        raise ValueError('レーザー直径とwaistは同時指定できません。')
    x = (np.arange(n)-(n-1)/2)*output_pitch
    if model == 'square':
        if not 0 <= edge_fraction < 1:
            raise ValueError('edge_fractionは[0,1)です。')
        half = (n-1)*output_pitch/2
        u = np.clip((np.abs(x)-half*(1-edge_fraction))/(half*edge_fraction),
                    0, 1) if edge_fraction else np.zeros_like(x)
        edge = 0.5*(1+np.cos(np.pi*u))
        return (edge[:, None]*edge[None, :]).astype(np.float32)
    if model == 'gaussian_fixed':
        waist_slm = laser_waist if laser_waist is not None else (
            laser_diameter/2 if laser_diameter is not None else None)
        if waist_slm is None or waist_slm <= 0:
            raise ValueError('Gaussian照明には正のレーザー直径またはwaistが必要です。')
        waist_output = magnification*waist_slm
        return np.exp(-(x[:, None]**2+x[None, :]**2)/waist_output**2).astype(np.float32)
    raise ValueError('modelはsquareまたはgaussian_fixed。')


def effective_output_radius(n, slm_pitch, magnification, *, laser_diameter=None,
                            laser_waist=None, lens_diameter_output=None,
                            iris_diameter_output=None, edge_fraction=0):
    """到達距離の幾何学的目安に使う半径。Gaussianのwaistは硬い開口ではない。"""
    if laser_diameter is not None and laser_waist is not None:
        raise ValueError('レーザー直径とwaistは同時指定できません。')
    if not 0 <= edge_fraction < 1:
        raise ValueError('edge_fractionは[0,1)です。')
    slm_radius = (n-1)*slm_pitch*magnification*(1-edge_fraction)/2
    candidates = {'SLM_ideal': slm_radius}
    for name, diameter in [('laser', laser_diameter), ('lens', lens_diameter_output),
                           ('iris', iris_diameter_output)]:
        if diameter is not None:
            if diameter <= 0:
                raise ValueError(f'{name}の直径は正である必要があります。')
            candidates[name] = diameter*(magnification if name == 'laser' else 1)/2
    if laser_waist is not None:
        if laser_waist <= 0:
            raise ValueError('laser_waistは正である必要があります。')
        candidates['laser_1e_amplitude'] = laser_waist*magnification
    unknown_limits = ((laser_diameter is None and laser_waist is None)
                      or lens_diameter_output is None or iris_diameter_output is None)
    return min(candidates.values()), candidates, unknown_limits


def geometric_reach(kr, wavelength, radius):
    """R/tan(theta) の幾何学的推定。伝搬強度を保証しない。"""
    k = 2*np.pi/wavelength
    kr = np.asarray(kr, dtype=float)
    if np.any(kr < 0) or np.any(kr >= k) or radius <= 0:
        raise ValueError('伝搬するkrと正の開口半径が必要です。')
    return np.divide(radius*np.sqrt(k*k-kr*kr), kr,
                     out=np.full_like(kr, np.inf), where=kr != 0)


def spatial_factors(uy, ux):
    """オフライン: FFT基底をIFFTして空間基底を作る。"""
    return (np.fft.ifft(uy, axis=0, norm='ortho'),
            np.fft.ifft(ux, axis=0, norm='ortho'))


def shift_operators(uy, ux, output_pitch, displacement_xy_m):
    """連続シフトのFourier位相を元の実直交基底へ再射影する。厳密な移動ではない。"""
    dy_m, dx_m = displacement_xy_m[1], displacement_xy_m[0]
    ky = 2*np.pi*np.fft.fftfreq(uy.shape[0], output_pitch)
    kx = 2*np.pi*np.fft.fftfreq(ux.shape[0], output_pitch)
    dy = np.exp(-1j*ky*dy_m)
    dx = np.exp(-1j*kx*dx_m)
    return uy.T @ (dy[:, None]*uy), ux.T @ (dx[:, None]*ux)


def sum_shifted_cores(cores, coefficients, uy, ux, output_pitch,
                      displacements_xy_m, operators=None):
    """各Threadで先にモード加算し、位置ごとに一度だけコアを再射影・加算。"""
    coefficients = np.asarray(coefficients)
    displacements_xy_m = np.asarray(displacements_xy_m, dtype=float)
    if coefficients.shape != (len(displacements_xy_m), len(cores)) or (
            displacements_xy_m.ndim != 2 or displacements_xy_m.shape[1] != 2):
        raise ValueError('係数は[Thread,mode]、位置は[Thread,x/y]です。')
    total = np.zeros(cores.shape[1:], dtype=np.complex128)
    for q, (weights, displacement) in enumerate(zip(coefficients, displacements_xy_m)):
        hq = np.tensordot(weights, cores, axes=(0, 0))
        ay, ax = (shift_operators(uy, ux, output_pitch, displacement)
                  if operators is None else operators[q])
        total += ay @ hq @ ax
    return total


def relative_error(reference, value):
    reference = np.asarray(reference, dtype=np.complex128)
    value = np.asarray(value, dtype=np.complex128)
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        return 0.0 if np.linalg.norm(value) == 0 else float('inf')
    return float(np.linalg.norm(value-reference)/denominator)


def phase_metrics(reference, value, amplitude_fraction=0.01):
    """マスクは理論振幅だけで固定。復元の失敗画素を隠さない。

    ゼロに復元された画素では位相は未定義なので、その割合も報告する。
    ±πのラップは複素積で評価し、通常の引き算は使わない。
    """
    reference = np.asarray(reference, dtype=np.complex128)
    value = np.asarray(value, dtype=np.complex128)
    amplitude = np.abs(reference)
    valid = amplitude > amplitude_fraction*amplitude.max()
    delta = np.angle(value*np.conj(reference))
    intensity = amplitude**2
    zero = np.abs(value) == 0
    defined = valid & ~zero
    return valid, delta, {
        'field_relative_L2': relative_error(reference, value),
        'amplitude_relative_L2': relative_error(amplitude, np.abs(value)),
        'intensity_relative_L2': relative_error(intensity, np.abs(value)**2),
        'phase_mae_rad': float(np.mean(np.abs(delta[defined]))) if defined.any() else None,
        'phase_rmse_rad': float(np.sqrt(np.mean(delta[defined]**2))) if defined.any() else None,
        'intensity_weighted_phase_mae_rad': (
            float(np.sum(intensity*np.abs(delta))/intensity.sum())
            if intensity.sum() and not np.any(zero & (intensity > 0)) else None),
        'phase_undefined_reference_energy_fraction': (
            float(intensity[zero].sum()/intensity.sum()) if intensity.sum() else None),
        'phase_mask_pixel_fraction': float(valid.mean()),
        'reference_bright_reconstruction_dark_fraction': (
            float(np.mean(np.abs(value)[valid] <= amplitude_fraction*amplitude.max()))
            if valid.any() else None),
    }


def fit_tucker(spectra, rank, fft_tensor=None):
    """同一アルゴリズムで共通因子とcomplex64コアを作る（空間2方向のみ圧縮）。"""
    if not 1 <= rank <= min(spectra.shape[-2:]):
        raise ValueError('空間ランクが範囲外です。')
    if fft_tensor is None:
        fft_tensor = np.stack((spectra.real, spectra.imag)).astype(np.float32)
    if fft_tensor.shape != (2, *spectra.shape):
        raise ValueError('FFTテンソルの軸が一致しません。')
    factors = []
    for mode in (2, 3):
        u, _, _ = randomized_svd(unfold(fft_tensor, mode), n_components=rank,
                                  n_iter=5, random_state=42)
        u, _ = np.linalg.qr(u.astype(np.float64), mode='reduced')
        factors.append(u)
    uy, ux = factors
    cores = np.stack([uy.T @ s @ ux for s in spectra]).astype(np.complex64)
    return uy.astype(np.float32), ux.astype(np.float32), cores


class PhaseOnlyCodec:
    """全評価に共通の振幅符号化、8bit量子化、一次回折光抽出。

    scaleは理論場から固定する。出力面の配列とSLM配列は理想4f結像で1対1対応。
    反転は座標方向の選択に吸収。収差、SLM位相LUT、照明、実レンズNAは未モデル化。
    """
    def __init__(self, shape, output_pitch, slm_pitch, kr_output, scale):
        if scale <= 0 or min(output_pitch, slm_pitch) <= 0:
            raise ValueError('振幅スケールとピッチは正の値が必要です。')
        self.scale = float(scale)
        self.coupling = 0.58
        self.f = np.linspace(0, 1.84118, 65536)
        self.a = j1(self.f)
        yy, xx = np.indices(shape)
        self.carrier = 2*np.pi*(xx+yy)/4
        fy = 2*np.pi*np.fft.fftfreq(shape[0], slm_pitch)
        fx = 2*np.pi*np.fft.fftfreq(shape[1], slm_pitch)
        self.fc = 2*np.pi/(4*slm_pitch)
        self.radius = 1.1*float(np.max(kr_output))*output_pitch/slm_pitch
        if self.radius >= self.fc:
            raise ValueError('CGHフィルタがサンプリング範囲を超えます。')
        self.pupil = ((fy[:, None]-self.fc)**2+(fx[None, :]-self.fc)**2
                      <= self.radius**2)

    def encode_phase(self, field):
        amplitude = np.abs(field)/self.scale
        saturation = float(np.mean(amplitude > 1))
        modulation = np.interp(self.coupling*np.clip(amplitude, 0, 1), self.a, self.f)
        phase = np.mod(modulation*np.sin(np.angle(field)+self.carrier), 2*np.pi)
        return phase, saturation

    @staticmethod
    def quantize(phase):
        """連続位相[rad]を8-bitへ。符号化時間と量子化時間を分離可能。"""
        codes = (np.rint(phase/(2*np.pi)*256).astype(np.uint16) % 256).astype(np.uint8)
        return codes

    def encode(self, field):
        phase, saturation = self.encode_phase(field)
        codes = self.quantize(phase)
        return phase, codes, saturation

    def replay(self, codes):
        transmittance = np.exp(1j*codes.astype(np.float64)*(2*np.pi/256))
        filtered = np.fft.ifft2(np.fft.fft2(transmittance, norm='ortho')*self.pupil,
                               norm='ortho')
        return filtered*np.exp(-1j*self.carrier)*self.scale/self.coupling


def config_signature(namespace):
    """カーネル内の設定と下流の保存結果が混ざることを防ぐ署名。"""
    names = ('WAVELENGTH', 'SPOT_RADIUS', 'DESIGN_LENGTH', 'SLM_PIXEL_PITCH',
             'FOUR_F_MAGNIFICATION', 'PIXEL_PITCH', 'dx', 'N', 'RANK', 'BASIS_RANK', 'APERTURE_EDGE_FRACTION',
             'THREAD_Z_START', 'THREAD_Z_END', 'PROPAGATION_PADDING',
             'LASER_DIAMETER', 'LASER_WAIST', 'LENS_APERTURE_DIAMETER_OUTPUT',
             'IRIS_APERTURE_DIAMETER_OUTPUT', 'ILLUMINATION_MODEL')
    values = {name: namespace[name] for name in names}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode())
    for name in ('KR_LIST', 'KZ_LIST', 'WEIGHT_LIST', 'SHIFT_LIST', 'BEAM_INDICES'):
        digest.update(np.ascontiguousarray(namespace[name]).tobytes())
    return digest.hexdigest()
