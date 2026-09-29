"""本検証・ランク調査・半径調査で共用する数値処理。import時は計算しない。"""
import hashlib
import json

import numpy as np
from scipy.special import j1
from sklearn.utils.extmath import randomized_svd
from tensorly import unfold


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

    def encode(self, field):
        amplitude = np.abs(field)/self.scale
        saturation = float(np.mean(amplitude > 1))
        modulation = np.interp(self.coupling*np.clip(amplitude, 0, 1), self.a, self.f)
        phase = np.mod(modulation*np.sin(np.angle(field)+self.carrier), 2*np.pi)
        codes = (np.rint(phase/(2*np.pi)*256).astype(np.uint16) % 256).astype(np.uint8)
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
             'THREAD_Z_START', 'THREAD_Z_END', 'PROPAGATION_PADDING')
    values = {name: namespace[name] for name in names}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode())
    for name in ('KR_LIST', 'KZ_LIST', 'WEIGHT_LIST', 'SHIFT_LIST', 'BEAM_INDICES'):
        digest.update(np.ascontiguousarray(namespace[name]).tobytes())
    return digest.hexdigest()
