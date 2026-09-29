"""LSH主経路の物理設定。座標原点 z=0 は4f系の出力結像面。"""
import numpy as np

WAVELENGTH = 532e-9
SLM_PIXEL_PITCH = 3.74e-6
FOUR_F_MAGNIFICATION = 0.25  # 絶対値。結像の反転は座標規約へ吸収する。
PIXEL_PITCH = SLM_PIXEL_PITCH * FOUR_F_MAGNIFICATION
N = 2048
SLM_WIDTH = N * SLM_PIXEL_PITCH
OUTPUT_WIDTH = N * PIXEL_PITCH

# r0 は出力面の中心J0モードの第1零点。開口半径ではない。
# 100 µmは未検証の開始値であり、30～120 µmの掃引で選定する。
SPOT_RADIUS = 100e-6
SPOT_RADIUS_SWEEP_UM = (30, 40, 50, 60, 80, 100, 120)
DESIGN_LENGTH = 0.50  # z方向の設計区間。到達距離の保証ではない。
RANK = 256
BASIS_RANK = 256
APERTURE_EDGE_FRACTION = 0.10
THREAD_Z_START = 0.15 * DESIGN_LENGTH
THREAD_Z_END = 0.65 * DESIGN_LENGTH
PROPAGATION_PADDING = 2

# SLM面での1/e^2強度直径または1/e振幅waist。両方指定しない。
# 未確定の間は全面一様照明を仮定し、到達距離は理想上限と表示する。
LASER_DIAMETER = None
LASER_WAIST = None
# 以下は4f出力面へ換算した直径。未確定なら制限を課さない。
LENS_APERTURE_DIAMETER_OUTPUT = None
IRIS_APERTURE_DIAMETER_OUTPUT = None
ILLUMINATION_MODEL = 'square'  # 'square' または 'gaussian_fixed'

if LASER_DIAMETER is not None and LASER_WAIST is not None:
    raise ValueError('LASER_DIAMETERとLASER_WAISTはどちらか一方のみ指定してください。')
if ILLUMINATION_MODEL not in ('square', 'gaussian_fixed'):
    raise ValueError('ILLUMINATION_MODELはsquareまたはgaussian_fixed。')
if ILLUMINATION_MODEL == 'gaussian_fixed' and LASER_DIAMETER is None and LASER_WAIST is None:
    raise ValueError('Gaussian照明にはLASER_DIAMETERまたはLASER_WAISTが必要です。')

K = 2 * np.pi / WAVELENGTH
KR_CENTER = 2.404825557695773 / SPOT_RADIUS
Q = np.sqrt(K**2 - KR_CENTER**2)
M_MAX = int(np.floor(min(K - Q, Q) * DESIGN_LENGTH / (2 * np.pi)))
MODE_INDEX = np.arange(-M_MAX, M_MAX + 1)
KZ_LIST = Q + 2 * np.pi * MODE_INDEX / DESIGN_LENGTH
KR_LIST = np.sqrt((K - KZ_LIST) * (K + KZ_LIST))

assert np.all((KZ_LIST > 0) & (KZ_LIST <= K))
assert np.all(KR_LIST < np.pi / PIXEL_PITCH)
assert np.isclose(KR_LIST[M_MAX], KR_CENTER)
