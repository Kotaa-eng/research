# 主検証の唯一の物理設定。test.ipynbの設定セルから再読込します。
"""4f出力面のLSH設計条件。SLM面との横倍率は1/4。"""
import numpy as np

WAVELENGTH = 532e-9
SPOT_RADIUS = 30e-6  # 4f出力面における中心J0の第1零点半径
DESIGN_LENGTH = 0.50  # 4f出力面からの伝搬距離
SLM_PIXEL_PITCH = 8e-6
FOUR_F_MAGNIFICATION = 0.25
PIXEL_PITCH = SLM_PIXEL_PITCH * FOUR_F_MAGNIFICATION
N = 1024
RANK = 256
BASIS_RANK = 256  # 本検証とランク調査で同じ最大ランク基底を使用
APERTURE_EDGE_FRACTION = 0.10
# 現在は1本のThreadの予備検証。0.5 m全区間を明るくする設計ではない。
THREAD_Z_START = 0.15 * DESIGN_LENGTH
THREAD_Z_END = 0.65 * DESIGN_LENGTH
PROPAGATION_PADDING = 2  # RS直接積分との照合で不足なら増やす

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
