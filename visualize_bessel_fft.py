# 日本語説明：Besselビームとそのフーリエ変換を画像で確認する可視化用スクリプトです。圧縮データの作成には不要です。
import contextlib
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

root = Path(__file__).resolve().parent
nb = json.loads((root / 'test.ipynb').read_text(encoding='utf-8'))
ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    for i in (1, 2, 3):
        exec(''.join(nb['cells'][i]['source']), ns)
kr = 10.0
field = ns['make_bessel'](kr)
spectrum = np.fft.fftshift(np.fft.fft2(field, norm='ortho'))
intensity = np.abs(field)**2
power = np.abs(spectrum)**2
power /= power.max()
k = np.fft.fftshift(2*np.pi*np.fft.fftfreq(ns['N'], d=ns['dx']))
step = k[1]-k[0]
extent_k = [k[0]-step/2, k[-1]+step/2]*2
x = ns['x']; dx = ns['dx']
extent_x = [x[0]-dx/2, x[-1]+dx/2]*2
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
im = axes[0].imshow(intensity/intensity.max(), extent=extent_x, origin='lower', cmap='gray')
axes[0].set(title='Spatial intensity |E|^2', xlabel='x / W0', ylabel='y / W0')
fig.colorbar(im, ax=axes[0], label='Normalized intensity', shrink=.8)
for ax, title, norm in zip(axes[1:], ['FFT intensity |F|^2 (linear)', 'FFT intensity |F|^2 (log scale)'], [None, LogNorm(vmin=1e-6, vmax=1)]):
    im = ax.imshow(power, extent=extent_k, origin='lower', cmap='gray', norm=norm, interpolation='nearest')
    ax.set(xlim=(-25,25), ylim=(-25,25), title=title, xlabel='kx [1/W0]', ylabel='ky [1/W0]')
    fig.colorbar(im, ax=ax, label='Normalized spectral intensity', shrink=.8)
fig.suptitle('Bessel beam: l = 0, kr = 10 / W0, N = 1024\nFFT of the actual notebook field, including its super-Gaussian aperture', fontsize=13)
output = root / 'bessel_fft_kr10_gray.png'
fig.savefig(output, dpi=170)
plt.close(fig)
print(output)
