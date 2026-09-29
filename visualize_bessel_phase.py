# 日本語説明：Besselビームの位相を確認する可視化用スクリプトです。振幅符号化したSLM用CGHとは区別してください。
import runpy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
a = runpy.run_path(str(root / 'visualize_bessel_fft.py'))
field, spectrum, k = a['field'], a['spectrum'], a['k']
# np.fft assumes sample index zero is the origin. Correct phase to the actual
# notebook coordinates x_j=x_min+j*dx (even N, endpoint-inclusive grid).
x_min = a['x'][0]
spectrum_physical = spectrum * np.exp(-1j * (k[:, None] + k[None, :]) * x_min)
assert np.linalg.norm(spectrum_physical.imag) / np.linalg.norm(spectrum_physical) < 1e-5
spectrum_physical = spectrum_physical.real  # Even real field: remove numerical imaginary roundoff.
threshold = 1e-6  # Mask where normalized intensity is below this threshold.

def phase_masked(z):
    power = np.abs(z)**2
    phase = np.mod(np.angle(z), 2*np.pi)
    phase[np.isclose(phase, 2*np.pi, atol=1e-6, rtol=0)] = 0.0
    phase[np.isclose(phase, 0, atol=1e-6, rtol=0)] = 0.0
    return np.ma.array(phase, mask=power < threshold * power.max())

cmap = plt.get_cmap('gray').copy()
cmap.set_bad('0.5')
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for ax, z, ext, title, xlabel, ylabel in [
    (axes[0], field, a['extent_x'], 'Spatial phase: arg(E)', 'x / W0', 'y / W0'),
    (axes[1], spectrum_physical, a['extent_k'], 'Fourier phase: arg(F)\nPhysical coordinate origin corrected', 'kx [1/W0]', 'ky [1/W0]'),
]:
    im = ax.imshow(phase_masked(z), origin='lower', extent=ext, cmap=cmap,
                   vmin=0, vmax=np.pi, interpolation='nearest')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    cb = fig.colorbar(im, ax=ax, shrink=.8, ticks=[0, np.pi/2, np.pi])
    cb.ax.set_yticklabels(['0', 'pi/2', 'pi'])
    cb.set_label('Phase [rad]')
axes[1].set(xlim=(-25,25), ylim=(-25,25))
fig.suptitle('l=0, kr=10/W0; reference plane, no propagation\nMid-gray background: masked low-intensity pixels (< 1e-6 of peak)', fontsize=12)
fig.savefig(root / 'bessel_phase_kr10_gray.png', dpi=170)
plt.close(fig)
# Standalone Fourier intensity images, cropped to the displayed frequency range.
select = np.flatnonzero(np.abs(k) <= 25)
power = a['power'][np.ix_(select, select)]
plt.imsave(root / 'bessel_fourier_intensity_gray.png', power, cmap='gray', vmin=0, vmax=1, origin='lower')
plt.imsave(root / 'bessel_spatial_phase_gray.png', phase_masked(field), cmap=cmap, vmin=0, vmax=np.pi, origin='lower')
print('Saved grayscale intensity and phase figures.')
