"""Unmodified projection functions from the pre-B6 two-state simulator."""
import numpy as np
from scipy.ndimage import map_coordinates

def centered_fft(a):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))

def ctf_grid(row):
    n, apix, dfu, dfv, angle, kv, cs, amp, phase = row
    axis = (np.arange(int(n)) - int(n)//2) / (n*apix)
    fx, fy = np.meshgrid(axis, axis, indexing='xy')
    s2 = fx*fx + fy*fy
    wavelength = 12.2639 / np.sqrt(kv*1000 + .97845e-6*(kv*1000)**2)
    df = .5*(dfu+dfv+(dfu-dfv)*np.cos(2*(np.arctan2(fy,fx)-np.deg2rad(angle))))
    gamma = 2*np.pi*(-.5*df*wavelength*s2 + .25*cs*1e7*wavelength**3*s2**2)-np.deg2rad(phase)
    return (np.sqrt(1-amp**2)*np.sin(gamma)-amp*np.cos(gamma)).astype(np.float32)

def project(fvolume, rotation):
    n = fvolume.shape[0]
    axis = np.arange(n)-n//2
    kx, ky = np.meshgrid(axis, axis, indexing='xy')
    # Object coordinates x_lab=R@x_object; reciprocal k_object=R.T@k_lab.
    points = np.stack((kx, ky, np.zeros_like(kx)), -1) @ rotation
    coords = np.moveaxis(points[..., ::-1]+n//2, -1, 0)  # array z,y,x
    # The discrete Fourier grid is periodic, including its Nyquist boundary.
    plane = map_coordinates(fvolume, coords, order=1, mode='grid-wrap', prefilter=False)
    plane[kx*kx+ky*ky >= (n/2)**2] = 0  # exclude ambiguous Nyquist edge
    inv = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(plane)))
    imag_ratio = np.linalg.norm(inv.imag)/max(np.linalg.norm(inv.real), 1e-30)
    if imag_ratio > 1e-5:
        raise ValueError(f'Projection is not Hermitian: imaginary/real={imag_ratio}')
    return inv.real.astype(np.float32)
