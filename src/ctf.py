import torch
import numpy as np


def compute_ctf(freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None):
    """
    Compute the 2D CTF
    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """

    assert freqs.shape[-1] == 2
    # units have been converted in ParticleDataset

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt ** 2) ** 0.5
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.atan2(y, x)
    s2 = x ** 2 + y ** 2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
            2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
            - phase_shift
    )
    ctf = (1 - w ** 2) ** 0.5 * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)

    return ctf


def convolve_ctf(proj, ctf):
    Fproj = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    Fproj = Fproj * ctf
    Fproj = torch.fft.ifftshift(Fproj, dim=(-2, -1))
    proj = torch.fft.ifft2(Fproj)
    proj = proj.real

    return proj

