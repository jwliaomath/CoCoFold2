import numpy as np
import torch
import mrcfile


def load_mrc_volume_and_apix(path: str):
    """
    Read an MRC volume and its pixel size (Angstrom/pixel).
    """
    with mrcfile.open(path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)
        # m.voxel_size.x/y/z are normally expressed in Angstrom.
        apix_x = float(m.voxel_size.x) if m.voxel_size.x > 0 else None
        apix_y = float(m.voxel_size.y) if m.voxel_size.y > 0 else None
        apix_z = float(m.voxel_size.z) if m.voxel_size.z > 0 else None

    if apix_x is None or apix_y is None or apix_z is None:
        raise ValueError(f"Cannot read valid voxel size from MRC header: {path}")

    if not (np.isclose(apix_x, apix_y) and np.isclose(apix_x, apix_z)):
        print(
            f"[WARN] voxel size anisotropic in {path}: "
            f"x={apix_x}, y={apix_y}, z={apix_z}. Use x as default."
        )

    return vol, apix_x


def smooth_1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1:
        return x.astype(np.float32)
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y.astype(np.float32)


def compute_halfmap_fsc_curve(
    half1: np.ndarray,
    half2: np.ndarray,
    apix: float,
    eps: float = 1e-8,
):
    """
    Compute the 3D half-map FSC curve and return:
      - freq_centers: physical frequency center of each shell (1/Angstrom)
      - fsc: FSC value for each shell

    Shells use integer-radius bins, following the 2D FRC ring convention,
    then map radii to physical frequencies:
        freq = r / (N * apix)
    where N = min(nz, ny, nx).
    """
    if half1.shape != half2.shape:
        raise ValueError(f"half-map shape mismatch: {half1.shape} vs {half2.shape}")

    nz, ny, nx = half1.shape
    nmin = min(nz, ny, nx)
    max_shell = nmin // 2

    f1 = np.fft.fftn(half1)
    f2 = np.fft.fftn(half2)

    # Build shells from pixel-coordinate radii, then map to physical frequencies.
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - nz // 2,
        np.arange(ny) - ny // 2,
        np.arange(nx) - nx // 2,
        indexing="ij",
    )
    rr_pix = np.sqrt(xx**2 + yy**2 + zz**2)

    # Align Fourier coefficients with the centered shell grid.
    f1 = np.fft.fftshift(f1)
    f2 = np.fft.fftshift(f2)

    frc_vals = []
    freq_centers = []

    df = 1.0 / (nmin * apix)  # frequency spacing (1/Angstrom)

    for r in range(1, max_shell + 1):
        mask = (rr_pix >= (r - 0.5)) & (rr_pix < (r + 0.5))
        if not np.any(mask):
            frc_vals.append(0.0)
            freq_centers.append(r * df)
            continue

        a = f1[mask]
        b = f2[mask]

        num = np.real(np.sum(a * np.conj(b)))
        den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2) + eps)
        fsc_r = num / (den + eps)

        frc_vals.append(float(np.clip(fsc_r, -1.0, 1.0)))
        freq_centers.append(r * df)

    return np.asarray(freq_centers, dtype=np.float32), np.asarray(frc_vals, dtype=np.float32)


def build_halfmap_shell_weights(
    halfmap1_path: str = None,
    halfmap2_path: str = None,
    gamma: float = 1.0,
    smooth_win: int = 0,
    clamp_min_zero: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Returns:
      - shell_weight_freqs: shape [K], frequency center of each 3D FSC shell (1/Angstrom)
      - shell_weights:      shape [K], corresponding weights

    Return (None, None) if either half-map path is absent.
    """
    if halfmap1_path is None or halfmap2_path is None:
        return None, None

    half1, apix1 = load_mrc_volume_and_apix(halfmap1_path)
    half2, apix2 = load_mrc_volume_and_apix(halfmap2_path)

    if not np.isclose(apix1, apix2):
        raise ValueError(f"half-map voxel size mismatch: {apix1} vs {apix2}")

    freq_centers, fsc = compute_halfmap_fsc_curve(half1, half2, apix=apix1)

    if clamp_min_zero:
        fsc = np.clip(fsc, 0.0, None)

    # ROCKET-inspired weights from the FSC curve.
    weights = fsc ** gamma

    if smooth_win and smooth_win > 1:
        weights = smooth_1d(weights, win=smooth_win)

    shell_weight_freqs = torch.tensor(freq_centers, dtype=dtype, device=device)
    shell_weights = torch.tensor(weights, dtype=dtype, device=device)

    return shell_weight_freqs, shell_weights


def interpolate_weights_by_frequency(
    src_freqs: torch.Tensor,
    src_weights: torch.Tensor,
    tgt_freqs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Interpolate by physical frequency rather than array length.
    Inputs:
      src_freqs   : [K_src]  3D FSC frequency centers
      src_weights : [K_src]  3D FSC shell weights
      tgt_freqs   : [K_tgt]  2D FRC ring frequency centers

    Returns:
      tgt_weights : [K_tgt]
    """
    if src_freqs is None or src_weights is None:
        return torch.ones_like(tgt_freqs, device=device, dtype=dtype)

    src_f = src_freqs.detach().cpu().numpy().astype(np.float32)
    src_w = src_weights.detach().cpu().numpy().astype(np.float32)
    tgt_f = tgt_freqs.detach().cpu().numpy().astype(np.float32)

    # Set weights beyond the maximum source frequency to zero.
    tgt_w = np.interp(
        tgt_f,
        src_f,
        src_w,
        left=float(src_w[0]),
        right=0.0,
    )

    return torch.tensor(tgt_w, dtype=dtype, device=device)