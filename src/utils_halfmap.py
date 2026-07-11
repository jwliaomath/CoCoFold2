import numpy as np
import torch
import mrcfile


def load_mrc_volume_and_apix(path: str):
    """
    读取 MRC 体数据和像素大小 (Å/pixel)。
    """
    with mrcfile.open(path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)
        # m.voxel_size.x/y/z 通常是 Å
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
    计算 3D half-map FSC 曲线，并返回:
      - freq_centers: 每个 shell 的物理频率中心 (1/Å)
      - fsc: 对应 shell 的 FSC 值

    shell 的定义采用与 2D FRC 一致的“整数半径 ring/shell”思想，
    再映射到物理频率:
        freq = r / (N * apix)
    其中 N = min(nz, ny, nx)
    """
    if half1.shape != half2.shape:
        raise ValueError(f"half-map shape mismatch: {half1.shape} vs {half2.shape}")

    nz, ny, nx = half1.shape
    nmin = min(nz, ny, nx)
    max_shell = nmin // 2

    f1 = np.fft.fftn(half1)
    f2 = np.fft.fftn(half2)

    # 使用像素坐标半径做 shell，再映射到物理频率
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - nz // 2,
        np.arange(ny) - ny // 2,
        np.arange(nx) - nx // 2,
        indexing="ij",
    )
    rr_pix = np.sqrt(xx**2 + yy**2 + zz**2)

    # 为了和 shift 后的 Fourier 对齐
    f1 = np.fft.fftshift(f1)
    f2 = np.fft.fftshift(f2)

    frc_vals = []
    freq_centers = []

    df = 1.0 / (nmin * apix)  # 频率步长 (1/Å)

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
    返回:
      - shell_weight_freqs: shape [K], 每个 3D FSC shell 的频率中心 (1/Å)
      - shell_weights:      shape [K], 对应权重

    若未传 half-map，则返回 (None, None)。
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

    # 最简单、最稳的 ROCKET-inspired 权重
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
    按物理频率插值，而不是按长度插值。
    输入:
      src_freqs   : [K_src]  3D FSC 的频率中心
      src_weights : [K_src]  3D FSC shell 权重
      tgt_freqs   : [K_tgt]  2D FRC ring 频率中心

    返回:
      tgt_weights : [K_tgt]
    """
    if src_freqs is None or src_weights is None:
        return torch.ones_like(tgt_freqs, device=device, dtype=dtype)

    src_f = src_freqs.detach().cpu().numpy().astype(np.float32)
    src_w = src_weights.detach().cpu().numpy().astype(np.float32)
    tgt_f = tgt_freqs.detach().cpu().numpy().astype(np.float32)

    # 超出 src 最大频率的部分设为 0，更合理
    tgt_w = np.interp(
        tgt_f,
        src_f,
        src_w,
        left=float(src_w[0]),
        right=0.0,
    )

    return torch.tensor(tgt_w, dtype=dtype, device=device)