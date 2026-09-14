import numpy as np
import torch
import os
import torch.nn.functional as F

os.environ['BMP_DUPLICATE_LIB_OB'] = 'TRUE'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def sum_of_gaussians_2d_torch(centers, coef, sdev, maxrange, matrices, batch_size=1000,k=5):
    device = centers.device
    maxrange = torch.tensor(maxrange, device=device).to(coef.dtype)
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    H, W = matrices.shape[1:]

    # Prepare the range for grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=centers.dtype, device=device),
        torch.arange(W, dtype=centers.dtype, device=device),
        indexing='ij'
    )
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(coef.dtype)  # Shape [1, 1, H, W]
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(coef.dtype)  # Shape [1, 1, H, W]

    centers = centers.unsqueeze(-1).unsqueeze(-1)  # Shape [B, N, 2, 1, 1]
    sdev = sdev.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape [1, N, 2, 1, 1]

    # Initialize density
    density = torch.zeros_like(matrices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        centers_batch = centers[:, start:end, :, :, :]
        sdev_batch = sdev[:, start:end, :, :, :]
        coef_batch = coef[start:end].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        coef_batch = coef_batch.expand(B, end - start, 1, 1)

        # Calculate the bounds dynamically based on centers and sdev
        x_min = torch.clamp(
            (centers_batch[..., 0, :, :] - k * sdev_batch[..., 0, :, :]).min().floor().int(), min=0
        )
        x_max = torch.clamp(
            (centers_batch[..., 0, :, :] + k * sdev_batch[..., 0, :, :]).max().ceil().int(),
            max=grid_x.shape[-1],
        )
        y_min = torch.clamp(
            (centers_batch[..., 1, :, :] - k * sdev_batch[..., 1, :, :]).min().floor().int(), min=0
        )
        y_max = torch.clamp(
            (centers_batch[..., 1, :, :] + k * sdev_batch[..., 1, :, :]).max().ceil().int(),
            max=grid_y.shape[-2],
        )

        # Subset grid for the relevant region
        grid_x_sub = grid_x[:, :, y_min:y_max, x_min:x_max]
        grid_y_sub = grid_y[:, :, y_min:y_max, x_min:x_max]

        # Compute the distance and Gaussian function within the subset grid
        dy = (grid_y_sub - centers_batch[..., 1, :, :]) / (sdev_batch[..., 1, :, :] + 1e-8)
        dx = (grid_x_sub - centers_batch[..., 0, :, :]) / (sdev_batch[..., 0, :, :] + 1e-8)
        d2 = dy**2 + dx**2

        # Mask and calculate Gaussians
        gaussians = coef_batch * torch.exp(-0.5 * d2)

        # Accumulate density into the main matrices
        density[:, y_min:y_max, x_min:x_max] += torch.sum(gaussians, dim=1)

    matrices += density

    return matrices

def _render_peak2d_checkpoint_block(centers_batch, coef_batch, sdev_batch,
                                   grid_x_sub, grid_y_sub):
    """Original pixel arithmetic, with all replay inputs passed explicitly."""
    dy = (grid_y_sub - centers_batch[..., 1, :, :]) / (sdev_batch[..., 1, :, :] + 1e-8)
    dx = (grid_x_sub - centers_batch[..., 0, :, :]) / (sdev_batch[..., 0, :, :] + 1e-8)
    d2 = dy**2 + dx**2
    gaussians = coef_batch * torch.exp(-0.5 * d2)
    return torch.sum(gaussians, dim=1)


def sum_of_gaussians_2d_torch_checkpointed(centers, coef, sdev, maxrange,
                                          matrices, batch_size=1000, k=5):
    """Checkpoint the original peak-2D blocks without changing their support.

    Keep the historical chunk size, rectangular crop, epsilon and accumulation
    order. No normalization, input mutation or random operation is replayed.
    maxrange remains unused, as in the original renderer; k sets the crop.
    """
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (centers, coef, sdev)):
        return sum_of_gaussians_2d_torch(centers, coef, sdev, maxrange,
                                         matrices, batch_size=batch_size, k=k)
    from torch.utils.checkpoint import checkpoint
    device = centers.device
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    H, W = matrices.shape[1:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=centers.dtype, device=device),
        torch.arange(W, dtype=centers.dtype, device=device), indexing='ij')
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(coef.dtype)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(coef.dtype)
    centers = centers.unsqueeze(-1).unsqueeze(-1)
    sdev = sdev.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    density = torch.zeros_like(matrices)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        centers_batch = centers[:, start:end, :, :, :]
        sdev_batch = sdev[:, start:end, :, :, :]
        coef_batch = coef[start:end].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        coef_batch = coef_batch.expand(B, end - start, 1, 1)
        x_min = torch.clamp(
            (centers_batch[..., 0, :, :] - k * sdev_batch[..., 0, :, :]).min().floor().int(), min=0)
        x_max = torch.clamp(
            (centers_batch[..., 0, :, :] + k * sdev_batch[..., 0, :, :]).max().ceil().int(), max=grid_x.shape[-1])
        y_min = torch.clamp(
            (centers_batch[..., 1, :, :] - k * sdev_batch[..., 1, :, :]).min().floor().int(), min=0)
        y_max = torch.clamp(
            (centers_batch[..., 1, :, :] + k * sdev_batch[..., 1, :, :]).max().ceil().int(), max=grid_y.shape[-2])
        grid_x_sub = grid_x[:, :, y_min:y_max, x_min:x_max]
        grid_y_sub = grid_y[:, :, y_min:y_max, x_min:x_max]
        block = checkpoint(_render_peak2d_checkpoint_block,
                           centers_batch, coef_batch, sdev_batch, grid_x_sub, grid_y_sub,
                           use_reentrant=False, preserve_rng_state=False)
        density[:, y_min:y_max, x_min:x_max] += block
    matrices += density
    return matrices


def pdb2img_peak2d_checkpointed(atoms_coord, resolution, atoms_weight, rotation,
                                trans, density_center, box_size=256, cutoff_range=5,
                                sigma_factor=1/(np.pi*np.sqrt(2)), apix=1, sdevs=None):
    """Training path with original pdb2img arithmetic and checkpointed pixels.

    Deliberately keep this path separate from the covariance renderer: min vs
    amin tie gradients, coordinate operation order, and scaling stay unchanged.
    The original pdb2img (including its optional masks) remains untouched.
    """
    B, _, _ = rotation.shape
    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor
    atoms_coord = atoms_coord / apix
    proj_rot = centers_rotation(atoms_coord, rotation)
    origin = torch.min(proj_rot, dim=1, keepdim=True).values
    proj_rot[..., 0] = proj_rot[..., 0] / step - origin[..., 0] / step
    proj_rot[..., 1] = proj_rot[..., 1] / step - origin[..., 1] / step
    proj_rot += pad
    img = sum_of_gaussians_2d_torch_checkpointed(
        centers=proj_rot.to(proj_rot.device), coef=atoms_weight.to(proj_rot.device),
        sdev=sdevs.to(proj_rot.device), maxrange=cutoff_range,
        matrices=torch.zeros(B, box_size, box_size).to(proj_rot.device).to(proj_rot.dtype))
    normalization = torch.pow(2 * torch.pi, torch.tensor(-1)) * torch.pow(sdev, torch.tensor(-2))
    img *= normalization
    img /= step
    img = img.unsqueeze(1)
    return translation_2d(img, trans / step, box_size, apix, density_center.to(img.device))


def centers_rotation(coords, rotations):
    """
    Output:
        rotated_coords[..., :2]: BxNx2
    """

    coords = coords.transpose(1, 2)
    rotated_coords = torch.matmul(rotations, coords)
    rotated_coords = rotated_coords.transpose(1, 2)
    return rotated_coords[..., :2]


def translation_2d(proj, trans, box_size, apix,density_center):
    """
    Inputs:
        proj: Bx1xbsxbs tensor
        trans: Bx2 tensor
    Output:
        proj_trans: Bx1xbsxbs tensor
    """
    B, _, H, W = proj.shape
    b = trans.shape[0]

    y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W))
    y_indices = y_indices.flatten().to(trans.dtype)
    x_indices = x_indices.flatten().to(trans.dtype)

    y_indices = y_indices.to(proj.device)
    x_indices = x_indices.to(proj.device)
    flat_images = proj.view(B, -1)
    # Compute the total intensity for each image
    total_intensity = flat_images.sum(dim=1)
    #if (total_intensity == 0).any():

    flat_images /= total_intensity.view(B,1)

    # Compute the weighted sum of coordinates
    x_weighted_sum = (flat_images * x_indices).sum(dim=1)
    y_weighted_sum = (flat_images * y_indices).sum(dim=1)

    # Compute the centroids
    centroid_x = x_weighted_sum
    centroid_y = y_weighted_sum

    centroids = torch.stack([centroid_x, centroid_y], dim=1)

    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device).to(proj.dtype)
    # trans *= apix

    trans -= (density_center - centroids)

    # trans -= (box_size / 2)

    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)
    grid = F.affine_grid(theta, size=proj.shape)

    proj_trans = F.grid_sample(proj, grid, mode='bicubic')


    return proj_trans


def translation_2d_robust(proj, trans, box_size, apix, density_center):
    """
    Robust 2D translation for cryo-EM density projections with negative backgrounds.
    Inputs:
        proj: Bx1xbsxbs tensor (negative values are allowed)
        trans: Bx2 tensor
        box_size: int
        apix: float
        density_center: Bx2 tensor
    Output:
        proj_trans: Bx1xbsxbs tensor
    """
    B, _, H, W = proj.shape
    b = trans.shape[0]

    # Specify indexing='ij' to avoid implicit meshgrid indexing.
    y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y_indices = y_indices.flatten().to(trans.dtype).to(proj.device)
    x_indices = x_indices.flatten().to(trans.dtype).to(proj.device)

    flat_images = proj.view(B, -1)

    # ==========================================
    # Use ReLU to exclude negative density from the centroid calculation.
    # ==========================================
    weights = torch.relu(flat_images)
    total_intensity = weights.sum(dim=1) + 1e-8 # Add epsilon to prevent division by zero.
    weights_normalized = weights / total_intensity.view(B, 1)

    # Compute the centroid using positive density only.
    centroid_x = (weights_normalized * x_indices).sum(dim=1)
    centroid_y = (weights_normalized * y_indices).sum(dim=1)
    centroids = torch.stack([centroid_x, centroid_y], dim=1)

    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device).to(proj.dtype)

    # Align to density_center using the existing convention.
    trans -= (density_center - centroids)
    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)

    # Explicitly use align_corners=False for both grid operations.
    grid = F.affine_grid(theta, size=proj.shape, align_corners=False)
    proj_trans = F.grid_sample(proj, grid, mode='bicubic', align_corners=False)

    return proj_trans

def pdb2img(atoms_coord,
            resolution,
            atoms_weight,
            rotation,
            trans,
            density_center,
            box_size=256,
            cutoff_range=5,  # in standard deviations
            sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution)
            apix=1,
            sdevs = None,
            affine_vector= None,
            masks = None,
            affine_matricies = None,
            cut_number = 3735,

            ):
    """
    Projection of 3D GMM without molmap
    Inputs:
        atoms_coord: BxNx3 tensor
        resolution: float
        atoms_wight: Nx1 tensor
        rotation: Bx2x3 tensor, only first two rows are needed since z will be integrated
        trans: Bx2 tensor
        box_size: int
        cutoff_range: int
        sigma_factor: float
    Output:
        img: Bx1xbsxbs
    """

    # get parameters for GMM
    _, N, _ = atoms_coord.shape
    B,_,_ = rotation.shape
    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor

    if masks is not None:
        for i in range(len(masks)):
            affine_matrix = torch.from_numpy(affine_matricies[i]).to(atoms_coord.device).to(atoms_coord.dtype)
            mask_indices = masks[i]
            atoms_coord[:,mask_indices,:] = torch.matmul(atoms_coord[:,mask_indices ,:], affine_matrix[:,:3].T) + affine_matrix[:,3]

    atoms_coord = atoms_coord/apix
    # rotation
    proj_rot = centers_rotation(atoms_coord, rotation)
    # transform xy to the grid ij and make it into the box
    origin = torch.min(proj_rot, dim=1, keepdim=True).values
    proj_rot[..., 0] = proj_rot[..., 0] / step - origin[..., 0] / step
    proj_rot[..., 1] = proj_rot[..., 1] / step - origin[..., 1] / step
    proj_rot += pad


    # projection
    img = sum_of_gaussians_2d_torch(centers=proj_rot.to(proj_rot.device), coef=atoms_weight.to(proj_rot.device), sdev=sdevs.to(proj_rot.device), maxrange=cutoff_range,matrices=torch.zeros(B, box_size, box_size).to(proj_rot.device).to(proj_rot.dtype))

    normalization = torch.pow(2 * torch.pi, torch.tensor(-1)) * torch.pow(sdev, torch.tensor(-2))
    img *= normalization

    # scale to integrate z
    img /= step
    img = img.unsqueeze(1)
    # move it to the center first and then modify the translation
    img = translation_2d(img, trans / step, box_size, apix,density_center.to(img.device))


    return img

def sum_of_gaussians_3d_torch(centers, coef, sdev, maxrange, matrices, batch_size=1):
    device = centers.device
    maxrange = torch.tensor(maxrange, device=device)
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    D, H, W = matrices.shape[1:]
    density = torch.zeros_like(matrices)
    for c in range(N):

        sd = sdev[c]
        cf = coef[c]
        center = centers[0,c,...]

        ijk_min = torch.ceil(center - maxrange * sd).to(torch.int)
        ijk_max = torch.floor(center + maxrange * sd).to(torch.int)
        ijk_min[ijk_min<=0] = 0
        ijk_min[ijk_max>=D] = D-1
        ijk_max[ijk_max<=0] = 0
        ijk_max[ijk_max>=D] = D-1
        z = torch.arange(ijk_min[2], ijk_max[2] + 1).to(device)
        y = torch.arange(ijk_min[1], ijk_max[1] + 1).to(device)
        x = torch.arange(ijk_min[0], ijk_max[0] + 1).to(device)
        Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

        dz = (Z - center[2]) / sd[2]
        dy = (Y - center[1]) / sd[1]
        dx = (X - center[0]) / sd[0]

        d2 = dz ** 2 + dy ** 2 + dx ** 2
        gauss = cf * torch.exp(-0.5 * d2)

        density[0, ijk_min[2]:ijk_max[2] + 1, ijk_min[1]:ijk_max[1] + 1, ijk_min[0]:ijk_max[0] + 1] += gauss
    matrices+=density

    return matrices


def translation_center(proj, box_size):
    """
    Inputs:
        proj: Bx1xDxHxW tensor
        box_size: int (size of the box)
    Output:
        proj_trans: Bx1xDxHxW tensor
    """
    B, _, D, H, W = proj.shape

    z_indices, y_indices, x_indices = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
    z_indices = z_indices.flatten().float()
    y_indices = y_indices.flatten().float()
    x_indices = x_indices.flatten().float()

    z_indices = z_indices.to(proj.device)
    y_indices = y_indices.to(proj.device)
    x_indices = x_indices.to(proj.device)
    flat_images = proj.view(B, -1)
    # Compute the total intensity for each image
    total_intensity = flat_images.sum(dim=1)
    if (total_intensity == 0).any():
        print("Warning: total_intensity contains zero values")

    # Compute the weighted sum of coordinates
    x_weighted_sum = (flat_images * x_indices).sum(dim=1)
    y_weighted_sum = (flat_images * y_indices).sum(dim=1)
    z_weighted_sum = (flat_images * z_indices).sum(dim=1)

    # Compute the centroids
    centroid_x = x_weighted_sum / (total_intensity + 1e-10)
    centroid_y = y_weighted_sum / (total_intensity + 1e-10)
    centroid_z = z_weighted_sum / (total_intensity + 1e-10)

    centroids = torch.stack([centroid_x, centroid_y, centroid_z], dim=1)

    eye = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(proj.device)

    trans = -(box_size / 2 - centroids)
    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)

    # Create the affine grid
    grid = F.affine_grid(theta, size=proj.shape, align_corners=True)
    # Sample the original image with the grid to get the translated image
    proj_trans = F.grid_sample(proj, grid, mode='nearest', align_corners=True)

    return proj_trans

def pdb2mrc(atoms_coord,
            resolution,
            atoms_weight,
            rotation=None,
            box_size=256,
            sdevs = None,
            affine_matrix1 = None,
            cutoff_range=5,  # in standard deviations
            sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution)
            apix=1,
            ):
    """
    Projection of 3D GMM without molmap
    Inputs:
        atoms_coord: BxNx3 tensor
        resolution: float
        atoms_wight: Nx1 tensor
        rotation: Bx2x3 tensor, only first two rows are needed since z will be integrated
        trans: Bx2 tensor
        box_size: int
        cutoff_range: int
        sigma_factor: float
    Output:
        img: Bx1xbsxbs
    """

    # get parameters for GMM
    B, N, _ = atoms_coord.shape

    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor
    if sdevs is None:
        sdevs = torch.zeros(N, 3)
        sdevs += sdev / step
    else:
        temp = torch.zeros(N,3).to(sdevs.device)
        temp += sdev/step
        temp[:,:2] = sdevs
        temp[:,2] = torch.mean(sdevs,dim=1)
        sdevs = temp
    # rotation
    if rotation is not None:
        atoms_coord = atoms_coord.transpose(1, 2)
        rotated_coords = torch.matmul(rotation, atoms_coord)
        rotated_coords = rotated_coords.transpose(1, 2)
    else:
        rotated_coords = atoms_coord
    rotated_coords/=apix
    # transform xy to the grid ij and make it into the box
    origin = torch.min(rotated_coords, dim=1, keepdim=True).values
    rotated_coords[..., 0] = rotated_coords[..., 0] - origin[..., 0]
    rotated_coords[..., 1] = rotated_coords[..., 1] - origin[..., 1]
    rotated_coords[..., 2] = rotated_coords[..., 2] - origin[..., 2]
    rotated_coords += pad

    grid = sum_of_gaussians_3d_torch(centers=rotated_coords, coef=atoms_weight, sdev=sdevs, maxrange=cutoff_range,
                                    matrices=torch.zeros(B, box_size, box_size, box_size).to(rotated_coords.device))

    normalization = torch.pow(2 * torch.pi, torch.tensor(-1.5)) * torch.pow(sdev, torch.tensor(-3))
    grid *= normalization

    # move it to the center first and then modify the translation
    grid = translation_center(grid.unsqueeze(1), box_size)

    return grid

# ==================== 3D Fourier-slice rendering ====================

def pdb2mrc_v2(atoms_coord, resolution, atoms_weight, rotation=None, box_size=256, sdevs=None, cutoff_range=5, sigma_factor=1 / (np.pi * np.sqrt(2)), apix=1):
    """
    Variant of pdb2mrc supporting an N x 3 sdevs tensor.
    """
    B, N, _ = atoms_coord.shape
    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor

    # Use the supplied [N, 3] sdevs directly, without padding 2D widths.
    if sdevs is None:
        sdevs = torch.zeros(N, 3, device=atoms_coord.device)
        sdevs += sdev / step

    if rotation is not None:
        atoms_coord = atoms_coord.transpose(1, 2)
        rotated_coords = torch.matmul(rotation, atoms_coord)
        rotated_coords = rotated_coords.transpose(1, 2)
    else:
        rotated_coords = atoms_coord
    rotated_coords /= apix

    origin = torch.min(rotated_coords, dim=1, keepdim=True).values
    rotated_coords[..., 0] = rotated_coords[..., 0] - origin[..., 0]
    rotated_coords[..., 1] = rotated_coords[..., 1] - origin[..., 1]
    rotated_coords[..., 2] = rotated_coords[..., 2] - origin[..., 2]
    rotated_coords += pad

    grid = sum_of_gaussians_3d_torch(centers=rotated_coords, coef=atoms_weight, sdev=sdevs, maxrange=cutoff_range,
                                     matrices=torch.zeros(B, box_size, box_size, box_size, device=rotated_coords.device))

    normalization = torch.pow(2 * torch.pi, torch.tensor(-1.5)) * torch.pow(sdev, torch.tensor(-3))
    grid *= normalization
    grid = translation_center(grid.unsqueeze(1), box_size)

    return grid, step

def prepare_fourier_volume(vol_3d):
    """Transform the generated 3D MRC volume into Fourier space."""
    # Fourier transform sequence: ifftshift -> fftn -> fftshift
    V_shift = torch.fft.ifftshift(vol_3d, dim=(-3, -2, -1))
    F_vol = torch.fft.fftn(V_shift, dim=(-3, -2, -1))
    F_vol = torch.fft.fftshift(F_vol, dim=(-3, -2, -1))
    return F_vol.real, F_vol.imag

def slice_fourier_volume(F_vol_real, F_vol_imag, rotations, trans, density_center, box_size, apix, step):
    B = rotations.shape[0]
    device = F_vol_real.device

    # 1. Construct the 2D base grid in the Z=0 Fourier plane.
    lin = torch.linspace(-1, 1, box_size, device=device)
    Y, X = torch.meshgrid(lin, lin, indexing='ij')
    Z = torch.zeros_like(X)
    grid_2d = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    # 2. Rotate the sampling grid inversely to the object rotation R.
    # For an object rotation R, the slice grid uses the inverse rotation R^T.
    # grid_2d contains (K, 3) row vectors; rotations has shape (B, 3, 3).
    # For row vectors, grid_2d @ rotations equals (R^T @ grid^T)^T.
    rotated_grid = torch.matmul(grid_2d, rotations)
    rotated_grid = rotated_grid.view(B, 1, box_size, box_size, 3)

    F_vol_real_exp = F_vol_real.expand(B, -1, -1, -1, -1)
    F_vol_imag_exp = F_vol_imag.expand(B, -1, -1, -1, -1)

    # 3. Sample slices with grid_sample interpolation.
    slice_real = F.grid_sample(F_vol_real_exp, rotated_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    slice_imag = F.grid_sample(F_vol_imag_exp, rotated_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    slice_real = slice_real.squeeze(2)
    slice_imag = slice_imag.squeeze(2)
    F_slice = torch.complex(slice_real, slice_imag)

    # 4. Transform 2D Fourier slices back into real-space images.
    F_slice = torch.fft.ifftshift(F_slice, dim=(-2, -1))
    proj = torch.fft.ifft2(F_slice, dim=(-2, -1)).real
    proj = torch.fft.fftshift(proj, dim=(-2, -1))

    # Correct the integration scale.
    proj = proj * box_size
    proj /= step

    # 5. Apply the final 2D translation.
    proj = translation_2d(proj, trans, box_size, apix, density_center)
    return proj


def project_gaussian_covariances(covariances, rotations):
    """Project shared reference-frame covariance [N,3,3] into each camera.

    Covariance is in internal rendering-grid units squared. Rotations can be
    [B,3,3] or their first two rows [B,2,3]. No additional apix scaling here.
    """
    if covariances.ndim != 3 or covariances.shape[-2:] != (3, 3):
        raise ValueError("covariances must have shape [N_atom,3,3]")
    if rotations.ndim != 3 or rotations.shape[-2:] not in ((2, 3), (3, 3)):
        raise ValueError("rotations must have shape [B,2,3] or [B,3,3]")
    dtype = torch.promote_types(covariances.dtype, rotations.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    u = rotations[:, :2, :].to(dtype=dtype)
    cov = covariances.to(device=u.device, dtype=dtype)
    return u[:, None] @ cov[None] @ u[:, None].transpose(-1, -2)


def _render_covariance_chunk(centers, mass, covariance, height, width, cutoff):
    """Functional raster block; can be recomputed by activation checkpointing."""
    a = covariance[..., 0, 0]
    b = covariance[..., 1, 1]
    c = covariance[..., 0, 1]
    det = a * b - c.square()
    # Bounds are a discrete support choice; all density computations remain
    # differentiable. The rectangle encloses the Mahalanobis cutoff ellipse.
    with torch.no_grad():
        sx, sy = a.sqrt(), b.sqrt()
        x0 = max(0, min(width, int((centers[..., 0] - cutoff * sx).min().floor())))
        x1 = max(0, min(width, int((centers[..., 0] + cutoff * sx).max().ceil()) + 1))
        y0 = max(0, min(height, int((centers[..., 1] - cutoff * sy).min().floor())))
        y1 = max(0, min(height, int((centers[..., 1] + cutoff * sy).max().ceil()) + 1))
    if x1 <= x0 or y1 <= y0:
        zero = (centers.sum() + mass.sum() + covariance.sum()) * 0
        return centers.new_zeros((centers.shape[0], height, width)) + zero
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=centers.device, dtype=centers.dtype),
        torch.arange(x0, x1, device=centers.device, dtype=centers.dtype),
        indexing="ij",
    )
    dx = xx[None, None] - centers[..., 0, None, None]
    dy = yy[None, None] - centers[..., 1, None, None]
    aa, bb, cc, dd = [v[..., None, None] for v in (a, b, c, det)]
    distance = (bb * dx.square() - 2 * cc * dx * dy + aa * dy.square()) / dd
    peak = mass[None, :, None, None] / (2 * torch.pi * dd.sqrt())
    # A per-Gaussian cutoff makes results independent of atom chunk partition.
    pixels = peak * torch.exp(-0.5 * distance) * (distance <= cutoff * cutoff)
    return F.pad(pixels.sum(dim=1), (x0, width - x1, y0, height - y1))


def sum_of_gaussians_2d_covariance(
    centers, mass, covariance, box_size, atom_chunk_size=64,
    cutoff=5.0, checkpoint_chunks=True,
):
    """Sum analytically projected Gaussians before global image normalization.

    centers [B,N,2], mass [N], covariance [B,N,2,2]. mass is the integral
    before finite support, finite-pixel quadrature and the caller's common scale.
    Checkpointing bounds saved raster activations; this is a reference PyTorch
    renderer, not a fused CUDA splatting implementation.
    """
    if centers.ndim != 3 or centers.shape[-1] != 2 or centers.shape[1] == 0:
        raise ValueError("centers must have nonempty shape [B,N_atom,2]")
    if mass.shape != (centers.shape[1],):
        raise ValueError("mass must have shape [N_atom]")
    if covariance.shape != (*centers.shape[:2], 2, 2):
        raise ValueError("covariance must have shape [B,N_atom,2,2]")
    if atom_chunk_size <= 0 or box_size <= 0 or cutoff <= 0:
        raise ValueError("chunk size, box size and cutoff must be positive")
    dtype = torch.promote_types(centers.dtype, covariance.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    centers = centers.to(dtype=dtype)
    mass = mass.to(device=centers.device, dtype=dtype)
    covariance = covariance.to(device=centers.device, dtype=dtype)
    with torch.no_grad():
        a, b, c = covariance[..., 0, 0], covariance[..., 1, 1], covariance[..., 0, 1]
        if not (torch.isfinite(centers).all() and torch.isfinite(mass).all()
                and torch.isfinite(covariance).all() and (a > 0).all()
                and (b > 0).all() and (a * b - c.square() > 0).all()):
            raise ValueError("non-finite inputs or non-positive projected covariance")
    image = centers.new_zeros((centers.shape[0], box_size, box_size))
    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled() and any(
        x.requires_grad for x in (centers, mass, covariance)
    )
    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint
    for start in range(0, centers.shape[1], atom_chunk_size):
        end = min(start + atom_chunk_size, centers.shape[1])
        args = (centers[:, start:end], mass[start:end], covariance[:, start:end],
                box_size, box_size, cutoff)
        block = (checkpoint(_render_covariance_chunk, *args, use_reentrant=False,
                            preserve_rng_state=False) if use_checkpoint
                 else _render_covariance_chunk(*args))
        image = image + block
    return image
