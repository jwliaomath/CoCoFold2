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

    '''
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        centers_batch = centers[:, start:end, :, :, :]
        sdev_batch = sdev[:, start:end, :, :, :]
        coef_batch = coef[start:end].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        coef_batch = coef_batch.expand(B, end - start, 1, 1)
        # print('y',torch.min(centers_batch[..., 1, :, :]),torch.max(centers_batch[..., 1, :, :]))
        # print('x',torch.min(centers_batch[..., 0, :, :]),torch.max(centers_batch[..., 0, :, :]))

        dy = (grid_y - centers_batch[..., 1, :, :]) / (sdev_batch[..., 1, :, :]+1e-8)
        dx = (grid_x - centers_batch[..., 0, :, :]) / (sdev_batch[..., 0, :, :]+1e-8)
        d2 = dy ** 2 + dx ** 2
        # print('d2',torch.min(d2),torch.max(d2))
        # print('coef_batch',torch.min(coef_batch),torch.max(coef_batch))
        
        gaussians = coef_batch * torch.exp(-0.5 * d2)  # Shape [B, batch_size, H, W]

        # Sum gaussians over the second dimension (current batch size)
        density += torch.sum(gaussians, dim=1)  # Shape [B, H, W]
    '''
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
    #    print("Warning: total_intensity contains zero values")

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
    '''
    if torch.isnan(theta).any():
        print("NaN detected in theta")
        print(f"theta: {theta}")
    '''
    grid = F.affine_grid(theta, size=proj.shape)

    proj_trans = F.grid_sample(proj, grid, mode='bicubic')

    '''
    new_H, new_W = int((H / apix)//2)*2, int((W * apix)//2)*2
    
    proj_downsampled = F.interpolate(proj_trans, size=(new_H, new_W), mode='bicubic', align_corners=False)
    
    pad_H = (box_size - new_H) // 2
    pad_W = (box_size - new_W) // 2
    
    proj_padded = F.pad(proj_downsampled, (pad_W, pad_W, pad_H, pad_H), mode='constant', value=0)
    '''

    return proj_trans


def translation_2d_robust(proj, trans, box_size, apix, density_center):
    """
    鲁棒的 2D 平移函数，专门针对含有负值背景的 Cryo-EM 密度图投影。
    Inputs:
        proj: Bx1xbsxbs tensor (允许包含负值)
        trans: Bx2 tensor
        box_size: int
        apix: float
        density_center: Bx2 tensor
    Output:
        proj_trans: Bx1xbsxbs tensor
    """
    B, _, H, W = proj.shape
    b = trans.shape[0]

    # 推荐加上 indexing='ij' 以消除 PyTorch 警告
    y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y_indices = y_indices.flatten().to(trans.dtype).to(proj.device)
    x_indices = x_indices.flatten().to(trans.dtype).to(proj.device)
    
    flat_images = proj.view(B, -1)
    
    # ==========================================
    # 核心修改：使用 ReLU 屏蔽负值，防止质心计算爆炸
    # ==========================================
    weights = torch.relu(flat_images)
    total_intensity = weights.sum(dim=1) + 1e-8 # 加上极小值防止除以 0
    weights_normalized = weights / total_intensity.view(B, 1)

    # 计算仅基于正密度的质心
    centroid_x = (weights_normalized * x_indices).sum(dim=1)
    centroid_y = (weights_normalized * y_indices).sum(dim=1)
    centroids = torch.stack([centroid_x, centroid_y], dim=1)

    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device).to(proj.dtype)

    # 像原来一样对齐到 density_center
    trans -= (density_center - centroids)
    trans = trans.unsqueeze(-1)
    trans = trans * 2 / box_size
    theta = torch.cat([eye, trans], dim=2)

    # 推荐显式声明 align_corners=False，符合现代 PyTorch 默认行为
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
    # print('atom coord xyz', torch.max(atoms_coord),torch.min(atoms_coord))
    # transform xy to the grid ij and make it into the box
    origin = torch.min(proj_rot, dim=1, keepdim=True).values
    proj_rot[..., 0] = proj_rot[..., 0] / step - origin[..., 0] / step
    proj_rot[..., 1] = proj_rot[..., 1] / step - origin[..., 1] / step
    proj_rot += pad

    # print('atom coord ijk', torch.max(atoms_coord),torch.min(atoms_coord))

    # projection
    img = sum_of_gaussians_2d_torch(centers=proj_rot.to(proj_rot.device), coef=atoms_weight.to(proj_rot.device), sdev=sdevs.to(proj_rot.device), maxrange=cutoff_range,matrices=torch.zeros(B, box_size, box_size).to(proj_rot.device).to(proj_rot.dtype))

    '''
    if torch.isnan(img).any():
        print("NaN detected in img after projection")
    '''
    normalization = torch.pow(2 * torch.pi, torch.tensor(-1)) * torch.pow(sdev, torch.tensor(-2))
    img *= normalization

    # scale to integrate z
    img /= step
    img = img.unsqueeze(1)
    # move it to the center first and then modify the translation
    img = translation_2d(img, trans / step, box_size, apix,density_center.to(img.device))

    '''
    if torch.isnan(img).any():
        print("NaN detected in img after translation")
    '''

    return img

def sum_of_gaussians_3d_torch(centers, coef, sdev, maxrange, matrices, batch_size=1):
    device = centers.device
    maxrange = torch.tensor(maxrange, device=device)
    sdev = sdev.to(device)
    B, N, _ = centers.shape
    D, H, W = matrices.shape[1:]
    '''
    # Prepare the range for grid
    grid_z ,grid_y, grid_x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    grid_z = grid_z.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)

    centers = centers.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape [B, N, 3, 1, 1]
    sdev = sdev.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Shape [1, N, 3, 1, 1]

    # Initialize density
    density = torch.zeros_like(matrices)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        centers_batch = centers[:, start:end, :, :, :]
        sdev_batch = sdev[:, start:end, :, :, :]

        coef_batch = coef[start:end].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        coef_batch = coef_batch.expand(B, end - start, 1, 1, 1)
        # print('y',torch.min(centers_batch[..., 1, :, :]),torch.max(centers_batch[..., 1, :, :]))
        # print('x',torch.min(centers_batch[..., 0, :, :]),torch.max(centers_batch[..., 0, :, :]))

        for i in range(240):
            dz = (grid_z[...,i:(i+1),:,:] - centers_batch[..., 2, :, :, :]) / sdev_batch[..., 2, :, :, :]
            dy = (grid_y[...,i:(i+1),:,:] - centers_batch[..., 1, :, :, :]) / sdev_batch[..., 1, :, :, :]
            dx = (grid_x[...,i:(i+1),:,:] - centers_batch[..., 0, :, :, :]) / sdev_batch[..., 0, :, :, :]
            d2= dx**2 + dy**2 + dz**2
            gaussians = coef_batch * torch.exp(-0.5 * d2)
            density[...,i:(i+1),:,:]+= torch.sum(gaussians, dim=1)

    matrices += density
    '''
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
    '''
    trans  = torch.zeros(B*3).reshape(B,3).to(proj.device)
    trans -= (box_size / 2)
        '''
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
        '''
    affine_matrix1 = torch.from_numpy(affine_matrix1).to(atoms_coord.device).to(torch.float)
    rotated_coords = torch.matmul(atoms_coord, affine_matrix1[:,:3].T) + affine_matrix1[:,3]
    rotated_coords /= apix
    '''
    # rotation
    if rotation is not None:
        atoms_coord = atoms_coord.transpose(1, 2)
        rotated_coords = torch.matmul(rotation, atoms_coord)
        rotated_coords = rotated_coords.transpose(1, 2)
    else:
        rotated_coords = atoms_coord 
    rotated_coords/=apix
    # print('atom coord xyz', torch.max(atoms_coord),torch.min(atoms_coord))
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

# ==================== 新增：3D 傅里叶切片渲染引擎 ====================

def pdb2mrc_v2(atoms_coord, resolution, atoms_weight, rotation=None, box_size=256, sdevs=None, cutoff_range=5, sigma_factor=1 / (np.pi * np.sqrt(2)), apix=1):
    """
    修改后的 pdb2mrc，原生支持 N x 3 的 sdevs 张量。
    """
    B, N, _ = atoms_coord.shape
    pad = 3 * resolution
    step = (1. / 3) * resolution
    sdev = resolution * sigma_factor
    
    # 直接使用传入的 [N, 3] sdevs，无需再做 2D 到 3D 的 padding
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
    """把生成的 3D MRC volume 转换至 3D 频域"""
    # 傅里叶变换三步曲: ifftshift -> fftn -> fftshift
    V_shift = torch.fft.ifftshift(vol_3d, dim=(-3, -2, -1))
    F_vol = torch.fft.fftn(V_shift, dim=(-3, -2, -1))
    F_vol = torch.fft.fftshift(F_vol, dim=(-3, -2, -1)) 
    return F_vol.real, F_vol.imag

def slice_fourier_volume(F_vol_real, F_vol_imag, rotations, trans, density_center, box_size, apix, step):
    B = rotations.shape[0]
    device = F_vol_real.device

    # 1. 建立 2D 基础网格 (频域平面 Z=0)
    lin = torch.linspace(-1, 1, box_size, device=device)
    Y, X = torch.meshgrid(lin, lin, indexing='ij')
    Z = torch.zeros_like(X)
    grid_2d = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3) 

    # 2. 根据旋转矩阵 R 反向旋转网格
    # 【核心修复】：由于物体正向旋转 R，切片网格必须反向旋转 R^T。
    # grid_2d 是 (K, 3) 的行向量，rotations 是 (B, 3, 3)
    # 行向量的 grid_2d @ rotations 刚好等效于 (R^T @ grid^T)^T，完美实现了 R^T 的旋转！
    rotated_grid = torch.matmul(grid_2d, rotations)
    rotated_grid = rotated_grid.view(B, 1, box_size, box_size, 3) 

    F_vol_real_exp = F_vol_real.expand(B, -1, -1, -1, -1)
    F_vol_imag_exp = F_vol_imag.expand(B, -1, -1, -1, -1)

    # 3. 使用 grid_sample 以极速双线性插值进行抽样切片
    slice_real = F.grid_sample(F_vol_real_exp, rotated_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    slice_imag = F.grid_sample(F_vol_imag_exp, rotated_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    slice_real = slice_real.squeeze(2)
    slice_imag = slice_imag.squeeze(2)
    F_slice = torch.complex(slice_real, slice_imag)

    # 4. 把 2D 频域切片转换回实空间图像
    F_slice = torch.fft.ifftshift(F_slice, dim=(-2, -1))
    proj = torch.fft.ifft2(F_slice, dim=(-2, -1)).real
    proj = torch.fft.fftshift(proj, dim=(-2, -1))
    
    # 积分缩放修正
    proj = proj * box_size 
    proj /= step

    # 5. 最后应用 2D 平移
    proj = translation_2d(proj, trans, box_size, apix, density_center)
    return proj