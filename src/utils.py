import torch
import numpy as np
import mrcfile
import gemmi
from torch.nn import functional as F
from utils_halfmap import interpolate_weights_by_frequency

def mrcread(fpath : str, iSlc = None):
    with mrcfile.mmap(fpath, permissive = True, mode = 'r') as mrc:
        data = mrc.data if iSlc is None or mrc.data.ndim == 2 else mrc.data[iSlc]
        return np.array(data, dtype = np.float64)

def cif_to_tensor(cif_path: str) -> tuple[torch.Tensor, torch.Tensor]:

    element_to_z = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8,
        'P': 15, 'S': 16, 'W': 74, 'K': 19, 'AU': 79
    }

    desired_tags = [
        "_atom_site.label_alt_id",
        "_atom_site.type_symbol",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]

    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()

        table = block.find(desired_tags)
        if not table:
            raise ValueError("missing required tags")

        col_map = {tag: i for i, tag in enumerate(table.tags)}

        coords = []
        atomic_numbers = []

        for row in table:
            alt_id = row[col_map["_atom_site.label_alt_id"]]
            if alt_id not in ('.', ''):
                continue

            elem = row[col_map["_atom_site.type_symbol"]].upper()
            if elem not in element_to_z:
                print(f"missing element")
                raise
            atomic_numbers.append(element_to_z[elem])


            coords.append([
                float(row[col_map["_atom_site.Cartn_x"]]),
                float(row[col_map["_atom_site.Cartn_y"]]),
                float(row[col_map["_atom_site.Cartn_z"]])
            ])

        return (
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(atomic_numbers, dtype=torch.float32, requires_grad=True)
        )

    except Exception as e:
        print(f"{str(e)}")
        raise


def kabsch_alignment(pred_coords: torch.Tensor,
                     ref_coords: torch.Tensor,
                     return_transform: bool = False
                    ) -> torch.Tensor | tuple:
    """
    Align predicted coordinates to reference coordinates using Kabsch algorithm.
    
    Args:
        pred_coords: Predicted coordinates tensor of shape [N, 3]
        ref_coords: Reference coordinates tensor of shape [N, 3]
        return_transform: Whether to return transformation components
    
    Returns:
        aligned_coords: Aligned coordinates tensor of shape [N, 3]
        (Optional) rotation: Optimal rotation matrix of shape [3, 3]
        (Optional) translation: Translation vector of shape [3]
        rmsd: Root Mean Square Deviation after alignment
    
    Note:
        - Requires exactly matched atom ordering between inputs
        - Handles reflection cases via determinant check
        - Centroid alignment removes translational component
    """
    # Input validation
    assert pred_coords.shape == ref_coords.shape, "Coordinate shape mismatch"
    assert pred_coords.dim() == 2 and pred_coords.size(1) == 3, "Expected [N, 3] shape"

    device = pred_coords.device
    pred_coords = pred_coords.float()
    ref_coords = ref_coords.to(device).float()

    # 1. Centroid alignment (remove translation)
    pred_centroid = pred_coords.mean(dim=0, keepdim=True)
    ref_centroid = ref_coords.mean(dim=0, keepdim=True)

    centered_pred = pred_coords - pred_centroid  # Centered predictions
    centered_ref = ref_coords - ref_centroid     # Centered reference

    # 2. Covariance matrix for optimal rotation
    cov = centered_pred.T @ centered_ref  # [3, 3] covariance matrix

    U, S, Vt = torch.linalg.svd(cov.float())
    D = torch.diag(torch.tensor([1., 1., torch.sign(torch.det(Vt.T @ U.T))], device=device))
    rotation = Vt.T @ D @ U.T

    # 5. Apply transformation
    aligned_coords = (rotation @ centered_pred.T).T + ref_centroid

    # Calculate RMSD
    # rmsd = torch.sqrt(torch.mean(torch.sum((aligned_coords - ref_coords)**2, dim=1)))

    if return_transform:
        translation = ref_centroid.squeeze() - rotation @ pred_centroid.squeeze()
        return aligned_coords.to(device), rotation.to(device), translation.to(device)
    else:
        return aligned_coords.to(device)
def compute_frc(proj, data, ctf, box_size = 240, max_freq=1):
    N = proj.shape[0]

    # Perform 2D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    F1 = F1 * ctf
    F2 = torch.fft.fftshift(torch.fft.fft2(data), dim=(-2,-1))

    # Calculate the frequency grid in polar coordinates
    ny = box_size
    nx = box_size
    y, x = torch.meshgrid(torch.arange(-ny // 2, ny // 2,device=proj.device), torch.arange(-nx // 2, nx // 2,device=proj.device))

    freq_radius = torch.sqrt(x ** 2 + y ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0).expand(N,1,box_size,box_size)
    # Number of frequency bins
    max_radius = int(box_size //2 * max_freq)
    frc = torch.zeros(N,max_radius,device=proj.device)

    for radius in range(1, max_radius):
        mask = (freq_radius == radius)
        if mask.sum() == 0:
            continue

        # Sum over all angles θ for the current radius k
        P_k_theta = F1[mask].reshape(N,-1)
        I_k_theta = F2[mask].reshape(N,-1)

        # Calculate the dot product for this ring
        numerator = torch.sum(P_k_theta.real * I_k_theta.real + P_k_theta.imag * I_k_theta.imag,dim=-1)

        denominator = torch.sqrt(
            torch.sum(P_k_theta.real ** 2 + P_k_theta.imag ** 2,dim=-1) * torch.sum(I_k_theta.real ** 2 + I_k_theta.imag ** 2,dim=-1))

        frc[:,radius] = numerator / (denominator + 1e-8)  # Add small value to prevent division by zero

    # Normalize FRC
    # frc = 2 * frc / (1 + frc)
    frc = torch.sum(frc)/max_radius

    return frc

def replace_cif_coordinates(
    input_cif: str,
    output_cif: str,
    new_coords: np.ndarray,
    coord_type: str = "cartesian"
) -> None:
    """
    Replace template atomic coordinates; this legacy helper writes PDB output.

    Args:
        input_cif: Input CIF template path.
        output_cif: Output path (legacy PDB writer).
        new_coords: New coordinates with shape [N_atoms, 3].
        coord_type: Coordinate type ('cartesian' or 'fractional').

    Raises:
        ValueError: Atom count mismatch or unsupported coordinate type.
    """
    # Read the CIF structure.
    struct = gemmi.read_structure(input_cif)

    # Collect all atoms and validate their count.
    all_atoms = [atom for model in struct for chain in model for residue in chain for atom in residue]
    if len(all_atoms) != new_coords.shape[0]:
        raise ValueError(f"Atom count mismatch: CIF contains {len(all_atoms)}, new coordinates contain {new_coords.shape[0]}")

    # Read the unit-cell parameters.
    cell = struct.cell

    # Replace coordinates for each atom.
    for i, atom in enumerate(all_atoms):
        x, y, z = new_coords[i]

        # Handle the requested coordinate convention.
        if coord_type == "cartesian":
            # Historical Cartesian-to-fractional conversion (inactive).
            # frac_pos = cell.fractionalize(gemmi.Position(x, y, z))
            # atom.pos = frac_pos
            atom.pos = gemmi.Position(x, y, z)
        elif coord_type == "fractional":
            # Convert fractional coordinates to Cartesian coordinates.
            cart_pos = cell.orthogonalize(gemmi.Fractional(x, y, z))
            atom.pos = cart_pos
        else:
            raise ValueError(f"Invalid coordinate type: {coord_type}; expected 'cartesian' or 'fractional'")

    # Write the output file.
    struct.write_minimal_pdb(output_cif)

def replace_cif_coordinates_cif(
    input_cif: str,
    output_cif: str,
    new_coords: np.ndarray,
    coord_type: str = "cartesian",
) -> None:
    import gemmi
    import numpy as np

    struct = gemmi.read_structure(input_cif)

    all_atoms = [
        atom
        for model in struct
        for chain in model
        for residue in chain
        for atom in residue
    ]

    if len(all_atoms) != new_coords.shape[0]:
        raise ValueError(
            f"Atom count mismatch: CIF contains {len(all_atoms)}, new coordinates contain {new_coords.shape[0]}"
        )

    cell = struct.cell

    for i, atom in enumerate(all_atoms):
        x, y, z = map(float, new_coords[i])

        if coord_type == "cartesian":
            atom.pos = gemmi.Position(x, y, z)

        elif coord_type == "fractional":
            cart_pos = cell.orthogonalize(gemmi.Fractional(x, y, z))
            atom.pos = cart_pos

        else:
            raise ValueError(
                f"Invalid coordinate type: {coord_type}; expected 'cartesian' or 'fractional'"
            )

    # Write mmCIF content with the mmCIF writer.
    doc = struct.make_mmcif_document()
    doc.write_file(output_cif)

def deep_clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deep_clone(item) for item in obj)
    else:
        return obj

def compute_frc_simulate(proj, data, box_size = 240, max_freq=1):
    N = proj.shape[0]

    # Perform 2D Fourier Transform
    F1 = torch.fft.fftshift(torch.fft.fft2(proj), dim=(-2, -1))
    F2 = torch.fft.fftshift(torch.fft.fft2(data), dim=(-2,-1))

    # Calculate the frequency grid in polar coordinates
    ny = box_size
    nx = box_size
    y, x = torch.meshgrid(torch.arange(-ny // 2, ny // 2), torch.arange(-nx // 2, nx // 2))

    freq_radius = torch.sqrt(x ** 2 + y ** 2).long()
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0).expand(N,1,box_size,box_size)
    # Number of frequency bins
    max_radius = int(box_size //2 * max_freq)
    frc = torch.zeros(N,max_radius)

    for radius in range(1, max_radius):
        mask = (freq_radius == radius)
        if mask.sum() == 0:
            continue

        # Sum over all angles θ for the current radius k
        P_k_theta = F1[mask].reshape(N,-1)
        I_k_theta = F2[mask].reshape(N,-1)

        # Calculate the dot product for this ring
        numerator = torch.sum(P_k_theta.real * I_k_theta.real + P_k_theta.imag * I_k_theta.imag,dim=-1)

        denominator = torch.sqrt(
            torch.sum(P_k_theta.real ** 2 + P_k_theta.imag ** 2,dim=-1) * torch.sum(I_k_theta.real ** 2 + I_k_theta.imag ** 2,dim=-1))

        frc[:,radius] = numerator / (denominator + 1e-8)  # Add small value to prevent division by zero

    # Normalize FRC
    # frc = 2 * frc / (1 + frc)

    frc = torch.sum(frc)/max_radius

    return frc

def discrete_radon_transform_3d(volume, rotation):
    volume = volume.expand(rotation.shape[0],1,volume.shape[-3],volume.shape[-2],volume.shape[-1])

    b = volume.shape[0]

    zeros = torch.zeros(b, 3, 1).to(volume.device)

    theta = torch.cat([rotation, zeros], dim=2)

    grid = F.affine_grid(theta, size=volume.shape)

    volume_rot = F.grid_sample(volume, grid, mode='bilinear')

    # volume_rot = volume_rot.permute(0, 1, 2, 3, 4)
    volume_rot = volume_rot.permute(0, 1, 3, 4, 2)
    proj = volume_rot.sum(dim=-1)

    return proj

def translation_2d(proj, trans):
    """
    Input:
        proj: Bx1xbsxbs tensor 
        trans: Bx2 tensor
    """

    b = trans.shape[0]

    eye = torch.eye(2).unsqueeze(0).repeat(b, 1, 1).to(proj.device)
    trans = trans.unsqueeze(-1)
    trans = trans * 2 / proj.shape[-1]
    theta = torch.cat([eye, trans], dim=2)

    grid = F.affine_grid(theta, size=proj.shape)
    proj_trans = F.grid_sample(proj, grid, mode='bicubic')

    return proj_trans

def compute_ncc_loss(proj, data):
    """
    Compute real-space normalized cross-correlation (NCC).
    Return negative correlation for minimization (higher correlation gives lower loss).
    """
    B = proj.shape[0]
    # Flatten the images.
    p = proj.view(B, -1)
    d = data.view(B, -1)

    # Subtract the mean to center each image.
    p_mean = p.mean(dim=1, keepdim=True)
    d_mean = d.mean(dim=1, keepdim=True)
    p_centered = p - p_mean
    d_centered = d - d_mean

    # Compute covariance and standard deviations.
    numerator = torch.sum(p_centered * d_centered, dim=1)
    denominator = torch.sqrt(torch.sum(p_centered ** 2, dim=1) * torch.sum(d_centered ** 2, dim=1) + 1e-8)

    # Compute NCC and sum the correlations.
    ncc = numerator / denominator

    # Return the negative mean correlation as the loss.
    return -torch.sum(ncc)