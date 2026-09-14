"""Template-based training exports; the reference CIF remains mandatory."""
from pathlib import Path

import gemmi
import numpy as np

from structure_io import write_coordinates
from utils import replace_cif_coordinates


def output_formats(output_format):
    if output_format not in ('cif', 'pdb', 'both'):
        raise ValueError('output_format must be cif, pdb or both')
    return ('cif', 'pdb') if output_format == 'both' else (output_format,)


def validate_output_template(template, n_atom, output_format):
    """Check unsupported topology/PDB limits before loading a diffusion model."""
    formats = output_formats(output_format)
    structure = gemmi.read_structure(str(template))
    if len(structure) != 1:
        raise ValueError('Training output requires a single-model reference CIF')
    atoms = [atom for chain in structure[0] for residue in chain for atom in residue]
    if len(atoms) != n_atom:
        raise ValueError('Reference CIF export atom count mismatch; alternate locations are not supported by training output')
    if 'pdb' in formats:
        if len(atoms) + len(structure[0]) > 99999:
            raise ValueError('PDB atom/TER serial limit exceeded; use --output-format cif')
        if any(len(chain.name) != 1 for chain in structure[0]):
            raise ValueError('PDB requires single-character chain IDs; use --output-format cif')
        if any(not -999 <= residue.seqid.num <= 9999 for chain in structure[0] for residue in chain):
            raise ValueError('PDB residue numbering limit exceeded; use --output-format cif')


def export_training_structure(template, stem, coordinates, output_format='cif'):
    xyz = np.asarray(coordinates)
    formats = output_formats(output_format)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Training output coordinates must be finite [N,3]')
    validate_output_template(template, len(xyz), output_format)
    if 'pdb' in formats and any(len(f'{float(v):8.3f}') > 8 for v in xyz.flat):
        raise ValueError('PDB coordinate field overflow; use --output-format cif')
    paths = [Path(str(stem) + '.' + fmt) for fmt in formats]
    for path in paths:
        if path.exists():
            raise FileExistsError(f'Structure output already exists: {path}')
    for fmt, path in zip(formats, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == 'cif':
            # Edit atom_site coordinates in place, preserving template identities/categories.
            write_coordinates(template, path, xyz)
        else:
            # Keep the historical PDB formatting for explicitly requested PDB output.
            replace_cif_coordinates(str(template), str(path), xyz)
    return [str(path) for path in paths]
