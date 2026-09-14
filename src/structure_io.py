"""Template-preserving structure I/O shared by public refinement entry points.

Keep template row order and the historical alternate-location filter. File
format follows the template, as in the existing block-refinement workflow.
Template-free export lives in cache_structure.
"""
from pathlib import Path

import gemmi
import numpy as np
import torch


def read_template(path):
    """Return coordinates, atomic numbers and labels in template atom order."""
    path = Path(path)
    coords, weights, labels = [], [], []
    if path.suffix.lower() in ('.cif', '.mmcif'):
        block = gemmi.cif.read_file(str(path)).sole_block()
        tags = ['label_alt_id', 'type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                'label_asym_id', 'label_seq_id', 'label_atom_id']
        table = block.find('_atom_site.', tags)
        if not table:
            raise ValueError('Template lacks required atom_site columns')
        for row in table:
            if row[0] not in ('.', ''):
                continue
            coords.append([float(row[k]) for k in (2, 3, 4)])
            weights.append(gemmi.Element(row[1]).atomic_number)
            labels.append('|'.join([row[5], row[6], row[7]]))
    else:
        structure = gemmi.read_structure(str(path))
        if len(structure) != 1:
            raise ValueError('Template must contain one model')
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    if atom.altloc not in ('\x00', ' '):
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    weights.append(atom.element.atomic_number)
                    labels.append(f'{chain.name}|{residue.seqid}|{atom.name}')
    if not coords or min(weights) <= 0:
        raise ValueError('Empty template or unknown atomic element')
    xyz = torch.tensor(coords, dtype=torch.float32)
    if not torch.isfinite(xyz).all():
        raise ValueError('Nonfinite template coordinates')
    return xyz, torch.tensor(weights, dtype=torch.float32), labels


def write_coordinates(template, output, coordinates):
    """Replace coordinates without changing template atom order or identities."""
    xyz = np.asarray(coordinates)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Coordinates must be finite [A,3]')
    if Path(template).suffix.lower() in ('.cif', '.mmcif'):
        doc = gemmi.cif.read_file(str(template))
        table = doc.sole_block().find(
            '_atom_site.', ['label_alt_id', 'Cartn_x', 'Cartn_y', 'Cartn_z'])
        rows = [r for r in table if r[0] in ('.', '')]
        if len(rows) != len(xyz):
            raise ValueError('Template atom count mismatch')
        for row, point in zip(rows, xyz):
            for k in range(3):
                row[k + 1] = f'{point[k]:.6f}'
        doc.write_file(str(output))
    else:
        structure = gemmi.read_structure(str(template))
        atoms = [a for c in structure[0] for r in c for a in r
                 if a.altloc in ('\x00', ' ')]
        if len(atoms) != len(xyz):
            raise ValueError('Template atom count mismatch')
        for atom, point in zip(atoms, xyz):
            atom.pos = gemmi.Position(*map(float, point))
        structure.write_pdb(str(output))
