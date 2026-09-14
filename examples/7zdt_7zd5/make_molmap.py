"""Run inside ChimeraX: --nogui --exit --script <this file>.

COCOFOLD2_MOLMAP_CONFIG names a JSON containing target_cif, output_dir,
boxsize, apix and resolution. Original input atoms are preserved in full.
"""
import json
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from chimerax.core.commands import run
from chimerax.map import volume_from_grid_data
from chimerax.map.molmap import molmap
from chimerax.map_data import ArrayGridData
from chimerax.map_data.mrc.writemrc import write_mrc2000_grid_data


def main(session):
    config_path = Path(os.environ['COCOFOLD2_MOLMAP_CONFIG']).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    source = Path(cfg['target_cif']).resolve()
    out = Path(cfg['output_dir']).resolve()
    n, apix, resolution = int(cfg['boxsize']), float(cfg['apix']), float(cfg['resolution'])
    if n < 8 or n % 2 or not np.isfinite([apix,resolution]).all() or min(apix,resolution) <= 0:
        raise ValueError('Invalid molmap grid/resolution')
    if not source.is_file():
        raise FileNotFoundError(source)
    out.mkdir(parents=True, exist_ok=False)
    models = run(session, 'open ' + json.dumps(str(source)))
    structures = [m for m in models if hasattr(m, 'atoms')]
    if len(structures) != 1:
        raise ValueError('Expected exactly one atomic structure')
    model = structures[0]
    coordinates = np.asarray(model.atoms.coords, dtype=np.float64)
    weights = np.array([atom.element.number for atom in model.atoms], dtype=np.float64)
    center = np.average(coordinates, weights=weights, axis=0)
    centered = coordinates-center
    margin = n*apix/2 - np.abs(centered).max()
    if margin < 4*resolution:
        raise ValueError(f'Map box too small: only {margin:.3f} Å margin; need >= 4*resolution')
    model.atoms.coords = centered
    origin = (-n*apix/2,)*3
    grid = volume_from_grid_data(ArrayGridData(np.zeros((n,n,n),dtype=np.float32),
        origin=origin, step=(apix,)*3), session, open_model=False)
    volume = molmap(session, model.atoms, resolution, on_grid=grid, replace=False,
                    show_dialog=False, open_model=False)
    values = volume.data.full_matrix()
    if values.shape != (n,n,n) or not np.isfinite(values).all():
        raise ValueError('Invalid ChimeraX molmap output')
    path = out/'7zd5_full_3A.mrc'
    write_mrc2000_grid_data(volume.data, str(path))
    report = dict(schema_version=1, status='completed', source_cif=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        atom_count=len(coordinates), atom_policy='All original CIF atoms loaded by ChimeraX; no residue completion, mutation or chain trimming',
        source_to_map=dict(rotation=np.eye(3).tolist(), translation=(-center).tolist(), convention='row coordinates @ rotation.T + translation'),
        boxsize=n, apix=apix, resolution_A=resolution, origin_A=list(origin),
        boundary_margin_A=float(margin), map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        command=sys.argv, config=cfg, molmap='ChimeraX molmap, original B6 kernel settings',
        python=sys.version)
    try:
        from chimerax.core import __version__
        report['chimerax_version'] = __version__
    except ImportError:
        report['chimerax_version'] = 'see ChimeraX log'
    (out/'molmap.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    session.models.close([model,volume])
    print(json.dumps(report, indent=2), flush=True)


main(session)
