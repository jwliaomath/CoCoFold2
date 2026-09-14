"""Public single-state case helpers: placement checks and real-run validation.

Manual map fitting is an explicit user step. These helpers never fit a model
to the target CIF or modify training defaults.
"""
import json
from pathlib import Path

import gemmi
import numpy as np
import torch

from simulate_particles import sha256, write_star
from structure_io import read_template, write_coordinates


def atom_keys(path):
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError('Placement must contain one model')
    keys = [(chain.name, residue.seqid.num, residue.seqid.icode, residue.name, atom.name, atom.element.name)
            for chain in structure[0] for residue in chain for atom in residue]
    if len(keys) != len(set(keys)) or any(atom.altloc not in ('\0',' ') for chain in structure[0] for residue in chain for atom in residue):
        raise ValueError('Duplicate atom identities or alternate locations in placement')
    return keys


def validate_placement(initial_cif, placed_cif, reference_structure=False):
    """Accept a rigidly moved copy of THIS prediction, preserving atom order."""
    if atom_keys(initial_cif) != atom_keys(placed_cif):
        raise ValueError('Placed CIF must preserve current prediction atom identities and order; do not use an old cache or target CIF')
    initial, _, labels = read_template(initial_cif)
    placed, _, placed_labels = read_template(placed_cif)
    if initial.shape != placed.shape or len(initial) < 3:
        raise ValueError('Placement atom count mismatch')
    x, y = initial.double().numpy(), placed.double().numpy()
    u, _, vt = np.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
    d = np.eye(3); d[-1,-1] = np.linalg.det(u@vt)
    rotation = u@d@vt
    translation = y.mean(0)-x.mean(0)@rotation
    mse = float(np.mean((x@rotation+translation-y)**2))
    if not reference_structure and mse > 1e-4:
        raise ValueError(f'Placement changed conformation (rigid-fit coordinate MSE={mse}); only rigid map fitting is allowed')
    return dict(atom_count=len(x), atom_identity_order='matched', rigid_component_mse_A2=mse,
        reference_kind='reference_structure' if reference_structure else 'rigidly_moved_prediction',
        initial_sha256=sha256(initial_cif), placed_sha256=sha256(placed_cif),
        rotation=rotation.T.tolist(), translation=translation.tolist(),
        convention='row coordinates @ rotation.T + translation',
        map_fit_quality='Not automatically measured; user must fit and inspect against the supplied molmap')


def canonical_placement(initial, placed, output, reference_structure=False):
    report = validate_placement(initial, placed, reference_structure)
    # Preserve cache label identities even if the visualization tool rewrites
    # mmCIF labels while retaining author identities and atom order.
    xyz = read_template(placed)[0].numpy()
    if Path(output).exists():
        raise FileExistsError(output)
    write_coordinates(initial, output, xyz)
    report['reference_sha256'] = sha256(output)
    return report


def select_particles(directory, output, count):
    import starfile
    directory = Path(directory).resolve()
    data = starfile.read(directory/'particles.star', always_dict=True)
    frame = data['images']
    if not 1 <= count <= len(frame):
        raise ValueError('Particle subset size out of range')
    frame = frame.iloc[:count].copy()
    # The subset lives in the run directory; retain explicit absolute paths to
    # the unchanged MRCS, no copy and no re-simulation of the subset.
    frame['rlnImageName'] = [f'{i+1}@{(directory/"particles.mrcs").as_posix()}' for i in range(count)]
    write_star(frame, output)


def fixed_particle_frc(initial, final, gmm, star, box, apix, device, count=8):
    """Mean unscaled FRC on the same first views; no decoder or backpropagation."""
    from particledataset import ParticleDataset
    from ctf import compute_ctf
    from utils import compute_frc
    data = ParticleDataset(str(star), '', apix, norm=False)
    axis = np.linspace(-.5, .5, box, endpoint=False)/apix
    freqs = torch.tensor(np.stack(np.meshgrid(axis,axis),-1).reshape(1,-1,2), dtype=torch.float32, device=device)
    scores = [[], []]
    with torch.no_grad():
        for i in range(min(count, len(data))):
            image, p, shift, rotation, _, _ = data[i]
            p = torch.tensor(p, dtype=torch.float32, device=device).reshape(8,1,1)
            ctf = compute_ctf(freqs,p[1],p[2],p[3],p[0],p[4],p[5],p[6]).reshape(1,1,box,box)
            for j, coordinates in enumerate((initial, final)):
                projection = -gmm(atoms_coord=coordinates.to(device).reshape(1,-1,3), resolution=3.,
                    rotation=torch.tensor(rotation, dtype=torch.float32, device=device)[None],
                    trans=torch.tensor(shift, dtype=torch.float32, device=device)[None],
                    density_center=torch.tensor([box/2,box/2], dtype=torch.float32, device=device),
                    box_size=box, cutoff_range=5, sigma_factor=1/(np.pi*np.sqrt(2)), apix=apix)
                score = float(compute_frc(projection.float(), torch.tensor(image,device=device)[None,None].float(),
                    ctf.float(), box_size=box, max_freq=2*apix/3.))
                if not np.isfinite(score):
                    raise ValueError('Nonfinite fixed-particle FRC')
                scores[j].append(score)
    return dict(indices=list(range(len(scores[0]))), initial_mean=float(np.mean(scores[0])),
        final_mean=float(np.mean(scores[1])), delta=float(np.mean(scores[1])-np.mean(scores[0])),
        initial_per_particle=scores[0], final_per_particle=scores[1],
        definition='Unscaled FRC averaged over the same first particles, 3 Å cutoff; report only, not a new training loss or pass threshold')


def validate_run(directory, epochs, expected_steps, reference, star, device='cpu'):
    """Validate saved state and files without executing the diffusion model again."""
    from gmm import GaussianProjector
    from checkpoint_sampling import global_coordinates
    directory = Path(directory)
    result = dict(passed=False, checks=[], rmsd=dict(status='manual', reference_target_A=2.,
        note='User may use US-align or another structural aligner; report matched chains/residues and coverage. Not an automatic gate.'))
    def check(name, condition, detail=''):
        result['checks'].append(dict(name=name, passed=bool(condition), detail=detail))
    try:
        cache = torch.load(directory/f'run_{epochs}.pth', map_location='cpu', weights_only=False)
        xyz, weights, labels = read_template(reference)
        initial_gmm = GaussianProjector(weights, kernel='legacy')
        saved_gmm = cache['gmm']['state_dict']
        check('GMM amplitude and width unchanged from original initializer',
            saved_gmm.keys() == initial_gmm.state_dict().keys() and all(torch.equal(saved_gmm[k], v) for k,v in initial_gmm.state_dict().items()))
        check('GMM learning disabled', cache['gmm_learning_enabled'] is False)
        check('Latent finite and updated', torch.isfinite(cache['z_bias']).all() and torch.any(cache['z_bias'] != 0))
        states = list(cache['opt_state']['state'].values())
        check('Optimizer has only latent group', len(cache['opt_state']['param_groups']) == 1 and len(states) == 1)
        check('Optimizer moments finite', bool(states) and all(torch.isfinite(s[k]).all() for s in states for k in ('exp_avg','exp_avg_sq')))
        check('Expected optimizer progress', cache['training_progress']['global_step'] == expected_steps and
            all(int(s['step']) == expected_steps for s in states))
        check('Checkpoint epoch complete', cache['training_resume']['epoch_complete'] and cache['training_resume']['next_epoch'] == epochs)
        final, _, final_labels = read_template(directory/f'run_{epochs}.cif')
        expected = global_coordinates(cache, cache['pred_dict']['coordinate'][0]).float()
        mse = float((expected-final).double().square().mean())
        check('CIF matches saved coordinates and atom identities', labels == final_labels and mse < 1e-8, f'coordinate MSE={mse} Å²')
        record = directory/'records'
        summary = json.loads((record/'run_summary.json').read_text(encoding='utf-8'))
        metrics = [json.loads(line) for line in (record/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        steps = [r for r in metrics if r['event'] == 'train_step']
        check('Structured record complete', summary['status'] == 'success' and len(steps) == expected_steps and
            (record/'resolved_config.json').is_file() and (record/'command.json').is_file())
        check('All recorded losses finite', bool(steps) and all(np.isfinite(r['total_loss']) for r in steps))
        result['runtime'] = dict(elapsed_seconds=summary['elapsed_seconds'], train_step_seconds=sum(r['elapsed_seconds'] for r in steps),
            peak_memory_bytes=max((r['peak_memory_bytes'] or 0) for r in steps) if steps else None)
        raw_initial = read_template(directory/'run__.cif')[0]
        if cache['training_resume']['science_args']['update_affine_mat']:
            raise ValueError('This fixed-initial placement comparison requires update_affine_mat disabled')
        initial = global_coordinates(cache, raw_initial).float()
        initial_gmm = GaussianProjector.from_checkpoint(cache['gmm'], device)
        initial_gmm.atom_chunk_size = 1000
        science = cache['training_resume']['science_args']
        result['fixed_particle_frc'] = fixed_particle_frc(initial, final, initial_gmm, star, science['boxsize'], science['apix'], device)
        result['passed'] = all(row['passed'] for row in result['checks'])
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)
    finally:
        (directory/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        lines = [f"Passed: {result['passed']}", f"Checks: {len(result['checks'])}"]
        lines.extend(f"{'PASS' if row['passed'] else 'FAIL'}: {row['name']} {row['detail']}" for row in result['checks'])
        if 'error' in result: lines.append(result['error'])
        lines.append('RMSD: manual inspection, not an automatic pass condition.')
        (directory/'validation.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
