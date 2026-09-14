"""Initial-prediction input checks and collision-safe cache paths."""
import json
import os
from pathlib import Path

from cli_utils import check_output_directory, require_file


def safe_name(name):
    if not isinstance(name, str) or not name.strip() or name in ('.', '..') or any(c in name for c in '/\\:\x00'):
        raise ValueError(f'Invalid sample name: {name!r}')
    return name


def read_inputs(path):
    path = require_file(path, '--input_json_path')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or not data:
        raise ValueError('Input JSON must be a nonempty list of target objects')
    names = []
    for target in data:
        if not isinstance(target, dict):
            raise ValueError('Each input target must be an object')
        names.append(safe_name(target.get('name')))
        if not isinstance(target.get('sequences'), list) or not target['sequences']:
            raise ValueError(f'Target {names[-1]} requires a nonempty sequences list')
        if 'modelSeeds' in target and (not isinstance(target['modelSeeds'], list) or not target['modelSeeds']
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in target['modelSeeds'])):
            raise ValueError(f'Target {names[-1]} has invalid modelSeeds')
    if len(names) != len(set(names)):
        raise ValueError('Input target names must be unique')
    return data


def cache_path(root, sample, seed, single=False, alias=None):
    root = Path(root).expanduser()
    name = safe_name(sample)
    if single:
        return root / f'{safe_name(alias) if alias else name}_diffusion_data.pth'
    return root / name / f'seed_{int(seed)}' / 'diffusion_data.pth'


def cache_plan(configs, targets, seeds):
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Seeds must be nonempty integers in [0, 2**32)')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate seeds would overwrite prediction outputs')
    if configs.output_model_dir is None:
        return {}
    planned = {}
    for target in targets:
        for seed in seeds:
            path = cache_path(configs.output_model_dir, target['name'], seed,
                              single=len(targets) * len(seeds) == 1, alias=configs.sample_name)
            if path.exists():
                raise FileExistsError(f'Diffusion cache already exists: {path}')
            planned[(target['name'], seed)] = path
    paths = [os.path.normcase(str(path.resolve())) for path in planned.values()]
    if len(paths) != len(set(paths)):
        raise ValueError('Cache destination paths collide on this filesystem')
    for path in planned.values():
        check_output_directory(path.parent)
    return planned


def save_diffusion_cache(data, configs):
    """Never overwrite an existing cache, including calls outside inference.py."""
    import torch
    output = getattr(configs, 'cache_output_path', None)
    path = Path(output) if output else cache_path(configs.output_model_dir, configs.sample_name, 0, single=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        try:
            torch.save(data, handle)
        except BaseException:
            # The exclusive create succeeded, so this incomplete file is ours.
            handle.close()
            path.unlink()
            raise
    return path


def resolve_resource_paths(configs, resource_root=None, forwarded_args=()):
    """Keep cwd defaults; explicit Protenix path flags override the default root."""
    root = Path(resource_root or '.').expanduser().resolve()
    explicit = {token.split('=', 1)[0] for token in forwarded_args if token.startswith('--')}
    if '--load_checkpoint_dir' not in explicit:
        configs.load_checkpoint_dir = str(root / 'checkpoint')
    for key, filename in {
        'ccd_components_file': 'components.cif',
        'ccd_components_rdkit_mol_file': 'components.cif.rdkit_mol.pkl',
        'pdb_cluster_file': 'clusters-by-entity-40.txt',
        'obsolete_release_data_csv': 'obsolete_release_date.csv',
    }.items():
        if '--data.' + key not in explicit:
            configs['data'][key] = str(root / 'common' / filename)
    if 'template' in configs['data']:
        for key, filename in {'release_dates_path': 'release_date_cache.json',
                              'obsolete_pdbs_path': 'obsolete_to_successor.json'}.items():
            if '--data.template.' + key not in explicit:
                configs['data']['template'][key] = str(root / 'common' / filename)
    return configs
