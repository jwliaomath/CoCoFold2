"""Rank-coordinated preflight and component-local alignment; no model imports."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import gemmi

from coordinate_transform import CoordinateTransform, fit_rigid_row, template_atom_keys, validate_alignment_manifest
from randomness import resolve_seeds
from run_recording import current_record
from structure_io import read_template


def gather(value):
    if not dist.is_initialized():
        return [value]
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def phase(name, function):
    """Finish local work on all ranks before any rank starts the next collective."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = gather(error)
    if any(errors):
        raise RuntimeError(f'{name} failed: ' + '; '.join(f'rank {i}: {e}' for i,e in enumerate(errors) if e))
    return value


def seeds(args):
    # Unlike single-card train, historical parallel already used an explicit
    # shared-seed DataLoader generator. Keep this behavior in legacy mode.
    local = copy.copy(args)
    local.data_seed = None
    settings = resolve_seeds(local)
    settings.update(data_seed=args.seed if args.data_seed is None else args.data_seed,
                    data_rng='parallel_shared_seed_generator')
    return settings


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return dict(path=str(path), size=path.stat().st_size, sha256=digest.hexdigest())


def preflight(args, entry):
    from input_validation import validate_train_inputs
    from checkpoint_sampling import is_refinement
    from coordinate_transform import read_alignment_manifest, validate_update_threshold
    local = copy.copy(args)
    local.diffusion_data_dir = str(entry.diffusion_data_dir)
    local.cif_path = str(entry.cif_path)
    local.data_seed = None  # Checked by the parallel seed resolver instead.
    local.device = 'cpu'
    local.coordinate_mode = 'global'
    local.block_alignment = None
    local.resume = local.warm_start = False
    local.output_trained_model_dir = f'{args.output_trained_model_dir}{entry.component_id}_rank{entry.rank}_'
    output_dir = Path(local.output_trained_model_dir).parent
    for name in ('chain_parallel_2d_run_metadata.json', f'chain_parallel_2d_metrics_rank{entry.rank}.jsonl'):
        if (output_dir/name).exists():
            raise FileExistsError(f'Choose a new run directory; output already exists: {output_dir/name}')
    raw = torch.load(entry.diffusion_data_dir, map_location='cpu', weights_only=False)
    if raw.get('enable_efficient_fusion') is None:
        raw['enable_efficient_fusion'] = getattr(raw.get('configs'), 'enable_efficient_fusion', True)
    inspected = dict(raw)
    # Saved transforms are validated against the component CIF below. The
    # single-card validator must not select a different diffusion sampler.
    inspected.pop('coordinate_transform', None)
    _, dataset, report = validate_train_inputs(local, inspected)
    if args.resume or args.warm_start:
        from training_restart import restart_gmm
        from checkpoint_sampling import effective_latent, global_coordinates
        effective_latent(raw,raw['z_trunk'],raw['pair_z'])
        restart_gmm(raw,args,read_template(entry.cif_path)[1],'cpu')
        if raw.get('rotation') is not None and raw.get('translation') is not None:
            global_coordinates(raw,read_template(entry.cif_path)[0])
            r=torch.as_tensor(raw['rotation']).double()
            if not torch.allclose(r.T@r,torch.eye(3,dtype=r.dtype),atol=1e-5,rtol=1e-5) or abs(torch.linalg.det(r).item()-1)>1e-5:
                raise ValueError('Saved global rotation is not a proper rigid transform')
    if is_refinement(raw) and not (args.resume or args.warm_start):
        raise ValueError('Refinement input requires --warm-start or an epoch --resume index')
    if raw['N_sample'] != 1:
        raise ValueError('Parallel refinement requires N_sample=1')
    keys = template_atom_keys(entry.cif_path)
    features = raw['input_feature_dict']
    topology_check = 'legacy cache: explicit CIF identities and atom-count check'
    if 'ref_atom_name_chars' in features and 'ref_element' in features:
        from cache_structure import _onehot
        chars = _onehot(features, 'ref_atom_name_chars', (len(keys),4,64))
        atom_names = [''.join(chr(int(c)+32) for c in row).rstrip() for row in chars]
        elements = _onehot(features, 'ref_element', (len(keys),128)) + 1
        if any(k[3] != name or gemmi.Element(k[4]).atomic_number != int(element)
               for k,name,element in zip(keys,atom_names,elements)):
            raise ValueError('CIF/cache atom name or element ORDER mismatch')
        topology_check = 'CIF/cache atom name and element order checked'
    if 'asym_id' in features:
        cached_chains = features['asym_id'].reshape(-1)[features['atom_to_token_idx'].reshape(-1)].tolist()
        pairs = {(int(a),k[0]) for a,k in zip(cached_chains,keys)}
        if len(pairs) != len({a for a,c in pairs}) or len(pairs) != len({c for a,c in pairs}):
            raise ValueError('CIF chains do not map one-to-one to component cache asym_id groups')
    else:
        pairs = set()
    names = list(dict.fromkeys(k[0] for k in keys))
    mapping = entry.chain_id_map or {n:n for n in names}
    if (not isinstance(mapping, dict) or set(mapping) != set(names) or
            any(not isinstance(v,str) or not v.strip() for v in mapping.values()) or len(set(mapping.values())) != len(mapping)):
        raise ValueError('chain_id_map must map every local CIF chain to a unique nonempty global chain ID')
    global_keys = [[mapping[k[0]], *k[1:]] for k in keys]
    stacks = {}
    for name in dataset.particles['rlnImageName']:
        _,path = dataset.image_location(name)
        path = path.resolve()
        if str(path) not in stacks:
            stat=path.stat()
            stacks[str(path)] = dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    saved_transform = raw.get('coordinate_transform')
    if saved_transform is not None:
        transform = CoordinateTransform.from_checkpoint(saved_transform, entry.cif_path, 'cpu')
        if args.update_affine_mat and transform.fit_targets is None:
            raise ValueError('Saved transform lacks update fitting targets/core')
        if entry.alignment_manifest:
            raise ValueError('Saved transform exists; omit alignment_manifest')
    elif entry.alignment_manifest:
        read_alignment_manifest(entry.alignment_manifest, entry.cif_path)
    elif args.by_chain:
        chain_manifest(entry.cif_path, args.fit_atoms)
    validate_update_threshold(args.block_update_trace_threshold)
    if (args.by_chain or entry.alignment_manifest or saved_transform) and not args.train_deterministic:
        raise ValueError('Per-chain alignment requires fixed stochasticity')
    meta = raw.get('contextual_split_metadata')
    if meta is not None:
        from chain_parallel.contextual_cache import validate_component_cache
        validate_component_cache(raw, require_atom_local=True)
    report.update(component_id=entry.component_id, rank=entry.rank, atom_keys=keys, global_atom_keys=global_keys, seed_settings=seeds(args),
                  topology_check=topology_check, cache_asym_to_cif_chain=sorted(pairs), stacks_identity=stacks,
                  chain_id_map=mapping, contextual=meta, cache=identity(entry.diffusion_data_dir),
                  reference=identity(entry.cif_path), star=identity(args.star_data_dir),
                  alignment=identity(entry.alignment_manifest) if entry.alignment_manifest else None)
    return raw, dataset, report


def validate_partition(reports):
    flattened = [tuple(k) for r in reports for k in r['global_atom_keys']]
    if len(flattened) != len(set(flattened)):
        raise ValueError('Duplicate global atom identities; provide explicit chain_id_map for renamed local chains')
    chains = [set(k[0] for k in r['global_atom_keys']) for r in reports]
    if sum(map(len,chains)) != len(set.union(*chains)):
        raise ValueError('A global chain occurs in multiple components; split-chain components are unsupported')
    if len({(r['star']['sha256'],r['n_particles']) for r in reports}) != 1:
        raise ValueError('All ranks must use the same STAR and particle count')
    if len({json.dumps(r['stacks_identity'],sort_keys=True) for r in reports}) != 1:
        raise ValueError('All ranks must resolve the same MRCS stacks')
    contextual = [r['contextual'] for r in reports]
    if any(m is not None for m in contextual):
        if any(m is None for m in contextual):
            raise ValueError('Cannot mix Independent and Contextual caches')
        hashes = {m.get('source_cache_sha256') for m in contextual}
        sizes = {m['source_n_token'] for m in contextual}
        indices = [i for m in contextual for i in m['source_token_indices']]
        if None in hashes or len(hashes) != 1 or len(sizes) != 1 or sorted(indices) != list(range(next(iter(sizes)))):
            raise ValueError('Contextual components must be a complete disjoint partition of the same source cache')
        if sum(r['n_atoms'] for r in reports) != contextual[0]['source_n_atom']:
            raise ValueError('Contextual partition atom count mismatch')
        if all('source_atom_indices' in m for m in contextual):
            atoms=[i for m in contextual for i in m['source_atom_indices']]
            if sorted(atoms)!=list(range(contextual[0]['source_n_atom'])):
                raise ValueError('Contextual source atom indices are not a complete disjoint partition')


def chain_manifest(template, fit_atoms):
    keys = template_atom_keys(template)
    xyz = read_template(template)[0]
    names = list(dict.fromkeys(k[0] for k in keys))
    data = dict(format='cocofold2-block-manifest-v1',grouping='auth_asym_id',body_names=names,
        atoms=[dict(key=k,body_id=names.index(k[0]),target=p,fit_core=fit_atoms=='all' or k[3]=='CA')
               for k,p in zip(keys,xyz.tolist())])
    validate_alignment_manifest(data,keys)
    return data


def fit_chains(template, raw, fit_atoms):
    data = chain_manifest(template,fit_atoms)
    keys = template_atom_keys(template)
    ids = torch.tensor([a['body_id'] for a in data['atoms']],device=raw.device)
    core = torch.tensor([a['fit_core'] for a in data['atoms']],device=raw.device)
    target = read_template(template)[0].to(raw.device).double()
    fits = [fit_rigid_row(raw[(ids==i)&core].double(),target[(ids==i)&core]) for i in range(len(data['body_names']))]
    return CoordinateTransform(torch.stack([r for r,t in fits]).to(raw),torch.stack([t for r,t in fits]).to(raw),
        ids,keys,data['body_names'],raw,fit_targets=target,fit_core=core,report=dict(grouping='auth_asym_id',fit_atoms=fit_atoms))
