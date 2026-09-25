"""Complete-epoch parallel checkpoints; no partial-rank or in-epoch resume."""
import copy
from dataclasses import replace
import json
import os
import tempfile
from pathlib import Path

import gemmi
import torch

from checkpoint_io import atomic_save_checkpoint
from checkpoint_sampling import preserve_sampling, sampling_snapshot
from randomness import capture_rng_state, restore_rng_state
from run_recording import current_record, json_value
from training_output import export_training_structure
from training_restart import SCIENCE_ARGS, runtime_signature
from chain_parallel.parallel_runtime import gather, identity, phase

SCIENCE = (*SCIENCE_ARGS, 'by_chain', 'fit_atoms')


def write_json_new(path, value):
    path = Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.'+path.name,suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name)
            json.dump(value,handle,indent=2,allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def resume_index(args, entries):
    if not args.resume:
        return entries, None
    source = Path(args.resume).resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    if data.get('format') != 'cocofold2-parallel-epoch-v1' or not data.get('epoch_complete'):
        raise ValueError('Resume requires a complete parallel epoch index; use warm-start for old/partial checkpoints')
    rows = data['components']
    if len(rows) != len(entries) or sorted((r['rank'],r['id']) for r in rows) != sorted((e.rank,e.component_id) for e in entries):
        raise ValueError('Resume cannot change world size or component grouping')
    updated=[]
    for entry in entries:
        row=next(r for r in rows if r['rank']==entry.rank)
        path=source.parent/row['path']
        if not path.is_file() or identity(path)['sha256'] != row['sha256']:
            raise ValueError(f'Incomplete/corrupt epoch component: {path}')
        updated.append(replace(entry,diffusion_data_dir=path))
    explicit=set(getattr(args,'_explicit_options',[]))
    for key, default in (('projection_frame', 'legacy'), ('projection_origin', (0., 0., 0.))):
        if key not in data['science_args']:
            value = getattr(args, key, default)
            requested = tuple(value) if key == 'projection_origin' and value is not None else value
            if key in explicit and requested != default:
                raise ValueError('Old parallel checkpoint has legacy projection; use --warm-start to change ' + key)
            setattr(args, key, default)
    for key,value in data['science_args'].items():
        if key in explicit and json_value(getattr(args,key)) != value:
            raise ValueError(f'Resume cannot change --{key}; use --warm-start for a new experiment')
        setattr(args,key,value)
    if args.epochs <= data['next_epoch']:
        raise ValueError('--epochs must exceed the completed epoch count')
    if args.max_steps is not None and args.max_steps <= data['global_step']:
        raise ValueError('--max_steps must exceed the resumed global step')
    return tuple(updated), data


def group_identity(reports):
    return [dict(rank=r['rank'],id=r['component_id'],reference_sha256=r['reference']['sha256'],
                 chain_id_map=r['chain_id_map'],global_atom_keys=r['global_atom_keys']) for r in reports]


def validate_resume(args, data, reports, raw, device):
    if data is None:
        return
    if group_identity(reports) != data['group_identity'] or reports[0]['star']['sha256'] != data['star_sha256']:
        raise ValueError('Resume reference, chain mapping or STAR changed; use warm-start')
    if reports[0]['stacks_identity'] != data['stacks_identity']:
        raise ValueError('Resume MRCS paths/size/mtime changed; use warm-start')
    state=raw.get('parallel_resume')
    if not state or state['global_step'] != data['global_step'] or state['next_epoch'] != data['next_epoch']:
        raise ValueError('Rank checkpoint and epoch index disagree')
    if not state['epoch_complete']:
        raise ValueError('Mid-epoch checkpoint cannot resume')
    if state['runtime'] != runtime_signature(device):
        raise ValueError('Resume runtime differs; use warm-start')


def restore_resume(raw, component, optimizer, loader):
    state=raw['parallel_resume']
    optimizer.load_state_dict(raw['opt_state'])
    loader.generator.set_state(state['data_generator_state'].cpu())
    component.configs.diffusion_rng_stream=copy.deepcopy(state['diffusion_rng_stream'])
    restore_rng_state(state['rng'])
    return state['next_epoch'],state['global_step']


def merge_cifs(paths, reports, output):
    """Merge placed components using explicit local-to-global author chain IDs."""
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    structure=gemmi.Structure(); structure.name='CoCoFold2_parallel'
    model=gemmi.Model('1'); seen=set()
    for path,report in zip(paths,reports):
        part=gemmi.read_structure(str(path))
        for chain in part[0]:
            name=report['chain_id_map'][chain.name]
            if name in seen: raise ValueError('Duplicate merged chain identity')
            seen.add(name)
            clone=chain.clone(); clone.name=name
            for residue in clone: residue.subchain=name
            model.add_chain(clone)
    structure.add_model(model); structure.setup_entities(); structure.assign_label_seq_id()
    structure.make_mmcif_document().write_file(str(output))
    # Check atom identity/order after serialization, not just the atom count.
    from coordinate_transform import template_atom_keys
    if template_atom_keys(output) != [k for r in reports for k in r['global_atom_keys']]:
        raise ValueError('Merged CIF changed global atom identities/order')


def save_epoch(args, component, optimizer, loader, reports, epoch, step, complete, prefix):
    record=current_record()
    def save_local():
        with preserve_sampling(component.configs), torch.no_grad():
            snapshot=sampling_snapshot(component.configs,component.device)
            coordinates=component.sample()
            placed=component.place_coordinates(coordinates[0],False)
        component.pred_dict['coordinate']=coordinates.detach()
        state={key:getattr(component,key) for key in ('input_feature_dict','s_inputs','s_trunk','z_trunk','pair_z',
            'p_lm','c_l','noise_schedule','inplace_safe','configs','enable_efficient_fusion','pred_dict')}
        state.update(checkpoint_schema=dict(name='cocofold2_refinement',version=1),
            model_state=component.diffusion_module.state_dict(), opt_state=optimizer.state_dict(),
            N_sample=component.n_sample, z_bias=component.z_bias,z_mul=None,s_inputs_bias=None,s_bias=None,
            atom_weights=component.atom_weights,sdevs=component.gmm.sdevs if component.gmm.kernel=='legacy' else None,
            gmm=component.gmm.export_checkpoint(),gmm_kernel=component.gmm.kernel,gmm_learning_enabled=args.learn_gmm,
            projection_frame=args.projection_frame,projection_origin=tuple(args.projection_origin),
            rotation=component.rotation,translation=component.translation,
            component_id=component.entry.component_id,contextual_split_metadata=component.contextual_split_metadata,
            refinement_sampler='global',refinement_seed_settings=component.seed_settings,export_sampling=snapshot,
            training_progress=dict(global_step=step,epoch=epoch+1,epoch_complete=complete),
            parallel_resume=dict(version=1,epoch_complete=complete,next_epoch=epoch+1 if complete else epoch,
                global_step=step,rng=capture_rng_state(),diffusion_rng_stream=copy.deepcopy(getattr(component.configs,'diffusion_rng_stream',None)),
                data_generator_state=loader.generator.get_state(),runtime=runtime_signature(component.device)))
        if component.transform is not None:
            state['coordinate_transform']=component.transform.export_checkpoint(coordinates[0])
        path=Path(f'{prefix}{epoch+1}.pth')
        # Always produce CIF for lossless assembly, plus requested PDB when needed.
        formats='both' if args.output_format in ('pdb','both') else 'cif'
        files=export_training_structure(component.entry.cif_path,f'{prefix}{epoch+1}',placed.cpu().numpy(),formats)
        atomic_save_checkpoint(state,path)
        record.artifact(path,'checkpoint')
        for item in files: record.artifact(item,'structure')
        return dict(rank=component.entry.rank,id=component.entry.component_id,path=str(path.resolve()),
                    sha256=identity(path)['sha256'],cif=str(Path(f'{prefix}{epoch+1}.cif').resolve()))
    local=phase('save component',save_local)
    rows=gather(local)
    def publish():
        if component.entry.rank != 0: return
        parent=Path(prefix).parent
        merged=Path(f'{args.output_trained_model_dir}merged_{epoch+1}.cif')
        merge_cifs([r['cif'] for r in rows],reports,merged)
        for row in rows:
            row['path']=str(Path(row['path']).relative_to(parent))
            row['cif']=str(Path(row['cif']).relative_to(parent))
        index=Path(f'{args.output_trained_model_dir}epoch_{epoch+1}.json')
        write_json_new(index,dict(format='cocofold2-parallel-epoch-v1',epoch_complete=complete,
            next_epoch=epoch+1 if complete else epoch,global_step=step,components=rows,
            merged_cif=merged.name,group_identity=group_identity(reports),star_sha256=reports[0]['star']['sha256'],
            stacks_identity=reports[0]['stacks_identity'],
            science_args={k:json_value(getattr(args,k)) for k in SCIENCE if hasattr(args,k)}))
        record.artifact(index,'parallel_epoch_index'); record.artifact(merged,'merged_structure')
    phase('publish epoch',publish)
