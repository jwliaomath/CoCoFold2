"""B8 launcher and output-audit tests; no model construction or training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_b2_inputs import train_fixture
from structure_io import read_template
from training_output import export_training_structure
from coordinate_transform import template_atom_keys
from chain_parallel.parallel_checkpoint import merge_cifs
from chain_parallel.train_chain_parallel_2d import build_parser

ROOT=Path(__file__).resolve().parents[2]
EXAMPLE=ROOT/'examples/6zbh_parallel'


def module(name):
    spec=importlib.util.spec_from_file_location('b8_'+name,EXAMPLE/(name+'.py'))
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def completed(tmp_path):
    """Synthetic saved state, serialized with the production CIF/merge helpers."""
    train_fixture(tmp_path)
    checker=module('check_run')
    xyz=read_template(tmp_path/'template.cif')[0]
    keys=template_atom_keys(tmp_path/'template.cif')
    out=tmp_path/'result'; out.mkdir()
    groups=[]
    for rank in range(2):
        mapping={'A':chr(65+2*rank),'B':chr(66+2*rank)}
        groups.append(dict(rank=rank,id=f'part{rank}',chain_id_map=mapping,
                           global_atom_keys=[[mapping[k[0]],*k[1:]] for k in keys]))
    science={'batch_size':2}
    for epoch in (1,2):
        components=[]
        for rank in range(2):
            base=out/f'model_part{rank}_rank{rank}_{epoch}'
            export_training_structure(tmp_path/'template.cif',base,xyz.numpy(),'cif')
            cache=dict(training_progress=dict(epoch=epoch,epoch_complete=True,global_step=2*epoch),
                parallel_resume=dict(epoch_complete=True),z_bias=torch.ones(2,2),
                gmm={'state_dict':{'weights':torch.ones(6)}},
                opt_state={'state':{0:dict(step=2*epoch,exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2))}},
                rotation=torch.eye(3),translation=torch.zeros(3),pred_dict={'coordinate':xyz[None]})
            torch.save(cache,base.with_suffix('.pth'))
            components.append(dict(rank=rank,id=f'part{rank}',path=base.name+'.pth',cif=base.name+'.cif',
                                   sha256=checker.digest(base.with_suffix('.pth'))))
        merged=out/f'model_merged_{epoch}.cif'
        merge_cifs([out/r['cif'] for r in components],groups,merged)
        dump(out/f'model_epoch_{epoch}.json',dict(format='cocofold2-parallel-epoch-v1',epoch_complete=True,
            next_epoch=epoch,global_step=2*epoch,components=components,merged_cif=merged.name,
            group_identity=groups,science_args=science))
    for rank in range(2):
        record=out/'records'/f'rank{rank}'
        dump(record/'run_summary.json',dict(status='success',error=None,elapsed_seconds=12))
        dump(record/'resolved_config.json',dict(stages=[dict(stage='parallel_inputs',values=[{'n_particles':3}]*2),
                                                      dict(stage='parallel_science',values=science)]))
        dump(record/'requested_config.json',dict(arguments=dict(epochs=2,backend='nccl',submission_script='source.slurm')))
        dump(record/'command.json',dict(argv=['train.py']))
        dump(record/'provenance.json',dict(source_sha256={'train.py':'fixture'}))
        dump(record/'run_identity.json',dict(run_id=f'fixture-{rank}'))
        dump(record/'environment.json',dict(process={'WORLD_SIZE':'2'},devices=[{'name':'synthetic device'}]))
        script=record/'submitted_script.sh'; script.write_text('#!/bin/bash\n',encoding='utf-8')
        dump(record/'submission_script.json',dict(sha256=checker.digest(script)))
        rows=[]; detailed=[]
        for step in range(4):
            epoch,batch=divmod(step,2)
            rows.append(dict(event='train_step',global_step=step+1,epoch=epoch,batch=batch,
                             particle_indices=[0,2] if batch==0 else [1],total_loss=.2,elapsed_seconds=1+rank))
            detailed.append(dict(epoch=epoch,batch=batch,batch_time_seconds=1+rank,
                                  peak_allocated_memory_mb=100+rank,peak_reserved_memory_mb=200+rank))
        (record/('metrics.jsonl' if rank==0 else f'metrics_rank{rank}.jsonl')).write_text('\n'.join(map(json.dumps,rows)))
        (out/f'chain_parallel_2d_metrics_rank{rank}.jsonl').write_text('\n'.join(map(json.dumps,detailed)))
    return out


def test_b8_completed_outputs_and_performance(completed):
    checker=module('check_run'); result=checker.check_run(completed,2)
    assert result['passed'],result
    assert result['performance']['distributed_steps']['sum_max_rank_seconds']==8
    assert result['performance']['rank1']['peak_reserved_memory_MiB']==201
    assert 'pending author review' in result['structure_quality']
    checker.write_report(completed,result)
    assert json.loads((completed/'summary.json').read_text())['passed']


@pytest.mark.parametrize('problem',['missing_checkpoint','bad_hash','partial_epoch','wrong_progress','nonfinite_loss',
                                    'missing_particle','different_rank_order','missing_script','missing_gpu_memory'])
def test_b8_audit_rejects_incomplete_or_corrupt_run(completed,problem):
    if problem=='missing_checkpoint':
        (completed/'model_part1_rank1_1.pth').unlink()
    elif problem=='missing_script':
        (completed/'records/rank1/submitted_script.sh').unlink()
    elif problem in ('bad_hash','partial_epoch','wrong_progress'):
        path=completed/'model_epoch_2.json'; data=json.loads(path.read_text())
        if problem=='bad_hash': data['components'][1]['sha256']='wrong'
        if problem=='partial_epoch': data['epoch_complete']=False
        if problem=='wrong_progress': data['global_step']=3
        dump(path,data)
    elif problem=='missing_gpu_memory':
        path=completed/'chain_parallel_2d_metrics_rank1.jsonl'
        rows=list(map(json.loads,path.read_text().splitlines())); rows[0]['peak_allocated_memory_mb']=None
        path.write_text('\n'.join(map(json.dumps,rows)))
    else:
        path=completed/'records/rank1/metrics_rank1.jsonl'
        rows=list(map(json.loads,path.read_text().splitlines()))
        if problem=='nonfinite_loss': rows[0]['total_loss']=float('nan')
        if problem=='missing_particle': rows[0]['particle_indices']=[0]
        if problem=='different_rank_order': rows[0]['particle_indices']=[2,0]
        path.write_text('\n'.join(map(json.dumps,rows)))
    checker=module('check_run'); result=checker.check_run(completed,2)
    assert not result['passed']
    checker.write_report(completed,result)


def test_b8_launcher_uses_full_star_and_validated_science(tmp_path):
    runner=module('run_case')
    args=SimpleNamespace(manifest=tmp_path/'manifest.yaml',star=tmp_path/'366.star',mrc_dir=tmp_path,
                         output=tmp_path/'out',epochs=10,submission_script=None)
    cmd=runner.command(args)
    parsed=build_parser().parse_args(cmd[6:])
    assert cmd[5].endswith('train_chain_parallel_2d.py')
    assert parsed.star_data_dir==str(args.star.resolve()) and parsed.max_steps is None
    assert (parsed.epochs,parsed.batch_size,parsed.mini_batch_size)==(10,32,16)
    assert parsed.learn_gmm and parsed.transR and parsed.train_deterministic
    assert parsed.projection_frame=='fixed'
    assert tuple(parsed.projection_origin)==(154.512,154.512,154.512)
    assert not parsed.gmm_checkpoint_peak2d and not parsed.by_chain and not parsed.update_affine_mat
    assert (parsed.lr_bias,parsed.lr_atom_weights,parsed.lr_sdevs)==(.01,.01,.005)


@pytest.mark.parametrize('stage',['run','check'])
def test_b8_cli_help(stage):
    result=subprocess.run([sys.executable,str(EXAMPLE/'run_case.py'),stage,'--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'usage:' in result.stdout


def test_b8_missing_run_is_failure_without_training(tmp_path):
    result=subprocess.run([sys.executable,str(EXAMPLE/'run_case.py'),'check','--output',str(tmp_path/'missing')],
                          capture_output=True,text=True)
    assert result.returncode==1
    assert not json.loads((tmp_path/'missing/summary.json').read_text())['passed']


def test_b8_launcher_records_training_failure_and_exact_preflight(tmp_path,monkeypatch):
    runner=module('run_case')
    manifest=tmp_path/'m.yaml'; manifest.write_text('fixture')
    star=tmp_path/'366.star'; star.write_text('fixture')
    args=SimpleNamespace(manifest=manifest,star=star,mrc_dir=tmp_path,resource_root=tmp_path,
                         output=tmp_path/'out',epochs=10,submission_script=None,dry_run=False)
    seen=[]
    def failed_child(cmd,**kwargs):
        seen.append(cmd)
        assert kwargs['env']['PROTENIX_ROOT_DIR']==str(tmp_path)
        if len(seen)==1:
            assert cmd[1].endswith('train_chain_parallel_2d.py')
            assert build_parser().parse_args(cmd[2:]).check_inputs
        return SimpleNamespace(returncode=0 if len(seen)==1 else 1)
    monkeypatch.setattr(runner.subprocess,'run',failed_child)
    with pytest.raises(RuntimeError,match='training exited 1'):
        runner.run(args)
    assert len(seen)==2 and seen[1]==runner.command(args)
    assert json.loads((args.output/'launcher.json').read_text())['status']=='failed'
    assert not json.loads((args.output/'summary.json').read_text())['passed']


def test_b8_dry_run_creates_no_outputs(tmp_path):
    args=[sys.executable,str(EXAMPLE/'run_case.py'),'run','--manifest',str(tmp_path/'missing.yaml'),
          '--star',str(tmp_path/'366.star'),'--mrc-dir',str(tmp_path),'--resource-root',str(tmp_path),
          '--output',str(tmp_path/'out'),'--dry-run']
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['particle_subset']=='entire supplied STAR; no truncation'
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('problem',['nonfinite_latent','coordinate_mismatch'])
def test_b8_final_state_numerics(completed,problem):
    checker=module('check_run')
    path=completed/'model_part1_rank1_2.pth'
    cache=torch.load(path,weights_only=False)
    if problem=='nonfinite_latent': cache['z_bias'][0,0]=float('nan')
    else: cache['translation'][0]=1
    torch.save(cache,path)
    index=json.loads((completed/'model_epoch_2.json').read_text())
    index['components'][1]['sha256']=checker.digest(path)
    dump(completed/'model_epoch_2.json',index)
    result=checker.check_run(completed,2)
    assert not result['passed'] and result['error'] is None
    checker.write_report(completed,result)


@pytest.mark.parametrize('status',['running','failed','success'])
def test_b8_prefix_preserves_original_target_status_and_summary(completed,status):
    for rank in range(2):
        record=completed/'records'/f'rank{rank}'
        path=record/'requested_config.json'; data=json.loads(path.read_text())
        data['arguments']['epochs']=10; dump(path,data)
        summary=dict(status=status,error='interrupted after accepted epochs' if status=='failed' else None)
        if status!='running': summary['elapsed_seconds']=12
        dump(record/'run_summary.json',summary)
    dump(completed/'launcher.json',dict(status=status,epochs=10,error='later interruption' if status=='failed' else None))
    dump(completed/'summary.json',{'original_full_run_summary':True})
    original=(completed/'summary.json').read_bytes()
    checker=module('check_run')
    result=checker.check_run(completed,1,completed_prefix=True,reason='Author accepts completed prefix')
    assert result['passed'],result
    assert result['acceptance']['original_target_epochs']==10
    assert result['acceptance']['requested_epochs']==1
    assert not result['acceptance']['whole_job_completion_claimed']
    assert result['acceptance']['observed_rank_records'][1]['status']==status
    assert result['performance']['distributed_steps']['count']==2
    checker.write_report(completed,result)
    assert (completed/'acceptance_epoch_1.json').is_file()
    assert (completed/'summary.json').read_bytes()==original
    assert not checker.check_run(completed,1)['passed']


def test_b8_prefix_does_not_parse_later_partial_jsonl(completed):
    for rank in range(2):
        paths=[completed/'records'/f'rank{rank}'/('metrics.jsonl' if rank==0 else f'metrics_rank{rank}.jsonl'),
               completed/f'chain_parallel_2d_metrics_rank{rank}.jsonl']
        for path in paths:
            rows=path.read_text().splitlines()
            path.write_text('\n'.join(rows[:2])+'\n{"later_incomplete":')
    checker=module('check_run')
    assert checker.check_run(completed,1,True,'author scope change')['passed']
    assert not checker.check_run(completed,2,True,'author scope change')['passed']


@pytest.mark.parametrize('problem',['incomplete_epoch','corrupt_checkpoint','missing_particle','bad_selected_json'])
def test_b8_prefix_still_rejects_invalid_selected_epochs(completed,problem):
    if problem=='incomplete_epoch':
        path=completed/'model_epoch_1.json'; data=json.loads(path.read_text()); data['epoch_complete']=False; dump(path,data)
    elif problem=='corrupt_checkpoint':
        (completed/'model_part0_rank0_1.pth').write_bytes(b'corrupt')
    else:
        path=completed/'records/rank1/metrics_rank1.jsonl'
        if problem=='bad_selected_json': path.write_text('{bad selected line')
        else:
            rows=list(map(json.loads,path.read_text().splitlines())); rows[0]['particle_indices']=[0]
            path.write_text('\n'.join(map(json.dumps,rows)))
    result=module('check_run').check_run(completed,1,True,'author scope change')
    assert not result['passed']


def test_b8_prefix_requires_reason(completed):
    assert not module('check_run').check_run(completed,1,True)['passed']
    result=subprocess.run([sys.executable,str(EXAMPLE/'run_case.py'),'check','--output',str(completed),
                           '--epochs','1','--completed-prefix'],capture_output=True,text=True)
    assert result.returncode==2 and '--reason' in result.stderr
