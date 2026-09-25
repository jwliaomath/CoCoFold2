"""Two real Gloo ranks with analytic denoiser, real sampler/renderer/optimizer."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,folder):
    from test_b7b_parallel import setup_case, install_sampler
    from test_b2b_randomness import sampler as sampler_fixture
    from test_foundations import fake_protenix
    from chain_parallel import train_chain_parallel_2d as trainer
    from chain_parallel.parallel_runtime import phase, fit_chains
    from chain_parallel.distributed_gmm import distributed_project_gaussians
    from gmm import GaussianProjector
    folder=Path(folder)
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
    patch=pytest.MonkeyPatch()
    fake_protenix.__wrapped__(patch)
    sampler=sampler_fixture.__wrapped__(patch)
    install_sampler(patch,sampler)
    def group(label):
        dist.init_process_group('gloo',init_method=(folder/f'rendezvous_{label}').as_uri(),
            rank=rank,world_size=2,timeout=timedelta(seconds=30))
    group('projection')
    try:
        # Four distinct per-chain translations, two chains on each rank.
        template=folder/'input/template.cif'
        raw=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[3.,0.,0.],[4.,0.,0.],[3.,1.,0.]])
        raw[:3]+=torch.tensor([2.+rank,1.,-1.]); raw[3:]+=torch.tensor([-2.,3.+rank,2.])
        transform=fit_chains(template,raw,'ca')
        local=raw.clone().requires_grad_(True)
        placed=transform(local)+rank*torch.tensor([4.,2.,0.])
        coords=[torch.zeros_like(placed) for _ in range(2)]; dist.all_gather(coords,placed.detach())
        fullcoords=torch.cat(coords).requires_grad_(True)
        weights=torch.full((6,),6.)
        gmm=GaussianProjector(weights); full=GaussianProjector(torch.cat([weights,weights]))
        gmm.requires_grad_(False); full.requires_grad_(False)
        rot=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]); trans=torch.zeros(1,2); center=torch.tensor([12.,12.])
        params=dict(rotations=rot,translations=trans,density_center=center,resolution=3.,box_size=24,apix=1.)
        image=distributed_project_gaussians(gmm,placed,**params)
        expected=full(atoms_coord=fullcoords[None],rotation=rot,trans=trans,density_center=center,
            resolution=3.,box_size=24,apix=1.,cutoff_range=5,sigma_factor=1/(torch.pi*2**.5))
        (image.square().mean()/2).backward(retain_graph=True)
        expected.square().mean().backward()
        torch.testing.assert_close(image,expected,atol=2e-5,rtol=2e-5)
        # Known transforms are translations; local/raw and placed gradients agree.
        torch.testing.assert_close(local.grad,fullcoords.grad[rank*6:(rank+1)*6],atol=2e-5,rtol=2e-5)
        local.grad.zero_(); fullcoords.grad.zero_()
        fixed_origin=(12., 12., 0.)
        fixed_image=distributed_project_gaussians(
            gmm, placed, **params, projection_frame='fixed', projection_origin=fixed_origin,
        )
        fixed_expected=full(
            atoms_coord=fullcoords[None], rotation=rot, trans=trans, density_center=center,
            resolution=3., box_size=24, apix=1., cutoff_range=5,
            sigma_factor=1/(torch.pi*2**.5), projection_frame='fixed',
            projection_origin=fixed_origin,
        )
        (fixed_image.square().mean()/2).backward()
        fixed_expected.square().mean().backward()
        torch.testing.assert_close(fixed_image,fixed_expected,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(local.grad,fullcoords.grad[rank*6:(rank+1)*6],atol=2e-5,rtol=2e-5)
        try:
            phase('injected rank fault',lambda: (_ for _ in ()).throw(ValueError('test fault')) if rank==1 else None)
        except RuntimeError as exc:
            assert 'rank 1' in str(exc)
        else: raise AssertionError('Fault was not propagated')
    finally:
        dist.destroy_process_group()
    base=trainer.build_parser().parse_args(['--component_manifest',str(folder/'input/components.yaml'),
        '--star_data_dir',str(folder/'input/particles.star'),'--output_trained_model_dir',str(folder/'first/model_'),
        '--device','cpu','--backend','gloo','--boxsize','24','--batch_size','2','--mini_batch_size','1',
        '--projection-frame','legacy',
        '--epochs','1','--no-learn-gmm','--by-chain','--record-dir',str(folder/'first/records')])
    for label in ('first','resumed','continuous','fixed','fault'):
        args=copy.deepcopy(base)
        args.output_trained_model_dir=str(folder/label/'model_')
        args.record_dir=str(folder/label/'records')
        if label=='resumed':
            args.resume=str(folder/'first/model_epoch_1.json'); args.epochs=2
        if label=='continuous': args.epochs=2
        if label=='fixed':
            args.projection_frame='fixed'
            args.projection_origin=(12.,12.,0.)
        if label=='fault':
            original=trainer._sample_diffusion
            counter=[0]
            def fail_after_initial(*a,**kw):
                counter[0]+=1
                if rank==1 and counter[0]>1: raise RuntimeError('injected decoder failure')
                return original(*a,**kw)
            trainer._sample_diffusion=fail_after_initial
        group(label)
        try:
            trainer.train(args)
        except RuntimeError as exc:
            if label!='fault' or 'injected decoder failure' not in str(exc): raise
        else:
            if label=='fault': raise AssertionError('Fault run unexpectedly passed')
    resumed=torch.load(folder/f'resumed/model_part{rank}_rank{rank}_2.pth',weights_only=False)
    full=torch.load(folder/f'continuous/model_part{rank}_rank{rank}_2.pth',weights_only=False)
    mse=float((resumed['z_bias']-full['z_bias']).square().mean())
    assert mse<1e-8
    assert resumed['parallel_resume']['global_step']==4
    assert torch.equal(resumed['parallel_resume']['data_generator_state'],full['parallel_resume']['data_generator_state'])
    fixed=torch.load(folder/f'fixed/model_part{rank}_rank{rank}_1.pth',weights_only=False)
    assert fixed['projection_frame']=='fixed' and fixed['projection_origin']==(12.,12.,0.)
    assert not list((folder/'fault').glob('*epoch*.json'))
    (folder/f'rank{rank}.json').write_text(json.dumps(dict(
        passed=True, resume_latent_mse=mse, projection_gradient=True,
        fixed_frame_projection_gradient=True, fixed_frame_training=True,
        fault_propagation=True,
    )),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    from test_b7b_parallel import setup_case
    setup_case(args.output/'input',2)
    mp.spawn(worker,args=(str(args.output),),nprocs=2,join=True)
    rows=[json.loads((args.output/f'rank{i}.json').read_text()) for i in range(2)]
    # Both ranks must report identical particle order for each optimizer step.
    metrics=[]
    for rank in range(2):
        path=args.output/f'continuous/records/rank{rank}'/('metrics.jsonl' if rank==0 else f'metrics_rank{rank}.jsonl')
        metrics.append([r['particle_indices'] for r in map(json.loads,path.read_text().splitlines()) if r['event']=='train_step'])
    assert metrics[0]==metrics[1] and len(metrics[0])==4
    result=dict(passed=True,world_size=2,real_protenix=False,real_weights=False,backend='gloo',ranks=rows,particle_order_equal=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
