"""Parallel public behavior with real sampler/GMM and an analytic denoiser."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from test_b2_inputs import train_fixture
from test_b2b_randomness import sampler
from test_foundations import fake_protenix
from chain_parallel import train_chain_parallel_2d as trainer
from chain_parallel.manifest import load_manifest
from chain_parallel.parallel_runtime import fit_chains, seeds, validate_partition, preflight
from chain_parallel.parallel_checkpoint import resume_index
from structure_io import read_template


def setup_case(folder, count=1):
    folder.mkdir(parents=True,exist_ok=True)
    old,cache=train_fixture(folder)
    cache.pop('z_bias',None); cache.pop('z_mul',None)
    torch.save(cache,folder/'cache.pt')
    rows=[dict(id=f'part{i}',rank=i,diffusion_data_dir='cache.pt',cif_path='template.cif',
               chain_id_map={'A':chr(65+i*2),'B':chr(66+i*2)}) for i in range(count)]
    manifest=folder/'components.yaml'
    manifest.write_text(yaml.safe_dump(dict(schema_version=1,components=rows)),encoding='utf-8')
    return trainer.build_parser().parse_args(['--component_manifest',str(manifest),
        '--star_data_dir',str(folder/'particles.star'),'--output_trained_model_dir',str(folder/'out/model_'),
        '--device','cpu','--backend','gloo','--boxsize','24','--batch_size','2','--mini_batch_size','1',
        '--epochs','2','--no-learn-gmm']),cache


def install_sampler(monkeypatch,sampler):
    from test_foundations import FakeDenoiser
    def forward(self,x_noisy,pair_z=None,z_trunk=None,**kwargs):
        pair = z_trunk if z_trunk is not None else pair_z
        return self.scale*x_noisy+pair.square().mean()
    monkeypatch.setattr(FakeDenoiser,'forward',forward)
    def sample(configs, training=False, **kwargs):
        return sampler(configs=configs,**configs.sample_diffusion,**kwargs)
    monkeypatch.setattr(trainer,'_sample_diffusion',sample)


def test_parallel_defaults_and_data_seed(tmp_path):
    args,_=setup_case(tmp_path)
    defaults=trainer.build_parser().parse_args(['--component_manifest','m','--star_data_dir','s','--output_trained_model_dir','p'])
    assert defaults.seed==42 and defaults.learn_gmm and defaults.epochs==10
    assert defaults.mini_batch_size==12 and defaults.train_deterministic
    assert (defaults.lr_bias,defaults.lr_atom_weights,defaults.lr_sdevs)==(.01,.01,.005)
    assert seeds(defaults)['data_seed']==42 and seeds(defaults)['rng_mode']=='legacy'
    defaults.data_seed=13
    assert seeds(defaults)['data_seed']==13


@pytest.mark.parametrize('kind',['missing_cache','bad_rank','duplicate_chain','bad_core'])
def test_parallel_preflight_errors(tmp_path,kind):
    args,_=setup_case(tmp_path,2)
    manifest=Path(args.component_manifest)
    data=yaml.safe_load(manifest.read_text())
    if kind=='missing_cache': data['components'][1]['diffusion_data_dir']='absent.pt'
    if kind=='bad_rank': data['components'][1]['rank']=-1
    if kind=='duplicate_chain': data['components'][1]['chain_id_map']={'A':'A','B':'B'}
    if kind=='bad_core':
        args.by_chain=True
        text=(tmp_path/'template.cif').read_text().replace(' CA ', ' CB ')
        (tmp_path/'template.cif').write_text(text)
    manifest.write_text(yaml.safe_dump(data))
    args.check_inputs=True
    with pytest.raises((ValueError,FileNotFoundError)):
        trainer.train(args)
    assert not list((tmp_path/'out').glob('*.pth'))


def test_parallel_by_chain_transform_and_gradient(tmp_path):
    args,_=setup_case(tmp_path)
    target=read_template(tmp_path/'template.cif')[0]
    raw=target.clone(); raw[:3]+=torch.tensor([2.,1.,3.]); raw[3:]+=torch.tensor([-4.,2.,1.])
    transform=fit_chains(tmp_path/'template.cif',raw,'ca')
    raw.requires_grad_(True)
    torch.testing.assert_close(transform(raw),target,atol=2e-6,rtol=0)
    transform(raw).square().sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0


@pytest.mark.parametrize('learn',[False,True])
@pytest.mark.parametrize('width_mode',['legacy','molmap'])
def test_parallel_training_resume_and_frozen_gmm(tmp_path,monkeypatch,sampler,fake_protenix,learn,width_mode):
    install_sampler(monkeypatch,sampler)
    args,initial=setup_case(tmp_path)
    args.by_chain=True
    args.learn_gmm=learn
    args.gmm_sdev_init_mode=width_mode
    args.gmm_molmap_resolution_A=2.0 if width_mode=='molmap' else None
    args.epochs=1
    trainer.train(args)
    prefix=tmp_path/'out/model_'
    index=Path(str(prefix)+'epoch_1.json')
    saved=torch.load(Path(str(prefix)+'part0_rank0_1.pth'),map_location='cpu',weights_only=False)
    assert saved['parallel_resume']['epoch_complete'] and saved['parallel_resume']['global_step']==2
    assert saved['gmm_learning_enabled'] is learn and len(saved['opt_state']['param_groups'])==(3 if learn else 1)
    assert torch.any(saved['z_bias']!=0)
    assert Path(str(prefix)+'merged_1.cif').is_file()
    resumed=copy.deepcopy(args); resumed.resume=str(index); resumed.epochs=2
    resumed.output_trained_model_dir=str(tmp_path/'resumed/model_')
    trainer.train(resumed)
    after=torch.load(tmp_path/'resumed/model_part0_rank0_2.pth',weights_only=False)
    assert after['parallel_resume']['global_step']==4
    full=copy.deepcopy(args); full.epochs=2; full.output_trained_model_dir=str(tmp_path/'continuous/model_')
    trainer.train(full)
    continuous=torch.load(tmp_path/'continuous/model_part0_rank0_2.pth',weights_only=False)
    assert torch.mean((continuous['z_bias']-after['z_bias'])**2)<1e-8
    if not learn:
        assert all(torch.equal(v,after['gmm']['state_dict'][k]) for k,v in saved['gmm']['state_dict'].items())
    assert all(torch.mean((v-after['gmm']['state_dict'][k])**2)<1e-8 for k,v in continuous['gmm']['state_dict'].items())
    assert torch.equal(continuous['parallel_resume']['data_generator_state'],after['parallel_resume']['data_generator_state'])
    data=json.loads(index.read_text()); data['epoch_complete']=False
    bad=tmp_path/'partial.json'; bad.write_text(json.dumps(data)); resumed.resume=str(bad)
    with pytest.raises(ValueError,match='complete'):
        resume_index(resumed,load_manifest(args.component_manifest))


def test_partial_checkpoint_and_warm_start(tmp_path,monkeypatch,sampler,fake_protenix):
    install_sampler(monkeypatch,sampler)
    args,_=setup_case(tmp_path)
    args.max_steps=1
    trainer.train(args)
    index=tmp_path/'out/model_epoch_1.json'
    assert json.loads(index.read_text())['epoch_complete'] is False
    manifest=yaml.safe_load(Path(args.component_manifest).read_text())
    manifest['components'][0]['diffusion_data_dir']='out/model_part0_rank0_1.pth'
    Path(args.component_manifest).write_text(yaml.safe_dump(manifest))
    args.warm_start=True; args.output_trained_model_dir=str(tmp_path/'warm/model_')
    trainer.train(args)
    saved=torch.load(tmp_path/'warm/model_part0_rank0_1.pth',weights_only=False)
    assert saved['parallel_resume']['global_step']==1


def test_default_component_matches_original(tmp_path,monkeypatch,sampler,fake_protenix):
    import importlib.util
    from randomness import seed_legacy
    install_sampler(monkeypatch,sampler)
    path=Path(__file__).parents[1]/'reference/parallel_trainer_before_b7b.py'
    spec=importlib.util.spec_from_file_location('legacy_parallel_component',path)
    legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    monkeypatch.setattr(legacy,'_sample_diffusion',trainer._sample_diffusion)
    args,cache=setup_case(tmp_path); args.learn_gmm=True
    entry=load_manifest(args.component_manifest)[0]
    values=[]
    for cls in (legacy.LocalComponent,trainer.LocalComponent):
        seed_legacy(42)
        component=cls(entry,torch.device('cpu'),True,args)
        sample=component.sample()
        placed=component.place_coordinates(sample[0],False)
        placed.square().sum().backward()
        values.append((sample.detach(),placed.detach(),component.z_bias.grad.clone(),component.gmm.state_dict()))
    for i in range(3): torch.testing.assert_close(values[0][i],values[1][i],atol=0,rtol=0)
    assert all(torch.equal(v,values[1][3][k]) for k,v in values[0][3].items())


@pytest.mark.parametrize('problem',['duplicate_tokens','different_source','mixed','missing_atoms'])
def test_contextual_partition_guards(tmp_path,problem):
    args,_=setup_case(tmp_path,2)
    reports=[preflight(args,e)[2] for e in load_manifest(args.component_manifest)]
    for i,r in enumerate(reports):
        r['contextual']=dict(source_cache_sha256='same',source_n_token=4,source_n_atom=12,
            source_token_indices=[i*2,i*2+1],source_atom_indices=list(range(i*6,(i+1)*6)))
    validate_partition(reports)
    if problem=='duplicate_tokens': reports[1]['contextual']['source_token_indices']=[0,1]
    if problem=='different_source': reports[1]['contextual']['source_cache_sha256']='other'
    if problem=='mixed': reports[1]['contextual']=None
    if problem=='missing_atoms': reports[1]['contextual']['source_atom_indices']=[]
    with pytest.raises(ValueError): validate_partition(reports)


def test_resume_rejects_missing_rank_and_science_change(tmp_path,monkeypatch,sampler,fake_protenix):
    install_sampler(monkeypatch,sampler)
    args,_=setup_case(tmp_path); args.epochs=1
    trainer.train(args)
    args.resume=str(tmp_path/'out/model_epoch_1.json'); args.epochs=2
    args.lr_bias=.2; args._explicit_options=['lr_bias']
    with pytest.raises(ValueError,match='lr_bias'):
        resume_index(args,load_manifest(args.component_manifest))
    args._explicit_options=[]
    checkpoint=tmp_path/'out/model_part0_rank0_1.pth'
    checkpoint.rename(checkpoint.with_suffix('.held'))
    with pytest.raises(ValueError,match='Incomplete'):
        resume_index(args,load_manifest(args.component_manifest))


@pytest.mark.parametrize('by_chain,fixed',[(False,True),(True,True),(False,False)])
def test_parallel_checkpoint_public_export(tmp_path,monkeypatch,sampler,fake_protenix,by_chain,fixed):
    from types import SimpleNamespace
    import get_pdb
    install_sampler(monkeypatch,sampler)
    monkeypatch.setattr(get_pdb,'_sample_diffusion',trainer._sample_diffusion)
    args,_=setup_case(tmp_path); args.epochs=1; args.by_chain=by_chain; args.train_deterministic=fixed
    args.rng_mode='isolated'
    trainer.train(args)
    get_pdb.main(SimpleNamespace(device='cpu',diffusion_data_dir=str(tmp_path/'out/model_part0_rank0_1.pth'),
        out_dir=str(tmp_path/'export'),cif_path=str(tmp_path/'template.cif'),pdbid='part',output_format='cif'))
    trained=read_template(tmp_path/'out/model_part0_rank0_1.cif')[0]
    exported=read_template(tmp_path/('export/part_block_prediction.cif' if by_chain else 'export/part_refined_prediction.cif'))[0]
    assert torch.mean((trained-exported)**2)<1e-8


def test_epoch_index_atomic_and_no_overwrite(tmp_path):
    from chain_parallel.parallel_checkpoint import write_json_new
    path=tmp_path/'epoch.json'
    write_json_new(path,{'epoch_complete':True})
    with pytest.raises(FileExistsError): write_json_new(path,{'epoch_complete':False})
    assert json.loads(path.read_text())=={'epoch_complete':True}
    with pytest.raises(ValueError): write_json_new(tmp_path/'invalid.json',{'value':float('nan')})
    assert not (tmp_path/'invalid.json').exists() and not list(tmp_path.glob('*.tmp'))


def test_contextual_success_and_atom_order_guard(tmp_path):
    args,cache=setup_case(tmp_path,2)
    manifest=yaml.safe_load(Path(args.component_manifest).read_text())
    for rank,row in enumerate(manifest['components']):
        raw=copy.deepcopy(cache); features=raw['input_feature_dict']
        features.update(asym_id=torch.tensor([rank*2,rank*2+1]),ref_pos=torch.zeros(6,3),
            ref_charge=torch.zeros(6),ref_mask=torch.ones(6),ref_space_uid=torch.zeros(6),
            ref_element=torch.nn.functional.one_hot(torch.full((6,),5),128).float(),
            ref_atom_name_chars=torch.nn.functional.one_hot(torch.tensor([[35,33,0,0]]*6),64).float(),
            d_lm=torch.zeros(1),v_lm=torch.zeros(1),pad_info={})
        raw['p_lm']=torch.zeros(1); raw['c_l']=torch.zeros(1)
        raw['contextual_split_metadata']=dict(component_n_token=2,component_n_atom=6,source_n_token=4,
            source_n_atom=12,source_cache_sha256='fixture',source_token_indices=[rank*2,rank*2+1])
        name=f'contextual{rank}.pth'; torch.save(raw,tmp_path/name); row['diffusion_data_dir']=name
    Path(args.component_manifest).write_text(yaml.safe_dump(manifest))
    entries=load_manifest(args.component_manifest)
    reports=[preflight(args,e)[2] for e in entries]; validate_partition(reports)
    assert 'element order checked' in reports[0]['topology_check']
    raw=torch.load(entries[0].diffusion_data_dir,weights_only=False)
    raw['input_feature_dict']['ref_atom_name_chars'][0]=torch.nn.functional.one_hot(torch.tensor([35,34,0,0]),64).float()
    torch.save(raw,entries[0].diffusion_data_dir)
    with pytest.raises(ValueError,match='ORDER mismatch'): preflight(args,entries[0])
