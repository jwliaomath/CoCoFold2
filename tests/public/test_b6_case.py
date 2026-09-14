"""B6 simulation, placement and case checks using real CPU I/O and small tensors."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mrcfile
import numpy as np
import pytest
import torch

import simulate_particles as sim
from benchmark_case import validate_placement, canonical_placement, select_particles, validate_run
from structure_io import read_template, write_coordinates
from test_foundations import write_fixture
from test_b2b_randomness import sampler
from test_epoch_restart import fixture, install

ROOT = Path(__file__).resolve().parents[2]


def tiny_map(path, n=24):
    z,y,x = np.meshgrid(*(np.arange(n)-n//2 for _ in range(3)), indexing='ij')
    value = np.exp(-((x-3)**2+(y+2)**2+(z-1)**2)/6.) + .6*np.exp(-((x+3)**2+y**2+(z+2)**2)/5.)
    with mrcfile.new(path) as handle:
        handle.set_data(value.astype(np.float32)); handle.voxel_size = 1.
        handle.header.origin = (-n/2,)*3
    return path


def options(tmp_path, name='generated'):
    source = tmp_path/'source.mrc'
    if not source.exists(): tiny_map(source)
    return sim.build_parser().parse_args(['--map', str(source), '--out-dir', str(tmp_path/name), '--n-particles','12'])


def test_b6_old_projection_and_ctf_preserved(tmp_path):
    spec = importlib.util.spec_from_file_location('legacy_sim', ROOT/'tests/reference/simulation_legacy.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    source = tiny_map(tmp_path/'map.mrc')
    volume, _ = sim.load_map(source, 1.)
    f = sim.centered_fft(volume).astype(np.complex64)
    for r in sim.Rotation.random(5, random_state=42).as_matrix():
        np.testing.assert_array_equal(sim.project(f,r), old.project(f,r))
    c = [24,1,11000,10500,23,300,2.7,.1,19]
    np.testing.assert_array_equal(sim.ctf_grid(c),old.ctf_grid(c))
    # Independent public CTF calculation checks the unit conversion, not just
    # equivalence to the historical NumPy implementation.
    from ctf import compute_ctf
    axis = np.arange(24)/24-.5
    freqs = torch.tensor(np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2), dtype=torch.float64)
    value = compute_ctf(freqs,11000,10500,np.deg2rad(23),300000,2.7e7,.1,np.deg2rad(19))
    np.testing.assert_allclose(value.numpy().reshape(24,24),sim.ctf_grid(c),atol=1e-7,rtol=1e-6)


def test_b6_simulation_roundtrip_reproducibility_and_move(tmp_path):
    args = options(tmp_path)
    report = sim.simulate(args)
    assert report['status'] == 'completed' and report['validation']['checked_particles'] == 12
    assert .9 < report['realized_snr'] < 1.1
    second = options(tmp_path, 'second'); sim.simulate(second)
    for name in ('particles.star','particles.mrcs','clean_ctf.mrcs'):
        assert (args.out_dir/name).read_bytes() == (second.out_dir/name).read_bytes()
    third = options(tmp_path, 'third'); third.seed=7; sim.simulate(third)
    assert sim.sha256(args.out_dir/'particles.mrcs') != sim.sha256(third.out_dir/'particles.mrcs')
    moved = tmp_path/'moved folder'; shutil.move(str(args.out_dir),moved)
    assert sim.validate_particles(moved)['passed']
    subset = tmp_path/'subset.star'; select_particles(moved,subset,4)
    from particledataset import ParticleDataset
    assert len(ParticleDataset(str(subset))) == 4


@pytest.mark.parametrize('field,value', [('snr',0),('snr',float('nan')),('n_particles',0),('seed',-1),
    ('apix',2),('amplitude_contrast',2),('cs',-1),('defocus_min',10),('astigmatism_max',float('inf'))])
def test_b6_bad_parameters_before_output(tmp_path, field, value):
    args = options(tmp_path); setattr(args,field,value)
    with pytest.raises(ValueError): sim.simulate(args)
    assert not args.out_dir.exists()


@pytest.mark.parametrize('bad', ['zero','nan','axes','starts','odd'])
def test_b6_invalid_map(tmp_path,bad):
    p = tiny_map(tmp_path/'map.mrc',n=23 if bad=='odd' else 24)
    with mrcfile.open(p,mode='r+') as h:
        if bad=='zero': h.data[:]=0
        if bad=='nan': h.data[0,0,0]=np.nan
        if bad=='axes': h.header.mapc=3
        if bad=='starts': h.header.nxstart=1
    with pytest.raises(ValueError): sim.load_map(p,1.)


def test_b6_refuse_existing_and_detect_metadata_change(tmp_path):
    args=options(tmp_path);sim.simulate(args)
    original=sim.sha256(args.out_dir/'particles.mrcs')
    with pytest.raises(FileExistsError):sim.simulate(args)
    assert original==sim.sha256(args.out_dir/'particles.mrcs')
    star=args.out_dir/'particles.star'
    star.write_text(star.read_text().replace('1@particles.mrcs','99@particles.mrcs'),encoding='utf-8')
    with pytest.raises(ValueError):sim.validate_particles(args.out_dir)


def test_b6_placement_modes_and_atom_guard(tmp_path):
    initial=write_fixture(tmp_path)
    placed=tmp_path/'placed.cif'; xyz=read_template(initial)[0].numpy()
    r=sim.Rotation.from_euler('z',35,degrees=True).as_matrix()
    write_coordinates(initial,placed,xyz@r.T+[1,2,3])
    assert validate_placement(initial,placed)['rigid_component_mse_A2']<1e-8
    canonical_placement(initial,placed,tmp_path/'reference.cif')
    distorted=tmp_path/'different.cif';xyz[0,0]+=2;write_coordinates(initial,distorted,xyz)
    with pytest.raises(ValueError,match='conformation'):validate_placement(initial,distorted)
    assert validate_placement(initial,distorted,True)['reference_kind']=='reference_structure'
    changed=placed.read_text().replace('ALA','GLY');placed.write_text(changed,encoding='utf-8')
    with pytest.raises(ValueError,match='identities'):validate_placement(initial,placed,True)



def test_b6_real_cpu_training_and_result_validator(tmp_path,monkeypatch,sampler):
    import train
    install(monkeypatch,sampler)
    args,_=fixture(tmp_path)
    args.epochs=2;args.batch_size=2;args.mini_batch_size=1
    args.learn_gmm=False;args.update_affine_mat=False;args.map_resolution=3.;args.resolution=3.
    args.record_dir=str(tmp_path/'records')
    train.main(args)
    result=validate_run(tmp_path,2,4,args.cif_path,args.star_data_dir,'cpu')
    assert result['passed'],result
    assert result['rmsd']['status']=='manual'
    assert np.isfinite(result['fixed_particle_frc']['delta'])
    cache=torch.load(tmp_path/'run_2.pth',weights_only=False)
    cache['gmm']['state_dict']['sdevs'][0,0]+=1
    torch.save(cache,tmp_path/'run_2.pth')
    result=validate_run(tmp_path,2,4,args.cif_path,args.star_data_dir,'cpu')
    assert not result['passed']


def case_module():
    spec=importlib.util.spec_from_file_location('b6_case_runner',ROOT/'examples/7zdt_7zd5/run_case.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_b6_predict_plan_uses_v1_and_stops_for_placement(tmp_path,monkeypatch):
    from types import SimpleNamespace
    module=case_module()
    source=tmp_path/'input.json';source.write_text(json.dumps([dict(name='7zdt',sequences=[dict(proteinChain=dict(sequence='AAA',count=1))])]))
    resource=tmp_path/'resources';(resource/'checkpoint').mkdir(parents=True)
    args=SimpleNamespace(input_json=source,resource_root=resource,output=tmp_path/'prediction',seed=42,device='cpu')
    with pytest.raises(FileNotFoundError,match='weights'):module.predict(args)
    assert not args.output.exists()
    (resource/'checkpoint'/f'{module.MODEL}.pt').write_bytes(b'placeholder for command planning test only')
    commands=[]
    def fake_execute(command,out,name,cwd=None):
        commands.append((name,list(map(str,command))))
        if name=='inference':
            (out/'params').mkdir();(out/'params/7zdt_diffusion_data.pth').write_bytes(b'no real weights')
        else:
            (out/'initial').mkdir();p=write_fixture(out/'initial');p.rename(out/'initial/7zdt_initial_prediction.cif')
    monkeypatch.setattr(module,'execute',fake_execute)
    module.predict(args)
    report=json.loads((args.output/'prediction.json').read_text())
    assert report['status']=='awaiting_map_placement'
    assert len(commands)==2 and commands[0][1][commands[0][1].index('--model_name')+1]==module.MODEL
    assert '--seeds' not in commands[0][1]
    assert commands[1][1][commands[1][1].index('--seed')+1]=='42'


def test_b6_smoke_refine_budgets_preserve_scientific_defaults(tmp_path):
    from types import SimpleNamespace
    module=case_module()
    for kind,epochs,count in [('smoke',2,32),('refine',10,1000)]:
        args=SimpleNamespace(kind=kind,device='cuda:0',seed=42,mini_batch_size=16)
        command,e,n=module.train_command(args,'ref.cif','particles.star','cache.pth',tmp_path)
        assert (e,n)==(epochs,count)
        assert '--no-learn-gmm' in command and '--train_deterministic' in command
        assert '--update_affine_mat' not in command and '--transR' not in command
        assert '--lr_bias' not in command  # actual train default stays authoritative
        assert command[command.index('--batch_size')+1]=='32'
