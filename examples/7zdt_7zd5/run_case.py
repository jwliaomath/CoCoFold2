"""B6: generate particles; predict v1; pause for map placement; smoke/refine.

Real model execution is only performed by the explicit predict/train commands.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from simulate_particles import sha256

MODEL = 'protenix_base_default_v1.0.0'


def execute(command, directory, name, cwd=ROOT):
    started = time.monotonic()
    command = list(map(str, command))
    record = dict(command=command, cwd=str(cwd), status='running')
    manifest = directory/(name+'_command.json')
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(json.dumps(record, indent=2), encoding='utf-8')
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    # Keep module resolution inside the public source plus installed packages.
    env['PYTHONPATH'] = str(ROOT/'src')
    with (directory/(name+'.log')).open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, status='success' if result.returncode == 0 else 'failed',
                  elapsed_seconds=time.monotonic()-started)
    manifest.write_text(json.dumps(record, indent=2), encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; see {directory/(name+".log")}')


def predict(args):
    from inference_io import read_inputs
    targets = read_inputs(args.input_json)
    if len(targets) != 1:
        raise ValueError('This case requires exactly one protein target in the input JSON')
    if any(set(row) != {'proteinChain'} for row in targets[0]['sequences']):
        raise ValueError('This B6 case requires protein-only JSON; do not silently drop ligands/nucleic acids')
    resource = args.resource_root.resolve()
    weights = resource/'checkpoint'/f'{MODEL}.pt'
    if not weights.is_file():
        raise FileNotFoundError(f'Prepare the existing v1 weights before this test: {weights}')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.input_json, out/'input_snapshot.json')
    request = dict(model_name=MODEL, input_json=str(args.input_json.resolve()), input_sha256=sha256(args.input_json),
        resource_root=str(resource), refinement_seed=args.seed,
        initial_prediction_seed='Existing inference default (101); recorded in inference run records',
        input_policy='Original 7ZDT JSON/MSA used as supplied; no sequence editing', status='running')
    (out/'prediction.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    command = [sys.executable, ROOT/'src/inference.py', '--input_json_path', args.input_json.resolve(),
        '--model_name', MODEL, '--sample_name', '7zdt', '--resource-root', resource,
        '--output_model_dir', out/'params', '--dump_dir', out/'inference']
    # Relative paths inside an existing JSON keep the original Protenix working root.
    execute(command, out, 'inference', cwd=resource)
    cache = out/'params/7zdt_diffusion_data.pth'
    execute([sys.executable, ROOT/'src/get_pdb.py', '--diffusion_data_dir', cache,
        '--pdbid', '7zdt', '--out_dir', out/'initial', '--seed', args.seed,
        '--output-format', 'cif', '--device', args.device], out, 'initial_export')
    initial = out/'initial/7zdt_initial_prediction.cif'
    request.update(status='awaiting_map_placement', cache=str(cache), cache_sha256=sha256(cache),
        initial_cif=str(initial), initial_sha256=sha256(initial),
        next='Fit this CIF to the supplied map and save it, or provide a topology-compatible placed reference CIF; then run train.')
    (out/'prediction.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    print(json.dumps(request, indent=2))


def train_command(args, reference, star, cache, output):
    count = 32 if args.kind == 'smoke' else 1000
    epochs = 2 if args.kind == 'smoke' else 10
    return [sys.executable, ROOT/'src/train.py', '--star_data_dir', star,
        '--diffusion_data_dir', cache, '--cif_path', reference,
        '--output_trained_model_dir', output/'run_', '--record-dir', output/'records',
        '--device', args.device, '--seed', args.seed, '--train_deterministic',
        '--no-learn-gmm', '--coordinate-mode', 'global', '--output-format', 'cif',
        '--apix', '1', '--boxsize', '192', '--particle_sign', '-1', '--resolution', '3', '--map_resolution', '3',
        '--batch_size', '32', '--mini_batch_size', str(args.mini_batch_size), '--epochs', str(epochs),
        '--gmm-kernel', 'legacy', '--gmm-amplitude', 'auto', '--gmm-atom-chunk-size', '1000', '--gmm-checkpoint-peak2d'], epochs, count


def train(args):
    from benchmark_case import canonical_placement, select_particles
    prediction_dir = args.prediction.resolve()
    prediction = json.loads((prediction_dir/'prediction.json').read_text(encoding='utf-8'))
    cache = prediction_dir/'params/7zdt_diffusion_data.pth'
    initial = prediction_dir/'initial/7zdt_initial_prediction.cif'
    if prediction.get('status') != 'awaiting_map_placement' or prediction.get('model_name') != MODEL:
        raise ValueError('Expected completed v1 prediction/export stage')
    if args.seed != prediction['refinement_seed'] or sha256(initial) != prediction['initial_sha256']:
        raise ValueError('Prediction/placement seed or initial CIF provenance mismatch')
    if sha256(cache) != prediction['cache_sha256']:
        raise ValueError('Prediction cache changed after initial export')
    data = args.particles.resolve()
    simulation = json.loads((data/'simulation.json').read_text(encoding='utf-8'))
    if simulation['status'] != 'completed' or simulation['n_particles'] != 1000 or simulation['boxsize'] != 192 or simulation['apix'] != 1. or simulation['requested_snr'] != 1.:
        raise ValueError('B6 requires completed 1000-particle SNR=1, 192 box, 1 Å/pixel simulation')
    if any(abs(float(v)) > 1e-6 for v in simulation['grid']['map_center_A']):
        raise ValueError('B6 map must be centered at physical (0,0,0); use the supplied molmap and its frame')
    for name, digest in simulation['outputs'].items():
        if sha256(data/name) != digest:
            raise ValueError(f'Simulation output changed: {name}')
    from benchmark_case import validate_placement
    validate_placement(initial, args.placed_cif, args.reference_structure)
    identity = dict(input_cache_sha256=prediction['cache_sha256'], placed_cif_sha256=sha256(args.placed_cif),
                    simulation_sha256=sha256(data/'simulation.json'), mini_batch_size=args.mini_batch_size, seed=args.seed)
    if args.kind == 'refine':
        if args.smoke_result is None:
            raise ValueError('Run smoke first, then provide --smoke-result pointing to its output directory')
        smoke = args.smoke_result.resolve()
        prior = json.loads((smoke/'case.json').read_text(encoding='utf-8'))
        passed = json.loads((smoke/'validation.json').read_text(encoding='utf-8'))
        if prior['kind'] != 'smoke' or not passed['passed'] or any(prior.get(key) != value for key,value in identity.items()):
            raise ValueError('Refine requires successful smoke with the same cache, reference, data, seed and mini-batch size')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    reference = out/'reference.cif'
    placement = canonical_placement(initial, args.placed_cif, reference, args.reference_structure)
    placement.update(source_reference=str(args.placed_cif.resolve()), source_map=simulation['source_map'],
        warning='Topology check does not establish map fit quality. The supplied reference must already be in the map frame.')
    (out/'placement.json').write_text(json.dumps(placement, indent=2), encoding='utf-8')
    star = out/'particles.star'
    command, epochs, count = train_command(args, reference, star, cache, out)
    select_particles(data, star, count)
    request = dict(kind=args.kind, n_particles=count, epochs=epochs, expected_steps=epochs*math.ceil(count/32),
        production_defaults_preserved=True, gmm_learning=False, prediction=str(prediction_dir),
        particle_source=str(data), simulation_sha256=sha256(data/'simulation.json'),
        code_sha256={p.name:sha256(p) for p in (ROOT/'src').glob('*.py')}, command=list(map(str, command)))
    request.update(identity)
    (out/'case.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    preflight = command.copy()
    record_index = preflight.index('--record-dir')
    del preflight[record_index:record_index+2]
    execute(preflight+['--check-inputs'], out, 'preflight')
    execute(command, out, 'train')
    # A fresh process frees diffusion model/optimizer GPU memory before rendering
    # the fixed-particle comparison. No second diffusion decode is performed.
    execute([sys.executable, Path(__file__).resolve(), 'validate', '--output', out, '--device', args.device], out, 'validate')
    print(f'{args.kind} finished; inspect {out/"validation.json"}')


def validate(args):
    from benchmark_case import validate_run
    out = args.output.resolve()
    request = json.loads((out/'case.json').read_text(encoding='utf-8'))
    result = validate_run(out, request['epochs'], request['expected_steps'], out/'reference.cif', out/'particles.star', args.device)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError(f'Case checks failed; see {out/"validation.txt"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    pred = sub.add_parser('predict', help='Generate v1 params and initial CIF; pause for map fitting.')
    pred.add_argument('--input-json', type=Path, required=True, help='Protenix input JSON, including sequence/MSA paths; relative path uses the current working directory. Default: %(default)s.')
    pred.add_argument('--resource-root', type=Path, required=True, help='Directory containing Protenix common/ and checkpoint/ resources. Default: %(default)s.')
    pred.add_argument('--output', type=Path, required=True, help='Output path for this command; relative paths use the working directory. Existing training output directories are refused. Default: %(default)s.')
    pred.add_argument('--seed', type=int, default=42, help='Initial export/refinement diffusion seed; prediction seed retains inference default.')
    pred.add_argument('--device', default='cuda:0', help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    pred.set_defaults(function=predict)
    tr = sub.add_parser('train', help='Run smoke/refine using a user-placed compatible reference CIF.')
    tr.add_argument('--kind', choices=['smoke','refine'], required=True, help='smoke uses the small particle subset; refine uses all simulated particles and requires successful smoke. Default: %(default)s.')
    tr.add_argument('--prediction', type=Path, required=True, help='Completed prediction-stage directory containing the cache and exported initial structure. Default: %(default)s.')
    tr.add_argument('--particles', type=Path, required=True, help='Particle simulation output directory containing particles.star and its report. Default: %(default)s.')
    tr.add_argument('--placed-cif', type=Path, required=True, help='Initial structure rigidly placed in the map frame, preserving cache atom identities/order. Default: %(default)s.')
    tr.add_argument('--reference-structure', action='store_true', help='Accept a compatible reference conformation; otherwise require a rigidly moved copy of the prediction.')
    tr.add_argument('--smoke-result', type=Path, help='Successful smoke directory required before refine with the same inputs.')
    tr.add_argument('--output', type=Path, required=True, help='Output path for this command; relative paths use the working directory. Existing training output directories are refused. Default: %(default)s.')
    tr.add_argument('--mini-batch-size', type=int, default=16, help='Keep fixed for comparison; changing this can change legacy loss scaling.')
    tr.add_argument('--seed', type=int, default=42, help='Random seed for this operation; does not modify previously generated input files. Default: %(default)s.')
    tr.add_argument('--device', default='cuda:0', help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    tr.set_defaults(function=train)
    val = sub.add_parser('validate', help='Check existing results without diffusion model execution.')
    val.add_argument('--output', type=Path, required=True, help='Existing smoke/refine directory to audit; does not rerun training.')
    val.add_argument('--device', default='cpu', help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    val.set_defaults(function=validate)
    args = p.parse_args()
    if hasattr(args, 'seed') and not 0 <= args.seed < 2**32:
        p.error('seed must be in [0, 2**32)')
    if hasattr(args, 'mini_batch_size') and args.mini_batch_size <= 0:
        p.error('mini-batch-size must be positive')
    args.function(args)


if __name__ == '__main__':
    main()
