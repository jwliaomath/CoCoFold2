import torch
import sys
from run_recording import recorded, current_record, add_record_arguments
import argparse
import os
import numpy as np
import json
from pathlib import Path
from typing import Any

from cache_structure import structure_from_cache, export_structure


def _resolve_efficient_fusion(cache):
    """Saved overrides precede the author-confirmed Protenix inference default."""
    from collections.abc import Mapping
    key = 'enable_efficient_fusion'
    if key in cache:
        return cache[key], 'checkpoint'
    config = cache.get('configs')
    if isinstance(config, Mapping):
        if key in config:
            return config[key], 'checkpoint_configs'
    elif hasattr(config, key):
        return getattr(config, key), 'checkpoint_configs'
    return True, 'protenix_inference_default_true'


def _tree_to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested cache object to one device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value

def _sample_diffusion(configs, training=False, **kwargs: Any) -> torch.Tensor:
    """
    Samples diffusion process based on the provided configurations.
    """
    from protenix.utils.torch_utils import autocasting_disable_decorator
    from model.generator import sample_diffusion

    _configs = {
        key: configs.sample_diffusion.get(key)
        for key in [
            "gamma0",
            "gamma_min",
            "noise_scale_lambda",
            "step_scale_eta",
        ]
    }
    _configs.update(
        {
            "attn_chunk_size": (
                configs.infer_setting.chunk_size if not training else None
            ),
            "diffusion_chunk_size": (
                configs.infer_setting.sample_diffusion_chunk_size
                if not training
                else None
            ),
        }
    )
    return autocasting_disable_decorator(configs.skip_amp.sample_diffusion)(
        sample_diffusion
    )(configs=configs, **_configs, **kwargs)

@torch.no_grad()
def _decode_aligned_global(cache, device, seeds, replay=True):
    """Decode initial or refined global-sampler caches; apply saved bias once."""
    from protenix.model.modules.diffusion import DiffusionModule
    from randomness import apply_seed_settings, seed_legacy
    from checkpoint_sampling import effective_latent, replay_sampling, is_refinement, validate_sampling_snapshot
    from contextlib import nullcontext
    import copy
    if replay and cache.get('export_sampling'):
        validate_sampling_snapshot(cache['export_sampling'], device)
    data = {key: _tree_to_device(cache[key], device) for key in (
        'input_feature_dict', 's_inputs', 's_trunk', 'z_trunk', 'pair_z', 'p_lm', 'c_l', 'noise_schedule')}
    config = copy.deepcopy(cache['configs'])
    config.train_deterministic = True
    apply_seed_settings(config, seeds)
    z, pair = effective_latent(cache, data['z_trunk'], data['pair_z']) if is_refinement(cache) else (data['z_trunk'], data['pair_z'])
    model = DiffusionModule(**config.model.diffusion_module).to(device).eval().requires_grad_(False)
    model.load_state_dict(cache['model_state'])
    seed_legacy(seeds['diffusion_seed'])
    context = replay_sampling(config, cache['export_sampling'], device) if replay and cache.get('export_sampling') else nullcontext()
    with context:
        return _sample_diffusion(configs=config, training=False, denoise_net=model,
            input_feature_dict=data['input_feature_dict'], s_inputs=data['s_inputs'], s_trunk=data['s_trunk'],
            z_trunk=z, pair_z=pair, p_lm=data['p_lm'], c_l=data['c_l'], N_sample=cache['N_sample'],
            noise_schedule=data['noise_schedule'], inplace_safe=False if is_refinement(cache) else cache['inplace_safe'],
            enable_efficient_fusion=_resolve_efficient_fusion(cache)[0])[0]


@recorded("export")
def main(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(device)
    diffusion_data_dir = args.diffusion_data_dir
    out_dir = args.out_dir
    cif_path = args.cif_path

    print(f"Loading diffusion data from: {diffusion_data_dir}")
    # Contextual split caches are intentionally saved on CPU for portability.
    # Loading every cache on CPU first also prevents stale source-device IDs
    # from controlling placement when a cache is moved between machines.
    diffusion_data = torch.load(
        diffusion_data_dir, map_location="cpu", weights_only=False
    )
    from randomness import resolve_seeds
    from checkpoint_sampling import is_refinement, global_coordinates
    seed_settings = resolve_seeds(args, "export", diffusion_data)
    refined = is_refinement(diffusion_data)
    frame = getattr(args, 'coordinate_frame', 'reference')
    replay = getattr(args, 'seed', None) is None
    efficient_fusion, fusion_source = _resolve_efficient_fusion(diffusion_data)
    current_record().resolved("decode_configuration", dict(cache_configs=diffusion_data["configs"], seeds=seed_settings,
                            cache_noise_schedule=diffusion_data["noise_schedule"].detach().cpu().tolist(),
                            saved_sampling_replay=bool(replay and diffusion_data.get('export_sampling')),
                            enable_efficient_fusion=efficient_fusion,
                            efficient_fusion_source=fusion_source,
                            saved_sampling_settings={k: v for k, v in diffusion_data.get('export_sampling', {}).items()
                                                     if k not in ('rng', 'diffusion_rng_stream')},
                            coordinate_frame=frame if refined else 'raw', refined=refined, device=str(device)),
                            "Saved export sample unless seed overridden; old checkpoints use fixed diffusion seed; saved transforms are not refitted")
    current_record().refresh_environment()
    output_format = getattr(args, "output_format", None)
    topology, topology_report = None, None
    if cif_path is None:
        # Validate identities before spending time on GPU sampling.
        topology, topology_report = structure_from_cache(
            diffusion_data["input_feature_dict"], name=args.pdbid
        )
        print("Cache topology:", json.dumps(topology_report))
        output_format = output_format or "cif"
    if diffusion_data.get('coordinate_transform') is not None:
        # A refined block checkpoint is defined by pair + its saved transform.
        # Never silently export unaligned coordinates or omit the saved pair bias.
        from coordinate_transform import CoordinateTransform
        from single_structure_decoder import load_protenix
        from refinement_runtime import saved_alignment_sampler
        from structure_io import write_coordinates
        import tempfile
        saved_transform=diffusion_data['coordinate_transform']
        sampler_name = saved_alignment_sampler(diffusion_data)
        current_record().resolved('aligned_decoder', dict(refinement_sampler=sampler_name),
                                 'Explicit checkpoint decoder; unmarked old block checkpoints retain block decoder')
        if sampler_name == 'global':
            raw = _decode_aligned_global(diffusion_data, device, seed_settings, replay=replay)
            del diffusion_data
        else:
            del diffusion_data
            decoder,base,_=load_protenix(diffusion_data_dir,device,seed=seed_settings["diffusion_seed"])
        with tempfile.TemporaryDirectory(prefix="cache_topology_") as temporary:
            transform_template = cif_path
            if transform_template is None:
                transform_template = export_structure(
                    topology, np.zeros((topology_report["n_atom"], 3)),
                    Path(temporary) / "topology", "cif",
                )[0]
            # Retains the exact saved atom-identity/order and handoff checks.
            transform=CoordinateTransform.from_checkpoint(saved_transform,transform_template,device)
        with torch.no_grad():
            if sampler_name == 'block':
                raw=decoder(base[None])[0]
            transform.check_handoff(raw)
            aligned=(raw if frame == 'raw' else transform(raw)).cpu().numpy()
        os.makedirs(out_dir,exist_ok=True)
        stem=os.path.join(out_dir,f'{args.pdbid}_block_prediction')
        if topology is None and output_format is None:
            suffix='.cif' if str(cif_path).lower().endswith(('.cif','.mmcif')) else '.pdb'
            output=stem+suffix
            write_coordinates(cif_path,output,aligned)
            current_record().artifact(output, 'structure')
            print(f'Block-aligned prediction saved to: {output}')
        else:
            _export(args, topology, topology_report, aligned, stem, output_format)
        return

    # Validate saved placement before allocating the diffusion model.
    global_coordinates(diffusion_data, torch.zeros((1, 3)), frame)
    generated_coordinates = _decode_aligned_global(diffusion_data, device, seed_settings, replay=replay)
    generated_coordinates = global_coordinates(diffusion_data, generated_coordinates, frame)
    prediction_name = 'refined_prediction' if refined else 'initial_prediction'
    if topology is not None or output_format is not None:
        _export(
            args, topology, topology_report,
            generated_coordinates.detach().cpu().numpy(),
            os.path.join(out_dir, f"{args.pdbid}_{prediction_name}"),
            output_format,
        )
        return

    # Preserve the historical template + default PDB path.
    from utils import cif_to_tensor, replace_cif_coordinates
    template_coordinates, _ = cif_to_tensor(cif_path)
    if generated_coordinates.shape != template_coordinates.shape:
        raise ValueError(
            "generated coordinates do not match the topology template: "
            f"generated {tuple(generated_coordinates.shape)}, "
            f"template {tuple(template_coordinates.shape)}"
        )

    # Create the output directory.
    os.makedirs(out_dir, exist_ok=True)
    out_pdb_path = os.path.join(out_dir, f"{args.pdbid}_{prediction_name}.pdb")

    print(f"Replacing coordinates into reference template...")
    replace_cif_coordinates(
        input_cif=cif_path,
        output_cif=out_pdb_path,
        new_coords=generated_coordinates.detach().cpu().numpy(),
    )

    current_record().artifact(out_pdb_path, "structure")
    print(f"✅ Prediction successfully saved to: {out_pdb_path}")

def _export(args, topology, report, coordinates, stem, output_format):
    if topology is None:
        import gemmi
        topology = gemmi.read_structure(args.cif_path)
    paths = export_structure(topology, coordinates, stem, output_format)
    if report is not None:
        report = dict(report, cache_path=os.path.abspath(args.diffusion_data_dir),
                      output_files=paths)
        Path(stem + "_topology.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        current_record().artifact(stem + "_topology.json", "topology_report")
    for path in paths:
        current_record().artifact(path, "structure")
        print(f"Prediction saved to: {path}")


def build_parser():
    parser = argparse.ArgumentParser()
    from randomness import add_seed_arguments
    add_seed_arguments(parser, "export")
    add_record_arguments(parser)
    parser.add_argument("--pdbid", required=True, help="Target PDB ID")
    parser.add_argument("--diffusion_data_dir", required=True, help="Path to the diffusion .pth data")
    parser.add_argument("--cif_path", default=None, help="Optional reference CIF/PDB topology; otherwise decode cache features")
    parser.add_argument("--output_format", "--output-format", choices=("cif", "pdb", "both"), default=None,
                        help="Default: CIF without template, historical format with template")
    parser.add_argument("--out_dir", required=True, help="Directory to save the prediction")
    parser.add_argument("--device", default="cuda:0", help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    parser.add_argument("--coordinate-frame", choices=("reference", "raw"), default="reference",
                        help="Refinement: apply saved R/t by default; raw omits placement. Initial caches stay raw.")

    return parser


if __name__ == "__main__":
    original_argv = list(sys.argv)
    args = build_parser().parse_args()
    args.original_argv = original_argv
    main(args)
