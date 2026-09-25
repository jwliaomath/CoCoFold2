import torch
from run_recording import recorded, current_record, add_record_arguments, file_identity
from training_output import export_training_structure
import argparse
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
from typing import Any


from utils import cif_to_tensor, kabsch_alignment, compute_frc, replace_cif_coordinates, discrete_radon_transform_3d, translation_2d, compute_frc_simulate, compute_ncc_loss
from utils_halfmap import build_halfmap_shell_weights

from particledataset import ParticleDataset
from ctf import compute_ctf
from pts2img import pdb2img,sum_of_gaussians_2d_torch
from pts2img import translation_2d_robust as pts_translation_2d
from gmm import add_gmm_arguments, gmm_from_arguments
from coordinate_transform import add_coordinate_arguments,prepare_coordinate_transform


from pathlib import Path
import matplotlib.pyplot as plt


def to_numpy_img(x):
    """Convert torch/numpy image tensor to 2D numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x)

    x = np.squeeze(x)

    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-2], x.shape[-1])[0]

    return x


def normalize_pair_for_display(a, b, p_low=1, p_high=99):
    """
    Normalize observed/predicted pair using shared percentile range.
    This keeps their contrast comparable.
    """
    a = to_numpy_img(a)
    b = to_numpy_img(b)

    both = np.concatenate([a.ravel(), b.ravel()])
    vmin, vmax = np.percentile(both, [p_low, p_high])

    if vmax <= vmin:
        vmax = vmin + 1e-6

    a = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    b = np.clip((b - vmin) / (vmax - vmin), 0, 1)

    return a, b


def save_gray_image(img, path):
    plt.imsave(path, img, cmap="gray", vmin=0, vmax=1)

def _sample_diffusion(configs, training=False, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
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
        )(configs = configs, **_configs, **kwargs)

@recorded("train")
def main(args):
    # Programmatic callers may supply a Namespace created before these CLI
    # options existed. Keep their historical renderer semantics.
    if not hasattr(args, 'projection_frame'):
        args.projection_frame = 'legacy'
    if not hasattr(args, 'projection_origin'):
        args.projection_origin = (0., 0., 0.)
    from randomness import resolve_seeds, apply_seed_settings, seed_legacy, data_loader_options
    from training_restart import prepare_restart, input_identity, restart_gmm, capture_epoch_state, restore_epoch_rng
    restart_cache = prepare_restart(args)
    resuming = getattr(args, 'resume', False)
    warm_start = getattr(args, 'warm_start', False)
    seed_settings = resolve_seeds(args)
    from input_validation import validate_train_inputs, move_tree
    diffusion_data, dataset, input_report = validate_train_inputs(args, cache=restart_cache)
    if getattr(args, 'check_inputs', False):
        import json
        current_record().resolved("preflight", input_report, "Explicitly recorded input check; no model execution")
        print(json.dumps(input_report))
        return input_report
    record = current_record()
    resume_state = diffusion_data.get('training_resume') if resuming else None
    saved_optimizer = diffusion_data.get('opt_state') if resuming else None
    resume_bias = diffusion_data.get('z_bias') if resuming else None
    restart_inputs = input_identity(args, input_report)
    restart_payload = diffusion_data if resuming or warm_start else None
    record.resolved("validated_inputs", dict(arguments=vars(args), inputs=input_report), "CLI defaults and typed input/path preflight")
    record.resolved("particle_stack_identity", [file_identity(row["path"]) for row in input_report.get("stack_headers", [])],
                    "Unique stacks from existing preflight; metadata only, no full MRCS hash scan")
    from protenix.model.modules.diffusion import DiffusionModule
    flag_update_mat = args.update_affine_mat
    device = args.device
    box_size = int(args.boxsize)
    particle_sign = float(args.particle_sign)
    if args.density_center is None:
        density_center =  torch.tensor([box_size/2,box_size/2], dtype=torch.float32).to(device)
    else:
        density_center = torch.tensor(args.density_center, dtype=torch.float32).to(device)
        density_center = density_center.unsqueeze(0)
    diffusion_data_dir = str(args.diffusion_data_dir)
    diffusion_data = move_tree({key: value for key, value in diffusion_data.items()
                               if key not in ('training_resume', 'opt_state')}, device)
    saved_coordinate_transform=diffusion_data.get('coordinate_transform')
    block_mode=(getattr(args,'coordinate_mode','global')=='blocks' or saved_coordinate_transform is not None)
    from refinement_runtime import alignment_sampler, apply_saved_global_bias
    sampler_name = alignment_sampler(args, diffusion_data)
    use_block_decoder = sampler_name == 'block'
    saved_gmm = diffusion_data.get('gmm') if saved_coordinate_transform is not None else None
    if getattr(args,'block_alignment',None) and not block_mode:
        raise ValueError('--block-alignment requires --coordinate-mode blocks')
    if block_mode and not args.train_deterministic:
        raise ValueError('Block mode requires fixed noise (--train_deterministic)')

    pred_dict = diffusion_data["pred_dict"]
    input_feature_dict = diffusion_data["input_feature_dict"]
    s_inputs = diffusion_data["s_inputs"]
    s = diffusion_data["s_trunk"]
    z = diffusion_data["z_trunk"]
    pair_z = diffusion_data["pair_z"]
    p_lm = diffusion_data["p_lm"]
    c_l = diffusion_data["c_l"]
    N_sample = diffusion_data["N_sample"]
    noise_schedule = diffusion_data["noise_schedule"]
    inplace_safe = diffusion_data["inplace_safe"]
    configs = diffusion_data["configs"]
    enable_efficient_fusion = diffusion_data["enable_efficient_fusion"]
    record.resolved("cache_configuration", configs, "Configuration read from input cache before runtime overrides")
    configs.train_deterministic = args.train_deterministic
    apply_seed_settings(configs, seed_settings)
    record.resolved("refinement_configuration", dict(configs=configs, seeds=seed_settings,
                    noise_schedule=noise_schedule.detach().cpu().tolist(), N_sample=int(N_sample),
                    halfmap_weights_used_in_loss=False, loss_reduction="legacy: sum(minibatch negative mean FRC / optimizer batch size) + sum(minibatch GMM penalty)"),
                    "train deterministic/seed CLI overrides; cached sampler settings retained; half-map weights remain unused")

    block_decoder=None
    if use_block_decoder:
        # Use the public single-structure sampler without changing the original bias target:
        # z_trunk remains trainable upstream of conditioning when present.
        from single_structure_decoder import load_protenix
        block_decoder,block_base,block_metadata=load_protenix(diffusion_data_dir,device,seed=seed_settings["diffusion_seed"], **({"apply_saved_bias": False} if resuming else {}))
        diffusion_module=block_decoder.module
        if z is None:
            pair_z=block_base
        else:
            from single_structure_decoder import single
            z=single(z.detach().to(device),3,'z_trunk').float()
            if not resuming and diffusion_data.get('z_mul') is not None:z=z*diffusion_data['z_mul'].to(device)
            if not resuming and diffusion_data.get('z_bias') is not None:z=z+diffusion_data['z_bias'].to(device)
            z=single(z,3,'refined z_trunk').detach()
            pair_z=None
        del block_base
        p_lm=None;c_l=None;N_sample=1
        input_feature_dict=block_decoder.features
        s_inputs=block_decoder.s_inputs;s=block_decoder.s_trunk
        noise_schedule=block_decoder.schedule
        enable_efficient_fusion=False
    else:
        diffusion_module = DiffusionModule(**configs.model.diffusion_module).to(device)
        diffusion_module.load_state_dict(diffusion_data["model_state"])
        if warm_start:
            from checkpoint_sampling import effective_latent
            z, pair_z = effective_latent(diffusion_data, z, pair_z)
        elif not resuming:
            z, pair_z = apply_saved_global_bias(diffusion_data, z, pair_z)
    del diffusion_data, restart_cache

    def decode_block(bias=None):
        value=(pair_z if z is None else z)
        if bias is not None:value=value+bias
        if z is not None:
            value=diffusion_module.diffusion_conditioning.prepare_cache(input_feature_dict['relp'],value,False)
        return block_decoder(value[None])

    seed_legacy(seed_settings["seed"])

    for name, param in diffusion_module.named_parameters():
        param.requires_grad = False
    diffusion_module.eval()

    if resuming:
        # Saved post-step coordinates are sufficient for initialization/handoff.
        # Do not decode the unperturbed base or refit saved geometry during resume.
        pred_dict = dict(pred_dict)
    else:
        preds = []
        for i in range(2):
            with torch.no_grad():
                start = time.time()
                diffusion_module.eval()
                pred_dict["coordinate"] = decode_block() if use_block_decoder else _sample_diffusion(
                    configs=configs,
                    training=False,
                    denoise_net=diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=z,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    N_sample=N_sample,
                    noise_schedule=noise_schedule,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                )
                end = time.time()
                print(pred_dict["coordinate"])
                preds.append(pred_dict["coordinate"])
        print(torch.mean((preds[0]-preds[1])**2))
        with torch.no_grad():
            aligned_coords = kabsch_alignment(preds[0][0],preds[1][0])
            print(torch.mean((aligned_coords-preds[1][0])**2))

    cif_path = str(args.cif_path)
    ref_coords, atom_weights = cif_to_tensor(cif_path)

    if restart_payload is not None:
        gmm, gmm_source = restart_gmm(restart_payload, args, atom_weights, device)
    else:
        if (use_block_decoder and block_metadata.get('gmm') is not None) or (block_mode and not use_block_decoder and saved_gmm is not None):
            from gmm import GaussianProjector
            gmm=GaussianProjector.from_checkpoint(block_metadata['gmm'] if use_block_decoder else saved_gmm,device)
            gmm.atom_chunk_size=args.gmm_atom_chunk_size
            gmm.checkpoint_chunks=args.gmm_checkpoint_chunks
            gmm.checkpoint_peak2d=args.gmm_checkpoint_peak2d
        else:
            gmm = gmm_from_arguments(atom_weights, args, shape_device=device)
        gmm_source = 'existing_initialization'
    learn_gmm = getattr(args, "learn_gmm", True)
    output_format = getattr(args, "output_format", "cif")
    gmm.set_learning_enabled(learn_gmm)
    atom_weights = gmm.atom_weights
    print('gmm_config', gmm.config())
    resolution = torch.tensor([float(args.resolution)]).to(torch.float).to(device)

    coordinate_transform=None
    with torch.no_grad():
        if block_mode:
            handoff_raw = pred_dict['coordinate'][0]
            if warm_start and saved_coordinate_transform is not None:
                saved_raw = restart_payload.get('pred_dict', {}).get('coordinate')
                if isinstance(saved_raw, torch.Tensor):
                    handoff_raw = saved_raw[0].to(device)
            coordinate_transform=prepare_coordinate_transform(args,handoff_raw,cif_path,saved_coordinate_transform)
            coordinate_transform.report.setdefault('source_cache', file_identity(diffusion_data_dir))
            coordinate_transform.report.setdefault('diffusion_seed', seed_settings['diffusion_seed'])
            rotation=torch.eye(3,device=pred_dict['coordinate'].device)
            translation=rotation.new_zeros(3)  # placeholders; block path bypasses them
        elif restart_payload is not None and restart_payload.get('rotation') is not None and restart_payload.get('translation') is not None:
            rotation = restart_payload['rotation'].to(device)
            translation = restart_payload['translation'].to(device)
        else:
            _, rotation, translation = kabsch_alignment(pred_dict["coordinate"][0],ref_coords,return_transform=True)


    apix =float(args.apix)
    if args.transR :
        transR = np.array([[1,0,0],[0,1,0],[0,0,-1]]).reshape(3,3)
    else:
        transR = None
    # Reuse the dataset already checked before model construction.
    print(f'The dataset contains {len(dataset)} particles.')

    batch_size = int(args.batch_size)
    mini_batch_size = int(args.mini_batch_size)
    print('batch_size',batch_size,'mini_batch_size',mini_batch_size)

    lr_atoms_weights = getattr(args, 'lr_atom_weights', 1e-2)
    lr_sdevs = getattr(args, 'lr_sdevs', 5e-3)
    lr_bias = getattr(args, 'lr_bias', 1e-2)
    lr_mul = 2e-4
    # z_reg_weight = 0

    # s_inputs_bias = torch.zeros_like(s_inputs,requires_grad=True).to(device)
    # s_bias = torch.zeros_like(s,requires_grad=True).to(device)
    s_inputs_bias = None
    s_bias = None
    if z is None:
        # z_mul = torch.ones_like(pair_z,requires_grad=True).to(device)
        z_mul = None
        z_bias = torch.zeros_like(pair_z,requires_grad=True).to(device)
    else:
        # z_mul = torch.ones_like(z,requires_grad=True).to(device)
        z_mul = None
        z_bias = torch.zeros_like(z,requires_grad=True).to(device)

    if resuming:
        z_bias = resume_bias.detach().to(device).clone().requires_grad_(True)
    placement_source = ('saved_chain_transform' if saved_coordinate_transform is not None else
                        'saved_global_transform' if restart_payload is not None and restart_payload.get('rotation') is not None
                        and restart_payload.get('translation') is not None else 'fitted_from_current_cif')
    record.resolved('restart', dict(mode='resume' if resuming else 'warm_start' if warm_start else 'initial',
                    source_checkpoint=str(args.diffusion_data_dir), gmm_source=gmm_source,
                    placement_source=placement_source,
                    parent_run_id=resume_state['parent_run_id'] if resuming else None),
                    'Completed-epoch resume inherits state; warm-start resets optimizer/progress and retains available parameters')
    if resuming:
        record.event('resume', global_step=resume_state['global_step'], epoch=resume_state['next_epoch'],
                     source_checkpoint=str(args.diffusion_data_dir), parent_run_id=resume_state['parent_run_id'])
    del restart_payload
    # optimizer.add_param_group({'params': [s_inputs_bias,s_bias,z_bias], 'lr': lr_bias})
    optimizer = torch.optim.AdamW([z_bias],lr=lr_bias)
    # optimizer.add_param_group({'params': [z_mul], 'lr': lr_mul})
    if learn_gmm:
        optimizer.add_param_group({'params': gmm.amplitude_parameters(), 'lr': lr_atoms_weights})
        optimizer.add_param_group({'params': gmm.shape_parameters(), 'lr': lr_sdevs})
    if resuming:
        optimizer.load_state_dict(saved_optimizer)
        del saved_optimizer
    print('lr_atom_weight',lr_atoms_weights)
    print('lr_sdevs',lr_sdevs)
    print('lr_bias',lr_bias)
    if z_mul is not None:
        print('lr_mul',lr_mul)
    output_trained_model_dir = str(args.output_trained_model_dir)

    os.makedirs(os.path.dirname(output_trained_model_dir) or '.', exist_ok=True)

    for output in export_training_structure(cif_path, output_trained_model_dir+'_',
                                            pred_dict['coordinate'][0].cpu().numpy(), output_format):
        record.artifact(output, 'initial_structure')
    if coordinate_transform is not None:
        import json
        for output in export_training_structure(cif_path, output_trained_model_dir+'aligned_initial',
                        coordinate_transform(pred_dict['coordinate'][0]).detach().cpu().numpy(), output_format):
            record.artifact(output, 'initial_structure')
        Path(output_trained_model_dir+'block_alignment_report.json').write_text(
            json.dumps(coordinate_transform.report,indent=2),encoding='utf-8')
        record.artifact(output_trained_model_dir+'block_alignment_report.json', 'alignment_report')

    epochs = getattr(args, 'epochs', 10)
    max_steps = getattr(args, 'max_steps', None)
    global_step = resume_state['global_step'] if resuming else 0
    start_epoch = resume_state['next_epoch'] if resuming else 0
    limit = [0.1,0.8,1,20]

    freqs = (
            np.stack(
                np.meshgrid(
                    np.linspace(-0.5, 0.5, box_size, endpoint=False),
                    np.linspace(-0.5, 0.5, box_size, endpoint=False),
                ),
                -1,
            )
            / apix
            )

    freqs = freqs.reshape(-1, 2)
    freqs = torch.from_numpy(freqs).unsqueeze(0).to(torch.float)
    apix = torch.tensor([float(args.apix)]).to(torch.float).to(device)
    target_resolution = float(args.map_resolution)
    # target_resolution = 2 * apix
    record.refresh_environment()
    record.resolved("optimizer_and_geometry", dict(learning_rates=[group["lr"] for group in optimizer.param_groups],
                    gmm=gmm.export_checkpoint(), coordinate_mode="blocks" if block_mode else "global",
                    alignment_bodies=None if coordinate_transform is None else coordinate_transform.body_names,
                    alignment_update=bool(flag_update_mat),
                    block_update_trace_threshold=getattr(args, 'block_update_trace_threshold', 2.5) if block_mode else None,
                    bias_target="pair_z" if z is None else "z_trunk", gmm_learning=learn_gmm, output_format=output_format,
                    N_sample=int(N_sample), noise_schedule=noise_schedule.detach().cpu().tolist(),
                    sampler="single_structure_fixed_noise" if use_block_decoder else "global",
                    cached_atom_conditioning=dict(p_lm=p_lm, c_l=c_l),
                    enable_efficient_fusion=enable_efficient_fusion,
                    input_dtype=str(s_inputs.dtype), epoch_index_base=0, batch_index_base=0,
                    regularization_limits=limit, target_resolution=target_resolution,
                    structure_checkpoint_step_consistency="post_step"), "Actual optimizer/GMM/coordinate setup")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **data_loader_options(seed_settings))
    # diffusion_module.train()
    # torch.autograd.set_detect_anomaly(True)
    ncc_weight = 0  # Historical suggested weight: 0.2 - 0.5; the active value remains zero.
    frc_weight = 1.0 - ncc_weight

    shell_weight_freqs, shell_weights = build_halfmap_shell_weights(
    halfmap1_path=args.halfmap1,
    halfmap2_path=args.halfmap2,
    gamma=float(args.fsc_gamma),
    smooth_win=int(args.fsc_smooth_win),
    device=device,
    dtype=torch.float32,
    )

    if shell_weights is None:
        print("[FRC] No half-maps provided. Using all-ones shell weights.")
    else:
        print(f"[FRC] Loaded half-map shell weights: {tuple(shell_weights.shape)}")

    if resuming:
        restore_epoch_rng(resume_state, configs, dataloader)
        del resume_state
    for epoch in range(start_epoch, epochs):

        #vis_dir = Path("paper_particle_examples")
        #vis_dir.mkdir(parents=True, exist_ok=True)

        #n_save_vis = 3
        #saved_vis = 0

        for num, batch in enumerate(dataloader):

                start = time.time()
                data,para,trans,R,R2,index = batch

                data = data.unsqueeze(1)
                data = data.to(device).to(torch.float)
                trans = trans.to(device).to(torch.float)
                R = R.to(device).to(torch.float)
                R2 = R2.to(device).to(torch.float)
                para = para.to(torch.float)
                voltage, defocusU, defocusV,astigmatism, Cs, amplitude, phase_shift, pixel_size = para.T
                voltage = voltage.unsqueeze(1)
                defocusU = defocusU.unsqueeze(1)
                defocusV = defocusV.unsqueeze(1)
                astigmatism = astigmatism.unsqueeze(1)
                Cs = Cs.unsqueeze(1)
                amplitude = amplitude.unsqueeze(1)
                phase_shift = phase_shift.unsqueeze(1)

                ctf = compute_ctf(
                    freqs=freqs,
                    dfu=defocusU,
                    dfv=defocusV,
                    dfang=astigmatism,
                    volt=voltage,
                    cs=Cs,
                    w=amplitude,
                    phase_shift=phase_shift,
                    bfactor=None
                )

                ctf = ctf.reshape(ctf.shape[0],1,box_size,box_size).to(device)
                ctf = ctf.to(torch.float)

                losses = 0
                penalties = 0
                # ncc_losses = 0
                for num_start in range(0, data.shape[0], mini_batch_size):
                    num_end = min(num_start + mini_batch_size,  data.shape[0])
                    # mb = num_end - num_start
                    # batch_weight = mb / data.shape[0]

                    if use_block_decoder:
                        pred_dict['coordinate']=decode_block(z_bias)
                    elif z is None:
                        pred_dict["coordinate"] = _sample_diffusion(
                            configs=configs,
                            training=False,
                            denoise_net=diffusion_module,
                            input_feature_dict=input_feature_dict,
                            # s_inputs=s_inputs + s_inputs_bias,
                            # s_trunk=s + s_bias,
                            s_inputs=s_inputs,
                            s_trunk=s,
                            z_trunk=z,
                            # pair_z=z_mul * pair_z+ z_bias,
                            pair_z=pair_z+ z_bias,
                            p_lm=p_lm,
                            c_l=c_l,
                            N_sample=N_sample,
                            noise_schedule=noise_schedule,
                            inplace_safe=False,
                            enable_efficient_fusion=enable_efficient_fusion,
                        )
                    else:
                        pred_dict["coordinate"] = _sample_diffusion(
                            configs=configs,
                            training=False,
                            denoise_net=diffusion_module,
                            input_feature_dict=input_feature_dict,
                            # s_inputs=s_inputs + s_inputs_bias,
                            # s_trunk=s + s_bias,
                            s_inputs=s_inputs,
                            s_trunk=s,
                            # z_trunk=z_mul * z + z_bias,
                            z_trunk=z + z_bias,
                            pair_z=pair_z,
                            p_lm=p_lm,
                            c_l=c_l,
                            N_sample=N_sample,
                            noise_schedule=noise_schedule,
                            inplace_safe=False,
                            enable_efficient_fusion=enable_efficient_fusion,
                        )
                    if flag_update_mat and block_mode:
                        updates = coordinate_transform.update_from_coordinates(pred_dict['coordinate'][0],
                                    getattr(args, 'block_update_trace_threshold', 2.5))
                        for update in updates:
                            record.event('alignment_update', epoch=epoch, global_step=global_step,
                                         minibatch_start=num_start, **update)
                    elif flag_update_mat:
                        with torch.no_grad():
                            _, current_rotation, current_translation = kabsch_alignment(pred_dict["coordinate"][0],ref_coords,return_transform=True)
                            rotation_diff = torch.trace(current_rotation @ rotation.T)
                        if rotation_diff < 2.5:
                            print('Flip happened')
                            rotation = current_rotation
                            translation = current_translation
                    atom_coord = (coordinate_transform(pred_dict['coordinate'][0]) if block_mode
                                  else pred_dict["coordinate"][0] @ rotation.T + translation)
                    proj = gmm(
                        atoms_coord=atom_coord.reshape(1,-1,3),
                        resolution=float(args.resolution),
                        rotation=R[num_start:num_end],
                        trans=trans[num_start:num_end],
                        density_center=density_center,
                        box_size=box_size,
                        cutoff_range=5,  # in standard deviations
                        sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution
                        apix = float(args.apix),
                        projection_frame=args.projection_frame,
                        projection_origin=args.projection_origin,
                    )
                    proj *= particle_sign

                    loss_frc = -compute_frc(
                        proj=proj.to(torch.float),
                        data=data[num_start:num_end].to(torch.float),
                        ctf=ctf[num_start:num_end].to(torch.float),
                        box_size=box_size,
                        max_freq=(2 * float(args.apix)) / target_resolution,
                        #apix=float(args.apix),
                        #shell_weight_freqs=shell_weight_freqs,
                        #shell_weights=shell_weights,
                    ) / data.shape[0]
                    losses += loss_frc.detach().to('cpu')
                    loss = loss_frc
                    penalty = gmm.regularization(limit)
                    penalties += penalty.detach().to('cpu')
                    loss += penalty
                    # loss += z_reg_weight * torch.mean(z_bias**2)
                    loss.backward()


                optimizer.step()
                global_step += 1

                print("peak memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
                optimizer.zero_grad()

                end = time.time()
                print('epoch',epoch,' batch num',num)
                print('frc_loss',losses)
                print('penalty',penalties)
                print('time',end-start)
                record.event('train_step', global_step=global_step, epoch=epoch, batch=num,
                             frc_loss=float(losses), penalty=float(penalties), total_loss=float(losses)+float(penalties),
                             learning_rates=[float(group['lr']) for group in optimizer.param_groups],
                             elapsed_seconds=end-start, n_particles=int(data.shape[0]),
                             peak_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if torch.device(device).type == 'cuda' else None,
                             memory_unavailable_reason=None if torch.device(device).type == 'cuda' else 'cpu_device',
                             peak_memory_scope='process_peak_since_last_external_reset',
                             rmsd=None, rmsd_unavailable_reason='not_measured',
                             loss_reduction='legacy_minibatch_sum; FRC divided by optimizer batch size')
                sys.stdout.flush()
                # Let an epoch-ending iterator finish normally so its RNG state
                # matches uninterrupted training; only break for a partial epoch.
                if max_steps is not None and global_step >= max_steps and num + 1 < len(dataloader):
                    break

                #if saved_vis >= n_save_vis:

        model_path = output_trained_model_dir+str(epoch+1)+'.pth'
        from checkpoint_sampling import sampling_snapshot, preserve_sampling
        from checkpoint_io import atomic_save_checkpoint
        export_sampling = sampling_snapshot(configs, device)
        with torch.no_grad(), preserve_sampling(configs):
            if use_block_decoder:
                pred_dict['coordinate'] = decode_block(z_bias)
            else:
                pred_dict['coordinate'] = _sample_diffusion(
                    configs=configs, training=False, denoise_net=diffusion_module,
                    input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s,
                    z_trunk=None if z is None else z + z_bias,
                    pair_z=pair_z + z_bias if z is None else pair_z,
                    p_lm=p_lm, c_l=c_l, N_sample=N_sample, noise_schedule=noise_schedule,
                    inplace_safe=False, enable_efficient_fusion=enable_efficient_fusion)
            atom_coord = (coordinate_transform(pred_dict['coordinate'][0]) if block_mode
                          else pred_dict['coordinate'][0] @ rotation.T + translation)
        epoch_complete = (num + 1 == len(dataloader))
        print('model_path',model_path)
        model_data = {
                    'checkpoint_schema':dict(name='cocofold2_refinement', version=1),
                    'training_progress':dict(epoch_completed=epoch+1 if epoch_complete else epoch, global_step=global_step, coordinate_step=global_step),
                    'training_resume':capture_epoch_state(args, configs, dataloader, restart_inputs, epoch, global_step, epoch_complete, record.run_id),
                    'export_sampling':export_sampling,
                    'refinement_sampler':sampler_name,
                    'gmm_learning_enabled':learn_gmm,
                    'structure_output_format':output_format,
                    'alignment_update_settings':dict(enabled=bool(flag_update_mat),
                        block_trace_threshold=getattr(args, 'block_update_trace_threshold', 2.5) if block_mode else None),
                    'refinement_seed_settings':dict(seed_settings),
                    'model_state':diffusion_module.state_dict(),
                    'opt_state':optimizer.state_dict(),
                    'atom_weights':atom_weights,
                    'sdevs':gmm.sdevs if gmm.kernel == 'legacy' else None,
                    'gmm':gmm.export_checkpoint(),
                    'gmm_kernel':gmm.kernel,
                    'projection_frame':args.projection_frame,
                    'projection_origin':tuple(args.projection_origin),
                    'rotation':rotation,
                    'translation':translation,
                    'enable_efficient_fusion':enable_efficient_fusion,
                    "pred_dict":pred_dict,
                    "input_feature_dict":input_feature_dict,
                    "s_inputs":s_inputs,
                    "s_trunk":s,
                    "z_trunk":z,
                    "pair_z":pair_z,
                    "p_lm":p_lm,
                    "c_l":c_l,
                    "N_sample":N_sample,
                    "noise_schedule":noise_schedule,
                    "inplace_safe":inplace_safe,
                    "configs":configs,
                    "s_inputs_bias":s_inputs_bias,
                    "s_bias":s_bias,
                    "z_bias":z_bias,
                    "z_mul":z_mul,
                    }
        if block_mode:
            model_data['refinement_sampler']=sampler_name
            model_data['coordinate_transform']=coordinate_transform.export_checkpoint(pred_dict['coordinate'][0])
            model_data['coordinate_mode']='blocks'
            model_data['coordinate_sampler']=(f'hetero_shared_noise_seed{seed_settings["diffusion_seed"]}'
                if use_block_decoder else f'global_fixed_noise_seed{seed_settings["diffusion_seed"]}')
        atomic_save_checkpoint(model_data, model_path)
        record.artifact(model_path, "checkpoint")
        record.event('checkpoint_saved', path=model_path, epoch_completed=epoch+1 if epoch_complete else epoch,
                     epoch_complete=epoch_complete, resume_supported=epoch_complete,
                     global_step=global_step, coordinate_step=global_step,
                     checkpoint_schema=model_data['checkpoint_schema'],
                     export_fixed_noise=export_sampling['fixed_noise'])

        del model_data
        torch.cuda.empty_cache()

        for output in export_training_structure(cif_path, output_trained_model_dir+str(epoch+1),
                                                atom_coord.detach().cpu().numpy(), output_format):
            record.artifact(output, 'structure')
        record.event("epoch_end" if epoch_complete else "epoch_stopped", epoch=epoch, global_step=global_step)
        if max_steps is not None and global_step >= max_steps:
            break


def build_parser():
    from cli_utils import positive_int, positive_float, finite_float, nonnegative_int
    from training_restart import RestartArgumentParser, add_restart_arguments
    parser = RestartArgumentParser(description="Particle-guided CoCoFold2 latent refinement.")
    add_restart_arguments(parser)
    from randomness import add_seed_arguments
    add_seed_arguments(parser, "train")
    add_record_arguments(parser)
    parser.add_argument("--output-format", "--output_format", choices=("cif", "pdb", "both"), default="cif",
                        help="Training structure format, including initial and epoch outputs (default: cif).")
    parser.add_argument("--learn-gmm", action=argparse.BooleanOptionalAction, default=True,
                        help="Learn both atom amplitudes and widths; --no-learn-gmm freezes both, retaining latent gradients.")
    parser.add_argument("--star_data_dir", required=True, help='RELION STAR file containing particle image references, poses and CTF metadata. Default: %(default)s.')
    parser.add_argument("--mrc_data_dir", default=None, help='Root for relative STAR image paths; omitted means the STAR directory. Absolute image paths are used directly. Default: %(default)s.')
    parser.add_argument("--output_trained_model_dir", required=True,
                        help="Legacy output filename prefix; a trailing slash means a directory.")
    parser.add_argument("--cif_path", required=True,
                        help="Initial model placed in the experimental coordinate frame.")
    parser.add_argument("--diffusion_data_dir", required=True, help='Diffusion .pth cache; use explicit resume/warm-start options for refinement checkpoints. Default: %(default)s.')
    parser.add_argument("--transR", action="store_true", default=False, help='Use the validated pose-convention matrix diag(1,1,-1); enable only for the matching upstream orientation convention. Default: %(default)s.')
    parser.add_argument("--particle_sign", default=-1., type=finite_float, help='Multiplier applied to rendered particle projections; keep the validated data sign convention. Default: %(default)s.')
    parser.add_argument("--boxsize", default=256, type=positive_int, help='Square particle image width/height in pixels; must match STAR/MRCS inputs. Default: %(default)s.')
    parser.add_argument("--apix", default=1., type=positive_float, help='Experimental pixel size in Angstrom per pixel; must be positive. Default: %(default)s.')
    parser.add_argument("--norm", action="store_true", default=False, help='Min-max normalize each observed particle to [0,1]; constant images are rejected. Default: %(default)s.')
    parser.add_argument("--resolution", default=3., type=positive_float, help='Legacy GMM coordinate/grid scale parameter; not generally a molmap resolution in Angstrom. Default: %(default)s.')
    parser.add_argument("--density_center", default=None, type=finite_float, nargs=2, help='Two image-center coordinates in pixels; omitted uses the box center. Default: %(default)s.')
    parser.add_argument('--projection-frame', choices=('legacy', 'fixed'), default='fixed',
                        help='fixed projects in one 3-D map frame (default); legacy dynamically recenters each GMM projection.')
    parser.add_argument('--projection-origin', type=finite_float, nargs=3, default=None,
                        metavar=('X_A', 'Y_A', 'Z_A'),
                        help='Required for fixed: 3-D reference point in the placed CIF/map frame, in Angstrom. No universal default.')
    parser.add_argument("--train_deterministic", "--train-deterministic",
                        dest="train_deterministic", action=argparse.BooleanOptionalAction, default=True, help='Reuse fixed diffusion stochasticity; disabling resamples noise. Per-chain placement requires fixed stochasticity. Default: %(default)s.')
    parser.add_argument("--device", default="cuda:0", help='PyTorch device for this operation; distributed CUDA ranks use LOCAL_RANK. Default: %(default)s.')
    parser.add_argument("--batch_size", default=32, type=positive_int, help='Particles per optimizer update; all parallel ranks process the same batch. Default: %(default)s.')
    parser.add_argument("--mini_batch_size", default=12, type=positive_int, help='Particles per loss/backward microbatch inside each update; smaller values trade memory for more decoding. Default: %(default)s.')
    parser.add_argument("--update_affine_mat", default=False, action="store_true", help='Enable the existing rigid-transform update safeguard; per-chain mode tests the shared threshold independently per chain. Default: %(default)s.')
    parser.add_argument("--map_resolution", default=5., type=positive_float, help='Angstrom resolution cutoff of the active particle FRC objective; not the GMM rendering width. Default: %(default)s.')
    parser.add_argument("--halfmap1", default=None, help='Optional first half-map file, paired with halfmap2. Weights are prepared but not consumed by the current active FRC loss. Default: %(default)s.')
    parser.add_argument("--halfmap2", default=None, help='Optional second half-map file, paired with halfmap1; current active FRC does not consume the prepared weights. Default: %(default)s.')
    parser.add_argument("--fsc_gamma", default=1., type=positive_float, help='Exponent for optional half-map weights; these weights currently do not alter the active FRC objective. Default: %(default)s.')
    parser.add_argument("--fsc_smooth_win", default=0, type=nonnegative_int, help='Nonnegative smoothing window for optional half-map weights; zero disables smoothing. Default: %(default)s.')
    parser.add_argument("--epochs", default=10, type=positive_int, help='Cumulative target epoch count; resume counts already completed epochs toward this target. Default: %(default)s.')
    parser.add_argument("--max_steps", "--max-steps", default=None, type=positive_int,
                        help="Total optimizer-step limit, including resumed steps; a partial-epoch save supports export/warm-start only.")
    parser.add_argument("--lr_bias", "--lr-bias", default=1e-2, type=positive_float, help='AdamW learning rate for the target-specific latent perturbation. Default: %(default)s.')
    parser.add_argument("--lr_atom_weights", "--lr-atom-weights", default=1e-2, type=positive_float, help='AdamW learning rate for GMM amplitudes; unused when GMM learning is disabled. Default: %(default)s.')
    parser.add_argument("--lr_sdevs", "--lr-sdevs", default=5e-3, type=positive_float, help='AdamW learning rate for GMM widths/shape parameters; unused when GMM learning is disabled. Default: %(default)s.')
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate inputs on CPU and exit without constructing the diffusion model.")
    add_gmm_arguments(parser)
    add_coordinate_arguments(parser)
    return parser


if __name__ == "__main__":
    original_argv = list(sys.argv)
    args = build_parser().parse_args()
    args.original_argv = original_argv
    print(args)
    main(args)
