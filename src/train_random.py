import torch
import argparse
import numpy as np
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import os
from typing import Any

from protenix.utils.torch_utils import autocasting_disable_decorator

from model.generator import (
    sample_diffusion
)

from protenix.model.modules.diffusion import DiffusionModule

from utils import cif_to_tensor, kabsch_alignment, compute_frc, replace_cif_coordinates, deep_clone, discrete_radon_transform_3d, translation_2d, compute_frc_simulate, compute_ncc_loss
from utils_halfmap import build_halfmap_shell_weights

from particledataset import ParticleDataset
from ctf import compute_ctf
from pts2img import pdb2img,sum_of_gaussians_2d_torch
from pts2img import translation_2d_robust as pts_translation_2d


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

def main(args):
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
    diffusion_data = torch.load(diffusion_data_dir,weights_only=False)

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
    configs.train_deterministic = args.train_deterministic
    print('train_deterministic',configs.train_deterministic)
    diffusion_module = DiffusionModule(**configs.model.diffusion_module).to(device)
    diffusion_module.load_state_dict(diffusion_data["model_state"])
    del diffusion_data
 
    #torch.manual_seed(42)
    #np.random.seed(42)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(42)
    
    for name, param in diffusion_module.named_parameters():
        param.requires_grad = False

    preds = []
    for i in range(2):
        with torch.no_grad():
            start = time.time()
            diffusion_module.eval()
            pred_dict["coordinate"] = _sample_diffusion(
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

    sdevs = torch.zeros(atom_weights.shape[0], 2)
    sdevs += 3/(np.pi * np.sqrt(2))
    sdevs = sdevs.to(torch.float).to(device)
    sdevs.requires_grad = True
    resolution = torch.tensor([float(args.resolution)]).to(torch.float).to(device)

    with torch.no_grad():
        _, rotation, translation = kabsch_alignment(pred_dict["coordinate"][0],ref_coords,return_transform=True)


    apix =float(args.apix)
    if args.transR :
        transR = np.array([[1,0,0],[0,1,0],[0,0,-1]]).reshape(3,3)
    else:
        transR = None
    dataset = ParticleDataset(str(args.star_data_dir),str(args.mrc_data_dir),apix,transR=transR,norm = args.norm)
    print(f'The dataset contains {len(dataset)} particles.')

    batch_size = int(args.batch_size)
    mini_batch_size = int(args.mini_batch_size)
    print('batch_size',batch_size,'mini_batch_size',mini_batch_size)

    lr_atoms_weights = 1e-2
    lr_sdevs = 5e-3
    lr_bias = 1e-2
    lr_mul = 2e-4
    z_reg_weight = 0

    # s_inputs_bias = torch.zeros_like(s_inputs,requires_grad=True).to(device)
    # s_bias = torch.zeros_like(s,requires_grad=True).to(device)
    s_inputs_bias = None
    s_bias = None
    '''
    if z is None:
        target = pair_z
    else:
        target = z
    
    z_mul = torch.nn.Parameter(
    torch.ones(
        (1,) * (target.ndim - 1) + (target.shape[-1],),
        device=target.device,
        dtype=target.dtype,
        )
    )

    z_bias = torch.nn.Parameter(
        torch.zeros(
            (1,) * (target.ndim - 1) + (target.shape[-1],),
            device=target.device,
            dtype=target.dtype,
        )
    )
    '''
    if z is None:
        # z_mul = torch.ones_like(pair_z,requires_grad=True).to(device)
        z_mul = None
        z_bias = torch.zeros_like(pair_z,requires_grad=True).to(device)
    else:
        # z_mul = torch.ones_like(z,requires_grad=True).to(device)
        z_mul = None
        z_bias = torch.zeros_like(z,requires_grad=True).to(device)

    # optimizer.add_param_group({'params': [s_inputs_bias,s_bias,z_bias], 'lr': lr_bias})
    optimizer = torch.optim.AdamW([z_bias],lr=lr_bias)
    # optimizer.add_param_group({'params': [z_mul], 'lr': lr_mul})
    optimizer.add_param_group({'params': atom_weights, 'lr': lr_atoms_weights})
    optimizer.add_param_group({'params': sdevs, 'lr': lr_sdevs})
    print('lr_atom_weight',lr_atoms_weights)
    print('lr_sdevs',lr_sdevs)
    print('lr_bias',lr_bias)
    print('z_reg_weight',z_reg_weight)
    if z_mul is not None:
        print('lr_mul',lr_mul)
    output_trained_model_dir = str(args.output_trained_model_dir)

    if not os.path.exists(output_trained_model_dir):
        os.makedirs(os.path.dirname(output_trained_model_dir), exist_ok=True)

    replace_cif_coordinates(
        input_cif = cif_path,
        output_cif = output_trained_model_dir+'_.pdb',
        new_coords = pred_dict["coordinate"][0].cpu().numpy(),
    )

    epochs = 10
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # diffusion_module.train()
    # torch.autograd.set_detect_anomaly(True)
    ncc_weight = 0  # 建议权重在 0.2 - 0.5 之间
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
        
    for epoch in range(epochs):

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

                    if z is None:
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

                    with torch.no_grad():
                        _, rotation, translation = kabsch_alignment(pred_dict["coordinate"][0],ref_coords,return_transform=True)

                    atom_coord = pred_dict["coordinate"][0] @ rotation.T + translation
                    proj = pdb2img(
                        atom_coord.reshape(1,-1,3),
                        float(args.resolution),
                        atom_weights,
                        R[num_start:num_end],
                        trans[num_start:num_end],
                        density_center=density_center,
                        box_size=box_size,
                        cutoff_range=5,  # in standard deviations
                        sigma_factor=1 / (np.pi * np.sqrt(2)),  # standard deviation / resolution
                        apix = float(args.apix),
                        sdevs = sdevs,
                        masks = None,
                        affine_matricies = None,
                    )
                    proj *= particle_sign
                    '''
                    if saved_vis < n_save_vis:
                        obs_batch = data[num_start:num_end]
                        pred_batch = proj

                        mb = num_end - num_start
                        n_to_save = min(n_save_vis - saved_vis, mb)

                        for j in range(n_to_save):
                            global_idx = num_start + j

                            obs_img, pred_img = normalize_pair_for_display(
                                obs_batch[j],
                                pred_batch[j],
                                p_low=1,
                                p_high=99
                            )
                            
                            save_gray_image(
                                obs_img,
                                vis_dir / f"particle_{global_idx:05d}_observed.png"
                            )
                            save_gray_image(
                                pred_img,
                                vis_dir / f"particle_{global_idx:05d}_predicted.png"
                            )

                            fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
                            axes[0].imshow(obs_img, cmap="gray",vmin=0, vmax=1)
                            axes[0].set_title("Observed", fontsize=8)
                            axes[0].axis("off")

                            axes[1].imshow(pred_img, cmap="gray",vmin=0, vmax=1)
                            axes[1].set_title("Predicted", fontsize=8)
                            axes[1].axis("off")

                            plt.tight_layout(pad=0.2)
                            fig.savefig(
                                vis_dir / f"particle_{global_idx:05d}_observed_vs_predicted.png",
                                bbox_inches="tight",
                                pad_inches=0.02,
                                transparent=True,
                            )
                            plt.close(fig)

                            saved_vis += 1

                            if saved_vis >= n_save_vis:
                                break
                    '''

                    loss_frc = -compute_frc(
                        proj=proj.to(torch.float),
                        data=data[num_start:num_end].to(torch.float),
                        ctf=ctf[num_start:num_end].to(torch.float),
                        box_size=box_size,
                        max_freq=(2 * float(args.apix)) / target_resolution,
                        # apix=float(args.apix),
                        # shell_weight_freqs=shell_weight_freqs,
                        # shell_weights=shell_weights,
                    ) / data.shape[0]
                    '''
                    proj_ft = torch.fft.fftshift(torch.fft.fft2(proj.to(torch.float)), dim=(-2, -1))
                    proj_ft_ctf = proj_ft * ctf[num_start:num_end].to(torch.float)

                    proj_ctf_real = torch.fft.ifft2(torch.fft.ifftshift(proj_ft_ctf, dim=(-2, -1))).real

                    loss_ncc = compute_ncc_loss(
                        proj=proj_ctf_real, 
                        data=data[num_start:num_end].to(torch.float)
                    ) / data.shape[0]


                    ncc_losses += loss_ncc.detach().to('cpu')

                    loss = frc_weight * loss_frc + ncc_weight * loss_ncc 
                    '''
                    losses += loss_frc.detach().to('cpu')
                    loss = loss_frc
                    penalty = torch.mean((torch.relu(sdevs.to(torch.float) - limit[1])) + 
                                            torch.relu(limit[0] - sdevs.to(torch.float))) + torch.mean((torch.relu(atom_weights.to(torch.float) - limit[3])) + torch.relu(limit[2] - atom_weights.to(torch.float)))
                    penalties += penalty.detach().to('cpu')
                    loss += penalty
                    loss += z_reg_weight * torch.mean(z_bias**2)
                    loss.backward()

                
                optimizer.step()
                
                print("peak memory:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
                optimizer.zero_grad()

                end = time.time()
                print('epoch',epoch,' batch num',num)
                print('frc_loss',losses)
                # print('ncc_loss',ncc_losses)
                print('penalty',penalties)
                print('time',end-start)
                sys.stdout.flush()  

                #if saved_vis >= n_save_vis:
                #    break

        model_path = output_trained_model_dir+str(epoch+1)+'.pth'
        print('model_path',model_path)
        model_data = {
                    'model_state':diffusion_module.state_dict(),
                    'opt_state':optimizer.state_dict(),
                    'atom_weights':atom_weights,
                    'sdevs':sdevs,
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
        model_data_clone = deep_clone(model_data)
        torch.save(model_data_clone, model_path)

        del model_data
        del model_data_clone
        torch.cuda.empty_cache()

        replace_cif_coordinates(
            input_cif = cif_path,
            output_cif = output_trained_model_dir+str(epoch+1)+'.pdb',
            new_coords = atom_coord.detach().cpu().numpy(),
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--star_data_dir", 
    )
    parser.add_argument(
        "--mrc_data_dir", 
    )
    parser.add_argument(
        "--output_trained_model_dir", 
    )
    parser.add_argument(
        "--cif_path", 
    )
    parser.add_argument(
        "--diffusion_data_dir", 
    )
    parser.add_argument(
        "--transR", action="store_true", default=False
    )
    parser.add_argument(
        "--particle_sign", default=-1,
    )    
    parser.add_argument(
        "--boxsize", default=256,
    )
    parser.add_argument(
        "--apix", default=1,
    )
    parser.add_argument(
        "--norm",action="store_true", default=False
    )
    parser.add_argument(
        "--resolution",default=3
    )
    parser.add_argument(
        "--density_center",default=None,type=float,nargs=2
    )
    parser.add_argument(
        "--train_deterministic",action="store_true", default=False
    )
    parser.add_argument(
        "--device", default="cuda:0"
    )
    parser.add_argument(
        "--batch_size", default=32
    )
    parser.add_argument(
        "--mini_batch_size", default=12
    )
    parser.add_argument(
        "--update_affine_mat", default=False,action="store_true"
    )
    parser.add_argument(
        "--map_resolution",default=5
    )
    parser.add_argument("--halfmap1", default=None, type=str)
    parser.add_argument("--halfmap2", default=None, type=str)
    parser.add_argument("--fsc_gamma", default=1.0, type=float)
    parser.add_argument("--fsc_smooth_win", default=0, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
