import torch
import argparse
import os
import numpy as np
from typing import Any

from protenix.utils.torch_utils import autocasting_disable_decorator
from model.generator import sample_diffusion
from protenix.model.modules.diffusion import DiffusionModule
from utils import replace_cif_coordinates

def _sample_diffusion(configs, training=False, **kwargs: Any) -> torch.Tensor:
    """
    Samples diffusion process based on the provided configurations.
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
    )(configs=configs, **_configs, **kwargs)

def main(args):
    device = args.device
    diffusion_data_dir = args.diffusion_data_dir
    out_dir = args.out_dir
    cif_path = args.cif_path

    print(f"Loading diffusion data from: {diffusion_data_dir}")
    diffusion_data = torch.load(diffusion_data_dir, weights_only=False)

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
    configs.train_deterministic = True

    print("Initializing Diffusion Module...")
    diffusion_module = DiffusionModule(**configs.model.diffusion_module).to(device)
    diffusion_module.load_state_dict(diffusion_data["model_state"])
    del diffusion_data

    # 锁定随机种子，保证预测的绝对可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    diffusion_module.eval()
    
    print("Running initial diffusion sampling...")
    with torch.no_grad():
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

    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)
    out_pdb_path = os.path.join(out_dir, f"{args.pdbid}_initial_prediction.pdb")

    print(f"Replacing coordinates into reference template...")
    replace_cif_coordinates(
        input_cif=cif_path,
        output_cif=out_pdb_path,
        new_coords=pred_dict["coordinate"][0].cpu().numpy(),
    )
    
    print(f"✅ Prediction successfully saved to: {out_pdb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbid", required=True, help="Target PDB ID")
    parser.add_argument("--diffusion_data_dir", required=True, help="Path to the diffusion .pth data")
    parser.add_argument("--cif_path", required=True, help="Path to reference cif/pdb file to borrow topology")
    parser.add_argument("--out_dir", required=True, help="Directory to save the prediction")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()
    main(args)