# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from run_recording import recorded, current_record, add_record_arguments, RecordingError
import logging
import os
import time
import traceback
import urllib.request
from argparse import Namespace
from contextlib import nullcontext
from os.path import exists as opexists, join as opjoin
from typing import Any, Mapping

import torch
import torch.distributed as dist

import argparse
import sys

logger = logging.getLogger(__name__)

def _load_runtime_imports():
    # Delay Protenix/config/kernel imports until basic input checks have passed.
    global configs_base, data_configs, inference_configs, model_configs
    global parse_configs, parse_sys_args, get_inference_dataloader, Protenix
    global DIST_WRAPPER, seed_everything, to_device, URL, DataDumper
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs, parse_sys_args
    from protenix.data.inference.infer_dataloader import get_inference_dataloader
    from model.protenix import Protenix
    from protenix.utils.distributed import DIST_WRAPPER
    from protenix.utils.seed import seed_everything
    from protenix.utils.torch_utils import to_device
    from protenix.web_service.dependency_url import URL
    from runner.dumper import DataDumper
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Namespace])


class InferenceRunner(object):
    """
    Runner class for AlphaFold3 model inference.
    Handles environment setup, model initialization, and running predictions.

    Args:
        configs (Any): Configuration object for inference.
    """

    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")

        if self.configs.triangle_attention == "deepspeed":
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert env is not None, (
                "If use deepspeed (ds4sci), set CUTLASS_PATH environment variable "
                "per instructions at "
                "https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            )
            logging.info(
                "Kernels will be compiled when DS4Sci_EvoformerAttention "
                "is first called."
            )

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "fast_layernorm")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Finished environment initialization.")

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """
        Initialize the Protenix model and move it to the appropriate device.
        """
        self.model = Protenix(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        """
        Load model weights from a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
        """
        checkpoint_path = opjoin(
            self.configs.load_checkpoint_dir, f"{self.configs.model_name}.pt"
        )
        if not opexists(checkpoint_path):
            raise FileNotFoundError(
                f"Given checkpoint path not exist [{checkpoint_path}]"
            )

        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = list(checkpoint["model"].keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model parameters: {count_parameters(self.model):.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        """
        Initialize the data dumper for saving predictions.

        Args:
            need_atom_confidence (bool): Whether to dump atom-level confidence.
            sorted_by_ranking_score (bool): Whether to sort results by ranking score.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    # Adapted from runner.train.AF3Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Run model prediction on the provided data.

        Args:
            data (Mapping[str, Mapping[str, Any]]): Input data dictionary.

        Returns:
            dict[str, torch.Tensor]: Prediction results.
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                mc_dropout_apply_rate=self.configs.mc_dropout_apply_rate,
            )

        return prediction

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        """
        Update the model's configuration.

        Args:
            new_configs (Any): New configuration object.
        """
        self.model.configs = new_configs


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for tracking download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    status = f"\r[{bar}] {percent:.1f}%"
    print(status, end="", flush=True)

    if downloaded >= total_size:
        print()


def download_from_url(
    tos_url: str, checkpoint_path: str, check_weight: bool = True
) -> None:
    """Internal helper to download from URL and verify weight files."""
    urllib.request.urlretrieve(tos_url, checkpoint_path, reporthook=progress_callback)
    if check_weight:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            del ckpt
        except Exception as e:
            if opexists(checkpoint_path):
                os.remove(checkpoint_path)
            raise RuntimeError(
                f"Download model checkpoint failed: {e}. Please download "
                f"manually with: wget {tos_url} -O {checkpoint_path}"
            ) from e


def download_inference_cache(configs: Any) -> None:
    """
    Download necessary data and model checkpoints for inference.

    Args:
        configs (Any): Configuration object containing paths and model names.
    """

    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
        "obsolete_release_data_csv",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath) or '.', exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    if configs.use_template:
        for cache_name in (
            "obsolete_pdbs_path",
            "release_dates_path",
        ):
            cur_cache_fpath = configs["data"]["template"][cache_name]
            if not opexists(cur_cache_fpath):
                os.makedirs(os.path.dirname(cur_cache_fpath) or '.', exist_ok=True)
                tos_url = URL[cache_name]
                assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                    f"{cache_name} file name is incorrect, `{tos_url}` and "
                    f"`{cur_cache_fpath}`. Please check and try again."
                )
                logger.info(
                    f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
                )
                download_from_url(tos_url, cur_cache_fpath, check_weight=False)
            else:
                logger.info(f"{cache_name} already exists at {cur_cache_fpath}")

    checkpoint_path = f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"
    checkpoint_dir = configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}...\n to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)

    if "esm" in configs.model_name:  # currently esm only support 3b model
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path)
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ckpt_path2)
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"

        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path)

        esm_3b_ism_ckpt_path2 = (
            f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"
        )
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            logger.info(
                f"Downloading model checkpoint from\n {tos_url}...\n to {esm_3b_ism_ckpt_path2}"
            )
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)


def update_inference_configs(configs: Any, n_token: int) -> Any:
    """
    Adjust inference configurations based on the number of tokens to avoid OOM.

    Args:
        configs (Any): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        Any: Updated configurations.
    """
    # Adjust configurations based on sequence length to manage memory usage
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    return configs


def _prepare_inference(configs):
    from cli_utils import check_output_directory
    from inference_io import read_inputs, cache_plan
    targets = read_inputs(configs.input_json_path)
    # Preserve the existing convention: the first target supplies JSON seeds.
    seeds = (targets[0].get('modelSeeds') if configs.use_seeds_in_json else None) or configs.seeds
    seeds = list(seeds)
    planned = cache_plan(configs, targets, seeds)
    check_output_directory(configs.dump_dir)
    return targets, seeds, planned


def infer_predict(runner: InferenceRunner, configs: Any, prepared=None) -> None:
    """Continue independent target/seed jobs, then fail if any job failed."""
    from pathlib import Path
    from inference_io import safe_name
    targets, seeds, planned = prepared if prepared is not None else _prepare_inference(configs)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()  # Every rank finishes preflight before any rank writes caches.
    expected = {(target['name'], seed) for target in targets for seed in seeds}
    outcomes = []
    rank = DIST_WRAPPER.rank

    def failure(sample, seed, exc, stage):
        message = f'{type(exc).__name__}: {exc}'
        outcomes.append(dict(sample_name=sample, seed=seed, status='failed',
                             stage=stage, error=message, rank=rank))
        logger.exception('Inference failed: target=%s seed=%s stage=%s', sample, seed, stage)
        # Use rank and outcome index, never an unchecked target name as a path.
        Path(runner.error_dir).mkdir(parents=True, exist_ok=True)
        with open(opjoin(runner.error_dir, f'rank_{rank}_errors.txt'), 'a', encoding='utf-8') as handle:
            handle.write(f'target={sample!r} seed={seed} stage={stage}: {message}\n')
            handle.write(traceback.format_exc() + '\n')

    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as exc:
        failure(None, None, exc, 'dataloader_init')
        dataloader = None

    if dataloader is not None:
        for seed in seeds:
            try:
                seed_everything(seed=seed, deterministic=configs.deterministic)
                for batch in dataloader:
                    sample_name = None
                    started = time.monotonic()
                    try:
                        data, atom_array, data_error_message = batch[0]
                        sample_name = safe_name(data['sample_name'])
                        if (sample_name, seed) not in expected:
                            raise ValueError(f'Unexpected dataloader target: {sample_name!r}')
                        if data_error_message:
                            raise ValueError(data_error_message)
                        new_configs = update_inference_configs(configs, data['N_token'].item())
                        cache = planned.get((sample_name, seed))
                        new_configs.cache_output_path = str(cache) if cache is not None else None
                        current_record().resolved(f"target_{sample_name}_seed_{seed}", new_configs, "Existing token-count-specific runtime configuration")
                        runner.update_model_configs(new_configs)
                        prediction = runner.predict(data)
                        runner.dumper.dump(
                            dataset_name='', pdb_id=sample_name, seed=seed,
                            pred_dict=prediction, atom_array=atom_array,
                            entity_poly_type={k: v for k, v in data['entity_poly_type'].items()
                                              if v != 'non-polymer'},
                        )
                        outcomes.append(dict(sample_name=sample_name, seed=seed, status='success',
                                             rank=rank, elapsed_seconds=time.monotonic() - started))
                        logger.info('Target %s seed %s succeeded.', sample_name, seed)
                    except RecordingError:
                        raise
                    except Exception as exc:
                        failure(sample_name, seed, exc, 'target')
                    finally:
                        torch.cuda.empty_cache()
            except RecordingError:
                raise
            except Exception as exc:
                # A broken iterator cannot safely resume; other seeds still run.
                failure(None, seed, exc, 'dataloader_iteration')

    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, outcomes)
        outcomes = [row for rows in gathered for row in rows]
    seen = {(row['sample_name'], row['seed']) for row in outcomes}
    for sample_name, seed in sorted(expected - seen):
        outcomes.append(dict(sample_name=sample_name, seed=seed, status='failed',
                             stage='not_processed', error='Target was not returned by the dataloader'))
    for row in outcomes:
        cache = planned.get((row['sample_name'], row['seed']))
        row['cache_path'] = str(cache) if cache is not None else None
        row['cache_exists'] = cache.is_file() if cache is not None else False
    failed = sum(row['status'] == 'failed' and (row['sample_name'], row['seed']) in expected for row in outcomes)
    execution_errors = sum(row['status'] == 'failed' and (row['sample_name'], row['seed']) not in expected for row in outcomes)
    summary = dict(expected_jobs=len(expected), succeeded=sum(row['status'] == 'success' for row in outcomes),
                   failed=failed, execution_errors=execution_errors, outcomes=outcomes)
    current_record().resolved('prediction_outcomes', summary, 'Collected per-target/seed outcomes including failures')
    for row in outcomes:
        current_record().event('prediction_result', **row)
    summary_path = Path(configs.dump_dir) / 'inference_summary.json'
    if rank == 0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        current_record().artifact(summary_path, 'inference_summary')
        for row in outcomes:
            if row['cache_exists']:
                current_record().artifact(row['cache_path'], 'diffusion_cache')
        logger.info('Inference summary: %s', summary_path)
    if failed or execution_errors:
        raise RuntimeError(f'Inference had {failed} target failure(s) and {execution_errors} execution error(s); see {summary_path}')
    return summary


def main(configs: Any, prepared=None) -> None:
    """
    Inference entry point.

    Args:
        configs (Any): Inference configurations.
    """
    prepared = prepared if prepared is not None else _prepare_inference(configs)
    runner = InferenceRunner(configs)
    infer_predict(runner, configs, prepared=prepared)


def update_gpu_compatible_configs(configs: Any) -> Any:
    """
    Update configurations to ensure compatibility with specific GPU architectures (e.g., V100).

    Args:
        configs (Any): Original configurations.

    Returns:
        Any: Updated configurations.
    """

    def is_gpu_capability_between_7_and_8() -> bool:
        # Check if 7.0 <= device_capability < 8.0
        if not torch.cuda.is_available():
            return False
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cc = major + minor / 10.0
        return 7.0 <= cc < 8.0

    if is_gpu_capability_between_7_and_8():
        # V100 and similar architectures don't support some kernels or BF16 effectively
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "Enforcing FP32 and torch kernels for compatibility with detected "
            "GPU (Compute Capability 7.x)."
        )
    return configs


@recorded("inference")
def run(args) -> None:
    """
    Initialize and execute the inference pipeline.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    from randomness import resolve_seeds, apply_seed_settings
    seed_settings = resolve_seeds(args, "inference")
    from inference_io import read_inputs, safe_name
    targets = read_inputs(args.input_json_path)
    if args.sample_name is not None:
        safe_name(args.sample_name)
    if getattr(args, 'check_inputs', False):
        report = dict(targets=[t['name'] for t in targets],
                      scope='basic JSON structure and target names only; model runtime and external MSA/template validation not run')
        current_record().resolved("preflight", report, "Explicitly recorded basic JSON check; no model execution")
        print(json.dumps(report))
        return report
    _load_runtime_imports()
    arg_str = parse_sys_args()

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    # 1. First pass to get model_name
    configs = parse_configs(
        configs=configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # 2. Get model specifics and merge into base defaults
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    if model_name not in model_configs:
        raise ValueError(f'Unknown model_name {model_name!r}')
    record = current_record()
    record.resolved("base_defaults", base_configs, "Existing base/data/inference defaults before model merge")
    model_specfics_configs = model_configs[model_name]
    record.resolved("model_overrides", model_specfics_configs, "Selected model-specific configuration")

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base_configs, model_specfics_configs)

    # 3. Second pass to apply sys_args with higher priority
    configs = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=True,
    )

    record.resolved("protenix_cli", configs, "Existing model/default merge followed by forwarded Protenix CLI arguments")
    configs.sample_diffusion = {
    "gamma0": float(args.gamma0),
    "gamma_min": float(args.gamma_min),
    "noise_scale_lambda": float(args.noise_scale_lambda),
    "step_scale_eta": float(args.step_scale_eta),
    "N_step": int(args.N_step),
    "N_sample": int(args.N_sample),
    "N_step_mini_rollout": int(args.N_step_mini_rollout),
    "N_sample_mini_rollout": int(args.N_sample_mini_rollout)
    }
    print('Original N_cycle',configs.model.N_cycle)
    print('enable_diffusion_shared_vars_cache',configs.enable_diffusion_shared_vars_cache)
    # configs.model.N_cycle = 10
    configs.train_deterministic = args.train_deterministic
    apply_seed_settings(configs, seed_settings)
    configs.input_json_path= str(args.input_json_path)
    configs.dump_dir= str(args.dump_dir)
    configs.save_pairformer_last_input = args.save_pairformer_last_input
    if args.output_model_dir is not None:
        configs.output_model_dir = str(args.output_model_dir)
    else:
        configs.output_model_dir = None
    configs.sample_name = args.sample_name

    record.resolved("cocofold_cli", dict(configs=configs, seeds=seed_settings), "Existing CoCoFold CLI replaces sample_diffusion; seed settings applied without changing prediction seeds")
    from inference_io import resolve_resource_paths
    resolve_resource_paths(configs, getattr(args, 'resource_root', None), getattr(args, 'protenix_args', []))
    logger.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    _, model_size, model_feature, model_version = model_name.split("_")
    logger.info(
        f"Inference by Protenix: model_size: {model_size}, "
        f"with_feature: {model_feature.replace('-',', ')}, "
        f"model_version: {model_version}, dtype: {configs.dtype}"
    )
    record.resolved("resource_paths", configs, "Explicit resource paths take precedence over resource_root/cwd defaults")
    configs = update_gpu_compatible_configs(configs)
    record.resolved("gpu_compatible", configs, "Existing hardware compatibility overrides")
    record.refresh_environment()
    logger.info(
        f"Triangle kernels: multiplicative={configs.triangle_multiplicative}, "
        f"attention={configs.triangle_attention}"
    )
    logger.info(
        f"Optimization: shared_vars_cache={configs.enable_diffusion_shared_vars_cache}, "
        f"efficient_fusion={configs.enable_efficient_fusion}, tf32={configs.enable_tf32}"
    )
    # Resolve all cache destinations before downloads/model construction.
    prepared = _prepare_inference(configs)
    record.resolved("prediction_plan", dict(targets=[t["name"] for t in prepared[0]], prediction_seeds=prepared[1],
                    caches=[dict(target=k[0], seed=k[1], path=str(v)) for k,v in prepared[2].items()]),
                    "Existing first-target JSON modelSeeds convention, otherwise Protenix configured seeds")
    download_inference_cache(configs)
    main(configs, prepared=prepared)


def build_parser():
    from cli_utils import positive_int, positive_float, nonnegative_float, boolean
    parser = argparse.ArgumentParser(description="Protenix initial predictions and CoCoFold2 caches.")
    from randomness import add_seed_arguments
    add_seed_arguments(parser, "inference")
    add_record_arguments(parser)
    parser.add_argument("--input_json_path", required=True, help='Protenix target JSON; may contain multiple targets. Relative CLI paths use the working directory. Default: %(default)s.')
    parser.add_argument("--sample_name", default=None,
                        help="Legacy cache name for one target and one seed; otherwise use actual target names.")
    parser.add_argument("--train-deterministic", "--train_deterministic",
                        action=argparse.BooleanOptionalAction, default=True, help='Reuse fixed diffusion stochasticity; disabling resamples noise. Default: %(default)s.')
    parser.add_argument("--output_model_dir", default=None,
                        help="Cache directory; trailing slash is optional. Existing caches are never overwritten.")
    parser.add_argument("--dump_dir", default='./output', help='Protenix prediction and inference-record output directory; relative to the working directory. Default: %(default)s.')
    parser.add_argument("--gamma0", default=0., type=nonnegative_float, help='Diffusion churn magnitude stored in the cache sampling configuration. Default: %(default)s.')
    parser.add_argument("--gamma_min", default=0., type=nonnegative_float, help='Noise-level threshold controlling diffusion churn. Default: %(default)s.')
    parser.add_argument("--noise_scale_lambda", default=1.003, type=nonnegative_float, help='Multiplier for diffusion noise injection. Default: %(default)s.')
    parser.add_argument("--step_scale_eta", default=1., type=positive_float, help='Scale of each diffusion integration update. Default: %(default)s.')
    parser.add_argument("--N_step", default=5, type=positive_int, help='Number of denoising integration steps saved in the diffusion sampling configuration. Default: %(default)s.')
    parser.add_argument("--N_sample", default=1, type=positive_int, help='Number of structure samples per target/seed; refinement requires a compatible single-structure cache. Default: %(default)s.')
    parser.add_argument("--N_step_mini_rollout", default=5, type=positive_int, help='Denoising steps for the saved mini-rollout configuration. Default: %(default)s.')
    parser.add_argument("--N_sample_mini_rollout", default=5, type=positive_int, help='Structure samples for the saved mini-rollout configuration; separate from particle mini-batches. Default: %(default)s.')
    parser.add_argument("--save_pairformer_last_input", default=False, type=boolean, nargs='?', const=True, help='Save the final Pairformer input for downstream cache preparation; optional explicit boolean. Default: %(default)s.')
    parser.add_argument("--resource-root", default=None,
                        help="Default checkpoint/common root; explicit Protenix paths take precedence.")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check basic JSON structure/names without Protenix or downloads; excludes external MSA/template validation.")
    return parser


if __name__ == "__main__":
    original_argv = list(sys.argv)
    args, leftovers = build_parser().parse_known_args()
    args.original_argv = original_argv
    args.protenix_args = list(leftovers)
    sys.argv = [sys.argv[0]] + leftovers
    run(args)
