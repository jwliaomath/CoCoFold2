"""Two actual Gloo processes; no Protenix or multi-GPU setup required.

Run from the repository root: python tests/check_gmm_distributed.py
Checks unequal component sizes, assembled images, and the loss/world_size
gradient convention used by train_chain_parallel_2d.py.
"""
from datetime import timedelta
import json
import math
from pathlib import Path
import tempfile
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_gmm import GaussianProjector, OLD_DIST, distributed_project_gaussians, image_objective


def worker(rank, rendezvous, report_dir):
    torch.set_num_threads(2)
    warnings.filterwarnings("ignore", message=".*grid_sample.*")
    warnings.filterwarnings("ignore", message=".*meshgrid.*")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        dtype = torch.float64
        xyz = torch.tensor([[-2.1, -.7, .1], [.8, -1.3, .4], [1.9, 2.2, -1.2],
                            [-.9, 1.1, .7], [1.4, -.3, -.8]], dtype=dtype)
        rotation = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                                 [[math.cos(.53), 0., math.sin(.53)],
                                  [0., 1., 0.]]], dtype=dtype)
        trans = torch.tensor([[.1, -.3], [-.2, .4]], dtype=dtype)
        center = torch.tensor([16., 16.], dtype=dtype)
        weights = torch.tensor([.5, 8., 22., 6., 7.], dtype=dtype)
        part = slice(0, 2) if rank == 0 else slice(2, 5)
        records = []
        configurations = [(mode, "auto") for mode in GaussianProjector.MODES]
        configurations += [(mode, "peak_3d") for mode in ("isotropic", "anisotropic")]
        for mode, amplitude in configurations:
            full = GaussianProjector(weights, mode, amplitude_convention=amplitude, atom_chunk_size=2)
            with torch.no_grad():
                shape = full.shape_parameters()[0]
                shape[0, 0] = .09 if mode == "legacy" else -3.
                shape[2, 0] = .91 if mode == "legacy" else .8
                if mode == "anisotropic":
                    shape[:, 3:] = torch.tensor([.13, -.08, .11], dtype=dtype)
            local = GaussianProjector(weights[part], mode, amplitude_convention=amplitude, atom_chunk_size=2)
            with torch.no_grad():
                local.shape_parameters()[0].copy_(full.shape_parameters()[0][part])
            x_full = xyz.clone().requires_grad_()
            x_local = xyz[part].clone().requires_grad_()
            expected = full(x_full, rotation, trans, 3.6, center, box_size=32, apix=1.2)
            actual = distributed_project_gaussians(local, x_local, rotation, trans,
                                                   center, 3.6, 32, 1.2)
            counts = torch.tensor(local.regularization_counts(), dtype=torch.long)
            dist.all_reduce(counts)
            assert tuple(counts.tolist()) == full.regularization_counts()
            local_penalty = local.regularization(global_atom_count=int(counts[0]),
                                                  global_width_count=int(counts[1]))
            summed_penalty = local_penalty.detach().clone()
            dist.all_reduce(summed_penalty)
            torch.testing.assert_close(summed_penalty, full.regularization(), rtol=1e-6, atol=1e-7)

            # Legacy truncates at a chunk-dependent rectangular bounding box;
            # splitting atoms can therefore change tiny tails. New covariance
            # kernels use per-atom support and should agree to round-off.
            rtol, atol = (3e-4, 3e-6) if mode == "legacy" else (1e-8, 1e-10)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            (image_objective(actual) / 2 + local_penalty).backward()
            (image_objective(expected) + full.regularization()).backward()
            errors = {}
            pairs = [("coordinates", x_local.grad, x_full.grad[part]),
                     ("amplitudes", local.atom_weights.grad, full.atom_weights.grad[part]),
                     ("shape", local.shape_parameters()[0].grad, full.shape_parameters()[0].grad[part])]
            for name, actual_grad, expected_grad in pairs:
                torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)
                errors[name] = float((actual_grad - expected_grad).abs().max())
            if mode == "legacy":
                # Stronger regression: exact same distributed partition against
                # the byte-for-byte original distributed renderer fixture.
                old_x = xyz[part].clone().requires_grad_()
                old_w = local.atom_weights.detach().clone().requires_grad_()
                old_s = local.sdevs.detach().clone().requires_grad_()
                original = OLD_DIST.distributed_pdb2img(old_x, old_w, old_s, rotation,
                                                        trans, center, 3.6, 32, 1.2)
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
                old_penalty = OLD_DIST.local_source_equivalent_penalty(old_w, old_s, 5, 10)
                (image_objective(original) / 2 + old_penalty).backward()
                for new_grad, old_grad in ((x_local.grad, old_x.grad),
                                           (local.atom_weights.grad, old_w.grad),
                                           (local.sdevs.grad, old_s.grad)):
                    torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)
            records.append({"kernel": mode, "amplitude": full.amplitude_convention, "rank": rank,
                            "image_max_abs_error": float((actual - expected).abs().max()),
                            "gradient_max_abs_errors": errors,
                            "legacy_original_exact": mode == "legacy"})
        Path(report_dir, f"rank_{rank}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Some Windows PyTorch builds cannot open FileStore under a non-ASCII user
    # temp path. Keep this self-cleaning, bounded test directory next to the test.
    with tempfile.TemporaryDirectory(prefix="gmm_gloo_", dir=Path(__file__).resolve().parent) as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        mp.spawn(worker, args=(rendezvous, directory), nprocs=2, join=True)
        for rank in range(2):
            print(Path(directory, f"rank_{rank}.json").read_text(encoding="utf-8"))
    print("PASS: two-process Gloo projection and gradients for all three kernels, including both peak_3d modes")
