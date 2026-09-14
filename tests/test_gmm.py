"""Run with: python -m unittest discover -s tests -p test_gmm.py -v

No Protenix weights or particle dataset needed. Original source fixtures are
the hash-verified pre-change versions, not rewritten reference implementations.
"""
import argparse
import ast
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest
import warnings

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/chain_parallel"))
from gmm import GaussianProjector, add_gmm_arguments
from pts2img import project_gaussian_covariances, sum_of_gaussians_2d_covariance
from distributed_gmm import distributed_project_gaussians


def load_reference(name):
    path = Path(__file__).parent / "reference" / (name + "_legacy.py")
    spec = importlib.util.spec_from_file_location(name + "_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = load_reference("pts2img")
OLD_DIST = load_reference("distributed_gmm")
torch.set_num_threads(2)


def scene(dtype=torch.float64, device="cpu"):
    xyz = torch.tensor([[-2.1, -.7, .1], [.8, -1.3, .4], [1.9, 2.2, -1.2]],
                       dtype=dtype, device=device)
    a = .63
    rotation = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                             [[math.cos(a), -math.sin(a), 0.],
                              [math.sin(a), math.cos(a), 0.]]], dtype=dtype, device=device)
    trans = torch.tensor([[.1, -.3], [-.2, .4]], dtype=dtype, device=device)
    center = torch.tensor([12., 12.], dtype=dtype, device=device)
    return xyz, rotation, trans, center


def image_objective(image):
    """Differentiable nonuniform Fourier objective, including a synthetic CTF."""
    yy, xx = torch.meshgrid(torch.arange(image.shape[-2], device=image.device),
                            torch.arange(image.shape[-1], device=image.device), indexing="ij")
    ctf = torch.cos(.017 * (xx.square() + yy.square())).to(image.dtype)
    target = torch.sin(.23 * xx + .11 * yy).to(image.dtype)
    residual = torch.fft.fft2(image) * ctf - torch.fft.fft2(target)
    # real^2 + imag^2 avoids the complex-abs Jiterator dependency of some
    # Windows PyTorch distributions while retaining the same Fourier objective.
    return (residual.real.square() + residual.imag.square()).mean() / 100


class GaussianTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message=".*grid_sample.*")
        warnings.filterwarnings("ignore", message=".*meshgrid.*")

    def test_all_original_renderer_functions_unchanged(self):
        def implementation(node):
            # B9 removes standalone string-literal comment blocks. Preserve real
            # docstrings and every executable expression, call and default.
            for parent in ast.walk(node):
                for field, items in ast.iter_fields(parent):
                    if not isinstance(items, list):
                        continue
                    keep = []
                    for index, item in enumerate(items):
                        inert = (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                                 and isinstance(item.value.value, str))
                        docstring = (field == 'body' and index == 0 and
                                     isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
                        if not inert or docstring:
                            keep.append(item)
                    setattr(parent, field, keep)
            return ast.dump(node, include_attributes=False)

        old = ast.parse((Path(__file__).parent / "reference/pts2img_legacy.py").read_text(encoding="utf-8"))
        new = ast.parse((ROOT / "src/pts2img.py").read_text(encoding="utf-8"))
        functions = {n.name: implementation(n)
                     for n in new.body if isinstance(n, ast.FunctionDef)}
        for node in old.body:
            if isinstance(node, ast.FunctionDef):
                self.assertEqual(implementation(node), functions[node.name], node.name)

    def test_legacy_projection_penalty_gradients_and_adamw_update(self):
        xyz, rotation, trans, center = scene()
        x_old, x_new = [xyz.clone().requires_grad_() for _ in range(2)]
        weights = torch.tensor([.5, 6., 21.], dtype=xyz.dtype, requires_grad=True)
        widths = torch.tensor([[.09, .6], [.65, .9], [.7, .6]], dtype=xyz.dtype, requires_grad=True)
        new = GaussianProjector(weights, "legacy")
        with torch.no_grad():
            new.sdevs.copy_(widths)
        original = OLD.pdb2img(x_old[None], 3., weights, rotation, trans, center,
                              box_size=24, sdevs=widths)
        actual = new(x_new[None], rotation, trans, 3., center, box_size=24)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        old_penalty = (torch.relu(widths.float() - .8) + torch.relu(.1 - widths.float())).mean()
        old_penalty = old_penalty + (torch.relu(weights.float() - 20) + torch.relu(1 - weights.float())).mean()
        torch.testing.assert_close(new.regularization(), old_penalty, rtol=0, atol=0)
        (image_objective(original) + old_penalty).backward()
        (image_objective(actual) + new.regularization()).backward()
        for a, b in [(x_new.grad, x_old.grad), (new.atom_weights.grad, weights.grad),
                     (new.sdevs.grad, widths.grad)]:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        old_opt = torch.optim.AdamW([{"params": [weights], "lr": .01},
                                      {"params": [widths], "lr": .005}])
        new_opt = torch.optim.AdamW([{"params": new.amplitude_parameters(), "lr": .01},
                                      {"params": new.shape_parameters(), "lr": .005}])
        old_opt.step()
        new_opt.step()
        torch.testing.assert_close(new.atom_weights, weights, rtol=0, atol=0)
        torch.testing.assert_close(new.sdevs, widths, rtol=0, atol=0)

    def test_legacy_distributed_wrapper_and_penalty(self):
        xyz, rotation, trans, center = scene()
        g = GaussianProjector(torch.tensor([.5, 8., 22.], dtype=xyz.dtype))
        with torch.no_grad():
            g.sdevs[0, 0] = .09
            g.sdevs[2, 1] = .91
        old = OLD_DIST.distributed_pdb2img(xyz, g.atom_weights, g.sdevs, rotation,
                                          trans, center, 3., 24, 1.)
        new = distributed_project_gaussians(g, xyz, rotation, trans, center, 3., 24, 1.)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        old_reg = OLD_DIST.local_source_equivalent_penalty(g.atom_weights, g.sdevs, 9, 18)
        new_reg = g.regularization(global_atom_count=9, global_width_count=18)
        torch.testing.assert_close(new_reg, old_reg, rtol=0, atol=0)

    def test_isotropic_peak_option_reuses_original_kernel(self):
        xyz, rotation, trans, center = scene()
        g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype),
                              "isotropic", amplitude_convention="peak_2d")
        sigma = g.widths()
        expected = OLD.pdb2img(xyz[None], 3., g.atom_weights, rotation, trans,
                               center, box_size=24, sdevs=sigma.expand(-1, 2))
        actual = g(xyz, rotation, trans, 3., center, box_size=24)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grad = torch.autograd.grad(image_objective(expected), g.raw_sigma,
                                            retain_graph=True)[0]
        actual_grad = torch.autograd.grad(image_objective(actual), g.raw_sigma)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_distinct_structure_per_particle_batch(self):
        xyz, rotation, trans, center = scene()
        xyz2 = xyz.clone()
        xyz2[1] += torch.tensor([.4, -.2, .3], dtype=xyz.dtype)
        for mode in GaussianProjector.MODES:
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), mode)
            coordinates = torch.stack((xyz, xyz2)).requires_grad_()
            batched = g(coordinates, rotation, trans, 3., center, box_size=24)
            separate = torch.cat([g(coordinates[i:i+1], rotation[i:i+1], trans[i:i+1],
                                    3., center, box_size=24) for i in range(2)])
            # Legacy crop rectangles depend on the entire particle batch.
            tolerance = dict(rtol=1e-4, atol=1e-6) if mode == "legacy" else dict(rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(batched, separate, **tolerance)
            gradients = [torch.autograd.grad(image_objective(im), coordinates,
                                             retain_graph=True)[0] for im in (batched, separate)]
            torch.testing.assert_close(*gradients, **tolerance)

    def test_both_trainer_parsers_accept_kernel_flags(self):
        # Public parser imports are lazy; use the actual factories without
        # extracting or executing a possibly changed __main__ block.
        from train import build_parser as train_parser
        from train_chain_parallel_2d import build_parser as parallel_parser
        for factory, required in (
            (train_parser, ["--star_data_dir", "dummy", "--cif_path", "dummy",
                            "--diffusion_data_dir", "dummy", "--output_trained_model_dir", "dummy"]),
            (parallel_parser, ["--component_manifest", "dummy", "--star_data_dir", "dummy",
                               "--mrc_data_dir", "dummy", "--output_trained_model_dir", "dummy"]),
        ):
            parser = factory()
            self.assertEqual(parser.parse_args(required).gmm_kernel, "legacy")
            for mode in GaussianProjector.MODES:
                args = parser.parse_args(required + ["--gmm-kernel", mode, "--gmm-atom-chunk-size", "16",
                                                       "--no-gmm-checkpoint-chunks"])
                self.assertEqual(args.gmm_kernel, mode)
                self.assertEqual(args.gmm_atom_chunk_size, 16)
                self.assertFalse(args.gmm_checkpoint_chunks)

    def test_spherical_limit_images_and_coordinate_gradients(self):
        xyz, rotation, trans, center = scene()
        results = []
        for kernel in ("isotropic", "anisotropic"):
            x = xyz.clone().requires_grad_()
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), kernel)
            im = g(x[None], rotation, trans, 4.2, center, box_size=24, apix=1.3)
            grad = torch.autograd.grad(image_objective(im), (x, g.atom_weights))
            results.append((im, grad))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-11, atol=1e-12)
        for a, b in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)

    def test_rotated_covariance_and_numerical_line_integral(self):
        dtype = torch.float64
        cov = torch.diag(torch.tensor([4., 1., 2.25], dtype=dtype))
        rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=dtype)
        torch.testing.assert_close(project_gaussian_covariances(cov[None], rz90[None])[0, 0],
                                   torch.diag(torch.tensor([1., 4.], dtype=dtype)))
        a, b = .7, .4
        ry = torch.tensor([[math.cos(a), 0., math.sin(a)], [0., 1., 0.],
                           [-math.sin(a), 0., math.cos(a)]], dtype=dtype)
        rz = torch.tensor([[math.cos(b), -math.sin(b), 0.],
                           [math.sin(b), math.cos(b), 0.], [0., 0., 1.]], dtype=dtype)
        rotation = rz @ ry
        cov2 = project_gaussian_covariances(cov[None], rotation[None])
        q = torch.tensor([2.3], dtype=dtype)
        image = sum_of_gaussians_2d_covariance(torch.tensor([[[16., 16.]]], dtype=dtype),
                                               q, cov2, 33, cutoff=12, checkpoint_chunks=False)
        cov_camera = rotation @ cov @ rotation.T
        points = torch.tensor([[0., 0.], [1., 2.], [-2., 1.]], dtype=dtype)
        z = torch.linspace(-25., 25., 10001, dtype=dtype)
        xyz = torch.cat([points[:, None].expand(-1, len(z), -1),
                         z[None, :, None].expand(len(points), -1, -1)], dim=-1)
        exponent = torch.einsum('mzi,ij,mzj->mz', xyz, torch.linalg.inv(cov_camera), xyz)
        value = q / ((2 * math.pi)**1.5 * torch.linalg.det(cov_camera).sqrt())
        numerical = torch.trapz(value * torch.exp(-.5 * exponent), z, dim=-1)
        predicted = image[0, (points[:, 1] + 16).long(), (points[:, 0] + 16).long()]
        torch.testing.assert_close(predicted, numerical, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(image.sum()), float(q[0]), places=6)

    def test_input_gradcheck_for_analytic_raster(self):
        dtype = torch.float64
        means = torch.tensor([[[9.2, 10.3], [12.4, 8.7]]], dtype=dtype, requires_grad=True)
        mass = torch.tensor([2., 3.], dtype=dtype, requires_grad=True)
        raw = torch.tensor([[1.3, 1.1, .9, .1, -.2, .2],
                            [1.4, 1.2, 1., -.1, .1, .3]], dtype=dtype, requires_grad=True)
        def fn(mu, q, r):
            lower = torch.zeros(2, 3, 3, dtype=dtype)
            lower[:, 0, 0], lower[:, 1, 1], lower[:, 2, 2] = r[:, 0], r[:, 1], r[:, 2]
            lower[:, 1, 0], lower[:, 2, 0], lower[:, 2, 1] = r[:, 3], r[:, 4], r[:, 5]
            cov = project_gaussian_covariances(lower @ lower.transpose(-1, -2), torch.eye(3, dtype=dtype)[None])
            image = sum_of_gaussians_2d_covariance(mu, q, cov, 24, cutoff=10, checkpoint_chunks=False)
            return image[:, 8:12, 8:12]
        self.assertTrue(torch.autograd.gradcheck(fn, (means, mass, raw), atol=1e-5, rtol=1e-4))

    def test_checkpoint_chunks_and_partition_independence(self):
        xyz, rotation, _, _ = scene()
        results = []
        for chunk, recompute in ((1, False), (2, True), (64, True)):
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), "anisotropic",
                                  atom_chunk_size=chunk, checkpoint_chunks=recompute)
            with torch.no_grad():
                g.raw_cholesky[:, 3] = .13
                g.raw_cholesky[:, 4] = -.08
            x = xyz.clone().requires_grad_()
            projected = g.project_coordinates(x, rotation, 1.3)
            raw = g.render_raw(projected, rotation, projected.amin(1, keepdim=True), 3., 24)
            grad = torch.autograd.grad(raw.square().sum(), (x, *g.parameters()))
            results.append((raw, grad))
        for image, gradients in results[1:]:
            torch.testing.assert_close(image, results[0][0], rtol=1e-12, atol=1e-12)
            for a, b in zip(gradients, results[0][1]):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)

    def test_checkpoint_roundtrip_and_legacy_import(self):
        xyz, rotation, trans, center = scene()
        for mode in GaussianProjector.MODES:
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), mode)
            if mode == "anisotropic":
                with torch.no_grad():
                    g.raw_cholesky[:, 3] = .17
            blob = io.BytesIO()
            torch.save({"gmm": g.export_checkpoint()}, blob)
            blob.seek(0)
            loaded = GaussianProjector.from_checkpoint(torch.load(blob, weights_only=False), device="cpu")
            self.assertEqual(loaded.config(), g.config())
            torch.testing.assert_close(loaded(xyz, rotation, trans, 3., center, box_size=24),
                                       g(xyz, rotation, trans, 3., center, box_size=24), rtol=0, atol=0)
        old = GaussianProjector(torch.tensor([6., 8.]))
        loaded = GaussianProjector.from_checkpoint({"atom_weights": old.atom_weights, "sdevs": old.sdevs})
        self.assertEqual(loaded.kernel, "legacy")
        with self.assertRaises(ValueError):
            GaussianProjector.from_checkpoint({"gmm_kernel": "isotropic", "atom_weights": old.atom_weights, "sdevs": old.sdevs})

    def test_positive_covariance_regularizer_and_mode_validation(self):
        for mode in ("isotropic", "anisotropic"):
            g = GaussianProjector(torch.tensor([.5, 21.], dtype=torch.float64), mode)
            with torch.no_grad():
                g.shape_parameters()[0][0, 0] = -6.
                g.shape_parameters()[0][1, 0] = 1.
                if mode == "anisotropic":
                    g.raw_cholesky[:, 3] = .4
            self.assertTrue((torch.linalg.eigvalsh(g.covariance()) > 0).all())
            g.regularization().backward()
            self.assertTrue(torch.isfinite(g.shape_parameters()[0].grad).all())
            self.assertGreater(float(g.shape_parameters()[0].grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            GaussianProjector(torch.ones(2), "anisotropic", amplitude_convention="peak_2d")
        parser = argparse.ArgumentParser()
        add_gmm_arguments(parser)
        self.assertEqual(parser.parse_args([]).gmm_kernel, "legacy")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward_all_modes_and_mixed_legacy_devices(self):
        xyz, rotation, trans, center = scene(dtype=torch.float32, device="cuda")
        for mode in GaussianProjector.MODES:
            # Main trainer historically keeps legacy amplitudes on CPU.
            weights = torch.tensor([6., 8., 7.], device="cpu" if mode == "legacy" else "cuda")
            g = GaussianProjector(weights, mode, shape_device="cuda", atom_chunk_size=2)
            x = xyz.detach().clone().requires_grad_()
            output = g(x, rotation, trans, 3., center, box_size=24)
            (image_objective(output) + g.regularization()).backward()
            for value in (output, x.grad, *[p.grad for p in g.parameters()]):
                self.assertIsNotNone(value)
                self.assertTrue(torch.isfinite(value).all())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
