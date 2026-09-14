"""Peak-3D projection, gradients, persistence, and pre-change regression.

Run together with the original checks:
python -m unittest discover -s tests -p "test_gmm*.py" -v
"""
import argparse
import importlib.util
import io
import math
from pathlib import Path
import unittest
import warnings

import torch
from torch.func import functional_call

from test_gmm import GaussianProjector, add_gmm_arguments, image_objective, scene


def load_previous():
    path = Path(__file__).parent / "reference/gmm_before_peak3d.py"
    spec = importlib.util.spec_from_file_location("gmm_before_peak3d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GaussianProjector


PREVIOUS = load_previous()


class Peak3DTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message=".*grid_sample.*")
        warnings.filterwarnings("ignore", message=".*meshgrid.*")

    def test_existing_four_configurations_are_unchanged(self):
        xyz, rotation, trans, center = scene()
        configurations = (("legacy", "auto"), ("isotropic", "peak_2d"),
                          ("isotropic", "auto"), ("anisotropic", "auto"))
        for mode, amplitude in configurations:
            with self.subTest(mode=mode, amplitude=amplitude):
                weights = torch.tensor([.5, 8., 21.], dtype=xyz.dtype)
                old = PREVIOUS(weights, mode, amplitude_convention=amplitude)
                with torch.no_grad():
                    old.shape_parameters()[0][0, 0] = .09 if mode == "legacy" else -3.
                    if mode == "anisotropic":
                        old.raw_cholesky[:, 3:] = torch.tensor([.13, -.08, .11], dtype=xyz.dtype)
                # Also checks loading a pre-peak_3d v1 GMM checkpoint.
                new = GaussianProjector.from_checkpoint(old.export_checkpoint())
                self.assertEqual({**old.config(), "checkpoint_peak2d": False}, new.config())
                old_x, new_x = [xyz.clone().requires_grad_() for _ in range(2)]
                images = [g(x, rotation, trans, 3., center, box_size=24)
                          for g, x in ((old, old_x), (new, new_x))]
                torch.testing.assert_close(*images, rtol=0, atol=0)
                torch.testing.assert_close(old.regularization(), new.regularization(), rtol=0, atol=0)
                for g, im in zip((old, new), images):
                    (image_objective(im) + g.regularization()).backward()
                torch.testing.assert_close(old_x.grad, new_x.grad, rtol=0, atol=0)
                for p, q in zip(old.parameters(), new.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                for g in (old, new):
                    torch.optim.AdamW(g.parameters(), lr=.005).step()
                for p, q in zip(old.parameters(), new.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_peak3d_raw_projection_matches_numerical_3d_integral(self):
        g = GaussianProjector(torch.tensor([2.3], dtype=torch.float64), "anisotropic",
                              sigma_floor=.2, amplitude_convention="peak_3d", checkpoint_chunks=False)
        with torch.no_grad():
            g.raw_cholesky[0, :3] = torch.log(torch.expm1(torch.tensor([1.4, .9, 1.1], dtype=torch.float64)))
            g.raw_cholesky[0, 3:] = torch.tensor([.2, -.15, .35], dtype=torch.float64)
        a, b = .7, .4
        ry = torch.tensor([[math.cos(a), 0., math.sin(a)], [0., 1., 0.],
                           [-math.sin(a), 0., math.cos(a)]], dtype=torch.float64)
        rz = torch.tensor([[math.cos(b), -math.sin(b), 0.],
                           [math.sin(b), math.cos(b), 0.], [0., 0., 1.]], dtype=torch.float64)
        rotation = (rz @ ry)[None]
        projected = g.project_coordinates(torch.zeros(1, 3, dtype=torch.float64), rotation, 1.)
        raw = g.render_raw(projected, rotation, projected.clone(), 3., 25, cutoff_range=12.)
        # The common legacy scale is separate from Gaussian amplitude semantics.
        common_scale = (2 * math.pi)**-1 * (3 / (math.pi * math.sqrt(2)))**-2
        cov_camera = rotation[0] @ g.covariance()[0] @ rotation[0].T
        points = torch.tensor([[0., 0.], [1., 2.], [-2., 1.]], dtype=torch.float64)
        z = torch.linspace(-25., 25., 10001, dtype=torch.float64)
        coordinates = torch.cat([points[:, None].expand(-1, len(z), -1),
                                 z[None, :, None].expand(len(points), -1, -1)], dim=-1)
        distance = torch.einsum("mzi,ij,mzj->mz", coordinates, torch.linalg.inv(cov_camera), coordinates)
        numerical = torch.trapz(g.atom_weights[0] * torch.exp(-.5 * distance), z, dim=-1)
        values = raw[0, 0, (points[:, 1] + 9).long(), (points[:, 0] + 9).long()] / common_scale
        torch.testing.assert_close(values, numerical, rtol=1e-11, atol=1e-12)

    def test_view_dependent_peaks_and_line_of_sight_width_gradient(self):
        g = GaussianProjector(torch.tensor([2.], dtype=torch.float64), "anisotropic",
                              sigma_floor=.2, amplitude_convention="peak_3d", checkpoint_chunks=False)
        sigma = torch.tensor([2., 1., 1.3], dtype=torch.float64)
        with torch.no_grad():
            lower_diagonal = (sigma.square() - g.sigma_floor**2).sqrt()
            g.raw_cholesky[0, :3] = torch.log(torch.expm1(lower_diagonal))
        # Look down z, then down x. Peak equals A3D * sqrt(2*pi) * LOS sigma.
        rotation = torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                                 [[0., 1., 0.], [0., 0., 1.]]], dtype=torch.float64)
        projected = torch.zeros(2, 1, 2, dtype=torch.float64)
        raw = g.render_raw(projected, rotation, projected.clone(), 3., 25)
        common_scale = (2 * math.pi)**-1 * (3 / (math.pi * math.sqrt(2)))**-2
        peaks = raw[:, 0, 9, 9] / common_scale
        expected = g.atom_weights[0] * math.sqrt(2 * math.pi) * sigma[[2, 0]]
        torch.testing.assert_close(peaks, expected, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(peaks[0], g.raw_cholesky)[0]
        d = torch.nn.functional.softplus(g.raw_cholesky[0, 2])
        expected_z_gradient = (g.atom_weights[0] * math.sqrt(2 * math.pi) * d
                               / sigma[2] * torch.sigmoid(g.raw_cholesky[0, 2]))
        torch.testing.assert_close(gradient[0, 2], expected_z_gradient, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradient[0, :2], torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12)

    def test_spherical_limit_and_initial_normalized_image(self):
        xyz, rotation, trans, center = scene()
        results = []
        for mode in ("isotropic", "anisotropic"):
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), mode,
                                  amplitude_convention="peak_3d")
            x = xyz.clone().requires_grad_()
            actual = g(x, rotation, trans, 3., center, box_size=24)
            gradients = torch.autograd.grad(image_objective(actual), (x, g.atom_weights))
            reference = GaussianProjector(g.atom_weights, mode)
            torch.testing.assert_close(actual, reference(xyz, rotation, trans, 3., center, box_size=24),
                                       rtol=1e-11, atol=1e-12)
            # Initial raw images differ by a common sqrt(2*pi)*sigma_init;
            # the existing final image normalization cancels that global scale.
            projected = g.project_coordinates(xyz, rotation, 1.)
            origin = projected.amin(1, keepdim=True)
            raw_peak = g.render_raw(projected, rotation, origin, 3., 24)
            raw_mass = reference.render_raw(projected, rotation, origin, 3., 24)
            torch.testing.assert_close(raw_peak, raw_mass * math.sqrt(2 * math.pi) * g.sigma_init,
                                       rtol=1e-11, atol=1e-12)
            results.append((actual, gradients))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-11, atol=1e-12)
        for p, q in zip(results[0][1], results[1][1]):
            torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)

    def test_peak3d_full_forward_finite_difference(self):
        xyz, rotation, trans, center = scene()
        for mode in ("isotropic", "anisotropic"):
            g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), mode,
                                  amplitude_convention="peak_3d", checkpoint_chunks=False)
            shape_name = "raw_sigma" if mode == "isotropic" else "raw_cholesky"
            shape = g.shape_parameters()[0].detach().clone()
            shape[:, 0] += .1
            if mode == "anisotropic":
                shape[:, 3:] = torch.tensor([.13, -.08, .11], dtype=xyz.dtype)
            shape.requires_grad_()
            weights = g.atom_weights.detach().clone().requires_grad_()
            x = xyz.clone().requires_grad_()
            def fn(coordinates, amplitudes, raw_shape):
                image = functional_call(g, {"atom_weights": amplitudes, shape_name: raw_shape},
                                        (coordinates, rotation, trans, 3., center),
                                        {"box_size": 24, "cutoff_range": 10.})
                return image[:, :, 11:13, 11:13]
            self.assertTrue(torch.autograd.gradcheck(fn, (x, weights, shape), atol=1e-5, rtol=1e-4))

    def test_trained_peak3d_checkpoint_roundtrip_and_regularization(self):
        xyz, rotation, trans, center = scene()
        for mode in ("isotropic", "anisotropic"):
            g = GaussianProjector(torch.tensor([.5, 8., 21.], dtype=xyz.dtype), mode,
                                  amplitude_convention="peak_3d")
            optimizer = torch.optim.AdamW(g.parameters(), lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                (image_objective(g(xyz, rotation, trans, 3., center, box_size=24))
                 + g.regularization()).backward()
                optimizer.step()
            # Same w and physical-width ReLU as reference_mass, including grads.
            reference = GaussianProjector(g.atom_weights, mode)
            reference.load_state_dict(g.state_dict())
            for source, target in zip(torch.autograd.grad(g.regularization(), tuple(g.parameters())),
                                      torch.autograd.grad(reference.regularization(), tuple(reference.parameters()))):
                torch.testing.assert_close(source, target, rtol=0, atol=0)
            blob = io.BytesIO()
            torch.save({"gmm": g.export_checkpoint(), "opt_state": optimizer.state_dict()}, blob)
            blob.seek(0)
            loaded = GaussianProjector.from_checkpoint(torch.load(blob, weights_only=False), device="cpu")
            self.assertEqual(loaded.config(), g.config())
            self.assertEqual(loaded.amplitude_convention, "peak_3d")
            for source, target in zip(g.parameters(), loaded.parameters()):
                torch.testing.assert_close(source, target, rtol=0, atol=0)
            torch.testing.assert_close(g(xyz, rotation, trans, 3., center, box_size=24),
                                       loaded(xyz, rotation, trans, 3., center, box_size=24), rtol=0, atol=0)

    def test_peak3d_chunk_checkpoint_consistency(self):
        xyz, rotation, trans, center = scene()
        for mode in ("isotropic", "anisotropic"):
            results = []
            for chunk, recompute in ((1, False), (2, True), (1000, True)):
                g = GaussianProjector(torch.tensor([6., 8., 7.], dtype=xyz.dtype), mode,
                                      amplitude_convention="peak_3d", atom_chunk_size=chunk,
                                      checkpoint_chunks=recompute)
                with torch.no_grad():
                    g.shape_parameters()[0][:, 0] += torch.tensor([.1, -.1, .2], dtype=xyz.dtype)
                    if mode == "anisotropic":
                        g.raw_cholesky[:, 3:] = torch.tensor([.13, -.08, .11], dtype=xyz.dtype)
                x = xyz.clone().requires_grad_()
                im = g(x, rotation, trans, 3., center, box_size=24)
                gradients = torch.autograd.grad(image_objective(im), (x, *g.parameters()))
                results.append((im, gradients))
            for im, gradients in results[1:]:
                torch.testing.assert_close(im, results[0][0], rtol=1e-11, atol=1e-12)
                for source, target in zip(gradients, results[0][1]):
                    torch.testing.assert_close(source, target, rtol=1e-9, atol=1e-11)

    def test_peak3d_cli_and_defaults(self):
        parser = argparse.ArgumentParser()
        add_gmm_arguments(parser)
        self.assertEqual(parser.parse_args([]).gmm_amplitude, "auto")
        for mode in ("isotropic", "anisotropic"):
            args = parser.parse_args(["--gmm-kernel", mode, "--gmm-amplitude", "peak_3d"])
            g = GaussianProjector(torch.ones(2), args.gmm_kernel, amplitude_convention=args.gmm_amplitude)
            self.assertEqual(g.amplitude_convention, "peak_3d")
            self.assertEqual(GaussianProjector(torch.ones(2), mode).amplitude_convention, "reference_mass")
        with self.assertRaises(ValueError):
            GaussianProjector(torch.ones(2), "legacy", amplitude_convention="peak_3d")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_peak3d_cuda_backward_including_cpu_amplitudes(self):
        xyz, rotation, trans, center = scene(dtype=torch.float32, device="cuda")
        for mode in ("isotropic", "anisotropic"):
            for amplitude_device in ("cpu", "cuda"):
                g = GaussianProjector(torch.tensor([6., 8., 7.], device=amplitude_device), mode,
                                      amplitude_convention="peak_3d", shape_device="cuda", atom_chunk_size=2)
                x = xyz.detach().clone().requires_grad_()
                image = g(x, rotation, trans, 3., center, box_size=24)
                (image_objective(image) + g.regularization()).backward()
                for value in (image, x.grad, *[p.grad for p in g.parameters()]):
                    self.assertIsNotNone(value)
                    self.assertTrue(torch.isfinite(value).all())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
