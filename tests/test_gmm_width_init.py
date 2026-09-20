"""CPU-only width initialization regressions; no Protenix or external data.

python -m unittest discover -s tests -p test_gmm_width_init.py -v
The frozen fixture is the unmodified src/gmm.py from origin/main 80f8f46.
"""
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gmm import (GaussianProjector, add_gmm_arguments, gmm_from_arguments,
                 initial_internal_sdev, describe_gmm_width, GAUSSIAN_SIGMA_FACTOR)
from training_restart import restart_gmm

spec = importlib.util.spec_from_file_location('gmm_before_width_init',
                                             ROOT / 'tests/reference/gmm_before_width_init.py')
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)


def options(*flags):
    parser = argparse.ArgumentParser()
    add_gmm_arguments(parser)
    args = parser.parse_args(list(flags))
    args.apix = 3.0
    args.resolution = 3.0
    return args


class WidthInitializationTests(unittest.TestCase):
    def test_frozen_reference_provenance(self):
        source = (ROOT / 'tests/reference/gmm_before_width_init.py').read_text(encoding='utf-8')
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         '0240d63c9c0f87be6e628394832ed5754371e45d68f0a50fff383cc486fcc473')

    def setUp(self):
        self.weights = torch.tensor([6., 7., 8.], dtype=torch.float64)
        self.xyz = torch.tensor([[-2., -1., .2], [1., -.5, .3], [2., 2., -.4]], dtype=torch.float64)
        self.rot = torch.eye(3, dtype=torch.float64)[None, :2]
        self.trans = torch.zeros(1, 2, dtype=torch.float64)
        self.center = torch.tensor([12., 12.], dtype=torch.float64)

    def build(self, args):
        return gmm_from_arguments(self.weights, args, record_initialization=False)

    def render(self, model):
        return model(self.xyz, self.rot, self.trans, 3., self.center, box_size=24, apix=3.)

    def same_state(self, a, b):
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]), key)

    def test_default_and_explicit_legacy_match_frozen_source(self):
        for kernel in GaussianProjector.MODES:
            with self.subTest(kernel=kernel):
                args = options('--gmm-kernel', kernel)
                old = OLD.gmm_from_arguments(self.weights, args)
                default = self.build(args)
                explicit = self.build(options('--gmm-kernel', kernel, '--gmm-sdev-init-mode', 'legacy'))
                self.same_state(old, default)
                self.same_state(old, explicit)
                self.assertTrue(torch.equal(self.render(old), self.render(default)))
                self.assertTrue(torch.equal(self.render(old), self.render(explicit)))
                # Simulate pre-extension Python configs without either new field.
                del args.gmm_sdev_init_mode
                del args.gmm_molmap_resolution_A
                self.same_state(old, self.build(args))

    def test_width_mapping_table(self):
        f = GAUSSIAN_SIGMA_FACTOR
        for apix, mode, target, s_multiple, sigma_multiple, equivalent in (
            (1, 'legacy', None, 3, 3, 3), (1, 'molmap', 3, 3, 3, 3),
            (3, 'legacy', None, 3, 9, 9), (3, 'molmap', 3, 1, 3, 3),
            (3, 'molmap', 4.5, 1.5, 4.5, 4.5), (3, 'molmap', 6, 2, 6, 6)):
            with self.subTest(apix=apix, mode=mode, target=target):
                data = describe_gmm_width(mode=mode, apix=apix, legacy_resolution=3,
                                          molmap_resolution_A=target)
                self.assertAlmostEqual(data['gmm_internal_sdev_initial'], s_multiple*f, places=14)
                self.assertAlmostEqual(data['gmm_physical_sigma_initial_A'], sigma_multiple*f, places=14)
                self.assertAlmostEqual(data['gmm_molmap_equivalent_resolution_initial_A'], equivalent, places=13)

    def test_unit_apix_same_named_resolution_is_identical(self):
        args = options('--gmm-sdev-init-mode', 'molmap', '--gmm-molmap-resolution-A', '3')
        args.apix = 1
        self.same_state(self.build(args), GaussianProjector(self.weights))

    def test_round_trip(self):
        rng = random.Random(101)
        for _ in range(50):
            a, r, target = [rng.uniform(.5, 12) for _ in range(3)]
            data = describe_gmm_width(mode='molmap', apix=a, legacy_resolution=r,
                                      molmap_resolution_A=target)
            self.assertAlmostEqual(data['gmm_molmap_equivalent_resolution_initial_A'], target, places=12)

    def test_coordinate_scale_and_amplitudes_unchanged(self):
        for kernel in GaussianProjector.MODES:
            old = self.build(options('--gmm-kernel', kernel))
            new = self.build(options('--gmm-kernel', kernel, '--gmm-sdev-init-mode', 'molmap',
                                     '--gmm-molmap-resolution-A', '3'))
            self.assertTrue(torch.equal(old.atom_weights, new.atom_weights))
            self.assertEqual(old.sigma_init, new.sigma_init)  # reference_mass baseline
            p_old = old.project_coordinates(self.xyz, self.rot, 3.)
            p_new = new.project_coordinates(self.xyz, self.rot, 3.)
            self.assertTrue(torch.equal(p_old, p_new))
            # Actual renderer receives the same scaled centers and coefficients.
            if kernel == 'legacy':
                from gmm import sum_of_gaussians_2d_torch
                seen = []
                def capture(*args, **kwargs):
                    seen.append((kwargs['centers'].clone(), kwargs['coef'].clone()))
                    return sum_of_gaussians_2d_torch(*args, **kwargs)
                with patch('gmm.sum_of_gaussians_2d_torch', side_effect=capture):
                    for model, projected in ((old, p_old), (new, p_new)):
                        model.render_raw(projected, self.rot, projected.amin(1, keepdim=True), 3., 24)
                self.assertTrue(torch.equal(seen[0][0], seen[1][0]))
                self.assertTrue(torch.equal(seen[0][1], seen[1][1]))
            image = self.render(new)
            self.assertTrue(torch.isfinite(image).all())
            image.square().sum().backward()
            for parameter in new.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_mapping_and_floor(self):
        for key in ('apix', 'legacy_resolution', 'molmap_resolution_A'):
            for value in (None, 0, -1, float('nan'), float('inf')):
                kwargs = dict(mode='molmap', apix=3, legacy_resolution=3, molmap_resolution_A=3)
                kwargs[key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'fresh GMM width initialization'):
                    initial_internal_sdev(**kwargs)
        with self.assertRaisesRegex(ValueError, 'unknown mode'):
            initial_internal_sdev(mode='unknown')
        with self.assertRaisesRegex(ValueError, 'sigma_floor'):
            self.build(options('--gmm-sdev-init-mode', 'molmap', '--gmm-molmap-resolution-A', '0.00001'))

    def test_unused_target_does_not_change_legacy(self):
        self.same_state(self.build(options()), self.build(options('--gmm-molmap-resolution-A', 'nan')))

    def test_metadata_record_and_learning_switch(self):
        args = options('--gmm-sdev-init-mode', 'molmap', '--gmm-molmap-resolution-A', '3')
        with patch('run_recording.current_record') as record:
            model = gmm_from_arguments(self.weights, args)
        metadata = record.return_value.resolved.call_args.args[1]
        self.assertEqual(metadata, model.config()['width_initialization'])
        self.assertEqual(metadata['gmm_sdev_init_mode'], 'molmap')
        self.assertAlmostEqual(metadata['internal_grid_spacing_A'], 3.)
        model.set_learning_enabled(False)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        xyz = self.xyz.clone().requires_grad_(True)
        model(xyz, self.rot, self.trans, 3., self.center, box_size=24, apix=3.).square().sum().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        model.set_learning_enabled(True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_old_payload_restores_without_new_metadata(self):
        for kernel in GaussianProjector.MODES:
            old = OLD.GaussianProjector(self.weights, kernel=kernel)
            with torch.no_grad():
                old.shape_parameters()[0].add_(.02)
            loaded = GaussianProjector.from_checkpoint(old.export_checkpoint())
            self.same_state(old, loaded)
            self.assertTrue(torch.equal(self.render(old), self.render(loaded)))

    def test_restart_saved_values_win_even_with_invalid_fresh_target(self):
        args = options('--gmm-sdev-init-mode', 'molmap')  # missing target: ignored on restore
        old = OLD.GaussianProjector(self.weights)
        with torch.no_grad():
            old.sdevs.fill_(.42)
        for payload in ({'gmm': old.export_checkpoint()},
                        {'atom_weights': old.atom_weights.detach(), 'sdevs': old.sdevs.detach()}):
            loaded, _ = restart_gmm(payload, args, self.weights, 'cpu')
            self.same_state(old, loaded)
            self.assertIsNone(loaded.width_initialization)

    def test_new_checkpoint_round_trip_preserves_learned_width_and_metadata(self):
        for kernel in GaussianProjector.MODES:
            new = self.build(options('--gmm-kernel', kernel, '--gmm-sdev-init-mode', 'molmap',
                                     '--gmm-molmap-resolution-A', '3'))
            with torch.no_grad():
                new.shape_parameters()[0].add_(.03)
            payload = new.export_checkpoint()
            self.assertEqual(payload['format_version'], 1)
            loaded = GaussianProjector.from_checkpoint(payload)
            self.same_state(new, loaded)
            self.assertEqual(new.config(), loaded.config())
            self.assertTrue(torch.equal(self.render(new), self.render(loaded)))


if __name__ == '__main__':
    unittest.main()
