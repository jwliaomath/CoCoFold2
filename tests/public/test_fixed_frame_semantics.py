"""Pure CPU fixed-frame renderer regressions; no dataset or hetero package."""
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_frame_probe import (scene, poses, project, compare, translation_rows,
                               deformation_rows, cache_rows, export_rows, gradient_rows)


def test_global_translation_and_view_axis():
    rows = translation_rows()
    exact = [r for r in rows if r['delta'] == 'integer_xy']
    assert len(exact) == 3
    assert all(r['equiv_relative_l2'] < 1e-4 for r in exact)
    assert all(r['same_shift_relative_l2'] > .1 for r in exact)
    axis = [r for r in rows if r['delta'] == 'view_axis']
    assert all(r['same_shift_relative_l2'] < 1e-4 for r in axis)
    # Fractional-pixel cases are reported by the private audit rather than
    # misclassified as a frame bug: bicubic resampling adds a numerical floor.
    assert all(r['equiv_real_cc'] > .999 for r in rows)


def test_legacy_cancels_global_translation():
    x, gmm = scene()
    delta = torch.tensor([5., -2., 1.], dtype=x.dtype)
    for rotation in poses():
        fixed = compare(project(x + delta, gmm, rotation), project(x, gmm, rotation))
        old = compare(project(x + delta, gmm, rotation, mode='legacy'),
                      project(x, gmm, rotation, mode='legacy'))
        assert fixed['relative_l2'] > .1
        assert old['relative_l2'] < 1e-4


def test_local_deformation_keeps_stationary_part_in_place():
    row = deformation_rows()
    assert row['centroid_shift_A'][0] == 3.
    assert row['fixed']['unchanged_part_A']['relative_l2'] < 1e-4
    assert row['legacy']['unchanged_part_A']['relative_l2'] > .1
    assert abs(row['fixed']['part_A_after_pixel'][0] -
               row['fixed']['part_A_before_pixel'][0]) < 1e-4
    assert abs(row['fixed']['part_B_after_pixel'][0] -
               row['fixed']['part_B_before_pixel'][0] - 6.) < 1e-2
    assert abs(row['legacy']['part_A_after_pixel'][0] -
               row['legacy']['part_A_before_pixel'][0]) > 1.


def test_gmm_checkpoint_and_structure_export_preserve_frame():
    cache = cache_rows()[0]
    exported = export_rows()[0]
    assert cache['relative_l2'] < 1e-8
    assert cache['source_centroid_A'] == cache['restored_centroid_A']
    assert exported['labels_match']
    assert exported['coordinate_rmsd_no_fit_A'] < 1e-3
    assert exported['relative_l2'] < 1e-3


def test_translation_gradient_matches_finite_difference():
    rows = {r['mode']: r for r in gradient_rows()}
    assert abs(rows['fixed']['autograd']) > 1e-5
    assert rows['fixed']['abs_error'] < 1e-4
    assert abs(rows['legacy']['autograd']) < 1e-5


def test_public_scoring_forwards_saved_frame(monkeypatch):
    import benchmark_case
    import particledataset
    import ctf
    import utils

    class Particles:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return (np.zeros((8, 8), dtype=np.float32),
                    np.ones(8, dtype=np.float32), np.zeros(2, dtype=np.float32),
                    np.eye(3, dtype=np.float32)[:2], None, None)

    class Capture:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append((kwargs['projection_frame'], kwargs['projection_origin']))
            return torch.ones((1, 1, 8, 8))

    monkeypatch.setattr(particledataset, 'ParticleDataset', Particles)
    monkeypatch.setattr(ctf, 'compute_ctf', lambda f, *args: torch.ones((1, len(f[0]))))
    monkeypatch.setattr(utils, 'compute_frc', lambda *args, **kwargs: torch.tensor(.5))
    gmm = Capture()
    benchmark_case.fixed_particle_frc(torch.zeros(1, 3), torch.zeros(1, 3),
                                      gmm, 'unused.star', 8, 1., 'cpu', count=1,
                                      projection_frame='fixed', projection_origin=(2., 3., 4.))
    assert gmm.calls == [('fixed', (2., 3., 4.)), ('fixed', (2., 3., 4.))]
