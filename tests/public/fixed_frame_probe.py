"""Small, dataset-independent numerical probes of the fixed projection frame.

No alignment or fitted translation is used in any comparison.  Rotations are
stored as [B,2,3], so the in-plane displacement of a row-coordinate delta is
``delta @ rotation.T``.  A positive particle shift is a *sampling* shift in
grid_sample and therefore moves the image in the negative direction.
"""
import math
import tempfile
from pathlib import Path

import torch
import gemmi

from gmm import GaussianProjector
from structure_io import read_template, write_coordinates


DTYPE = torch.float64
BOX = 64
CENTER = (31.5, 31.5)
ORIGIN = (4., -3., 2.)


def scene():
    coordinates = torch.tensor([
        [-8., -2., 1.], [-7., 2., -1.], [-4., .5, 2.],
        [5., -1., 0.], [8., 2., 1.], [9., -2., -2.],
    ], dtype=DTYPE) + torch.tensor(ORIGIN, dtype=DTYPE)
    weights = torch.tensor([3., 5., 4., 7., 5., 3.], dtype=DTYPE)
    projector = GaussianProjector(weights, kernel='legacy').to(dtype=DTYPE)
    with torch.no_grad():
        projector.sdevs.fill_(1.15)
    return coordinates, projector


def poses():
    a, b = .61, .43
    rz = torch.tensor([[math.cos(a), -math.sin(a), 0.],
                       [math.sin(a), math.cos(a), 0.], [0., 0., 1.]], dtype=DTYPE)
    ry = torch.tensor([[math.cos(b), 0., math.sin(b)],
                       [0., 1., 0.], [-math.sin(b), 0., math.cos(b)]], dtype=DTYPE)
    return [torch.eye(3, dtype=DTYPE)[:2], rz[:2], (rz @ ry)[:2]]


def project(x, gmm, rotation, shift=(0., 0.), mode='fixed', origin=ORIGIN):
    shift = torch.as_tensor(shift, dtype=DTYPE).reshape(1, 2)
    return gmm(x, rotation.reshape(1, 2, 3), shift, 3.,
               torch.tensor(CENTER, dtype=DTYPE), box_size=BOX, apix=1.,
               projection_frame=mode, projection_origin=origin)


def compare(a, b):
    av, bv = a.double().flatten(), b.double().flatten()
    diff = av - bv
    rel = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(bv).clamp_min(1e-12))
    centered_a, centered_b = av - av.mean(), bv - bv.mean()
    cc = float(torch.dot(centered_a, centered_b) /
               (torch.linalg.vector_norm(centered_a) * torch.linalg.vector_norm(centered_b)).clamp_min(1e-12))
    fa, fb = torch.fft.fft2(a.double()), torch.fft.fft2(b.double())
    fcc = float((fa.conj() * fb).real.sum() /
                (fa.abs().square().sum() * fb.abs().square().sum()).sqrt().clamp_min(1e-12))
    return dict(relative_l2=rel, max_abs=float(diff.abs().max()), real_cc=cc,
                fourier_cc=fcc)


def translation_rows():
    x, gmm = scene()
    rows = []
    for pose_id, rotation in enumerate(poses()):
        view = torch.linalg.cross(rotation[0], rotation[1], dim=0)
        for label, delta in (
            ('integer_xy', 3 * rotation[0] - 2 * rotation[1] + view),
            ('5A_x', torch.tensor([5., 0., 0.], dtype=DTYPE)),
            ('5A_y', torch.tensor([0., 5., 0.], dtype=DTYPE)),
            ('mixed', torch.tensor([3., -4., 2.], dtype=DTYPE)),
            ('view_axis', 5 * view),
        ):
            xy = delta @ rotation.T  # Angstrom; apix=1, resolution/3=1.
            baseline = project(x, gmm, rotation)
            moved = project(x + delta, gmm, rotation)
            # grid_sample samples input at output+shift: positive shift moves
            # the rendered image to the left.  Compensating shift is -xy.
            adjusted = project(x, gmm, rotation, -xy)
            rows.append(dict(pose=pose_id, delta=label, delta_A=delta.tolist(),
                             expected_image_shift_pixels=xy.tolist(),
                             shift_parameter_pixels=(-xy).tolist(),
                             **{f'equiv_{k}': v for k, v in compare(moved, adjusted).items()},
                             **{f'same_shift_{k}': v for k, v in compare(moved, baseline).items()}))
    return rows


def deformation_rows():
    x, gmm = scene()
    x2 = x.clone()
    x2[3:] += torch.tensor([6., 0., 0.], dtype=DTYPE)
    rotation = poses()[0]
    outputs = {}
    def location(image, x0, x1):
        patch = image[0, 0, 20:44, x0:x1].clamp_min(0)
        yy, xx = torch.meshgrid(torch.arange(20, 44, dtype=DTYPE),
                                torch.arange(x0, x1, dtype=DTYPE), indexing='ij')
        mass = patch.sum().clamp_min(1e-30)
        return [float((patch * xx).sum() / mass), float((patch * yy).sum() / mass)]
    for mode in ('fixed', 'legacy'):
        before, after = project(x, gmm, rotation, mode=mode), project(x2, gmm, rotation, mode=mode)
        # The left-hand ROI contains only part A; Gaussian cutoff makes the
        # moving right-hand atoms exactly absent from it.
        roi_before = before[..., 20:44, 12:29]
        roi_after = after[..., 20:44, 12:29]
        outputs[mode] = dict(unchanged_part_A=compare(roi_after, roi_before),
                             whole_image=compare(after, before),
                             part_A_before_pixel=location(before, 12, 29),
                             part_A_after_pixel=location(after, 12, 29),
                             part_B_before_pixel=location(before, 33, 57),
                             part_B_after_pixel=location(after, 33, 57))
    return dict(centroid_before_A=x.mean(0).tolist(),
                centroid_after_A=x2.mean(0).tolist(),
                centroid_shift_A=(x2.mean(0)-x.mean(0)).tolist(),
                fixed=outputs['fixed'], legacy=outputs['legacy'])


def cache_rows():
    x, gmm = scene()
    pose = poses()[1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'gmm.pt'
        torch.save(gmm.export_checkpoint(), path)
        restored = GaussianProjector.from_checkpoint(
            torch.load(path, map_location='cpu', weights_only=False), device='cpu').to(dtype=DTYPE)
        before = project(x, gmm, pose)
        after = project(x, restored, pose)
        return [dict(path='GaussianProjector.export_checkpoint/from_checkpoint',
                     source_centroid_A=x.mean(0).tolist(),
                     restored_centroid_A=x.mean(0).tolist(), **compare(after, before))]


def export_rows():
    x, gmm = scene()
    pose = poses()[1]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for suffix in ('pdb', 'cif'):
            source = Path(directory) / ('source.' + suffix)
            exported = Path(directory) / ('exported.' + suffix)
            if suffix == 'pdb':
                # PDB coordinates have three decimal places; never fit a
                # rigid transform to hide their rounding or a frame shift.
                with source.open('w') as handle:
                    for i, point in enumerate(x.tolist(), 1):
                        handle.write(f'ATOM  {i:5d}  CA  ALA A{i:4d}    '
                                     f'{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}'
                                     f'{1.:6.2f}{20.:6.2f}           C\n')
                    handle.write('END\n')
            else:
                document = gemmi.cif.Document()
                block = document.add_new_block('fixture')
                loop = block.init_loop('_atom_site.', [
                    'label_alt_id', 'type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z',
                    'label_asym_id', 'label_seq_id', 'label_atom_id'])
                for i, point in enumerate(x.tolist(), 1):
                    loop.add_row(['.', 'C', *[f'{value:.6f}' for value in point],
                                  'A', str(i), 'CA'])
                document.write_file(str(source))
            original, _, labels = read_template(source)
            write_coordinates(source, exported, original.numpy())
            reloaded, _, labels2 = read_template(exported)
            coordinate_delta = reloaded.double() - original.double()
            before = project(original.double(), gmm, pose)
            after = project(reloaded.double(), gmm, pose)
            rows.append(dict(path='structure_io.write_coordinates/read_template',
                             format=suffix, labels_match=labels == labels2,
                             centroid_before_A=original.double().mean(0).tolist(),
                             centroid_after_A=reloaded.double().mean(0).tolist(),
                             coordinate_rmsd_no_fit_A=float(
                                 coordinate_delta.square().sum(-1).mean().sqrt()),
                             **compare(after, before)))
    return rows


def gradient_rows():
    x, gmm = scene()
    rotation = poses()[2]
    target = project(x + torch.tensor([1., -.5, 0.], dtype=DTYPE), gmm, rotation).detach()
    rows = []
    for mode in ('fixed', 'legacy'):
        def loss(delta):
            image = project(x + torch.stack((delta, delta*0, delta*0)), gmm, rotation, mode=mode)
            return (image - target).square().mean()
        delta = torch.tensor(.1, dtype=DTYPE, requires_grad=True)
        value = loss(delta)
        value.backward()
        h = 1e-3
        finite = float((loss(delta.detach()+h)-loss(delta.detach()-h))/(2*h))
        rows.append(dict(mode=mode, autograd=float(delta.grad), finite_difference=finite,
                         abs_error=abs(float(delta.grad)-finite)))
    return rows
