"""Single-map Fourier-slice particles with portable STAR and explicit provenance.

No Protenix or private research code is required. The centered projection/CTF
conventions preserve the original pilot; noise statistics use this single map.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import linecache
from pathlib import Path
import sys
import time
from importlib import metadata

import mrcfile
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def centered_fft(a):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a)))


def ctf_grid(row):
    n, apix, dfu, dfv, angle, kv, cs, amp, phase = row
    axis = (np.arange(int(n)) - int(n)//2) / (n*apix)
    fx, fy = np.meshgrid(axis, axis, indexing='xy')
    s2 = fx*fx + fy*fy
    wavelength = 12.2639 / np.sqrt(kv*1000 + .97845e-6*(kv*1000)**2)
    df = .5*(dfu+dfv+(dfu-dfv)*np.cos(2*(np.arctan2(fy,fx)-np.deg2rad(angle))))
    gamma = 2*np.pi*(-.5*df*wavelength*s2 + .25*cs*1e7*wavelength**3*s2**2)-np.deg2rad(phase)
    return (np.sqrt(1-amp**2)*np.sin(gamma)-amp*np.cos(gamma)).astype(np.float32)


def project(fvolume, rotation):
    n = fvolume.shape[0]
    axis = np.arange(n)-n//2
    kx, ky = np.meshgrid(axis, axis, indexing='xy')
    points = np.stack((kx, ky, np.zeros_like(kx)), -1) @ rotation
    coords = np.moveaxis(points[..., ::-1]+n//2, -1, 0)
    plane = map_coordinates(fvolume, coords, order=1, mode='grid-wrap', prefilter=False)
    plane[kx*kx+ky*ky >= (n/2)**2] = 0
    inv = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(plane)))
    imag_ratio = np.linalg.norm(inv.imag)/max(np.linalg.norm(inv.real), 1e-30)
    if imag_ratio > 1e-5:
        raise ValueError(f'Projection is not Hermitian: imaginary/real={imag_ratio}')
    return inv.real.astype(np.float32)


def load_map(path, apix):
    with mrcfile.open(path, permissive=False) as handle:
        volume = np.array(handle.data, dtype=np.float32, copy=True)
        voxel = np.array([handle.voxel_size[x] for x in ('x', 'y', 'z')], dtype=float)
        origin = np.array([handle.header.origin[x] for x in ('x', 'y', 'z')], dtype=float)
        axes = [int(handle.header[x]) for x in ('mapc', 'mapr', 'maps')]
        starts = [int(handle.header[x]) for x in ('nxstart', 'nystart', 'nzstart')]
    if volume.ndim != 3 or len(set(volume.shape)) != 1 or volume.shape[0] < 8 or volume.shape[0] % 2:
        raise ValueError('Map must be an even cubic grid with side >= 8')
    if not np.isfinite(volume).all() or not np.isfinite(voxel).all() or not np.allclose(voxel, apix, atol=1e-6, rtol=0):
        raise ValueError('Map must be finite and isotropic with the requested apix')
    if axes != [1, 2, 3] or starts != [0, 0, 0]:
        raise ValueError('Map must use standard MRC x/y/z axes and zero start indices; resample explicitly')
    if not np.isfinite(origin).all():
        raise ValueError('Nonfinite map origin')
    if float(np.linalg.norm(volume)) == 0:
        raise ValueError('Map has no signal')
    return volume, dict(shape=list(volume.shape), origin_A=origin.tolist(), voxel_A=voxel.tolist(),
        array_axes='zyx', map_center_A=(origin + volume.shape[0]//2 * apix).tolist())


def write_star(frame, path):
    # Numeric/string loop independent of installed starfile writer version.
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        handle.write('data_images\n\nloop_\n')
        for i, column in enumerate(frame.columns, 1):
            handle.write(f'_{column} #{i}\n')
        frame.to_csv(handle, sep=' ', index=False, header=False, float_format='%.10g')
        handle.write('\n')


def validate_particles(directory):
    """Read every particle through the production reader, not a mock reader."""
    from particledataset import ParticleDataset
    directory = Path(directory)
    # starfile 0.4.x uses linecache; validate the current bytes if the caller
    # checks a file again after an edit.
    linecache.checkcache(str(directory / 'particles.star'))
    meta = np.load(directory / 'particle_metadata.npz', allow_pickle=False)
    rotations, ctfs = meta['rotations'], meta['ctfs']
    n, apix = int(ctfs[0, 0]), float(ctfs[0, 1])
    data = ParticleDataset(str(directory / 'particles.star'), '', apix, norm=False)
    data.validate(n)
    if len(data) != len(rotations):
        raise ValueError('STAR/metadata particle count mismatch')
    max_rotation = max_ctf = 0.
    with mrcfile.mmap(directory / 'particles.mrcs', mode='r') as stack:
        if stack.data.shape != (len(data), n, n):
            raise ValueError('MRCS count/box mismatch')
        for i in range(len(data)):
            image, ctf, shift, rotation, inverse, index = data[i]
            expected = ctfs[i, [5, 2, 3, 4, 6, 7, 8, 1]].astype(np.float64)
            expected[[0, 4]] *= [1000, 1e7]
            expected[[3, 6]] = np.deg2rad(expected[[3, 6]])
            np.testing.assert_array_equal(image, stack.data[i])
            np.testing.assert_array_equal(shift, [0., 0.])
            np.testing.assert_allclose(ctf, expected, atol=2e-4, rtol=2e-7)
            np.testing.assert_allclose(rotation, rotations[i, :2], atol=2e-6, rtol=0)
            np.testing.assert_allclose(inverse, rotations[i].T, atol=2e-6, rtol=0)
            assert int(index) == i
            max_rotation = max(max_rotation, float(abs(rotation-rotations[i, :2]).max()))
            max_ctf = max(max_ctf, float(abs(ctf-expected).max()))
    return dict(passed=True, checked_particles=len(data), max_rotation_error=max_rotation, max_ctf_error=max_ctf,
                images='all exact', shifts='all zero', reader='public ParticleDataset')


def simulate(args):
    started = time.monotonic()
    for key in ('apix', 'snr', 'voltage', 'defocus_min', 'defocus_max'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not isinstance(args.n_particles, int) or args.n_particles < 1 or not 0 <= args.seed < 2**32:
        raise ValueError('Positive particle count and seed in [0, 2**32) required')
    if not 0 <= args.amplitude_contrast <= 1 or not np.isfinite(args.cs) or args.cs < 0 or not np.isfinite(args.phase_shift):
        raise ValueError('Invalid CTF amplitude, Cs or phase')
    if not 0 <= args.astigmatism_min <= args.astigmatism_max < 2*args.defocus_min or args.defocus_min > args.defocus_max:
        raise ValueError('Invalid defocus/astigmatism range')
    volume, grid = load_map(args.map, args.apix)
    n, count = volume.shape[0], args.n_particles
    rng = np.random.default_rng(args.seed)
    rotations = Rotation.random(count, random_state=rng).as_matrix().astype(np.float32)
    dfs = rng.uniform(args.defocus_min, args.defocus_max, count)
    astig = rng.uniform(args.astigmatism_min, args.astigmatism_max, count)
    ctfs = np.column_stack((np.full(count,n), np.full(count,args.apix), dfs+astig/2, dfs-astig/2,
        rng.uniform(0,180,count), np.full(count,args.voltage), np.full(count,args.cs),
        np.full(count,args.amplitude_contrast), np.full(count,args.phase_shift))).astype(np.float32)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', command=list(getattr(args, 'original_argv', sys.argv)),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_map=dict(path=str(Path(args.map).resolve()), sha256=sha256(args.map)), grid=grid,
        n_particles=count, boxsize=n, apix=args.apix, particle_sign=-1, shifts_pixels=0,
        requested_snr=args.snr, seed=args.seed, created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        simulator_sha256=sha256(__file__), python=sys.version,
        versions={name:metadata.version(name) for name in ('numpy','scipy','mrcfile','starfile')},
        snr_definition='mean per-image variance of noiseless CTF signal / one common noise variance; circular mask radius 0.85*D/2',
        projector='centered Fourier slice, linear interpolation, radial Nyquist cutoff',
        note='Single state; no paired-state labels or private dependency; not byte-identical to the old two-state RNG sequence.')
    def save_report():
        (out/'simulation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save_report()
    try:
        axis = np.arange(n)-n//2
        xx, yy = np.meshgrid(axis, axis, indexing='xy')
        mask = xx*xx + yy*yy < (.85*n/2)**2
        fvolume = centered_fft(volume).astype(np.complex64)
        expected = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(centered_fft(volume.sum(0))*(xx*xx+yy*yy<(n/2)**2)))).real
        np.testing.assert_allclose(project(fvolume, np.eye(3)), expected, rtol=2e-5, atol=2e-5)
        signal = 0.
        with mrcfile.new_mmap(out/'clean_ctf.mrcs', shape=(count,n,n), mrc_mode=2, overwrite=False) as clean:
            clean.voxel_size = args.apix
            clean.header.label[:] = b''
            clean.header.label[0] = b'CoCoFold2 noiseless CTF particles; provenance in simulation.json'
            clean.header.nlabl = 1
            for i in range(count):
                projected = project(fvolume, rotations[i])
                image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(centered_fft(projected)*ctf_grid(ctfs[i])))).real
                image -= image.mean()
                clean.data[i] = -image
                signal += np.var(clean.data[i][mask], dtype=np.float64)
                if (i+1) % 100 == 0:
                    print(f'Projected {i+1}/{count}', flush=True)
            clean.update_header_stats()
            signal /= count
            if not np.isfinite(signal) or signal <= 0:
                raise ValueError('No finite nonzero projected CTF signal')
            sigma = np.sqrt(signal / args.snr)
            noise_variance = 0.
            with mrcfile.new_mmap(out/'particles.mrcs', shape=(count,n,n), mrc_mode=2, overwrite=False) as stack:
                stack.voxel_size = args.apix
                stack.header.label[:] = b''
                stack.header.label[0] = b'CoCoFold2 noisy particles; provenance in simulation.json'
                stack.header.nlabl = 1
                for i in range(count):
                    noise = rng.normal(0, sigma, size=(n,n)).astype(np.float32)
                    stack.data[i] = clean.data[i] + noise
                    noise_variance += np.var(noise[mask], dtype=np.float64)
                stack.update_header_stats()
        angles = Rotation.from_matrix(rotations.transpose(0,2,1)).as_euler('ZYZ', degrees=True)
        frame = pd.DataFrame(dict(rlnImageName=[f'{i+1}@particles.mrcs' for i in range(count)],
            rlnAngleRot=angles[:,0], rlnAngleTilt=angles[:,1], rlnAnglePsi=angles[:,2],
            rlnOriginX=np.zeros(count), rlnOriginY=np.zeros(count), rlnVoltage=ctfs[:,5],
            rlnDefocusU=ctfs[:,2], rlnDefocusV=ctfs[:,3], rlnDefocusAngle=ctfs[:,4],
            rlnSphericalAberration=ctfs[:,6], rlnAmplitudeContrast=ctfs[:,7], rlnPhaseShift=ctfs[:,8]))
        write_star(frame, out/'particles.star')
        np.savez(out/'particle_metadata.npz', rotations=rotations, ctfs=ctfs)
        report.update(validation=validate_particles(out), signal_variance=signal, noise_sigma=float(sigma),
            realized_snr=float(signal/(noise_variance/count)), elapsed_seconds=time.monotonic()-started,
            outputs={name: sha256(out/name) for name in ('particles.star','particles.mrcs','clean_ctf.mrcs','particle_metadata.npz')}, status='completed')
        save_report()
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        save_report()
        raise
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map', type=Path, required=True, help='Finite, cubic MRC density map (standard axes).')
    p.add_argument('--out-dir', type=Path, required=True, help='New directory, never overwritten.')
    p.add_argument('--n-particles', type=int, default=1000, help='Number of simulated single-state particles to generate. Default: %(default)s.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for this operation; does not modify previously generated input files. Default: %(default)s.')
    p.add_argument('--apix', type=float, default=1., help='Angstrom/pixel; must match map voxel size.')
    p.add_argument('--snr', type=float, default=1., help='Masked signal variance / noise variance.')
    p.add_argument('--defocus-min', type=float, default=10000., help='Angstrom')
    p.add_argument('--defocus-max', type=float, default=20000., help='Angstrom')
    p.add_argument('--astigmatism-min', type=float, default=200., help='Defocus U-V minimum, Angstrom')
    p.add_argument('--astigmatism-max', type=float, default=1000., help='Defocus U-V maximum, Angstrom')
    p.add_argument('--voltage', type=float, default=300., help='kV')
    p.add_argument('--cs', type=float, default=2.7, help='mm')
    p.add_argument('--amplitude-contrast', type=float, default=.1, help='CTF amplitude-contrast fraction (dimensionless). Default: %(default)s.')
    p.add_argument('--phase-shift', type=float, default=0., help='degrees')
    return p


if __name__ == '__main__':
    parsed = build_parser().parse_args()
    parsed.original_argv = sys.argv
    print(json.dumps(simulate(parsed), indent=2))
