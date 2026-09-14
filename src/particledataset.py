import os
from pathlib import Path
import mrcfile
import starfile
import numpy as np
from copy import copy
from torch.utils.data import Dataset
from utils import mrcread

class ParticleDataset(Dataset):
    '''
    Dataset class for particles.

    The parameters of particles, like ctfs, will be loaded when
    the object is created. However, the data of particles will not
    be loaded until the __getitem__ method is called.
    '''

    def __init__(self, star_path : str, data_dir : str = '', pixel_size : float = 1., transR = None, norm = False):
        if not os.path.exists(star_path):
            raise FileNotFoundError(f'{star_path} does not exist.')
        star = starfile.read(star_path, always_dict = True)
        self.star_path = Path(star_path).expanduser().resolve()
        self.data_dir = data_dir
        self.pixel_size = pixel_size
        self.norm = norm
        if transR is None:
            print('Check the alignment')
            self.transR = np.array([
            [1, 0, 0],
            [0, 1., 0],
            [0, 0, 1]
        ])
        else:
            self.transR = transR
        # <Relion 3.1
        # For supporting starfile>=0.5 in the future.
        if len(star) == 1 and (0 in star or '' in star or 'images' in star):
            self.version = 2
            self.particles = star[0] if 0 in star else star[''] if '' in star else star['images']
            if len(self.particles) == 0:
                raise ValueError(f'No particles in {star_path}')

            # Check keys.
            for key in ['rlnOriginX', 'rlnOriginY', 'rlnAngleRot', 'rlnAngleTilt',
                        'rlnAnglePsi', 'rlnVoltage', 'rlnDefocusU', 'rlnDefocusV',
                        'rlnDefocusAngle', 'rlnSphericalAberration',
                        'rlnAmplitudeContrast', 'rlnImageName']:
                if key not in self.particles:
                    raise ValueError(f'Key {key} missed in star file {star_path}.')

        # >=Relion 3.1
        elif len(star) == 2 and ('optics' in star and 'particles' in star):
            self.version = 3
            self.optics = star['optics']
            self.particles = star['particles']
            if len(self.particles) == 0:
                raise ValueError(f'No particles in {star_path}')

            # Check keys.
            for key in ['rlnVoltage', 'rlnImagePixelSize', 'rlnSphericalAberration',
                        'rlnAmplitudeContrast', 'rlnOpticsGroup']:
                if key not in self.optics:
                    raise ValueError(f'Key {key} missed in block data_optics in star file {star_path}.')

            for key in ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnAngleRot', 'rlnAngleTilt',
                        'rlnAnglePsi', 'rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle',
                        'rlnOpticsGroup', 'rlnImageName']:
                if key not in self.particles:
                    raise ValueError(f'Key {key} missed in block data_particles in star file {star_path}.')

        else:
            raise ValueError('Invalid particle star file.')

        self.indices = np.arange(len(self.particles), dtype = np.int32)
        self.subsets = self.particles['rlnRandomSubset'].to_numpy().astype(np.int32) \
                        if 'rlnRandomSubset' in self.particles \
                        else np.ones(len(self.particles), dtype = np.int32)

    def __len__(self) -> int:
        return len(self.indices)

    def image_location(self, image_name):
        """Resolve a one-based STAR index without relying on trailing slashes."""
        try:
            index, name = str(image_name).split('@', 1)
            index = int(index)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid rlnImageName {image_name!r}; expected index@stack') from exc
        if index < 1 or not name:
            raise ValueError(f'Invalid rlnImageName {image_name!r}; index must be >= 1')
        stack = Path(name).expanduser()
        if not stack.is_absolute():
            base = Path(self.data_dir).expanduser() if self.data_dir else self.star_path.parent
            stack = base / stack
        return index - 1, stack

    def validate(self, box_size=None):
        """Check metadata and each referenced stack header, without reading all pixels."""
        if not len(self):
            raise ValueError(f'No particles in {self.star_path}')
        if not np.isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError('Particle pixel size must be positive and finite')
        common = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnDefocusU',
                  'rlnDefocusV', 'rlnDefocusAngle']
        columns = common + (['rlnOriginX', 'rlnOriginY', 'rlnVoltage',
                             'rlnSphericalAberration', 'rlnAmplitudeContrast'] if self.version == 2
                            else ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOpticsGroup'])
        if 'rlnPhaseShift' in self.particles:
            columns.append('rlnPhaseShift')
        for column in columns:
            try:
                values = self.particles[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Non-numeric {column} in {self.star_path}') from exc
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite {column} in {self.star_path}')
        optical = self.particles if self.version == 2 else self.optics
        for column in ('rlnVoltage', 'rlnSphericalAberration', 'rlnAmplitudeContrast'):
            values = optical[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite {column} in {self.star_path}')
            if column == 'rlnVoltage' and np.any(values <= 0):
                raise ValueError('rlnVoltage must be positive')
            if column == 'rlnAmplitudeContrast' and np.any((values < 0) | (values > 1)):
                raise ValueError('rlnAmplitudeContrast must be in [0,1]')
        if self.version == 3:
            groups = self.optics['rlnOpticsGroup']
            group_values = groups.to_numpy(dtype=float)
            if (not np.isfinite(group_values).all() or np.any(group_values < 1)
                    or np.any(group_values != np.floor(group_values)) or groups.duplicated().any()):
                raise ValueError('Optics groups must be unique positive integers')
            if not self.particles['rlnOpticsGroup'].isin(groups).all():
                raise ValueError('Particle references a missing optics group')
            pixels = self.optics['rlnImagePixelSize'].to_numpy(dtype=float)
            if not np.isfinite(pixels).all() or not np.allclose(pixels, self.pixel_size, rtol=1e-5, atol=1e-6):
                raise ValueError('Optics pixel size differs from --apix; mixed pixel sizes are unsupported')
        headers = {}
        for image_name in self.particles.iloc[self.indices]['rlnImageName']:
            index, stack = self.image_location(image_name)
            key = str(stack.resolve())
            if key not in headers:
                if not stack.is_file():
                    raise FileNotFoundError(f'Particle stack does not exist: {stack}')
                with mrcfile.mmap(stack, mode='r') as mrc:
                    shape = tuple(mrc.data.shape)
                if len(shape) not in (2, 3) or shape[-1] != shape[-2]:
                    raise ValueError(f'Expected square 2D particles in {stack}, got {shape}')
                if box_size is not None and shape[-1] != box_size:
                    raise ValueError(f'Stack box size {shape[-1]} differs from --boxsize {box_size}: {stack}')
                headers[key] = (1 if len(shape) == 2 else shape[0], shape[-1])
            if index >= headers[key][0]:
                raise ValueError(f'Particle index out of range: {image_name}')
        return dict(n_particles=len(self), n_stacks=len(headers), box_sizes=sorted({x[1] for x in headers.values()}),
                    pixel_validation='deferred until each image is read',
                    stack_headers=[dict(path=key, n_images=value[0], box_size=value[1]) for key, value in headers.items()])

    def __getitem__(self, i : int):
        assert 0 <= i < len(self.indices)
        idx = self.indices[i]

        
        # Common parameters.
        image_index, stack_path = self.image_location(self.particles.loc[idx, 'rlnImageName'])
        psi         = np.radians(self.particles.loc[idx, 'rlnAngleRot'])
        theta       = np.radians(self.particles.loc[idx, 'rlnAngleTilt'])
        phi         = np.radians(self.particles.loc[idx, 'rlnAnglePsi'])
        qw          =  np.cos((phi + psi) / 2) * np.cos(theta / 2)
        qx          = -np.sin((phi - psi) / 2) * np.sin(theta / 2)
        qy          = -np.cos((phi - psi) / 2) * np.sin(theta / 2)
        qz          = -np.sin((phi + psi) / 2) * np.cos(theta / 2)
        defocusU    = self.particles.loc[idx, 'rlnDefocusU']
        defocusV    = self.particles.loc[idx, 'rlnDefocusV']
        astigmatism = np.radians(self.particles.loc[idx, 'rlnDefocusAngle'])
        phase_shift = np.radians(self.particles.loc[idx, 'rlnPhaseShift']) \
                    if 'rlnPhaseShift' in self.particles else 0.
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
        ])
        R2 = R.T
        R = R@self.transR
        R1 = R[:2,:]

        # Different versions.
        if self.version == 2:
            dx          = self.particles.loc[idx, 'rlnOriginX']
            dy          = self.particles.loc[idx, 'rlnOriginY']
            voltage     = self.particles.loc[idx, 'rlnVoltage'] * 1000
            Cs          = self.particles.loc[idx, 'rlnSphericalAberration'] * 1e7
            amplitude   = self.particles.loc[idx, 'rlnAmplitudeContrast']
            pixel_size  = self.pixel_size
        else:
            group       = self.particles.loc[idx, 'rlnOpticsGroup']
            row         = self.optics.query(f'rlnOpticsGroup == {group}')
            if len(row) == 0:
                raise ValueError(f'Optic group {group} does not exist.')
            elif len(row) > 1:
                raise ValueError(f'Find multiple optic group {group}.')
            row         = row.iloc[0]
            pixel_size  = row['rlnImagePixelSize']
            dx          = self.particles.loc[idx, 'rlnOriginXAngst'] / pixel_size
            dy          = self.particles.loc[idx, 'rlnOriginYAngst'] / pixel_size
            voltage     = row['rlnVoltage'] * 1000
            Cs          = row['rlnSphericalAberration'] * 1e7
            amplitude   = row['rlnAmplitudeContrast']
        data = mrcread(str(stack_path), image_index)
        data = np.array(data)
        if not np.isfinite(data).all():
            raise ValueError(f'Nonfinite particle pixels: {image_index + 1}@{stack_path}')
        if self.norm:
            if np.max(data) == np.min(data):
                raise ValueError(f'Cannot normalize a constant particle: {image_index + 1}@{stack_path}')
            # data = (data-np.mean(data))/(np.std(data)+1e-10)
            data = (data-np.min(data))/(np.max(data)-np.min(data))
        trans = np.array([dx,dy])
        return data, \
            np.array([voltage, defocusU, defocusV,
                        astigmatism, Cs, amplitude, phase_shift, pixel_size], dtype = np.float64),trans,R1,R2,np.array(idx)

    def save(self, output_path : str):
        if self.version == 2:
            starfile.write({'images' : self.particles.iloc[self.indices]}, output_path, overwrite = True)
        else:
            starfile.write({'optics' : self.optics, 'particles' : self.particles.iloc[self.indices]},
                           output_path, overwrite = True)

    def reset(self, subset = None):
        self.indices = np.arange(len(self.particles), dtype = np.int32) if subset is None else \
                       np.where(self.subsets == subset)[0]

    def subset(self, indices):
        sub = copy(self)
        sub.indices = self.indices[indices]
        return sub

    def split(self, mask):
        sub1, sub2 = copy(self), copy(self)
        sub1.indices = self.indices[mask]
        sub2.indices = self.indices[~mask]
        return sub1, sub2
    
    def get_subset_data(self):
        """Extracts the data used for averaging and replacement."""
        all_data = []
        for i in range(len(self)):
            data, *_ = self[i]
            all_data.append(data)
        return np.array(all_data)
