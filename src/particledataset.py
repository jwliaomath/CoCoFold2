import os
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

    def __getitem__(self, i : int):
        assert 0 <= i < len(self.indices)
        idx = self.indices[i]

        
        # Common parameters.
        slc, name, *_ = self.particles.loc[idx, 'rlnImageName'].split('@')
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
        data = mrcread(self.data_dir + name, int(slc) - 1)
        data = np.array(data)
        if self.norm:
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
