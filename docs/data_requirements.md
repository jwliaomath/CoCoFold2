# Data requirements for particle-guided CoCoFold2

This document describes the inputs consumed by the current particle-guided release. CoCoFold2 assumes that standard upstream cryo-EM processing has already produced particle orientations, translations and CTF estimates.

## Required inputs

### 1. Protenix input JSON

The JSON must be valid for the compatible Protenix 1.0.2 inference pipeline and reference the protein sequence and any MSA/template inputs required by that pipeline.

### 2. Protein sequence and Protenix resources

Provide all sequence, MSA and template resources referenced by the input JSON. Protenix model parameters are not distributed by CoCoFold2.

### 3. RELION-format STAR metadata

`ParticleDataset` supports two layouts:

- pre-RELION-3.1-style STAR metadata represented as one `images`-like table;
- RELION 3.1+ STAR metadata containing `optics` and `particles` blocks.

#### Pre-RELION-3.1 required columns

```text
rlnOriginX
rlnOriginY
rlnAngleRot
rlnAngleTilt
rlnAnglePsi
rlnVoltage
rlnDefocusU
rlnDefocusV
rlnDefocusAngle
rlnSphericalAberration
rlnAmplitudeContrast
rlnImageName
```

For this layout, the pixel size is supplied to CoCoFold2 through `--apix`.

#### RELION 3.1+ required optics columns

```text
rlnVoltage
rlnImagePixelSize
rlnSphericalAberration
rlnAmplitudeContrast
rlnOpticsGroup
```

#### RELION 3.1+ required particle columns

```text
rlnOriginXAngst
rlnOriginYAngst
rlnAngleRot
rlnAngleTilt
rlnAnglePsi
rlnDefocusU
rlnDefocusV
rlnDefocusAngle
rlnOpticsGroup
rlnImageName
```

Across the workflow, the important STAR fields include:

```text
rlnVoltage
rlnImagePixelSize
rlnSphericalAberration
rlnAmplitudeContrast
rlnOpticsGroup
rlnOriginXAngst
rlnOriginYAngst
rlnAngleRot
rlnAngleTilt
rlnAnglePsi
rlnDefocusU
rlnDefocusV
rlnDefocusAngle
rlnImageName
```

`rlnPhaseShift` and `rlnRandomSubset` are optional. If `rlnRandomSubset` is absent, all particles are assigned to subset 1 internally.

### 4. MRC/MRCS particle stack

Each `rlnImageName` entry uses a one-based `index@path/to/stack.mrcs` reference.
Absolute paths are used directly. Relative paths resolve against
`--mrc_data_dir` when supplied, otherwise against the STAR file's directory.
No trailing slash is required. Prefer a documented relative layout for portability.
The input preflight checks referenced stacks, image indices and header shape.
Pixel finiteness is checked when each particle is read; preflight is not a full
scan of every pixel in every stack.

### 5. Particle poses and translations

CoCoFold2 reads Euler angles and particle origins from the STAR file. The workflow does not optimize these quantities. Use `--transR` only when required by the orientation convention of the upstream processing workflow.

### 6. CTF parameters

Voltage, defocus values, defocus angle, spherical aberration, amplitude contrast and optional phase shift are read from STAR metadata and used to construct particle-specific CTFs. CoCoFold2 does not re-estimate CTF parameters.

### 7. Pixel size and box size

- `--boxsize` must match the particle images.
- For pre-RELION-3.1 metadata, `--apix` supplies the pixel size.
- For RELION 3.1+ metadata, `ParticleDataset` reads `rlnImagePixelSize` from the optics table; the command-line `--apix` is also used by the frequency-domain loss setup and should be consistent with the data.

### 8. Initial Protenix structure and topology

`get_pdb.py` can reconstruct topology from compatible cache features without a template. If `--cif_path` is supplied, its atom identities and topology must correspond to the generated coordinate tensor; use the matching Protenix-generated CIF/PDB. Do not substitute a deposited reference structure as this topology template.

### 9. Rigidly fitted initial model

Before refinement, the initial Protenix model must be rigidly placed once into the coordinate frame defined by the particle poses or reconstructed density. This fitted model is supplied to `train.py --cif_path` and is used as the optimization-frame topology and placement reference.

New runs use fixed-frame projection. Record the 3D map/CIF reference point in
Angstrom and pass it as `--projection-origin X_A Y_A Z_A`; the trainer cannot
infer it safely from a STAR file alone. Confirm the reconstructed map's origin,
starts, axis order, voxel size and box size before selecting the point. See
[choosing the projection origin](parameter_guide.md#choosing-the-projection-origin)
for the zero-origin-box-center and centered-molmap examples. Multi-GPU
component references must all use this same physical frame.

The deposited reference structure must not be used for this step in the reported experiment.

### 10. Cached diffusion tensor file

`inference.py --output_model_dir` writes a cache such as:

```text
params/6zbh_diffusion_data.pth
```

It contains the frozen diffusion-module state, conditional representations, cached `pair_z` or `z_trunk`, atom-level caches, noise schedule, configuration and initial prediction dictionary required for iterative refinement.
