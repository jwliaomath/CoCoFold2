# Public CLI reference

## Fresh GMM width options

Both `src/train.py` and `src/chain_parallel/train_chain_parallel_2d.py` additionally
accept the following options (the help snapshots below predate this extension):

| Option | Default | Meaning |
|---|---|---|
| `--gmm-sdev-init-mode {legacy,molmap}` | `legacy` | Choose fresh width initialization; saved GMM tensors take precedence |
| `--gmm-molmap-resolution-A FLOAT` | None | Finite positive molmap width resolution in Angstrom, required only for `molmap` |

`--resolution` remains a legacy coordinate/grid scale parameter, not generally
a molmap resolution in Angstrom. See [Gaussian kernels](gaussian_kernels.md#optional-physical-width-initialization)
for the mapping and the distinction from amplitude normalization.

## Projection frame options (`src/train.py` and component-parallel trainer)

| Option | Default | Meaning |
|---|---|---|
| `--projection-frame {legacy,fixed}` | `fixed` | Fixed 3D reference frame for new runs; choose `legacy` for historical per-image recentering and normalization |
| `--projection-origin X_A Y_A Z_A` | Required for fixed; omitted for legacy | Fixed-frame 3D reference point in the placed CIF/map frame, in Angstrom; no universal numerical default |

These options are distinct from `--gmm-kernel legacy` and
`--gmm-sdev-init-mode legacy`. See [projection-frame behavior](parameter_guide.md#projection-frame-single-gpu-and-component-parallel-refinement)
for the placement and normalization differences. Component-parallel ranks
render into the same fixed frame before their images are summed. For the
zero-origin 288-pixel/1.073 Å map in the 6ZBH examples, use
`--projection-origin 154.512 154.512 154.512` after checking the map/CIF frame.
The centered 7ZDT/7ZD5 molmap example uses `--projection-origin 0 0 0`.

## Existing CLI help snapshots

Generated from the public entrypoints with `--help`; no model or weights were loaded.
The help snapshots below predate the additional GMM width and projection-frame
options described above.
Run from the repository root. Required input paths have no usable default.
Protenix configuration flags forwarded by inference use the installed upstream configuration;
the list below covers the CoCoFold2 parser. For resolved settings consult run records.
Single-GPU independent data seed requires isolated RNG; parallel uses a dedicated data generator.
See [outputs and restart](outputs_and_restart.md) for checkpoint-derived defaults.

## src/train.py

```text
usage: train.py [-h] [--resume | --warm-start] [--seed SEED]
                [--diffusion-seed DIFFUSION_SEED] [--data-seed DATA_SEED]
                [--rng-mode {legacy,isolated}] [--record-dir RECORD_DIR]
                [--submission-script SUBMISSION_SCRIPT]
                [--output-format {cif,pdb,both}]
                [--learn-gmm | --no-learn-gmm] --star_data_dir STAR_DATA_DIR
                [--mrc_data_dir MRC_DATA_DIR] --output_trained_model_dir
                OUTPUT_TRAINED_MODEL_DIR --cif_path CIF_PATH
                --diffusion_data_dir DIFFUSION_DATA_DIR [--transR]
                [--particle_sign PARTICLE_SIGN] [--boxsize BOXSIZE]
                [--apix APIX] [--norm] [--resolution RESOLUTION]
                [--density_center DENSITY_CENTER DENSITY_CENTER]
                [--train_deterministic | --no-train_deterministic | --train-deterministic | --no-train-deterministic]
                [--device DEVICE] [--batch_size BATCH_SIZE]
                [--mini_batch_size MINI_BATCH_SIZE] [--update_affine_mat]
                [--map_resolution MAP_RESOLUTION] [--halfmap1 HALFMAP1]
                [--halfmap2 HALFMAP2] [--fsc_gamma FSC_GAMMA]
                [--fsc_smooth_win FSC_SMOOTH_WIN] [--epochs EPOCHS]
                [--max_steps MAX_STEPS] [--lr_bias LR_BIAS]
                [--lr_atom_weights LR_ATOM_WEIGHTS] [--lr_sdevs LR_SDEVS]
                [--check-inputs] [--gmm-kernel {legacy,isotropic,anisotropic}]
                [--gmm-amplitude {auto,peak_2d,reference_mass,peak_3d}]
                [--gmm-sigma-floor GMM_SIGMA_FLOOR]
                [--gmm-atom-chunk-size GMM_ATOM_CHUNK_SIZE]
                [--gmm-checkpoint-chunks | --no-gmm-checkpoint-chunks]
                [--gmm-checkpoint-peak2d | --no-gmm-checkpoint-peak2d]
                [--coordinate-mode {global,blocks}]
                [--block-alignment BLOCK_ALIGNMENT]
                [--alignment-sampler {auto,global,block}]
                [--coordinate-handoff-tolerance COORDINATE_HANDOFF_TOLERANCE]
                [--block-update-trace-threshold BLOCK_UPDATE_TRACE_THRESHOLD]

Particle-guided CoCoFold2 latent refinement.

options:
  -h, --help            show this help message and exit
  --resume              Continue a completed-epoch checkpoint supplied with
                        --diffusion_data_dir; --epochs is the total target.
  --warm-start          Start a new run from saved latent/GMM/placement; reset
                        optimizer and progress.
  --seed SEED           Run seed (42 for new runs; export inherits recorded
                        diffusion seed, else 42).
  --diffusion-seed DIFFUSION_SEED
                        Diffusion noise seed; train defaults to --seed,
                        inference to 42.
  --data-seed DATA_SEED
                        Particle shuffle seed override; single-GPU train
                        requires isolated RNG, parallel uses its dedicated
                        data generator.
  --rng-mode {legacy,isolated}
                        legacy preserves global RNG side effects; isolated
                        keeps sampling RNG separate.
  --record-dir RECORD_DIR
                        New directory for this invocation; existing
                        directories are rejected.
  --submission-script SUBMISSION_SCRIPT
                        Copy this submitted shell/Slurm script verbatim into
                        the run record.
  --output-format {cif,pdb,both}, --output_format {cif,pdb,both}
                        Training structure format, including initial and epoch
                        outputs (default: cif).
  --learn-gmm, --no-learn-gmm
                        Learn both atom amplitudes and widths; --no-learn-gmm
                        freezes both, retaining latent gradients.
  --star_data_dir STAR_DATA_DIR
                        RELION STAR file containing particle image references,
                        poses and CTF metadata. Default: None.
  --mrc_data_dir MRC_DATA_DIR
                        Root for relative STAR image paths; omitted means the
                        STAR directory. Absolute image paths are used
                        directly. Default: None.
  --output_trained_model_dir OUTPUT_TRAINED_MODEL_DIR
                        Legacy output filename prefix; a trailing slash means
                        a directory.
  --cif_path CIF_PATH   Initial model placed in the experimental coordinate
                        frame.
  --diffusion_data_dir DIFFUSION_DATA_DIR
                        Diffusion .pth cache; use explicit resume/warm-start
                        options for refinement checkpoints. Default: None.
  --transR              Use the validated pose-convention matrix diag(1,1,-1);
                        enable only for the matching upstream orientation
                        convention. Default: False.
  --particle_sign PARTICLE_SIGN
                        Multiplier applied to rendered particle projections;
                        keep the validated data sign convention. Default:
                        -1.0.
  --boxsize BOXSIZE     Square particle image width/height in pixels; must
                        match STAR/MRCS inputs. Default: 256.
  --apix APIX           Experimental pixel size in Angstrom per pixel; must be
                        positive. Default: 1.0.
  --norm                Min-max normalize each observed particle to [0,1];
                        constant images are rejected. Default: False.
  --resolution RESOLUTION
                        GMM rendering resolution parameter in Angstrom;
                        distinct from the FRC cutoff. Default: 3.0.
  --density_center DENSITY_CENTER DENSITY_CENTER
                        Two image-center coordinates in pixels; omitted uses
                        the box center. Default: None.
  --train_deterministic, --no-train_deterministic, --train-deterministic, --no-train-deterministic
                        Reuse fixed diffusion stochasticity; disabling
                        resamples noise. Per-chain placement requires fixed
                        stochasticity. Default: True.
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
  --batch_size BATCH_SIZE
                        Particles per optimizer update; all parallel ranks
                        process the same batch. Default: 32.
  --mini_batch_size MINI_BATCH_SIZE
                        Particles per loss/backward microbatch inside each
                        update; smaller values trade memory for more decoding.
                        Default: 12.
  --update_affine_mat   Enable the existing rigid-transform update safeguard;
                        per-chain mode tests the shared threshold
                        independently per chain. Default: False.
  --map_resolution MAP_RESOLUTION
                        Angstrom resolution cutoff of the active particle FRC
                        objective; not the GMM rendering width. Default: 5.0.
  --halfmap1 HALFMAP1   Optional first half-map file, paired with halfmap2.
                        Weights are prepared but not consumed by the current
                        active FRC loss. Default: None.
  --halfmap2 HALFMAP2   Optional second half-map file, paired with halfmap1;
                        current active FRC does not consume the prepared
                        weights. Default: None.
  --fsc_gamma FSC_GAMMA
                        Exponent for optional half-map weights; these weights
                        currently do not alter the active FRC objective.
                        Default: 1.0.
  --fsc_smooth_win FSC_SMOOTH_WIN
                        Nonnegative smoothing window for optional half-map
                        weights; zero disables smoothing. Default: 0.
  --epochs EPOCHS       Cumulative target epoch count; resume counts already
                        completed epochs toward this target. Default: 10.
  --max_steps MAX_STEPS, --max-steps MAX_STEPS
                        Total optimizer-step limit, including resumed steps; a
                        partial-epoch save supports export/warm-start only.
  --lr_bias LR_BIAS, --lr-bias LR_BIAS
                        AdamW learning rate for the target-specific latent
                        perturbation. Default: 0.01.
  --lr_atom_weights LR_ATOM_WEIGHTS, --lr-atom-weights LR_ATOM_WEIGHTS
                        AdamW learning rate for GMM amplitudes; unused when
                        GMM learning is disabled. Default: 0.01.
  --lr_sdevs LR_SDEVS, --lr-sdevs LR_SDEVS
                        AdamW learning rate for GMM widths/shape parameters;
                        unused when GMM learning is disabled. Default: 0.005.
  --check-inputs        Validate inputs on CPU and exit without constructing
                        the diffusion model.
  --gmm-kernel {legacy,isotropic,anisotropic}
                        Gaussian kernel parameterization; legacy preserves the
                        original width convention. Default: legacy.
  --gmm-amplitude {auto,peak_2d,reference_mass,peak_3d}
                        Amplitude convention; auto uses the selected kernel
                        convention. See docs/gaussian_kernels.md. Default:
                        auto.
  --gmm-sigma-floor GMM_SIGMA_FLOOR
                        Positive covariance width floor in renderer units,
                        used by the covariance kernels. Default: 0.0001.
  --gmm-atom-chunk-size GMM_ATOM_CHUNK_SIZE
                        Atom chunk size for new covariance kernels only
  --gmm-checkpoint-chunks, --no-gmm-checkpoint-chunks
                        Recompute new raster blocks during backward
  --gmm-checkpoint-peak2d, --no-gmm-checkpoint-peak2d
                        Opt-in recomputation for legacy/isotropic peak_2d;
                        preserves original 1000-atom chunks
  --coordinate-mode {global,blocks}
                        Original global alignment by default; saved block
                        checkpoints retain their mode
  --block-alignment BLOCK_ALIGNMENT
                        Block manifest JSON for FIRST initialization only
  --alignment-sampler {auto,global,block}
                        auto: new per-chain manifests use original train
                        sampler; old block checkpoints keep their saved
                        decoder. Explicit override only for new
                        initialization.
  --coordinate-handoff-tolerance COORDINATE_HANDOFF_TOLERANCE
                        Maximum raw-coordinate RMS difference in Angstrom at
                        block checkpoint handoff
  --block-update-trace-threshold BLOCK_UPDATE_TRACE_THRESHOLD
                        Shared threshold for optional per-body updates:
                        trace(R_new @ R_old.T) below this value triggers R/t
                        replacement; default 2.5, valid (-1,3)
```

## src/inference.py

```text
usage: inference.py [-h] [--diffusion-seed DIFFUSION_SEED]
                    [--rng-mode {legacy,isolated}] [--record-dir RECORD_DIR]
                    [--submission-script SUBMISSION_SCRIPT] --input_json_path
                    INPUT_JSON_PATH [--sample_name SAMPLE_NAME]
                    [--train-deterministic | --no-train-deterministic | --train_deterministic | --no-train_deterministic]
                    [--output_model_dir OUTPUT_MODEL_DIR]
                    [--dump_dir DUMP_DIR] [--gamma0 GAMMA0]
                    [--gamma_min GAMMA_MIN]
                    [--noise_scale_lambda NOISE_SCALE_LAMBDA]
                    [--step_scale_eta STEP_SCALE_ETA] [--N_step N_STEP]
                    [--N_sample N_SAMPLE]
                    [--N_step_mini_rollout N_STEP_MINI_ROLLOUT]
                    [--N_sample_mini_rollout N_SAMPLE_MINI_ROLLOUT]
                    [--save_pairformer_last_input [SAVE_PAIRFORMER_LAST_INPUT]]
                    [--resource-root RESOURCE_ROOT] [--check-inputs]

Protenix initial predictions and CoCoFold2 caches.

options:
  -h, --help            show this help message and exit
  --diffusion-seed DIFFUSION_SEED
                        Diffusion noise seed; train defaults to --seed,
                        inference to 42.
  --rng-mode {legacy,isolated}
                        legacy preserves global RNG side effects; isolated
                        keeps sampling RNG separate.
  --record-dir RECORD_DIR
                        New directory for this invocation; existing
                        directories are rejected.
  --submission-script SUBMISSION_SCRIPT
                        Copy this submitted shell/Slurm script verbatim into
                        the run record.
  --input_json_path INPUT_JSON_PATH
                        Protenix target JSON; may contain multiple targets.
                        Relative CLI paths use the working directory. Default:
                        None.
  --sample_name SAMPLE_NAME
                        Legacy cache name for one target and one seed;
                        otherwise use actual target names.
  --train-deterministic, --no-train-deterministic, --train_deterministic, --no-train_deterministic
                        Reuse fixed diffusion stochasticity; disabling
                        resamples noise. Default: True.
  --output_model_dir OUTPUT_MODEL_DIR
                        Cache directory; trailing slash is optional. Existing
                        caches are never overwritten.
  --dump_dir DUMP_DIR   Protenix prediction and inference-record output
                        directory; relative to the working directory. Default:
                        ./output.
  --gamma0 GAMMA0       Diffusion churn magnitude stored in the cache sampling
                        configuration. Default: 0.0.
  --gamma_min GAMMA_MIN
                        Noise-level threshold controlling diffusion churn.
                        Default: 0.0.
  --noise_scale_lambda NOISE_SCALE_LAMBDA
                        Multiplier for diffusion noise injection. Default:
                        1.003.
  --step_scale_eta STEP_SCALE_ETA
                        Scale of each diffusion integration update. Default:
                        1.0.
  --N_step N_STEP       Number of denoising integration steps saved in the
                        diffusion sampling configuration. Default: 5.
  --N_sample N_SAMPLE   Number of structure samples per target/seed;
                        refinement requires a compatible single-structure
                        cache. Default: 1.
  --N_step_mini_rollout N_STEP_MINI_ROLLOUT
                        Denoising steps for the saved mini-rollout
                        configuration. Default: 5.
  --N_sample_mini_rollout N_SAMPLE_MINI_ROLLOUT
                        Structure samples for the saved mini-rollout
                        configuration; separate from particle mini-batches.
                        Default: 5.
  --save_pairformer_last_input [SAVE_PAIRFORMER_LAST_INPUT]
                        Save the final Pairformer input for downstream cache
                        preparation; optional explicit boolean. Default:
                        False.
  --resource-root RESOURCE_ROOT
                        Default checkpoint/common root; explicit Protenix
                        paths take precedence.
  --check-inputs        Check basic JSON structure/names without Protenix or
                        downloads; excludes external MSA/template validation.
```

## src/get_pdb.py

```text
usage: get_pdb.py [-h] [--seed SEED] [--rng-mode {legacy,isolated}]
                  [--record-dir RECORD_DIR]
                  [--submission-script SUBMISSION_SCRIPT] --pdbid PDBID
                  --diffusion_data_dir DIFFUSION_DATA_DIR
                  [--cif_path CIF_PATH] [--output_format {cif,pdb,both}]
                  --out_dir OUT_DIR [--device DEVICE]
                  [--coordinate-frame {reference,raw}]

options:
  -h, --help            show this help message and exit
  --seed SEED           Run seed (42 for new runs; export inherits recorded
                        diffusion seed, else 42).
  --rng-mode {legacy,isolated}
                        legacy preserves global RNG side effects; isolated
                        keeps sampling RNG separate.
  --record-dir RECORD_DIR
                        New directory for this invocation; existing
                        directories are rejected.
  --submission-script SUBMISSION_SCRIPT
                        Copy this submitted shell/Slurm script verbatim into
                        the run record.
  --pdbid PDBID         Target PDB ID
  --diffusion_data_dir DIFFUSION_DATA_DIR
                        Path to the diffusion .pth data
  --cif_path CIF_PATH   Optional reference CIF/PDB topology; otherwise decode
                        cache features
  --output_format {cif,pdb,both}, --output-format {cif,pdb,both}
                        Default: CIF without template, historical format with
                        template
  --out_dir OUT_DIR     Directory to save the prediction
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
  --coordinate-frame {reference,raw}
                        Refinement: apply saved R/t by default; raw omits
                        placement. Initial caches stay raw.
```

## src/prepare_block_alignment.py

```text
usage: prepare_block_alignment.py [-h] --target TARGET
                                  (--atom-map ATOM_MAP | --by-chain)
                                  [--fit-atoms {ca,all}] --output OUTPUT

Build a portable atom-identity-based block manifest from an assembled model.

options:
  -h, --help            show this help message and exit
  --target TARGET       Assembled CIF/PDB retaining source atom identities
  --atom-map ATOM_MAP   Split atom_mapping.csv with
                        six_body_file/suggested_fit_core
  --by-chain            One alignment body per author chain; one whole cache
                        can optimize all bodies
  --fit-atoms {ca,all}  For --by-chain: fit C-alpha atoms (default ca) or all
                        atoms; retain all atoms in either case
  --output OUTPUT       New alignment manifest JSON path; parent directory is
                        created if needed. Default: None.
```

## src/simulate_particles.py

```text
usage: simulate_particles.py [-h] --map MAP --out-dir OUT_DIR
                             [--n-particles N_PARTICLES] [--seed SEED]
                             [--apix APIX] [--snr SNR]
                             [--defocus-min DEFOCUS_MIN]
                             [--defocus-max DEFOCUS_MAX]
                             [--astigmatism-min ASTIGMATISM_MIN]
                             [--astigmatism-max ASTIGMATISM_MAX]
                             [--voltage VOLTAGE] [--cs CS]
                             [--amplitude-contrast AMPLITUDE_CONTRAST]
                             [--phase-shift PHASE_SHIFT]

Single-map Fourier-slice particles with portable STAR and explicit provenance.
No Protenix or private research code is required. The centered projection/CTF
conventions preserve the original pilot; noise statistics use this single map.

options:
  -h, --help            show this help message and exit
  --map MAP             Finite, cubic MRC density map (standard axes).
  --out-dir OUT_DIR     New directory, never overwritten.
  --n-particles N_PARTICLES
                        Number of simulated single-state particles to
                        generate. Default: 1000.
  --seed SEED           Random seed for this operation; does not modify
                        previously generated input files. Default: 42.
  --apix APIX           Angstrom/pixel; must match map voxel size.
  --snr SNR             Masked signal variance / noise variance.
  --defocus-min DEFOCUS_MIN
                        Angstrom
  --defocus-max DEFOCUS_MAX
                        Angstrom
  --astigmatism-min ASTIGMATISM_MIN
                        Defocus U-V minimum, Angstrom
  --astigmatism-max ASTIGMATISM_MAX
                        Defocus U-V maximum, Angstrom
  --voltage VOLTAGE     kV
  --cs CS               mm
  --amplitude-contrast AMPLITUDE_CONTRAST
                        CTF amplitude-contrast fraction (dimensionless).
                        Default: 0.1.
  --phase-shift PHASE_SHIFT
                        degrees
```

## src/chain_parallel/train_chain_parallel_2d.py

```text
usage: train_chain_parallel_2d.py [-h] --component_manifest COMPONENT_MANIFEST
                                  --star_data_dir STAR_DATA_DIR
                                  [--mrc_data_dir MRC_DATA_DIR]
                                  --output_trained_model_dir
                                  OUTPUT_TRAINED_MODEL_DIR [--transR]
                                  [--particle_sign PARTICLE_SIGN]
                                  [--boxsize BOXSIZE] [--apix APIX] [--norm]
                                  [--resolution RESOLUTION]
                                  [--density_center DENSITY_CENTER DENSITY_CENTER]
                                  [--train_deterministic | --no-train_deterministic]
                                  [--device DEVICE] [--backend {nccl,gloo}]
                                  [--batch_size BATCH_SIZE]
                                  [--mini_batch_size MINI_BATCH_SIZE]
                                  [--update_affine_mat]
                                  [--map_resolution MAP_RESOLUTION]
                                  [--halfmap1 HALFMAP1] [--halfmap2 HALFMAP2]
                                  [--fsc_gamma FSC_GAMMA]
                                  [--fsc_smooth_win FSC_SMOOTH_WIN]
                                  [--epochs EPOCHS] [--max_steps MAX_STEPS]
                                  [--lr_bias LR_BIAS]
                                  [--lr_atom_weights LR_ATOM_WEIGHTS]
                                  [--lr_sdevs LR_SDEVS]
                                  [--learn-gmm | --no-learn-gmm]
                                  [--output-format {cif,pdb,both}]
                                  [--by-chain] [--fit-atoms {ca,all}]
                                  [--block-update-trace-threshold BLOCK_UPDATE_TRACE_THRESHOLD]
                                  [--check-inputs]
                                  [--distributed-timeout DISTRIBUTED_TIMEOUT]
                                  [--resume RESUME | --warm-start]
                                  [--seed SEED]
                                  [--diffusion-seed DIFFUSION_SEED]
                                  [--data-seed DATA_SEED]
                                  [--rng-mode {legacy,isolated}]
                                  [--record-dir RECORD_DIR]
                                  [--submission-script SUBMISSION_SCRIPT]
                                  [--gmm-kernel {legacy,isotropic,anisotropic}]
                                  [--gmm-amplitude {auto,peak_2d,reference_mass,peak_3d}]
                                  [--gmm-sigma-floor GMM_SIGMA_FLOOR]
                                  [--gmm-atom-chunk-size GMM_ATOM_CHUNK_SIZE]
                                  [--gmm-checkpoint-chunks | --no-gmm-checkpoint-chunks]
                                  [--gmm-checkpoint-peak2d | --no-gmm-checkpoint-peak2d]

One component cache per rank; component-wide or per-chain rigid placement.

options:
  -h, --help            show this help message and exit
  --component_manifest COMPONENT_MANIFEST
                        YAML assigning one component cache/reference CIF per
                        rank; embedded paths are relative to the manifest.
                        Default: None.
  --star_data_dir STAR_DATA_DIR
                        RELION STAR file containing particle image references,
                        poses and CTF metadata. Default: None.
  --mrc_data_dir MRC_DATA_DIR
                        Root for relative STAR image paths; omitted means the
                        STAR directory. Absolute image paths are used
                        directly. Default: None.
  --output_trained_model_dir OUTPUT_TRAINED_MODEL_DIR
                        Output filename prefix; a trailing slash selects a
                        directory. Use a new run location. Default: None.
  --transR              Use the validated pose-convention matrix diag(1,1,-1);
                        enable only for the matching upstream orientation
                        convention. Default: False.
  --particle_sign PARTICLE_SIGN
                        Multiplier applied to rendered particle projections;
                        keep the validated data sign convention. Default:
                        -1.0.
  --boxsize BOXSIZE     Square particle image width/height in pixels; must
                        match STAR/MRCS inputs. Default: 256.
  --apix APIX           Experimental pixel size in Angstrom per pixel; must be
                        positive. Default: 1.0.
  --norm                Min-max normalize each observed particle to [0,1];
                        constant images are rejected. Default: False.
  --resolution RESOLUTION
                        GMM rendering resolution parameter in Angstrom;
                        distinct from the FRC cutoff. Default: 3.0.
  --density_center DENSITY_CENTER DENSITY_CENTER
                        Two image-center coordinates in pixels; omitted uses
                        the box center. Default: None.
  --train_deterministic, --no-train_deterministic
                        Reuse fixed diffusion stochasticity; disabling
                        resamples noise. Per-chain placement requires fixed
                        stochasticity. Default: True.
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
  --backend {nccl,gloo}
                        Distributed communication backend: NCCL for CUDA
                        training, Gloo for CPU tests. Default: nccl.
  --batch_size BATCH_SIZE
                        Particles per optimizer update; all parallel ranks
                        process the same batch. Default: 32.
  --mini_batch_size MINI_BATCH_SIZE
                        Particles per loss/backward microbatch inside each
                        update; smaller values trade memory for more decoding.
                        Default: 12.
  --update_affine_mat   Enable the existing rigid-transform update safeguard;
                        per-chain mode tests the shared threshold
                        independently per chain. Default: False.
  --map_resolution MAP_RESOLUTION
                        Angstrom resolution cutoff of the active particle FRC
                        objective; not the GMM rendering width. Default: 5.0.
  --halfmap1 HALFMAP1   Optional first half-map file, paired with halfmap2.
                        Weights are prepared but not consumed by the current
                        active FRC loss. Default: None.
  --halfmap2 HALFMAP2   Optional second half-map file, paired with halfmap1;
                        current active FRC does not consume the prepared
                        weights. Default: None.
  --fsc_gamma FSC_GAMMA
                        Exponent for optional half-map weights; these weights
                        currently do not alter the active FRC objective.
                        Default: 1.0.
  --fsc_smooth_win FSC_SMOOTH_WIN
                        Nonnegative smoothing window for optional half-map
                        weights; zero disables smoothing. Default: 0.
  --epochs EPOCHS       Total target epochs, including resumed epochs.
  --max_steps MAX_STEPS
                        Stop after this cumulative optimizer step; partial
                        epoch is warm-start only.
  --lr_bias LR_BIAS     AdamW learning rate for the target-specific latent
                        perturbation. Default: 0.01.
  --lr_atom_weights LR_ATOM_WEIGHTS
                        AdamW learning rate for GMM amplitudes; unused when
                        GMM learning is disabled. Default: 0.01.
  --lr_sdevs LR_SDEVS   AdamW learning rate for GMM widths/shape parameters;
                        unused when GMM learning is disabled. Default: 0.005.
  --learn-gmm, --no-learn-gmm
                        Learn both GMM amplitudes and widths; --no-learn-gmm
                        freezes both without blocking coordinate gradients.
                        Default: True.
  --output-format {cif,pdb,both}
                        CIF always retained for merged output; pdb also writes
                        PDB.
  --by-chain            Each local CIF author chain fits independently within
                        the same component decoder.
  --fit-atoms {ca,all}  Rigid-fitting core within each local chain: ca uses
                        C-alpha, all uses all atoms; all atoms remain in the
                        output. Default: ca.
  --block-update-trace-threshold BLOCK_UPDATE_TRACE_THRESHOLD
                        Common rotation trace threshold (-1,3) for independent
                        per-chain affine updates. Default: 2.5.
  --check-inputs        Check all component inputs on CPU without constructing
                        models.
  --distributed-timeout DISTRIBUTED_TIMEOUT
                        Collective timeout in seconds; torchrun terminates
                        peers on worker failure.
  --resume RESUME       Complete epoch JSON index; fixed grouping/world size
                        and inherited science settings.
  --warm-start          Component manifest points to old/new refinement
                        checkpoints; reset optimizer/progress.
  --seed SEED           Run seed (42 for new runs; export inherits recorded
                        diffusion seed, else 42).
  --diffusion-seed DIFFUSION_SEED
                        Diffusion noise seed; train defaults to --seed,
                        inference to 42.
  --data-seed DATA_SEED
                        Particle shuffle seed override; single-GPU train
                        requires isolated RNG, parallel uses its dedicated
                        data generator.
  --rng-mode {legacy,isolated}
                        legacy preserves global RNG side effects; isolated
                        keeps sampling RNG separate.
  --record-dir RECORD_DIR
                        New directory for this invocation; existing
                        directories are rejected.
  --submission-script SUBMISSION_SCRIPT
                        Copy this submitted shell/Slurm script verbatim into
                        the run record.
  --gmm-kernel {legacy,isotropic,anisotropic}
                        Gaussian kernel parameterization; legacy preserves the
                        original width convention. Default: legacy.
  --gmm-amplitude {auto,peak_2d,reference_mass,peak_3d}
                        Amplitude convention; auto uses the selected kernel
                        convention. See docs/gaussian_kernels.md. Default:
                        auto.
  --gmm-sigma-floor GMM_SIGMA_FLOOR
                        Positive covariance width floor in renderer units,
                        used by the covariance kernels. Default: 0.0001.
  --gmm-atom-chunk-size GMM_ATOM_CHUNK_SIZE
                        Atom chunk size for new covariance kernels only
  --gmm-checkpoint-chunks, --no-gmm-checkpoint-chunks
                        Recompute new raster blocks during backward
  --gmm-checkpoint-peak2d, --no-gmm-checkpoint-peak2d
                        Opt-in recomputation for legacy/isotropic peak_2d;
                        preserves original 1000-atom chunks
```

## src/chain_parallel/prepare_contextual_diffusion_caches.py inspect

```text
usage: prepare_contextual_diffusion_caches.py inspect [-h] --cache CACHE
                                                      [--output-json OUTPUT_JSON]

options:
  -h, --help            show this help message and exit
  --cache CACHE         Existing diffusion cache to inspect on CPU. Default:
                        None.
  --output-json OUTPUT_JSON
                        Optional inspection JSON output; otherwise print the
                        inspection. Default: None.
```

## src/chain_parallel/prepare_contextual_diffusion_caches.py split

```text
usage: prepare_contextual_diffusion_caches.py split [-h] --spec SPEC
                                                    [--device DEVICE]
                                                    [--overwrite]

options:
  -h, --help       show this help message and exit
  --spec SPEC      Contextual split YAML; paths inside it are resolved
                   relative to that YAML. Default: None.
  --device DEVICE  PyTorch device for this operation; distributed CUDA ranks
                   use LOCAL_RANK. Default: cpu.
  --overwrite      Allow replacing existing prepared cache outputs; disabled
                   unless explicitly supplied. Default: False.
```

## src/chain_parallel/materialize_local_diffusion_cache.py

```text
usage: materialize_local_diffusion_cache.py [-h] --input-cache INPUT_CACHE
                                            --output-cache OUTPUT_CACHE
                                            [--device DEVICE]
                                            [--report-json REPORT_JSON]
                                            [--chunk-elements CHUNK_ELEMENTS]
                                            [--overwrite]

Convert one split contextual z_trunk cache into a component-local shared
pair_z/p_lm/c_l cache

options:
  -h, --help            show this help message and exit
  --input-cache INPUT_CACHE
                        Input component cache containing local z_trunk; not
                        modified. Default: None.
  --output-cache OUTPUT_CACHE
                        Destination cache with materialized pair_z, p_lm and
                        c_l; refuses overwrite unless explicitly allowed.
                        Default: None.
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
  --report-json REPORT_JSON
                        Optional JSON report path, resolved from the working
                        directory. Default: None.
  --chunk-elements CHUNK_ELEMENTS
                        Number of tensor elements per chunk in numerical
                        comparison/reporting. Default: 1000000.
  --overwrite           Allow replacing existing prepared cache outputs;
                        disabled unless explicitly supplied. Default: False.
```

## examples/7zdt_7zd5/run_case.py predict

```text
usage: run_case.py predict [-h] --input-json INPUT_JSON --resource-root
                           RESOURCE_ROOT --output OUTPUT [--seed SEED]
                           [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --input-json INPUT_JSON
                        Protenix input JSON, including sequence/MSA paths;
                        relative path uses the current working directory.
                        Default: None.
  --resource-root RESOURCE_ROOT
                        Directory containing Protenix common/ and checkpoint/
                        resources. Default: None.
  --output OUTPUT       Output path for this command; relative paths use the
                        working directory. Existing training output
                        directories are refused. Default: None.
  --seed SEED           Initial export/refinement diffusion seed; prediction
                        seed retains inference default.
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
```

## examples/7zdt_7zd5/run_case.py train

```text
usage: run_case.py train [-h] --kind {smoke,refine} --prediction PREDICTION
                         --particles PARTICLES --placed-cif PLACED_CIF
                         [--reference-structure] [--smoke-result SMOKE_RESULT]
                         --output OUTPUT [--mini-batch-size MINI_BATCH_SIZE]
                         [--seed SEED] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --kind {smoke,refine}
                        smoke uses the small particle subset; refine uses all
                        simulated particles and requires successful smoke.
                        Default: None.
  --prediction PREDICTION
                        Completed prediction-stage directory containing the
                        cache and exported initial structure. Default: None.
  --particles PARTICLES
                        Particle simulation output directory containing
                        particles.star and its report. Default: None.
  --placed-cif PLACED_CIF
                        Initial structure rigidly placed in the map frame,
                        preserving cache atom identities/order. Default: None.
  --reference-structure
                        Accept a compatible reference conformation; otherwise
                        require a rigidly moved copy of the prediction.
  --smoke-result SMOKE_RESULT
                        Successful smoke directory required before refine with
                        the same inputs.
  --output OUTPUT       Output path for this command; relative paths use the
                        working directory. Existing training output
                        directories are refused. Default: None.
  --mini-batch-size MINI_BATCH_SIZE
                        Keep fixed for comparison; changing this can change
                        legacy loss scaling.
  --seed SEED           Random seed for this operation; does not modify
                        previously generated input files. Default: 42.
  --device DEVICE       PyTorch device for this operation; distributed CUDA
                        ranks use LOCAL_RANK. Default: cuda:0.
```

## examples/7zdt_7zd5/run_case.py validate

```text
usage: run_case.py validate [-h] --output OUTPUT [--device DEVICE]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Existing smoke/refine directory to audit; does not rerun
                   training.
  --device DEVICE  PyTorch device for this operation; distributed CUDA ranks
                   use LOCAL_RANK. Default: cpu.
```

## examples/6zbh_parallel/run_case.py run

```text
usage: run_case.py run [-h] --manifest MANIFEST --star STAR --mrc-dir MRC_DIR
                       --resource-root RESOURCE_ROOT --output OUTPUT
                       [--epochs EPOCHS]
                       [--submission-script SUBMISSION_SCRIPT] [--dry-run]

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST   Two-component YAML; embedded paths are manifest-
                        relative.
  --star STAR           Full STAR file; no particle truncation.
  --mrc-dir MRC_DIR     Root for relative STAR image paths.
  --resource-root RESOURCE_ROOT
                        Directory containing common/ and checkpoint/.
  --output OUTPUT       New output directory; refuses overwrite.
  --epochs EPOCHS       Example target epochs; default 10.
  --submission-script SUBMISSION_SCRIPT
                        Optional submitted shell/Slurm script copied verbatim
                        into run records. Default: None.
  --dry-run             Print exact command without imports, files or GPU
                        work.
```

## examples/6zbh_parallel/run_case.py check

```text
usage: run_case.py check [-h] --output OUTPUT [--epochs EPOCHS]
                         [--completed-prefix] [--reason REASON]

options:
  -h, --help          show this help message and exit
  --output OUTPUT     Output path for this command; relative paths use the
                      working directory. Existing training output directories
                      are refused. Default: None.
  --epochs EPOCHS     Number of epochs to run or audit; partial-prefix
                      acceptance must be explicitly requested. Default: 10.
  --completed-prefix  Audit only the first N complete epochs; preserve
                      original target/status and write a separate acceptance
                      report.
  --reason REASON     Required with --completed-prefix; record why the
                      acceptance scope changed.
```

## tests/run_public_tests.py

```text
usage: run_public_tests.py [-h] --output OUTPUT [--timeout TIMEOUT]
                           [--groups {public,entrypoints,gmm,b6,b6_cli,b7b,b7b_cli,b8,b9} [{public,entrypoints,gmm,b6,b6_cli,b7b,b7b_cli,b8,b9} ...]]

Run the public T0-T3 CPU gate from a fresh allowlisted copy. Usage: python
tests/run_public_tests.py --output /path/to/new/results No GPU, Protenix
installation, weights, or research checkout is needed.

options:
  -h, --help            show this help message and exit
  --output OUTPUT       New results directory; never overwrite an earlier
                        report.
  --timeout TIMEOUT     Whole pytest process timeout in seconds (default:
                        900).
  --groups {public,entrypoints,gmm,b6,b6_cli,b7b,b7b_cli,b8,b9} [{public,entrypoints,gmm,b6,b6_cli,b7b,b7b_cli,b8,b9} ...]
                        Run only affected groups; omitted means the complete
                        public gate.
```

## tests/run_parallel_tests.py

```text
usage: run_parallel_tests.py [-h] --output OUTPUT [--timeout TIMEOUT]

Run T4 in a clean public copy: two CPU Gloo ranks, analytic denoiser only.

options:
  -h, --help         show this help message and exit
  --output OUTPUT    New results directory; refuses overwrite.
  --timeout TIMEOUT  Two-process test timeout in seconds (default: 300).
```

## tools/build_public_release.py

```text
usage: build_public_release.py [-h] --output OUTPUT

Copy only the public inventory into a new release directory and ZIP. Run from
any working directory. No Git operation or publishing is performed.

options:
  -h, --help       show this help message and exit
  --output OUTPUT  New delivery directory; existing directories are refused.
```

## tools/inspect_environment.py

```text
usage: inspect_environment.py [-h] --output OUTPUT [--model-imports]

Record installed dependencies and optionally check model imports in a fresh
environment. No weights are loaded. Does not modify or install any dependency.

options:
  -h, --help       show this help message and exit
  --output OUTPUT  New JSON report file; refuses overwrite.
  --model-imports  Import installed Protenix inference dependencies and
                   inspect visible GPUs; no weights.
```
