# B6 test execution notes

For the step-by-step tutorial, see [README](README.md). These notes describe the test scope. It uses the original
7ZD5 CIF for a ChimeraX 3 Angstrom molmap on a 192-cubed, 1 Angstrom/pixel grid,
and the supplied 7ZDT JSON/MSA for `protenix_base_default_v1.0.0` prediction.
No target residue completion, mutation, common-atom trimming, or heterogeneity
code is used. Input CIFs are preserved unchanged.

1. Generate/obtain the map and check its `molmap.json` hash and coordinate frame.
   `make_molmap.py` runs inside ChimeraX with `COCOFOLD2_MOLMAP_CONFIG` pointing to
   a JSON containing `target_cif`, new `output_dir`, `boxsize: 192`, `apix: 1`
   and `resolution: 3`. The supplied map was generated locally. Its coordinates
   are centered and its origin is (-96, -96, -96) Angstrom.
2. Run `src/simulate_particles.py --map MAP --out-dir NEW_DIR --n-particles 1000
   --snr 1 --apix 1 --seed 42`. Every STAR row is checked through the real public
   reader. STAR paths are relative to its own directory. The clean CTF stack
   is retained for diagnostics; this doubles particle storage to about 281 MiB.
3. Run `run_case.py predict --input-json JSON --resource-root RESOURCES --output
   NEW_PREDICTION_DIR`. It uses existing v1 weights, the supplied JSON/MSA and
   unchanged inference sampling defaults. Initial prediction seed remains 101;
   the exported/refinement diffusion seed defaults to 42. It records both and
   stops after exporting the initial CIF. Existing outputs are never overwritten.
4. Fit the exported initial CIF to the supplied **map**, save only that atomic
   model to a new CIF, and inspect placement visually. Preserve its atom identities,
   number and order. Rigid motion is checked, but map-fitting quality is a user
   decision. This workflow does not substitute fitting to the target structure
   for fitting to the map.
5. Alternatively, use a pre-placed reference conformation with the same atom
   identities/order as the prediction and pass `--reference-structure`. This mode
   checks topology, not conformation equality. Same protein/PDB name alone does
   not establish compatibility: sequence length, missing residues, atom names,
   chain names, ligands and ordering must match. A reference bundled later must
   be checked against the accompanying params; an old Mini reference is not
   automatically compatible with a newly generated v1 cache.
6. Run `run_case.py train --kind smoke --prediction PREDICTION_DIR --particles
   PARTICLE_DIR --placed-cif PLACED_CIF --output NEW_SMOKE_DIR`. Smoke uses 32
   particles, two epochs/two optimizer steps, batch 32 and mini-batch 16.
7. Inspect `validation.json` and `validation.txt`. Only after successful smoke,
   run the same command with `--kind refine --smoke-result SMOKE_DIR` and a new
   output directory. Refine uses all 1000 particles, 10 epochs/320 optimizer
   steps. Both stages freeze GMM amplitudes/widths explicitly; train's default
   learning behavior, learning rates and legacy RNG/loss scaling are unchanged.
   Changing mini-batch size changes legacy loss scaling, so keep it consistent.

Runtime checks verify latent/optimizer finiteness and progress, exact GMM
freezing against its initializer, output/checkpoint consistency, and structured
records. Validation occurs in a fresh process after model memory is freed; it
does not repeat diffusion decoding. It reports unscaled FRC for the same eight
particles before/after training with a 3 Angstrom cutoff. This is a comparison
metric, not a change to the training objective or an automatic improvement
threshold. These eight particles are training views, not held-out validation.

RMSD is left to the user (e.g. US-align). A fitted RMSD within 2 Angstrom is the
suggested inspection target, **not an automated pass condition**. Include chain
mapping, aligned residues and coverage when reporting it. Sequences and atoms
of 7ZD5 and the modeled 7ZDT input may differ.

Run `python tests/run_public_tests.py --groups b6 b6_cli --output NEW_CPU_REPORT`
for affected CPU tests only. These use an analytic denoiser, not real weights.
Real prediction and refinement belong on a Slurm compute node; supplied server
scripts use a100_high and one GPU. The author subsequently accepted real smoke/refine in the existing environment. No runtime/VRAM guarantee or new-installation acceptance is inferred from that report.
