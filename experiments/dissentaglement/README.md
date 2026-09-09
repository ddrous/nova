# BinaryShapes video world models — v3

The scripts are intentionally notebook-style Python files with `#%%` cells and no `main()` wrappers.

## Model modes

Set only `model.mode` in `config.yaml`:

- `weight_ab`: weight-space representation + `A(z) + B(a)` FDM
- `weight_joint`: weight-space representation + monolithic joint FDM
- `standard_ab`: ConvNet latent/decoder + `A(z) + B(a)` FDM
- `standard_joint`: ConvNet latent/decoder + monolithic joint FDM

The numerical hyperparameters from the supplied `wm_train.py` are retained in the shared `train:` block.

## Phase 1 — encoder / representation

Run `wm_train_p1.py` first when you need to train or refresh an encoder. Phase 1 still creates a normal isolated run folder, exactly like the rest of the project:

```text
runs/<timestamp>-<mode>-p1/
```

That run folder contains the copied code/config, plots, loss histories, final evaluation, and the phase-1 checkpoint used for reproducibility. Phase 1 trains only the representation from frame reconstruction:

- weight-space modes: `WeightCNN`, including its `theta_base` renderer parameters
- standard modes: ConvNet encoder + its reconstruction decoder

The FDM and IDM are never optimized in phase 1.

After phase 1 completes, the final reusable representation is additionally exported to exactly one canonical file per representation family:

```text
encs/weight.eqx
encs/standard.eqx
```

There are no timestamped run folders inside `encs/`. For the standard model, `standard.eqx` is still a **single Equinox file**; it stores the learned encoder and decoder together because the decoder is part of the phase-1 reconstruction representation and must stay frozen in phase 2.

Set `phase_1.enabled: false` and `phase_1.load_dir` to evaluate an existing phase-1 run without training.

## Phase 2 — IDM + FDM

Run `wm_train_p2.py` after a suitable phase-1 encoder exists. Phase 2 selects the checkpoint only from the representation family:

- `weight_ab` / `weight_joint` -> `encs/weight.eqx`
- `standard_ab` / `standard_joint` -> `encs/standard.eqx`

You may override that location with `run.encoder_path`. If the required file is absent, or if its Equinox tree is incompatible with the current representation hyperparameters, phase 2 stops immediately with a message asking you to run phase 1 first. It does not silently initialise or retrain an encoder.

The loaded representation remains frozen; only the IDM/action model and FDM are optimized. Every phase-2 training run still gets its normal isolated folder:

```text
runs/<timestamp>-<mode>/
```

The run's copied `config.yaml` records the exact encoder checkpoint path used for training. Set `phase_2.enabled: false` and `run.load_dir` to evaluate an existing world-model run cleanly; evaluation loads the complete saved world model and does not depend on `encs/`.

## Training diagnostics

Both phases track losses at **every train step** and at epoch resolution. The full histories are exported as NPZ and CSV files. With the default 100 epochs, `plot_every` and `save_every` are computed as 10 epochs, so diagnostics are displayed and checkpoints are saved about ten times over training rather than every epoch.

Phase 1 displays reconstruction loss and IoU. Phase 2 displays every objective term: rollout pixel loss, latent-dynamics loss, VQ codebook loss, VQ commitment loss, and total loss, together with rollout IoU and representation/action diagnostics. Final figures use enlarged publication-style fonts and tick labels.

At the end of phase 2, the final GT/predicted rollout is saved as a GIF and displayed inline with Jupyter's `IPython.display` functions.

## Data

The loader calls `gendata.sample_video` on the fly for every sample. The default three-frame simulator has eight factors:

```text
shape, rotation, x0, y0, vx0, vy0, vx1, vy1
```

The safe position support guarantees the sprite stays inside the frame for every supported factor combination, and velocity values are checked to be multiples of four.

## Disentanglement evaluation

Use `wm_eval.py` with one or more phase-2 run folders. The representation is

```text
concat(z_0, a_0, a_1)
```

The fixed-factor design also holds the initial `(x0, y0)` fixed within each classifier feature, so random starting position cannot dominate the IDM-derived representation differences.
