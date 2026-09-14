# BinaryShapes video world models

Notebook-style Python scripts (`#%%` cells), with no `main()` wrappers.

## Train

Set `model.mode` in `config.yaml` to one of:

- `weight_ab`: weight-space world model, `A(z) + B(a)` FDM.
- `weight_joint`: weight-space world model, joint monolithic FDM.
- `standard_ab`: standard ConvNet latent + decoder, `A(z) + B(a)` FDM.
- `standard_joint`: standard ConvNet latent + decoder, joint monolithic FDM.

This gives a clean 2×2 comparison: **representation/renderer** (`weight` vs `standard`) × **forward dynamics** (`ab` vs `joint`). All four modes use the same latent dimension and the same IDM/action dimensionality. The legacy mode name `standard` is accepted as an alias for `standard_joint` so older v1 run folders can still be evaluated.

Then run `train.py` (or execute its cells). Training creates `runs/<timestamp>-<mode>/` and copies the exact code/config into the run. The default simulator makes 3-frame videos, so the eight factors are:

`shape, rotation, x0, y0, vx0, vy0, vx1, vy1`.

The loader calls the simulator for every sample. Position supports are chosen conservatively so the full Cartesian product of factors remains inside the frame; velocity supports are validated to be multiples of four.

## Evaluate without training

Set:

```yaml
train:
  enabled: false
run:
  load_dir: runs/<your-run>
```

and execute `wm_train.py`. It loads `artefacts/model.eqx`, runs the fixed evaluation stream, and writes/updates final diagnostic outputs in that run.

## Disentanglement comparison

In `wm_eval.py`, edit `RUN_DIRS`, or pass run folders as command-line arguments. It uses the same simulator parameter pairs for every model.

A video representation is

`r(video) = concat(z_0, a_0, a_1)`

where the actions are inferred by the IDM from adjacent encoded frames. For each of the eight simulator factors, the experiment fixes that factor, samples the other factors independently, forms `L` paired videos, and uses the mean absolute representation difference as input to a linear classifier that predicts which factor was fixed.

The output folder contains a single compressed NumPy archive with the shared simulator design, example videos, per-model metric features, classifier weights, predictions, confusion matrices and scores.
