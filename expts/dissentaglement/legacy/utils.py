#%% Imports
import csv
import datetime
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml


#%% Run management

def count_trainable_params(model):
    leaves = jax.tree_util.tree_leaves(model)
    return int(sum(x.size for x in leaves if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact)))


def setup_run_dir(config, train, config_path="config.yaml"):
    """Create a new run when training; reuse an existing run for evaluation."""
    if not train:
        requested = config.get("run", {}).get("load_dir")
        if requested:
            run_dir = Path(requested)
        elif (Path("artefacts") / "model.eqx").exists():
            run_dir = Path(".")
        else:
            raise ValueError("Training is disabled. Set run.load_dir to a trained run folder.")
        if not (run_dir / "artefacts" / "model.eqx").exists():
            raise FileNotFoundError(run_dir / "artefacts" / "model.eqx")
        return run_dir

    base_dir = Path(config.get("run", {}).get("base_dir", "runs"))
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    mode = config["model"]["mode"]
    run_dir = base_dir / f"{stamp}-{mode}"
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    for name in ("train.py", "disentangle.py", "models.py", "loaders.py", "gendata.py", "utils.py", "requirements.txt", "README.md"):
        src = Path(name)
        if src.exists():
            shutil.copy2(src, run_dir / name)
    if Path(config_path).exists() and Path(config_path).name != "config.yaml":
        shutil.copy2(config_path, run_dir / Path(config_path).name)
    return run_dir


def get_coords_grid(H, W):
    y = jnp.linspace(-1.0, 1.0, H)
    x = jnp.linspace(-1.0, 1.0, W)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx, yy], axis=-1)


#%% Diagnostics

def plot_videos(video, ref_video, save_name, title="GT / prediction"):
    video, ref_video = np.asarray(video), np.asarray(ref_video)
    T = video.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(1.8 * T, 3.8), squeeze=False)
    for t in range(T):
        for row, frames, label in ((0, ref_video, "GT"), (1, video, "Pred")):
            frame = frames[t]
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
            axes[row, t].imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[row, t].set_axis_off()
            if t == 0:
                axes[row, t].set_ylabel(label)
            axes[row, t].set_title(f"t={t}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history, path):
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([history[k][i] for k in keys])
