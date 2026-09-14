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
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
sns.set_theme(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'


#%% Plot style

def configure_plots():
    """Large, clean defaults suitable for notebook and paper figures."""
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 19,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.1,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


configure_plots()


def display_figure(fig):
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        plt.show()


#%% Run and encoder management

def count_trainable_params(model):
    leaves = jax.tree_util.tree_leaves(model)
    return int(sum(x.size for x in leaves if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.inexact)))


def representation_family(config):
    mode = config["model"]["mode"]
    return "weight" if mode.startswith("weight_") else "standard"


def encoder_checkpoint_path(config):
    """Canonical reusable phase-1 checkpoint: encs/weight.eqx or encs/standard.eqx."""
    requested = config.get("run", {}).get("encoder_path")
    if requested:
        return Path(requested)
    base = Path(config.get("run", {}).get("encoder_base_dir", "encs"))
    return base / f"{representation_family(config)}.eqx"


def resolve_encoder_checkpoint(config):
    """Find the reusable encoder required by phase 2, or tell the user to run phase 1."""
    path = encoder_checkpoint_path(config)
    if not path.exists():
        family = representation_family(config)
        raise FileNotFoundError(
            f"No pretrained {family} encoder found at {path}. "
            f"Run wm_train_p1.py with a {family} model first, then rerun phase 2."
        )
    return path


def _stamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def _copy_source_files(destination):
    names = (
        "wm_train_p1.py", "wm_train_p2.py", "wm_eval.py",
        "models.py", "loaders.py", "gendata.py", "utils.py",
        "requirements.txt", "README.md",
    )
    for name in names:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, destination / name)


def setup_run_dir(config, train, config_path="config.yaml", phase="p2"):
    """Create the normal isolated run folder for either training phase."""
    if not train:
        if phase == "p1":
            requested = config.get("phase_1", {}).get("load_dir")
            expected = Path("artefacts") / "encoder.eqx"
            message = "Phase 1 training is disabled. Set phase_1.load_dir to a phase-1 run folder."
        else:
            requested = config.get("run", {}).get("load_dir")
            expected = Path("artefacts") / "model.eqx"
            message = "Phase 2 training is disabled. Set run.load_dir to a trained run folder."

        if requested:
            run_dir = Path(requested)
        elif expected.exists():
            run_dir = Path(".")
        else:
            raise ValueError(message)

        if not (run_dir / expected).exists():
            raise FileNotFoundError(run_dir / expected)
        return run_dir

    base = Path(config.get("run", {}).get("base_dir", "runs"))
    mode = config["model"]["mode"]
    suffix = f"-{phase}" if phase == "p1" else ""
    run_dir = base / f"{_stamp()}-{mode}{suffix}"
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    _copy_source_files(run_dir)
    if Path(config_path).exists() and Path(config_path).name != "config.yaml":
        shutil.copy2(config_path, run_dir / Path(config_path).name)
    return run_dir


def get_coords_grid(H, W):
    y = jnp.linspace(-1.0, 1.0, H)
    x = jnp.linspace(-1.0, 1.0, W)
    xx, yy = jnp.meshgrid(x, y)
    return jnp.stack([xx, yy], axis=-1)


#%% Training diagnostics

def tree_l2_norm(tree):
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def save_history_csv(history, path):
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([history[k][i] for k in keys])


def save_histories(epoch_history, step_history, artefact_dir, stem):
    artefact_dir = Path(artefact_dir)
    arrays = {f"epoch_{k}": np.asarray(v) for k, v in epoch_history.items()}
    arrays.update({f"step_{k}": np.asarray(v) for k, v in step_history.items()})
    np.savez_compressed(artefact_dir / f"{stem}_metrics.npz", **arrays)
    save_history_csv(epoch_history, artefact_dir / f"{stem}_epoch_metrics.csv")
    save_history_csv(step_history, artefact_dir / f"{stem}_step_metrics.csv")


def _loss_axis(ax):
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.grid(alpha=0.18, linewidth=0.8)


def plot_loss_history(step_history, epoch_history, loss_terms, save_name, title, show=True):
    """Show every loss component at train-step and epoch resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

    x_step = np.asarray(step_history.get("step", []))
    for term in loss_terms:
        values = np.asarray(step_history.get(term, []))
        if len(values):
            axes[0].plot(x_step, values, label=term.replace("_", " "))
    axes[0].set_xlabel("train step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("per-step training losses")
    _loss_axis(axes[0])
    axes[0].legend()

    x_epoch = np.asarray(epoch_history.get("epoch", []))
    for term in loss_terms:
        train_values = np.asarray(epoch_history.get(f"train_{term}", []))
        eval_values = np.asarray(epoch_history.get(f"eval_{term}", []))
        if len(train_values):
            axes[1].plot(x_epoch, train_values, label=f"train {term.replace('_', ' ')}")
        if len(eval_values):
            axes[1].plot(x_epoch, eval_values, linestyle="--", label=f"eval {term.replace('_', ' ')}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_title("epoch means")
    _loss_axis(axes[1])
    axes[1].legend(ncol=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    if show:
        display_figure(fig)
    plt.close(fig)


def plot_diagnostic_history(epoch_history, series, save_name, title, ylabel="value", show=True):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.asarray(epoch_history.get("epoch", []))
    for key, label in series:
        values = np.asarray(epoch_history.get(key, []))
        if len(values):
            ax.plot(x, values, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    if show:
        display_figure(fig)
    plt.close(fig)


#%% Video diagnostics

def plot_videos(video, ref_video, save_name=None, title="GT / prediction", show=False):
    video, ref_video = np.asarray(video), np.asarray(ref_video)
    T = video.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2.4 * T, 5.0), squeeze=False)
    for t in range(T):
        for row, frames, label in ((0, ref_video, "GT"), (1, video, "Pred")):
            frame = frames[t]
            if frame.shape[-1] == 1:
                frame = frame[..., 0]
            axes[row, t].imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[row, t].set_xticks([])
            axes[row, t].set_yticks([])
            axes[row, t].set_title(f"t={t}")
            if t == 0:
                axes[row, t].set_ylabel(label, fontsize=17, fontweight="bold")
    fig.suptitle(title)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(save_name, bbox_inches="tight")
    if show:
        display_figure(fig)
    plt.close(fig)


def _to_rgb(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    frame = np.clip(frame, 0.0, 1.0)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    return (255.0 * frame[..., :3]).astype(np.uint8)


def save_rollout_gif(pred_video, ref_video, path, scale=4, duration_ms=450):
    """Save a simple GT/pred rollout GIF for inline Jupyter display."""
    pred_video, ref_video = np.asarray(pred_video), np.asarray(ref_video)
    frames = []
    gap = 8 * scale
    header = 24 * scale

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13 * scale)
    except OSError:
        font = ImageFont.load_default()

    for t in range(pred_video.shape[0]):
        gt = Image.fromarray(_to_rgb(ref_video[t])).resize(
            (ref_video.shape[2] * scale, ref_video.shape[1] * scale), Image.Resampling.NEAREST
        )
        pred = Image.fromarray(_to_rgb(pred_video[t])).resize(
            (pred_video.shape[2] * scale, pred_video.shape[1] * scale), Image.Resampling.NEAREST
        )
        canvas = Image.new("RGB", (gt.width + gap + pred.width, header + max(gt.height, pred.height)), "white")
        canvas.paste(gt, (0, header))
        canvas.paste(pred, (gt.width + gap, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((gt.width // 2, 2 * scale), "GT", fill="black", font=font, anchor="ma")
        draw.text((gt.width + gap + pred.width // 2, 2 * scale), "Pred", fill="black", font=font, anchor="ma")
        draw.text((canvas.width // 2, header - 2 * scale), f"t={t}", fill="black", font=font, anchor="ms")
        frames.append(canvas)

    path = Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return path


def display_gif(path):
    try:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=str(path)))
    except ImportError:
        print(f"Saved rollout GIF to {path}")
