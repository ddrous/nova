#%% Imports and configuration
import math
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from loaders import get_dataloaders
from models import WorldModel, fourier_encode
from utils import (
    count_trainable_params,
    get_coords_grid,
    plot_diagnostic_history,
    plot_loss_history,
    plot_videos,
    encoder_checkpoint_path,
    save_histories,
    setup_run_dir,
    tree_l2_norm,
)

CONFIG_PATH = "config.yaml"
if len(sys.argv) > 1 and Path(sys.argv[1]).suffix in (".yaml", ".yml") and Path(sys.argv[1]).exists():
    CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

TRAIN = bool(CONFIG.get("phase_1", {}).get("enabled", True))

# Phase 1 is a normal run like phase 2. The reusable encoder is exported separately
# to encs/weight.eqx or encs/standard.eqx after training finishes.
run_dir = setup_run_dir(CONFIG, train=TRAIN, config_path=CONFIG_PATH, phase="p1")

# Evaluation reconstructs the exact representation configuration saved with the phase-1 run.
if not TRAIN:
    with open(run_dir / "config.yaml", "r") as f:
        trained_config = yaml.safe_load(f)
    trained_config.setdefault("phase_1", {})["enabled"] = False
    trained_config["phase_1"]["load_dir"] = str(run_dir)
    CONFIG = trained_config

encoder_path = encoder_checkpoint_path(CONFIG)

np.random.seed(int(CONFIG["seed"]))
key = jax.random.PRNGKey(int(CONFIG["seed"]))


#%% Data: phase 1 sees frames, generated on the fly
train_loader, test_loader = get_dataloaders(
    CONFIG,
    fixed_parameters=CONFIG["simulation"].get("fixed_parameters", {}),
)
vis_videos, vis_parameters = next(iter(test_loader))
B, T, H, W, C = vis_videos.shape
coords_grid = get_coords_grid(H, W)

print(f"Run directory: {run_dir}")
print(f"Reusable encoder checkpoint: {encoder_path}")
print(f"Representation family: {'weight' if CONFIG['model']['mode'].startswith('weight_') else 'standard'}")
print(f"Video shape: {(T, H, W, C)}; batch size: {CONFIG['data']['batch_size']}")
print(f"First simulator parameters: {np.round(vis_parameters[0], 3)}")


#%% Representation model only
key, model_key = jax.random.split(key)
template_model = WorldModel(CONFIG, frame_shape=(H, W, C), key=model_key)
representation = (template_model.encoder, template_model.decoder)

print(f"Latent dimension: {template_model.latent_dim}")
print(f"Phase-1 trainable parameters: {count_trainable_params(representation):,}")
print(f"  encoder: {count_trainable_params(representation[0]):,}")
print(f"  decoder: {count_trainable_params(representation[1]) if representation[1] is not None else 0:,}")


def representation_tree(rep):
    # Weight-space needs only WeightCNN: theta_base lives inside it.
    # Standard phase 1 also learns the decoder, so keep encoder+decoder together
    # inside the single standard.eqx checkpoint.
    return rep[0] if rep[1] is None else rep


def save_representation(rep, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, representation_tree(rep))


def load_representation(rep, path):
    path = Path(path)
    if rep[1] is None:
        return eqx.tree_deserialise_leaves(path, rep[0]), None
    return eqx.tree_deserialise_leaves(path, rep)


if not TRAIN:
    representation = load_representation(representation, run_dir / "artefacts" / "encoder.eqx")
    print("Loaded pretrained phase-1 representation from the run folder.")


#%% Phase-1 reconstruction objective
METRIC_NAMES = ("total", "reconstruction", "reconstruction_iou", "latent_std")
LOSS_TERMS = ("total", "reconstruction")


def binary_iou(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = jnp.sum(pred & target)
    union = jnp.sum(pred | target)
    return intersection / jnp.maximum(union, 1)


def encode_frame(rep, frame):
    return rep[0](jnp.transpose(frame, (2, 0, 1)))


def decode_frame(rep, z, coords, time_value=0.0):
    encoder, decoder = rep
    if template_model.mode.startswith("standard_"):
        return jnp.transpose(decoder(z), (1, 2, 0))

    theta = z + encoder.theta_base
    root = template_model.unravel_fn(theta)
    flat_xy = coords.reshape(-1, 2)

    def render_point(xy):
        encoded = fourier_encode(xy, template_model.num_freqs)
        if template_model.use_time_in_root:
            encoded = jnp.concatenate([jnp.asarray([time_value], dtype=z.dtype), encoded])
        return root(encoded)

    return jax.vmap(render_point)(flat_xy).reshape(H, W, -1)


def forward_video(rep, video, coords):
    times = jnp.linspace(0.0, 1.0, video.shape[0])
    latents = jax.vmap(lambda frame: encode_frame(rep, frame))(video)
    recon = jax.vmap(lambda z, t: decode_frame(rep, z, coords, t))(latents, times)
    return latents, recon


def loss_and_metrics(rep, batch_videos, coords):
    latents, recon = jax.vmap(forward_video, in_axes=(None, 0, None))(rep, batch_videos, coords)
    reconstruction = jnp.mean((recon - batch_videos) ** 2)
    total = float(CONFIG["train"]["loss_weights"]["reconstruction"]) * reconstruction
    metrics = jnp.asarray([
        total,
        reconstruction,
        binary_iou(recon, batch_videos),
        jnp.std(latents),
    ])
    return total, (metrics, recon, latents)


#%% Optimiser and jitted steps
steps_per_epoch = math.ceil(CONFIG["data"]["train_samples_per_epoch"] / CONFIG["data"]["batch_size"])
epochs = int(CONFIG["train"]["epochs"])
total_steps = max(1, epochs * steps_per_epoch)
learning_rate = optax.cosine_decay_schedule(
    init_value=float(CONFIG["train"]["learning_rate"]),
    decay_steps=total_steps,
    alpha=float(CONFIG["train"].get("min_lr_ratio", 0.05)),
)
optimizer = optax.chain(
    optax.clip_by_global_norm(float(CONFIG["train"].get("grad_clip", 1.0))),
    optax.adamw(learning_rate, weight_decay=float(CONFIG["train"].get("weight_decay", 0.0))),
)
opt_state = optimizer.init(eqx.filter(representation, eqx.is_inexact_array))


@eqx.filter_jit
def train_step(rep, state, batch_videos, coords):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_and_metrics, has_aux=True)(rep, batch_videos, coords)
    updates, state = optimizer.update(grads, state, eqx.filter(rep, eqx.is_inexact_array))
    rep = eqx.apply_updates(rep, updates)
    return rep, state, loss, aux[0], tree_l2_norm(grads), tree_l2_norm(eqx.filter(rep, eqx.is_inexact_array))


@eqx.filter_jit
def eval_step(rep, batch_videos, coords):
    _, aux = loss_and_metrics(rep, batch_videos, coords)
    return aux


def evaluate_loader(rep, loader, max_batches):
    rows, last = [], None
    for batch_idx, (videos, parameters) in enumerate(loader):
        metrics, recon, latents = eval_step(rep, videos, coords_grid)
        rows.append(np.asarray(metrics))
        last = np.asarray(videos), np.asarray(parameters), np.asarray(recon), np.asarray(latents)
        if batch_idx + 1 >= max_batches:
            break
    return np.mean(rows, axis=0), last


#%% Phase-1 training
# With the current 100 epochs this gives exactly ten visible plots and ten checkpoint saves.
plot_every = max(1, epochs // 10)
save_every = max(1, epochs // 10)
print(f"plot_every={plot_every} epochs; save_every={save_every} epochs")

epoch_history = {
    "epoch": [],
    "train_total": [], "eval_total": [],
    "train_reconstruction": [], "eval_reconstruction": [],
    "train_reconstruction_iou": [], "eval_reconstruction_iou": [],
    "train_latent_std": [], "eval_latent_std": [],
    "grad_norm": [], "param_norm": [], "learning_rate": [],
}
step_history = {
    "step": [], "total": [], "reconstruction": [],
    "reconstruction_iou": [], "latent_std": [],
    "grad_norm": [], "param_norm": [], "learning_rate": [],
}
global_step = 0

if TRAIN:
    print("\nPhase 1: training the encoder/representation only...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        rows, grad_rows, param_rows = [], [], []

        for videos, _ in train_loader:
            representation, opt_state, _, metrics, grad_norm, param_norm = train_step(
                representation, opt_state, videos, coords_grid
            )
            row = np.asarray(metrics)
            rows.append(row)
            grad_rows.append(float(grad_norm))
            param_rows.append(float(param_norm))

            step_history["step"].append(global_step + 1)
            for idx, name in enumerate(METRIC_NAMES):
                step_history[name].append(float(row[idx]))
            step_history["grad_norm"].append(float(grad_norm))
            step_history["param_norm"].append(float(param_norm))
            step_history["learning_rate"].append(float(learning_rate(global_step)))
            global_step += 1

        train_metrics = np.mean(rows, axis=0)
        eval_metrics, eval_pack = evaluate_loader(
            representation, test_loader, int(CONFIG["train"].get("eval_batches", 8))
        )

        epoch_history["epoch"].append(epoch + 1)
        for idx, name in enumerate(METRIC_NAMES):
            epoch_history[f"train_{name}"].append(float(train_metrics[idx]))
            epoch_history[f"eval_{name}"].append(float(eval_metrics[idx]))
        epoch_history["grad_norm"].append(float(np.mean(grad_rows)))
        epoch_history["param_norm"].append(float(np.mean(param_rows)))
        epoch_history["learning_rate"].append(float(learning_rate(max(global_step - 1, 0))))

        if (epoch + 1) % int(CONFIG["train"].get("print_every", 1)) == 0:
            print(
                f"epoch {epoch+1:03d}/{epochs} | "
                f"train total {train_metrics[0]:.6f} | eval total {eval_metrics[0]:.6f} | "
                f"train recon {train_metrics[1]:.6f} | eval recon {eval_metrics[1]:.6f} | "
                f"eval IoU {eval_metrics[2]:.3f} | z_std {eval_metrics[3]:.3f} | "
                f"grad {np.mean(grad_rows):.3f} | lr {epoch_history['learning_rate'][-1]:.2e}",
                flush=True,
            )

        if (epoch + 1) % plot_every == 0 or epoch + 1 == epochs:
            plot_loss_history(
                step_history, epoch_history, LOSS_TERMS,
                run_dir / "plots" / "p1_losses.png",
                "Phase 1 — encoder reconstruction training",
                show=True,
            )
            plot_diagnostic_history(
                epoch_history,
                (("train_reconstruction_iou", "train IoU"), ("eval_reconstruction_iou", "eval IoU")),
                run_dir / "plots" / "p1_reconstruction_iou.png",
                "Phase 1 — reconstruction quality",
                ylabel="IoU",
                show=True,
            )

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_representation(representation, run_dir / "artefacts" / "encoder.eqx")
            save_histories(epoch_history, step_history, run_dir / "artefacts", "p1")

    # Only the completed phase-1 representation is exported to the shared encs/ cache.
    save_representation(representation, encoder_path)
    print(f"Training wall time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print(f"Saved phase-1 run to: {run_dir}")
    print(f"Exported reusable encoder to: {encoder_path}")

else:
    metrics_path = run_dir / "artefacts" / "p1_metrics.npz"
    if metrics_path.exists():
        stored = np.load(metrics_path)
        for key_name in epoch_history:
            key = f"epoch_{key_name}"
            if key in stored:
                epoch_history[key_name] = stored[key].tolist()
        for key_name in step_history:
            key = f"step_{key_name}"
            if key in stored:
                step_history[key_name] = stored[key].tolist()


#%% Final phase-1 evaluation
final_metrics, final_pack = evaluate_loader(
    representation, test_loader, int(CONFIG["train"].get("eval_batches", 8))
)
print("\nFinal phase-1 evaluation")
for name, value in zip(METRIC_NAMES, final_metrics):
    print(f"  {name:>20s}: {float(value):.6f}")

videos, parameters, recon, latents = final_pack
plot_videos(
    recon[0], videos[0],
    run_dir / "plots" / "final_reconstruction.png",
    "Phase 1 — final reconstruction",
    show=True,
)

np.savez_compressed(
    run_dir / "artefacts" / "p1_final_evaluation.npz",
    metrics=np.asarray(final_metrics),
    metric_names=np.asarray(METRIC_NAMES),
    example_parameters=parameters[:8],
    example_videos=videos[:8],
    example_reconstructions=recon[:8],
    example_latents=latents[:8],
)

if step_history["step"]:
    plot_loss_history(
        step_history, epoch_history, LOSS_TERMS,
        run_dir / "plots" / "p1_losses.png",
        "Phase 1 — encoder reconstruction training",
        show=False,
    )
