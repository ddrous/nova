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
from models import WorldModel
from utils import (
    count_trainable_params,
    display_gif,
    get_coords_grid,
    plot_diagnostic_history,
    plot_loss_history,
    plot_videos,
    resolve_encoder_checkpoint,
    save_histories,
    save_rollout_gif,
    setup_run_dir,
    tree_l2_norm,
)

CONFIG_PATH = "config.yaml"
if len(sys.argv) > 1 and Path(sys.argv[1]).suffix in (".yaml", ".yml") and Path(sys.argv[1]).exists():
    CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

TRAIN = bool(CONFIG.get("phase_2", {}).get("enabled", True))

if TRAIN:
    # Phase 2 never trains or searches for an encoder implicitly. It uses the one
    # canonical reusable checkpoint for the selected representation family.
    encoder_path = resolve_encoder_checkpoint(CONFIG)
    CONFIG.setdefault("run", {})["encoder_path"] = str(encoder_path)
    run_dir = setup_run_dir(CONFIG, train=True, config_path=CONFIG_PATH)
else:
    run_dir = setup_run_dir(CONFIG, train=False, config_path=CONFIG_PATH)
    with open(run_dir / "config.yaml", "r") as f:
        trained_config = yaml.safe_load(f)
    trained_config.setdefault("phase_2", {})["enabled"] = False
    trained_config.setdefault("run", {})["load_dir"] = str(run_dir)
    CONFIG = trained_config
    encoder_path = None

np.random.seed(int(CONFIG["seed"]))
key = jax.random.PRNGKey(int(CONFIG["seed"]))


#%% Data: every video is generated on the fly
train_loader, test_loader = get_dataloaders(
    CONFIG,
    fixed_parameters=CONFIG["simulation"].get("fixed_parameters", {}),
)
vis_videos, vis_parameters = next(iter(test_loader))
B, T, H, W, C = vis_videos.shape
coords_grid = get_coords_grid(H, W)

print(f"Run directory: {run_dir}")
print(f"Mode: {CONFIG['model']['mode']}")
print(f"Video shape: {(T, H, W, C)}; batch size: {CONFIG['data']['batch_size']}")
print(f"First simulator parameters: {np.round(vis_parameters[0], 3)}")


#%% World model: load and freeze the phase-1 representation
key, model_key = jax.random.split(key)
model = WorldModel(CONFIG, frame_shape=(H, W, C), key=model_key)

if TRAIN:
    try:
        if model.decoder is None:
            encoder = eqx.tree_deserialise_leaves(encoder_path, model.encoder)
            model = eqx.tree_at(lambda m: m.encoder, model, encoder)
        else:
            # standard.eqx is still one file; it contains the phase-1 encoder and
            # decoder together because both are learned by the reconstruction objective.
            encoder, decoder = eqx.tree_deserialise_leaves(
                encoder_path, (model.encoder, model.decoder)
            )
            model = eqx.tree_at(lambda m: m.encoder, model, encoder)
            model = eqx.tree_at(lambda m: m.decoder, model, decoder)
    except Exception as exc:
        family = "weight" if model.mode.startswith("weight_") else "standard"
        raise RuntimeError(
            f"Found {encoder_path}, but it is not compatible with the current {family} "
            "configuration. Run wm_train_p1.py with the current config first."
        ) from exc
    print(f"Loaded frozen phase-1 encoder from: {encoder_path}")
else:
    model = eqx.tree_deserialise_leaves(run_dir / "artefacts" / "model.eqx", model)
    print("Loaded pretrained world model.")

print(f"Latent dimension: {model.latent_dim}")
print(f"Action dimension: {model.action_dim}")
print(f"Total model parameters: {count_trainable_params(model):,}")
print(f"  frozen encoder: {count_trainable_params(model.encoder):,}")
print(f"  frozen decoder: {count_trainable_params(model.decoder) if model.decoder is not None else 0:,}")
print(f"  trainable FDM: {count_trainable_params(model.transition_model):,}")
print(f"  trainable IDM/action: {count_trainable_params(model.action_model):,}")

# Only this tuple is differentiated/updated in phase 2.
dynamics = (model.transition_model, model.action_model)


#%% Phase-2 IDM + FDM objective
METRIC_NAMES = (
    "total", "reconstruction", "rollout_pixel", "latent_dynamics",
    "vq_codebook", "vq_commitment", "reconstruction_iou", "rollout_iou",
    "latent_std", "action_std", "action_norm",
)
# LOSS_TERMS = ("total", "rollout_pixel", "latent_dynamics", "vq_codebook", "vq_commitment")
LOSS_TERMS = ("total", "rollout_pixel", "latent_dynamics")


def binary_iou(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = jnp.sum(pred & target)
    union = jnp.sum(pred | target)
    return intersection / jnp.maximum(union, 1)


def infer_action(action_model, z_prev, z_next):
    raw, quant, idx = action_model.decode(z_prev, z_next)
    if action_model.discrete:
        action = raw + jax.lax.stop_gradient(quant - raw)
    else:
        action = raw
    return raw, quant, action, idx


def forward_video(dyn, frozen_model, video, coords):
    transition_model, action_model = dyn
    times = jnp.linspace(0.0, 1.0, video.shape[0])

    # The representation is fixed throughout phase 2.
    gt_latents = jax.vmap(frozen_model.encode_frame)(video)
    gt_latents = jax.lax.stop_gradient(gt_latents)
    recon_video = jax.vmap(lambda z, t: frozen_model.decode_frame(z, coords, t))(gt_latents, times)

    raw_actions, quant_actions, actions, action_ids = jax.vmap(
        lambda z0, z1: infer_action(action_model, z0, z1)
    )(gt_latents[:-1], gt_latents[1:])

    def transition_step(z_prev, action):
        _, z_next = transition_model(z_prev, action)
        return z_next, z_next

    _, pred_tail = jax.lax.scan(transition_step, gt_latents[0], actions)
    pred_latents = jnp.concatenate([gt_latents[:1], pred_tail], axis=0)
    rollout_video = jax.vmap(lambda z, t: frozen_model.decode_frame(z, coords, t))(pred_latents, times)

    return gt_latents, pred_latents, raw_actions, quant_actions, action_ids, recon_video, rollout_video


def loss_and_metrics(dyn, frozen_model, batch_videos, coords):
    outputs = jax.vmap(forward_video, in_axes=(None, None, 0, None))(dyn, frozen_model, batch_videos, coords)
    gt_z, pred_z, raw_a, quant_a, action_ids, recon, rollout = outputs

    # Reconstruction is a frozen-encoder diagnostic in phase 2, not an optimisation term.
    reconstruction = jnp.mean((recon - batch_videos) ** 2)
    rollout_pixel = jnp.mean((rollout[:, 1:] - batch_videos[:, 1:]) ** 2)

    target_z = gt_z[:, 1:]
    if CONFIG["train"].get("latent_target_stop_gradient", True):
        target_z = jax.lax.stop_gradient(target_z)
    latent_dynamics = jnp.mean((pred_z[:, 1:] - target_z) ** 2)

    action_model = dyn[1]
    if action_model.discrete:
        vq_codebook = jnp.mean((jax.lax.stop_gradient(raw_a) - quant_a) ** 2)
        vq_commitment = jnp.mean((raw_a - jax.lax.stop_gradient(quant_a)) ** 2)
    else:
        vq_codebook = jnp.asarray(0.0)
        vq_commitment = jnp.asarray(0.0)

    w = CONFIG["train"]["loss_weights"]
    # total = (
    #     w["rollout_pixel"] * rollout_pixel
    #     + w["latent_dynamics"] * latent_dynamics
    #     + w["vq_codebook"] * vq_codebook
    #     + w["vq_commitment"] * vq_commitment
    # )

    total = latent_dynamics

    metrics = jnp.asarray([
        total,
        reconstruction,
        rollout_pixel,
        latent_dynamics,
        vq_codebook,
        vq_commitment,
        binary_iou(recon, batch_videos),
        binary_iou(rollout[:, 1:], batch_videos[:, 1:]),
        jnp.std(gt_z),
        jnp.std(quant_a),
        jnp.mean(jnp.linalg.norm(quant_a, axis=-1)),
    ])
    return total, (metrics, recon, rollout, gt_z, quant_a, action_ids)


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
opt_state = optimizer.init(eqx.filter(dynamics, eqx.is_inexact_array))


@eqx.filter_jit
def train_step(dyn, state, frozen_model, batch_videos, coords):
    def objective(d):
        return loss_and_metrics(d, frozen_model, batch_videos, coords)

    (loss, aux), grads = eqx.filter_value_and_grad(objective, has_aux=True)(dyn)
    updates, state = optimizer.update(grads, state, eqx.filter(dyn, eqx.is_inexact_array))
    dyn = eqx.apply_updates(dyn, updates)
    return dyn, state, loss, aux[0], tree_l2_norm(grads), tree_l2_norm(eqx.filter(dyn, eqx.is_inexact_array))


@eqx.filter_jit
def eval_step(dyn, frozen_model, batch_videos, coords):
    _, aux = loss_and_metrics(dyn, frozen_model, batch_videos, coords)
    return aux


def evaluate_loader(dyn, frozen_model, loader, max_batches):
    rows, last = [], None
    for batch_idx, (videos, parameters) in enumerate(loader):
        metrics, recon, rollout, gt_z, quant_a, action_ids = eval_step(
            dyn, frozen_model, videos, coords_grid
        )
        rows.append(np.asarray(metrics))
        last = (
            np.asarray(videos), np.asarray(parameters), np.asarray(recon), np.asarray(rollout),
            np.asarray(gt_z), np.asarray(quant_a), np.asarray(action_ids),
        )
        if batch_idx + 1 >= max_batches:
            break
    return np.mean(rows, axis=0), last


def inject_dynamics(frozen_model, dyn):
    out = eqx.tree_at(lambda m: m.transition_model, frozen_model, dyn[0])
    return eqx.tree_at(lambda m: m.action_model, out, dyn[1])


#%% Phase-2 training
plot_every = max(1, epochs // 10)
save_every = max(1, epochs // 10)
print(f"plot_every={plot_every} epochs; save_every={save_every} epochs")

epoch_history = {"epoch": []}
for name in METRIC_NAMES:
    epoch_history[f"train_{name}"] = []
    epoch_history[f"eval_{name}"] = []
epoch_history.update({"grad_norm": [], "param_norm": [], "learning_rate": []})

step_history = {"step": []}
for name in METRIC_NAMES:
    step_history[name] = []
step_history.update({"grad_norm": [], "param_norm": [], "learning_rate": []})
global_step = 0

if TRAIN:
    print("\nPhase 2: training IDM + FDM only; encoder/decoder remain frozen...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loader.dataset.set_epoch(epoch)
        rows, grad_rows, param_rows = [], [], []

        for videos, _ in train_loader:
            dynamics, opt_state, _, metrics, grad_norm, param_norm = train_step(
                dynamics, opt_state, model, videos, coords_grid
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
            dynamics, model, test_loader, int(CONFIG["train"].get("eval_batches", 8))
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
                f"train {train_metrics[0]:.6f} | eval {eval_metrics[0]:.6f} | "
                f"rollout {eval_metrics[2]:.6f} | latent {eval_metrics[3]:.6f} | "
                f"vq_cb {eval_metrics[4]:.6f} | vq_commit {eval_metrics[5]:.6f} | "
                f"rollout IoU {eval_metrics[7]:.3f} | a_std {eval_metrics[9]:.3f} | "
                f"grad {np.mean(grad_rows):.3f} | lr {epoch_history['learning_rate'][-1]:.2e}",
                flush=True,
            )

        if (epoch + 1) % plot_every == 0 or epoch + 1 == epochs:
            plot_loss_history(
                step_history, epoch_history, LOSS_TERMS,
                run_dir / "plots" / "p2_losses.png",
                "Phase 2 — IDM + FDM training",
                show=True,
            )
            plot_diagnostic_history(
                epoch_history,
                (("train_rollout_iou", "train rollout IoU"), ("eval_rollout_iou", "eval rollout IoU")),
                run_dir / "plots" / "p2_rollout_iou.png",
                "Phase 2 — rollout quality",
                ylabel="IoU",
                show=True,
            )

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            checkpoint_model = inject_dynamics(model, dynamics)
            eqx.tree_serialise_leaves(run_dir / "artefacts" / "model.eqx", checkpoint_model)
            save_histories(epoch_history, step_history, run_dir / "artefacts", "p2")

    model = inject_dynamics(model, dynamics)
    eqx.tree_serialise_leaves(run_dir / "artefacts" / "model.eqx", model)
    save_histories(epoch_history, step_history, run_dir / "artefacts", "p2")
    print(f"Training wall time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print("Saved phase-2 world model and diagnostics.")

else:
    dynamics = (model.transition_model, model.action_model)
    metrics_path = run_dir / "artefacts" / "p2_metrics.npz"
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


#%% Final diagnostics and clean evaluation
final_metrics, final_pack = evaluate_loader(
    dynamics, model, test_loader, int(CONFIG["train"].get("eval_batches", 8))
)
print("\nFinal phase-2 evaluation")
for name, value in zip(METRIC_NAMES, final_metrics):
    print(f"  {name:>20s}: {float(value):.6f}")

videos, parameters, recon, rollout, gt_z, actions, action_ids = final_pack
plot_videos(
    recon[0], videos[0],
    run_dir / "plots" / "final_reconstruction.png",
    "Frozen phase-1 reconstruction",
    show=True,
)
plot_videos(
    rollout[0], videos[0],
    run_dir / "plots" / "final_rollout.png",
    "Phase 2 — final IDM-conditioned rollout",
    show=True,
)

per_frame_mse = np.mean((rollout - videos) ** 2, axis=(0, 2, 3, 4))
print("Per-frame rollout MSE:", np.round(per_frame_mse, 6))
print("Latent mean/std:", float(gt_z.mean()), float(gt_z.std()))
print("Action mean/std:", float(actions.mean()), float(actions.std()))
if model.action_model.discrete:
    ids, counts = np.unique(action_ids[action_ids >= 0], return_counts=True)
    print("Discrete action usage:", dict(zip(ids.tolist(), counts.tolist())))

np.savez_compressed(
    run_dir / "artefacts" / "final_evaluation.npz",
    metrics=np.asarray(final_metrics),
    metric_names=np.asarray(METRIC_NAMES),
    per_frame_mse=per_frame_mse,
    example_parameters=parameters[:8],
    example_videos=videos[:8],
    example_reconstructions=recon[:8],
    example_rollouts=rollout[:8],
    example_latents=gt_z[:8],
    example_actions=actions[:8],
)

# Jupyter-native inline animation of the final rollout.
gif_path = save_rollout_gif(
    rollout[0], videos[0], run_dir / "plots" / "final_rollout.gif"
)
print(f"Final rollout video: {gif_path}")
display_gif(gif_path)

if step_history["step"]:
    plot_loss_history(
        step_history, epoch_history, LOSS_TERMS,
        run_dir / "plots" / "p2_losses.png",
        "Phase 2 — IDM + FDM training",
        show=False,
    )
