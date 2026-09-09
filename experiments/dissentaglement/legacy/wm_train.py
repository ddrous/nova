#%% Imports and configuration
import math
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml

from loaders import get_dataloaders
from models import WorldModel
from utils import count_trainable_params, get_coords_grid, plot_videos, save_history_csv, setup_run_dir

CONFIG_PATH = "config.yaml"
if len(sys.argv) > 1 and Path(sys.argv[1]).suffix in (".yaml", ".yml") and Path(sys.argv[1]).exists():
    CONFIG_PATH = sys.argv[1]

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

TRAIN = bool(CONFIG["train"]["enabled"])
run_dir = setup_run_dir(CONFIG, train=TRAIN, config_path=CONFIG_PATH)

# Evaluation always reconstructs the exact architecture/configuration saved by the run.
if not TRAIN:
    with open(run_dir / "config.yaml", "r") as f:
        trained_config = yaml.safe_load(f)
    trained_config["train"]["enabled"] = False
    trained_config["run"]["load_dir"] = str(run_dir)
    CONFIG = trained_config

np.random.seed(int(CONFIG["seed"]))
key = jax.random.PRNGKey(int(CONFIG["seed"]))


#%% Data: every sample is generated on the fly
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


#%% Model
key, model_key = jax.random.split(key)
model = WorldModel(CONFIG, frame_shape=(H, W, C), key=model_key)

print(f"Latent dimension: {model.latent_dim}")
print(f"Action dimension: {model.action_dim}")
print(f"Total trainable parameters: {count_trainable_params(model):,}")
print(f"  encoder: {count_trainable_params(model.encoder):,}")
print(f"  decoder: {count_trainable_params(model.decoder) if model.decoder is not None else 0:,}")
print(f"  FDM: {count_trainable_params(model.transition_model):,}")
print(f"  IDM/action: {count_trainable_params(model.action_model):,}")

if not TRAIN:
    model = eqx.tree_deserialise_leaves(run_dir / "artefacts" / "model.eqx", model)
    print("Loaded pretrained model.")


#%% Joint phase-1 + phase-2 forward pass
METRIC_NAMES = (
    "total", "reconstruction", "rollout_pixel", "latent_dynamics",
    "vq_codebook", "vq_commitment", "reconstruction_iou", "rollout_iou",
    "latent_std", "action_std", "action_norm",
)


def binary_iou(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = jnp.sum(pred & target)
    union = jnp.sum(pred | target)
    return intersection / jnp.maximum(union, 1)


def forward_video(m, video, coords):
    times = jnp.linspace(0.0, 1.0, video.shape[0])
    gt_latents = jax.vmap(m.encode_frame)(video)
    recon_video = jax.vmap(lambda z, t: m.decode_frame(z, coords, t))(gt_latents, times)

    raw_actions, quant_actions, actions, action_ids = jax.vmap(m.infer_action)(
        gt_latents[:-1], gt_latents[1:]
    )

    def transition_step(z_prev, action):
        _, z_next = m.transition_model(z_prev, action)
        return z_next, z_next

    _, pred_tail = jax.lax.scan(transition_step, gt_latents[0], actions)
    pred_latents = jnp.concatenate([gt_latents[:1], pred_tail], axis=0)
    rollout_video = jax.vmap(lambda z, t: m.decode_frame(z, coords, t))(pred_latents, times)

    return gt_latents, pred_latents, raw_actions, quant_actions, action_ids, recon_video, rollout_video


def loss_and_metrics(m, batch_videos, coords):
    outputs = jax.vmap(forward_video, in_axes=(None, 0, None))(m, batch_videos, coords)
    gt_z, pred_z, raw_a, quant_a, action_ids, recon, rollout = outputs

    reconstruction = jnp.mean((recon - batch_videos) ** 2)
    rollout_pixel = jnp.mean((rollout[:, 1:] - batch_videos[:, 1:]) ** 2)

    target_z = gt_z[:, 1:]
    if CONFIG["train"].get("latent_target_stop_gradient", True):
        target_z = jax.lax.stop_gradient(target_z)
    latent_dynamics = jnp.mean((pred_z[:, 1:] - target_z) ** 2)

    if m.action_model.discrete:
        vq_codebook = jnp.mean((jax.lax.stop_gradient(raw_a) - quant_a) ** 2)
        vq_commitment = jnp.mean((raw_a - jax.lax.stop_gradient(quant_a)) ** 2)
    else:
        vq_codebook = jnp.asarray(0.0)
        vq_commitment = jnp.asarray(0.0)

    w = CONFIG["train"]["loss_weights"]
    total = (
        w["reconstruction"] * reconstruction
        + w["rollout_pixel"] * rollout_pixel
        + w["latent_dynamics"] * latent_dynamics
        + w["vq_codebook"] * vq_codebook
        + w["vq_commitment"] * vq_commitment
    )

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


def tree_l2_norm(tree):
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


#%% Optimiser and jitted steps
steps_per_epoch = math.ceil(CONFIG["data"]["train_samples_per_epoch"] / CONFIG["data"]["batch_size"])
total_steps = max(1, int(CONFIG["train"]["epochs"]) * steps_per_epoch)
learning_rate = optax.cosine_decay_schedule(
    init_value=float(CONFIG["train"]["learning_rate"]),
    decay_steps=total_steps,
    alpha=float(CONFIG["train"].get("min_lr_ratio", 0.05)),
)
optimizer = optax.chain(
    optax.clip_by_global_norm(float(CONFIG["train"].get("grad_clip", 1.0))),
    optax.adamw(learning_rate, weight_decay=float(CONFIG["train"].get("weight_decay", 0.0))),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))


@eqx.filter_jit
def train_step(m, state, batch_videos, coords):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_and_metrics, has_aux=True)(m, batch_videos, coords)
    updates, state = optimizer.update(grads, state, eqx.filter(m, eqx.is_inexact_array))
    m = eqx.apply_updates(m, updates)
    grad_norm = tree_l2_norm(grads)
    param_norm = tree_l2_norm(eqx.filter(m, eqx.is_inexact_array))
    return m, state, loss, aux[0], grad_norm, param_norm


@eqx.filter_jit
def eval_step(m, batch_videos, coords):
    _, aux = loss_and_metrics(m, batch_videos, coords)
    return aux


#%% Evaluation helper

def evaluate_loader(m, loader, max_batches):
    metric_rows = []
    last = None
    for batch_idx, (videos, parameters) in enumerate(loader):
        metrics, recon, rollout, gt_z, quant_a, action_ids = eval_step(m, videos, coords_grid)
        metric_rows.append(np.asarray(metrics))
        last = (
            np.asarray(videos), np.asarray(parameters), np.asarray(recon), np.asarray(rollout),
            np.asarray(gt_z), np.asarray(quant_a), np.asarray(action_ids),
        )
        if batch_idx + 1 >= max_batches:
            break
    return np.mean(metric_rows, axis=0), last


#%% Joint end-to-end training
history = {
    "epoch": [], "train_total": [], "eval_total": [],
    "train_reconstruction": [], "eval_reconstruction": [],
    "train_rollout_pixel": [], "eval_rollout_pixel": [],
    "train_latent_dynamics": [], "eval_latent_dynamics": [],
    "train_reconstruction_iou": [], "eval_reconstruction_iou": [],
    "train_rollout_iou": [], "eval_rollout_iou": [],
    "latent_std": [], "action_std": [], "action_norm": [],
    "grad_norm": [], "param_norm": [], "learning_rate": [],
}
step_total, step_grad, step_lr = [], [], []
global_step = 0

if TRAIN:
    print("\nTraining encoder, IDM, FDM and decoder jointly end-to-end...")
    start_time = time.time()
    for epoch in range(int(CONFIG["train"]["epochs"])):
        train_loader.dataset.set_epoch(epoch)
        rows, grad_rows, param_rows = [], [], []

        for videos, _ in train_loader:
            model, opt_state, _, metrics, grad_norm, param_norm = train_step(
                model, opt_state, videos, coords_grid
            )
            rows.append(np.asarray(metrics))
            grad_rows.append(float(grad_norm))
            param_rows.append(float(param_norm))
            step_total.append(float(metrics[0]))
            step_grad.append(float(grad_norm))
            step_lr.append(float(learning_rate(global_step)))
            global_step += 1

        train_metrics = np.mean(rows, axis=0)
        eval_metrics, eval_pack = evaluate_loader(
            model, test_loader, int(CONFIG["train"].get("eval_batches", 8))
        )

        history["epoch"].append(epoch + 1)
        history["train_total"].append(train_metrics[0])
        history["eval_total"].append(eval_metrics[0])
        history["train_reconstruction"].append(train_metrics[1])
        history["eval_reconstruction"].append(eval_metrics[1])
        history["train_rollout_pixel"].append(train_metrics[2])
        history["eval_rollout_pixel"].append(eval_metrics[2])
        history["train_latent_dynamics"].append(train_metrics[3])
        history["eval_latent_dynamics"].append(eval_metrics[3])
        history["train_reconstruction_iou"].append(train_metrics[6])
        history["eval_reconstruction_iou"].append(eval_metrics[6])
        history["train_rollout_iou"].append(train_metrics[7])
        history["eval_rollout_iou"].append(eval_metrics[7])
        history["latent_std"].append(eval_metrics[8])
        history["action_std"].append(eval_metrics[9])
        history["action_norm"].append(eval_metrics[10])
        history["grad_norm"].append(np.mean(grad_rows))
        history["param_norm"].append(np.mean(param_rows))
        history["learning_rate"].append(float(learning_rate(max(global_step - 1, 0))))

        if (epoch + 1) % int(CONFIG["train"].get("print_every", 1)) == 0:
            print(
                f"epoch {epoch+1:03d} | "
                f"train {train_metrics[0]:.5f} | eval {eval_metrics[0]:.5f} | "
                f"recon {eval_metrics[1]:.5f} | rollout {eval_metrics[2]:.5f} | "
                f"latent {eval_metrics[3]:.5f} | IoU {eval_metrics[7]:.3f} | "
                f"z_std {eval_metrics[8]:.3f} | a_std {eval_metrics[9]:.3f} | "
                f"grad {np.mean(grad_rows):.3f} | lr {history['learning_rate'][-1]:.2e}",
                flush=True,
            )

        if (epoch + 1) % int(CONFIG["train"].get("visualise_every", 2)) == 0:
            videos, _, recon, rollout, *_ = eval_pack
            plot_videos(recon[0], videos[0], run_dir / "plots" / f"reconstruction_epoch{epoch+1:03d}.png", "reconstruction")
            plot_videos(rollout[0], videos[0], run_dir / "plots" / f"rollout_epoch{epoch+1:03d}.png", "IDM-conditioned rollout")

    eqx.tree_serialise_leaves(run_dir / "artefacts" / "model.eqx", model)
    np.savez_compressed(
        run_dir / "artefacts" / "training_metrics.npz",
        **{k: np.asarray(v) for k, v in history.items()},
        step_total=np.asarray(step_total),
        step_grad_norm=np.asarray(step_grad),
        step_learning_rate=np.asarray(step_lr),
    )
    save_history_csv(history, run_dir / "artefacts" / "training_metrics.csv")
    print(f"Training wall time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    print("Saved model and training diagnostics.")

else:
    metrics_path = run_dir / "artefacts" / "training_metrics.npz"
    if metrics_path.exists():
        stored = np.load(metrics_path)
        for key_name in history:
            if key_name in stored:
                history[key_name] = stored[key_name].tolist()
        step_total = stored["step_total"].tolist() if "step_total" in stored else []
        step_grad = stored["step_grad_norm"].tolist() if "step_grad_norm" in stored else []
        step_lr = stored["step_learning_rate"].tolist() if "step_learning_rate" in stored else []


#%% Final diagnostics and clean evaluation
final_metrics, final_pack = evaluate_loader(
    model, test_loader, int(CONFIG["train"].get("eval_batches", 8))
)
final = dict(zip(METRIC_NAMES, final_metrics.tolist()))
print("\nFinal evaluation")
for name in METRIC_NAMES:
    print(f"  {name:>20s}: {final[name]:.6f}")

videos, parameters, recon, rollout, gt_z, actions, action_ids = final_pack
plot_videos(recon[0], videos[0], run_dir / "plots" / "final_reconstruction.png", "final reconstruction")
plot_videos(rollout[0], videos[0], run_dir / "plots" / "final_rollout.png", "final IDM-conditioned rollout")

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


#%% Diagnostic plots
if history["epoch"]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["train_total"], label="train total")
    ax.plot(history["epoch"], history["eval_total"], label="eval total")
    ax.plot(history["epoch"], history["eval_reconstruction"], label="eval reconstruction")
    ax.plot(history["epoch"], history["eval_rollout_pixel"], label="eval rollout")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "losses.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["eval_reconstruction_iou"], label="reconstruction IoU")
    ax.plot(history["epoch"], history["eval_rollout_iou"], label="rollout IoU")
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "iou.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["latent_std"], label="latent std")
    ax.plot(history["epoch"], history["action_std"], label="action std")
    ax.plot(history["epoch"], history["action_norm"], label="action norm")
    ax.plot(history["epoch"], history["grad_norm"], label="gradient norm")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "representation_diagnostics.png", dpi=160)
    plt.close(fig)
