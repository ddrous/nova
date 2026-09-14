#%% Imports and experiment paths
import csv
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from gendata import (
    PARAMETER_NAMES,
    make_sprite,
    make_trajectory,
    parameter_supports,
    sample_parameters,
    simulate_video,
)
from models import WorldModel
from utils import configure_plots, get_coords_grid

configure_plots()
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 23,
    "axes.labelsize": 21,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 25,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "savefig.dpi": 260,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Edit this list in Jupyter, or pass run folders on the command line.
RUN_DIRS = []
# cli_runs = [Path(x) for x in sys.argv[1:] if Path(x).is_dir()]
wm_folders = [
    "260909-122231-weight_ab",
    "260909-130907-weight_joint",
    "260909-151104-standard_ab",
    "260909-155921-standard_joint",
]
cli_runs = [Path("runs") / f for f in wm_folders]

if cli_runs:
    RUN_DIRS = cli_runs
else:
    RUN_DIRS = [Path(x) for x in RUN_DIRS]

if not RUN_DIRS:
    raise ValueError("Provide one or more trained run folders in RUN_DIRS or on the command line.")

OUTPUT_DIR = Path("runs") / f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}_multieval"
ARTEFACT_DIR = OUTPUT_DIR / "artefacts"
PLOT_DIR = OUTPUT_DIR / "plots"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=False)
PLOT_DIR.mkdir(exist_ok=True)

if Path("config.yaml").exists():
    shutil.copy2("config.yaml", OUTPUT_DIR / "config.yaml")
if Path("wm_multieval.py").exists():
    shutil.copy2("wm_multieval.py", OUTPUT_DIR / "wm_multieval.py")


#%% Load model/run configurations and multi-evaluation settings
run_configs = []
for run_dir in RUN_DIRS:
    with open(run_dir / "config.yaml", "r") as f:
        run_configs.append(yaml.safe_load(f))

SIM_CONFIG = run_configs[0]["simulation"]
if int(SIM_CONFIG["num_frames"]) != 3:
    raise ValueError("The trained models are expected to use simulation.num_frames=3.")
for cfg in run_configs[1:]:
    if cfg["simulation"] != SIM_CONFIG:
        raise ValueError("All compared runs must use exactly the same training simulation configuration.")

if Path("config.yaml").exists():
    with open("config.yaml", "r") as f:
        CURRENT_CONFIG = yaml.safe_load(f)
else:
    CURRENT_CONFIG = run_configs[0]

# Preserve the existing disentanglement hyperparameters unless explicitly overridden.
probe_cfg = dict(run_configs[0].get("disentanglement", {}))
probe_cfg.update(CURRENT_CONFIG.get("disentanglement", {}))
multieval_cfg = dict(CURRENT_CONFIG.get("multieval", {}))

HORIZON = int(multieval_cfg.get("horizon", 9))
if HORIZON <= 3:
    raise ValueError("multieval.horizon must be > 3 for the long-horizon evaluation.")

VIDEO_REPRESENTATION = multieval_cfg.get(
    "video_repreentation",
    probe_cfg.get("video_repreentation", "state_actions"),
)
if VIDEO_REPRESENTATION not in ("state_actions", "states_only"):
    raise ValueError("video_repreentation must be 'state_actions' or 'states_only'.")

L = int(multieval_cfg.get("pairs_per_feature", probe_cfg.get("pairs_per_feature", 16)))
TRAIN_PER_FACTOR = int(multieval_cfg.get("train_features_per_factor", probe_cfg.get("train_features_per_factor", 256)))
TEST_PER_FACTOR = int(multieval_cfg.get("test_features_per_factor", probe_cfg.get("test_features_per_factor", 128)))
REP_BATCH = int(multieval_cfg.get("representation_batch_size", probe_cfg.get("representation_batch_size", 128)))
FEATURE_CHUNK = int(multieval_cfg.get("feature_chunk_size", probe_cfg.get("feature_chunk_size", 16)))
CLASSIFIER_MAX_ITER = int(multieval_cfg.get("classifier_max_iter", probe_cfg.get("classifier_max_iter", 3000)))
CLASSIFIER_C = float(multieval_cfg.get("classifier_C", probe_cfg.get("classifier_C", 1.0)))
CLASSIFIER_TOL = float(multieval_cfg.get("classifier_tol", probe_cfg.get("classifier_tol", 1e-9)))
CLASSIFIER_GRAD_TOL = float(multieval_cfg.get("classifier_grad_tol", probe_cfg.get("classifier_grad_tol", 1e-6)))

EVAL_VIDEOS = int(multieval_cfg.get("eval_videos", 256))
SWAP_PAIRS = int(multieval_cfg.get("swap_pairs", 128))
EVAL_BATCH = int(multieval_cfg.get("eval_batch_size", 64))
HIST_BINS = int(multieval_cfg.get("hist_bins", 64))
INTERVENTION_STEP = int(multieval_cfg.get("intervention_step", max(1, HORIZON // 2)))
if not 0 <= INTERVENTION_STEP < HORIZON - 1:
    raise ValueError("multieval.intervention_step must be in [0, horizon-2].")
INTERVENTION_EFFECT_FRAME = INTERVENTION_STEP + 1

SEED = int(multieval_cfg.get("seed", probe_cfg.get("seed", 991) + 1000))
rng = np.random.default_rng(SEED)

# x0/y0 remain fixed nuisance variables in the Beta-VAE-style probe.
FACTOR_NAMES = [name for name in PARAMETER_NAMES[:8] if name not in ("x0", "y0")]
N_FACTORS = len(FACTOR_NAMES)
SUPPORTS = parameter_supports(SIM_CONFIG)
VELOCITY_VALUES = np.asarray(SIM_CONFIG.get("velocity_values", [-4, 0, 4]), dtype=int)
PARAM_DIM = 4 + 2 * (HORIZON - 1)
H = W = int(SIM_CONFIG["image_size"])
C = 1
COORDS_GRID = get_coords_grid(H, W)

print("Comparing runs:")
for run_dir, cfg in zip(RUN_DIRS, run_configs):
    print(f"  {run_dir} -> {cfg['model']['mode']}")
print(f"Long horizon: T={HORIZON}")
print(f"Video representation: {VIDEO_REPRESENTATION}")
print(f"Probe factors: {FACTOR_NAMES}")
print(f"Reconstruction/rollout videos: {EVAL_VIDEOS}")
print(f"Swap pairs: {SWAP_PAIRS}; intervention transition={INTERVENTION_STEP} -> effect from frame {INTERVENTION_EFFECT_FRAME}")

EFFECTIVE_CONFIG = {
    "runs": [str(p) for p in RUN_DIRS],
    "simulation_training": SIM_CONFIG,
    "multieval": {
        "horizon": HORIZON,
        "video_repreentation": VIDEO_REPRESENTATION,
        "pairs_per_feature": L,
        "train_features_per_factor": TRAIN_PER_FACTOR,
        "test_features_per_factor": TEST_PER_FACTOR,
        "representation_batch_size": REP_BATCH,
        "feature_chunk_size": FEATURE_CHUNK,
        "classifier_max_iter": CLASSIFIER_MAX_ITER,
        "classifier_C": CLASSIFIER_C,
        "eval_videos": EVAL_VIDEOS,
        "swap_pairs": SWAP_PAIRS,
        "eval_batch_size": EVAL_BATCH,
        "hist_bins": HIST_BINS,
        "intervention_step": INTERVENTION_STEP,
        "seed": SEED,
    },
}
with open(ARTEFACT_DIR / "multieval_config.yaml", "w") as f:
    yaml.safe_dump(EFFECTIVE_CONFIG, f, sort_keys=False)


#%% Small shared utilities
def safe_name(path, mode):
    text = f"{path.name}_{mode}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def pretty_mode(mode):
    names = {
        "weight_ab": "Weight A/B",
        "weight_joint": "Weight joint",
        "standard_ab": "Standard A/B",
        "standard_joint": "Standard joint",
        "standard": "Standard joint",
    }
    return names.get(mode, mode.replace("_", " ").title())


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.9)
    ax.set_axisbelow(True)


def show_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def add_bar_labels(ax, bars, fmt="{:.3f}"):
    labels = [fmt.format(float(bar.get_height())) for bar in bars]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=13)


def load_model(run_dir, cfg):
    key = jax.random.PRNGKey(int(cfg["seed"]))
    model = WorldModel(cfg, frame_shape=(H, W, C), key=key)
    return eqx.tree_deserialise_leaves(run_dir / "artefacts" / "model.eqx", model)


#%% Long-horizon simulator: same one-step support, extended safely on the fly
def sprite_centre_bounds(shape, rotation):
    sprite = make_sprite(shape, rotation, float(SIM_CONFIG["shape_size"]))
    half = sprite.shape[0] // 2
    yy, xx = np.where(sprite == 1)
    dx = xx - half
    dy = yy - half
    return (
        1 - int(dx.min()),
        W - 2 - int(dx.max()),
        1 - int(dy.min()),
        H - 2 - int(dy.max()),
    )


def intersect_bounds(*contents):
    bounds = [sprite_centre_bounds(shape, rotation) for shape, rotation in contents]
    x_min = max(b[0] for b in bounds)
    x_max = min(b[1] for b in bounds)
    y_min = max(b[2] for b in bounds)
    y_max = min(b[3] for b in bounds)
    if x_min > x_max or y_min > y_max:
        raise ValueError("No common in-frame support for the requested contents.")
    return x_min, x_max, y_min, y_max


def sample_motion_sequence(motion_rng, x0, y0, bounds, horizon=HORIZON):
    """Sample a bounded random walk using only the configured velocity support."""
    x_min, x_max, y_min, y_max = bounds
    velocity_pairs = np.asarray(
        [(vx, vy) for vx in VELOCITY_VALUES for vy in VELOCITY_VALUES],
        dtype=int,
    )
    x, y = int(x0), int(y0)
    motion = np.empty((horizon - 1, 2), dtype=np.float32)

    for t in range(horizon - 1):
        candidate_positions = velocity_pairs + np.asarray([x, y])
        valid = (
            (candidate_positions[:, 0] >= x_min)
            & (candidate_positions[:, 0] <= x_max)
            & (candidate_positions[:, 1] >= y_min)
            & (candidate_positions[:, 1] <= y_max)
        )
        choices = velocity_pairs[valid]
        if len(choices) == 0:
            raise RuntimeError("No valid continuation velocity. Check the configured velocity support.")
        vx, vy = choices[motion_rng.integers(len(choices))]
        motion[t] = (vx, vy)
        x += int(vx)
        y += int(vy)

    return motion


def compose_parameters(shape, rotation, x0, y0, motion):
    return np.concatenate([
        np.asarray([shape, rotation, x0, y0], dtype=np.float32),
        np.asarray(motion, dtype=np.float32).reshape(-1),
    ])


def extend_base_parameters(base_parameters, extension_rng):
    """Keep the original eight factors, then sample later velocities as nuisance variables."""
    base_parameters = np.asarray(base_parameters, dtype=np.float32)
    if HORIZON == 3:
        return base_parameters.copy()

    shape, rotation, x0, y0 = base_parameters[:4]
    initial_motion = base_parameters[4:8].reshape(2, 2).astype(int)
    bounds = sprite_centre_bounds(shape, rotation)
    positions = make_trajectory(x0, y0, initial_motion)
    x, y = map(int, positions[-1])

    remaining = np.empty((HORIZON - 3, 2), dtype=np.float32)
    velocity_pairs = np.asarray(
        [(vx, vy) for vx in VELOCITY_VALUES for vy in VELOCITY_VALUES],
        dtype=int,
    )
    x_min, x_max, y_min, y_max = bounds
    for t in range(HORIZON - 3):
        candidate_positions = velocity_pairs + np.asarray([x, y])
        valid = (
            (candidate_positions[:, 0] >= x_min)
            & (candidate_positions[:, 0] <= x_max)
            & (candidate_positions[:, 1] >= y_min)
            & (candidate_positions[:, 1] <= y_max)
        )
        choices = velocity_pairs[valid]
        if len(choices) == 0:
            raise RuntimeError("No valid long-horizon continuation from a valid three-frame sample.")
        vx, vy = choices[extension_rng.integers(len(choices))]
        remaining[t] = (vx, vy)
        x += int(vx)
        y += int(vy)

    return np.concatenate([base_parameters, remaining.reshape(-1)]).astype(np.float32)


def sample_long_parameters(sample_rng, fixed=None):
    base = sample_parameters(sample_rng, SIM_CONFIG, fixed=fixed)
    params = extend_base_parameters(base, sample_rng)
    # Final assertion: every generated frame must remain strictly inside the border.
    simulate_video(params, image_size=H, shape_size=float(SIM_CONFIG["shape_size"]))
    return params


def parameters_to_videos(parameters):
    parameters = np.asarray(parameters, dtype=np.float32)
    flat = parameters.reshape(-1, parameters.shape[-1])
    videos = [
        simulate_video(p, image_size=H, shape_size=float(SIM_CONFIG["shape_size"]))
        for p in flat
    ]
    videos = np.asarray(videos, dtype=np.float32)[..., None]
    return videos.reshape(*parameters.shape[:-1], HORIZON, H, W, C)


def sample_long_dataset(n, sample_rng):
    params = np.stack([sample_long_parameters(sample_rng) for _ in range(n)])
    return params, parameters_to_videos(params)


#%% Model inference helpers
@eqx.filter_jit
def representation_batch_state_actions(model, videos):
    return jax.vmap(model.video_representation)(videos)


@eqx.filter_jit
def representation_batch_states_only(model, videos):
    def one_video(video):
        latents = jax.vmap(model.encode_frame)(video)
        return latents.reshape(-1)
    return jax.vmap(one_video)(videos)


def represent(model, videos):
    rows = []
    for start in range(0, len(videos), REP_BATCH):
        batch = jnp.asarray(videos[start:start + REP_BATCH])
        if VIDEO_REPRESENTATION == "state_actions":
            rows.append(np.asarray(representation_batch_state_actions(model, batch)))
        else:
            rows.append(np.asarray(representation_batch_states_only(model, batch)))
    return np.concatenate(rows, axis=0)


@eqx.filter_jit
def reconstruct_rollout_batch(model, videos, coords_grid):
    def one_video(video):
        times = jnp.linspace(0.0, 1.0, video.shape[0])
        gt_z = jax.vmap(model.encode_frame)(video)
        recon = jax.vmap(lambda z, t: model.decode_frame(z, coords_grid, t))(gt_z, times)
        _, _, actions, _ = jax.vmap(model.infer_action)(gt_z[:-1], gt_z[1:])

        def step(z_prev, action):
            _, z_next = model.transition_model(z_prev, action)
            return z_next, z_next

        _, pred_tail = jax.lax.scan(step, gt_z[0], actions)
        pred_z = jnp.concatenate([gt_z[:1], pred_tail], axis=0)
        rollout = jax.vmap(lambda z, t: model.decode_frame(z, coords_grid, t))(pred_z, times)
        return gt_z, pred_z, actions, recon, rollout

    return jax.vmap(one_video)(videos)


def reconstruct_rollout(model, videos):
    outputs = [[], [], [], [], []]
    for start in range(0, len(videos), EVAL_BATCH):
        batch = jnp.asarray(videos[start:start + EVAL_BATCH])
        chunk = reconstruct_rollout_batch(model, batch, COORDS_GRID)
        for store, value in zip(outputs, chunk):
            store.append(np.asarray(value))
    return tuple(np.concatenate(parts, axis=0) for parts in outputs)


@eqx.filter_jit
def motion_swap_batch(model, source_videos, alien_videos, coords_grid):
    def one_pair(source_video, alien_video):
        times = jnp.linspace(0.0, 1.0, source_video.shape[0])
        source_z = jax.vmap(model.encode_frame)(source_video)
        alien_z = jax.vmap(model.encode_frame)(alien_video)
        _, _, source_actions, _ = jax.vmap(model.infer_action)(source_z[:-1], source_z[1:])
        _, _, alien_actions, _ = jax.vmap(model.infer_action)(alien_z[:-1], alien_z[1:])
        time_idx = jnp.arange(source_video.shape[0] - 1)

        def step(z_prev, inputs):
            t, source_action, alien_action = inputs
            action = jnp.where(t >= INTERVENTION_STEP, alien_action, source_action)
            _, z_next = model.transition_model(z_prev, action)
            return z_next, z_next

        _, pred_tail = jax.lax.scan(step, source_z[0], (time_idx, source_actions, alien_actions))
        pred_z = jnp.concatenate([source_z[:1], pred_tail], axis=0)
        video = jax.vmap(lambda z, t: model.decode_frame(z, coords_grid, t))(pred_z, times)
        return video, source_actions, alien_actions

    return jax.vmap(one_pair)(source_videos, alien_videos)


@eqx.filter_jit
def content_swap_batch(model, source_videos, alien_videos, coords_grid):
    def one_pair(source_video, alien_video):
        times = jnp.linspace(0.0, 1.0, source_video.shape[0])
        source_z = jax.vmap(model.encode_frame)(source_video)
        alien_z = jax.vmap(model.encode_frame)(alien_video)
        _, _, source_actions, _ = jax.vmap(model.infer_action)(source_z[:-1], source_z[1:])
        time_idx = jnp.arange(source_video.shape[0] - 1)

        def step(z_prev, inputs):
            t, source_action, alien_state = inputs
            # Algorithm 2 intervention occurs immediately before the FDM step.
            z_for_transition = jnp.where(t >= INTERVENTION_STEP, alien_state, z_prev)
            _, z_next = model.transition_model(z_for_transition, source_action)
            return z_next, z_next

        _, pred_tail = jax.lax.scan(step, source_z[0], (time_idx, source_actions, alien_z[:-1]))
        pred_z = jnp.concatenate([source_z[:1], pred_tail], axis=0)
        video = jax.vmap(lambda z, t: model.decode_frame(z, coords_grid, t))(pred_z, times)
        return video, source_actions

    return jax.vmap(one_pair)(source_videos, alien_videos)


def run_swap_batches(model, source_videos, alien_videos, kind):
    videos = []
    auxiliary = []
    fn = motion_swap_batch if kind == "motion" else content_swap_batch
    for start in range(0, len(source_videos), EVAL_BATCH):
        s = jnp.asarray(source_videos[start:start + EVAL_BATCH])
        a = jnp.asarray(alien_videos[start:start + EVAL_BATCH])
        out = fn(model, s, a, COORDS_GRID)
        videos.append(np.asarray(out[0]))
        auxiliary.append(tuple(np.asarray(x) for x in out[1:]))
    merged_aux = []
    for i in range(len(auxiliary[0])):
        merged_aux.append(np.concatenate([chunk[i] for chunk in auxiliary], axis=0))
    return np.concatenate(videos, axis=0), tuple(merged_aux)


#%% Quantitative image/distribution metrics
def _histp(data):
    h, _ = np.histogram(np.asarray(data).ravel(), bins=HIST_BINS, range=(0.0, 1.0))
    h = h.astype(np.float64) + 1e-12
    return h / h.sum()


def _histv():
    edges = np.linspace(0.0, 1.0, HIST_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def metric_wasserstein(target, pred):
    v = _histv()
    return float(wasserstein_distance(v, v, u_weights=_histp(target), v_weights=_histp(pred)))


def metric_jsd(target, pred):
    return float(jensenshannon(_histp(target), _histp(pred)))


def metric_bhattacharyya(target, pred):
    p, q = _histp(target), _histp(pred)
    return float(-np.log(np.sum(np.sqrt(p * q)) + 1e-15))


def metric_ssim(target, pred):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(target, pred, data_range=1.0))
    except ImportError:
        mu1, mu2 = target.mean(), pred.mean()
        s1, s2 = target.std(), pred.std()
        cov = float(np.mean((target - mu1) * (pred - mu2)))
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (s1**2 + s2**2 + c2)))


def metric_fft(target, pred):
    mag_t = np.abs(np.fft.fft2(target))
    mag_p = np.abs(np.fft.fft2(pred))
    return float(np.linalg.norm(mag_t - mag_p) / mag_t.size)


def binary_iou_np(target, pred):
    target = target > 0.5
    pred = pred > 0.5
    intersection = np.logical_and(target, pred).sum()
    union = np.logical_or(target, pred).sum()
    return float(intersection / max(union, 1))


def binary_dice_np(target, pred):
    target = target > 0.5
    pred = pred > 0.5
    intersection = np.logical_and(target, pred).sum()
    denom = target.sum() + pred.sum()
    return float(2.0 * intersection / max(denom, 1))


def centre_of_mass(frame):
    mask = np.asarray(frame) > 0.5
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.asarray([xx.mean(), yy.mean()], dtype=np.float64)


def score_video_pairs(predictions, targets, start_frame=0):
    """Per-sequence/per-frame scores against an exact simulator target."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    T_eval = predictions.shape[1] - start_frame
    metric_names = (
        "mse", "mae", "psnr", "ssim", "iou", "dice",
        "wasserstein", "jsd", "bhattacharyya", "fft", "position_error_px",
    )
    scores = {name: np.empty((len(predictions), T_eval), dtype=np.float64) for name in metric_names}

    for n in range(len(predictions)):
        for j, t in enumerate(range(start_frame, predictions.shape[1])):
            pred_raw = np.asarray(predictions[n, t, ..., 0], dtype=np.float64)
            target = np.asarray(targets[n, t, ..., 0], dtype=np.float64)
            pred = np.clip(pred_raw, 0.0, 1.0)

            mse = float(np.mean((pred_raw - target) ** 2))
            mae = float(np.mean(np.abs(pred_raw - target)))
            clipped_mse = float(np.mean((pred - target) ** 2))
            psnr = float(10.0 * np.log10(1.0 / max(clipped_mse, 1e-12)))

            scores["mse"][n, j] = mse
            scores["mae"][n, j] = mae
            scores["psnr"][n, j] = psnr
            scores["ssim"][n, j] = metric_ssim(target, pred)
            scores["iou"][n, j] = binary_iou_np(target, pred)
            scores["dice"][n, j] = binary_dice_np(target, pred)
            scores["wasserstein"][n, j] = metric_wasserstein(target, pred)
            scores["jsd"][n, j] = metric_jsd(target, pred)
            scores["bhattacharyya"][n, j] = metric_bhattacharyya(target, pred)
            scores["fft"][n, j] = metric_fft(target, pred)
            scores["position_error_px"][n, j] = np.linalg.norm(centre_of_mass(target) - centre_of_mass(pred))

    return scores


def summarise_scores(scores):
    return {
        metric: {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "median": float(np.nanmedian(values)),
        }
        for metric, values in scores.items()
    }


#%% Shared summary/result helpers
summary_rows = []


def record_scores(experiment, model_name, mode, scores):
    for metric, values in scores.items():
        summary_rows.append({
            "experiment": experiment,
            "model": model_name,
            "mode": mode,
            "metric": metric,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "median": float(np.nanmedian(values)),
        })


def plot_metric_over_time(results, metric, title, ylabel, path, start_frame=0):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(start_frame, HORIZON)
    for item in results:
        values = item["scores"][metric]
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        ax.plot(x, mean, label=pretty_mode(item["mode"]))
        ax.fill_between(x, mean - std, mean + std, alpha=0.10)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    show_and_save(fig, path)


def plot_grouped_metrics(results, metrics, title, path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 5.8))
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(results))
    labels = [pretty_mode(item["mode"]) for item in results]
    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        means = [np.nanmean(item["scores"][metric]) for item in results]
        stds = [np.nanstd(item["scores"][metric]) for item in results]
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title() + (" ↑" if higher_better else " ↓"))
        style_axis(ax)
        add_bar_labels(ax, bars)
    fig.suptitle(title)
    show_and_save(fig, path)


#%% GROUP 1 — long-video Beta-VAE-style linear-probe experiment
def make_probe_design(features_per_factor, design_rng):
    n = N_FACTORS * features_per_factor
    params_a = np.empty((n, L, PARAM_DIM), dtype=np.float32)
    params_b = np.empty_like(params_a)
    labels = np.empty(n, dtype=np.int64)
    fixed_values = np.empty(n, dtype=np.float32)

    row = 0
    for factor_idx, factor_name in enumerate(FACTOR_NAMES):
        support = SUPPORTS[factor_name]
        for _ in range(features_per_factor):
            fixed_value = support[design_rng.integers(len(support))]
            fixed = {
                "x0": SUPPORTS["x0"][design_rng.integers(len(SUPPORTS["x0"]))],
                "y0": SUPPORTS["y0"][design_rng.integers(len(SUPPORTS["y0"]))],
                factor_name: fixed_value,
            }
            for pair_idx in range(L):
                params_a[row, pair_idx] = sample_long_parameters(design_rng, fixed=fixed)
                params_b[row, pair_idx] = sample_long_parameters(design_rng, fixed=fixed)
            labels[row] = factor_idx
            fixed_values[row] = fixed_value
            row += 1

    order = design_rng.permutation(n)
    return params_a[order], params_b[order], labels[order], fixed_values[order]


def metric_features(model, params_a, params_b):
    n_features = params_a.shape[0]
    features = []
    for start in range(0, n_features, FEATURE_CHUNK):
        pa = params_a[start:start + FEATURE_CHUNK]
        pb = params_b[start:start + FEATURE_CHUNK]
        va = parameters_to_videos(pa).reshape(-1, HORIZON, H, W, C)
        vb = parameters_to_videos(pb).reshape(-1, HORIZON, H, W, C)
        ra = represent(model, va).reshape(len(pa), L, -1)
        rb = represent(model, vb).reshape(len(pb), L, -1)
        features.append(np.mean(np.abs(ra - rb), axis=1))
    return np.concatenate(features, axis=0)


def fit_linear_probe(train_x, train_y, test_x, test_y):
    """Same standardised multinomial linear logistic probe as wm_eval.py."""
    n_classes = N_FACTORS
    n_features = train_x.shape[1]
    l2 = 1.0 / (CLASSIFIER_C * max(len(train_x), 1))

    def unpack(theta):
        cut = n_classes * n_features
        return theta[:cut].reshape(n_classes, n_features), theta[cut:]

    def log_probs(theta, x):
        weights, intercept = unpack(theta)
        scores = x @ weights.T + intercept
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def data_loss(theta, x, y):
        lp = log_probs(theta, x)
        return float(-np.mean(lp[np.arange(len(y)), y]))

    def objective_and_grad(theta):
        weights, intercept = unpack(theta)
        scores = train_x @ weights.T + intercept
        lp = scores - logsumexp(scores, axis=1, keepdims=True)
        probs = np.exp(lp)
        loss = -np.mean(lp[np.arange(len(train_y)), train_y]) + 0.5 * l2 * np.sum(weights ** 2)
        residual = probs
        residual[np.arange(len(train_y)), train_y] -= 1.0
        residual /= len(train_y)
        grad_w = residual.T @ train_x + l2 * weights
        grad_b = residual.sum(axis=0)
        return float(loss), np.concatenate([grad_w.reshape(-1), grad_b]).astype(np.float64, copy=False)

    theta0 = np.zeros(n_classes * n_features + n_classes, dtype=np.float64)
    history = {
        "iteration": [0],
        "train_loss": [data_loss(theta0, train_x, train_y)],
        "test_loss": [data_loss(theta0, test_x, test_y)],
    }

    def callback(theta):
        history["iteration"].append(len(history["iteration"]))
        history["train_loss"].append(data_loss(theta, train_x, train_y))
        history["test_loss"].append(data_loss(theta, test_x, test_y))

    result = minimize(
        objective_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": CLASSIFIER_MAX_ITER, "ftol": CLASSIFIER_TOL, "gtol": CLASSIFIER_GRAD_TOL},
    )
    weights, intercept = unpack(result.x)
    train_pred = np.argmax(train_x @ weights.T + intercept, axis=1)
    test_pred = np.argmax(test_x @ weights.T + intercept, axis=1)
    history = {k: np.asarray(v) for k, v in history.items()}
    return weights, intercept, train_pred, test_pred, history, result


print("\n" + "=" * 86)
print("GROUP 1/4 — LONG-VIDEO DISENTANGLEMENT")
print("=" * 86)
train_params_a, train_params_b, train_y, train_fixed = make_probe_design(TRAIN_PER_FACTOR, rng)
test_params_a, test_params_b, test_y, test_fixed = make_probe_design(TEST_PER_FACTOR, rng)

probe_archive = {
    "factor_names": np.asarray(FACTOR_NAMES),
    "video_repreentation": np.asarray(VIDEO_REPRESENTATION),
    "train_labels": train_y,
    "test_labels": test_y,
    "train_fixed_values": train_fixed,
    "test_fixed_values": test_fixed,
    "train_parameters_a": train_params_a,
    "train_parameters_b": train_params_b,
    "test_parameters_a": test_params_a,
    "test_parameters_b": test_params_b,
}
probe_results = []
loaded_models = []

for run_dir, cfg in zip(RUN_DIRS, run_configs):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    print(f"\n{name}: extracting T={HORIZON} video representations...")
    model = load_model(run_dir, cfg)
    loaded_models.append(model)

    train_x = metric_features(model, train_params_a, train_params_b)
    test_x = metric_features(model, test_params_a, test_params_b)
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    weights, intercept, train_pred, test_pred, history, opt_result = fit_linear_probe(
        train_x_scaled, train_y, test_x_scaled, test_y
    )

    train_accuracy = accuracy_score(train_y, train_pred)
    test_accuracy = accuracy_score(test_y, test_pred)
    confusion = confusion_matrix(test_y, test_pred, labels=np.arange(N_FACTORS))
    per_factor = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    print(f"  train accuracy: {100 * train_accuracy:.2f}%")
    print(f"  test accuracy:  {100 * test_accuracy:.2f}% (chance={100 / N_FACTORS:.2f}%)")
    for factor_name, score in zip(FACTOR_NAMES, per_factor):
        print(f"    {factor_name:>8s}: {100 * score:6.2f}%")

    probe_archive[f"{name}__mode"] = np.asarray(mode)
    probe_archive[f"{name}__run_dir"] = np.asarray(str(run_dir))
    probe_archive[f"{name}__train_features"] = train_x.astype(np.float32)
    probe_archive[f"{name}__test_features"] = test_x.astype(np.float32)
    probe_archive[f"{name}__scaler_mean"] = scaler.mean_.astype(np.float32)
    probe_archive[f"{name}__scaler_scale"] = scaler.scale_.astype(np.float32)
    probe_archive[f"{name}__classifier_coef"] = weights.astype(np.float32)
    probe_archive[f"{name}__classifier_intercept"] = intercept.astype(np.float32)
    probe_archive[f"{name}__train_prediction"] = train_pred.astype(np.int64)
    probe_archive[f"{name}__test_prediction"] = test_pred.astype(np.int64)
    probe_archive[f"{name}__train_accuracy"] = np.asarray(train_accuracy, dtype=np.float32)
    probe_archive[f"{name}__test_accuracy"] = np.asarray(test_accuracy, dtype=np.float32)
    probe_archive[f"{name}__per_factor_accuracy"] = per_factor.astype(np.float32)
    probe_archive[f"{name}__confusion_matrix"] = confusion.astype(np.int64)
    probe_archive[f"{name}__probe_iteration"] = history["iteration"].astype(np.int64)
    probe_archive[f"{name}__probe_train_loss"] = history["train_loss"].astype(np.float32)
    probe_archive[f"{name}__probe_test_loss"] = history["test_loss"].astype(np.float32)
    probe_archive[f"{name}__probe_converged"] = np.asarray(opt_result.success)

    probe_results.append({
        "name": name,
        "mode": mode,
        "train_accuracy": train_accuracy,
        "accuracy": test_accuracy,
        "per_factor": per_factor,
        "confusion": confusion,
        "history": history,
    })
    summary_rows.append({
        "experiment": "long_disentanglement", "model": name, "mode": mode,
        "metric": "accuracy", "mean": float(test_accuracy), "std": 0.0, "median": float(test_accuracy),
    })

np.savez_compressed(ARTEFACT_DIR / "group1_long_disentanglement.npz", **probe_archive)

# Probe optimisation curves.
fig, ax = plt.subplots(figsize=(10.8, 6.3))
for item in probe_results:
    h = item["history"]
    ax.plot(h["iteration"], h["test_loss"], label=pretty_mode(item["mode"]))
ax.set_xlabel("Linear-probe optimisation iteration")
ax.set_ylabel("Held-out cross-entropy")
ax.set_title(f"Long-video linear probes — {VIDEO_REPRESENTATION.replace('_', ' ')}")
ax.legend(frameon=False, ncol=2)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "group1_probe_loss.png")

# Overall probe score.
x = np.arange(len(probe_results))
labels = [pretty_mode(r["mode"]) for r in probe_results]
fig, ax = plt.subplots(figsize=(9.6, 6.2))
bars = ax.bar(x, [r["accuracy"] for r in probe_results], width=0.68)
ax.axhline(1.0 / N_FACTORS, linestyle="--", linewidth=2.0, label=f"Chance ({100/N_FACTORS:.1f}%)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylim(0, 1.06)
ax.set_ylabel("Held-out accuracy")
ax.set_title(f"Disentanglement from T={HORIZON} videos")
ax.legend(frameon=False)
ax.bar_label(bars, labels=[f"{100*r['accuracy']:.1f}%" for r in probe_results], padding=5, fontsize=14)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "group1_probe_accuracy_overall.png")

# Per-factor comparison.
fig, ax = plt.subplots(figsize=(13.0, 6.5))
width = 0.82 / len(probe_results)
factor_x = np.arange(N_FACTORS)
for i, item in enumerate(probe_results):
    ax.bar(factor_x - 0.41 + width / 2 + i * width, item["per_factor"], width=width, label=pretty_mode(item["mode"]))
ax.axhline(1.0 / N_FACTORS, linestyle="--", linewidth=1.8)
ax.set_xticks(factor_x)
ax.set_xticklabels(FACTOR_NAMES)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Fixed simulator factor")
ax.set_ylabel("Classification accuracy")
ax.set_title("Disentanglement by simulator factor")
ax.legend(frameon=False, ncol=2)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "group1_probe_accuracy_per_factor.png")


#%% GROUP 2 — long-horizon reconstruction and teacher-action rollout accuracy
print("\n" + "=" * 86)
print("GROUP 2/4 — RECONSTRUCTION AND LONG-HORIZON ROLLOUT")
print("=" * 86)
eval_parameters, eval_videos = sample_long_dataset(EVAL_VIDEOS, rng)
accuracy_archive = {
    "parameters": eval_parameters,
    "ground_truth": eval_videos,
}
accuracy_results = []

for run_dir, cfg, model in zip(RUN_DIRS, run_configs, loaded_models):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    print(f"\n{name}: reconstruction + teacher-action rollout...")
    gt_z, pred_z, actions, recon, rollout = reconstruct_rollout(model, eval_videos)
    recon_scores = score_video_pairs(recon, eval_videos, start_frame=0)
    rollout_scores = score_video_pairs(rollout, eval_videos, start_frame=0)
    latent_mse = np.mean((pred_z - gt_z) ** 2, axis=-1)
    rollout_scores["latent_mse"] = latent_mse

    print(f"  reconstruction MSE: {np.mean(recon_scores['mse']):.6f}")
    print(f"  reconstruction SSIM: {np.mean(recon_scores['ssim']):.4f}")
    print(f"  rollout MSE:        {np.mean(rollout_scores['mse']):.6f}")
    print(f"  rollout SSIM:       {np.mean(rollout_scores['ssim']):.4f}")
    print(f"  rollout IoU:        {np.mean(rollout_scores['iou']):.4f}")

    accuracy_archive[f"{name}__mode"] = np.asarray(mode)
    accuracy_archive[f"{name}__reconstruction"] = recon[:8].astype(np.float32)
    accuracy_archive[f"{name}__rollout"] = rollout[:8].astype(np.float32)
    accuracy_archive[f"{name}__actions"] = actions.astype(np.float32)
    for metric, values in recon_scores.items():
        accuracy_archive[f"{name}__reconstruction__{metric}"] = values.astype(np.float32)
    for metric, values in rollout_scores.items():
        accuracy_archive[f"{name}__rollout__{metric}"] = values.astype(np.float32)

    record_scores("reconstruction", name, mode, recon_scores)
    record_scores("long_rollout", name, mode, rollout_scores)
    accuracy_results.append({
        "name": name,
        "mode": mode,
        "recon_scores": recon_scores,
        "scores": rollout_scores,
        "recon": recon,
        "rollout": rollout,
    })

np.savez_compressed(ARTEFACT_DIR / "group2_reconstruction_rollout.npz", **accuracy_archive)

plot_grouped_metrics(
    [{"mode": r["mode"], "scores": r["recon_scores"]} for r in accuracy_results],
    [("mse", "MSE", False), ("ssim", "SSIM", True), ("iou", "IoU", True)],
    "Encoder reconstruction accuracy",
    PLOT_DIR / "group2_reconstruction_summary.png",
)
plot_grouped_metrics(
    accuracy_results,
    [("mse", "MSE", False), ("ssim", "SSIM", True), ("iou", "IoU", True), ("position_error_px", "Position error [px]", False)],
    f"Long-horizon rollout accuracy (T={HORIZON})",
    PLOT_DIR / "group2_rollout_summary.png",
)
plot_metric_over_time(
    accuracy_results, "mse", "Rollout error accumulation", "MSE",
    PLOT_DIR / "group2_rollout_mse_over_time.png",
)
plot_metric_over_time(
    accuracy_results, "iou", "Rollout silhouette preservation", "IoU",
    PLOT_DIR / "group2_rollout_iou_over_time.png",
)
plot_metric_over_time(
    accuracy_results, "latent_mse", "Latent transition error accumulation", "Latent MSE",
    PLOT_DIR / "group2_rollout_latent_mse_over_time.png",
)

# Qualitative long-horizon example: GT plus each model rollout.
frame_indices = np.unique(np.linspace(0, HORIZON - 1, min(HORIZON, 7), dtype=int))
fig, axes = plt.subplots(1 + len(accuracy_results), len(frame_indices), figsize=(2.0 * len(frame_indices), 2.0 * (1 + len(accuracy_results))))
for c, t in enumerate(frame_indices):
    axes[0, c].imshow(eval_videos[0, t, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[0, c].set_title(f"t={t}")
for r, item in enumerate(accuracy_results, start=1):
    for c, t in enumerate(frame_indices):
        axes[r, c].imshow(np.clip(item["rollout"][0, t, ..., 0], 0, 1), cmap="gray", vmin=0, vmax=1)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("GT", fontsize=18, fontweight="bold")
for r, item in enumerate(accuracy_results, start=1):
    axes[r, 0].set_ylabel(pretty_mode(item["mode"]), fontsize=15, fontweight="bold")
fig.suptitle("Long-horizon rollout example")
show_and_save(fig, PLOT_DIR / "group2_rollout_example.png")


#%% Swap-design helpers: exact simulator counterfactuals
def sample_content(content_rng, different_shape=None):
    if different_shape is None:
        shape = int(SUPPORTS["shape"][content_rng.integers(len(SUPPORTS["shape"]))])
    else:
        choices = SUPPORTS["shape"][SUPPORTS["shape"] != different_shape]
        shape = int(choices[content_rng.integers(len(choices))])
    rotation = float(SUPPORTS["rotation"][content_rng.integers(len(SUPPORTS["rotation"]))])
    return shape, rotation


def sample_common_start(pair_rng, bounds):
    x_min, x_max, y_min, y_max = bounds
    valid_x = SUPPORTS["x0"][(SUPPORTS["x0"] >= x_min) & (SUPPORTS["x0"] <= x_max)]
    valid_y = SUPPORTS["y0"][(SUPPORTS["y0"] >= y_min) & (SUPPORTS["y0"] <= y_max)]
    if len(valid_x) == 0 or len(valid_y) == 0:
        raise RuntimeError("No common starting position inside the training support.")
    x0 = int(valid_x[pair_rng.integers(len(valid_x))])
    y0 = int(valid_y[pair_rng.integers(len(valid_y))])
    return x0, y0


def make_motion_swap_design(n_pairs, pair_rng):
    """Different content + different motion; exact target = source content with alien motion."""
    source_params, alien_params, target_params = [], [], []
    for _ in range(n_pairs):
        source_content = sample_content(pair_rng)
        alien_content = sample_content(pair_rng, different_shape=source_content[0])
        bounds = intersect_bounds(source_content, alien_content)
        x0, y0 = sample_common_start(pair_rng, bounds)
        source_motion = sample_motion_sequence(pair_rng, x0, y0, bounds)
        alien_motion = sample_motion_sequence(pair_rng, x0, y0, bounds)
        source_params.append(compose_parameters(*source_content, x0, y0, source_motion))
        alien_params.append(compose_parameters(*alien_content, x0, y0, alien_motion))
        target_params.append(compose_parameters(*source_content, x0, y0, alien_motion))
    return tuple(np.stack(x).astype(np.float32) for x in (source_params, alien_params, target_params))


def make_content_swap_design(n_pairs, pair_rng):
    """Different content + identical motion; alien video is the exact desired retargeted target."""
    source_params, alien_params = [], []
    for _ in range(n_pairs):
        source_content = sample_content(pair_rng)
        alien_content = sample_content(pair_rng, different_shape=source_content[0])
        bounds = intersect_bounds(source_content, alien_content)
        x0, y0 = sample_common_start(pair_rng, bounds)
        shared_motion = sample_motion_sequence(pair_rng, x0, y0, bounds)
        source_params.append(compose_parameters(*source_content, x0, y0, shared_motion))
        alien_params.append(compose_parameters(*alien_content, x0, y0, shared_motion))
    return tuple(np.stack(x).astype(np.float32) for x in (source_params, alien_params))


def plot_swap_examples(source, alien, target, model_results, title, path):
    frame_idx = np.unique(np.asarray([0, INTERVENTION_STEP, INTERVENTION_EFFECT_FRAME, HORIZON - 1]).clip(0, HORIZON - 1))
    n_rows = 3 + len(model_results)
    fig, axes = plt.subplots(n_rows, len(frame_idx), figsize=(2.25 * len(frame_idx), 2.1 * n_rows))
    row_videos = [("Source", source), ("Alien", alien), ("Target", target)]
    row_videos += [(pretty_mode(r["mode"]), r["video"]) for r in model_results]
    for r, (label, videos) in enumerate(row_videos):
        for c, t in enumerate(frame_idx):
            axes[r, c].imshow(np.clip(videos[0, t, ..., 0], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"t={t}")
        axes[r, 0].set_ylabel(label, fontsize=15, fontweight="bold")
    fig.suptitle(title)
    show_and_save(fig, path)


#%% GROUP 3 — Algorithm-2 motion swapping + quantitative target fidelity
print("\n" + "=" * 86)
print("GROUP 3/4 — MOTION SWAPPING")
print("=" * 86)
motion_source_params, motion_alien_params, motion_target_params = make_motion_swap_design(SWAP_PAIRS, rng)
motion_source_videos = parameters_to_videos(motion_source_params)
motion_alien_videos = parameters_to_videos(motion_alien_params)
motion_target_videos = parameters_to_videos(motion_target_params)

motion_archive = {
    "source_parameters": motion_source_params,
    "alien_parameters": motion_alien_params,
    "target_parameters": motion_target_params,
    "source_videos": motion_source_videos[:8],
    "alien_videos": motion_alien_videos[:8],
    "target_videos": motion_target_videos[:8],
    "intervention_step": np.asarray(INTERVENTION_STEP, dtype=np.int64),
}
motion_results = []

for run_dir, cfg, model in zip(RUN_DIRS, run_configs, loaded_models):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    print(f"\n{name}: replacing native actions with alien actions from transition {INTERVENTION_STEP} onward...")
    swapped, aux = run_swap_batches(model, motion_source_videos, motion_alien_videos, kind="motion")
    scores = score_video_pairs(swapped, motion_target_videos, start_frame=INTERVENTION_EFFECT_FRAME)
    print(f"  W1:   {np.nanmean(scores['wasserstein']):.5f}")
    print(f"  JSD:  {np.nanmean(scores['jsd']):.5f}")
    print(f"  B:    {np.nanmean(scores['bhattacharyya']):.5f}")
    print(f"  SSIM: {np.nanmean(scores['ssim']):.4f}")
    print(f"  FFT:  {np.nanmean(scores['fft']):.5f}")
    print(f"  IoU:  {np.nanmean(scores['iou']):.4f}")

    motion_archive[f"{name}__mode"] = np.asarray(mode)
    motion_archive[f"{name}__swapped_examples"] = swapped[:8].astype(np.float32)
    motion_archive[f"{name}__source_actions"] = aux[0].astype(np.float32)
    motion_archive[f"{name}__alien_actions"] = aux[1].astype(np.float32)
    for metric, values in scores.items():
        motion_archive[f"{name}__{metric}"] = values.astype(np.float32)
    record_scores("motion_swap", name, mode, scores)
    motion_results.append({"name": name, "mode": mode, "scores": scores, "video": swapped})

np.savez_compressed(ARTEFACT_DIR / "group3_motion_swap.npz", **motion_archive)

plot_grouped_metrics(
    motion_results,
    [("wasserstein", "$W_1$", False), ("jsd", "JSD", False), ("bhattacharyya", "Bhattacharyya", False)],
    "Motion retargeting — distributional fidelity to exact target",
    PLOT_DIR / "group3_motion_distributional.png",
)
plot_grouped_metrics(
    motion_results,
    [("ssim", "SSIM", True), ("fft", "FFT magnitude distance", False), ("iou", "IoU", True)],
    "Motion retargeting — structural fidelity to exact target",
    PLOT_DIR / "group3_motion_structural.png",
)
plot_metric_over_time(
    motion_results, "ssim", "Motion retargeting after intervention", "SSIM",
    PLOT_DIR / "group3_motion_ssim_over_time.png", start_frame=INTERVENTION_EFFECT_FRAME,
)
plot_swap_examples(
    motion_source_videos, motion_alien_videos, motion_target_videos, motion_results,
    "Motion swap: source content + alien motion",
    PLOT_DIR / "group3_motion_examples.png",
)


#%% GROUP 4 — Algorithm-2 content retargeting + quantitative alien-target fidelity
print("\n" + "=" * 86)
print("GROUP 4/4 — CONTENT RETARGETING")
print("=" * 86)
content_source_params, content_alien_params = make_content_swap_design(SWAP_PAIRS, rng)
content_source_videos = parameters_to_videos(content_source_params)
content_alien_videos = parameters_to_videos(content_alien_params)
# Because motion and initial position are shared, the alien sequence is the exact desired target.
content_target_videos = content_alien_videos

content_archive = {
    "source_parameters": content_source_params,
    "alien_parameters": content_alien_params,
    "source_videos": content_source_videos[:8],
    "alien_target_videos": content_alien_videos[:8],
    "intervention_step": np.asarray(INTERVENTION_STEP, dtype=np.int64),
}
content_results = []

for run_dir, cfg, model in zip(RUN_DIRS, run_configs, loaded_models):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    print(f"\n{name}: replacing current state with alien state from transition {INTERVENTION_STEP} onward...")
    retargeted, aux = run_swap_batches(model, content_source_videos, content_alien_videos, kind="content")
    scores = score_video_pairs(retargeted, content_target_videos, start_frame=INTERVENTION_EFFECT_FRAME)
    print(f"  W1:   {np.nanmean(scores['wasserstein']):.5f}")
    print(f"  JSD:  {np.nanmean(scores['jsd']):.5f}")
    print(f"  B:    {np.nanmean(scores['bhattacharyya']):.5f}")
    print(f"  SSIM: {np.nanmean(scores['ssim']):.4f}")
    print(f"  FFT:  {np.nanmean(scores['fft']):.5f}")
    print(f"  IoU:  {np.nanmean(scores['iou']):.4f}")

    content_archive[f"{name}__mode"] = np.asarray(mode)
    content_archive[f"{name}__retargeted_examples"] = retargeted[:8].astype(np.float32)
    content_archive[f"{name}__source_actions"] = aux[0].astype(np.float32)
    for metric, values in scores.items():
        content_archive[f"{name}__{metric}"] = values.astype(np.float32)
    record_scores("content_retarget", name, mode, scores)
    content_results.append({"name": name, "mode": mode, "scores": scores, "video": retargeted})

np.savez_compressed(ARTEFACT_DIR / "group4_content_retarget.npz", **content_archive)

plot_grouped_metrics(
    content_results,
    [("wasserstein", "$W_1$", False), ("jsd", "JSD", False), ("bhattacharyya", "Bhattacharyya", False)],
    "Content retargeting — distributional fidelity to alien target",
    PLOT_DIR / "group4_content_distributional.png",
)
plot_grouped_metrics(
    content_results,
    [("ssim", "SSIM", True), ("fft", "FFT magnitude distance", False), ("iou", "IoU", True)],
    "Content retargeting — structural fidelity to alien target",
    PLOT_DIR / "group4_content_structural.png",
)
plot_metric_over_time(
    content_results, "ssim", "Content retargeting after intervention", "SSIM",
    PLOT_DIR / "group4_content_ssim_over_time.png", start_frame=INTERVENTION_EFFECT_FRAME,
)
plot_swap_examples(
    content_source_videos, content_alien_videos, content_target_videos, content_results,
    "Content retargeting: alien content + native motion",
    PLOT_DIR / "group4_content_examples.png",
)


#%% Final cross-experiment summaries
with open(ARTEFACT_DIR / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["experiment", "model", "mode", "metric", "mean", "std", "median"])
    writer.writeheader()
    writer.writerows(summary_rows)

summary_json = {
    "description": "Four-part long-horizon evaluation of all world-model configurations.",
    "groups": [
        "long-video fixed-factor linear probe",
        "encoder reconstruction and teacher-action long-horizon rollout",
        "Algorithm-2-style motion swapping",
        "Algorithm-2-style content retargeting",
    ],
    "video_repreentation": VIDEO_REPRESENTATION,
    "probe_factors": FACTOR_NAMES,
    "excluded_probe_factors": ["x0", "y0"],
    "horizon": HORIZON,
    "intervention_step": INTERVENTION_STEP,
    "intervention_effect_frame": INTERVENTION_EFFECT_FRAME,
    "rollout_action_source": "IDM actions inferred from consecutive ground-truth encoded states; this project has no GCM.",
    "motion_swap_target": "exact simulator counterfactual = source shape/rotation + alien velocity sequence, with common initial position",
    "content_swap_target": "alien sequence with the same initial position and motion as the source, so it is the exact desired content target",
    "retarget_metrics": ["wasserstein", "jsd", "bhattacharyya", "ssim", "fft", "mse", "mae", "psnr", "iou", "dice", "position_error_px"],
    "hist_bins": HIST_BINS,
    "runs": [str(p) for p in RUN_DIRS],
}
with open(ARTEFACT_DIR / "metadata.json", "w") as f:
    json.dump(summary_json, f, indent=2)

# Compact final comparison: exact-target SSIM for rollout and both interventions.
fig, ax = plt.subplots(figsize=(10.5, 6.3))
x = np.arange(len(loaded_models))
width = 0.24
labels = [pretty_mode(cfg["model"]["mode"]) for cfg in run_configs]
rollout_ssim = [np.nanmean(r["scores"]["ssim"]) for r in accuracy_results]
motion_ssim = [np.nanmean(r["scores"]["ssim"]) for r in motion_results]
content_ssim = [np.nanmean(r["scores"]["ssim"]) for r in content_results]
ax.bar(x - width, rollout_ssim, width=width, label="Long rollout")
ax.bar(x, motion_ssim, width=width, label="Motion swap")
ax.bar(x + width, content_ssim, width=width, label="Content retarget")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylabel("SSIM")
ax.set_ylim(0, 1.05)
ax.set_title("Structural fidelity across long-horizon experiments")
ax.legend(frameon=False, ncol=3)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "summary_ssim.png")

print("\n" + "=" * 86)
print("MULTI-EVALUATION COMPLETE")
print("=" * 86)
print(f"Run directory: {OUTPUT_DIR}")
print(f"Summary CSV:   {ARTEFACT_DIR / 'results.csv'}")
print(f"Metadata:      {ARTEFACT_DIR / 'metadata.json'}")
print("Saved group archives:")
print(f"  {ARTEFACT_DIR / 'group1_long_disentanglement.npz'}")
print(f"  {ARTEFACT_DIR / 'group2_reconstruction_rollout.npz'}")
print(f"  {ARTEFACT_DIR / 'group3_motion_swap.npz'}")
print(f"  {ARTEFACT_DIR / 'group4_content_retarget.npz'}")
