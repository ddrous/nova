#%% Imports and experiment paths
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from gendata import PARAMETER_NAMES, parameter_supports, sample_parameters, simulate_video
from models import WorldModel
from utils import configure_plots

configure_plots()
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 24,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.4,
    "savefig.dpi": 240,
})

# Edit this list in Jupyter, or pass run folders on the command line.
RUN_DIRS = []
# cli_runs = [Path(x) for x in sys.argv[1:] if Path(x).is_dir()]
# wm_folders = ["260909-122231-weight_ab", "260909-130907-weight_joint", "260909-151104-standard_ab", "260909-155921-standard_joint"]
wm_folders = ["260910-220611-weight_ab", "260910-220716-weight_joint", "260910-220755-standard_ab", "260910-220808-standard_joint"]
cli_runs = [Path("runs") / f for f in wm_folders]

# print(f"cli_runs: {cli_runs}")

if cli_runs:
    RUN_DIRS = cli_runs
else:
    RUN_DIRS = [Path(x) for x in RUN_DIRS]

# print(f"RUN_DIRS: {RUN_DIRS}")

if not RUN_DIRS:
    raise ValueError("Provide one or more trained run folders in RUN_DIRS or on the command line.")

OUTPUT_DIR = Path("runs") / f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}_dissentglement"
ARTEFACT_DIR = OUTPUT_DIR / "artefacts"
PLOT_DIR = OUTPUT_DIR / "plots"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=False)
PLOT_DIR.mkdir(exist_ok=True)

if Path("config.yaml").exists():
    shutil.copy2("config.yaml", OUTPUT_DIR / "config.yaml")
if Path("wm_eval.py").exists():
    shutil.copy2("wm_eval.py", OUTPUT_DIR / "wm_eval.py")


#%% Load run configurations and enforce a fair simulator setting
run_configs = []
for run_dir in RUN_DIRS:
    with open(run_dir / "config.yaml", "r") as f:
        run_configs.append(yaml.safe_load(f))

SIM_CONFIG = run_configs[0]["simulation"]
if int(SIM_CONFIG["num_frames"]) != 3:
    raise ValueError("This video disentanglement experiment requires simulation.num_frames=3.")
for cfg in run_configs[1:]:
    if cfg["simulation"] != SIM_CONFIG:
        raise ValueError("All compared runs must use exactly the same simulation configuration.")

# Metric settings come from the current config.yaml when present; otherwise from the first run.
if Path("config.yaml").exists():
    with open("config.yaml", "r") as f:
        metric_cfg = yaml.safe_load(f).get("disentanglement", run_configs[0]["disentanglement"])
else:
    metric_cfg = run_configs[0]["disentanglement"]

# x0 and y0 are deliberately held fixed in every classifier feature, so they are not targets.
FACTOR_NAMES = [name for name in PARAMETER_NAMES[:8] if name not in ("x0", "y0")]
N_FACTORS = len(FACTOR_NAMES)
SUPPORTS = parameter_supports(SIM_CONFIG)
L = int(metric_cfg["pairs_per_feature"])
TRAIN_PER_FACTOR = int(metric_cfg["train_features_per_factor"])
TEST_PER_FACTOR = int(metric_cfg["test_features_per_factor"])
REP_BATCH = int(metric_cfg.get("representation_batch_size", 128))
FEATURE_CHUNK = int(metric_cfg.get("feature_chunk_size", 16))
VIDEO_REPRESENTATION = metric_cfg.get("video_repreentation", "state_actions")
if VIDEO_REPRESENTATION not in ("state_actions", "states_only"):
    raise ValueError("disentanglement.video_repreentation must be 'state_actions' or 'states_only'.")
rng = np.random.default_rng(int(metric_cfg.get("seed", 991)))

print("Comparing runs:")
for run_dir, cfg in zip(RUN_DIRS, run_configs):
    print(f"  {run_dir} -> {cfg['model']['mode']}")
print(f"Evaluated factors: {FACTOR_NAMES}")
print(f"Video representation: {VIDEO_REPRESENTATION}")
print(f"Metric: L={L}, train={TRAIN_PER_FACTOR}/factor, test={TEST_PER_FACTOR}/factor")


#%% Beta-VAE-style video metric design
# One fixed ground-truth factor y per feature. For each feature, create L pairs of videos
# sharing y and the same initial (x0, y0), then later average |r_1-r_2| over L.
def make_design(features_per_factor, design_rng):
    n = len(FACTOR_NAMES) * features_per_factor
    params_a = np.empty((n, L, 8), dtype=np.float32)
    params_b = np.empty_like(params_a)
    labels = np.empty(n, dtype=np.int64)
    fixed_values = np.empty(n, dtype=np.float32)

    row = 0
    for factor_idx, factor_name in enumerate(FACTOR_NAMES):
        support = SUPPORTS[factor_name]
        for _ in range(features_per_factor):
            fixed_value = support[design_rng.integers(len(support))]
            # Position is a nuisance variable for this metric: fix it for the whole feature.
            fixed = {
                "x0": SUPPORTS["x0"][design_rng.integers(len(SUPPORTS["x0"]))],
                "y0": SUPPORTS["y0"][design_rng.integers(len(SUPPORTS["y0"]))],
            }
            fixed[factor_name] = fixed_value

            for pair_idx in range(L):
                params_a[row, pair_idx] = sample_parameters(design_rng, SIM_CONFIG, fixed=fixed)
                params_b[row, pair_idx] = sample_parameters(design_rng, SIM_CONFIG, fixed=fixed)
            labels[row] = factor_idx
            fixed_values[row] = fixed_value
            row += 1

    order = design_rng.permutation(n)
    return params_a[order], params_b[order], labels[order], fixed_values[order]


train_params_a, train_params_b, train_y, train_fixed = make_design(TRAIN_PER_FACTOR, rng)
test_params_a, test_params_b, test_y, test_fixed = make_design(TEST_PER_FACTOR, rng)

# Fixed examples: one pair for each evaluated factor, taken from the test design.
example_indices = np.asarray([np.flatnonzero(test_y == k)[0] for k in range(N_FACTORS)])
example_params_a = test_params_a[example_indices, 0]
example_params_b = test_params_b[example_indices, 0]


def parameters_to_videos(parameters):
    flat = parameters.reshape(-1, 8)
    videos = [
        simulate_video(p, image_size=int(SIM_CONFIG["image_size"]), shape_size=float(SIM_CONFIG["shape_size"]))
        for p in flat
    ]
    videos = np.asarray(videos, dtype=np.float32)[..., None]
    return videos.reshape(*parameters.shape[:-1], int(SIM_CONFIG["num_frames"]), int(SIM_CONFIG["image_size"]), int(SIM_CONFIG["image_size"]), 1)


example_videos_a = parameters_to_videos(example_params_a)
example_videos_b = parameters_to_videos(example_params_b)


#%% Model loading and representation extraction
@eqx.filter_jit
def representation_batch_state_actions(model, videos):
    return jax.vmap(model.video_representation)(videos)


@eqx.filter_jit
def representation_batch_states_only(model, videos):
    def one_video(video):
        latents = jax.vmap(model.encode_frame)(video)
        return latents.reshape(-1)
    return jax.vmap(one_video)(videos)


def load_model(run_dir, cfg):
    H = W = int(cfg["simulation"]["image_size"])
    key = jax.random.PRNGKey(int(cfg["seed"]))
    model = WorldModel(cfg, frame_shape=(H, W, 1), key=key)
    return eqx.tree_deserialise_leaves(run_dir / "artefacts" / "model.eqx", model)


def represent(model, videos):
    rows = []
    for start in range(0, len(videos), REP_BATCH):
        batch = videos[start:start + REP_BATCH]
        if VIDEO_REPRESENTATION == "state_actions":
            rows.append(np.asarray(representation_batch_state_actions(model, batch)))
        else:
            rows.append(np.asarray(representation_batch_states_only(model, batch)))
    return np.concatenate(rows, axis=0)


def metric_features(model, params_a, params_b):
    n_features = params_a.shape[0]
    features = []
    for start in range(0, n_features, FEATURE_CHUNK):
        pa = params_a[start:start + FEATURE_CHUNK]
        pb = params_b[start:start + FEATURE_CHUNK]
        va = parameters_to_videos(pa).reshape(-1, *example_videos_a.shape[1:])
        vb = parameters_to_videos(pb).reshape(-1, *example_videos_a.shape[1:])
        ra = represent(model, va).reshape(len(pa), L, -1)
        rb = represent(model, vb).reshape(len(pb), L, -1)
        features.append(np.mean(np.abs(ra - rb), axis=1))
    return np.concatenate(features, axis=0)


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


def show_and_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    try:
        from IPython.display import display
        display(fig)
    except ImportError:
        plt.show()
    plt.close(fig)


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.9)
    ax.set_axisbelow(True)


#%% Linear probe with optimisation history
def fit_linear_probe(train_x, train_y, test_x, test_y):
    """Multinomial linear logistic probe; records train/test cross-entropy each L-BFGS iteration."""
    n_classes = N_FACTORS
    n_features = train_x.shape[1]
    max_iter = int(metric_cfg.get("classifier_max_iter", 3000))
    classifier_c = float(metric_cfg.get("classifier_C", 1.0))
    l2 = 1.0 / (classifier_c * max(len(train_x), 1))

    def unpack(theta):
        cut = n_classes * n_features
        return theta[:cut].reshape(n_classes, n_features), theta[cut:]

    def probabilities(theta, x):
        weights, intercept = unpack(theta)
        scores = x @ weights.T + intercept
        scores = scores - logsumexp(scores, axis=1, keepdims=True)
        return np.exp(scores), scores

    def data_loss(theta, x, y):
        _, log_probs = probabilities(theta, x)
        return float(-np.mean(log_probs[np.arange(len(y)), y]))

    def objective_and_grad(theta):
        weights, intercept = unpack(theta)
        scores = train_x @ weights.T + intercept
        log_norm = logsumexp(scores, axis=1, keepdims=True)
        log_probs = scores - log_norm
        probs = np.exp(log_probs)

        loss = -np.mean(log_probs[np.arange(len(train_y)), train_y])
        loss += 0.0 * l2 * np.sum(weights ** 2)

        residual = probs
        residual[np.arange(len(train_y)), train_y] -= 1.0
        residual /= len(train_y)
        grad_w = residual.T @ train_x + l2 * weights
        grad_b = residual.sum(axis=0)
        grad = np.concatenate([grad_w.reshape(-1), grad_b])
        return float(loss), grad.astype(np.float64, copy=False)

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
        options={
            "maxiter": max_iter,
            "ftol": float(metric_cfg.get("classifier_tol", 1e-9)),
            "gtol": float(metric_cfg.get("classifier_grad_tol", 1e-6)),
        },
    )

    weights, intercept = unpack(result.x)
    train_pred = np.argmax(train_x @ weights.T + intercept, axis=1)
    test_pred = np.argmax(test_x @ weights.T + intercept, axis=1)
    history = {k: np.asarray(v) for k, v in history.items()}
    return weights, intercept, train_pred, test_pred, history, result


#%% Evaluate every run with exactly the same train/test simulator pairs
archive = {
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
    "example_factor_index": np.arange(N_FACTORS, dtype=np.int64),
    "example_parameters_a": example_params_a,
    "example_parameters_b": example_params_b,
    "example_videos_a": example_videos_a,
    "example_videos_b": example_videos_b,
}
results = []

for run_dir, cfg in zip(RUN_DIRS, run_configs):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    print(f"\n{name}: loading and extracting video representations...")
    model = load_model(run_dir, cfg)

    train_x = metric_features(model, train_params_a, train_params_b)
    test_x = metric_features(model, test_params_a, test_params_b)

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    weights, intercept, train_pred, pred, probe_history, optimiser_result = fit_linear_probe(
        train_x_scaled, train_y, test_x_scaled, test_y
    )

    train_accuracy = accuracy_score(train_y, train_pred)
    accuracy = accuracy_score(test_y, pred)
    confusion = confusion_matrix(test_y, pred, labels=np.arange(N_FACTORS))
    per_factor = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    print(f"train accuracy: {100 * train_accuracy:.2f}%")
    print(f"test accuracy:  {100 * accuracy:.2f}% (chance = {100 / N_FACTORS:.2f}%)")
    print(f"probe iterations: {len(probe_history['iteration']) - 1}; converged={optimiser_result.success}")
    for factor_name, score in zip(FACTOR_NAMES, per_factor):
        print(f"  {factor_name:>8s}: {100 * score:6.2f}%")

    # Show the probe loss curve immediately after fitting this model.
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(probe_history["iteration"], probe_history["train_loss"], label="Train")
    ax.plot(probe_history["iteration"], probe_history["test_loss"], linestyle="--", label="Test")
    ax.set_xlabel("Linear-probe optimisation iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_yscale("log")
    ax.set_title(f"Linear probe — {pretty_mode(mode)}")
    ax.legend(frameon=False)
    style_axis(ax)
    show_and_save(fig, PLOT_DIR / f"probe_loss_{name}.png")

    example_rep_a = represent(model, example_videos_a)
    example_rep_b = represent(model, example_videos_b)

    archive[f"{name}__mode"] = np.asarray(mode)
    archive[f"{name}__run_dir"] = np.asarray(str(run_dir))
    archive[f"{name}__train_features"] = train_x.astype(np.float32)
    archive[f"{name}__test_features"] = test_x.astype(np.float32)
    archive[f"{name}__scaler_mean"] = scaler.mean_.astype(np.float32)
    archive[f"{name}__scaler_scale"] = scaler.scale_.astype(np.float32)
    archive[f"{name}__classifier_coef"] = weights.astype(np.float32)
    archive[f"{name}__classifier_intercept"] = intercept.astype(np.float32)
    archive[f"{name}__train_prediction"] = train_pred.astype(np.int64)
    archive[f"{name}__test_prediction"] = pred.astype(np.int64)
    archive[f"{name}__train_accuracy"] = np.asarray(train_accuracy, dtype=np.float32)
    archive[f"{name}__accuracy"] = np.asarray(accuracy, dtype=np.float32)
    archive[f"{name}__confusion_matrix"] = confusion.astype(np.int64)
    archive[f"{name}__per_factor_accuracy"] = per_factor.astype(np.float32)
    archive[f"{name}__probe_iteration"] = probe_history["iteration"].astype(np.int64)
    archive[f"{name}__probe_train_loss"] = probe_history["train_loss"].astype(np.float32)
    archive[f"{name}__probe_test_loss"] = probe_history["test_loss"].astype(np.float32)
    archive[f"{name}__probe_converged"] = np.asarray(optimiser_result.success)
    archive[f"{name}__example_representation_a"] = example_rep_a.astype(np.float32)
    archive[f"{name}__example_representation_b"] = example_rep_b.astype(np.float32)

    results.append((name, mode, accuracy, per_factor, confusion, train_accuracy, probe_history))


#%% Export one organised NumPy archive and human-readable summaries
representation_text = {
    "state_actions": "encoder z_0 concatenated with IDM actions a_0 and a_1",
    "states_only": "encoder states z_0, z_1 and z_2 concatenated",
}[VIDEO_REPRESENTATION]

metadata = {
    "description": "Beta-VAE-style fixed-factor metric extended to 3-frame videos.",
    "representation": representation_text,
    "video_repreentation": VIDEO_REPRESENTATION,
    "feature": "mean over L pairs of absolute representation difference",
    "classifier": "standardised multinomial linear logistic regression (L-BFGS)",
    "factor_names": list(FACTOR_NAMES),
    "excluded_factors": ["x0", "y0"],
    "pairs_per_feature": L,
    "train_features_per_factor": TRAIN_PER_FACTOR,
    "test_features_per_factor": TEST_PER_FACTOR,
    "simulation": SIM_CONFIG,
    "runs": [str(p) for p in RUN_DIRS],
}
archive["metadata_json"] = np.asarray(json.dumps(metadata))
np.savez_compressed(ARTEFACT_DIR / "disentanglement_results.npz", **archive)

with open(ARTEFACT_DIR / "results.csv", "w") as f:
    f.write("method,mode,train_accuracy,test_accuracy," + ",".join(FACTOR_NAMES) + "\n")
    for name, mode, accuracy, per_factor, _, train_accuracy, _ in results:
        values = ",".join(f"{x:.8f}" for x in per_factor)
        f.write(f"{name},{mode},{train_accuracy:.8f},{accuracy:.8f},{values}\n")

with open(ARTEFACT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


#%% Publication-ready comparison plots
plot_labels = [pretty_mode(r[1]) for r in results]
x_models = np.arange(len(results))
chance = 1.0 / N_FACTORS

# Overall fixed-factor score.
fig, ax = plt.subplots(figsize=(max(9.0, 2.0 * len(results)), 6.2))
bars = ax.bar(x_models, [r[2] for r in results], width=0.68)
ax.axhline(chance, linestyle="--", linewidth=2.0, label=f"Chance ({100 * chance:.1f}%)")
ax.set_xticks(x_models)
ax.set_xticklabels(plot_labels, rotation=15, ha="right")
ax.set_ylim(0, 1.06)
ax.set_ylabel("Test accuracy")
ax.set_title("Video-representation disentanglement")
ax.legend(frameon=False)
ax.bar_label(bars, labels=[f"{100 * r[2]:.1f}%" for r in results], padding=5, fontsize=15)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "scores_overall.png")

# Per-factor comparison across all models.
fig, ax = plt.subplots(figsize=(max(11.0, 1.8 * N_FACTORS), 6.8))
x_factor = np.arange(N_FACTORS)
width = 0.82 / max(len(results), 1)
for model_idx, result in enumerate(results):
    offset = (model_idx - (len(results) - 1) / 2.0) * width
    ax.bar(x_factor + offset, result[3], width=width, label=plot_labels[model_idx])
ax.axhline(chance, linestyle="--", linewidth=1.8, label=f"Chance ({100 * chance:.1f}%)")
ax.set_xticks(x_factor)
ax.set_xticklabels(FACTOR_NAMES)
ax.set_ylim(0, 1.04)
ax.set_xlabel("Fixed simulation factor")
ax.set_ylabel("Classification accuracy")
ax.set_title("Disentanglement by simulation factor")
ax.legend(frameon=False, ncol=min(3, len(results) + 1))
style_axis(ax)
show_and_save(fig, PLOT_DIR / "scores_per_factor.png")

# Train/test probe accuracy exposes overfitting of the linear readout itself.
fig, ax = plt.subplots(figsize=(max(9.0, 2.0 * len(results)), 6.2))
width = 0.36
train_bars = ax.bar(x_models - width / 2, [r[5] for r in results], width=width, label="Train")
test_bars = ax.bar(x_models + width / 2, [r[2] for r in results], width=width, label="Test")
ax.axhline(chance, linestyle="--", linewidth=1.8, label="Chance")
ax.set_xticks(x_models)
ax.set_xticklabels(plot_labels, rotation=15, ha="right")
ax.set_ylim(0, 1.06)
ax.set_ylabel("Accuracy")
ax.set_title("Linear-probe generalisation")
ax.legend(frameon=False)
ax.bar_label(train_bars, fmt="%.2f", padding=3, fontsize=13)
ax.bar_label(test_bars, fmt="%.2f", padding=3, fontsize=13)
style_axis(ax)
show_and_save(fig, PLOT_DIR / "probe_generalisation.png")

# Linear-probe loss curves for all methods on one figure.
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), sharey=True)
for result, label in zip(results, plot_labels):
    history = result[6]
    axes[0].plot(history["iteration"], history["train_loss"], label=label)
    axes[1].plot(history["iteration"], history["test_loss"], label=label)
for ax, title in zip(axes, ("Training loss", "Held-out loss")):
    ax.set_xlabel("Linear-probe optimisation iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    style_axis(ax)
axes[1].legend(frameon=False)
fig.suptitle("Linear-probe optimisation")
show_and_save(fig, PLOT_DIR / "probe_losses_comparison.png")

# Normalised confusion matrices make factor-specific confusions directly comparable.
for name, mode, _, _, confusion, _, _ in results:
    normalised = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    image = ax.imshow(normalised, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(N_FACTORS), FACTOR_NAMES, rotation=40, ha="right")
    ax.set_yticks(np.arange(N_FACTORS), FACTOR_NAMES)
    ax.set_xlabel("Predicted fixed factor")
    ax.set_ylabel("True fixed factor")
    ax.set_title(f"Confusion matrix — {pretty_mode(mode)}")
    for i in range(N_FACTORS):
        for j in range(N_FACTORS):
            ax.text(j, i, f"{100 * normalised[i, j]:.0f}%", ha="center", va="center", fontsize=12)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of examples")
    show_and_save(fig, PLOT_DIR / f"confusion_{name}.png")


#%% Example simulator pairs
fig, axes = plt.subplots(N_FACTORS, 2 * int(SIM_CONFIG["num_frames"]), figsize=(12, max(9, 2.0 * N_FACTORS)), squeeze=False)
for factor_idx, factor_name in enumerate(FACTOR_NAMES):
    for t in range(int(SIM_CONFIG["num_frames"])):
        axes[factor_idx, t].imshow(example_videos_a[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[factor_idx, t + int(SIM_CONFIG["num_frames"])].imshow(example_videos_b[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1)
    for ax in axes[factor_idx]:
        ax.set_axis_off()
    axes[factor_idx, 0].set_ylabel(factor_name, fontsize=18, fontweight="bold")
fig.suptitle("Example fixed-factor video pairs")
show_and_save(fig, PLOT_DIR / "fixed_factor_examples.png")

print(f"\nSaved disentanglement experiment to: {OUTPUT_DIR}")
print(f"Archive: {ARTEFACT_DIR / 'disentanglement_results.npz'}")
print(f"Plots: {PLOT_DIR}")
