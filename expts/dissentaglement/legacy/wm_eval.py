#%% Imports and experiment paths
import datetime
import json
import re
import sys
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from experiments.dissentaglement.legacy.gendata import PARAMETER_NAMES, parameter_supports, sample_parameters, simulate_video
from models import WorldModel

# Edit this list in Jupyter, or pass run folders on the command line.
RUN_DIRS = []
cli_runs = [Path(x) for x in sys.argv[1:] if Path(x).is_dir()]
if cli_runs:
    RUN_DIRS = cli_runs
else:
    RUN_DIRS = [Path(x) for x in RUN_DIRS]

if not RUN_DIRS:
    raise ValueError("Provide one or more trained run folders in RUN_DIRS or on the command line.")

OUTPUT_DIR = Path("disentanglement") / datetime.datetime.now().strftime("%y%m%d-%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


#%% Load run configurations and enforce a fair simulator setting
run_configs = []
for run_dir in RUN_DIRS:
    with open(run_dir / "config.yaml", "r") as f:
        run_configs.append(yaml.safe_load(f))

SIM_CONFIG = run_configs[0]["simulation"]
if int(SIM_CONFIG["num_frames"]) != 3:
    raise ValueError("This eight-factor video experiment requires simulation.num_frames=3.")
for cfg in run_configs[1:]:
    if cfg["simulation"] != SIM_CONFIG:
        raise ValueError("All compared runs must use exactly the same simulation configuration.")

# Metric settings come from the current config.yaml when present; otherwise from the first run.
if Path("config.yaml").exists():
    with open("config.yaml", "r") as f:
        metric_cfg = yaml.safe_load(f).get("disentanglement", run_configs[0]["disentanglement"])
else:
    metric_cfg = run_configs[0]["disentanglement"]

FACTOR_NAMES = PARAMETER_NAMES[:8]
SUPPORTS = parameter_supports(SIM_CONFIG)
L = int(metric_cfg["pairs_per_feature"])
TRAIN_PER_FACTOR = int(metric_cfg["train_features_per_factor"])
TEST_PER_FACTOR = int(metric_cfg["test_features_per_factor"])
REP_BATCH = int(metric_cfg.get("representation_batch_size", 128))
FEATURE_CHUNK = int(metric_cfg.get("feature_chunk_size", 16))
rng = np.random.default_rng(int(metric_cfg.get("seed", 991)))

print("Comparing runs:")
for run_dir, cfg in zip(RUN_DIRS, run_configs):
    print(f"  {run_dir} -> {cfg['model']['mode']}")
print(f"Eight factors: {FACTOR_NAMES}")
print(f"Metric: L={L}, train={TRAIN_PER_FACTOR}/factor, test={TEST_PER_FACTOR}/factor")


#%% Beta-VAE-style video metric design
# One fixed ground-truth factor y per feature. For each feature, create L pairs of videos
# sharing y, vary all other factors independently, and later average |r_1-r_2| over L.
def make_design(features_per_factor, design_rng):
    n = len(FACTOR_NAMES) * features_per_factor
    params_a = np.empty((n, L, 8), dtype=np.float32)
    params_b = np.empty_like(params_a)
    labels = np.empty(n, dtype=np.int64)
    fixed_values = np.empty(n, dtype=np.float32)

    row = 0
    for factor_idx, factor_name in enumerate(FACTOR_NAMES):     #@TODO: never vary the initial position factor, because it is not disentangled in the IDM representation
        support = SUPPORTS[factor_name]
        for _ in range(features_per_factor):
            fixed_value = support[design_rng.integers(len(support))]
            # fixed = {factor_name: fixed_value}

            ## fix the init position. The worst classifier may have up
            fixed = {"x_0": SUPPORTS["x_0"][design_rng.integers(len(SUPPORTS["x_0"]))],
                     "y_0": SUPPORTS["y_0"][design_rng.integers(len(SUPPORTS["y_0"]))],
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

# Fixed examples: one pair for each factor, taken from the test design.
example_indices = np.asarray([np.flatnonzero(test_y == k)[0] for k in range(8)])
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
def representation_batch(model, videos):
    return jax.vmap(model.video_representation)(videos)


def load_model(run_dir, cfg):
    H = W = int(cfg["simulation"]["image_size"])
    key = jax.random.PRNGKey(int(cfg["seed"]))
    model = WorldModel(cfg, frame_shape=(H, W, 1), key=key)
    return eqx.tree_deserialise_leaves(run_dir / "artefacts" / "model.eqx", model)


def represent(model, videos):
    rows = []
    for start in range(0, len(videos), REP_BATCH):
        rows.append(np.asarray(representation_batch(model, videos[start:start + REP_BATCH])))
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


#%% Evaluate every run with exactly the same train/test simulator pairs
archive = {
    "factor_names": np.asarray(FACTOR_NAMES),
    "train_labels": train_y,
    "test_labels": test_y,
    "train_fixed_values": train_fixed,
    "test_fixed_values": test_fixed,
    "train_parameters_a": train_params_a,
    "train_parameters_b": train_params_b,
    "test_parameters_a": test_params_a,
    "test_parameters_b": test_params_b,
    "example_factor_index": np.arange(8, dtype=np.int64),
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
    classifier = LogisticRegression(max_iter=int(metric_cfg.get("classifier_max_iter", 3000)))
    classifier.fit(train_x_scaled, train_y)
    pred = classifier.predict(test_x_scaled)

    accuracy = accuracy_score(test_y, pred)
    confusion = confusion_matrix(test_y, pred, labels=np.arange(8))
    per_factor = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    print(f"accuracy: {100 * accuracy:.2f}% (chance = 12.50%)")
    for factor_name, score in zip(FACTOR_NAMES, per_factor):
        print(f"  {factor_name:>8s}: {100 * score:6.2f}%")

    example_rep_a = represent(model, example_videos_a)
    example_rep_b = represent(model, example_videos_b)

    archive[f"{name}__mode"] = np.asarray(mode)
    archive[f"{name}__run_dir"] = np.asarray(str(run_dir))
    archive[f"{name}__train_features"] = train_x.astype(np.float32)
    archive[f"{name}__test_features"] = test_x.astype(np.float32)
    archive[f"{name}__scaler_mean"] = scaler.mean_.astype(np.float32)
    archive[f"{name}__scaler_scale"] = scaler.scale_.astype(np.float32)
    archive[f"{name}__classifier_coef"] = classifier.coef_.astype(np.float32)
    archive[f"{name}__classifier_intercept"] = classifier.intercept_.astype(np.float32)
    archive[f"{name}__test_prediction"] = pred.astype(np.int64)
    archive[f"{name}__accuracy"] = np.asarray(accuracy, dtype=np.float32)
    archive[f"{name}__confusion_matrix"] = confusion.astype(np.int64)
    archive[f"{name}__per_factor_accuracy"] = per_factor.astype(np.float32)
    archive[f"{name}__example_representation_a"] = example_rep_a.astype(np.float32)
    archive[f"{name}__example_representation_b"] = example_rep_b.astype(np.float32)

    results.append((name, mode, accuracy, per_factor, confusion))


#%% Export one organised NumPy archive and human-readable summaries
metadata = {
    "description": "Beta-VAE-style fixed-factor metric extended to 3-frame videos.",
    "representation": "encoder z_0 concatenated with IDM actions for t0->t1 and t1->t2",
    "feature": "mean over L pairs of absolute representation difference",
    "classifier": "standardised multinomial linear logistic regression",
    "factor_names": list(FACTOR_NAMES),
    "pairs_per_feature": L,
    "train_features_per_factor": TRAIN_PER_FACTOR,
    "test_features_per_factor": TEST_PER_FACTOR,
    "simulation": SIM_CONFIG,
    "runs": [str(p) for p in RUN_DIRS],
}
archive["metadata_json"] = np.asarray(json.dumps(metadata))
np.savez_compressed(OUTPUT_DIR / "disentanglement_results.npz", **archive)

with open(OUTPUT_DIR / "results.csv", "w") as f:
    f.write("method,mode,accuracy," + ",".join(FACTOR_NAMES) + "\n")
    for name, mode, accuracy, per_factor, _ in results:
        values = ",".join(f"{x:.8f}" for x in per_factor)
        f.write(f"{name},{mode},{accuracy:.8f},{values}\n")

with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


#%% Result plots and example simulator pairs
fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(results)), 4.5))
ax.bar(np.arange(len(results)), [r[2] for r in results])
ax.axhline(1.0 / 8.0, linestyle="--", linewidth=1, label="chance")
ax.set_xticks(np.arange(len(results)))
ax.set_xticklabels([r[0] for r in results], rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("fixed-factor classification accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "scores.png", dpi=180)
plt.close(fig)

for name, _, _, _, confusion in results:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, interpolation="nearest")
    ax.set_xticks(np.arange(8), FACTOR_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(8), FACTOR_NAMES)
    ax.set_xlabel("predicted fixed factor")
    ax.set_ylabel("true fixed factor")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_{name}.png", dpi=180)
    plt.close(fig)

fig, axes = plt.subplots(8, 2 * int(SIM_CONFIG["num_frames"]), figsize=(9, 14))
for factor_idx, factor_name in enumerate(FACTOR_NAMES):
    for t in range(int(SIM_CONFIG["num_frames"])):
        axes[factor_idx, t].imshow(example_videos_a[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[factor_idx, t + int(SIM_CONFIG["num_frames"])].imshow(example_videos_b[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1)
    for ax in axes[factor_idx]:
        ax.set_axis_off()
    axes[factor_idx, 0].set_ylabel(factor_name)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fixed_factor_examples.png", dpi=180)
plt.close(fig)

print(f"\nSaved disentanglement experiment to: {OUTPUT_DIR}")
print(f"Archive: {OUTPUT_DIR / 'disentanglement_results.npz'}")
