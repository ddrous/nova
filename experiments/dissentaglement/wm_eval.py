#%% Imports and experiment paths
import datetime
import json
import re
import shutil
import sys
import warnings
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

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

# Literature metrics implemented below:
#   Higgins et al. (ICLR 2017): beta-VAE fixed-factor score.
#   Kim & Mnih (ICML 2018): FactorVAE score.
#   Chen et al. (NeurIPS 2018): Mutual Information Gap (MIG).
#   Kumar et al. (ICLR 2018): Separated Attribute Predictability (SAP).
#   Eastwood & Williams (ICLR 2018): DCI.
#   Ridgeway & Mozer (NeurIPS 2018): Modularity + Explicitness.
#   Suter et al. (ICML 2019): Interventional Robustness Score (IRS).
#
# The point is not to find one magic number. These metrics encode different
# notions of disentanglement and are deliberately reported side-by-side.
#
# Reading guide:
#   beta_vae / factor_vae : fixed-factor invariance / axis alignment.
#   MIG / SAP             : compactness; does one factor concentrate in one code?
#   DCI                    : modularity, completeness, and factor recoverability.
#   Modularity             : does one code dimension specialise in one factor?
#   Explicitness           : can a simple linear readout recover each factor from
#                            the *whole* representation? This is deliberately
#                            tolerant of an invertible linear reparameterisation.
#   IRS                    : robustness of a factor-related code to nuisance changes.
#
# For this project Explicitness + DCI informativeness are especially important:
# an IDM is allowed to encode e.g. "go to x,y" rather than literal dx,dy. Such a
# representation can be useful and simple even when a fixed-factor variance
# metric does not regard it as perfectly axis-aligned.

# Edit this list in Jupyter, or pass run folders on the command line.
RUN_DIRS = []
# cli_runs = [Path(x) for x in sys.argv[1:] if Path(x).is_dir()]
# wm_folders = [
#     "260909-122231-weight_ab",
#     "260909-130907-weight_joint",
#     "260909-151104-standard_ab",
#     "260909-155921-standard_joint",
# ]
wm_folders = ["260910-220611-weight_ab", "260910-220716-weight_joint", "260910-220755-standard_ab", "260910-220808-standard_joint"]

cli_runs = [Path("runs") / f for f in wm_folders]

if cli_runs:
    RUN_DIRS = cli_runs
else:
    RUN_DIRS = [Path(x) for x in RUN_DIRS]

if not RUN_DIRS:
    raise ValueError("Provide one or more trained run folders in RUN_DIRS or on the command line.")

OUTPUT_DIR = Path("runs") / f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}_dissentglement2"
ARTEFACT_DIR = OUTPUT_DIR / "artefacts"
PLOT_DIR = OUTPUT_DIR / "plots"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=False)
PLOT_DIR.mkdir(exist_ok=True)

if Path("config.yaml").exists():
    shutil.copy2("config.yaml", OUTPUT_DIR / "config.yaml")
if Path("wm_eval2.py").exists():
    shutil.copy2("wm_eval2.py", OUTPUT_DIR / "wm_eval2.py")


#%% Load run configurations and metric settings
run_configs = []
for run_dir in RUN_DIRS:
    with open(run_dir / "config.yaml", "r") as f:
        run_configs.append(yaml.safe_load(f))

SIM_CONFIG = run_configs[0]["simulation"]
if int(SIM_CONFIG["num_frames"]) != 3:
    raise ValueError("This disentanglement benchmark currently expects simulation.num_frames=3.")
for cfg in run_configs[1:]:
    if cfg["simulation"] != SIM_CONFIG:
        raise ValueError("All compared runs must use exactly the same simulation configuration.")

if Path("config.yaml").exists():
    with open("config.yaml", "r") as f:
        metric_cfg = yaml.safe_load(f).get("disentanglement", run_configs[0]["disentanglement"])
else:
    metric_cfg = run_configs[0]["disentanglement"]

# x0/y0 are nuisance factors, not targets. For the fixed-factor metrics we also
# hold them fixed within each feature/vote, exactly as in wm_eval.py. For MIG,
# SAP, DCI, Explicitness and IRS they vary normally, which is useful: those
# metrics can test whether the six target factors stay readable despite position.
FACTOR_NAMES = [name for name in PARAMETER_NAMES[:8] if name not in ("x0", "y0")]
N_FACTORS = len(FACTOR_NAMES)
FACTOR_COLUMNS = [PARAMETER_NAMES.index(name) for name in FACTOR_NAMES]
SUPPORTS = parameter_supports(SIM_CONFIG)

L = int(metric_cfg.get("pairs_per_feature", 16))
TRAIN_PER_FACTOR = int(metric_cfg.get("train_features_per_factor", 128))
TEST_PER_FACTOR = int(metric_cfg.get("test_features_per_factor", 64))
REP_BATCH = int(metric_cfg.get("representation_batch_size", 128))
FEATURE_CHUNK = int(metric_cfg.get("feature_chunk_size", 16))
RANDOM_TRAIN = int(metric_cfg.get("literature_train_samples", 4096))
RANDOM_TEST = int(metric_cfg.get("literature_test_samples", 2048))
MI_BINS = int(metric_cfg.get("mi_bins", 20))
DCI_TREES = int(metric_cfg.get("dci_trees", 96))
REPORT_COMPONENTS = bool(metric_cfg.get("report_components", True))
VIDEO_REPRESENTATION = metric_cfg.get("video_repreentation", "state_actions")
if VIDEO_REPRESENTATION not in ("state_actions", "states_only"):
    raise ValueError("disentanglement.video_repreentation must be 'state_actions' or 'states_only'.")

SEED = int(metric_cfg.get("seed", 991))
rng = np.random.default_rng(SEED)

fixed_video_count = 2 * L * N_FACTORS * (TRAIN_PER_FACTOR + TEST_PER_FACTOR)
random_video_count = RANDOM_TRAIN + RANDOM_TEST
print("Comparing runs:")
for run_dir, cfg in zip(RUN_DIRS, run_configs):
    print(f"  {run_dir} -> {cfg['model']['mode']}")
print(f"Evaluated factors: {FACTOR_NAMES}")
print(f"Video representation: {VIDEO_REPRESENTATION}")
print(f"Fixed-factor design: L={L}, train={TRAIN_PER_FACTOR}/factor, test={TEST_PER_FACTOR}/factor")
print(f"Generic labelled design: train={RANDOM_TRAIN}, test={RANDOM_TEST}")
print(f"Approx. representations/model: {fixed_video_count + random_video_count:,}")


#%% Shared simulator designs
# beta-VAE / FactorVAE design: fix one target factor and one nuisance position.
def make_fixed_factor_design(features_per_factor, design_rng):
    n = N_FACTORS * features_per_factor
    params_a = np.empty((n, L, 8), dtype=np.float32)
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
                params_a[row, pair_idx] = sample_parameters(design_rng, SIM_CONFIG, fixed=fixed)
                params_b[row, pair_idx] = sample_parameters(design_rng, SIM_CONFIG, fixed=fixed)
            labels[row] = factor_idx
            fixed_values[row] = fixed_value
            row += 1

    order = design_rng.permutation(n)
    return params_a[order], params_b[order], labels[order], fixed_values[order]


def make_random_design(n_samples, design_rng):
    return np.asarray(
        [sample_parameters(design_rng, SIM_CONFIG) for _ in range(n_samples)],
        dtype=np.float32,
    )


def factor_labels_from_parameters(parameters):
    """Map simulator values to integer class IDs for the six evaluated factors."""
    labels = np.empty((len(parameters), N_FACTORS), dtype=np.int64)
    for factor_idx, (name, col) in enumerate(zip(FACTOR_NAMES, FACTOR_COLUMNS)):
        support = np.asarray(SUPPORTS[name], dtype=np.float64)
        values = np.asarray(parameters[:, col], dtype=np.float64)
        labels[:, factor_idx] = np.argmin(np.abs(values[:, None] - support[None, :]), axis=1)
    return labels


train_params_a, train_params_b, train_y, train_fixed = make_fixed_factor_design(TRAIN_PER_FACTOR, rng)
test_params_a, test_params_b, test_y, test_fixed = make_fixed_factor_design(TEST_PER_FACTOR, rng)
random_train_params = make_random_design(RANDOM_TRAIN, rng)
random_test_params = make_random_design(RANDOM_TEST, rng)
random_train_y = factor_labels_from_parameters(random_train_params)
random_test_y = factor_labels_from_parameters(random_test_params)

example_indices = np.asarray([np.flatnonzero(test_y == k)[0] for k in range(N_FACTORS)])
example_params_a = test_params_a[example_indices, 0]
example_params_b = test_params_b[example_indices, 0]


def parameters_to_videos(parameters):
    flat = parameters.reshape(-1, 8)
    videos = [
        simulate_video(
            p,
            image_size=int(SIM_CONFIG["image_size"]),
            shape_size=float(SIM_CONFIG["shape_size"]),
        )
        for p in flat
    ]
    videos = np.asarray(videos, dtype=np.float32)[..., None]
    return videos.reshape(
        *parameters.shape[:-1],
        int(SIM_CONFIG["num_frames"]),
        int(SIM_CONFIG["image_size"]),
        int(SIM_CONFIG["image_size"]),
        1,
    )


example_videos_a = parameters_to_videos(example_params_a)
example_videos_b = parameters_to_videos(example_params_b)


#%% Model loading and video representations
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


def represent_parameters(model, parameters):
    rows = []
    for start in range(0, len(parameters), REP_BATCH):
        videos = parameters_to_videos(parameters[start:start + REP_BATCH])
        rows.append(represent(model, videos))
    return np.concatenate(rows, axis=0)


def fixed_factor_statistics(model, params_a, params_b, global_std):
    """One pass supplies both Higgins features and FactorVAE low-variance votes."""
    beta_features, winners = [], []
    active = global_std > float(metric_cfg.get("active_std_threshold", 1e-8))
    safe_std = np.where(active, global_std, 1.0)

    for start in range(0, len(params_a), FEATURE_CHUNK):
        pa = params_a[start:start + FEATURE_CHUNK]
        pb = params_b[start:start + FEATURE_CHUNK]
        va = parameters_to_videos(pa).reshape(-1, *example_videos_a.shape[1:])
        vb = parameters_to_videos(pb).reshape(-1, *example_videos_a.shape[1:])
        ra = represent(model, va).reshape(len(pa), L, -1)
        rb = represent(model, vb).reshape(len(pb), L, -1)

        beta_features.append(np.mean(np.abs(ra - rb), axis=1))

        # Kim & Mnih: normalize code dimensions by global std, then choose the
        # dimension with the smallest variance under a fixed factor.
        group = np.concatenate([ra, rb], axis=1) / safe_std[None, None, :]
        variance = np.var(group, axis=1)
        variance[:, ~active] = np.inf
        winners.append(np.argmin(variance, axis=1))

    return np.concatenate(beta_features, axis=0), np.concatenate(winners, axis=0), active


def representation_components(model, codes):
    """No extra model calls: inspect where information lives in the same representation."""
    if not REPORT_COMPONENTS:
        return {"full": codes}

    d_z, d_u = int(model.latent_dim), int(model.action_dim)
    if VIDEO_REPRESENTATION == "state_actions":
        return {
            "full": codes,
            "state_z0": codes[:, :d_z],
            "actions": codes[:, d_z:],
        }
    return {
        "full": codes,
        "state_z0": codes[:, :d_z],
    }


def safe_name(path, mode):
    text = f"{path.name}_{mode}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def pretty_mode(mode):
    names = {
        "weight_ab": "NOVA A/B",
        "weight_abc": "NOVA ABC",
        "weight_joint": "NOVA Joint",
        "standard_ab": "Standard A/B",
        "standard_abc": "Standard ABC",
        "standard_joint": "Standard Joint",
        "standar_abc": "Standard ABC",
        "standard": "Standard Joint",
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


#%% Metric 1: Higgins et al. beta-VAE fixed-factor score
def fit_beta_probe(train_x, train_y, test_x, test_y):
    """Multinomial linear probe with a consistent L2 objective and loss history."""
    n_classes = N_FACTORS
    n_features = train_x.shape[1]
    max_iter = int(metric_cfg.get("classifier_max_iter", 1000))
    classifier_c = float(metric_cfg.get("classifier_C", 1.0))
    l2 = 1.0 / (classifier_c * max(len(train_x), 1))

    def unpack(theta):
        cut = n_classes * n_features
        return theta[:cut].reshape(n_classes, n_features), theta[cut:]

    def probabilities(theta, x):
        weights, intercept = unpack(theta)
        scores = x @ weights.T + intercept
        log_probs = scores - logsumexp(scores, axis=1, keepdims=True)
        return np.exp(log_probs), log_probs

    def data_loss(theta, x, y):
        _, log_probs = probabilities(theta, x)
        return float(-np.mean(log_probs[np.arange(len(y)), y]))

    def objective_and_grad(theta):
        weights, intercept = unpack(theta)
        scores = train_x @ weights.T + intercept
        log_probs = scores - logsumexp(scores, axis=1, keepdims=True)
        probs = np.exp(log_probs)

        loss = -np.mean(log_probs[np.arange(len(train_y)), train_y])
        loss += 0.5 * l2 * np.sum(weights ** 2)

        residual = probs.copy()
        residual[np.arange(len(train_y)), train_y] -= 1.0
        residual /= len(train_y)
        grad_w = residual.T @ train_x + l2 * weights
        grad_b = residual.sum(axis=0)
        return float(loss), np.concatenate([grad_w.reshape(-1), grad_b]).astype(np.float64)

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


#%% Metric 2: Kim & Mnih FactorVAE score
def compute_factor_vae_score(train_winners, train_labels, test_winners, test_labels, n_codes):
    votes = np.zeros((n_codes, N_FACTORS), dtype=np.int64)
    for winner, label in zip(train_winners, train_labels):
        votes[int(winner), int(label)] += 1

    mapping = np.full(n_codes, -1, dtype=np.int64)
    used = votes.sum(axis=1) > 0
    mapping[used] = np.argmax(votes[used], axis=1)
    pred = mapping[test_winners]
    score = float(np.mean(pred == test_labels))
    per_factor = np.asarray([
        np.mean(pred[test_labels == k] == k) if np.any(test_labels == k) else np.nan
        for k in range(N_FACTORS)
    ])
    return score, per_factor, votes, mapping, pred


#%% Metric 3/5: MIG and Modularity from one shared mutual-information matrix
def quantile_discretize(codes, n_bins):
    discrete = np.zeros_like(codes, dtype=np.int32)
    q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j in range(codes.shape[1]):
        col = codes[:, j]
        if np.std(col) < 1e-12:
            continue
        edges = np.unique(np.quantile(col, q))
        if len(edges):
            discrete[:, j] = np.digitize(col, edges, right=False)
    return discrete


def compute_mi_matrix(codes, factors, n_bins=20):
    discrete = quantile_discretize(codes, n_bins)
    matrix = np.zeros((codes.shape[1], factors.shape[1]), dtype=np.float64)
    for j in range(codes.shape[1]):
        if np.all(discrete[:, j] == discrete[0, j]):
            continue
        for k in range(factors.shape[1]):
            matrix[j, k] = mutual_info_score(factors[:, k], discrete[:, j])
    return matrix


def compute_mig_and_modularity(mi_matrix, factors):
    factor_entropy = np.asarray([
        mutual_info_score(factors[:, k], factors[:, k])
        for k in range(factors.shape[1])
    ])
    sorted_mi = np.sort(mi_matrix, axis=0)[::-1]
    top1 = sorted_mi[0]
    top2 = sorted_mi[1] if len(sorted_mi) > 1 else np.zeros_like(top1)
    mig_per_factor = (top1 - top2) / np.maximum(factor_entropy, 1e-12)
    mig = float(np.mean(mig_per_factor))

    squared = mi_matrix ** 2
    max_squared = np.max(squared, axis=1)
    denom = max_squared * max(N_FACTORS - 1, 1)
    delta = np.zeros(len(max_squared), dtype=np.float64)
    active = max_squared > 0
    delta[active] = (np.sum(squared[active], axis=1) - max_squared[active]) / denom[active]
    modularity_per_code = np.zeros_like(max_squared)
    modularity_per_code[active] = 1.0 - delta[active]
    modularity = float(np.mean(modularity_per_code))
    return mig, mig_per_factor, modularity, modularity_per_code, factor_entropy


#%% Metric 4: Kumar et al. SAP score
def compute_sap(train_codes, train_factors, test_codes, test_factors):
    score_matrix = np.zeros((train_codes.shape[1], N_FACTORS), dtype=np.float64)
    max_iter = int(metric_cfg.get("sap_max_iter", 3000))

    for code_idx in range(train_codes.shape[1]):
        x_train = train_codes[:, code_idx:code_idx + 1]
        x_test = test_codes[:, code_idx:code_idx + 1]
        if np.std(x_train) < 1e-12:
            continue

        for factor_idx in range(N_FACTORS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=max_iter)
                clf.fit(x_train, train_factors[:, factor_idx])
            score_matrix[code_idx, factor_idx] = np.mean(
                clf.predict(x_test) == test_factors[:, factor_idx]
            )

    sorted_scores = np.sort(score_matrix, axis=0)
    gaps = sorted_scores[-1] - sorted_scores[-2]
    return float(np.mean(gaps)), gaps, score_matrix


#%% Metric 5: Eastwood & Williams DCI
def _normalised_entropy(values, base):
    values = np.asarray(values, dtype=np.float64) + 1e-11
    return entropy(values, base=base)


def compute_dci(train_codes, train_factors, test_codes, test_factors):
    importance = np.zeros((train_codes.shape[1], N_FACTORS), dtype=np.float64)
    train_acc, test_acc = [], []

    for factor_idx in range(N_FACTORS):
        # Eastwood & Williams allow a nonlinear regressor/classifier for the
        # importance matrix. A random forest is also the strongest DCI variant
        # in the later Carbonneau et al. metric review, and is dramatically
        # cheaper here than fitting one boosted model per high-cardinality factor.
        clf = RandomForestClassifier(
            n_estimators=DCI_TREES,
            random_state=SEED + factor_idx,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=int(metric_cfg.get("dci_min_samples_leaf", 2)),
            class_weight="balanced_subsample",
        )
        clf.fit(train_codes, train_factors[:, factor_idx])
        importance[:, factor_idx] = np.abs(clf.feature_importances_)
        train_acc.append(np.mean(clf.predict(train_codes) == train_factors[:, factor_idx]))
        test_acc.append(np.mean(clf.predict(test_codes) == test_factors[:, factor_idx]))

    disent_per_code = np.asarray([
        1.0 - _normalised_entropy(importance[j], N_FACTORS)
        for j in range(importance.shape[0])
    ])
    code_weight = importance.sum(axis=1)
    if code_weight.sum() > 0:
        code_weight = code_weight / code_weight.sum()
    else:
        code_weight = np.ones_like(code_weight) / len(code_weight)
    disentanglement_score = float(np.sum(disent_per_code * code_weight))

    completeness_per_factor = np.asarray([
        1.0 - _normalised_entropy(importance[:, k], importance.shape[0])
        for k in range(N_FACTORS)
    ])
    factor_weight = importance.sum(axis=0)
    if factor_weight.sum() > 0:
        factor_weight = factor_weight / factor_weight.sum()
    else:
        factor_weight = np.ones_like(factor_weight) / len(factor_weight)
    completeness_score = float(np.sum(completeness_per_factor * factor_weight))

    return {
        "disentanglement": disentanglement_score,
        "completeness": completeness_score,
        "informativeness_train": float(np.mean(train_acc)),
        "informativeness_test": float(np.mean(test_acc)),
        "per_factor_train_accuracy": np.asarray(train_acc),
        "per_factor_test_accuracy": np.asarray(test_acc),
        "importance_matrix": importance,
        "disentanglement_per_code": disent_per_code,
        "completeness_per_factor": completeness_per_factor,
    }


#%% Metric 6: Ridgeway & Mozer Explicitness
def compute_explicitness(train_codes, train_factors, test_codes, test_factors):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_codes)
    x_test = scaler.transform(test_codes)
    train_auc, test_auc = [], []

    for factor_idx in range(N_FACTORS):
        y_train = train_factors[:, factor_idx]
        y_test = test_factors[:, factor_idx]
        clf = LogisticRegression(
            max_iter=int(metric_cfg.get("explicitness_max_iter", 1500)),
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(x_train, y_train)
        p_train = clf.predict_proba(x_train)
        p_test = clf.predict_proba(x_test)
        classes = clf.classes_

        if len(classes) == 2:
            train_auc.append(roc_auc_score(y_train, p_train[:, 1], labels=classes))
            test_auc.append(roc_auc_score(y_test, p_test[:, 1], labels=classes))
        else:
            train_auc.append(roc_auc_score(y_train, p_train, labels=classes, multi_class="ovr", average="macro"))
            test_auc.append(roc_auc_score(y_test, p_test, labels=classes, multi_class="ovr", average="macro"))

    return float(np.mean(train_auc)), float(np.mean(test_auc)), np.asarray(train_auc), np.asarray(test_auc)


#%% Metric 7: Suter et al. Interventional Robustness Score (IRS)
def compute_irs(codes, factors):
    """Observational estimator of IRS, valid here because simulator factors are independently sampled."""
    overall_mean = np.mean(codes, axis=0)
    normaliser = np.max(np.abs(codes - overall_mean), axis=0)
    robust_matrix = np.zeros((codes.shape[1], N_FACTORS), dtype=np.float64)

    for factor_idx in range(N_FACTORS):
        y = factors[:, factor_idx]
        empida = np.zeros(codes.shape[1], dtype=np.float64)
        for value in np.unique(y):
            group = codes[y == value]
            group_mean = np.mean(group, axis=0)
            mpida = np.max(np.abs(group - group_mean), axis=0)
            empida += (len(group) / len(codes)) * mpida
        robust_matrix[:, factor_idx] = 1.0 - empida / np.maximum(normaliser, 1e-12)

    robust_matrix = np.clip(robust_matrix, 0.0, 1.0)
    per_code = np.max(robust_matrix, axis=1)
    weights = normaliser.copy()
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights[:] = 1.0 / len(weights)
    score = float(np.sum(per_code * weights))

    # A factor-centric summary is useful for our simulator: for each factor,
    # report the best robust latent coordinate that predominantly tracks it.
    per_factor = np.max(robust_matrix, axis=0)
    return score, per_factor, robust_matrix, per_code, normaliser


#%% Evaluate every run on exactly the same simulator designs
archive = {
    "factor_names": np.asarray(FACTOR_NAMES),
    "video_repreentation": np.asarray(VIDEO_REPRESENTATION),
    "fixed_train_labels": train_y,
    "fixed_test_labels": test_y,
    "fixed_train_values": train_fixed,
    "fixed_test_values": test_fixed,
    "fixed_train_parameters_a": train_params_a,
    "fixed_train_parameters_b": train_params_b,
    "fixed_test_parameters_a": test_params_a,
    "fixed_test_parameters_b": test_params_b,
    "random_train_parameters": random_train_params,
    "random_test_parameters": random_test_params,
    "random_train_factor_labels": random_train_y,
    "random_test_factor_labels": random_test_y,
    "example_parameters_a": example_params_a,
    "example_parameters_b": example_params_b,
    "example_videos_a": example_videos_a,
    "example_videos_b": example_videos_b,
}
results = []
component_results = []

for run_dir, cfg in zip(RUN_DIRS, run_configs):
    mode = cfg["model"]["mode"]
    name = safe_name(run_dir, mode)
    label = pretty_mode(mode)
    print(f"\n{'=' * 80}\n{label}: loading model and extracting shared representation sets...")
    model = load_model(run_dir, cfg)

    random_train_codes = represent_parameters(model, random_train_params)
    random_test_codes = represent_parameters(model, random_test_params)
    global_std = np.std(np.concatenate([random_train_codes, random_test_codes], axis=0), axis=0)

    train_beta_x, train_winners, active = fixed_factor_statistics(
        model, train_params_a, train_params_b, global_std
    )
    test_beta_x, test_winners, _ = fixed_factor_statistics(
        model, test_params_a, test_params_b, global_std
    )

    # 1) Historical beta-VAE score.
    scaler = StandardScaler()
    train_beta_scaled = scaler.fit_transform(train_beta_x)
    test_beta_scaled = scaler.transform(test_beta_x)
    beta_w, beta_b, beta_train_pred, beta_test_pred, beta_history, beta_opt = fit_beta_probe(
        train_beta_scaled, train_y, test_beta_scaled, test_y
    )
    beta_train_acc = accuracy_score(train_y, beta_train_pred)
    beta_score = accuracy_score(test_y, beta_test_pred)
    beta_confusion = confusion_matrix(test_y, beta_test_pred, labels=np.arange(N_FACTORS))
    beta_per_factor = np.diag(beta_confusion) / np.maximum(beta_confusion.sum(axis=1), 1)

    # 2) FactorVAE score on the same fixed-factor design.
    factor_vae, factor_vae_per_factor, fv_votes, fv_mapping, fv_pred = compute_factor_vae_score(
        train_winners, train_y, test_winners, test_y, random_train_codes.shape[1]
    )

    # 3) MIG + Modularity share the same MI matrix.
    mi_matrix = compute_mi_matrix(random_train_codes, random_train_y, MI_BINS)
    mig, mig_per_factor, modularity, modularity_per_code, factor_entropy = compute_mig_and_modularity(
        mi_matrix, random_train_y
    )

    # 4) SAP.
    sap, sap_per_factor, sap_matrix = compute_sap(
        random_train_codes, random_train_y, random_test_codes, random_test_y
    )

    # 5) DCI.
    dci = compute_dci(random_train_codes, random_train_y, random_test_codes, random_test_y)

    # 6) Explicitness. (Modularity was already computed from the MI matrix.)
    explicit_train, explicit_test, explicit_train_pf, explicit_test_pf = compute_explicitness(
        random_train_codes, random_train_y, random_test_codes, random_test_y
    )

    # 7) IRS on both generic splits pooled for a more stable worst-case estimate.
    pooled_codes = np.concatenate([random_train_codes, random_test_codes], axis=0)
    pooled_factors = np.concatenate([random_train_y, random_test_y], axis=0)
    irs, irs_per_factor, irs_matrix, irs_per_code, irs_normaliser = compute_irs(
        pooled_codes, pooled_factors
    )

    scalar_scores = {
        "beta_vae": float(beta_score),
        "factor_vae": float(factor_vae),
        "mig": float(mig),
        "sap": float(sap),
        "dci_disentanglement": float(dci["disentanglement"]),
        "dci_completeness": float(dci["completeness"]),
        "dci_informativeness": float(dci["informativeness_test"]),
        "modularity": float(modularity),
        "explicitness": float(explicit_test),
        "irs": float(irs),
    }

    print("Literature metrics (higher is better):")
    for metric_name, value in scalar_scores.items():
        print(f"  {metric_name:>22s}: {value:.4f}")
    print(f"  {'active dimensions':>22s}: {int(np.sum(active))}/{len(active)}")

    # Component analysis is especially useful for state_actions: it tells us
    # whether readable information is in z0, in the IDM actions, or only in
    # their combination. Reuse the already-computed full-code metrics.
    train_components = representation_components(model, random_train_codes)
    test_components = representation_components(model, random_test_codes)
    component_results.append({
        "name": name,
        "mode": mode,
        "component": "full",
        "mig": mig,
        "modularity": modularity,
        "explicitness": explicit_test,
    })
    for component in [c for c in train_components if c != "full"]:
        comp_mi = compute_mi_matrix(train_components[component], random_train_y, MI_BINS)
        comp_mig, _, comp_mod, _, _ = compute_mig_and_modularity(comp_mi, random_train_y)
        _, comp_exp, _, _ = compute_explicitness(
            train_components[component], random_train_y,
            test_components[component], random_test_y,
        )
        component_results.append({
            "name": name,
            "mode": mode,
            "component": component,
            "mig": comp_mig,
            "modularity": comp_mod,
            "explicitness": comp_exp,
        })

    archive[f"{name}__mode"] = np.asarray(mode)
    archive[f"{name}__run_dir"] = np.asarray(str(run_dir))
    archive[f"{name}__random_train_codes"] = random_train_codes.astype(np.float32)
    archive[f"{name}__random_test_codes"] = random_test_codes.astype(np.float32)
    archive[f"{name}__global_std"] = global_std.astype(np.float32)
    archive[f"{name}__active_dimensions"] = active.astype(np.bool_)

    archive[f"{name}__beta_train_features"] = train_beta_x.astype(np.float32)
    archive[f"{name}__beta_test_features"] = test_beta_x.astype(np.float32)
    archive[f"{name}__beta_train_accuracy"] = np.asarray(beta_train_acc, dtype=np.float32)
    archive[f"{name}__beta_accuracy"] = np.asarray(beta_score, dtype=np.float32)
    archive[f"{name}__beta_per_factor"] = beta_per_factor.astype(np.float32)
    archive[f"{name}__beta_confusion"] = beta_confusion.astype(np.int64)
    archive[f"{name}__beta_probe_coef"] = beta_w.astype(np.float32)
    archive[f"{name}__beta_probe_intercept"] = beta_b.astype(np.float32)
    archive[f"{name}__beta_probe_iteration"] = beta_history["iteration"].astype(np.int64)
    archive[f"{name}__beta_probe_train_loss"] = beta_history["train_loss"].astype(np.float32)
    archive[f"{name}__beta_probe_test_loss"] = beta_history["test_loss"].astype(np.float32)
    archive[f"{name}__beta_probe_converged"] = np.asarray(beta_opt.success)

    archive[f"{name}__factor_vae_accuracy"] = np.asarray(factor_vae, dtype=np.float32)
    archive[f"{name}__factor_vae_per_factor"] = factor_vae_per_factor.astype(np.float32)
    archive[f"{name}__factor_vae_votes"] = fv_votes.astype(np.int64)
    archive[f"{name}__factor_vae_mapping"] = fv_mapping.astype(np.int64)
    archive[f"{name}__factor_vae_test_prediction"] = fv_pred.astype(np.int64)

    archive[f"{name}__mi_matrix"] = mi_matrix.astype(np.float32)
    archive[f"{name}__mig"] = np.asarray(mig, dtype=np.float32)
    archive[f"{name}__mig_per_factor"] = mig_per_factor.astype(np.float32)
    archive[f"{name}__factor_entropy"] = factor_entropy.astype(np.float32)
    archive[f"{name}__modularity"] = np.asarray(modularity, dtype=np.float32)
    archive[f"{name}__modularity_per_code"] = modularity_per_code.astype(np.float32)

    archive[f"{name}__sap"] = np.asarray(sap, dtype=np.float32)
    archive[f"{name}__sap_per_factor"] = sap_per_factor.astype(np.float32)
    archive[f"{name}__sap_score_matrix"] = sap_matrix.astype(np.float32)

    archive[f"{name}__dci_disentanglement"] = np.asarray(dci["disentanglement"], dtype=np.float32)
    archive[f"{name}__dci_completeness"] = np.asarray(dci["completeness"], dtype=np.float32)
    archive[f"{name}__dci_informativeness_train"] = np.asarray(dci["informativeness_train"], dtype=np.float32)
    archive[f"{name}__dci_informativeness_test"] = np.asarray(dci["informativeness_test"], dtype=np.float32)
    archive[f"{name}__dci_per_factor_train_accuracy"] = dci["per_factor_train_accuracy"].astype(np.float32)
    archive[f"{name}__dci_per_factor_test_accuracy"] = dci["per_factor_test_accuracy"].astype(np.float32)
    archive[f"{name}__dci_importance_matrix"] = dci["importance_matrix"].astype(np.float32)
    archive[f"{name}__dci_disentanglement_per_code"] = dci["disentanglement_per_code"].astype(np.float32)
    archive[f"{name}__dci_completeness_per_factor"] = dci["completeness_per_factor"].astype(np.float32)

    archive[f"{name}__explicitness_train"] = np.asarray(explicit_train, dtype=np.float32)
    archive[f"{name}__explicitness_test"] = np.asarray(explicit_test, dtype=np.float32)
    archive[f"{name}__explicitness_per_factor_train"] = explicit_train_pf.astype(np.float32)
    archive[f"{name}__explicitness_per_factor_test"] = explicit_test_pf.astype(np.float32)

    archive[f"{name}__irs"] = np.asarray(irs, dtype=np.float32)
    archive[f"{name}__irs_per_factor"] = irs_per_factor.astype(np.float32)
    archive[f"{name}__irs_matrix"] = irs_matrix.astype(np.float32)
    archive[f"{name}__irs_per_code"] = irs_per_code.astype(np.float32)
    archive[f"{name}__irs_normaliser"] = irs_normaliser.astype(np.float32)

    results.append({
        "name": name,
        "mode": mode,
        "label": label,
        **scalar_scores,
        "beta_train_accuracy": float(beta_train_acc),
        "beta_per_factor": beta_per_factor,
        "factor_vae_per_factor": factor_vae_per_factor,
        "mig_per_factor": mig_per_factor,
        "sap_per_factor": sap_per_factor,
        "dci_per_factor": dci["per_factor_test_accuracy"],
        "explicitness_per_factor": explicit_test_pf,
        "irs_per_factor": irs_per_factor,
        "beta_history": beta_history,
    })


#%% Export organised archives and CSVs
metric_columns = [
    "beta_vae",
    "factor_vae",
    "mig",
    "sap",
    "dci_disentanglement",
    "dci_completeness",
    "dci_informativeness",
    "modularity",
    "explicitness",
    "irs",
]

metadata = {
    "description": "Multi-metric disentanglement evaluation for short binary-shape videos.",
    "video_repreentation": VIDEO_REPRESENTATION,
    "representation": (
        "z0 concatenated with IDM actions a0,a1" if VIDEO_REPRESENTATION == "state_actions"
        else "z0,z1,z2 concatenated"
    ),
    "factor_names": FACTOR_NAMES,
    "excluded_target_factors": ["x0", "y0"],
    "important_protocol_note": (
        "x0/y0 are fixed within Higgins/FactorVAE fixed-factor batches, but vary as nuisance factors "
        "in the generic labelled set used by MIG/SAP/DCI/Modularity/Explicitness/IRS."
    ),
    "metric_references": {
        "beta_vae": "Higgins et al., ICLR 2017",
        "factor_vae": "Kim & Mnih, ICML 2018",
        "mig": "Chen et al., NeurIPS 2018",
        "sap": "Kumar et al., ICLR 2018",
        "dci": "Eastwood & Williams, ICLR 2018",
        "modularity_explicitness": "Ridgeway & Mozer, NeurIPS 2018",
        "irs": "Suter et al., ICML 2019",
    },
    "pairs_per_fixed_feature": L,
    "fixed_train_features_per_factor": TRAIN_PER_FACTOR,
    "fixed_test_features_per_factor": TEST_PER_FACTOR,
    "generic_train_samples": RANDOM_TRAIN,
    "generic_test_samples": RANDOM_TEST,
    "mi_bins": MI_BINS,
    "dci_trees": DCI_TREES,
    "simulation": SIM_CONFIG,
    "runs": [str(p) for p in RUN_DIRS],
}
archive["metadata_json"] = np.asarray(json.dumps(metadata))
np.savez_compressed(ARTEFACT_DIR / "disentanglement_metrics.npz", **archive)

with open(ARTEFACT_DIR / "results.csv", "w") as f:
    f.write("method,mode," + ",".join(metric_columns) + "\n")
    for r in results:
        f.write(
            f"{r['label']},{r['mode']}," + ",".join(f"{r[m]:.8f}" for m in metric_columns) + "\n"
        )

with open(ARTEFACT_DIR / "per_factor.csv", "w") as f:
    f.write("method,mode,factor,beta_vae,factor_vae,mig,sap,dci_informativeness,explicitness,irs\n")
    for r in results:
        for k, factor_name in enumerate(FACTOR_NAMES):
            f.write(
                f"{r['label']},{r['mode']},{factor_name},"
                f"{r['beta_per_factor'][k]:.8f},{r['factor_vae_per_factor'][k]:.8f},"
                f"{r['mig_per_factor'][k]:.8f},{r['sap_per_factor'][k]:.8f},"
                f"{r['dci_per_factor'][k]:.8f},{r['explicitness_per_factor'][k]:.8f},"
                f"{r['irs_per_factor'][k]:.8f}\n"
            )

with open(ARTEFACT_DIR / "component_scores.csv", "w") as f:
    f.write("method,mode,component,mig,modularity,explicitness\n")
    for r in component_results:
        f.write(
            f"{pretty_mode(r['mode'])},{r['mode']},{r['component']},"
            f"{r['mig']:.8f},{r['modularity']:.8f},{r['explicitness']:.8f}\n"
        )

with open(ARTEFACT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)


#%% Publication-ready comparison plots
labels = [r["label"] for r in results]
x_models = np.arange(len(results))

# 1) Overall literature metrics in three readable groups.
metric_groups = [
    ("Fixed-factor", ["beta_vae", "factor_vae"]),
    ("Compactness / robustness", ["mig", "sap", "modularity", "irs"]),
    ("Predictive structure", ["dci_disentanglement", "dci_completeness", "dci_informativeness", "explicitness"]),
]
fig, axes = plt.subplots(1, 3, figsize=(20.0, 6.4), sharey=True)
for ax, (title, metrics) in zip(axes, metric_groups):
    width = 0.82 / len(metrics)
    for metric_idx, metric_name in enumerate(metrics):
        offset = (metric_idx - (len(metrics) - 1) / 2.0) * width
        ax.bar(x_models + offset, [r[metric_name] for r in results], width=width, label=metric_name.replace("_", " "))
    ax.set_xticks(x_models)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=12)
    style_axis(ax)
axes[0].set_ylabel("Score (higher is better)")
fig.suptitle("Complementary disentanglement metrics")
show_and_save(fig, PLOT_DIR / "scores_overview.png")

# 2) Model x metric rank heatmap. Raw scores have different chance/baseline
# levels (e.g. fixed-factor accuracy vs ROC-AUC), so ranking each metric is a
# more honest visual summary than pretending all [0,1] scales mean the same thing.
heat = np.asarray([[r[m] for m in metric_columns] for r in results], dtype=np.float64)
ranks = np.empty_like(heat)
for j in range(heat.shape[1]):
    order = np.argsort(-heat[:, j], kind="stable")
    ranks[order, j] = np.arange(1, len(results) + 1)

fig, ax = plt.subplots(figsize=(15.5, max(5.5, 1.3 * len(results))))
im = ax.imshow(ranks, aspect="auto", vmin=1, vmax=max(len(results), 2))
ax.set_yticks(np.arange(len(results)), labels)
ax.set_xticks(
    np.arange(len(metric_columns)),
    [m.replace("_", "\n") for m in metric_columns],
    rotation=35, ha="right",
)
for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        ax.text(
            j, i, f"#{int(ranks[i, j])}\n{heat[i, j]:.2f}",
            ha="center", va="center", fontsize=10,
        )
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
cbar.set_label("Rank (1 = best)")
ax.set_title("Metric agreement and disagreement")
show_and_save(fig, PLOT_DIR / "scores_rank_heatmap.png")

# 3) Factor-level view for the three most complementary summaries.
fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.3), sharey=True)
per_factor_specs = [
    ("MIG", "mig_per_factor"),
    ("DCI informativeness", "dci_per_factor"),
    ("IRS", "irs_per_factor"),
]
for ax, (title, key) in zip(axes, per_factor_specs):
    x = np.arange(N_FACTORS)
    width = 0.82 / max(len(results), 1)
    for model_idx, r in enumerate(results):
        offset = (model_idx - (len(results) - 1) / 2.0) * width
        ax.bar(x + offset, r[key], width=width, label=r["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(FACTOR_NAMES, rotation=25, ha="right")
    ax.set_ylim(0, 1.04)
    ax.set_title(title)
    style_axis(ax)
axes[0].set_ylabel("Per-factor score")
axes[-1].legend(frameon=False, fontsize=12)
fig.suptitle("Which physical factors are actually clean?")
show_and_save(fig, PLOT_DIR / "scores_per_factor.png")

# 4) state/actions decomposition: where does readable information live?
if component_results:
    components = []
    for r in component_results:
        if r["component"] not in components:
            components.append(r["component"])
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.0), sharey=True)
    for ax, metric_name in zip(axes, ("mig", "modularity", "explicitness")):
        x = np.arange(len(results))
        width = 0.82 / len(components)
        for comp_idx, component in enumerate(components):
            vals = []
            for result in results:
                match = [r for r in component_results if r["name"] == result["name"] and r["component"] == component]
                vals.append(match[0][metric_name] if match else np.nan)
            offset = (comp_idx - (len(components) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=component.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.04)
        ax.set_title(metric_name.title())
        style_axis(ax)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, fontsize=12)
    fig.suptitle("Representation components")
    show_and_save(fig, PLOT_DIR / "component_scores.png")

# 5) Historical beta-VAE probe optimisation, kept as a diagnostic rather than a gold standard.
fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), sharey=True)
for r in results:
    h = r["beta_history"]
    axes[0].plot(h["iteration"], h["train_loss"], label=r["label"])
    axes[1].plot(h["iteration"], h["test_loss"], label=r["label"])
for ax, title in zip(axes, ("Training", "Held-out")):
    ax.set_xlabel("Linear-probe optimisation iteration")
    ax.set_ylabel("Cross-entropy")
    ax.set_yscale("log")
    ax.set_title(title)
    style_axis(ax)
axes[1].legend(frameon=False)
fig.suptitle("Historical beta-VAE probe")
show_and_save(fig, PLOT_DIR / "beta_probe_losses.png")


#%% Example fixed-factor simulator pairs
fig, axes = plt.subplots(
    N_FACTORS,
    2 * int(SIM_CONFIG["num_frames"]),
    figsize=(12, max(9, 2.0 * N_FACTORS)),
    squeeze=False,
)
for factor_idx, factor_name in enumerate(FACTOR_NAMES):
    for t in range(int(SIM_CONFIG["num_frames"])):
        axes[factor_idx, t].imshow(example_videos_a[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[factor_idx, t + int(SIM_CONFIG["num_frames"])].imshow(
            example_videos_b[factor_idx, t, ..., 0], cmap="gray", vmin=0, vmax=1
        )
    for ax in axes[factor_idx]:
        ax.set_axis_off()
    axes[factor_idx, 0].set_ylabel(factor_name, fontsize=18, fontweight="bold")
fig.suptitle("Example fixed-factor video pairs")
show_and_save(fig, PLOT_DIR / "fixed_factor_examples.png")


print(f"\nSaved multi-metric disentanglement experiment to: {OUTPUT_DIR}")
print(f"Archive: {ARTEFACT_DIR / 'disentanglement_metrics.npz'}")
print(f"Scalar summary: {ARTEFACT_DIR / 'results.csv'}")
print(f"Per-factor summary: {ARTEFACT_DIR / 'per_factor.csv'}")
print(f"Plots: {PLOT_DIR}")
