#%% Paths used by every plotting cell
# This file is meant to sit next to wm_eval2.py.
# Each plotting cell below is self-contained: it imports what it needs, resolves
# the paths again, loads the source data again, and writes its own figure.
#
# Edit these values once when working interactively. If a plotting cell is run
# by itself in a fresh kernel, the same defaults are used automatically.

from pathlib import Path

wm_folders = [
    "260910-220611-weight_ab",
    "260910-220716-weight_joint",
    "260910-220755-standard_ab",
    "260910-220808-standard_joint",
]

disentanglement_folder = "260911-111431_dissentglement2"

# This is the multieval run used for the longer-horizon and retargeting figures
# in the manuscript. Change it here if you rerun wm_multieval.py.
multieval_folder = "260910-095913_multieval"

runs_dir = Path("runs")
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

print("Training runs:")
for folder in wm_folders:
    print(" ", runs_dir / folder)
print("Disentanglement:", runs_dir / disentanglement_folder)
print("Multieval:", runs_dir / multieval_folder)
print("Plots:", plots_dir.resolve())


#%% Figure 1a — Phase-2 optimisation over 2,500 epochs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_theme(style="white", context="talk")
plt.rcParams['savefig.facecolor'] = 'white'

WM_FOLDERS = globals().get("wm_folders", [
    "260910-220611-weight_ab",
    "260910-220716-weight_joint",
    "260910-220755-standard_ab",
    "260910-220808-standard_joint",
])
RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#0072B2",
    "weight_joint": "#D55E00",
    "standard_ab": "#009E73",
    "standard_joint": "#CC79A7",
}
MARKERS = {
    "weight_ab": "o",
    "weight_joint": "s",
    "standard_ab": "^",
    "standard_joint": "D",
}

def _mode_from_folder(name):
    for mode in MODE_ORDER:
        if name.endswith(mode) or f"-{mode}" in name:
            return mode
    raise ValueError(f"Could not infer model mode from {name!r}")

def _metric_file(folder):
    run_dir = Path(folder)
    if not run_dir.is_absolute():
        run_dir = RUNS_DIR / run_dir
    candidates = [
        run_dir / "artefacts" / "p2_metrics.npz",
        run_dir / "p2_metrics.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No Phase-2 diagnostics found for {run_dir}. "
        f"Expected {candidates[0]}."
    )

metric_paths = {_mode_from_folder(Path(f).name): _metric_file(f) for f in WM_FOLDERS}

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(10.5, 5.8))

for mode in MODE_ORDER:
    data = np.load(metric_paths[mode])
    epochs = np.asarray(data["epoch_epoch"])
    train = np.asarray(data["epoch_train_rollout_pixel"])
    eval_ = np.asarray(data["epoch_eval_rollout_pixel"])

    # Training: deliberately broad and faint.
    ax.plot(
        epochs, train,
        color=COLORS[mode],
        linewidth=4.0,
        alpha=0.24,
        solid_capstyle="round",
        zorder=1,
    )

    # Evaluation: crisp, dashed, and model-specific marker.
    ax.plot(
        epochs, eval_,
        color=COLORS[mode],
        linewidth=2.35,
        linestyle="--",
        alpha=1.0,
        marker=MARKERS[mode],
        markersize=5.2,
        markevery=max(1, len(epochs) // 10),
        markerfacecolor="white",
        markeredgewidth=1.35,
        zorder=3,
    )

ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Pixel MSE")
ax.set_xlim(float(epochs.min()), float(epochs.max()))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", which="major", linewidth=0.8, alpha=0.18)
ax.grid(axis="y", which="minor", linewidth=0.5, alpha=0.08)
ax.set_axisbelow(True)

method_handles = [
    Line2D(
        [0], [0],
        color=COLORS[m],
        marker=MARKERS[m],
        linestyle="-",
        linewidth=2.2,
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.35,
        label=LABELS[m],
    )
    for m in MODE_ORDER
]
style_handles = [
    Line2D([0], [0], color="0.25", linewidth=4.0, alpha=0.24, label="Train"),
    Line2D([0], [0], color="0.25", linewidth=2.35, linestyle="--", label="Eval"),
]

legend_models = ax.legend(
    handles=method_handles,
    loc="upper right",
    frameon=False,
    ncol=2,
    columnspacing=1.2,
    handlelength=2.2,
)
ax.add_artist(legend_models)
ax.legend(
    handles=style_handles,
    loc="lower left",
    frameon=False,
    ncol=2,
    handlelength=2.8,
)

fig.tight_layout()
path = PLOTS_DIR / "training_epoch_rollout_mse_2500.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Figure 1b — Longer-horizon rollout
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
MULTIEVAL_FOLDER = globals().get("multieval_folder", "260910-095913_multieval")
MULTIEVAL_DIR = Path(MULTIEVAL_FOLDER)
if not MULTIEVAL_DIR.is_absolute():
    MULTIEVAL_DIR = RUNS_DIR / MULTIEVAL_DIR

archive_path = MULTIEVAL_DIR / "artefacts" / "group2_reconstruction_rollout.npz"
if not archive_path.exists():
    raise FileNotFoundError(f"Missing {archive_path}")

archive = np.load(archive_path, allow_pickle=True)

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#0072B2",
    "weight_joint": "#D55E00",
    "standard_ab": "#009E73",
    "standard_joint": "#CC79A7",
}
MARKERS = {
    "weight_ab": "o",
    "weight_joint": "s",
    "standard_ab": "^",
    "standard_joint": "D",
}

prefix_by_mode = {}
for key in archive.files:
    if key.endswith("__mode"):
        prefix_by_mode[str(archive[key])] = key[:-6]

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# fig, ax = plt.subplots(figsize=(6.6, 5.2))
fig, ax = plt.subplots(figsize=(10.5, 5.8))
frames = np.arange(archive["ground_truth"].shape[1])

for mode in MODE_ORDER:
    values = np.asarray(
        archive[prefix_by_mode[mode] + "__rollout__mse"],
        dtype=float,
    )
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0) / np.sqrt(values.shape[0])

    ax.plot(
        frames,
        mean,
        color=COLORS[mode],
        label=LABELS[mode],
        linestyle="--" if mode.endswith("_joint") else "-",
        marker=MARKERS[mode],
        markersize=5.2,
        linewidth=2.4,
    )
    ax.fill_between(
        frames,
        mean - sem,
        mean + sem,
        color=COLORS[mode],
        alpha=0.10,
    )

ax.set_xlabel("Frame")
ax.set_ylabel("Pixel MSE")
ax.set_xlim(float(frames.min()), float(frames.max()))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linewidth=0.8, alpha=0.18)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
path = PLOTS_DIR / "figure1b_longer_horizon_rollout_v2.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Disentanglement overview — all complementary metrics
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
DIS_FOLDER = globals().get("disentanglement_folder", "260911-111431_dissentglement2")
DIS_DIR = Path(DIS_FOLDER)
if not DIS_DIR.is_absolute():
    DIS_DIR = RUNS_DIR / DIS_DIR

results_path = DIS_DIR / "artefacts" / "results.csv"
if not results_path.exists():
    raise FileNotFoundError(f"Missing {results_path}")
results = pd.read_csv(results_path)

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}

# Family hue + darker/hatched Joint formulation.
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

METRICS = [
    ("beta_vae", r"$\beta$-VAE"),
    ("factor_vae", "FactorVAE"),
    ("dci_disentanglement", "DCI-D"),
    ("dci_completeness", "DCI-C"),
    ("dci_informativeness", "DCI-I"),
    ("modularity", "Modularity"),
    ("explicitness", "Explicitness"),
    ("irs", "IRS"),
]

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(METRICS))
width = 0.19
fig, ax = plt.subplots(figsize=(13.2, 6.2))

for i, mode in enumerate(MODE_ORDER):
    row = results.loc[results["mode"] == mode].iloc[0]
    values = [float(row[column]) for column, _ in METRICS]
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width=width,
        label=LABELS[mode],
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
    )

ax.set_xticks(x)
ax.set_xticklabels([label for _, label in METRICS], rotation=24, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
path = PLOTS_DIR / "disentanglement_metrics_overview_v3.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Disentanglement gap metrics — MIG and SAP
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
DIS_FOLDER = globals().get("disentanglement_folder", "260911-111431_dissentglement2")
DIS_DIR = Path(DIS_FOLDER)
if not DIS_DIR.is_absolute():
    DIS_DIR = RUNS_DIR / DIS_DIR

results = pd.read_csv(DIS_DIR / "artefacts" / "results.csv")

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

metric_specs = [("mig", "MIG"), ("sap", "SAP")]
x = np.arange(len(metric_specs))
width = 0.19
fig, ax = plt.subplots(figsize=(7.8, 5.6))

for i, mode in enumerate(MODE_ORDER):
    row = results.loc[results["mode"] == mode].iloc[0]
    values = [float(row[column]) for column, _ in metric_specs]
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width=width,
        label=LABELS[mode],
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
    )

upper = 1.35 * max(float(results["mig"].max()), float(results["sap"].max()))
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in metric_specs])
ax.set_ylabel("Gap")
ax.set_ylim(0, upper)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
path = PLOTS_DIR / "disentanglement_gap_metrics_v3.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% DCI breakdown — disentanglement, completeness, informativeness
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
DIS_FOLDER = globals().get("disentanglement_folder", "260911-111431_dissentglement2")
DIS_DIR = Path(DIS_FOLDER)
if not DIS_DIR.is_absolute():
    DIS_DIR = RUNS_DIR / DIS_DIR

results = pd.read_csv(DIS_DIR / "artefacts" / "results.csv")

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

metric_specs = [
    ("dci_disentanglement", "Disentanglement"),
    ("dci_completeness", "Completeness"),
    ("dci_informativeness", "Informativeness"),
]
x = np.arange(len(metric_specs))
width = 0.19
fig, ax = plt.subplots(figsize=(9.6, 5.7))

for i, mode in enumerate(MODE_ORDER):
    row = results.loc[results["mode"] == mode].iloc[0]
    values = [float(row[column]) for column, _ in metric_specs]
    ax.bar(
        x + (i - 1.5) * width,
        values,
        width=width,
        label=LABELS[mode],
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
    )

ax.set_xticks(x)
ax.set_xticklabels([label for _, label in metric_specs])
ax.set_ylabel("DCI")
ax.set_ylim(0, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
path = PLOTS_DIR / "dci_breakdown_v3.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Component explicitness — state, actions, and full representation
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
DIS_FOLDER = globals().get("disentanglement_folder", "260911-111431_dissentglement2")
DIS_DIR = Path(DIS_FOLDER)
if not DIS_DIR.is_absolute():
    DIS_DIR = RUNS_DIR / DIS_DIR

components = pd.read_csv(DIS_DIR / "artefacts" / "component_scores.csv")

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

COMPONENTS = ["state_z0", "actions", "full"]
COMPONENT_LABELS = {
    "state_z0": r"Initial state $z_0$",
    "actions": r"Actions $u_{0:1}$",
    "full": "Full representation",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(COMPONENTS))
width = 0.19
fig, ax = plt.subplots(figsize=(9.8, 5.7))

for i, mode in enumerate(MODE_ORDER):
    values = []
    for component in COMPONENTS:
        row = components.loc[
            (components["mode"] == mode)
            & (components["component"] == component)
        ].iloc[0]
        values.append(float(row["explicitness"]))

    ax.bar(
        x + (i - 1.5) * width,
        values,
        width=width,
        label=LABELS[mode],
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
    )

ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.3)
ax.set_xticks(x)
ax.set_xticklabels([COMPONENT_LABELS[c] for c in COMPONENTS])
ax.set_ylabel("Explicitness")
ax.set_ylim(0.45, 1.02)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=2)

fig.tight_layout()
path = PLOTS_DIR / "component_explicitness_v3.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Retargeting distributional error — Wasserstein-1
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
MULTIEVAL_FOLDER = globals().get("multieval_folder", "260910-095913_multieval")
MULTIEVAL_DIR = Path(MULTIEVAL_FOLDER)
if not MULTIEVAL_DIR.is_absolute():
    MULTIEVAL_DIR = RUNS_DIR / MULTIEVAL_DIR

results = pd.read_csv(MULTIEVAL_DIR / "artefacts" / "results.csv")

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

def _metric(experiment, mode, metric):
    row = results.loc[
        (results["experiment"] == experiment)
        & (results["mode"] == mode)
        & (results["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {experiment}/{mode}/{metric}, got {len(row)}")
    return float(row["mean"].iloc[0])

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(MODE_ORDER))
width = 0.34
fig, ax = plt.subplots(figsize=(9.8, 5.6))

motion = [_metric("motion_swap", m, "wasserstein") for m in MODE_ORDER]
content = [_metric("content_retarget", m, "wasserstein") for m in MODE_ORDER]

for i, mode in enumerate(MODE_ORDER):
    ax.bar(
        x[i] - width / 2,
        motion[i],
        width=width,
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
        alpha=0.58,
    )
    ax.bar(
        x[i] + width / 2,
        content[i],
        width=width,
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
        alpha=1.0,
    )

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in MODE_ORDER], rotation=13, ha="right")
ax.set_ylabel(r"Wasserstein-1 $W_1$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Patch(facecolor="#777777", edgecolor="#333333", alpha=0.58, label="Motion swap"),
        Patch(facecolor="#777777", edgecolor="#333333", alpha=1.0, label="Content retargeting"),
        # Patch(facecolor="white", edgecolor="#333333", hatch="///", label="Joint FDM"),
    ],
    frameon=False,
    ncol=3,
)

fig.tight_layout()
path = PLOTS_DIR / "retargeting_wasserstein_v2.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Retargeting spatial overlap — IoU
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
MULTIEVAL_FOLDER = globals().get("multieval_folder", "260910-095913_multieval")
MULTIEVAL_DIR = Path(MULTIEVAL_FOLDER)
if not MULTIEVAL_DIR.is_absolute():
    MULTIEVAL_DIR = RUNS_DIR / MULTIEVAL_DIR

results = pd.read_csv(MULTIEVAL_DIR / "artefacts" / "results.csv")

MODE_ORDER = ["weight_ab", "weight_joint", "standard_ab", "standard_joint"]
LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}
COLORS = {
    "weight_ab": "#6BAED6",
    "weight_joint": "#2171B5",
    "standard_ab": "#FDBE85",
    "standard_joint": "#E6550D",
}
HATCHES = {
    "weight_ab": "",
    "weight_joint": "///",
    "standard_ab": "",
    "standard_joint": "///",
}

def _metric(experiment, mode, metric):
    row = results.loc[
        (results["experiment"] == experiment)
        & (results["mode"] == mode)
        & (results["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {experiment}/{mode}/{metric}, got {len(row)}")
    return float(row["mean"].iloc[0])

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(MODE_ORDER))
width = 0.34
fig, ax = plt.subplots(figsize=(9.8, 5.6))

motion = [_metric("motion_swap", m, "iou") for m in MODE_ORDER]
content = [_metric("content_retarget", m, "iou") for m in MODE_ORDER]

for i, mode in enumerate(MODE_ORDER):
    ax.bar(
        x[i] - width / 2,
        motion[i],
        width=width,
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
        alpha=0.58,
    )
    ax.bar(
        x[i] + width / 2,
        content[i],
        width=width,
        facecolor=COLORS[mode],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=HATCHES[mode],
        alpha=1.0,
    )

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in MODE_ORDER], rotation=13, ha="right")
ax.set_ylabel("IoU")
ax.set_ylim(0, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Patch(facecolor="#777777", edgecolor="#333333", alpha=0.58, label="Motion swap"),
        Patch(facecolor="#777777", edgecolor="#333333", alpha=1.0, label="Content retargeting"),
        # Patch(facecolor="white", edgecolor="#333333", hatch="///", label="Joint FDM"),
    ],
    frameon=False,
    ncol=3,
)

fig.tight_layout()
path = PLOTS_DIR / "retargeting_iou_v2.pdf"
fig.savefig(path, bbox_inches="tight")
plt.show()
plt.close(fig)
print(path)


#%% Qualitative motion-swapping figure
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from IPython.display import display

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
MULTIEVAL_FOLDER = globals().get("multieval_folder", "260910-095913_multieval")
MULTIEVAL_DIR = Path(MULTIEVAL_FOLDER)
if not MULTIEVAL_DIR.is_absolute():
    MULTIEVAL_DIR = RUNS_DIR / MULTIEVAL_DIR

archive = np.load(
    MULTIEVAL_DIR / "artefacts" / "group3_motion_swap.npz",
    allow_pickle=True,
)

MODE_LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}

prefix_by_mode = {}
for key in archive.files:
    if key.endswith("__mode"):
        prefix_by_mode[str(archive[key])] = key[:-6]

font_path = fm.findfont("DejaVu Sans")
font_big = ImageFont.truetype(font_path, 32)
font_med = ImageFont.truetype(font_path, 27)
font_small = ImageFont.truetype(font_path, 24)

def _tile(frame, size=150):
    image = np.asarray(frame, dtype=float)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3:
        image = image[..., 0]
    image = Image.fromarray((255 * image).astype(np.uint8), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)

example_index = 6
frame_ids = [0, 4, 5, 8]
rows = [
    ("Source", archive["source_videos"]),
    ("Alien", archive["alien_videos"]),
    ("Target", archive["target_videos"]),
    (MODE_LABELS["weight_ab"], archive[prefix_by_mode["weight_ab"] + "__swapped_examples"]),
    (MODE_LABELS["weight_joint"], archive[prefix_by_mode["weight_joint"] + "__swapped_examples"]),
    (MODE_LABELS["standard_ab"], archive[prefix_by_mode["standard_ab"] + "__swapped_examples"]),
    (MODE_LABELS["standard_joint"], archive[prefix_by_mode["standard_joint"] + "__swapped_examples"]),
]

tile = 150
gap = 16
label_width = 250
header_height = 90
bottom = 24
canvas_width = label_width + len(frame_ids) * tile + (len(frame_ids) - 1) * gap + 20
canvas_height = header_height + len(rows) * tile + (len(rows) - 1) * gap + bottom

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)
draw.text(
    (label_width, 8),
    "Motion retargeting: source content + alien motion",
    fill="black",
    font=font_big,
)

for column, frame_id in enumerate(frame_ids):
    x0 = label_width + column * (tile + gap)
    text = f"t={frame_id}"
    box = draw.textbbox((0, 0), text, font=font_med)
    draw.text(
        (x0 + (tile - (box[2] - box[0])) / 2, 53),
        text,
        fill="black",
        font=font_med,
    )

for row_index, (label, videos) in enumerate(rows):
    y0 = header_height + row_index * (tile + gap)
    box = draw.textbbox((0, 0), label, font=font_small)
    draw.text(
        (label_width - 18 - (box[2] - box[0]), y0 + (tile - (box[3] - box[1])) / 2),
        label,
        fill="black",
        font=font_small,
    )
    for column, frame_id in enumerate(frame_ids):
        x0 = label_width + column * (tile + gap)
        canvas.paste(_tile(videos[example_index, frame_id], tile), (x0, y0))

path = PLOTS_DIR / "motion_swap_examples.png"
canvas.save(path, dpi=(300, 300))
display(canvas)
print(path)


#%% Qualitative content-retargeting figure
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from IPython.display import display

RUNS_DIR = Path(globals().get("runs_dir", Path("runs")))
PLOTS_DIR = Path(globals().get("plots_dir", Path("plots")))
PLOTS_DIR.mkdir(exist_ok=True)
MULTIEVAL_FOLDER = globals().get("multieval_folder", "260910-095913_multieval")
MULTIEVAL_DIR = Path(MULTIEVAL_FOLDER)
if not MULTIEVAL_DIR.is_absolute():
    MULTIEVAL_DIR = RUNS_DIR / MULTIEVAL_DIR

archive = np.load(
    MULTIEVAL_DIR / "artefacts" / "group4_content_retarget.npz",
    allow_pickle=True,
)

MODE_LABELS = {
    "weight_ab": "WINR Additive",
    "weight_joint": "WINR Joint",
    "standard_ab": "Standard Additive",
    "standard_joint": "Standard Joint",
}

prefix_by_mode = {}
for key in archive.files:
    if key.endswith("__mode"):
        prefix_by_mode[str(archive[key])] = key[:-6]

font_path = fm.findfont("DejaVu Sans")
font_big = ImageFont.truetype(font_path, 32)
font_med = ImageFont.truetype(font_path, 27)
font_small = ImageFont.truetype(font_path, 24)

def _tile(frame, size=150):
    image = np.asarray(frame, dtype=float)
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3:
        image = image[..., 0]
    image = Image.fromarray((255 * image).astype(np.uint8), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)

example_index = 7
frame_ids = [0, 4, 5, 8]
rows = [
    ("Source", archive["source_videos"]),
    ("Alien / target", archive["alien_target_videos"]),
    (MODE_LABELS["weight_ab"], archive[prefix_by_mode["weight_ab"] + "__retargeted_examples"]),
    (MODE_LABELS["weight_joint"], archive[prefix_by_mode["weight_joint"] + "__retargeted_examples"]),
    (MODE_LABELS["standard_ab"], archive[prefix_by_mode["standard_ab"] + "__retargeted_examples"]),
    (MODE_LABELS["standard_joint"], archive[prefix_by_mode["standard_joint"] + "__retargeted_examples"]),
]

tile = 150
gap = 16
label_width = 250
header_height = 90
bottom = 24
canvas_width = label_width + len(frame_ids) * tile + (len(frame_ids) - 1) * gap + 20
canvas_height = header_height + len(rows) * tile + (len(rows) - 1) * gap + bottom

canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(canvas)
draw.text(
    (label_width, 8),
    "Content retargeting: alien content + native motion",
    fill="black",
    font=font_big,
)

for column, frame_id in enumerate(frame_ids):
    x0 = label_width + column * (tile + gap)
    text = f"t={frame_id}"
    box = draw.textbbox((0, 0), text, font=font_med)
    draw.text(
        (x0 + (tile - (box[2] - box[0])) / 2, 53),
        text,
        fill="black",
        font=font_med,
    )

for row_index, (label, videos) in enumerate(rows):
    y0 = header_height + row_index * (tile + gap)
    box = draw.textbbox((0, 0), label, font=font_small)
    draw.text(
        (label_width - 18 - (box[2] - box[0]), y0 + (tile - (box[3] - box[1])) / 2),
        label,
        fill="black",
        font=font_small,
    )
    for column, frame_id in enumerate(frame_ids):
        x0 = label_width + column * (tile + gap)
        canvas.paste(_tile(videos[example_index, frame_id], tile), (x0, y0))

path = PLOTS_DIR / "content_retarget_examples.png"
canvas.save(path, dpi=(300, 300))
display(canvas)
print(path)
