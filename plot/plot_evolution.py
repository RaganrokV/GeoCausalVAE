from release_paths import INPUT_ROOT, output_path
# -*- coding: utf-8 -*-
#%%

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams

# =====================================================
# Plot style
# =====================================================

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'svg',
    'savefig.bbox': 'tight',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.alpha': 0.4
})

# =====================================================
# Paths
# =====================================================

results_dir = str(INPUT_ROOT / 'GeoCausalVAE_results/')
prior_dir = str(INPUT_ROOT / 'Results/')
figures_dir = output_path('figures/')

os.makedirs(
    figures_dir,
    exist_ok=True
)

# =====================================================
# Choose vehicle and threshold
# =====================================================

# 可选: "Average", "Sedan", "Bus", "Logistic"
target_col = os.environ.get("BEEC_VEHICLE", "Bus")

# Learned DAG 二值化阈值
learned_dag_threshold = 0.15

# 是否保存图片
save_figure = True

# =====================================================
# Variable names
# =====================================================

predictors = [
    'Building density',
    'POI diversity',
    'Land use diversity',
    'Charging station',
    'Bus stop',
    'Road density',
    'Road curvature',
    'Arterial ratio',
    'Connectivity',
    'Slope',
    'Elevation',
    'Population'
]

variable_names = predictors + [target_col]

n_vars = len(variable_names)

# =====================================================
# Load causal evolution
# =====================================================

causal_evolution_path = os.path.join(
    results_dir,
    f"causal_evolution_{target_col}.pkl"
)

if not os.path.exists(causal_evolution_path):

    raise FileNotFoundError(
        f"Cannot find causal evolution file: {causal_evolution_path}"
    )

with open(causal_evolution_path, "rb") as f:

    causal_evolution = pickle.load(f)

print("=" * 80)
print(f"Loaded causal evolution for: {target_col}")
print(f"File: {causal_evolution_path}")
print(f"Number of snapshots: {len(causal_evolution)}")
print(f"Last saved epoch: {causal_evolution[-1]['epoch']}")
print("=" * 80)

# =====================================================
# Load prior DAG and transpose
# =====================================================

prior_path = os.path.join(
    prior_dir,
    f"binary_table_{target_col}.csv"
)

if not os.path.exists(prior_path):

    raise FileNotFoundError(
        f"Cannot find prior DAG file: {prior_path}"
    )

prior_df = pd.read_csv(
    prior_path,
    index_col=0
)

# Important:
# saved binary table:
# row = dependent variable
# column = explanatory variable
#
# model convention:
# A[source, target]
#
# Therefore transpose the matrix.
prior_dag = prior_df.loc[variable_names, variable_names].to_numpy(dtype=float).T.copy()

np.fill_diagonal(
    prior_dag,
    0.0
)

print(f"Prior DAG shape after transpose: {prior_dag.shape}")
print(f"Prior edge count: {int(prior_dag.sum())}")

# =====================================================
# Helper functions
# =====================================================

def get_matrix_by_epoch(
    causal_evolution,
    target_epoch
):
    """
    Return raw_matrix for the exact or closest saved epoch.
    """

    epochs = [
        item["epoch"]
        for item in causal_evolution
    ]

    if target_epoch in epochs:

        idx = epochs.index(
            target_epoch
        )

        return causal_evolution[idx]["raw_matrix"], epochs[idx]

    else:

        idx = int(
            np.argmin(
                np.abs(
                    np.array(epochs)
                    -
                    target_epoch
                )
            )
        )

        print(
            f"Warning: epoch {target_epoch} not found. "
            f"Using closest epoch {epochs[idx]}."
        )

        return causal_evolution[idx]["raw_matrix"], epochs[idx]


def binarize_matrix(
    matrix,
    threshold
):
    """
    Binarize learned DAG using absolute edge weights.
    """

    binary_matrix = (
        np.abs(matrix)
        >
        threshold
    ).astype(float)

    np.fill_diagonal(
        binary_matrix,
        0.0
    )

    return binary_matrix


# =====================================================
# Extract matrices
# =====================================================

initial_matrix = causal_evolution[0]["raw_matrix"]
initial_epoch = causal_evolution[0]["epoch"]

epoch_20k_matrix, epoch_20k_used = get_matrix_by_epoch(
    causal_evolution,
    20000
)

epoch_40k_matrix, epoch_40k_used = get_matrix_by_epoch(
    causal_evolution,
    40000
)

final_matrix = causal_evolution[-1]["raw_matrix"]
final_epoch = causal_evolution[-1]["epoch"]

binarized_matrix = binarize_matrix(
    final_matrix,
    learned_dag_threshold
)

print(f"Initial epoch: {initial_epoch}")
print(f"Epoch used for 20k panel: {epoch_20k_used}")
print(f"Epoch used for 40k panel: {epoch_40k_used}")
print(f"Final epoch: {final_epoch}")
print(f"Learned DAG threshold: {learned_dag_threshold}")
print(f"Final binary edge count: {int(binarized_matrix.sum())}")

# =====================================================
# Matrices and titles
# =====================================================

causal_matrices = [
    initial_matrix,
    epoch_20k_matrix,
    epoch_40k_matrix,
    final_matrix,
    binarized_matrix,
    prior_dag
]

titles = [
    "(a) Initial graph",
    f"(b) Epoch {epoch_20k_used:,}",
    f"(c) Epoch {epoch_40k_used:,}",
    "(d) Final graph",
    f"(e) Thresholded graph",
    "(f) Empirical prior"
]

# =====================================================
# Plot
# =====================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(22, 14)
)

axes = axes.flatten()

base_size = 1800 / (
    n_vars
    *
    n_vars
)

# Use absolute value so negative learned edges can also be visualized
all_max = max(
    [
        np.max(
            np.abs(mat)
        )
        for mat in causal_matrices
    ]
    +
    [1.0]
)

for i, (ax, matrix, title) in enumerate(
    zip(
        axes,
        causal_matrices,
        titles
    )
):

    x, y = np.meshgrid(
        range(n_vars),
        range(n_vars)
    )

    plot_values = np.abs(
        matrix
    )

    sizes = (
        plot_values.flatten()
        *
        base_size
        *
        (100 / all_max)
    )

    # Prior DAG uses orange, learned matrices use blue
    if i == 5:

        color = "#ff7f0e"

    else:

        color = "#1f77b4"

    ax.grid(
        True,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        zorder=1
    )

    ax.scatter(
        x,
        y,
        s=sizes,
        c=color,
        marker="o",
        alpha=0.7,
        zorder=2
    )

    ax.set_title(
        title,
        fontsize=23,
        pad=15
    )

    ax.set_xlim(
        -0.5,
        n_vars - 0.5
    )

    ax.set_ylim(
        -0.5,
        n_vars - 0.5
    )

    ax.invert_yaxis()

    ax.set_xticks(
        np.arange(n_vars)
    )

    ax.set_yticks(
        np.arange(n_vars)
    )

    ax.set_xticklabels(variable_names, rotation=90, fontsize=8)
    ax.set_yticklabels(variable_names, fontsize=8)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')

    ax.tick_params(
        length=0
    )

    # Diagonal cross: no self-loop
    for j in range(n_vars):

        ax.text(
            j,
            j,
            "×",
            fontsize=25,
            ha="center",
            va="center",
            color="red",
            alpha=0.7,
            fontweight="bold",
            zorder=3
        )

plt.subplots_adjust(
    wspace=0.48,
    hspace=0.75,
    top=0.92,
    bottom=0.15
)

# =====================================================
# Save and show
# =====================================================

fig.suptitle('Circle area represents absolute edge weight; thresholding does not guarantee a DAG.', fontsize=12)
if save_figure:

    save_path = os.path.join(
        figures_dir,
        f"M2_dag_evolution_{target_col}_thr{learned_dag_threshold}.svg"
    )

    plt.savefig(
        save_path,
        dpi=600,
        format="svg",
        bbox_inches="tight"
    )

    print(f"Saved DAG evolution figure to: {save_path}")

plt.show()