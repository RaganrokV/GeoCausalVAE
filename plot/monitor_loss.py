
from release_paths import INPUT_ROOT, output_path
# -*- coding: utf-8 -*-
#%%

import os
import pickle
import numpy as np
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
figures_dir = output_path('figures/')

os.makedirs(
    figures_dir,
    exist_ok=True
)


# =====================================================
# Choose vehicle
# =====================================================

# 可选: "Average", "Sedan", "Bus", "Logistic"
target_col = os.environ.get("BEEC_VEHICLE", "Logistic")


# =====================================================
# Load loss history
# =====================================================

loss_history_path = os.path.join(
    results_dir,
    f"loss_history_{target_col}.pkl"
)

if not os.path.exists(loss_history_path):

    raise FileNotFoundError(
        f"Cannot find loss history file: {loss_history_path}"
    )

with open(loss_history_path, "rb") as f:

    loss_history = pickle.load(f)

print("=" * 80)
print(f"Loaded loss history for: {target_col}")
print(f"Path: {loss_history_path}")
print("=" * 80)

print("Available keys:")
print(loss_history.keys())


# =====================================================
# Prepare epoch axis
# =====================================================

# 每 10 个 epoch 记录一次
n_points = len(
    loss_history['total_loss']
)

# Original training logs at epoch indices 0, 10, 20, ...
# Prefer explicit epoch metadata if supplied by another training version.
actual_epochs = np.asarray(loss_history.get('epochs', np.arange(n_points) * 10))
if len(actual_epochs) != n_points or n_points == 0:
    raise ValueError('Epoch metadata and nonempty loss history must have matching lengths.')


# =====================================================
# Colors
# =====================================================

colors = {
    'total_loss': '#e41a1c',
    'recon_loss': '#1f77b4',
    'kl_loss': '#2ca02c',
    'dag_loss': '#8c564b',
    'prior_dag_loss': '#d62728',
    'spatial_reg': '#9467bd',
    'lag_reg': '#ff7f0e',
    'lr': '#000000'
}


# =====================================================
# Plot loss and learning rate
# =====================================================

fig, ax1 = plt.subplots(
    figsize=(14, 6)
)

# -----------------------------------------------------
# Left y-axis: loss components
# -----------------------------------------------------

loss_labels = {
    'total_loss': 'Total loss', 'recon_loss': 'Reconstruction loss',
    'kl_loss': 'KL loss', 'dag_loss': 'DAG diagnostic (recorded)',
    'prior_dag_loss': 'Empirical-prior loss (recorded)',
    'spatial_reg': 'Spatial diagnostic (recorded)',
    'lag_reg': 'Lag diagnostic (recorded)'
}
for key, label in loss_labels.items():
    if key in loss_history:
        values = np.asarray(loss_history[key])
        if len(values) != n_points:
            raise ValueError(f'History length mismatch for {key}')
        ax1.plot(actual_epochs, values, label=label, color=colors[key])

ax1.set_xlabel(
    'Epoch',
    fontsize=20
)

ax1.set_ylabel(
    'Loss',
    fontsize=20,
    color='black'
)

ax1.tick_params(
    axis='y',
    labelcolor='black',
    labelsize=18
)


# -----------------------------------------------------
# Right y-axis: learning rate
# -----------------------------------------------------

ax2 = ax1.twinx()

if 'lr' in loss_history:

    ax2.plot(
        actual_epochs,
        loss_history['lr'],
        label='Learning Rate',
        color=colors['lr'],
        linestyle='--',
        linewidth=2.5,
        alpha=0.5,
        zorder=5
    )

ax2.set_ylabel(
    'Learning Rate',
    fontsize=20,
    color=colors['lr']
)

ax2.tick_params(
    axis='y',
    labelcolor=colors['lr'],
    labelsize=20
)


# =====================================================
# Grid and x-axis
# =====================================================

ax1.grid(
    True,
    which="major",
    ls="-",
    alpha=0.3
)

ax1.grid(
    True,
    which="minor",
    ls=":",
    alpha=0.2
)

max_epoch = actual_epochs[-1]

ax1.set_xticks(
    np.arange(
        0,
        max_epoch + 1,
        10000
    )
)

ax1.set_xticklabels(
    [
        f'{int(x / 1000)}k'
        for x in np.arange(
            0,
            max_epoch + 1,
            10000
        )
    ],
    fontsize=20
)


# =====================================================
# Legend
# =====================================================

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper right',
    fontsize=18,
    frameon=True,
    framealpha=0.9,
    edgecolor='black'
)


# =====================================================
# Save and show
# =====================================================

fig.tight_layout()

save_path = os.path.join(
    figures_dir,
    f"M2_loss_lr_{target_col}.svg"
)

plt.savefig(
    save_path,
    dpi=600,
    format='svg',
    bbox_inches='tight'
)

print(f"Saved loss figure to: {save_path}")

plt.show()
# %%
