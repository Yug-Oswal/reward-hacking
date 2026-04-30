"""
Plotting utilities for all experiment phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.hidden_states import get_probe_view


def plot_phase1_aurocs(layers, auc_data, savefig=""):
    """Line plot of AUROC vs layer for each probe view."""
    plt.figure(figsize=(10, 6))

    for view, style in zip(
        ["last", "mean", "max"],
        ["-o", "-s", "-^"],
    ):
        mean = auc_data[view]["mean"]
        std = auc_data[view]["std"]

        plt.plot(layers, mean, style, label=view)
        plt.fill_between(layers, mean - std, mean + std, alpha=0.2)

    plt.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Phase 1A — Linear probe AUROC vs layer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_probe_margins(layer_margins, savefig=""):
    """Violin plot of probe margin distributions across layers."""
    layers = sorted(layer_margins.keys())
    data = [layer_margins[l] for l in layers]

    plt.figure(figsize=(14, 6))
    plt.violinplot(data, positions=layers, showmeans=True, widths=0.8)

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Margin (signed distance to decision boundary)")
    plt.title("Probe Margin Distributions Across Layers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_phase1_bars(layers, auc_data, savefig=""):
    """Stacked bar charts of AUROC for each view."""
    views = ["last", "mean", "max"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    for ax, view in zip(axes, views):
        mean = auc_data[view]["mean"]
        std = auc_data[view]["std"]

        ax.bar(layers, mean, yerr=std, capsize=3)
        ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)

        ax.set_ylabel("AUROC")
        ax.set_title(f"{view} probe view")
        ax.set_ylim(0.45, 1.01)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Layer")

    fig.suptitle("Phase 1A — AUROC vs Layer (Bar Charts)", y=1.02)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_grouped_bars(layers, auc_data, savefig=""):
    """Grouped bar chart of AUROC for all views side by side."""
    views = ["last", "mean", "max"]
    width = 0.25

    x = np.arange(len(layers))

    plt.figure(figsize=(14, 4))

    for i, view in enumerate(views):
        plt.bar(
            x + i * width,
            auc_data[view]["mean"],
            width=width,
            yerr=auc_data[view]["std"],
            capsize=3,
            label=view,
        )

    plt.axhline(0.5, linestyle="--", linewidth=1, alpha=0.6)
    plt.xticks(x + width, layers)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.title("Phase 1A — AUROC vs Layer (Grouped Bars)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_pca_all_layers(
    hidden_data, view="last", max_layers=28, cols=7, savefig=""
):
    """PCA scatter plots for all layers in a grid."""
    layers = list(range(max_layers))
    rows = int(np.ceil(max_layers / cols))

    plt.figure(figsize=(cols * 3, rows * 3))

    for idx, layer in enumerate(layers):
        layer_hs = hidden_data["hidden_states"][layer]
        y = hidden_data["labels"]

        X = get_probe_view(layer_hs, view)

        pca = PCA(n_components=2)
        pts = pca.fit_transform(X)

        ax = plt.subplot(rows, cols, idx + 1)
        ax.scatter(pts[:, 0], pts[:, 1], c=y, cmap="bwr", s=8, alpha=0.8)
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(
        f"PCA of {view}-representation for all layers", fontsize=16
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def plot_phase1B_results(
    results, top_k=5, title="Phase 1B: Causal Layer Effectiveness"
):
    """Line plot of Phase 1B probe AUC with top-k highlights."""
    layers = sorted(results.keys())
    means = np.array([results[l]["mean_auc"] for l in layers])
    stds = np.array([results[l]["std_auc"] for l in layers])

    top_indices = np.argsort(means)[-top_k:]
    top_layers = [layers[i] for i in top_indices]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, means, marker="o", label="Mean AUC")
    plt.fill_between(layers, means - stds, means + stds, alpha=0.2)

    for l in top_layers:
        plt.scatter(l, results[l]["mean_auc"], s=100, zorder=3)

    plt.xlabel("Layer")
    plt.ylabel("Probe AUC (Base vs Steered)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axhline(0.5, linestyle="--")
    plt.tight_layout()
    plt.show()

    print(
        f"Top-{top_k} layers:",
        sorted(
            top_layers,
            key=lambda l: results[l]["mean_auc"],
            reverse=True,
        ),
    )
