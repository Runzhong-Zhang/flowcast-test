"""Plot per-layer expert probabilities directly from exported CSV matrices."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    labels = ("cross_modal_gated", "cmi_gated")
    matrices = {
        (layer, label): np.loadtxt(
            args.directory / f"{label}_layer_{layer}.csv", delimiter=",", skiprows=1
        )
        for layer in range(1, 5) for label in labels
    }
    for key, matrix in matrices.items():
        if matrix.shape != (12, 8) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid CSV matrix: {key}")
    ymax = np.ceil(max(m.max() for m in matrices.values()) * 20) / 20
    colors = plt.get_cmap("tab10").colors
    styles = ("-", "--", "-.", ":", "-", "--", "-.", ":")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 15})
    for layer in range(1, 5):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
        for ax, label in zip(axes, labels):
            for expert in range(8):
                ax.plot(
                    range(1, 13), matrices[layer, label][:, expert],
                    label=f"Expert {expert + 1}", color=colors[expert],
                    linestyle=styles[expert], marker=markers[expert],
                    linewidth=1.8, markersize=4,
                )
            ax.set(title=label, xlabel="Forecast lead (frame)",
                   xticks=range(1, 13), xlim=(0.7, 12.3), ylim=(0, ymax))
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Mean pre-top-2 probability")
        fig.suptitle(f"MoE layer {layer}: expert probability across forecast leads", fontsize=18)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                   ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.025))
        fig.text(0.5, 0.008, "Expert IDs are specific to each model and layer.",
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))
        output = args.directory / f"layer_{layer}_line_comparison.png"
        fig.savefig(output, dpi=200, facecolor="white")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
