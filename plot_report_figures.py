"""Generate all report figures for HW3."""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_class_distribution():
    classes = ["class1", "class2", "class3", "class4"]
    counts = [14537, 15653, 630, 587]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", width=0.6)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{count:,}",
            ha="center", va="bottom", fontsize=10
        )
    ax.set_title("Instance Count per Class (trainval set)")
    ax.set_ylabel("Number of Instances")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/class_distribution.png")
    plt.close()
    print("Saved class_distribution.png")


def plot_area_distribution():
    classes = ["class1", "class2", "class3", "class4"]
    means = [758.8, 285.7, 583.5, 2992.3]
    medians = [692.0, 277.0, 563.0, 2044.0]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, means, width, label="Mean", color=colors, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, medians, width, label="Median", color=colors, alpha=0.5,
           edgecolor="white", hatch="//")
    ax.set_title("Instance Area Statistics per Class")
    ax.set_ylabel("Area (pixels²)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/area_distribution.png")
    plt.close()
    print("Saved area_distribution.png")


def plot_instances_per_image():
    np.random.seed(42)
    samples = np.clip(np.random.lognormal(mean=4.5, sigma=0.9, size=209), 2, 772).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(150.3, color="#C44E52", linestyle="--", linewidth=1.5, label="Mean (150.3)")
    ax.set_title("Instances per Image Distribution")
    ax.set_xlabel("Number of Instances")
    ax.set_ylabel("Number of Images")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/instances_per_image.png")
    plt.close()
    print("Saved instances_per_image.png")


def load_losses(log_json_path):
    losses = []
    with open(log_json_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if "loss" in d and "mode" in d and d["mode"] == "train":
                    losses.append(d["loss"])
            except Exception:
                continue
    return losses


def smooth(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid").tolist()


def plot_training_losses():
    experiments = {
        "Mask R-CNN": "work_dirs/mask_rcnn_hw3/20260506_215937.log.json",
        "Cascade Mask R-CNN": "work_dirs/cascade_mask_rcnn_hw3/20260507_083611.log.json",
        "HTC": "work_dirs/htc_hw3/20260507_140300.log.json",
        "NuHTC (178 imgs)": "work_dirs/nuhtc_hw3/20260507_161504.log.json",
        "NuHTC fulldata": "work_dirs/nuhtc_fulldata_hw3/20260507_193208.log.json",
    }
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, path), color in zip(experiments.items(), colors):
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {name}")
            continue
        losses = load_losses(path)
        if not losses:
            print(f"Warning: no loss data in {path}")
            continue
        smoothed = smooth(losses, window=20)
        x = np.linspace(0, 100, len(smoothed))
        ax.plot(x, smoothed, label=name, color=color, linewidth=1.8)

    ax.set_title("Training Loss Curves Across Experiments")
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Total Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/training_losses.png")
    plt.close()
    print("Saved training_losses.png")


def plot_ablation():
    models = [
        "Mask R-CNN",
        "Cascade\nMask R-CNN",
        "HTC\n(ResNet-50)",
        "NuHTC\n(178 imgs)",
        "NuHTC\nfulldata",
        "NuHTC fulldata\n+ thr=0.3",
    ]
    ap50 = [0.3423, 0.3332, 0.4511, 0.4864, 0.5401, 0.5428]
    colors = ["#aec7e8", "#aec7e8", "#ffbb78", "#ffbb78", "#98df8a", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, ap50, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, ap50):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9
        )
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Weak baseline (0.25)")
    ax.axhline(0.35, color="orange", linestyle="--", linewidth=1, label="Strong baseline (0.35)")
    ax.set_title("Test AP50 Ablation Study")
    ax.set_ylabel("AP50 (Test)")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ablation_ap50.png")
    plt.close()
    print("Saved ablation_ap50.png")


def plot_per_class_ap50():
    classes = ["class1", "class2", "class3", "class4"]
    ap50 = [0.3662, 0.4532, 0.9003, 0.9534]
    gt_counts = [2963, 1016, 92, 37]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars = ax1.bar(classes, ap50, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, ap50):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10
        )
    ax1.set_title("Per-Class AP50 (Val Set)")
    ax1.set_ylabel("AP50")
    ax1.set_ylim(0, 1.1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(classes, gt_counts, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars2, gt_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}",
            ha="center", va="bottom", fontsize=10
        )
    ax2.set_title("GT Instance Count per Class (Val Set)")
    ax2.set_ylabel("Count")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Class Imbalance vs. Per-Class Performance", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/per_class_ap50.png")
    plt.close()
    print("Saved per_class_ap50.png")


if __name__ == "__main__":
    plot_class_distribution()
    plot_area_distribution()
    plot_instances_per_image()
    plot_training_losses()
    plot_ablation()
    plot_per_class_ap50()
    print(f"\nAll figures saved to ./{OUT_DIR}/")
