#!/usr/bin/env python3
"""
Build qualitative + accuracy figures for DP object-erasure reports.

Reuses ResNet-50 evaluation from compare_accuracy.py.
Does not regenerate diffusion images — only classifies existing folders.

Outputs (under --out_dir):
  qualitative_grid.png
  acc_by_class.png
  acc_drop_by_class.png
  target_pred_hist_before_after.png
  comparison_table.csv
  summary.txt
  predictions_original.csv
  predictions_erased.csv
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from compare_accuracy import (
    ErasureAccuracyEvaluator,
    print_terminal_summary,
)


def find_image(folder: str, *name_candidates: str) -> str | None:
    for name in name_candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def make_qualitative_grid(original_path: str, erased_path: str, out_path: str) -> list[str]:
    """3 rows x 2 cols: cassette x2 + church; left original, right erased."""
    pairs = [
        (
            "cassette player (case 000)",
            find_image(original_path, "000_cassette_player_case000_var00.png"),
            find_image(erased_path, "000_cassette_player_case000_var00.png"),
        ),
        (
            "cassette player (case 001)",
            find_image(original_path, "001_cassette_player_case001_var00.png"),
            find_image(erased_path, "001_cassette_player_case001_var00.png"),
        ),
        (
            "church (non-target, case 020)",
            find_image(original_path, "020_church_case020_var00.png"),
            find_image(erased_path, "020_church_case020_var00.png"),
        ),
    ]

    missing = []
    for title, o, e in pairs:
        if o is None or e is None:
            missing.append(title)

    if missing:
        raise FileNotFoundError(
            "Missing qualitative pair images for: " + ", ".join(missing)
        )

    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    col_titles = ["Original SD 1.4", "After DP erase"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight="bold")

    used = []
    for r, (title, o_path, e_path) in enumerate(pairs):
        for c, path in enumerate([o_path, e_path]):
            img = Image.open(path).convert("RGB")
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            used.append(path)
        axes[r, 0].set_ylabel(title, fontsize=9, rotation=90, labelpad=8)

    fig.suptitle(
        "DP Cassette Player — qualitative original vs erased",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.03, 0.01, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return used


def plot_acc_by_class(comparison: pd.DataFrame, target: str, out_path: str) -> None:
    df = comparison.copy()
    # Keep target first (already ordered by compare_accuracy)
    classes = df["class"].tolist()
    x = range(len(classes))
    w = 0.38
    orig = df["accuracy_semantic_original"].values * 100
    eras = df["accuracy_semantic_erased"].values * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar([i - w / 2 for i in x], orig, w, label="Original", color="#4C72B0")
    bars2 = ax.bar([i + w / 2 for i in x], eras, w, label="After DP", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Semantic Top-1 Acc (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-class semantic accuracy: original vs DP-erased")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)
    # Highlight target tick
    for i, c in enumerate(classes):
        if c == target:
            ax.get_xticklabels()[i].set_fontweight("bold")
            ax.get_xticklabels()[i].set_color("#C44E52")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acc_drop(comparison: pd.DataFrame, target: str, out_path: str) -> None:
    df = comparison.copy()
    classes = df["class"].tolist()
    drops = df["accuracy_drop (%)"].values
    colors = ["#C44E52" if c == target else "#55A868" for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(classes)), drops, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Semantic Acc drop (pp)")
    ax.set_title("Accuracy drop by class (red = erased target; green = others)")
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_pred_hist(
    original_df: pd.DataFrame,
    erased_df: pd.DataFrame,
    target: str,
    out_path: str,
    top_n: int = 8,
) -> None:
    """Top-1 predicted labels for target-class prompts only, before vs after."""
    o = original_df[original_df["class"].str.lower() == target.lower()]
    e = erased_df[erased_df["class"].str.lower() == target.lower()]
    if o.empty or e.empty:
        raise ValueError(f"No rows for target concept '{target}' in predictions")

    def top_counts(series: pd.Series) -> pd.Series:
        c = Counter([str(x).lower() for x in series.tolist()])
        most = c.most_common(top_n)
        return pd.Series({k: v for k, v in most})

    co = top_counts(o["category_top1"])
    ce = top_counts(e["category_top1"])
    all_labels = sorted(
        set(co.index) | set(ce.index),
        key=lambda L: -(co.get(L, 0) + ce.get(L, 0)),
    )[:top_n]

    vo = [co.get(L, 0) for L in all_labels]
    ve = [ce.get(L, 0) for L in all_labels]
    x = range(len(all_labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - w / 2 for i in x], vo, w, label="Original", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], ve, w, label="After DP", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Count (target prompts)")
    ax.set_title(
        f'Top-1 ResNet predictions on "{target}" prompts (before vs after)'
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(comparison: pd.DataFrame, target: str, out_path: str) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 80)
    if target in comparison["class"].values:
        t = comparison[comparison["class"] == target].iloc[0]
        lines.append(f"\nTarget Concept ({target}):")
        lines.append(
            f"  • Original Accuracy (semantic): {t['accuracy_semantic_original']:.1%}"
        )
        lines.append(
            f"  • Erased Accuracy (semantic):   {t['accuracy_semantic_erased']:.1%}"
        )
        lines.append(f"  • Accuracy Drop:                {t['accuracy_drop (%)']:.1f}%")
        lines.append(
            f"  • Original Accuracy (strict):   {t['accuracy_strict_original']:.1%}"
        )
        lines.append(
            f"  • Erased Accuracy (strict):     {t['accuracy_strict_erased']:.1%}"
        )
        other = comparison[comparison["class"] != target]
        if not other.empty:
            lines.append("\nOther Classes (Average):")
            lines.append(
                f"  • Original Accuracy: {other['accuracy_semantic_original'].mean():.1%}"
            )
            lines.append(
                f"  • Erased Accuracy:   {other['accuracy_semantic_erased'].mean():.1%}"
            )
            lines.append(
                f"  • Average Drop:      {other['accuracy_drop (%)'].mean():.1f}%"
            )
    lines.append("=" * 80)
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make DP evaluation figures from existing generated images"
    )
    parser.add_argument("--original_path", type=str, required=True)
    parser.add_argument("--erased_path", type=str, required=True)
    parser.add_argument("--prompts_csv", type=str, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="figures_cassette_player")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument(
        "--skip_classify",
        action="store_true",
        help="Only build qualitative grid (needs no GPU/ResNet)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Qualitative grid (no classifier)
    qual_path = os.path.join(args.out_dir, "qualitative_grid.png")
    print("Building qualitative grid...")
    used = make_qualitative_grid(args.original_path, args.erased_path, qual_path)
    print(f"  Saved {qual_path}")
    for p in used:
        print(f"    used: {p}")

    if args.skip_classify:
        print("skip_classify set — done.")
        return

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("CUDA not available; falling back to CPU")
        args.device = "cpu"

    print("Initializing ResNet-50 evaluator...")
    evaluator = ErasureAccuracyEvaluator(
        device=args.device, batch_size=args.batch_size
    )
    prompts_df = pd.read_csv(args.prompts_csv)

    print("Classifying original images...")
    original_results = evaluator.classify_images(args.original_path, prompts_df)
    print("Classifying erased images...")
    erased_results = evaluator.classify_images(args.erased_path, prompts_df)

    comparison = evaluator.compare_accuracy(
        original_results, erased_results, target_concept=args.concept
    )
    print_terminal_summary(comparison, args.concept)

    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    comparison.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    original_results.to_csv(
        os.path.join(args.out_dir, "predictions_original.csv"), index=False
    )
    erased_results.to_csv(
        os.path.join(args.out_dir, "predictions_erased.csv"), index=False
    )

    summary_path = os.path.join(args.out_dir, "summary.txt")
    write_summary(comparison, args.concept, summary_path)
    print(f"Saved {summary_path}")

    plot_acc_by_class(
        comparison, args.concept, os.path.join(args.out_dir, "acc_by_class.png")
    )
    plot_acc_drop(
        comparison, args.concept, os.path.join(args.out_dir, "acc_drop_by_class.png")
    )
    plot_target_pred_hist(
        original_results,
        erased_results,
        args.concept,
        os.path.join(args.out_dir, "target_pred_hist_before_after.png"),
    )
    print(f"Saved plots under {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
