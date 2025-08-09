import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Tuple

ALLOWED_METRICS = [
    "rougeL", "rougeLsum", "rouge1", "rouge2", "meteor",
    "word_count_prediction", "word_count_reference",
]


def extract_scores_from_file(file_path: str, plot_metric: str) -> List[float]:
    scores: List[float] = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:
                continue
            data = json.loads(line)
            if plot_metric in {"rougeL", "rougeLsum", "rouge1", "rouge2"}:
                scores.append(data["rouge"][plot_metric])
            elif plot_metric == "meteor":
                scores.append(data["meteor"]["meteor"])
            elif plot_metric == "word_count_prediction":
                scores.append(data["word_count_prediction"])
            elif plot_metric == "word_count_reference":
                scores.append(data["word_count_reference"])
    return scores


def compute_summary(scores: List[float]) -> Tuple[float, float, float]:
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")
    avg_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    return avg_score, min_score, max_score


def main():
    parser = argparse.ArgumentParser()
    # Backward-compatible single-file arg
    parser.add_argument("--input_path", type=str, default=None,
                        help="Single evaluation results JSONL file")
    # New: multiple files support
    parser.add_argument("--input_paths", type=str, nargs="*", default=None,
                        help="Multiple evaluation results JSONL files to compare (max 4)")
    parser.add_argument("--labels", type=str, nargs="*", default=None,
                        help="Optional labels for inputs; must match number of input_paths")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plot_metric", type=str, required=True)
    parser.add_argument("--auto_bin", action="store_true", default=False)
    parser.add_argument("--bin_size", type=int, default=50)
    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = args.output_dir
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Validate metric
    if args.plot_metric not in ALLOWED_METRICS:
        print(f"Invalid plot metric: {args.plot_metric}")
        sys.exit(1)

    # Resolve inputs
    resolved_paths: List[str] = []
    if args.input_paths:
        resolved_paths = args.input_paths
    elif args.input_path:
        resolved_paths = [args.input_path]
    else:
        print("You must provide --input_path or --input_paths")
        sys.exit(1)

    if len(resolved_paths) > 4:
        print("Please provide at most 4 input files for comparison.")
        sys.exit(1)

    # Labels
    if args.labels and len(args.labels) != len(resolved_paths):
        print("Number of --labels must match number of input files. Ignoring labels.")
        labels = None
    else:
        labels = args.labels
    if not labels:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in resolved_paths]

    # Load all scores
    series_list: List[List[float]] = []
    for p in resolved_paths:
        series_list.append(extract_scores_from_file(p, args.plot_metric))

    print("Plotting the histograms...")

    # Compute shared bins
    all_scores = [s for series in series_list for s in series]
    if len(all_scores) == 0:
        print("No scores found in the provided files.")
        sys.exit(1)

    global_min = float(np.min(all_scores))
    global_max = float(np.max(all_scores))

    if args.auto_bin:
        bins = np.linspace(global_min, global_max, args.bin_size + 1)
    else:
        if args.plot_metric in ["word_count_prediction", "word_count_reference"]:
            step = max(1, int(round((global_max - global_min) / max(args.bin_size, 1))))
            bins = np.arange(int(global_min), int(global_max) + step, step)
        else:
            bins = np.linspace(0, 1, args.bin_size + 1)

    # Figure with sidebar for summary stats
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1])
    ax_main = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Colors for up to 4 series
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    colors = default_colors[: len(series_list)]

    # Plot overlay histograms
    for scores, label, color in zip(series_list, labels, colors):
        ax_main.hist(scores, bins=bins, alpha=0.55, color=color, edgecolor='black', label=label)

    ax_main.set_title(f'Distribution of {args.plot_metric} Scores', fontsize=14, fontweight='bold')
    ax_main.set_xlabel(f'{args.plot_metric} Score', fontsize=12)
    ax_main.set_ylabel('Frequency', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    # Sidebar: text-only stats per series
    ax_side.axis('off')
    y = 0.95
    line_height = 0.12 if len(series_list) <= 3 else 0.09
    ax_side.text(0.0, y, 'Summary', fontsize=12, fontweight='bold', transform=ax_side.transAxes)
    y -= line_height
    for scores, label, color in zip(series_list, labels, colors):
        avg_score, min_score, max_score = compute_summary(scores)
        ax_side.text(0.0, y, f'{label}', color=color, fontsize=11, fontweight='bold', transform=ax_side.transAxes)
        y -= line_height * 0.8
        ax_side.text(0.02, y, f'avg: {avg_score:.4f}', color=color, fontsize=10, transform=ax_side.transAxes)
        y -= line_height * 0.8
        ax_side.text(0.02, y, f'min: {min_score:.4f}', color=color, fontsize=10, transform=ax_side.transAxes)
        y -= line_height * 0.8
        ax_side.text(0.02, y, f'max: {max_score:.4f}', color=color, fontsize=10, transform=ax_side.transAxes)
        y -= line_height

    # Output path
    suffix = f"{args.plot_metric}_distribution"
    if len(series_list) > 1:
        suffix += "_overlay"
    output_path = os.path.join(dirpath, f"{suffix}.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    for scores, label in zip(series_list, labels):
        avg_score, min_score, max_score = compute_summary(scores)
        print(f"[{label}] avg {args.plot_metric}: {avg_score:.4f}")
        print(f"[{label}] min {args.plot_metric}: {min_score:.4f}")
        print(f"[{label}] max {args.plot_metric}: {max_score:.4f}")


if __name__ == "__main__":
    main()