"""Plot forward / gradient timings collected by the benchmark scripts.

Each benchmark writes a JSON file with the following shape:

    {
      "entries": [
        {"label": "<framework>", "forward_us": <float>, "grad_us": <float>},
        ...
      ]
    }

This script concatenates the entries (in input-file order) and produces a
side-by-side bar chart. By default it reads the three result files produced
by ``run_all.sh`` in the current directory.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUTS = [
    "results_torch.json",
    "results_jax.json",
    "results_sofie_clad.json",
]


def load_entries(paths):
    entries = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        entries.extend(data["entries"])
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Result JSON files to plot, in display order.",
    )
    parser.add_argument(
        "--out",
        default="plot.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Forward vs Gradient Time per Sample (CPU) - MLP with ~70k parameters",
    )
    args = parser.parse_args()

    entries = load_entries(args.inputs)
    labels = [e["label"] for e in entries]
    forward_times = [e["forward_us"] for e in entries]
    grad_times = [e["grad_us"] for e in entries]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, forward_times, width, label="Forward", color="#4C72B0")
    rects2 = ax.bar(x + width / 2, grad_times, width, label="Gradient", color="#55A868")

    ax.set_ylabel("Time per sample [µs]")
    ax.set_title(args.title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Plot written to {args.out}")


if __name__ == "__main__":
    main()
