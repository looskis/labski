"""Plot the three four-client trials for each measured Qwen DFlash2 configuration."""

import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

HERE = Path(__file__).resolve().parent
CONFIGS = [
    ("../0", "Ordinary · 3 GiB"),
    ("../3", "3 proposals · 3 GiB"),
    ("../7", "7 proposals · 3 GiB"),
    ("7-kv3-repeat", "7 proposals · 3 GiB repeat"),
    ("7-kv5376m", "7 proposals · 5.25 GiB"),
    ("7-kv5632m", "7 proposals · 5.5 GiB"),
    ("7-kv9", "7 proposals · 9 GiB"),
    ("7-dynamic73", "7 or 3 by batch size · 3 GiB"),
]


def main():
    panels = []
    for folder, label in CONFIGS:
        p = HERE / folder / "trials.jsonl"
        if p.exists():
            trials = [r for r in map(json.loads, p.read_text().splitlines()) if r["concurrency"] == 4]
            if len(trials) != 3:
                raise ValueError(f"expected three four-client trials in {p}")
            panels.append((label, trials))

    height = 2.2 + 0.62 * len(panels)
    fig, axes = plt.subplots(1, 2, figsize=(10, height), sharey=True)
    fig.subplots_adjust(left=.24, right=.96, bottom=.19, top=1 - 1.4 / height, wspace=.20)
    fig.text(.025, 1 - .32 / height, "Qwen3.8 27B on one DGX Spark", weight="bold", fontsize=18)
    fig.text(.025, 1 - .66 / height, "Four simultaneous clients, 96 tokens per request. Lines show the three-trial range. Dots mark medians.",
             color="#425466", fontsize=10)
    for ax, field, label, color in zip(
        axes, ("output_tokens_per_s", "median_first_delta_ms"),
        ("Aggregate output (tokens/s)", "First streamed delta (ms, log scale)"),
        ("#08745e", "#98510b"),
    ):
        ax.set_title(label, fontsize=11, loc="left", pad=12, weight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#a9b4bc")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", color="#a9b4bc", labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e0e6ea", lw=.8)
        for index, (_, trials) in enumerate(panels):
            values = [t[field] for t in trials]
            ax.hlines(index, min(values), max(values), linewidth=3, color=color, alpha=.65)
            ax.scatter(median(values), index, s=42, color=color, zorder=3)
        if field == "median_first_delta_ms":
            ax.set_xscale("log")
            ax.set_xlim(160, 3000)
            ax.xaxis.set_major_locator(FixedLocator([200, 300, 500, 1000, 2000]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            ax.minorticks_off()
        else:
            ax.set_xlim(0, max(t[field] for _, trials in panels for t in trials) * 1.1)
    axes[0].set_yticks(range(len(panels)), [p[0] for p in panels], fontsize=10)
    axes[0].set_ylim(len(panels)-.5, -.5)
    fig.savefig(HERE / "four-clients.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
