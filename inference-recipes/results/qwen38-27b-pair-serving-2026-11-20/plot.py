# /// script
# dependencies = ["matplotlib>=3.8,<4"]
# ///
"""Plot the six measured concurrency levels from the archived trial records."""
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parent
CLIENTS = [1, 4, 8, 16, 32, 64]


def main():
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained")
    for deploy, color, name in [("tp2", "#176b66", "Tensor parallel"),
                                ("replicas", "#935329", "Two replicas")]:
        for budget, style, marker in [(3, "--", "o"), (9, "-", "s")]:
            source = ROOT / f"{budget}g" / deploy / "trials.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            for ax, metric, units in zip(axes, ("output_tokens_per_s", "p95_first_delta_ms"), (1, 1000)):
                cells = [[r[metric] / units for r in rows if r["concurrency"] == c] for c in CLIENTS]
                if any(len(cell) != 3 for cell in cells):
                    raise ValueError(f"expected three trials at each load in {source}")
                ax.plot(CLIENTS, [median(cell) for cell in cells], color=color, linestyle=style,
                        marker=marker, markersize=5, markerfacecolor=color if budget == 9 else "white",
                        linewidth=1.8, label=f"{name}, {budget} GiB")
    for ax, title in zip(axes, ("Aggregate output tokens/s", "p95 time to first token (seconds)")):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(CLIENTS))
        ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in CLIENTS]))
        ax.set_xlabel("Concurrent clients")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylim(bottom=0)
    axes[1].set_yscale("log")
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    axes[1].yaxis.set_major_locator(FixedLocator(ticks))
    axes[1].yaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    axes[1].set_ylim(0.08, 35)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    figure.savefig(ROOT / "serving.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
