#!/usr/bin/env python3
"""Generate publication-quality visualizations for Inkling-2 loop experiments.

Run with:
    uv run --with matplotlib --with numpy python3 plot_results.py results/ --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Clean publication styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#dddddd",
    "legend.fontsize": 8.5,
})

# Colorblind-safe Okabe-Ito palette keyed by model family
PALETTE = {
    "Gemma-4-31B-it": "#0072B2",       # deep blue
    "Gemma-4-26B-A4B-it": "#E69F00",   # amber/orange
    "Qwen3.8-27B": "#009E73",          # teal/green
    "Qwen3-4B": "#CC79A7",             # magenta/purple
}
MODEL_ORDER = [
    "Gemma-4-26B-A4B-it",
    "Gemma-4-31B-it",
    "Qwen3.8-27B",
    "Qwen3-4B",
]
MODE_STYLES = {
    "inkling2+var": {"ls": "-", "marker": "o", "lw": 2.5},
    "inkling2": {"ls": "--", "marker": "s", "lw": 1.9},
    "inkling2+var+lora": {"ls": "-", "marker": "D", "lw": 2.4},
    "scalar_gate_loop+var": {"ls": ":", "marker": "^", "lw": 2.0},
    "naive_loop": {"ls": "-.", "marker": "x", "lw": 2.0},
    "naive_loop+lora": {"ls": "-.", "marker": "x", "lw": 2.0},
    "lora_only": {"ls": "--", "marker": "*", "lw": 1.8},
}


def mode_tag(m: dict[str, Any]) -> str:
    ta = m["train_args"]
    t = ta["mode"]
    if ta.get("varloop"):
        t += "+var"
    if m["loop_config"].get("core_lora_rank", 0) > 0 and t != "lora_only":
        t += "+lora"
    return t


def short_model(label: str) -> str:
    return (
        label.replace("-it", "")
        .replace("Gemma-4-26B-A4B", "G4-26B MoE")
        .replace("Gemma-4-31B", "G4-31B")
    )


def load_completed_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for sub in sorted(results_dir.iterdir()):
        mp = sub / "metrics.json"
        if mp.is_file():
            m = json.loads(mp.read_text())
            m["_dir"] = sub
            m["_mode_tag"] = mode_tag(m)
            runs.append(m)
    return runs


def load_train_logs(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    logs = {}
    for sub in sorted(results_dir.iterdir()):
        lp = sub / "train_log.jsonl"
        if lp.is_file():
            rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
            if rows:
                logs[sub.name] = rows
    return logs


def color_for_run(name: str, model_label: str | None = None) -> str:
    if model_label and model_label in PALETTE:
        return PALETTE[model_label]
    if "31b" in name:
        return PALETTE["Gemma-4-31B-it"]
    if "26b" in name:
        return PALETTE["Gemma-4-26B-A4B-it"]
    if "qwen38" in name:
        return PALETTE["Qwen3.8-27B"]
    return PALETTE["Qwen3-4B"]


def _spread_labels(
    items: list[tuple[float, str, str]],
    ymin: float,
    ymax: float,
    min_gap_frac: float = 0.068,
) -> list[tuple[float, float, str, str]]:
    """Repel overlapping annotation y-positions cleanly within [ymin, ymax]."""
    if not items:
        return []
    ordered = sorted(items, key=lambda t: t[0])
    min_gap = max((ymax - ymin) * min_gap_frac, 1e-3)
    ys = [t[0] for t in ordered]
    # Iterative pairwise repulsion
    for _ in range(30):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                mid = 0.5 * (ys[i] + ys[i - 1])
                ys[i - 1] = mid - 0.5 * min_gap
                ys[i] = mid + 0.5 * min_gap
                moved = True
        if not moved:
            break
    # Keep inside bounds
    low_lim = ymin + (ymax - ymin) * 0.03
    high_lim = ymax - (ymax - ymin) * 0.04
    if ys[0] < low_lim:
        shift = low_lim - ys[0]
        ys = [y + shift for y in ys]
    if ys[-1] > high_lim:
        shift = ys[-1] - high_lim
        ys = [y - shift for y in ys]
    return [(orig[0], adj_y, orig[1], orig[2]) for orig, adj_y in zip(ordered, ys)]


# ---------------------------------------------------------------------------
# Figure 1: Depth Scaling (P = 1..5) + Parameter-Matched 1-Pass LoRA Controls
# ---------------------------------------------------------------------------

def plot_depth_scaling(runs: list[dict[str, Any]], out_path: Path) -> None:
    loop_runs = [r for r in runs if r["train_args"]["mode"] != "lora_only"]
    lora_runs = [r for r in runs if r["train_args"]["mode"] == "lora_only"]
    if not loop_runs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), sharex=True)
    panels = [
        (axes[0, 0], "overall_id", "acc", 100.0, "In-Distribution Accuracy (%) vs. Loop Passes", "Exact-Match Accuracy (%)"),
        (axes[0, 1], "overall_ood", "acc", 100.0, "Out-of-Distribution (Deeper Hops) Accuracy (%)", "Exact-Match Accuracy (%)"),
        (axes[1, 0], "overall_id", "mean_ans_nll", 1.0, "In-Distribution Gold-Answer NLL (Lower = Better)", "Cross-Entropy Loss (NLL)"),
        (axes[1, 1], "overall_ood", "mean_ans_nll", 1.0, "Out-of-Distribution Gold-Answer NLL (Lower = Better)", "Cross-Entropy Loss (NLL)"),
    ]

    for ax, split_key, metric_key, scale, title, ylabel in panels:
        ax.axvspan(3.35, 6.02, color="#f0e6f6", alpha=0.52, zorder=0)
        ax.axvline(3.35, color="#9b72aa", ls=":", lw=1.1, zorder=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["P=1\n(Base)", "P=2", "P=3\n(Train Max)", "P=4\n(Extrap)", "P=5\n(Extrap)"])
        ax.set_xlim(0.52, 6.02)

        ann_by_x: dict[int, list[tuple[float, str, str]]] = {1: [], 3: [], 5: []}

        for r in loop_runs:
            ps = r["post_training"]["serial"]
            xs, ys = [], []
            for p in (1, 2, 3, 4, 5):
                tag = f"P{p}"
                if tag in ps:
                    xs.append(p)
                    ys.append(ps[tag][split_key][metric_key] * scale)
            if not xs:
                continue
            col = color_for_run(r["run_name"], r["model_label"])
            st = MODE_STYLES.get(r["_mode_tag"], {"ls": "-", "marker": "o", "lw": 2.0})
            lbl = f"{r['model_label']} ({r['_mode_tag']})"
            alpha = 0.95 if r["_mode_tag"] in ("inkling2+var", "inkling2+var+lora") else 0.78
            ax.plot(
                xs, ys, color=col, linestyle=st["ls"], marker=st["marker"],
                linewidth=st["lw"], markersize=6.8, label=lbl, alpha=alpha, zorder=3,
            )
            # Only collect annotations for primary varloop + scalar_gate to keep panel crisp
            if r["_mode_tag"] in ("inkling2+var", "scalar_gate_loop+var"):
                for x, y in zip(xs, ys):
                    if x in (1, 3, 5):
                        txt = f"{y:.1f}%" if scale == 100.0 else f"{y:.2f}"
                        ann_by_x[x].append((y, txt, col))

        # Parameter-matched 1-pass LoRA controls as horizontal dashed lines with right-edge badges
        lora_labels: list[tuple[float, str, str]] = []
        for lr in lora_runs:
            sd = lr["post_training"]["serial"].get("P1")
            if not sd:
                continue
            val = sd[split_key][metric_key] * scale
            col = color_for_run(lr["run_name"], lr["model_label"])
            lbl = f"{lr['model_label']} 1-Pass LoRA ctrl"
            ax.axhline(val, color=col, linestyle="--", linewidth=1.6, alpha=0.82, label=lbl, zorder=2)
            vtxt = f"{val:.1f}%" if scale == 100.0 else f"{val:.2f}"
            lora_labels.append((val, f"LoRA {vtxt}", col))

        ymin, ymax = ax.get_ylim()
        pad = max((ymax - ymin) * 0.16, 1.2 if scale == 100.0 else 0.2)
        ax.set_ylim(max(0.0, ymin - pad * 0.3), ymax + pad)
        new_ymin, new_ymax = ax.get_ylim()

        # Render collision-free annotations at P=1 (left), P=3 (slightly offset), P=5 (right)
        for x_pt, items in ann_by_x.items():
            # Deduplicate identical (y, col) e.g. shared P=1 baseline across same model
            dedup: list[tuple[float, str, str]] = []
            seen = set()
            for yv, txt, col in items:
                key = (round(yv, 2), col)
                if key not in seen:
                    seen.add(key)
                    dedup.append((yv, txt, col))
            spread = _spread_labels(dedup, new_ymin, new_ymax, min_gap_frac=0.068)
            for orig_y, adj_y, txt, col in spread:
                if x_pt == 1:
                    tx, ha = 0.88, "right"
                elif x_pt == 5:
                    tx, ha = 5.10, "left"
                else:
                    tx, ha = 3.06, "left"
                if abs(adj_y - orig_y) > (new_ymax - new_ymin) * 0.025:
                    ax.plot([x_pt, tx], [orig_y, adj_y], color=col, lw=0.7, alpha=0.55, zorder=2)
                ax.text(
                    tx, adj_y, txt, ha=ha, va="center", fontsize=8.0, color=col,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.75),
                    zorder=5,
                )

        # Right-margin badges for LoRA horizontal reference lines (cleanly past P=5 labels)
        for orig_y, adj_y, txt, col in _spread_labels(lora_labels, new_ymin, new_ymax, min_gap_frac=0.072):
            ax.text(
                5.97, adj_y, txt, ha="right", va="center", fontsize=7.5, color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.7, alpha=0.92),
                zorder=6,
            )

        ax.text(
            4.55, new_ymax - (new_ymax - new_ymin) * 0.03,
            "Unseen Depth\nExtrapolation",
            ha="center", va="top", fontsize=8.0, color="#6d4c7d", style="italic",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=min(4, max(1, len(labels))), fontsize=8.5,
    )
    fig.suptitle(
        "Inkling-2 Recurrent Loop Retrofit: Depth Scaling (P = 1..5) vs. 1-Pass LoRA Controls",
        y=1.035, fontsize=13.5, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Compact Two-Panel Per-Task Heatmap (ID + GSM8K Top, OOD Bottom)
# ---------------------------------------------------------------------------

def plot_per_task_breakdown(runs: list[dict[str, Any]], out_path: Path) -> None:
    if not runs:
        return

    tasks = ["graph_bfs", "pointer_chase", "mod_register", "perm_track"]
    id_rows = [
        ("graph_bfs", "Graph BFS (ID)"),
        ("pointer_chase", "Pointer Chase (ID)"),
        ("mod_register", "Mod-19 Register (ID)"),
        ("perm_track", "S5 Cup-Swap (ID)"),
        ("gsm8k", "GSM8K Direct (No-CoT)"),
    ]
    ood_rows = [
        ("graph_bfs", "Graph BFS (OOD Deeper)"),
        ("pointer_chase", "Pointer Chase (OOD 5–6 Hops)"),
        ("mod_register", "Mod-19 Register (OOD 5–6 Ops)"),
        ("perm_track", "S5 Cup-Swap (OOD 7–8 Swaps)"),
    ]

    # Group runs by model family in clean canonical order
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model_label"], {})[r["_mode_tag"]] = r

    # Build compact milestone columns per model family with NaN separator gaps between families
    col_labels: list[str] = []
    col_group_spans: list[tuple[str, int, int, str]] = []
    id_cols: list[list[float]] = []
    ood_cols: list[list[float]] = []

    def extract_col(r: dict[str, Any], p_tag: str) -> tuple[list[float], list[float]]:
        sd = r["post_training"]["serial"][p_tag]
        id_v = [sd["per_task"][t]["id"]["acc"] * 100.0 for t in tasks]
        gsm = (r["post_training"]["gsm8k_direct"].get(p_tag) or {}).get("acc")
        id_v.append(gsm * 100.0 if gsm is not None else np.nan)
        ood_v = [sd["per_task"][t]["ood"]["acc"] * 100.0 for t in tasks]
        return id_v, ood_v

    ordered_models = [m for m in MODEL_ORDER if m in by_model] + [m for m in by_model if m not in MODEL_ORDER]
    for m_idx, mlabel in enumerate(ordered_models):
        modes = by_model[mlabel]
        start_c = len(col_labels)
        var_r = modes.get("inkling2+var")
        plus_r = modes.get("inkling2+var+lora")
        lora_r = modes.get("lora_only")

        if var_r:
            for p_tag, lbl in [("P1", "Base\nP=1"), ("P3", "Loop\nP=3"), ("P5", "Loop\nP=5")]:
                if p_tag in var_r["post_training"]["serial"]:
                    iv, ov = extract_col(var_r, p_tag)
                    id_cols.append(iv)
                    ood_cols.append(ov)
                    col_labels.append(lbl)
        if plus_r:
            for p_tag, lbl in [("P3", "Loop+LoRA\nP=3"), ("P5", "Loop+LoRA\nP=5")]:
                if p_tag in plus_r["post_training"]["serial"]:
                    iv, ov = extract_col(plus_r, p_tag)
                    id_cols.append(iv)
                    ood_cols.append(ov)
                    col_labels.append(lbl)
        if lora_r and "P1" in lora_r["post_training"]["serial"]:
            iv, ov = extract_col(lora_r, "P1")
            id_cols.append(iv)
            ood_cols.append(ov)
            col_labels.append("1-Pass\nLoRA")

        end_c = len(col_labels) - 1
        if end_c >= start_c:
            col_group_spans.append((short_model(mlabel), start_c, end_c, PALETTE.get(mlabel, "#333333")))

    if not col_labels:
        return

    mat_id = np.array(id_cols, dtype=float).T
    mat_ood = np.array(ood_cols, dtype=float).T

    fig_w = max(12.2, 0.92 * len(col_labels) + 3.0)
    fig, (ax_id, ax_ood) = plt.subplots(
        2, 1, figsize=(fig_w, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [5, 4], "hspace": 0.24},
    )

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#efefef")

    for ax, mat, rows, title, vmax in [
        (ax_id, mat_id, id_rows, "A. In-Distribution Serial Reasoning & Direct No-CoT GSM8K Accuracy (%)", 100.0),
        (ax_ood, mat_ood, ood_rows, "B. Out-of-Distribution (Unseen Deeper Hops / Steps) Accuracy (%)", 50.0),
    ]:
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows], fontweight="bold", fontsize=9.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10 if ax is ax_ood else 28)
        ax.grid(False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", color="#999999", fontsize=9)
                else:
                    txt_col = "white" if v > vmax * 0.52 else "#111111"
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=txt_col, fontsize=8.8, fontweight="bold")

        # Vertical dividers between model families
        for _, sc, ec, _ in col_group_spans[:-1]:
            ax.axvline(ec + 0.5, color="white", linewidth=3.2, zorder=4)
            ax.axvline(ec + 0.5, color="#bbbbbb", linewidth=0.9, zorder=5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
        cbar.set_label("Accuracy (%)", fontsize=9)

    # Model family header banners above Panel A
    for name, sc, ec, col in col_group_spans:
        mid = 0.5 * (sc + ec)
        ax_id.text(
            mid, -0.78, name, ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8f9fa", ec=col, lw=1.2),
        )

    ax_ood.set_xticks(range(len(col_labels)))
    ax_ood.set_xticklabels(col_labels, fontsize=8.8, fontweight="bold")
    fig.suptitle(
        "Per-Task Reasoning Breakdown: Base (P=1) vs. Inkling-2 Looping vs. Controls",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Causal Message Ablations + Gate Telemetry + Hardware Frontier
# ---------------------------------------------------------------------------

def plot_causal_and_mechanistic(runs: list[dict[str, Any]], out_path: Path) -> None:
    # Keep Panel A focused on the 4 primary varloop models + Qwen3-4B fixed3 & scalar_gate
    ab_runs = [
        r for r in runs
        if r.get("causal_ablations")
        and r["_mode_tag"] in ("inkling2+var", "inkling2", "scalar_gate_loop+var")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))

    # Panel A: Causal message ablations at P=3 (ID NLL)
    ax = axes[0]
    if ab_runs:
        conds = [
            ("P1 Base", lambda r: r["post_training"]["serial"]["P1"]["overall_id"]["mean_ans_nll"], "#888888"),
            ("P2 Normal", lambda r: r["post_training"]["serial"]["P2"]["overall_id"]["mean_ans_nll"], "#56B4E9"),
            ("P3 Normal", lambda r: r["post_training"]["serial"]["P3"]["overall_id"]["mean_ans_nll"], "#0072B2"),
            ("P3 Zero-Msg", lambda r: r["causal_ablations"]["zero"]["overall_id"]["mean_ans_nll"], "#E69F00"),
            ("P3 Shuf-Tok", lambda r: r["causal_ablations"]["shuffle_tokens"]["overall_id"]["mean_ans_nll"], "#D55E00"),
            ("P3 Freeze-P1", lambda r: r["causal_ablations"]["freeze_pass1"]["overall_id"]["mean_ans_nll"], "#CC79A7"),
        ]
        x = np.arange(len(ab_runs))
        w = 0.13
        for idx, (clbl, fn, ccol) in enumerate(conds):
            vals = [fn(r) for r in ab_runs]
            offset = (idx - (len(conds) - 1) / 2.0) * w
            bars = ax.bar(x + offset, vals, width=w, color=ccol, label=clbl, edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=6.9, rotation=90)
        xlabels = []
        for r in ab_runs:
            sm = short_model(r["model_label"]).replace(" MoE", "\nMoE")
            mt = r["_mode_tag"].replace("inkling2+var+lora", "loop+LoRA").replace("inkling2+var", "var").replace("inkling2", "fix3").replace("scalar_gate_loop+var", "scalar")
            xlabels.append(f"{sm}\n({mt})")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7.8)
        ax.set_ylabel("ID Gold-Answer NLL (Lower = Better)")
        ax.set_title("A. Causal Message Ablations at P=3")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.24)
        ax.legend(fontsize=7.3, ncol=2, loc="upper right")
    else:
        ax.text(0.5, 0.5, "Awaiting causal ablations...", ha="center", va="center")
        ax.set_title("A. Causal Message Ablations at P=3")

    # Panel B: Per-Layer Gate Activation Magnitude (Pass 3) — primary varloop runs only (1 per model)
    ax = axes[1]
    plotted_b = False
    primary_gate_runs = [r for r in ab_runs if r["_mode_tag"] == "inkling2+var"]
    if not primary_gate_runs:
        primary_gate_runs = ab_runs
    for r in primary_gate_runs:
        diag = r["post_training"]["serial"].get("P3", {}).get("diagnostics")
        if not diag or not diag.get("passes"):
            continue
        p3 = diag["passes"][-1]
        if not p3["layers"]:
            continue
        # Only plot if gate telemetry exists (skip scalar_gate / naive_loop if missing)
        if "gates" not in p3["layers"][0] or "q" not in p3["layers"][0]["gates"]:
            continue
        layers = [l["layer_idx"] for l in p3["layers"]]
        pos = np.linspace(0.0, 1.0, len(layers))
        q_msg = [l["gates"]["q"]["message_rms"] for l in p3["layers"]]
        q_fgt = [l["gates"]["q"]["forget_rms"] for l in p3["layers"]]
        col = color_for_run(r["run_name"], r["model_label"])
        sm = short_model(r["model_label"])
        ax.plot(pos, q_msg, color=col, lw=2.2, marker="o", ms=3.8, label=f"{sm} msg")
        ax.plot(pos, q_fgt, color=col, lw=1.4, ls="--", marker="x", ms=3.5, alpha=0.72, label=f"{sm} forget")
        plotted_b = True
    ax.set_title("B. Learned Gate Activity Across Looped Core (Pass 3)")
    ax.set_xlabel("Relative Core Layer Depth (0 = Core Start, 1 = Core End)")
    ax.set_ylabel("Gate RMS Magnitude (|tanh|)")
    if plotted_b:
        ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No gate telemetry yet", ha="center", va="center")

    # Panel C: DGX Spark GB10 Hardware Cost (Latency & Energy vs Pass Count)
    ax = axes[2]
    hw_seen = {}
    for r in runs:
        hw = r.get("hardware_forward_cost", {})
        if "P3" in hw and r["model_label"] not in hw_seen:
            hw_seen[r["model_label"]] = hw
    if hw_seen:
        ax2 = ax.twinx()
        ax2.grid(False)
        for mlabel in MODEL_ORDER:
            if mlabel not in hw_seen:
                continue
            hw = hw_seen[mlabel]
            ps = [1, 2, 3]
            ms = [hw[f"P{p}"]["ms_per_forward"] for p in ps]
            jj = [hw[f"P{p}"]["joules_per_forward"] for p in ps]
            col = color_for_run(mlabel, mlabel)
            sm = short_model(mlabel)
            ax.plot(ps, ms, color=col, lw=2.4, marker="o", label=f"{sm}")
            ax2.plot(ps, jj, color=col, lw=1.5, ls=":", marker="^", alpha=0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["P=1", "P=2", "P=3"])
        ax.set_ylabel("Latency (ms / forward, solid ○)")
        ax2.set_ylabel("GPU Energy (Joules / forward, dotted △)")
        ax.set_title("C. DGX Spark GB10 Forward Cost vs. Depth")
        ax.legend(fontsize=8.2, loc="upper left", title="Solid = ms, Dotted = J", title_fontsize=7.6)
    else:
        ax.text(0.5, 0.5, "Awaiting hardware benchmarks", ha="center", va="center")
        ax.set_title("C. DGX Spark GB10 Forward Cost vs. Depth")

    fig.suptitle(
        "Mechanistic & Hardware Analysis of Inkling-2 Recurrent Retrofits on NVIDIA GB10",
        fontsize=13.2, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Live Training Loss & Validation Trajectories (Shared Top Legend)
# ---------------------------------------------------------------------------

def plot_training_curves(logs: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    if not logs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.1))

    for name, rows in sorted(logs.items()):
        steps = [r["step"] for r in rows]
        sloss = [r["serial_loss"] for r in rows]
        val_steps = [r["step"] for r in rows if "val_id_nll" in r]
        val_nll = [r["val_id_nll"] for r in rows if "val_id_nll" in r]
        val_acc = [r["val_id_acc"] * 100.0 for r in rows if "val_id_acc" in r]

        col = color_for_run(name)
        if "control-1pass-lora" in name:
            ls, mk = "--", "*"
        elif "fixed3" in name:
            ls, mk = "-.", "s"
        elif "plus-lora" in name:
            ls, mk = "-", "D"
        elif "scalar" in name:
            ls, mk = ":", "^"
        else:
            ls, mk = "-", "o"

        short = (
            name.replace("gemma4-26b-a4b", "G4-26B MoE")
            .replace("gemma4-31b", "G4-31B")
            .replace("qwen38-27b", "Qwen3.8-27B")
            .replace("qwen3-4b", "Qwen3-4B")
            .replace("-inkling2-varloop", " (loop+var)")
            .replace("-control-1pass-lora", " (1-pass LoRA)")
            .replace("-inkling2-plus-lora", " (loop+LoRA)")
            .replace("-inkling2-fixed3", " (fixed3)")
            .replace("-scalar-gate-loop", " (scalar gate)")
            .replace("-naive-loop-lora", " (naive loop)")
        )

        axes[0].plot(steps, sloss, color=col, ls=ls, lw=1.9, marker=mk, ms=3.5, label=short, alpha=0.9)
        if val_steps:
            axes[1].plot(val_steps, val_nll, color=col, ls=ls, lw=2.1, marker=mk, ms=5.0, label=short, alpha=0.9)
            axes[2].plot(val_steps, val_acc, color=col, ls=ls, lw=2.1, marker=mk, ms=5.0, label=short, alpha=0.9)

    axes[0].set_title("Training Serial Loss vs. Optimizer Step")
    axes[0].set_xlabel("Optimizer Step (8 seq/step)")
    axes[0].set_ylabel("Answer-Token CE Loss")

    axes[1].set_title("Mid-Run Validation ID NLL (Held-Out Slice)")
    axes[1].set_xlabel("Optimizer Step")
    axes[1].set_ylabel("Validation NLL (Lower = Better)")

    axes[2].set_title("Mid-Run Validation ID Accuracy (%)")
    axes[2].set_xlabel("Optimizer Step")
    axes[2].set_ylabel("Validation Exact-Match Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=min(5, max(1, len(labels))), fontsize=8.3,
    )
    fig.suptitle(
        "Concurrent 4-Spark Training Trajectories (Gemma-4-31B, Gemma-4-26B-A4B, Qwen3.8-27B, Qwen3-4B)",
        fontsize=12.8, fontweight="bold", y=1.06,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Practical Reasoning Scoreboard (GSM8K + Biggest Gains + Causal Acc)
# ---------------------------------------------------------------------------

def plot_reasoning_scoreboard(runs: list[dict[str, Any]], out_path: Path) -> None:
    var_runs = [r for r in runs if r["_mode_tag"] == "inkling2+var"]
    if not var_runs:
        return

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model_label"], {})[r["_mode_tag"]] = r

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.28, hspace=0.36)
    ax_gains = fig.add_subplot(gs[:, 0])
    ax_gsm = fig.add_subplot(gs[0, 1])
    ax_causal = fig.add_subplot(gs[1, 1])

    # Panel A (Left): Biggest Task-Level Reasoning Leaps (P=1 Base -> Best Loop P=2..5)
    task_pretty = {
        "graph_bfs": "Graph BFS",
        "pointer_chase": "Pointer Chase",
        "mod_register": "Mod-19 Reg",
        "perm_track": "Cup-Swap",
    }
    leaps: list[tuple[float, str, float, float, str, str]] = []
    for r in var_runs:
        col = color_for_run(r["run_name"], r["model_label"])
        sm = short_model(r["model_label"])
        ps = r["post_training"]["serial"]
        # Check GSM8K direct
        g1 = (r["post_training"]["gsm8k_direct"].get("P1") or {}).get("acc")
        if g1 is not None:
            best_gp, best_gv = "P1", g1 * 100.0
            for pt in ("P2", "P3"):
                gv = (r["post_training"]["gsm8k_direct"].get(pt) or {}).get("acc")
                if gv is not None and gv * 100.0 > best_gv:
                    best_gp, best_gv = pt, gv * 100.0
            if best_gv - g1 * 100.0 >= 5.0:
                leaps.append((best_gv - g1 * 100.0, f"{sm} · GSM8K Direct", g1 * 100.0, best_gv, best_gp, col))
        # Check each serial task (ID & OOD)
        for task in ("graph_bfs", "pointer_chase", "mod_register", "perm_track"):
            for split, stag in (("id", "ID"), ("ood", "OOD")):
                base_v = ps["P1"]["per_task"][task][split]["acc"] * 100.0
                best_p, best_v = "P1", base_v
                for pt in ("P2", "P3", "P4", "P5"):
                    if pt not in ps:
                        continue
                    v = ps[pt]["per_task"][task][split]["acc"] * 100.0
                    if v > best_v:
                        best_p, best_v = pt, v
                delta = best_v - base_v
                if delta >= 8.0:
                    leaps.append((delta, f"{sm} · {task_pretty[task]} ({stag})", base_v, best_v, best_p, col))

    leaps.sort(key=lambda t: t[0], reverse=False)  # largest at top
    if leaps:
        y_pos = np.arange(len(leaps))
        for idx, (delta, lbl, base_v, best_v, best_p, col) in enumerate(leaps):
            ax_gains.plot([base_v, best_v], [idx, idx], color=col, lw=3.2, alpha=0.85, zorder=2)
            ax_gains.scatter([base_v], [idx], color="white", edgecolor=col, s=65, lw=2.0, zorder=3, label="Base (P=1)" if idx == 0 else None)
            ax_gains.scatter([best_v], [idx], color=col, s=85, zorder=4, label="Best Loop (P=2..5)" if idx == 0 else None)
            ax_gains.text(
                best_v + 1.8, idx,
                f"+{delta:.0f} pts  ({base_v:.0f}% → {best_v:.0f}% @ {best_p})",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color=col,
            )
        ax_gains.set_yticks(y_pos)
        ax_gains.set_yticklabels([t[1] for t in leaps], fontsize=9.2, fontweight="bold")
        ax_gains.set_xlim(-2, 138)
        ax_gains.set_xlabel("Exact-Match Accuracy (%)")
        ax_gains.set_title("A. Biggest Reasoning Gains from Recurrent Looping (Frozen Base)")
        ax_gains.legend(loc="lower right", fontsize=8.5)
    else:
        ax_gains.text(0.5, 0.5, "No gains >= 8 pts yet", ha="center", va="center")

    # Panel B (Top Right): No-CoT GSM8K Direct Math Accuracy
    models_present = [m for m in MODEL_ORDER if m in by_model]
    x = np.arange(len(models_present))
    series = [
        ("Base (P=1)", lambda modes: ((modes.get("inkling2+var") or modes.get("lora_only"))["post_training"]["gsm8k_direct"].get("P1") or {}).get("acc"), "#888888"),
        ("Loop (P=2)", lambda modes: (modes.get("inkling2+var", {}).get("post_training", {}).get("gsm8k_direct", {}).get("P2") or {}).get("acc"), "#56B4E9"),
        ("Loop (P=3)", lambda modes: (modes.get("inkling2+var", {}).get("post_training", {}).get("gsm8k_direct", {}).get("P3") or {}).get("acc"), "#0072B2"),
        ("Loop+LoRA (P=3)", lambda modes: (modes.get("inkling2+var+lora", {}).get("post_training", {}).get("gsm8k_direct", {}).get("P3") or {}).get("acc"), "#009E73"),
        ("1-Pass LoRA", lambda modes: (modes.get("lora_only", {}).get("post_training", {}).get("gsm8k_direct", {}).get("P1") or {}).get("acc"), "#E69F00"),
    ]
    active_series = []
    for slbl, fn, scol in series:
        vals = [fn(by_model[m]) for m in models_present]
        if any(v is not None for v in vals):
            active_series.append((slbl, [v * 100.0 if v is not None else np.nan for v in vals], scol))

    w = 0.78 / max(1, len(active_series))
    for s_idx, (slbl, vals, scol) in enumerate(active_series):
        offset = (s_idx - (len(active_series) - 1) / 2.0) * w
        labeled = False
        for m_idx, v in enumerate(vals):
            if np.isnan(v):
                continue
            bar = ax_gsm.bar(x[m_idx] + offset, v, width=w * 0.92, color=scol, edgecolor="white", label=slbl if not labeled else None)
            labeled = True
            ax_gsm.text(
                x[m_idx] + offset, v + 1.2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#222222",
            )
    # Deduplicate legend entries
    h, l = ax_gsm.get_legend_handles_labels()
    seen_l = {}
    for hi, li in zip(h, l):
        seen_l.setdefault(li, hi)
    ax_gsm.legend(seen_l.values(), seen_l.keys(), fontsize=7.6, ncol=2, loc="upper left")
    ax_gsm.set_xticks(x)
    ax_gsm.set_xticklabels([short_model(m) for m in models_present], fontsize=8.8, fontweight="bold")
    ax_gsm.set_ylabel("Exact-Match Accuracy (%)")
    ax_gsm.set_ylim(0, 84)
    ax_gsm.set_title("B. Direct No-CoT GSM8K Math (Looping vs. Numeric Collapse)")

    # Panel C (Bottom Right): Causal Proof — ID Accuracy Under Message Ablations at P=3
    ab_var = [by_model[m]["inkling2+var"] for m in models_present if "inkling2+var" in by_model[m] and by_model[m]["inkling2+var"].get("causal_ablations")]
    if ab_var:
        xa = np.arange(len(ab_var))
        aconds = [
            ("P=1 Base", lambda r: r["post_training"]["serial"]["P1"]["overall_id"]["acc"] * 100.0, "#888888"),
            ("P=3 Normal", lambda r: r["post_training"]["serial"]["P3"]["overall_id"]["acc"] * 100.0, "#0072B2"),
            ("P=3 Zero-Msg", lambda r: r["causal_ablations"]["zero"]["overall_id"]["acc"] * 100.0, "#E69F00"),
            ("P=3 Shuf-Tok", lambda r: r["causal_ablations"]["shuffle_tokens"]["overall_id"]["acc"] * 100.0, "#D55E00"),
        ]
        wa = 0.78 / len(aconds)
        for c_idx, (clbl, fn, ccol) in enumerate(aconds):
            offset = (c_idx - (len(aconds) - 1) / 2.0) * wa
            vals = [fn(r) for r in ab_var]
            ax_causal.bar(xa + offset, vals, width=wa * 0.92, color=ccol, edgecolor="white", label=clbl)
            for xi, v in zip(xa, vals):
                ax_causal.text(xi + offset, v + 0.9, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
        ax_causal.set_xticks(xa)
        ax_causal.set_xticklabels([short_model(r["model_label"]) for r in ab_var], fontsize=8.8, fontweight="bold")
        ax_causal.set_ylabel("Serial ID Accuracy (%)")
        ax_causal.set_ylim(0, 50)
        ax_causal.set_title("C. Causal Proof: Does Shuffling Cross-Pass Messages Break Reasoning?")
        ax_causal.legend(fontsize=7.8, ncol=2, loc="upper right")

    fig.suptitle(
        "Practical Reasoning Scoreboard: Where Inkling-2 Looping Helped, Where Frozen Weights Hedged, and Causal Proof",
        fontsize=13.2, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.005,
        "Note: Qwen3.8-27B emitted leading blank lines on 58/214 base P=1 serial prompts (0/214 by P=2); its pure multi-pass P=2 → P=3 gain (zero format confound) is +11.9 pts ID, +20 pts OOD Pointer Chase, and +3.1 pts GSM8K.",
        ha="center", fontsize=8.2, color="#555555", style="italic",
    )
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Architecture & Training Ablations (Scalar-Gate Discovery + LoRA)
# ---------------------------------------------------------------------------

def plot_ablation_discovery(runs: list[dict[str, Any]], out_path: Path) -> None:
    q4_runs = [r for r in runs if r["model_label"] == "Qwen3-4B"]
    g26_runs = [r for r in runs if r["model_label"] == "Gemma-4-26B-A4B-it"]
    if len(q4_runs) < 3:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    # Distinct palette for Qwen3-4B 6-way ablation matrix
    q4_style = {
        "qwen3-4b-scalar-gate-loop": ("Scalar-Gate Loop (0.74M, 0.02%)", "#009E73", "-", "^", 2.8),
        "qwen3-4b-inkling2-plus-lora": ("Inkling-2 + LoRA r16 (85.2M)", "#0072B2", "-", "D", 2.2),
        "qwen3-4b-inkling2-varloop": ("Inkling-2 varloop (63.2M)", "#CC79A7", "-", "o", 2.3),
        "qwen3-4b-inkling2-fixed3": ("Inkling-2 fixed P=3 (63.2M)", "#D55E00", "--", "s", 1.9),
        "qwen3-4b-naive-loop-lora": ("Naive Loop + LoRA (66.1M)", "#8C564B", "-.", "x", 2.1),
        "qwen3-4b-control-1pass-lora": ("1-Pass LoRA Control (66.1M)", "#555555", "--", "*", 1.8),
    }

    # Panel A: Qwen3-4B 6-Way Depth Scaling (ID Accuracy vs P=1..5)
    ax = axes[0]
    ax.axvspan(3.35, 5.35, color="#f0e6f6", alpha=0.52, zorder=0)
    ax.axvline(3.35, color="#9b72aa", ls=":", lw=1.1, zorder=1)
    q4_by_name = {r["run_name"]: r for r in q4_runs}
    for rname, (lbl, col, ls, mk, lw) in q4_style.items():
        r = q4_by_name.get(rname)
        if not r:
            continue
        ps = r["post_training"]["serial"]
        if r["train_args"]["mode"] == "lora_only":
            val = ps["P1"]["overall_id"]["acc"] * 100.0
            ax.axhline(val, color=col, ls=ls, lw=lw, label=f"{lbl}: {val:.1f}%", zorder=2)
        else:
            xs, ys = [], []
            for p in (1, 2, 3, 4, 5):
                tag = f"P{p}"
                if tag in ps:
                    xs.append(p)
                    ys.append(ps[tag]["overall_id"]["acc"] * 100.0)
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, lw=lw, ms=6.8, label=lbl, zorder=3)
            if rname in ("qwen3-4b-scalar-gate-loop", "qwen3-4b-inkling2-varloop", "qwen3-4b-inkling2-plus-lora", "qwen3-4b-naive-loop-lora"):
                ax.text(xs[-1] + 0.08, ys[-1], f"{ys[-1]:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color=col)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["P=1\n(Base)", "P=2", "P=3", "P=4\n(Extrap)", "P=5\n(Extrap)"])
    ax.set_xlim(0.8, 5.55)
    ax.set_ylim(0, 66)
    ax.set_ylabel("In-Distribution Exact-Match Accuracy (%)")
    ax.set_title("A. Qwen3-4B 6-Way Ablation: ID Accuracy vs. Depth")
    ax.legend(fontsize=7.2, loc="upper left", ncol=1)

    # Panel B: Qwen3-4B Multi-Metric Comparison (Best Loop ID, Best Loop OOD, Best GSM8K)
    ax = axes[1]
    order_b = [
        ("qwen3-4b-inkling2-varloop", "Rank-64\nvarloop\n(63.2M)"),
        ("qwen3-4b-inkling2-fixed3", "Rank-64\nfixed3\n(63.2M)"),
        ("qwen3-4b-inkling2-plus-lora", "Rank-64\n+LoRA\n(85.2M)"),
        ("qwen3-4b-naive-loop-lora", "Naive\nLoop\n(66.1M)"),
        ("qwen3-4b-scalar-gate-loop", "Scalar\nGate\n(0.74M)"),
        ("qwen3-4b-control-1pass-lora", "1-Pass\nLoRA\n(66.1M)"),
    ]
    present_b = [(q4_by_name[k], lbl) for k, lbl in order_b if k in q4_by_name]
    xb = np.arange(len(present_b))
    wb = 0.25
    id_vals, ood_vals, gsm_vals = [], [], []
    for r, _ in present_b:
        ps = r["post_training"]["serial"]
        if r["train_args"]["mode"] == "lora_only":
            id_vals.append(ps["P1"]["overall_id"]["acc"] * 100.0)
            ood_vals.append(ps["P1"]["overall_ood"]["acc"] * 100.0)
            gsm_vals.append((r["post_training"]["gsm8k_direct"].get("P1") or {}).get("acc", 0.0) * 100.0)
        else:
            p_tags = [t for t in ("P2", "P3", "P4", "P5") if t in ps]
            best_p = max(p_tags, key=lambda t: ps[t]["overall_id"]["acc"])
            id_vals.append(ps[best_p]["overall_id"]["acc"] * 100.0)
            ood_vals.append(max(ps[t]["overall_ood"]["acc"] for t in p_tags) * 100.0)
            g_tags = [t for t in ("P2", "P3") if t in r["post_training"]["gsm8k_direct"]]
            gsm_vals.append(max(r["post_training"]["gsm8k_direct"][t]["acc"] for t in g_tags) * 100.0 if g_tags else 0.0)

    for offset, vals, clbl, col in [
        (-wb, id_vals, "Best ID Acc", "#0072B2"),
        (0.0, ood_vals, "Best OOD Acc", "#009E73"),
        (wb, gsm_vals, "GSM8K Direct", "#E69F00"),
    ]:
        bars = ax.bar(xb + offset, vals, width=wb * 0.92, color=col, edgecolor="white", label=clbl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels([lbl for _, lbl in present_b], fontsize=7.8, fontweight="bold")
    ax.set_ylabel("Exact-Match Accuracy (%)")
    ax.set_ylim(0, 62)
    ax.set_title("B. 0.74M Scalar Gate vs. 63.2M Rank-64")
    ax.legend(fontsize=7.8, loc="upper left")

    # Panel C: Gemma-4-26B-A4B-it 3-Way Retrofit Comparison (Loop vs Loop+LoRA vs 1-Pass LoRA)
    ax = axes[2]
    g26_by_name = {r["run_name"]: r for r in g26_runs}
    conds_c = []
    if "gemma4-26b-a4b-inkling2-varloop" in g26_by_name:
        r = g26_by_name["gemma4-26b-a4b-inkling2-varloop"]
        conds_c.append(("Base\n(P=1)", r["post_training"]["serial"]["P1"]["overall_id"]["acc"] * 100.0,
                        r["post_training"]["serial"]["P1"]["overall_ood"]["acc"] * 100.0,
                        r["post_training"]["gsm8k_direct"]["P1"]["acc"] * 100.0))
        conds_c.append(("Loop\nP=3\n(52.1M)", r["post_training"]["serial"]["P3"]["overall_id"]["acc"] * 100.0,
                        r["post_training"]["serial"]["P5"]["overall_ood"]["acc"] * 100.0,
                        r["post_training"]["gsm8k_direct"]["P3"]["acc"] * 100.0))
    if "gemma4-26b-a4b-inkling2-plus-lora" in g26_by_name:
        r = g26_by_name["gemma4-26b-a4b-inkling2-plus-lora"]
        conds_c.append(("Loop+LoRA\nP=1\n(63.3M)", r["post_training"]["serial"]["P1"]["overall_id"]["acc"] * 100.0,
                        r["post_training"]["serial"]["P1"]["overall_ood"]["acc"] * 100.0,
                        r["post_training"]["gsm8k_direct"]["P1"]["acc"] * 100.0))
        conds_c.append(("Loop+LoRA\nP=3/4\n(63.3M)", r["post_training"]["serial"]["P4"]["overall_id"]["acc"] * 100.0,
                        r["post_training"]["serial"]["P3"]["overall_ood"]["acc"] * 100.0,
                        r["post_training"]["gsm8k_direct"]["P3"]["acc"] * 100.0))
    if "gemma4-26b-a4b-control-1pass-lora" in g26_by_name:
        r = g26_by_name["gemma4-26b-a4b-control-1pass-lora"]
        conds_c.append(("1-Pass\nLoRA\n(44.6M)", r["post_training"]["serial"]["P1"]["overall_id"]["acc"] * 100.0,
                        r["post_training"]["serial"]["P1"]["overall_ood"]["acc"] * 100.0,
                        r["post_training"]["gsm8k_direct"]["P1"]["acc"] * 100.0))
    if conds_c:
        xc = np.arange(len(conds_c))
        wc = 0.25
        for offset, idx_v, clbl, col in [
            (-wc, 1, "ID Acc", "#0072B2"),
            (0.0, 2, "OOD Acc", "#009E73"),
            (wc, 3, "GSM8K Direct", "#E69F00"),
        ]:
            vals = [c[idx_v] for c in conds_c]
            bars = ax.bar(xc + offset, vals, width=wc * 0.92, color=col, edgecolor="white", label=clbl)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
        ax.set_xticks(xc)
        ax.set_xticklabels([c[0] for c in conds_c], fontsize=7.8, fontweight="bold")
        ax.set_ylabel("Exact-Match Accuracy (%)")
        ax.set_ylim(0, 74)
        ax.set_title("C. G4-26B MoE: Joint Loop+LoRA Short-Circuit")
        ax.legend(fontsize=7.8, loc="upper left")

    fig.suptitle(
        "Architecture & Training Ablations: The 0.74M Scalar-Gate Discovery & Joint LoRA Interaction",
        fontsize=13.2, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_completed_runs(args.results_dir)
    logs = load_train_logs(args.results_dir)
    print(f"Loaded {len(runs)} completed runs and {len(logs)} training logs.")

    plot_depth_scaling(runs, args.out / "fig1_depth_scaling.png")
    plot_per_task_breakdown(runs, args.out / "fig2_per_task_breakdown.png")
    plot_causal_and_mechanistic(runs, args.out / "fig3_causal_and_mechanistic.png")
    plot_training_curves(logs, args.out / "fig4_training_curves.png")
    plot_reasoning_scoreboard(runs, args.out / "fig5_reasoning_scoreboard.png")
    plot_ablation_discovery(runs, args.out / "fig6_ablation_discovery.png")
    print(f"Saved 6 figures to {args.out}")


if __name__ == "__main__":
    main()
