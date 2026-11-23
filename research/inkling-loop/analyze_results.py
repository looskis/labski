#!/usr/bin/env python3
"""Aggregate multi-Spark Inkling-2 recurrent loop results with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any


def paired_bootstrap_ci(
    diffs: list[float],
    n_draws: int = 10000,
    seed: int = 170521,
) -> tuple[float, float, float]:
    if not diffs:
        return (0.0, 0.0, 0.0)
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    draws = []
    for _ in range(n_draws):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        draws.append(s / n)
    draws.sort()
    return (mean, draws[int(0.025 * n_draws)], draws[int(0.975 * n_draws)])


def summarize_run(run_dir: Path) -> dict[str, Any]:
    m = json.loads((run_dir / "metrics.json").read_text())
    preds_path = run_dir / "eval_predictions.jsonl"
    by_tag: dict[str, list[dict[str, Any]]] = {}
    if preds_path.is_file():
        for line in preds_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_tag.setdefault(r["eval_tag"], []).append(r)

    paired_ci = {}
    base_rows = by_tag.get("baseline_P1") or by_tag.get("P1")
    if base_rows:
        for tag, rows in sorted(by_tag.items()):
            if tag in ("baseline_P1", "P1") or len(rows) != len(base_rows):
                continue
            acc_diffs = [r["correct"] - b["correct"] for b, r in zip(base_rows, rows)]
            nll_diffs = [r["ans_nll"] - b["ans_nll"] for b, r in zip(base_rows, rows)]
            id_acc_diffs = [r["correct"] - b["correct"] for b, r in zip(base_rows, rows) if b["split"] == "eval_id"]
            ood_acc_diffs = [r["correct"] - b["correct"] for b, r in zip(base_rows, rows) if b["split"] == "eval_ood"]
            gained = sum(1 for d in acc_diffs if d > 0)
            lost = sum(1 for d in acc_diffs if d < 0)
            paired_ci[tag] = {
                "overall_acc_delta_ci": [round(x, 4) for x in paired_bootstrap_ci(acc_diffs)],
                "id_acc_delta_ci": [round(x, 4) for x in paired_bootstrap_ci(id_acc_diffs)],
                "ood_acc_delta_ci": [round(x, 4) for x in paired_bootstrap_ci(ood_acc_diffs)],
                "nll_delta_ci": [round(x, 4) for x in paired_bootstrap_ci(nll_diffs)],
                "gained": gained,
                "lost": lost,
            }

    return {
        "run_name": m["run_name"],
        "model_label": m["model_label"],
        "hostname": m["hostname"],
        "mode": m["train_args"]["mode"],
        "varloop": m["train_args"]["varloop"],
        "steps": m["train_args"]["steps"],
        "params": m["param_stats"],
        "peak_mem_gib": m["peak_train_mem_gib"],
        "train_elapsed_s": m["train_elapsed_s"],
        "step0_max_diff_p3_p1": m["step0_identity_check"]["max_abs_diff_p3_vs_p1"],
        "hw": m["hardware_forward_cost"],
        "baseline_p1": {
            "id_acc": m["baseline_p1"]["serial"]["overall_id"]["acc"],
            "ood_acc": m["baseline_p1"]["serial"]["overall_ood"]["acc"],
            "id_nll": m["baseline_p1"]["serial"]["overall_id"]["mean_ans_nll"],
            "ood_nll": m["baseline_p1"]["serial"]["overall_ood"]["mean_ans_nll"],
            "wt_ppl": (m["baseline_p1"]["wikitext2"] or {}).get("perplexity"),
            "gsm_acc": (m["baseline_p1"]["gsm8k_direct"] or {}).get("acc"),
        },
        "post_serial": {
            k: {
                "id_acc": v["overall_id"]["acc"],
                "ood_acc": v["overall_ood"]["acc"],
                "id_nll": v["overall_id"]["mean_ans_nll"],
                "ood_nll": v["overall_ood"]["mean_ans_nll"],
            }
            for k, v in m["post_training"]["serial"].items()
        },
        "post_wt_ppl": {k: v["perplexity"] for k, v in m["post_training"]["wikitext2"].items()},
        "post_gsm_acc": {k: v["acc"] for k, v in m["post_training"]["gsm8k_direct"].items()},
        "causal_ablations": {
            k: {
                "id_acc": v["overall_id"]["acc"],
                "ood_acc": v["overall_ood"]["acc"],
                "id_nll": v["overall_id"]["mean_ans_nll"],
                "ood_nll": v["overall_ood"]["mean_ans_nll"],
            }
            for k, v in m.get("causal_ablations", {}).items()
        },
        "paired_ci_vs_p1": paired_ci,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    summaries = []
    for sub in sorted(args.runs_dir.iterdir()):
        if (sub / "metrics.json").is_file():
            summaries.append(summarize_run(sub))

    if args.out:
        args.out.write_text(json.dumps(summaries, indent=2))

    print("| Run | Model | Mode | Trainable | P1 ID/OOD | P2 ID/OOD | P3 ID/OOD | P4 ID/OOD | P5 ID/OOD | Zero-Msg | Shuf-Msg |")
    print("| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |")
    for s in summaries:
        ps = s["post_serial"]
        def fmt(tag):
            if tag not in ps:
                return "-"
            return f"{ps[tag]['id_acc']*100:.1f}% / {ps[tag]['ood_acc']*100:.1f}%"
        ab = s["causal_ablations"]
        def fmt_ab(k):
            if k not in ab:
                return "-"
            return f"{ab[k]['id_acc']*100:.1f}% / {ab[k]['ood_acc']*100:.1f}%"
        tp = f"{s['params']['trainable_params']/1e6:.1f}M"
        print(
            f"| {s['run_name']} | {s['model_label']} | {s['mode']}{'+var' if s['varloop'] else ''} "
            f"| {tp} | {fmt('P1')} | {fmt('P2')} | {fmt('P3')} | {fmt('P4')} | {fmt('P5')} "
            f"| {fmt_ab('zero')} | {fmt_ab('shuffle_tokens')} |"
        )


if __name__ == "__main__":
    main()
