"""Estimate four-client admission from the three measured fixed-depth DFlash2 pools."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CALIBRATION = [("7-kv3-repeat", 3072), ("7-kv5632m", 5632), ("7-kv9", 9216)]
MIB = 2**20


def read(folder, name):
    return json.loads((HERE / folder / name).read_text())


def estimate(mib):
    control = read("7-kv3-repeat", "effective-cache.json")
    full_request = round(control["num_gpu_blocks"] / control["kv_cache_max_concurrency"])
    observed = []
    for folder, pool_mib in CALIBRATION:
        capacity = read(folder, "kv-capacity.json")
        context = capacity["max_request_tokens"]
        slots = capacity["kv_tokens"]
        # vLLM reports floor(context * blocks / blocks_per_full_request).
        blocks = math.ceil(Fraction(slots * full_request, context))
        if context <= full_request or blocks * context // full_request != slots:
            raise ValueError(f"capacity in {folder} is inconsistent with the control")
        trials = (HERE / folder / "scheduler-trials.jsonl").read_text().splitlines()
        short_blocks = set()
        peak_running = set()
        for trial in map(json.loads, trials):
            used = trial["max_kv_fraction"] * (blocks - 1)
            if not math.isclose(used, round(used), abs_tol=1e-7):
                raise ValueError(f"nonintegral occupied blocks in {folder}")
            if trial["concurrency"] == 1:
                short_blocks.add(round(used))
            if trial["concurrency"] == 4:
                peak_running.add(int(trial["max_running"]))
        if len(short_blocks) != 1 or len(peak_running) != 1:
            raise ValueError(f"occupancy varied across repeats in {folder}")
        occupied = short_blocks.pop()
        running = peak_running.pop()
        if running != min(4, (blocks - 1) // occupied):
            raise ValueError(f"observed admission differs from estimate in {folder}")
        observed.append(dict(pool_mib=pool_mib, total_blocks=blocks,
                             blocks_per_short_request=occupied, peak_running=running))
    short = observed[0]["blocks_per_short_request"]
    if any(r["blocks_per_short_request"] != short for r in observed):
        raise ValueError("per-request block use differs across budgets")
    # If blocks = floor(budget / cost), the lower bound on cost is open.
    lower = max(Fraction(r["pool_mib"] * MIB, r["total_blocks"] + 1) for r in observed)
    upper = min(Fraction(r["pool_mib"] * MIB, r["total_blocks"]) for r in observed)
    if lower >= upper:
        raise ValueError("no constant block footprint fits the measured budgets")
    budget = mib * MIB
    least = int(budget // upper)
    most = math.ceil(budget / lower) - 1
    required = short * 4 + 1  # vLLM reserves block zero.
    return dict(calibration=observed, blocks_per_full_request=full_request,
                four_request_threshold_gib=[float(required * cost / 2**30) for cost in (lower, upper)],
                query_mib=mib, predicted_total_blocks=[least, most],
                predicted_running_of_four=[max(0, min(4, (blocks - 1) // short)) for blocks in (least, most)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib", type=int, default=5376)
    args = parser.parse_args()
    if args.mib <= 0:
        parser.error("supply a positive KV pool size")
    print(json.dumps(estimate(args.mib), indent=2))
