#!/usr/bin/env python3
"""Unit tests for Inkling-2 loop task generators and mathematical properties."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks


def test_task_determinism_and_correctness():
    ev1 = tasks.build_serial_eval_suite(seed=20261122, per_depth=4)
    ev2 = tasks.build_serial_eval_suite(seed=20261122, per_depth=4)
    assert len(ev1) == len(ev2) and len(ev1) > 0
    for a, b in zip(ev1, ev2):
        assert a.prompt == b.prompt and a.answer == b.answer

    # Verify train and eval keys are disjoint
    assert set(tasks.TRAIN_KEYS).isdisjoint(set(tasks.EVAL_KEYS))

    # Verify pointer_chase answers directly from prompt table
    for ex in ev1:
        if ex.task == "pointer_chase":
            table_line = [l for l in ex.prompt.splitlines() if l.startswith("Table:")][-1]
            pairs = [p.strip() for p in table_line.removeprefix("Table:").split(",")]
            mp = dict(p.split("->") for p in pairs)
            cur = ex.meta["start"]
            for _ in range(ex.meta["hops"]):
                cur = mp[cur]
            assert cur == ex.meta["target"]
        elif ex.task == "perm_track":
            line = ex.prompt.splitlines()[-1]
            slots_part = line.split("|")[0].removeprefix("Slots:").strip()
            state = [s.split("=")[1].strip() for s in slots_part.split(",")]
            swaps_part = line.split("|")[1].removeprefix(" Swaps:").strip()
            for tok in swaps_part.split(", "):
                ij = tok.removeprefix("swap(").removesuffix(")").split(",")
                i, j = int(ij[0]) - 1, int(ij[1]) - 1
                state[i], state[j] = state[j], state[i]
            assert state[ex.meta["query_slot"] - 1] == ex.meta["target"]

    print(f"All task tests passed ({len(ev1)} eval examples verified)!")


if __name__ == "__main__":
    test_task_determinism_and_correctness()
