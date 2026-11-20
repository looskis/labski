"""Check every routed expert tensor against the pinned source through the safetensors reader."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open


def verify(source, output):
    layout = json.loads((output / "layout.json").read_bytes())
    world = layout["world_size"]
    per_rank = layout["experts_per_rank"]
    totals = [{"rank": r, "files": 0, "tensors": 0, "payload_bytes": 0} for r in range(world)]
    for row in layout["files"]:
        rank, name = row["rank"], row["file"]
        total = totals[rank]
        with safe_open(source / name, framework="numpy") as original, \
                safe_open(output / f"rank{rank}" / name, framework="numpy") as subset:
            owned = set()
            for key in original.keys():
                expert = int(re.search(r"\.experts\.(\d+)\.", key)[1])
                if expert // per_rank == rank:
                    owned.add(key)
            if owned != set(subset.keys()) or len(owned) != row["tensors"] or original.metadata() != subset.metadata():
                raise ValueError(f"incorrect rank {rank} tensor set or file metadata in {name}")
            for key in owned:
                a, b = original.get_tensor(key), subset.get_tensor(key)
                if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(
                        a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)):
                    raise ValueError(f"rank {rank} differs from source at {key}")
                total["tensors"] += 1
                total["payload_bytes"] += a.nbytes
            total["files"] += 1
    return totals


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    print(json.dumps(verify(args.source, args.output), indent=2))
