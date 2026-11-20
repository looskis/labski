"""Compare compact K2 indexes to vLLM's native two-rank linear expert placement."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import platform

import vllm
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

from inventory import EXPERT, REVISION


def check(original, indexes):
    mapping = json.loads(original.read_bytes())["weight_map"]
    results = []
    for rank, path in enumerate(indexes):
        count, table, _ = determine_expert_map(ep_size=2, ep_rank=rank, global_num_experts=192,
                                               expert_placement_strategy="linear")
        local = {i: local_id for i, local_id in enumerate(table.tolist()) if local_id >= 0}
        if count != 96 or sorted(local.values()) != list(range(count)):
            raise ValueError("unexpected vLLM expert placement")
        layers = Counter()
        selected = {}
        for key, shard in mapping.items():
            match = EXPERT.fullmatch(key)
            if match:
                layer, expert = int(match[1]), int(match[2])
                if expert not in local:
                    continue
                layers[layer] += 1
            elif ".mlp.experts." in key:
                raise ValueError(f"unknown expert tensor {key}")
            selected[key] = shard
        data = path.read_bytes()
        if len(layers) != 58 or set(layers.values()) != {96 * 3 * 4}:
            raise ValueError("expected three projections and four tensors for each local expert")
        if json.loads(data)["weight_map"] != selected:
            raise ValueError(f"rank {rank} index differs from vLLM's placement")
        results.append(dict(rank=rank, first_global_expert=min(local), last_global_expert=max(local),
                            local_experts=count, routed_layers=len(layers), routed_tensors=sum(layers.values()),
                            common_tensors=len(selected) - sum(layers.values()), index_sha256=hashlib.sha256(data).hexdigest()))
    return dict(revision=REVISION, vllm=vllm.__version__, machine=platform.machine(), ranks=results,
                placement_module_sha256=hashlib.sha256(Path(inspect.getfile(determine_expert_map)).read_bytes()).hexdigest())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source_index", type=Path)
    cli.add_argument("rank_indexes", nargs=2, type=Path)
    args = cli.parse_args()
    print(json.dumps(check(args.source_index, args.rank_indexes), indent=2))
