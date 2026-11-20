"""Measure the byte ranges for contiguous expert placement in K2 Horizon 375B NVFP4."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


REVISION = "d3a6e26e4d59243ed6e569e39ca6facf4da84c05"
EXPERT = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
    r"(input_global_scale|weight_global_scale|weight_packed|weight_scale)"
)
GAPS = (0, 4096, 65536, 1048576)


def intervals(ranges, gap):
    merged = []
    for start, stop in ranges:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    lengths = [stop - start for start, stop in merged]
    block = 64 << 20
    return len(lengths), sum(lengths), sum((length + block - 1) // block for length in lengths)


def inventory(snapshot):
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
    shards = sorted(set(index.values()))
    if len(shards) != 12 or any(Path(name).name != name for name in shards):
        raise ValueError("expected the 12 indexed NVFP4 shards")
    placements = {(n, rank): {gap: [0, 0, 0] for gap in GAPS} for n in (2, 4) for rank in range(n)}
    experts, expert_bytes, common_bytes, header_bytes = 0, 0, 0, 0
    components = {}
    files = []
    for filename in shards:
        path = snapshot / filename
        with path.open("rb") as f:
            prefix = f.read(8)
            size, = struct.unpack("<Q", prefix)
            if not 2 <= size <= 100_000_000:
                raise ValueError(f"invalid header length in {filename}")
            raw = f.read(size)
        if len(raw) != size:
            raise ValueError(f"truncated header in {filename}")
        fields = json.loads(raw)
        fields.pop("__metadata__", None)
        if set(fields) != {key for key, shard in index.items() if shard == filename}:
            raise ValueError(f"index differs from {filename}")
        offset = len(prefix) + size
        header_bytes += offset
        spans = {p: [(0, offset)] for p in placements}
        previous = 0
        for key, field in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
            a, b = field["data_offsets"]
            if a != previous or b <= a:
                raise ValueError(f"invalid tensor offsets for {key}")
            previous = b
            expert = EXPERT.fullmatch(key)
            if expert:
                layer, expert_id = map(int, expert.groups()[:2])
                if not (3 <= layer < 61 and 0 <= expert_id < 192):
                    raise ValueError(f"unexpected expert placement for {key}")
                experts += 1
                expert_bytes += b - a
                component = expert[4]
            else:
                if ".mlp.experts." in key:
                    raise ValueError(f"unknown routed weight {key}")
                common_bytes += b - a
                component = "common"
            totals = components.setdefault((component, field["dtype"]), [0, 0])
            totals[0] += 1
            totals[1] += b - a
            for (n, rank), kept in spans.items():
                if expert and expert_id // (192 // n) != rank:
                    continue
                if kept[-1][1] == offset + a:
                    kept[-1] = (kept[-1][0], offset + b)
                else:
                    kept.append((offset + a, offset + b))
        if previous + offset != path.stat().st_size:
            raise ValueError(f"file length differs from the header in {filename}")
        files.append(dict(file=filename, tensors=len(fields), file_bytes=previous + offset,
                          header_bytes=offset, header_sha256=hashlib.sha256(prefix + raw).hexdigest()))
        for p, kept in spans.items():
            for gap in GAPS:
                count, total, bounded = intervals(kept, gap)
                placements[p][gap][0] += count
                placements[p][gap][1] += total
                placements[p][gap][2] += bounded
    if experts != 58 * 192 * 3 * 4 or len(index) != 134300:
        raise ValueError("unexpected number of tensors")
    pinned = json.loads(Path(__file__).with_name("375b-nvfp4.json").read_bytes())
    if files != pinned["indexed_files"]:
        raise ValueError("the indexed headers differ from the pinned K2 Horizon checkpoint")
    ranks = [dict(ranks=n, rank=r, experts_per_sparse_layer=192 // n,
                  ranges=[dict(max_gap_bytes=g, requests=v[0], transfer_bytes=v[1],
                               requests_with_64mib_limit=v[2]) for g, v in totals.items()])
             for (n, r), totals in placements.items()]
    return dict(revision=REVISION, indexed_files=files, tensors=len(index), expert_tensors=experts,
                expert_bytes=expert_bytes, common_tensor_bytes=common_bytes, header_bytes=header_bytes,
                components=[dict(component=name, dtype=dtype, tensors=count, bytes=size)
                            for (name, dtype), (count, size) in sorted(components.items())],
                rank_placement=ranks)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    args = cli.parse_args()
    print(json.dumps(inventory(args.snapshot), indent=2))
