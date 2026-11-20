"""Write the expert files each of four Step-5-Preview-Spark ranks actually opens."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile

REVISION = "51a46a7f62eee1225d124170da903bba23d04a26"
MANIFEST_SHA256 = "a427f0ae33b2158f7550d21caa5e508a93b1e859283e0fcddf9ce7973b69f3c8"
EXPERTS = 352
RANKS = 4
BUFFER = 8 << 20
TENSOR = re.compile(r"model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(?:gate|up|down)_proj\.(?:trellis|suh|svh|mul1|mcg)")


def open_header(source):
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError("missing safetensors header")
    length, = struct.unpack("<Q", prefix)
    if not 2 <= length <= 100_000_000:
        raise ValueError("invalid safetensors header length")
    encoded = source.read(length)
    if len(encoded) != length:
        raise ValueError("truncated safetensors header")
    return prefix + encoded, json.loads(encoded)


def make_header(fields):
    data = json.dumps(fields, separators=(",", ":"), ensure_ascii=False).encode()
    data += b" " * (-len(data) % 8)
    return struct.pack("<Q", len(data)) + data


def partition_layer(snapshot, layer, expected_hash, output, ranks, reference):
    relative = Path("exl3", "experts", f"L{layer:02d}.safetensors")
    original = snapshot / relative
    records = []
    with original.open("rb") as source, ExitStack() as stack:
        original_header, fields = open_header(source)
        source_hash = hashlib.sha256(original_header) if reference is None else None
        metadata = fields.pop("__metadata__", None)
        headers = [{} if metadata is None else {"__metadata__": metadata} for _ in range(RANKS)]
        sizes = [0] * RANKS
        owners = set()
        segments = []
        cursor = 0
        for key, field in sorted(fields.items(), key=lambda item: item[1]["data_offsets"][0]):
            match = TENSOR.fullmatch(key)
            if not match or int(match[1]) != layer or not 0 <= int(match[2]) < EXPERTS:
                raise ValueError(f"unexpected expert tensor {key}")
            expert = int(match[2])
            owners.add(expert)
            rank = expert // (EXPERTS // RANKS)
            begin, end = field["data_offsets"]
            if begin != cursor or end <= begin:
                raise ValueError(f"invalid or noncontiguous tensor range for {key}")
            count = end - begin
            headers[rank][key] = {**field, "data_offsets": [sizes[rank], sizes[rank] + count]}
            sizes[rank] += count
            if segments and segments[-1][0] == rank:
                segments[-1][1] += count
            else:
                segments.append([rank, count])
            cursor = end
        if cursor + len(original_header) != original.stat().st_size or len(owners) != EXPERTS:
            raise ValueError(f"incomplete expert coverage in {relative}")

        streams, digests = {}, {}
        for rank in ranks:
            header = headers[rank]
            path = output / f"rank{rank}" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("xb"))
            data = make_header(header)
            stream.write(data)
            digests[rank] = hashlib.sha256(data)
            streams[rank] = stream
        for rank, count in segments:
            if rank not in streams:
                source.seek(count, 1)
                continue
            while count:
                data = source.read(min(BUFFER, count))
                if not data:
                    raise ValueError(f"truncated tensor payload in {relative}")
                streams[rank].write(data)
                digests[rank].update(data)
                if source_hash is not None:
                    source_hash.update(data)
                count -= len(data)
        if source_hash is not None and source_hash.hexdigest() != expected_hash:
            raise ValueError(f"source SHA-256 mismatch for {relative}")
        for rank in ranks:
            row = dict(rank=rank, file=str(relative),
                       tensors=len(headers[rank]) - int(metadata is not None),
                       bytes=streams[rank].tell(), sha256=digests[rank].hexdigest())
            if reference is not None and reference.get((rank, str(relative))) != row:
                raise ValueError(f"rank {rank} output differs from the verified layout for {relative}")
            records.append(row)
    return records


def partition(snapshot, destination, rank=None, layout_path=None):
    snapshot = snapshot.resolve()
    destination = destination.resolve()
    if destination == snapshot or snapshot in destination.parents:
        raise ValueError("choose a separate destination outside the source snapshot")
    if destination.exists():
        raise FileExistsError(destination)
    manifest_data = (snapshot / "sha256-manifest.txt").read_bytes()
    if hashlib.sha256(manifest_data).hexdigest() != MANIFEST_SHA256:
        raise ValueError(f"expected the checksum manifest from Step-5-Preview-Spark {REVISION}")
    sums = dict((path, sha) for sha, path in
                (line.split(maxsplit=1) for line in manifest_data.decode().splitlines()))
    config_data = (snapshot / "config.json").read_bytes()
    if hashlib.sha256(config_data).hexdigest() != sums["config.json"]:
        raise ValueError("source config differs from the pinned manifest")
    config = json.loads(config_data)["quantization_config"]
    if config["n_experts"] != EXPERTS or config["quant_method"] != "step5_exl3":
        raise ValueError("unexpected expert layout")
    ranks = range(RANKS) if rank is None else (rank,)
    reference = None
    if (rank is None) != (layout_path is None):
        raise ValueError("use --rank and --layout together")
    if layout_path is not None:
        layout = json.loads(layout_path.read_bytes())
        if layout["revision"] != REVISION or layout["world_size"] != RANKS or rank not in range(RANKS):
            raise ValueError("incompatible rank or layout")
        reference = {(r["rank"], r["file"]): r for r in layout["files"]}
        if len(reference) != RANKS * len(config["expert_layers"]):
            raise ValueError("expected the verified layout for all four ranks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".step5-layout-") as temp:
        output = Path(temp) / "layout"
        records = []
        for layer in config["expert_layers"]:
            relative = f"exl3/experts/L{layer:02d}.safetensors"
            rows = partition_layer(snapshot, layer, sums[relative], output, ranks, reference)
            records.extend(rows)
            print(f"Verified and partitioned {relative}", flush=True)
        layout = dict(revision=REVISION, world_size=RANKS, experts_per_rank=EXPERTS // RANKS,
                      files=records)
        (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
        output.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rank", type=int, choices=range(RANKS), help="materialize one rank")
    parser.add_argument("--layout", type=Path, help="verified four-rank layout for --rank")
    args = parser.parse_args()
    partition(args.snapshot, args.destination, args.rank, args.layout)
