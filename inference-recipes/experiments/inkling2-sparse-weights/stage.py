"""Copy the byte ranges needed by a four-way Inkling 2 Small expert rank."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

REVISION = "4d564489295e30e9341a9822a8046f0c9ded1a7b"
RANKS = 4
EXPERTS = 384
EXPERT = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.(w13|w2)_weight(_scale|_amax)?$")
GLOBAL = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.(w13|w2)_weight_outer_scale_denominator$")
CHUNK = 8 << 20


def header(fp):
    prefix = fp.read(8)
    if len(prefix) != 8:
        raise ValueError("missing SafeTensors header")
    size, = struct.unpack("<Q", prefix)
    if not 2 <= size <= 100_000_000:
        raise ValueError("invalid header length")
    raw = fp.read(size)
    if len(raw) != size:
        raise ValueError("truncated header")
    fields = json.loads(raw)
    fields.pop("__metadata__", None)
    return prefix + raw, fields


def ranges(fields, rank, header_bytes, file_bytes):
    if rank not in range(RANKS):
        raise ValueError("rank must be 0 through 3")
    kept = [(0, header_bytes)]
    expert_bytes = 0
    end = 0
    for name, tensor in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
        start, stop = tensor["data_offsets"]
        if start != end or stop < start:
            raise ValueError(f"invalid tensor offsets for {name}")
        end = stop
        if EXPERT.fullmatch(name):
            if tensor["shape"][0] != EXPERTS or (stop - start) % EXPERTS:
                raise ValueError(f"expected {EXPERTS} expert rows in {name}")
            expert_bytes += stop - start
            chunk = (stop - start) // RANKS
            start += rank * chunk
            stop = start + chunk
        elif ".mlp.experts." in name and not GLOBAL.fullmatch(name):
            raise ValueError(f"unexpected expert tensor {name}")
        start, stop = start + header_bytes, stop + header_bytes
        if start == stop:
            continue
        if start == kept[-1][1]:
            kept[-1] = (kept[-1][0], stop)
        else:
            kept.append((start, stop))
    if header_bytes + end != file_bytes:
        raise ValueError("header offsets do not cover the input file")
    return kept, expert_bytes


def project(source, rank, destination=None, expected=None):
    size = source.stat().st_size
    with source.open("rb") as fp:
        raw, fields = header(fp)
        if expected and (expected["indexed_bytes"] != size or
                         expected["header_payload_sha256"] != hashlib.sha256(raw[8:]).hexdigest()):
            raise ValueError(f"{source.name} differs from the pinned index")
        kept, expert_bytes = ranges(fields, rank, len(raw), size)
        result = dict(file=source.name, rank=rank, logical_bytes=size,
                      header_sha256=hashlib.sha256(raw).hexdigest(),
                      expert_bytes=expert_bytes, ranges=len(kept),
                      retained_bytes=sum(end - start for start, end in kept))
        if destination is None:
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        # Publish only after the sparse file is complete. An existing result is never replaced.
        with tempfile.NamedTemporaryFile(dir=destination.parent) as temp:
            temp.truncate(size)
            digest = hashlib.sha256()
            for start, end in kept:
                fp.seek(start)
                temp.seek(start)
                remaining = end - start
                while remaining:
                    part = fp.read(min(CHUNK, remaining))
                    if not part:
                        raise ValueError(f"truncated source {source.name}")
                    temp.write(part)
                    digest.update(part)
                    remaining -= len(part)
            temp.flush()
            os.link(temp.name, destination)
        result.update(physical_bytes=destination.stat().st_blocks * 512,
                      retained_sha256=digest.hexdigest())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--rank", type=int, choices=range(RANKS), required=True)
    parser.add_argument("--file", help="limit the operation to one indexed shard")
    parser.add_argument("--out", type=Path, help="write sparse shards into a new rank directory")
    args = parser.parse_args()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_bytes())
    files = sorted(set(index["weight_map"].values()))
    plan = json.loads(Path(__file__).with_name("plan.json").read_bytes())[args.rank]
    by_name = {row["file"]: row for row in plan["files"]}
    if plan["rank"] != args.rank or set(files) != set(by_name):
        parser.error("checkpoint index does not match the pinned shard names")
    if len(files) != 24 or any(Path(name).name != name for name in files):
        parser.error("expected the 24 indexed Inkling 2 Small text/model shards")
    if args.file and args.file not in files:
        parser.error("requested file is absent from the checkpoint index")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        marker = args.out / "labski-rank.json"
        description = {"model": "thinkingmachines/Inkling-2-Small", "revision": REVISION, "rank": args.rank, "ranks": RANKS}
        if marker.exists():
            if json.loads(marker.read_bytes()) != description:
                parser.error("output directory belongs to a different checkpoint or rank")
        else:
            if any(args.out.iterdir()):
                parser.error("choose an empty rank directory")
            with marker.open("x") as f:
                f.write(json.dumps(description) + "\n")
    for name in [args.file] if args.file else files:
        result = project(args.snapshot / name, args.rank, args.out / name if args.out else None, by_name[name])
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
