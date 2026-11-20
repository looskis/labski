"""Build or verify a complete two-rank K2 Horizon snapshot, one Spark directory at a time."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from inventory import REVISION
from slice_shard import read_header, write_shard


def file_hash(path, git_blob=False):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode()) if git_blob else hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, record):
    if path.stat().st_size != record["packed_bytes"] or file_hash(path) != record["packed_file_sha256"]:
        raise ValueError(f"compact shard differs from the published rank {record['rank']} file: {path.name}")


def publish_bytes(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing file differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent) as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        try:
            os.link(f.name, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"concurrent write differs: {path}")


def encoded(obj):
    return (json.dumps(obj, indent=2) + "\n").encode()


def assemble(source, out, rank):
    proof = json.loads(Path(__file__).with_name("two-rank-parity.json").read_bytes())
    if proof["revision"] != REVISION or rank not in (0, 1) or source.resolve() == out.resolve():
        raise ValueError("use a separate destination and a valid two-way expert rank")
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "model.safetensors.index.json").read_bytes())["weight_map"]
    records = [r for r in proof["shards"] if r["rank"] == rank]
    expected_files = {row["file"] for row in records}
    if len(records) != 12 or expected_files != set(original.values()):
        raise ValueError("source index does not match the 12 verified shards")
    extra_files = {p.name for p in out.glob("*.safetensors")} - expected_files
    if extra_files:
        raise ValueError(f"unrelated weight files in the destination: {sorted(extra_files)}")
    size, elements, mapping = 0, 0, {}
    for record in records:
        name = record["file"]
        destination = out / name
        if destination.exists():
            verify_file(destination, record)
        else:
            with tempfile.TemporaryDirectory(dir=out) as work:
                staged = Path(work) / name
                write_shard(source / name, staged, 2, rank)
                verify_file(staged, record)
                try:
                    os.link(staged, destination)
                except FileExistsError:
                    verify_file(destination, record)
        with destination.open("rb") as fp:
            _, fields = read_header(fp)
        fields.pop("__metadata__", None)
        for key, field in fields.items():
            if key in mapping or original.get(key) != name:
                raise ValueError(f"unexpected tensor {key} in {name}")
            mapping[key] = name
            start, end = field["data_offsets"]
            size += end - start
            elements += math.prod(field["shape"])
    for asset in proof["assets"]:
        path = source / asset["file"]
        if path.stat().st_size != asset["bytes"] or file_hash(path, asset["algorithm"] == "git-blob-sha1") != asset["digest"]:
            raise ValueError(f"source asset differs: {path.name}")
        publish_bytes(out / path.name, path.read_bytes())
    index = encoded(dict(metadata=dict(total_size=size, total_parameters=elements), weight_map=mapping))
    if hashlib.sha256(index).hexdigest() != proof["rank_indexes"][str(rank)]["sha256"]:
        raise ValueError("generated index differs from the measured compact layout")
    publish_bytes(out / "model.safetensors.index.json", index)
    publish_bytes(out / "labski-expert-placement.json", encoded(dict(model=proof["model"], revision=REVISION,
                  rank=rank, ranks=2, first_expert=rank * 96, experts=96, index_tensors=len(mapping),
                  index_tensor_bytes=size)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--rank", required=True, choices=(0, 1), type=int)
    args = cli.parse_args()
    assemble(args.source, args.destination, args.rank)
