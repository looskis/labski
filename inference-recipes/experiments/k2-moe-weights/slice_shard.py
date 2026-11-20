"""Copy the common tensors and one contiguous expert partition into a compact K2 shard."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from inventory import EXPERT


def read_header(fp):
    prefix = fp.read(8)
    size, = struct.unpack("<Q", prefix)
    if not 2 <= size <= 100_000_000:
        raise ValueError("invalid SafeTensors header length")
    raw = fp.read(size)
    if len(raw) != size:
        raise ValueError("truncated SafeTensors header")
    return prefix + raw, json.loads(raw)


def write_shard(source, destination, ranks, rank):
    if ranks not in (2, 4) or not 0 <= rank < ranks:
        raise ValueError("choose a valid two- or four-way expert rank")
    catalog = json.loads(Path(__file__).with_name("375b-nvfp4.json").read_bytes())["indexed_files"]
    expected = next((r for r in catalog if r["file"] == source.name), None)
    if not expected or source.stat().st_size != expected["file_bytes"]:
        raise ValueError("input is absent from the pinned K2 file inventory")
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src:
        raw, fields = read_header(src)
        if hashlib.sha256(raw).hexdigest() != expected["header_sha256"]:
            raise ValueError("source header differs from the pinned revision")
        compact = {}
        if "__metadata__" in fields:
            compact["__metadata__"] = fields.pop("__metadata__")
        spans, expert_count, common_count, size = [], 0, 0, 0
        for name, f in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
            match = EXPERT.fullmatch(name)
            if match:
                if int(match[2]) // (192 // ranks) != rank:
                    continue
                expert_count += 1
            else:
                if ".mlp.experts." in name:
                    raise ValueError(f"unrecognized expert tensor {name}")
                common_count += 1
            start, end = f["data_offsets"]
            length = end - start
            compact[name] = dict(f, data_offsets=[size, size + length])
            size += length
            if spans and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=destination.parent) as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for start, end in spans:
                src.seek(len(raw) + start)
                remaining = end - start
                while remaining:
                    block = src.read(min(8 << 20, remaining))
                    if not block:
                        raise ValueError("source ended inside a retained tensor")
                    out.write(block)
                    digest.update(block)
                    remaining -= len(block)
            out.flush()
            os.link(out.name, destination)
    return dict(file=source.name, ranks=ranks, rank=rank, source_bytes=source.stat().st_size,
                packed_bytes=destination.stat().st_size, payload_bytes=size,
                common_tensors=common_count, routed_tensors=expert_count,
                payload_sha256=digest.hexdigest(), input_header_sha256=expected["header_sha256"])


def verify_shard(source, destination, ranks, rank):
    from safetensors import safe_open

    if ranks not in (2, 4) or rank not in range(ranks):
        raise ValueError("choose a valid two- or four-way expert rank")
    with source.open("rb") as src, destination.open("rb") as dst:
        src_header, src_fields = read_header(src)
        dst_header, dst_fields = read_header(dst)
        dst_fields.pop("__metadata__", None)
        digest, total = hashlib.sha256(), 0
        with safe_open(source, framework="numpy") as original, safe_open(destination, framework="numpy") as packed:
            expected = set()
            for name in original.keys():
                fields = name.split(".")
                if "experts" not in fields or int(fields[fields.index("experts") + 1]) // (192 // ranks) == rank:
                    expected.add(name)
            if set(packed.keys()) != expected:
                raise ValueError("compact shard contains a different set of tensors than this rank needs")
            for name, field in sorted(dst_fields.items(), key=lambda item: item[1]["data_offsets"]):
                first, second = original.get_slice(name), packed.get_slice(name)
                a, b = src_fields[name]["data_offsets"]
                x, y = field["data_offsets"]
                if first.get_shape() != second.get_shape() or first.get_dtype() != second.get_dtype() or b - a != y - x:
                    raise ValueError(f"tensor layout changed for {name}")
                src.seek(len(src_header) + a)
                dst.seek(len(dst_header) + x)
                remaining = y - x
                while remaining:
                    left = src.read(min(8 << 20, remaining))
                    right = dst.read(len(left))
                    if not left or left != right:
                        raise ValueError(f"tensor contents differ for {name}")
                    digest.update(right)
                    remaining -= len(right)
                    total += len(right)
    return dict(file=source.name, ranks=ranks, rank=rank, tensors=len(expected),
                payload_bytes=total, payload_sha256=digest.hexdigest())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--ranks", type=int, choices=(2, 4), required=True)
    cli.add_argument("--rank", type=int, required=True)
    cli.add_argument("--verify-existing", action="store_true",
                     help="check an existing compact shard against its source using the SafeTensors reader")
    args = cli.parse_args()
    operation = verify_shard if args.verify_existing else write_shard
    print(json.dumps(operation(args.source, args.out, args.ranks, args.rank), indent=2))
