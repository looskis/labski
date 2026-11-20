"""Restore one official MoVA BF16 shard, reusing compatible tensors from its FP8 sibling."""

import argparse
import fcntl
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

CATALOG = Path(__file__).with_name("mova-shards.json")
BLOCK = 64 << 20


def http_range(url, start, end, size):
    for attempt in range(5):
        separator = "&" if "?" in url else "?"
        request = urllib.request.Request(f"{url}{separator}download=true&range={start}-{end}",
            headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=90) as reply:
                if reply.status != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{size}":
                    raise ValueError("server returned an unexpected range")
                data = reply.read(end - start + 1)
            if len(data) != end - start:
                raise OSError("truncated response body")
            return data
        except (OSError, IncompleteRead):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def header_size(prefix):
    length, = struct.unpack("<Q", prefix)
    if not 2 <= length <= 100_000_000:
        raise ValueError("invalid SafeTensors header size")
    return length


def offsets(header, size):
    fields = json.loads(header[8:])
    fields.pop("__metadata__", None)
    end = 0
    for field in sorted(fields.values(), key=lambda f: f["data_offsets"]):
        start, stop = field["data_offsets"]
        if start != end or stop < start:
            raise ValueError("noncontiguous tensor layout")
        end = stop
    if end + len(header) != size:
        raise ValueError("header and file sizes disagree")
    return fields


def spans(target, source, target_header, source_header):
    segments = []
    for key, field in sorted(target.items(), key=lambda item: item[1]["data_offsets"]):
        a, b = field["data_offsets"]
        output = target_header + a
        sibling = source.get(key)
        local = bool(sibling and field["dtype"] == sibling["dtype"] == "BF16" and
                     field["shape"] == sibling["shape"] and b - a == sibling["data_offsets"][1] - sibling["data_offsets"][0])
        origin = source_header + sibling["data_offsets"][0] if local else output
        if segments and segments[-1][0] == local and origin == segments[-1][1] + segments[-1][2]:
            segments[-1][2] += b - a
        else:
            segments.append([local, origin, b - a])
    return segments


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def restore(sibling, directory, bf16_shard_url=None):
    state = directory / '.mova-reuse'
    state.mkdir(parents=True, exist_ok=True)
    with (state / (sibling.name + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return restore_locked(sibling, directory, state / (sibling.name + '.part'), bf16_shard_url)


def restore_locked(sibling, directory, partial, bf16_shard_url):
    catalog = json.loads(CATALOG.read_bytes())
    record = next((r for r in catalog["shards"] if r["file"] == sibling.name), None)
    if not record or sibling.stat().st_size != record["fp8_bytes"]:
        raise ValueError("expected a shard from the pinned FP8 checkpoint")
    out = directory / sibling.name
    if out.exists():
        if digest(out) != record["bf16_sha256"]:
            raise ValueError(f"existing BF16 destination has unexpected contents: {out.name}")
        partial.unlink(missing_ok=True)
        return dict(file=out.name, already_verified=True)
    url = bf16_shard_url or f"https://huggingface.co/{catalog['bf16']}/resolve/{catalog['bf16_revision']}/{out.name}"
    size = record["bf16_bytes"]
    prefix = http_range(url, 0, 8, size)
    head = prefix + http_range(url, 8, 8 + header_size(prefix), size)
    if hashlib.sha256(head).hexdigest() != record["bf16_header_sha256"]:
        raise ValueError("unexpected BF16 header")
    source_bytes = 0
    response_bytes, requests = len(head), 2
    initial = partial.stat().st_size if partial.exists() else 0
    if initial > size:
        raise ValueError("partial BF16 file exceeds the expected size")
    with sibling.open("rb") as src, partial.open("a+b") as dest:
        fp8_prefix = src.read(8)
        fp8_head = fp8_prefix + src.read(header_size(fp8_prefix))
        if hashlib.sha256(fp8_head).hexdigest() != record["fp8_header_sha256"]:
            raise ValueError("unexpected FP8 header")
        storage = spans(offsets(head, size), offsets(fp8_head, record["fp8_bytes"]), len(head), len(fp8_head))
        dest.seek(0)
        if dest.read(min(initial, len(head))) != head[:initial]:
            raise ValueError("partial BF16 header differs from the pinned source")
        dest.seek(0, os.SEEK_END)
        if initial < len(head):
            dest.write(head[initial:])
        position = len(head)
        for local, start, length in storage:
            skip = min(length, max(0, initial - position))
            position += length
            if local:
                src.seek(start + skip)
            for offset in range(skip, length, BLOCK):
                count = min(length - offset, BLOCK)
                if local:
                    data = src.read(count)
                    if len(data) != count:
                        raise OSError("truncated local tensor")
                    source_bytes += count
                else:
                    data = http_range(url, start + offset, start + offset + count, size)
                    response_bytes += count
                    requests += 1
                dest.write(data)
                dest.flush()
        os.fsync(dest.fileno())
    if partial.stat().st_size != size or digest(partial) != record["bf16_sha256"]:
        raise ValueError(f"publisher SHA-256 mismatch; retained partial file at {partial}")
    try:
        os.link(partial, out)
    except FileExistsError:
        if digest(out) != record["bf16_sha256"]:
            raise ValueError("concurrent write differs from the verified shard")
    partial.unlink()
    return dict(file=out.name, bf16_bytes=size, fp8_tensor_bytes_reused=source_bytes,
                preserved_bf16_bytes=initial, bf16_response_bytes=response_bytes, http_ranges=requests,
                bf16_sha256=record["bf16_sha256"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fp8_shard", type=Path)
    cli.add_argument("bf16_directory", type=Path)
    cli.add_argument("--bf16-shard-url", help="read the pinned BF16 shard from a peer supporting HTTP Range")
    args = cli.parse_args()
    print(json.dumps(restore(args.fp8_shard, args.bf16_directory, args.bf16_shard_url), indent=2))
