"""Fetch one rank's Step-5 expert tensors from the pinned Hub files using HTTP ranges."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.error
import urllib.request

from partition import BUFFER, EXPERTS, RANKS, REVISION, TENSOR, open_header, partition_layer

HUB = f"https://huggingface.co/0xSero/Step-5-Preview-Spark/resolve/{REVISION}"
HEADER_PREFIX = 1 << 20


def ranges(fields, rank, header_bytes, gap):
    spans = []
    for name, info in sorted(fields.items(), key=lambda item: item[1]["data_offsets"][0]):
        match = TENSOR.fullmatch(name)
        if not match:
            raise ValueError(f"unexpected tensor {name}")
        if int(match[2]) // (EXPERTS // RANKS) != rank:
            continue
        start, end = (header_bytes + pos for pos in info["data_offsets"])
        start = max(start, HEADER_PREFIX)
        if start >= end:
            continue
        if spans and start - spans[-1][1] <= gap:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def fetch_range(url, start, end, target, size=None):
    """Write [start, end) at its original file offset. Reject full-file responses."""
    interval = f"{start}-{end - 1}"
    request = urllib.request.Request(url + "?labski_range=" + interval,
                                     headers={"Range": "bytes=" + interval, "Accept-Encoding": "identity"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match:
                    raise ValueError("the server did not honor the byte range")
                first, last, total = map(int, match.groups())
                if (first, last) != (start, end - 1) or total < end or (size is not None and total != size):
                    raise ValueError("wrong byte range or source file size")
                left = end - start
                target.seek(start)
                while left:
                    block = response.read(min(left, BUFFER))
                    if not block:
                        raise OSError("short HTTP range")
                    target.write(block)
                    left -= len(block)
                return total
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def check_file(path, row):
    if path.stat().st_size != row["bytes"]:
        return False
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(BUFFER), b""):
            h.update(block)
    return h.hexdigest() == row["sha256"]


def fetch_layer(destination, row, gap):
    rank, name = row["rank"], row["file"]
    layer = int(re.fullmatch(r"exl3/experts/L(\d{2})\.safetensors", name)[1])
    final = destination / f"rank{rank}" / name
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if not check_file(final, row):
            raise ValueError(f"existing file fails verification: {final}")
        return dict(rank=rank, file=name, reused=True)
    url = HUB + "/" + name
    with tempfile.TemporaryDirectory(dir=final.parent, prefix=".fetch-") as d:
        tmp = Path(d)
        sparse = tmp / "source" / name
        sparse.parent.mkdir(parents=True)
        with sparse.open("w+b") as out:
            total = fetch_range(url, 0, HEADER_PREFIX, out)
            out.truncate(total)
            out.seek(0)
            header, fields = open_header(out)
            if len(header) > HEADER_PREFIX:
                raise ValueError("the prefix did not cover the source header")
            fields.pop("__metadata__", None)
            spans = ranges(fields, rank, len(header), gap)
            for start, end in spans:
                fetch_range(url, start, end, out, total)
        partition_layer(tmp / "source", layer, None, tmp, (rank,), {(rank, name): row})
        # Concurrent fetches can only replace this path with an identical verified file.
        os.replace(tmp / f"rank{rank}" / name, final)
    return dict(rank=rank, file=name, requests=1 + len(spans), source_file_bytes=total,
                range_body_bytes=HEADER_PREFIX + sum(end - start for start, end in spans),
                output_file_bytes=row["bytes"], sha256=row["sha256"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--rank", required=True, type=int, choices=range(RANKS))
    cli.add_argument("--layer", action="append", type=int, choices=range(3, 91),
                     help="restrict the fetch to selected MoE layers (default all 88)")
    cli.add_argument("--merge-gap", type=int, default=1 << 20,
                     help="fetch through gaps up to this many bytes (default 1 MiB)")
    args = cli.parse_args()
    if args.merge_gap < 0:
        cli.error("merge gap must be nonnegative")
    layout = json.loads(Path(__file__).with_name("layout.json").read_bytes())
    if layout["revision"] != REVISION or layout["world_size"] != RANKS:
        raise ValueError("unexpected reference layout")
    selected = {f"exl3/experts/L{n:02d}.safetensors" for n in args.layer} if args.layer else None
    for row in layout["files"]:
        if row["rank"] == args.rank and (selected is None or row["file"] in selected):
            print(json.dumps(fetch_layer(args.destination, row, args.merge_gap)), flush=True)
