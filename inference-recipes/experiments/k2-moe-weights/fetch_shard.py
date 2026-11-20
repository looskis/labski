"""Download only the public Hugging Face ranges needed for one compact K2 shard."""

import argparse
import fcntl
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

from assemble import verify_file
from inventory import EXPERT, REVISION
from slice_shard import write_shard

MODEL = "IFM/K2-Horizon-375B-A23B-NVFP4"
HERE = Path(__file__).resolve().parent
BLOCK = 64 << 20
GAP = 4096


def get_range(name, start, end, size):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true&labski={start}-{end}"
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{size}":
                    raise ValueError(f"unexpected HTTP byte range for {name}")
                body = response.read(end - start + 1)
            if len(body) != end - start:
                raise OSError(f"short response for {name}")
            return body
        except (OSError, IncompleteRead):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def selected_ranges(header, rank):
    fields = json.loads(header[8:])
    fields.pop("__metadata__", None)
    spans = []
    for key, field in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
        expert = EXPERT.fullmatch(key)
        if expert and int(expert[2]) // 96 != rank:
            continue
        if not expert and ".mlp.experts." in key:
            raise ValueError(f"unknown expert tensor {key}")
        start, end = (len(header) + x for x in field["data_offsets"])
        if spans and start - spans[-1][1] <= GAP:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return [(offset, min(offset + BLOCK, end)) for start, end in spans for offset in range(start, end, BLOCK)]


def download(record, meta, out, work, read_range=None):
    if read_range is None:
        read_range = get_range
    name, rank = record["file"], record["rank"]
    final = out / name
    source = work / name
    state = work / "progress.json"
    if final.exists():
        verify_file(final, record)
        source.unlink(missing_ok=True)
        state.unlink(missing_ok=True)
        return dict(file=name, rank=rank, already_verified=True)
    size = record["source_bytes"]
    header = read_range(name, 0, meta["header_bytes"], size)
    if hashlib.sha256(header).hexdigest() != meta["header_sha256"]:
        raise ValueError(f"source header mismatch in {name}")
    groups = selected_ranges(header, rank)
    completed = 0
    if state.exists():
        progress = json.loads(state.read_bytes())
        completed = progress["completed"]
        if progress["header_sha256"] != meta["header_sha256"] or not 0 <= completed <= len(groups):
            raise ValueError("invalid saved range progress")
        with source.open("rb") as fp:
            if source.stat().st_size != size or fp.read(len(header)) != header:
                raise ValueError("partial file differs from its recorded header")
    else:
        with source.open("wb") as fp:
            fp.write(header)
            fp.truncate(size)
            fp.flush()
            os.fsync(fp.fileno())
    with source.open("r+b") as fp:
        for index in range(completed, len(groups)):
            start, end = groups[index]
            fp.seek(start)
            fp.write(read_range(name, start, end, size))
            fp.flush()
            os.fsync(fp.fileno())
            next_state = state.with_suffix(".next")
            next_state.write_text(json.dumps(dict(completed=index + 1, header_sha256=meta["header_sha256"])))
            next_state.replace(state)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        candidate = Path(directory) / name
        write_shard(source, candidate, 2, rank)
        try:
            verify_file(candidate, record)
        except ValueError:
            state.unlink(missing_ok=True)
            raise
        try:
            os.link(candidate, final)
        except FileExistsError:
            verify_file(final, record)
    source.unlink()
    state.unlink(missing_ok=True)
    return dict(file=name, rank=rank, compact_file_bytes=record["packed_bytes"], source_file_bytes=size,
                planned_response_bytes=len(header) + sum(b - a for a, b in groups), planned_ranges=1 + len(groups),
                compact_sha256=record["packed_file_sha256"])


def fetch(name, rank, out, read_range=None):
    proof = json.loads((HERE / "two-rank-parity.json").read_bytes())
    catalog = json.loads((HERE / "375b-nvfp4.json").read_bytes())["indexed_files"]
    record = next((r for r in proof["shards"] if r["rank"] == rank and r["file"] == name), None)
    if record is None:
        raise ValueError("choose a shard and rank in the verified checkpoint")
    meta = next(r for r in catalog if r["file"] == name)
    work = out / ".k2-ranges" / str(rank) / Path(name).stem
    work.mkdir(parents=True, exist_ok=True)
    with (work / "lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return download(record, meta, out, work, read_range)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("shard")
    cli.add_argument("destination", type=Path)
    cli.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = cli.parse_args()
    print(json.dumps(fetch(args.shard, args.rank, args.destination), indent=2))
