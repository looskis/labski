"""Observe synthetic PLE gathers in the pinned FP4-Local image after its reader returns."""
import hashlib
import json
import os
from pathlib import Path
import time


def install(reader_class, directory):
    directory = Path(directory)
    original = reader_class._read_staged
    seen = {}
    last_label = None

    def traced(reader, count):
        nonlocal last_label
        marker = directory / "active"
        try:
            label = marker.read_text().strip()
        except FileNotFoundError:
            return original(reader, count)
        if not label or not label.replace("-", "").isalnum():
            raise ValueError("invalid PLE trace label")
        if label != last_label:
            seen.clear()
            last_label = label
        start = time.perf_counter_ns()
        original(reader, count)
        finished = time.perf_counter_ns()
        ids = reader._ids_buffer[:count].tolist()
        counters = reader._native.ple_reader_stats(reader._reader)
        size, scale_size = reader.weight_row_bytes, reader.scale_row_bytes
        weights = reader._weight_buffer.cast("B")[:count * size]
        scales = reader._scale_buffer.cast("B")[:count * scale_size] if scale_size else b""
        compared = mismatches = 0
        for i, row_id in enumerate(ids):
            if not reader.shard_start <= row_id < reader.shard_end:
                continue
            packed = bytes(weights[i*size:(i+1)*size]) + bytes(scales[i*scale_size:(i+1)*scale_size])
            previous = seen.setdefault(row_id, packed)
            if previous is not packed:
                compared += 1
                mismatches += int(previous != packed)
        batch = hashlib.sha256(weights)
        batch.update(scales)
        row = dict(nanoseconds=finished-start, ids=ids, weight_bytes=size, scale_bytes=scale_size,
                   shard_start=reader.shard_start, shard_end=reader.shard_end,
                   read_bytes=counters["read_bytes"], unique_blocks=counters["unique_blocks"],
                   read_calls=counters["read_calls"], batch_sha256=batch.hexdigest(),
                   repeated_rows_compared=compared, byte_mismatches=mismatches)
        with (directory / f"{label}-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row,separators=(",",":"))+"\n")
    reader_class._read_staged = traced
