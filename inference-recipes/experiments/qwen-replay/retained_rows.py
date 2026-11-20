"""Retain exact immutable PLE row bytes across gathers in one B12X reader."""
from array import array
from collections import OrderedDict
import json
import os
from pathlib import Path
import time


class RetainedRows:
    def __init__(self, reader, budget):
        self.capacity = budget // (reader.weight_row_bytes + reader.scale_row_bytes)
        self.entries = OrderedDict()
        self.hits = self.distinct = 0

    def gather(self, reader, count):
        ids = reader._ids_buffer[:count].tolist()
        distinct = dict.fromkeys(ids)
        resolved = {i: self.entries[i] for i in distinct if i in self.entries}
        missing = [i for i in distinct if i not in resolved]
        self.hits += len(resolved)
        self.distinct += len(distinct)
        w, s = reader.weight_row_bytes, reader.scale_row_bytes
        if missing:
            weights = memoryview(bytearray(w * len(missing)))
            scales = memoryview(bytearray(s * len(missing))) if s else None
            reader._native.ple_reader_run(reader._reader, array("q", missing), weights, scales, len(missing))
            for pos, row in enumerate(missing):
                resolved[row] = (bytes(weights[pos*w:(pos+1)*w]),
                                 bytes(scales[pos*s:(pos+1)*s]) if s else b"")
        output = reader._weight_buffer.cast("B")
        scales_out = reader._scale_buffer.cast("B") if s else None
        for pos, row in enumerate(ids):
            packed, scale = resolved[row]
            output[pos*w:(pos+1)*w] = packed
            if s:
                scales_out[pos*s:(pos+1)*s] = scale
        for row in distinct:
            if reader.shard_start <= row < reader.shard_end:
                self.entries[row] = resolved[row]
                self.entries.move_to_end(row)
                if len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
        return bool(missing)


def rss_bytes():
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def install(reader_class, directory):
    directory = Path(directory)
    control = directory / "control"
    original = reader_class._read_staged

    def gather(reader, count):
        if reader._gds is not None:
            raise ValueError("this experiment requires the pinned io_uring reader")
        previous = getattr(reader, "_labski_epoch", None)
        try:
            info = control.stat()
            stamp = (info.st_ino, info.st_mtime_ns)
        except FileNotFoundError:
            stamp = None
        if previous is None or previous["inode"] != stamp:
            if previous and previous["mode"] != "idle":
                cache = previous["cache"]
                stats = previous["stats"]
                stats.update(resident_bytes_at_end=rss_bytes(), payload_bytes_at_end=0)
                if cache:
                    stats.update(hits=cache.hits, distinct=cache.distinct,
                                 payload_bytes_at_end=len(cache.entries)*(reader.weight_row_bytes+reader.scale_row_bytes))
                path = directory / (previous["label"] + "-reader-" + str(os.getpid()) + ".json")
                path.write_text(json.dumps(stats) + "\n")
            mode, label = control.read_text().split() if stamp is not None else ("idle", "idle")
            if mode not in ("idle", "stock", "cache64", "verify64") or not label.replace("-", "").isalnum():
                raise ValueError("unsupported PLE experiment control")
            previous = reader._labski_epoch = dict(
                inode=stamp, mode=mode, label=label,
                cache=RetainedRows(reader, 64*1024**2) if mode in ("cache64", "verify64") else None,
                stats=dict(calls=0, read_bytes=0, read_calls=0, unique_blocks=0,
                           nanoseconds=0, verified_rows=0, resident_bytes_at_start=rss_bytes()))
        mode = previous["mode"]
        if mode == "idle":
            return original(reader, count)
        stats = previous["stats"]
        begin = time.perf_counter_ns()
        if mode == "stock":
            original(reader, count)
            ran = True
        else:
            # The caller holds the reader transaction lock. The event covers both ID
            # production and completion of the prior GPU consumer of mapped outputs.
            reader._ids_ready.synchronize()
            ran = previous["cache"].gather(reader, count)
        if ran:
            counters = reader._native.ple_reader_stats(reader._reader)
            for key in ("read_bytes", "read_calls", "unique_blocks"):
                stats[key] += counters[key]
        stats["calls"] += 1
        stats["nanoseconds"] += time.perf_counter_ns() - begin
        if mode == "verify64":
            w, s = reader.weight_row_bytes * count, reader.scale_row_bytes * count
            weights = reader._weight_buffer.cast("B")[:w]
            scales = reader._scale_buffer.cast("B")[:s] if s else b""
            saved = bytes(weights), bytes(scales)
            original(reader, count)
            if saved != (bytes(weights), bytes(scales)):
                raise RuntimeError("cached PLE gather differs from the full native gather")
            stats["verified_rows"] += count
    reader_class._read_staged = gather
