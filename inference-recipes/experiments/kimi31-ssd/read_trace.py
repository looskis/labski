"""Measure direct NVMe reads of 16 distinct Kimi experts per simulated routing step."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import statistics
import threading
import time

EXPERT_BYTES = 17547264
ORDERS = ((1, 4, 8, 16), (16, 8, 4, 1), (4, 1, 16, 8), (8, 16, 1, 4))


def process_read_bytes():
    return int(next(line.split()[1] for line in Path("/proc/self/io").read_text().splitlines()
                    if line.startswith("read_bytes:")))


def trial(paths, workers, trace, warmup, disk_stat):
    buffers = threading.local()

    def read_file(index):
        if not hasattr(buffers, "data"):
            buffers.data = mmap.mmap(-1, EXPERT_BYTES)
        fd = os.open(paths[index], os.O_RDONLY | os.O_DIRECT)
        try:
            if os.preadv(fd, [buffers.data], 0) != EXPERT_BYTES:
                raise OSError(f"short expert read from {paths[index].name}")
            return buffers.data[:8] + buffers.data[-8:]
        finally:
            os.close(fd)

    def sectors():
        return int(disk_stat.read_text().split()[2])

    fingerprint = hashlib.sha256()
    steps = []
    with ThreadPoolExecutor(workers) as pool:
        for indices in warmup:
            list(pool.map(read_file, indices))
        before = process_read_bytes()
        sector0 = sectors()
        started = time.perf_counter()
        for indices in trace:
            start = time.perf_counter()
            edges = list(pool.map(read_file, indices))
            steps.append(1000 * (time.perf_counter() - start))
            fingerprint.update(b"".join(edges))
        elapsed = time.perf_counter() - started
        disk_bytes = 512 * (sectors() - sector0)
        actual = process_read_bytes() - before
    requested = len(trace) * 16 * EXPERT_BYTES
    if actual < requested or disk_bytes < requested:
        raise RuntimeError("physical read counters did not cover requested bytes")
    return dict(workers=workers, elapsed_s=elapsed, requested_bytes=requested, process_read_bytes=actual,
                device_read_bytes=disk_bytes, decimal_GB_per_s=requested / elapsed / 1e9,
                median_step_ms=statistics.median(steps), step_ms=steps, edge_sha256=fingerprint.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expert_dir", type=Path)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", required=True, help="Linux block device containing the expert store")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or Path(args.device).name != args.device:
        parser.error("specify a positive step count and a block device name")
    disk_stat = Path("/sys/class/block") / args.device / "stat"
    paths = [args.expert_dir / f"L{args.layer}-E{i}.bin" for i in range(896)]
    if any(path.stat().st_size != EXPERT_BYTES for path in paths):
        raise ValueError("expected one complete 896-expert layer")
    rng = random.Random(args.seed)
    warmup = [rng.sample(range(896), 16) for _ in range(4)]
    trace = [rng.sample(range(896), 16) for _ in range(args.steps)]
    signature = hashlib.sha256(json.dumps(trace, separators=(",", ":")).encode()).hexdigest()
    ref = None
    with args.out.open("x") as out:
        for repeat, order in enumerate(ORDERS, 1):
            for workers in order:
                result = trial(paths, workers, trace, warmup, disk_stat)
                if ref is None:
                    ref = result["edge_sha256"]
                if result["edge_sha256"] != ref:
                    raise ValueError("sampled expert bytes changed across conditions")
                result.update(repeat=repeat, trace_sha256=signature, layer=args.layer, steps=args.steps, seed=args.seed)
                out.write(json.dumps(result, separators=(",", ":")) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
