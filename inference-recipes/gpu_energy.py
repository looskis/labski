"""Sample the GB10's cumulative NVML energy counter, or join rank traces to a timed trial."""
import argparse
import bisect
import csv
import ctypes
import json
from pathlib import Path
import time


def check(status):
    if status:
        raise RuntimeError(f"NVML call failed with return code {status}")


def sample(args):
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    check(nvml.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        check(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)))
        counter = ctypes.c_ulonglong()
        deadline = time.monotonic() + args.seconds
        with args.out.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unix_ns", "energy_mj"])
            while time.monotonic() < deadline:
                before = time.time_ns()
                check(nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(counter)))
                writer.writerow([(before + time.time_ns()) // 2, counter.value])
                stream.flush()
                time.sleep(args.interval_ms / 1000)
    finally:
        check(nvml.nvmlShutdown())


def energy_delta(path, start, end):
    with path.open(newline="") as stream:
        samples = [(int(row["unix_ns"]), int(row["energy_mj"])) for row in csv.DictReader(stream)]
    times = [timestamp for timestamp, _ in samples]
    if not times or start < times[0] or end > times[-1] or end <= start:
        raise ValueError(f"trial falls outside {path}")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"clock moved backwards in {path}")
    window = samples[max(bisect.bisect_right(times, start) - 1, 0):bisect.bisect_left(times, end) + 1]
    for (ta, ea), (tb, eb) in zip(window, window[1:]):
        if eb < ea or tb - ta > 1_000_000_000:
            raise ValueError(f"counter reset or sample gap above one second in {path}")

    def interpolate(t):
        right = bisect.bisect_left(times, t)
        tb, eb = samples[right]
        if t == tb:
            return eb
        ta, ea = samples[right - 1]
        return ea + (eb - ea) * (t - ta) / (tb - ta)

    return (interpolate(end) - interpolate(start)) / 1000


def join(args):
    trial = json.loads(args.trial.read_text())
    result = trial["summary"]
    if result["failed"] or not result["completed"] or len(trial["requests"]) != result["completed"]:
        raise ValueError("a complete trial is required")
    tokens = sum(row["output_tokens"] for row in trial["requests"])
    if tokens <= 0 or len({p.resolve() for p in args.traces}) != len(args.traces):
        raise ValueError("supply one distinct trace per participating Spark and a positive token count")
    start, end = result["start_unix_ns"], result["end_unix_ns"]
    if abs((end - start) / 1e9 - result["wall_s"]) > 0.01:
        raise ValueError("wall clock and monotonic trial duration differ by more than 10 ms")
    joules = [energy_delta(path, start, end) for path in args.traces]
    if any(energy <= 0 for energy in joules):
        raise ValueError("a rank has no positive energy increment")
    print(json.dumps(dict(rank_joules=joules, output_tokens=tokens,
                          total_joules=sum(joules), joules_per_output_token=sum(joules) / tokens)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    record = sub.add_parser("sample")
    record.add_argument("--out", required=True, type=Path)
    record.add_argument("--seconds", type=float, required=True)
    record.add_argument("--interval-ms", type=int, default=200)
    report = sub.add_parser("join")
    report.add_argument("trial", type=Path)
    report.add_argument("traces", type=Path, nargs="+")
    args = parser.parse_args()
    if args.mode == "sample":
        if args.seconds <= 0 or args.interval_ms <= 0:
            parser.error("sampling duration and interval must be positive")
        sample(args)
    else:
        join(args)
