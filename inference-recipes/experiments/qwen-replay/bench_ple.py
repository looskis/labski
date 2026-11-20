"""Verify every packed byte, then compare stock and a cold 64 MiB row cache in ABBA order."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import replay


def benchmark(args):
    directory = args.directory
    control = directory / "control"
    def select(mode, label):
        pending = control.with_suffix(".next")
        pending.write_text(f"{mode} {label}\n")
        pending.replace(control)

    schedule = ("trace",) if args.trace else ("verify64", "stock", "cache64", "cache64", "stock")
    marker = directory / "active"
    for block, mode in enumerate(schedule):
        for clients in (1, 4):
            for kind in ("edits", "prose"):
                label = f"b{block}-{mode}-{kind}-c{clients}"
                out = directory / (label + ".jsonl")
                params = SimpleNamespace(workload=kind, clients=clients, model="qwen38-flash-next-ple-ab",
                                         url=args.url, out=out)
                select("idle", "idle")
                marker.unlink(missing_ok=True)
                def begin():
                    if args.trace:
                        marker.write_text(label)
                    else:
                        select(mode, label)
                    device["before"] = args.device_stat.read_text().split()
                device = {}
                try:
                    replay.run(params, after_warmup=begin)
                finally:
                    marker.unlink(missing_ok=True)
                device["after"] = args.device_stat.read_text().split()
                (directory / (label + "-device.json")).write_text(json.dumps(device) + "\n")
                print(label, flush=True)
    select("idle", "idle")
    replay.request(params, 0, 0, warmup=True)  # Flush the final reader receipt.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trace", action="store_true", help="run the gather observer once per workload")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--device-stat", type=Path, default=Path("/sys/class/block/nvme0n1/stat"))
    benchmark(parser.parse_args())
