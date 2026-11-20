"""Measure an exact PLE cache on long inventory lookups with known answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bench


def inventory(kind, turn, session):
    seed = 1009 + session + (4*turn if kind == "fresh" else 0)
    rng = random.Random(seed)
    values = rng.sample(range(100000, 1000000), 256)
    target = (43 + 79*turn + 17*session) % len(values)
    data = "\n".join(f"bin-{i:04d} = {val}" for i, val in enumerate(values))
    return f"Return only the value of bin-{target:04d}.\n{data}", str(values[target])


def benchmark(args):
    def select(mode, label):
        pending = args.directory / "control.next"
        pending.write_text(f"{mode} {label}\n")
        pending.replace(args.directory / "control")

    def send(prompt):
        return bench.stream(args.url, "qwen38-flash-next-ple-ab", prompt, 16, 300, "off")

    def request(kind, turn, session):
        prompt, expected = inventory(kind, turn, session)
        row = send(prompt)
        if row["first_text_s"] is None:
            raise ValueError("missing first visible text")
        return dict(**row, turn=turn, session=session, expected=expected,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    correct=row["finish"]=="stop" and bench.matches(row["answer"], expected))

    for block, mode in enumerate(("verify64", "stock", "cache64", "cache64", "stock")):
        for clients in (1,4):
            for kind in ("repeated", "fresh"):
                label = f"b{block}-{mode}-{kind}-c{clients}"
                select("idle", "idle")
                send("Reply with just the word ready.")
                path = args.directory / (label + ".jsonl")
                with path.open("x") as out, ThreadPoolExecutor(clients) as pool:
                    select(mode, label)
                    before = args.device_stat.read_text().split()
                    for turn in range(3):
                        start_ns = time.time_ns()
                        begin = time.perf_counter()
                        rows = list(pool.map(lambda session: request(kind, turn, session), range(4)))
                        elapsed = time.perf_counter()-begin
                        end_ns = time.time_ns()
                        record = dict(block=block, mode=mode, clients=clients, workload=kind, turn=turn,
                                      wall_s=elapsed, start_unix_ns=start_ns, end_unix_ns=end_ns, requests=rows)
                        out.write(json.dumps(record)+"\n")
                        out.flush()
                    after = args.device_stat.read_text().split()
                (args.directory / (label+"-device.json")).write_text(json.dumps(dict(before=before, after=after))+"\n")
                print(label, flush=True)
    select("idle", "idle")
    send("Reply with just the word ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--device-stat", type=Path, default=Path("/sys/class/block/nvme0n1/stat"))
    benchmark(parser.parse_args())
