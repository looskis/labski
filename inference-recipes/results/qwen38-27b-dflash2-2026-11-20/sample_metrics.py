"""Poll queue, KV and speculative-decoding counters from one vLLM engine."""

import argparse
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

GAUGES = {"num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"}
COUNTERS = {"spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total"}
METRIC = re.compile(r"^vllm:(\w+)(?:\{[^}]*\})?\s+([\d.eE+-]+)$", re.MULTILINE)


def sample(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        body = response.read().decode()
    values = {}
    for name, value in METRIC.findall(body):
        if name in GAUGES or name in COUNTERS:
            if name in values:
                raise ValueError(f"expected a single engine, received multiple samples for {name}")
            values[name] = float(value)
    if not GAUGES.issubset(values):
        raise ValueError(f"missing engine metrics: {sorted(GAUGES.difference(values))}")
    return {"unix_ns": time.time_ns(), **values}


def main(args):
    url = f"http://127.0.0.1:{args.port}/metrics"
    deadline = time.monotonic() + args.startup_timeout
    started = False
    with args.dest.open("x") as stream:
        while not args.done.is_file():
            try:
                row = sample(url)
            except (urllib.error.URLError, OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    if started:
                        return
                    raise
            else:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                started = True
                deadline = time.monotonic() + 8
            time.sleep(args.interval)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dest", type=Path, help="new JSONL output file")
    cli.add_argument("done", type=Path, help="stop polling when this marker exists")
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--interval", type=float, default=0.25, help="seconds between polls")
    cli.add_argument("--startup-timeout", type=float, default=1200)
    args = cli.parse_args()
    if args.interval <= 0 or args.startup_timeout <= 0 or not 1 <= args.port <= 65535:
        cli.error("choose positive timeouts and a valid port")
    main(args)
