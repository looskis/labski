"""Extract vLLM capacity and time-align its periodic scheduler samples with client trials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

STATUS = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
CAPACITY = re.compile(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
PAGE = re.compile(r"Setting attention block size to (\d+) tokens")


def extract(log, trials):
    first = datetime.fromtimestamp(min(x["start_unix_ns"] for x in trials) / 1e9, tz=timezone.utc)
    cap = CAPACITY.search(log)
    page = PAGE.search(log)
    if not cap or not page:
        raise ValueError("missing capacity or page-size record in engine log")
    samples = []
    for stamp, active, waiting, usage in STATUS.findall(log):
        observed = datetime.strptime(f"{first.year}-{stamp}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        ns = int(observed.timestamp()) * 10**9
        # The engine log truncates seconds. Require the entire second to fall inside a trial.
        for trial in trials:
            if trial["start_unix_ns"] <= ns and ns + 10**9 <= trial["end_unix_ns"]:
                samples.append(dict(utc=observed.isoformat(), clients=trial["concurrency"],
                                    repeat=trial["repeat"], running=int(active), waiting=int(waiting),
                                    kv_used_percent=float(usage)))
                break
    return dict(cache_tokens=int(cap[1].replace(",", "")), max_context=int(cap[2].replace(",", "")),
                max_context_concurrency=float(cap[3]), attention_block_tokens=int(page[1]), samples=samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path, help="vLLM server log in UTC")
    parser.add_argument("trials", type=Path, help="trials.jsonl from the client")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.trials.read_text().splitlines()]
    print(json.dumps(extract(args.server_log.read_text(), records), indent=2))
