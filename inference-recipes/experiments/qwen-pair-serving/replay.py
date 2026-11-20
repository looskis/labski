"""Replay a matched concurrency sweep and quality panel on one or two API endpoints."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bench

MODEL = "qwen38-27b-nvfp4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--corpus", type=Path, help="the lab's pinned 112-item MMLU-Pro panel")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.endpoint) not in (1, 2) or len(set(args.clients)) != len(args.clients) or min(args.clients) < 1:
        parser.error("supply one or two endpoints and distinct positive concurrency levels")
    endpoints = [endpoint.rstrip("/") for endpoint in args.endpoint]
    if len(set(endpoints)) != len(endpoints):
        parser.error("replicas require distinct endpoints")
    subset = []
    if not args.timing_only:
        if args.corpus is None:
            parser.error("the quality panel requires --corpus")
        cases = [json.loads(line) for line in args.corpus.read_text().splitlines()]
        subjects = Counter()
        if len(cases) != 112 or len({row["id"] for row in cases}) != 112:
            parser.error("expected 112 distinct diagnostic questions")
        for row in cases:
            subjects[row["subject"]] += 1
            if subjects[row["subject"]] <= 2:
                subset.append(row)
        if sorted(subjects.values()) != [8] * 14:
            parser.error("expected eight items per subject across 14 subjects")
    args.out.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, str(ROOT / "bench.py"), MODEL]
    for endpoint in endpoints:
        common.extend(["--endpoint", endpoint])
    common.extend(["--thinking", "off", "--records", "32"])

    def measure(workload, clients, requests, repeats, budget, name):
        subprocess.run([*common, "--workload", workload, "--concurrency", str(clients),
                        "--requests", str(requests), "--repeats", str(repeats),
                        "--max-tokens", str(budget), "--out", str(args.out / name)], check=True)

    for clients in args.clients:
        measure("prose", clients, max(4, clients), 3, 96, f"prose-c{clients}")
    if args.timing_only:
        return
    measure("lookup", 4, 8, 1, 128, "lookup")
    with (args.out / "quality.jsonl").open("x") as stream:
        for index, item in enumerate(subset):
            rank = index % len(endpoints)
            row = bench.stream(endpoints[rank], MODEL, item["prompt"], 16, 300, "off")
            digest = hashlib.sha256(item["prompt"].encode()).hexdigest()
            correct = row["finish"] == "stop" and row["answer"].strip() == item["expected"]
            result = dict(id=item["id"], subject=item["subject"], expected=item["expected"],
                          prompt_sha256=digest, endpoint_index=rank, correct=correct, **row)
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
