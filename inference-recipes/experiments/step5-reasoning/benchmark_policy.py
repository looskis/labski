"""Time Step-5's inventory retry client against one native completion per query."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from lookup_policy import lookup_with_retry, render_prompt, request_mode
from lookup_value import lookup_value
from pair import workload


def measure(args, prompt, key, policy):
    start_ns, start = time.time_ns(), time.perf_counter()
    ids = render_prompt(args, prompt)
    if policy == "retry":
        value, calls = lookup_with_retry(key, lambda mode: request_mode(args, ids, mode))
    else:
        row = request_mode(args, ids, "native")
        calls = [row]
        value = lookup_value(key, row["answer"], row["finish"])
    elapsed, end_ns = time.perf_counter() - start, time.time_ns()
    return dict(policy=policy, value=value, elapsed_s=elapsed, start_unix_ns=start_ns, end_unix_ns=end_ns,
                calls=[{k: row[k] for k in ("mode", "finish", "prompt_tokens", "output_tokens",
                                             "submitted_ids_sha256", "generated_ids_sha256")} for row in calls])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()
    if args.count < 1 or args.limit < 1:
        parser.error("--count and --limit must be positive")
    with args.out.open("x") as out:
        for index in range(args.count):
            seed = args.seed + index
            prompt, expected = workload("lookup", seed, 32)
            key = re.search(r"Return only the value of (bin-[0-9]{4})\.", prompt)[1]
            order = ("native", "retry") if index % 2 == 0 else ("retry", "native")
            for policy in order:
                row = measure(args, prompt, key, policy)
                row.update(seed=seed, key=key, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           expected=expected, correct=row["value"] == expected)
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                out.flush()
