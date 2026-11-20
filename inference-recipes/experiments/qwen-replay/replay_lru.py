"""Simulate a cold cross-step LRU on one ordered, single-reader PLE trace."""
import argparse
from collections import OrderedDict
import json
from pathlib import Path


def simulate(path, budgets):
    caches = {mib: OrderedDict() for mib in budgets}
    hits = dict.fromkeys(budgets, 0)
    totals = dict(calls=0, row_requests=0, distinct_rows_per_call_sum=0, read_bytes=0,
                  unique_blocks=0, read_calls=0, reader_nanoseconds=0, repeated_rows_compared=0,
                  byte_mismatches=0)
    payload = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            size = row["weight_bytes"] + row["scale_bytes"]
            if size < 1 or (payload is not None and payload != size):
                raise ValueError("expected a single packed table geometry")
            payload = size
            valid = [i for i in row["ids"] if row["shard_start"] <= i < row["shard_end"]]
            distinct = list(dict.fromkeys(valid))
            totals["calls"] += 1
            totals["row_requests"] += len(valid)
            totals["distinct_rows_per_call_sum"] += len(distinct)
            totals["reader_nanoseconds"] += row["nanoseconds"]
            for key in ("read_bytes", "unique_blocks", "read_calls", "repeated_rows_compared", "byte_mismatches"):
                totals[key] += row[key]
            for mib, cache in caches.items():
                slots = (mib * 1024**2) // payload
                if not slots:
                    continue
                # All IDs in one gather arrive together. Count only rows resident before this call.
                hits[mib] += sum(i in cache for i in distinct)
                for i in distinct:
                    cache[i] = None
                    cache.move_to_end(i)
                    if len(cache) > slots:
                        cache.popitem(last=False)
    if not totals["calls"]:
        raise ValueError("empty PLE trace")
    return dict(**totals, packed_bytes_per_row=payload, cache_payload_mib_to_cross_step_hits=hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(simulate(args.trace, (0, 64, 256)), indent=2))
