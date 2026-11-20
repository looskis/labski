"""Replay three one-token precision policies on held-out items through two resident workers."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import shlex
import subprocess
import time

MARGIN_NATS = 1.0


class Worker:
    def __init__(self, command):
        self.process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        if self.process.stdout.readline().strip() != "READY":
            self.close()
            raise RuntimeError("scoring worker failed startup")

    def get(self, key):
        self.process.stdin.write(str(key)+"\n")
        self.process.stdin.flush()
        row = json.loads(self.process.stdout.readline())
        if row["id"] != key:
            raise ValueError("worker returned the wrong example")
        return row

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)


def replay(args):
    cases = [json.loads(l) for l in args.corpus.read_text().splitlines()]
    seen = Counter()
    held = []
    for case in cases:
        if seen[case["subject"]] >= 8:
            held.append(case["id"])
        seen[case["subject"]] += 1
    if len(seen)!=14 or set(seen.values())!={16} or len(set(held))!=112:
        raise ValueError("expected sixteen distinct cases for each of fourteen subjects")
    random.Random(20261119).shuffle(held)
    orders = list(itertools.permutations(("mixed","bf16","route")))
    with args.out.open("x") as out:
        mixed = Worker(args.mixed_worker)
        try:
            high = Worker(args.bf16_worker)
            try:
                mixed.get(cases[0]["id"])
                high.get(cases[0]["id"])
                for start in range(0,len(held),16):
                    group = start//16
                    for policy in orders[group % len(orders)]:
                        rows = []
                        begin_ns = time.time_ns()
                        begin = time.perf_counter()
                        for key in held[start:start+16]:
                            first = time.perf_counter()
                            primary = high.get(key) if policy=="bf16" else mixed.get(key)
                            fallback = policy=="route" and primary["margin"] < MARGIN_NATS
                            selected = high.get(key) if fallback else primary
                            if selected["input_ids_sha256"] != primary["input_ids_sha256"]:
                                raise ValueError("precision endpoints tokenized the input differently")
                            rows.append(dict(id=key,expected=selected["expected"],answer=selected["answer"],
                                             correct=selected["correct"],fallback=fallback,margin=primary["margin"],
                                             input_ids_sha256=primary["input_ids_sha256"],elapsed_s=time.perf_counter()-first))
                        wall = time.perf_counter()-begin
                        entry = dict(policy=policy,group=group,threshold_nats=MARGIN_NATS,wall_s=wall,
                                     start_unix_ns=begin_ns,end_unix_ns=time.time_ns(),requests=rows)
                        out.write(json.dumps(entry,separators=(",",":"))+"\n")
                        out.flush()
            finally:
                high.close()
        finally:
            mixed.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus",type=Path)
    p.add_argument("--mixed-worker",required=True,help="command that starts the mixed-precision probe with --worker")
    p.add_argument("--bf16-worker",required=True,help="command that starts the BF16 probe with --worker")
    p.add_argument("--out",type=Path,required=True)
    replay(p.parse_args())
