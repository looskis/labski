"""Prepare and run a deterministic zero-shot, short-answer MMLU-Pro diagnostic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import bench

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
PARQUET_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
PREFIX = "labski-mmlu-pro-v1"
LETTERS = "ABCDEFGHIJ"


def prepare(args):
    import pyarrow.parquet as pq
    if hashlib.sha256(args.parquet.read_bytes()).hexdigest() != PARQUET_SHA256:
        raise ValueError("expected the pinned MMLU-Pro test parquet")
    groups = defaultdict(list)
    for row in pq.read_table(args.parquet).to_pylist():
        key = hashlib.sha256(f"{PREFIX}/{row['question_id']}".encode()).hexdigest()
        groups[row["category"]].append((key, row))
    with args.out.open("x") as stream:
        for category, group in sorted(groups.items()):
            if len(group) < args.per_subject:
                raise ValueError(f"insufficient test items in {category}")
            for _, row in sorted(group)[:args.per_subject]:
                options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["options"]))
                prompt = f"{row['question']}\n\n{options}\n\nReply with the letter of the correct option only."
                item = dict(id=row["question_id"], subject=category, prompt=prompt, expected=row["answer"])
                stream.write(json.dumps(item) + "\n")


def run(args):
    cases = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("empty corpus or duplicate IDs")
    with args.out.open("x") as stream:
        for item in cases:
            result = bench.stream(args.url, args.model, item["prompt"], 16, 180, "off")
            # A truncated or non-letter answer is an incorrect answer.
            answer = result["answer"].strip()
            row = dict(id=item["id"], subject=item["subject"], expected=item["expected"],
                       prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                       correct=result["finish"] == "stop" and answer == item["expected"], **result)
            stream.write(json.dumps(row) + "\n")
            stream.flush()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cmds = cli.add_subparsers(dest="command", required=True)
    prep = cmds.add_parser("prepare", help="requires pyarrow and the pinned test parquet")
    prep.add_argument("parquet", type=Path)
    prep.add_argument("--per-subject", type=int, default=8)
    trial = cmds.add_parser("run")
    trial.add_argument("model")
    trial.add_argument("corpus", type=Path)
    trial.add_argument("--url", default="http://127.0.0.1:8000")
    for parser in (prep, trial):
        parser.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    if args.command == "prepare" and args.per_subject < 1:
        cli.error("per-subject must be positive")
    (prepare if args.command == "prepare" else run)(args)
