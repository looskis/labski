"""Count shared prefix tokens in the archived Qwen continuations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def read_run(directory, tokenizer):
    with gzip.open(directory / "requests.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    answers = {}
    for row in rows:
        key = row["concurrency"], row["case"]
        if key in answers or row["finish"] != "length":
            raise ValueError("expected one length-capped continuation per case and concurrency")
        tokens = tokenizer.encode(row["answer"], add_special_tokens=False).ids
        answers[key] = row["prompt_sha256"], tokens
    if set(answers) != {(c, i) for c in (1, 4) for i in range(42, 54)}:
        raise ValueError("incomplete prose panel")
    return answers


def observation(left, right, lhs, rhs, case):
    (prompt, a), (other, b) = left, right
    if prompt != other:
        raise ValueError("prompts differ")
    shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(case=case, left=lhs, right=rhs, prompt_sha256=prompt, left_tokens=len(a),
                right_tokens=len(b), shared_prefix_tokens=shared)


def compare(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("expected tokenizer.json from the pinned NVIDIA Qwen3.8-27B-NVFP4 target")
    tokenizer = Tokenizer.from_file(str(path))
    base = Path(__file__).resolve().parent
    runs = {k: read_run(base / str(k), tokenizer) for k in (0, 3, 7)}
    for k, run in runs.items():
        for case in range(42, 54):
            yield observation(run[1, case], run[4, case], f"k{k}-c1", f"k{k}-c4", case)
        if k:
            for c in (1, 4):
                for case in range(42, 54):
                    yield observation(runs[0][c, case], run[c, case], f"k0-c{c}", f"k{k}-c{c}", case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    for row in compare(parser.parse_args().tokenizer):
        print(json.dumps(row, separators=(",", ":")))
