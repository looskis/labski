"""Measure paired target-token likelihoods across three 2,048-token prefill chunks."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from wikitext_probe import post

CORPUS_SHA256 = "ec6b6a121622680b2278495bb92bcda9ef8bfd98862624d90dcb3a8eb9628081"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
WIDTH = 6144
STARTS = tuple(range(0, 16 * WIDTH, 2 * WIDTH))


def ids_digest(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def prepare(corpus, tokenizer, dest):
    from tokenizers import Tokenizer

    for path, expected in ((corpus, CORPUS_SHA256), (tokenizer, TOKENIZER_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path.name} differs from the pinned input")
    articles = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]
    ids = Tokenizer.from_file(str(tokenizer)).encode("\n\n".join(articles), add_special_tokens=False).ids
    if len(articles) != 62 or len(ids) != 103340:
        raise ValueError("unexpected source token stream")
    with dest.open("x") as out:
        for start in STARTS:
            window = ids[start:start + WIDTH]
            out.write(json.dumps(dict(start=start, ids=window, token_ids_sha256=ids_digest(window))) + "\n")


def score(model, url, windows, dest):
    cases = [json.loads(line) for line in windows.read_text().splitlines()]
    if [case["start"] for case in cases] != list(STARTS):
        raise ValueError("unexpected window offsets")
    with gzip.open(dest, "xt") as out:
        for call, case in enumerate([*cases, cases[0]]):
            ids = case["ids"]
            if len(ids) != WIDTH or ids_digest(ids) != case["token_ids_sha256"]:
                raise ValueError("invalid input window")
            reply = post(url.rstrip("/") + "/v1/completions", dict(model=model, prompt=ids,
                         max_tokens=1, temperature=0, add_special_tokens=False, prompt_logprobs=0))
            scores = reply["choices"][0]["prompt_logprobs"]
            usage = reply["usage"]
            if usage["prompt_tokens"] != WIDTH or len(scores) != WIDTH or scores[0] is not None:
                raise ValueError("server scored unexpected positions")
            logps = [score[str(token)]["logprob"] for token, score in zip(ids[1:], scores[1:])]
            if not all(math.isfinite(x) and x <= 0 for x in logps):
                raise ValueError("invalid target log probability")
            record = dict(call=call, start=case["start"], token_ids_sha256=case["token_ids_sha256"],
                          repeated=call == len(cases), logprobs=logps)
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            out.flush()


def read_scores(path):
    with gzip.open(path, "rt") as src:
        rows = [json.loads(line) for line in src]
    offsets = [*STARTS, STARTS[0]]
    if [r["start"] for r in rows] != offsets or rows[0]["token_ids_sha256"] != rows[-1]["token_ids_sha256"]:
        raise ValueError(f"unexpected windows in {path}")
    for i, row in enumerate(rows):
        values = row["logprobs"]
        if row["call"] != i or row["repeated"] != (i == len(STARTS)) or len(values) != WIDTH - 1:
            raise ValueError(f"invalid score record in {path}")
        if not all(math.isfinite(x) and x <= 0 for x in values):
            raise ValueError(f"invalid log probability in {path}")
    return rows


def differences(pairs):
    scores = list(pairs)
    count = len(scores)
    delta = [a - b for a, b in scores]
    return dict(scored_tokens=count, identical=sum(a == b for a, b in scores),
                control_mean_nll=-math.fsum(a for a, _ in scores) / count,
                candidate_mean_nll=-math.fsum(b for _, b in scores) / count,
                candidate_minus_control_mean_nll=math.fsum(delta) / count,
                max_abs_logprob_delta=max(map(abs, delta)))


def compare(control, candidate):
    a, b = read_scores(control), read_scores(candidate)
    if [r["token_ids_sha256"] for r in a] != [r["token_ids_sha256"] for r in b]:
        raise ValueError("input token IDs differ between runs")
    chunks = []
    for first, last in ((1, 2047), (2048, 4095), (4096, 6143)):
        pairs = ((x, y) for orig, test in zip(a[:-1], b[:-1])
                 for x, y in zip(orig["logprobs"][first - 1:last], test["logprobs"][first - 1:last]))
        chunks.append(dict(first_position=first, last_position=last, **differences(pairs)))
    repeats = {label: differences(zip(run[0]["logprobs"], run[-1]["logprobs"]))
               for label, run in (("control", a), ("candidate", b))}
    windows = [dict(start=orig["start"], **differences(zip(orig["logprobs"], test["logprobs"])))
               for orig, test in zip(a[:-1], b[:-1])]
    overall = differences((x, y) for orig, test in zip(a[:-1], b[:-1])
                          for x, y in zip(orig["logprobs"], test["logprobs"]))
    return dict(overall=overall, windows=windows, nominal_chunks=chunks, within_launch_first_window=repeats)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("corpus", type=Path)
    prep.add_argument("tokenizer", type=Path)
    prep.add_argument("out", type=Path)
    run = commands.add_parser("score")
    run.add_argument("model")
    run.add_argument("windows", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--url", default="http://127.0.0.1:8000")
    paired = commands.add_parser("compare")
    paired.add_argument("control", type=Path)
    paired.add_argument("candidate", type=Path)
    args = cli.parse_args()
    if args.command == "compare":
        print(json.dumps(compare(args.control, args.candidate), indent=2))
        sys.exit(0)
    if args.out.exists():
        cli.error(f"output already exists at {args.out}")
    with tempfile.TemporaryDirectory(dir=args.out.parent) as staging:
        result = Path(staging) / "result"
        if args.command == "prepare":
            prepare(args.corpus, args.tokenizer, result)
        else:
            score(args.model, args.url, args.windows, result)
        os.link(result, args.out)
