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
    args = cli.parse_args()
    if args.out.exists():
        cli.error(f"output already exists at {args.out}")
    with tempfile.TemporaryDirectory(dir=args.out.parent) as staging:
        result = Path(staging) / "result"
        if args.command == "prepare":
            prepare(args.corpus, args.tokenizer, result)
        else:
            score(args.model, args.url, args.windows, result)
        os.link(result, args.out)
