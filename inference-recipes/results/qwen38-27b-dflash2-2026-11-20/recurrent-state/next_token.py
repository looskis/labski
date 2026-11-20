"""Compare teacher-forced top tokens with direct greedy decoding at fixed prefixes."""

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import tempfile

from long_likelihood import STARTS, WIDTH, ids_digest, post

PREFIXES = (512, 2047, 2048, 2049, 4095, 4096, 4097, 6143)


def completion(url, model, ids, **options):
    reply = post(url.rstrip("/") + "/v1/completions", dict(model=model, prompt=ids,
                 max_tokens=1, temperature=0, add_special_tokens=False,
                 ignore_eos=True, return_token_ids=True, **options))
    choice = reply["choices"][0]
    if (reply["usage"]["prompt_tokens"] != len(ids) or reply["usage"]["completion_tokens"] != 1
            or len(choice["token_ids"]) != 1):
        raise ValueError("unexpected token counts from completion")
    return choice


def top_tokens(scores, observed):
    if str(observed) not in scores or not all(math.isfinite(s["logprob"]) for s in scores.values()):
        raise ValueError("missing or non-finite teacher score")
    top = sorted(int(k) for k, v in scores.items() if v["rank"] == 1)
    if not top or any(scores[str(k)]["logprob"] < max(v["logprob"] for v in scores.values()) for k in top):
        raise ValueError("missing or inconsistent first-ranked token")
    return top


def measure(url, model, cases, dest):
    if [c["start"] for c in cases] != list(STARTS):
        raise ValueError("unexpected window offsets")
    for case in cases:
        if len(case["ids"]) != WIDTH or ids_digest(case["ids"]) != case["token_ids_sha256"]:
            raise ValueError("invalid source window")
    with gzip.open(dest, "xt") as out:
        for call, case in enumerate([*cases, cases[0]]):
            ids = case["ids"]
            scores = completion(url, model, ids, prompt_logprobs=1)["prompt_logprobs"]
            if len(scores) != WIDTH or scores[0] is not None:
                raise ValueError("unexpected teacher-scored positions")
            top = [top_tokens(s, observed) for s, observed in zip(scores[1:], ids[1:])]
            prefixes = []
            for length in PREFIXES:
                generated = completion(url, model, ids[:length])["token_ids"][0]
                prefixes.append(dict(prefix_tokens=length, prefix_ids_sha256=ids_digest(ids[:length]),
                                     observed_next_id=ids[length], generated_id=generated))
            row = dict(call=call, start=case["start"], token_ids_sha256=case["token_ids_sha256"],
                       repeated=call == len(cases), top_ids=top, prefixes=prefixes,
                       observed_in_top=sum(token in ranked for token, ranked in zip(ids[1:], top)))
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            out.flush()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("windows", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    args = cli.parse_args()
    if args.out.exists():
        cli.error("choose a new output path")
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temp:
        target = Path(temp) / "scores.gz"
        measure(args.url, args.model, [json.loads(line) for line in args.windows.read_text().splitlines()], target)
        os.link(target, args.out)
