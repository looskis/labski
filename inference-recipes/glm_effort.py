"""Evaluate native GLM reasoning effort on a fixed subset of the MMLU-Pro probe."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def post(url, path, body):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def run(args):
    items = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    counts = Counter()
    with args.out.open("x") as stream:
        for item in items:
            subject = item["subject"]
            counts[subject] += 1
            if counts[subject] > args.per_subject:
                continue
            body = dict(model=args.model, messages=[dict(role="user", content=item["prompt"])],
                        add_generation_prompt=True, chat_template_kwargs={"reasoning_effort": args.effort})
            tokens = post(args.url, "/tokenize", body)["tokens"]
            start = time.perf_counter()
            data = post(args.url, "/v1/completions", dict(model=args.model, prompt=tokens,
                        temperature=0, max_tokens=args.max_tokens, add_special_tokens=False))
            elapsed = time.perf_counter() - start
            choice = data["choices"][0]
            reasoning, end, answer = choice["text"].partition("</think>")
            usage = data["usage"]
            if usage["prompt_tokens"] != len(tokens) or usage["completion_tokens"] < 1:
                raise ValueError("token count mismatch")
            row = dict(id=item["id"], subject=subject, expected=item["expected"], effort=args.effort,
                       prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                       rendered_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                       input_tokens=len(tokens), output_tokens=usage["completion_tokens"],
                       elapsed_s=elapsed, finish=choice["finish_reason"], closed_thinking=bool(end),
                       reasoning_characters=len(reasoning), answer=answer.strip(),
                       correct=bool(end) and choice["finish_reason"] == "stop" and answer.strip() == item["expected"])
            stream.write(json.dumps(row) + "\n")
            stream.flush()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("corpus", type=Path)
    cli.add_argument("--effort", choices=("low", "high", "max"), required=True)
    cli.add_argument("--max-tokens", type=int, default=256)
    cli.add_argument("--per-subject", type=int, default=2)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    if min(args.per_subject, args.max_tokens) < 1:
        cli.error("per-subject and max-tokens must be positive")
    run(args)
