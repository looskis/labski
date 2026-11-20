"""Extract a six-digit Step-5 inventory answer, retrying with native reasoning on a format failure."""

import argparse
from pathlib import Path
import re
import sys

from lookup_value import lookup_value
from pair import completion, post


def lookup_with_retry(key, generate):
    """Call generate(mode) at most twice. Never consult a reference answer."""
    if not re.fullmatch(r"bin-[0-9]{4}", key):
        raise ValueError("expected a key such as bin-0022")
    attempts = []
    for mode in ("prefilled", "native"):
        row = generate(mode)
        attempts.append(row)
        value = lookup_value(key, row["answer"], row["finish"])
        if value is not None:
            return value, attempts
    return None, attempts


def render_prompt(args, prompt):
    body = dict(model=args.model, messages=[dict(role="user", content=prompt)],
                add_generation_prompt=True, chat_template_kwargs={"reasoning_effort": "low"})
    ids = post(args.url, "/tokenize", body)["tokens"]
    closing = post(args.url, "/tokenize", dict(model=args.model, prompt="</think>", add_special_tokens=False))
    if not ids or ids[-1] != 128798 or closing["tokens"] != [128799]:
        raise ValueError("the endpoint uses a different reasoning template or tokenizer")
    return ids


def request_mode(args, ids, mode):
    return completion(args, ids, mode, 128798, 128799, expected=None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("key", help="key requested by the prompt")
    cli.add_argument("prompt", type=Path, help="UTF-8 inventory and question")
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--limit", type=int, default=384)
    args = cli.parse_args()
    if args.limit < 1:
        cli.error("--limit must be positive")
    ids = render_prompt(args, args.prompt.read_text())
    value, _ = lookup_with_retry(args.key, lambda mode: request_mode(args, ids, mode))
    if value is None:
        sys.exit("neither completion supplied an accepted value")
    print(value)
