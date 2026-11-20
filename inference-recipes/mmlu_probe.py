"""Prepare and run a deterministic zero-shot, short-answer MMLU-Pro diagnostic."""
import argparse
from collections import Counter
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import urllib.request

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


def post(url, body, timeout=180):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def letter_ids(url, model):
    allowed = []
    for letter in LETTERS:
        tokens = post(url + "/tokenize", dict(model=model, prompt=letter, add_special_tokens=False))["tokens"]
        if len(tokens) != 1 or tokens[0] in allowed:
            raise ValueError("this probe requires distinct single-token uppercase letters A through J")
        allowed.append(tokens[0])
    return allowed


def one_letter(args, prompt, allowed):
    body = dict(model=args.model, messages=[dict(role="user", content=prompt)], temperature=0,
                max_tokens=1, allowed_token_ids=allowed, chat_template_kwargs={"enable_thinking": False})
    by_token = getattr(args, "one_letter_token", False)
    if by_token:
        body["return_token_ids"] = True
    start = time.perf_counter()
    data = post(args.url + "/v1/chat/completions", body)
    elapsed = time.perf_counter() - start
    choice = data["choices"][0]
    message = choice["message"]
    answer = message.get("content")
    if by_token:
        ids = choice.get("token_ids")
        if (data["usage"]["completion_tokens"] != 1 or not isinstance(ids, list) or len(ids) != 1
                or type(ids[0]) is not int or ids[0] not in allowed):
            raise ValueError(f"server did not return one of the ten allowed token IDs: {ids!r}")
        return dict(answer=LETTERS[allowed.index(ids[0])], answer_token_id=ids[0], visible_answer=answer,
                    reasoning_answer=message.get("reasoning") or message.get("reasoning_content") or "",
                    finish=choice["finish_reason"], elapsed_s=elapsed, input_tokens=data["usage"]["prompt_tokens"],
                    output_tokens=1, allowed_token_ids=allowed)
    if not isinstance(answer, str) or len(answer) != 1 or answer not in LETTERS or data["usage"]["completion_tokens"] != 1:
        raise ValueError("server did not return exactly one allowed answer token")
    return dict(answer=answer, finish=choice["finish_reason"], elapsed_s=elapsed,
                input_tokens=data["usage"]["prompt_tokens"], output_tokens=1, allowed_token_ids=allowed)


def native_chat(args, prompt):
    start = time.perf_counter()
    budget = args.max_tokens or 256
    if args.reasoning_effort:
        options = {"chat_template_kwargs": {"reasoning_effort": args.reasoning_effort}}
        settings = {"requested_effort": args.reasoning_effort}
    else:
        options = {"enable_thinking": args.enable_thinking == "on"}
        settings = dict(options)
    data = post(args.url + "/v1/chat/completions", dict(
        model=args.model, messages=[{"role": "user", "content": prompt}], temperature=0,
        max_tokens=budget, **options), timeout=600)
    choice = data["choices"][0]
    message = choice["message"]
    usage = data["usage"]
    if usage["prompt_tokens"] <= 0 or not 0 < usage["completion_tokens"] <= budget:
        raise ValueError("server token usage falls outside the request budget")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return dict(answer=message.get("content") or "", finish=choice["finish_reason"],
                elapsed_s=time.perf_counter() - start, input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"], reasoning_chars=len(reasoning),
                max_tokens=budget, **settings)


def run(args):
    cases = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("empty corpus or duplicate IDs")
    allowed = letter_ids(args.url, args.model) if args.one_letter or args.one_letter_token else None
    subjects = Counter()
    with args.out.open("x") as stream:
        for item in cases:
            subjects[item["subject"]] += 1
            if args.per_subject and subjects[item["subject"]] > args.per_subject:
                continue
            if allowed:
                result = one_letter(args, item["prompt"], allowed)
                complete = result["finish"] in ("stop", "length")
            elif args.reasoning_effort or args.enable_thinking:
                result = native_chat(args, item["prompt"])
                complete = result["finish"] == "stop"
            else:
                result = bench.stream(args.url, args.model, item["prompt"], args.max_tokens or 16, 180, "off")
                complete = result["finish"] == "stop"
            answer = result["answer"].strip()
            row = dict(id=item["id"], subject=item["subject"], expected=item["expected"],
                       prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                       correct=complete and answer == item["expected"], **result)
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
    mode = trial.add_mutually_exclusive_group()
    mode.add_argument("--reasoning-effort", choices=("none", "low", "high", "max"),
                      help="request the model's native chat template effort and score final content")
    mode.add_argument("--enable-thinking", choices=("on", "off"),
                      help="set the top-level enable_thinking field on servers such as mlx-serve")
    trial.add_argument("--max-tokens", type=int, help="output budget (16 short-answer, 256 native effort)")
    trial.add_argument("--per-subject", type=int, help="score the first N pinned cases of each subject")
    mode.add_argument("--one-letter", action="store_true", help="restrict greedy decoding to one A–J token")
    mode.add_argument("--one-letter-token", action="store_true",
                      help="score the one allowed generated token ID and preserve its visible rendering")
    for parser in (prep, trial):
        parser.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    if args.per_subject is not None and args.per_subject < 1:
        cli.error("--per-subject must be positive")
    if args.command == "run" and args.max_tokens is not None:
        if args.max_tokens < 1 or ((args.one_letter or args.one_letter_token) and args.max_tokens != 1):
            cli.error("use a positive output budget, or 1 for a one-letter probe")
    (prepare if args.command == "prepare" else run)(args)
