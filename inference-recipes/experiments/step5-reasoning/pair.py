"""Compare Step's native low effort with a prefilled empty reasoning block."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import workload
from glm_effort import post


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def cases(corpus, lookups_only, lookup_seed, lookup_count):
    if not lookups_only:
        yield from question_cases(corpus)
    for case in range(lookup_seed, lookup_seed + lookup_count):
        prompt, expected = workload("lookup", case, 32)
        yield "lookup", case, prompt, expected


def question_cases(corpus):
    seen = set()
    for row in (json.loads(line) for line in corpus.read_text().splitlines()):
        if row["subject"] not in seen:
            seen.add(row["subject"])
            yield "question", row["id"], row["prompt"], row["expected"]
    if len(seen) != 14:
        raise ValueError("expected 14 MMLU-Pro subjects")


def completion(args, base_ids, mode, opened, closed, expected):
    ids = base_ids + ([closed] if mode == "prefilled" else [])
    body = dict(model=args.model, prompt=ids, max_tokens=args.limit, temperature=0,
                add_special_tokens=False, return_token_ids=True, skip_special_tokens=True)
    start_ns, start = time.time_ns(), time.perf_counter()
    result = post(args.url, "/v1/completions", body)
    elapsed, end_ns = time.perf_counter() - start, time.time_ns()
    choice = result["choices"][0]
    generated, raw = choice["token_ids"], choice["text"]
    usage = result["usage"]
    if usage["prompt_tokens"] != len(ids) or usage["completion_tokens"] != len(generated):
        raise ValueError("server token usage disagrees with submitted or returned token IDs")
    if mode == "native":
        has_answer = closed in generated
        _, delimiter, answer = raw.partition("</think>")
        if bool(delimiter) != has_answer or generated.count(closed) > 1:
            raise ValueError("unexpected native reasoning boundary")
        reasoning_tokens = generated.index(closed) if has_answer else len(generated)
    else:
        answer = raw
        reasoning_tokens = None
    answer = answer.strip()
    return dict(mode=mode, base_ids_sha256=ids_hash(base_ids), submitted_ids_sha256=ids_hash(ids),
                generated_ids_sha256=ids_hash(generated), prompt_tokens=len(ids),
                output_tokens=len(generated), new_open_markers=generated.count(opened),
                new_close_markers=generated.count(closed), native_reasoning_tokens=reasoning_tokens,
                finish=choice["finish_reason"], answer=answer, expected=expected,
                correct=choice["finish_reason"] == "stop" and answer == expected,
                start_unix_ns=start_ns, end_unix_ns=end_ns, elapsed_s=elapsed)


def main(args):
    def plain(text):
        return post(args.url, "/tokenize", dict(model=args.model, prompt=text, add_special_tokens=False))["tokens"]
    if plain("<think>") != [128798] or plain("</think>") != [128799]:
        raise ValueError("expected the pinned Step-5 reasoning tokens")
    with args.out.open("x") as out:
        for i, (group, case, prompt, expected) in enumerate(cases(
                args.corpus, args.lookups_only, args.lookup_seed, args.lookup_count)):
            ids = post(args.url, "/tokenize", dict(model=args.model, messages=[dict(role="user", content=prompt)],
                       add_generation_prompt=True, chat_template_kwargs={"reasoning_effort": "low"}))["tokens"]
            if not ids or ids[-1] != 128798:
                raise ValueError("the official low-effort template did not end at the opening marker")
            order = ("native", "prefilled") if i % 2 == 0 else ("prefilled", "native")
            for mode in order:
                row = completion(args, ids, mode, 128798, 128799, expected)
                row.update(group=group, case=case, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           max_tokens=args.limit)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--lookups-only", action="store_true")
    parser.add_argument("--lookup-seed", type=int, default=42)
    parser.add_argument("--lookup-count", type=int, default=8)
    args = parser.parse_args()
    if min(args.limit, args.lookup_count) < 1:
        parser.error("--limit and --lookup-count must be positive")
    main(args)
