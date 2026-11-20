"""Compare streamed and nonstreamed K2 responses on held-out record lookups."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import workload


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def measure(url, model, seed, streamed):
    prompt, expected = workload("lookup", seed, 32)
    body = dict(model=model, messages=[dict(role="user", content=prompt)],
                temperature=0, max_tokens=128, return_token_ids=True,
                chat_template_kwargs={"reasoning_effort": "low"}, stream=streamed)
    if streamed:
        body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    ids, final, reason_chars, finish, usage = [], "", 0, None, None
    with urllib.request.urlopen(request, timeout=180) as reply:
        if streamed:
            def chunks():
                for line in reply:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        yield json.loads(line[6:])
            events = chunks()
        else:
            events = (json.load(reply),)
        for event in events:
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                msg = choice["delta" if streamed else "message"]
                final += msg.get("content") or ""
                reason_chars += len(msg.get("reasoning") or msg.get("reasoning_content") or "")
                ids.extend(choice.get("token_ids") or [])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not usage or not finish:
        raise ValueError("no terminal usage or finish reason")
    answer = final.strip()
    return dict(seed=seed, stream=streamed, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected=expected, returned_digits=answer if len(answer) == 6 and answer.isascii() and answer.isdigit() else None,
                correct=finish == "stop" and answer == expected, visible_characters=len(final), reasoning_characters=reason_chars,
                visible_sha256=hashlib.sha256(final.encode()).hexdigest(), finish=finish,
                input_tokens=usage["prompt_tokens"], generated_tokens=usage["completion_tokens"],
                returned_token_count=len(ids), token_ids_sha256=ids_hash(ids) if len(ids) == usage["completion_tokens"] else None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("out", type=Path)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--seed", type=int, default=1000)
    cli.add_argument("--count", type=int, default=16)
    args = cli.parse_args()
    if args.count < 1:
        cli.error("choose a positive case count")
    with args.out.open("x") as f:
        for case in range(args.seed, args.seed + args.count):
            for stream in (False, True) if case % 2 == 0 else (True, False):
                f.write(json.dumps(measure(args.url, args.model, case, stream), separators=(",", ":")) + "\n")
                f.flush()
