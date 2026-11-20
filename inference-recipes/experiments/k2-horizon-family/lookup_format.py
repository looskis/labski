"""Compare free and six-digit constrained K2 responses on fresh record lookups."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import workload


def lookup_value(text, finish):
    """Extract one standalone ASCII six-digit value from a completed response."""
    values = re.findall(r"(?<![0-9])[0-9]{6}(?![0-9])", text)
    return values[0] if finish == "stop" and len(values) == 1 else None


def request(url, model, seed, records, constrained):
    prompt, expected = workload("lookup", seed, records)
    body = dict(model=model, messages=[dict(role="user", content=prompt)],
                temperature=0, max_tokens=128, chat_template_kwargs={"reasoning_effort": "low"})
    if constrained:
        body["structured_outputs"] = {"regex": "[0-9]{6}"}
    call = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(call, timeout=180) as reply:
        response = json.load(reply)
    elapsed = time.perf_counter() - start
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    values = re.findall(r"(?<![0-9])[0-9]{6}(?![0-9])", text)
    stopped = choice["finish_reason"] == "stop"
    return dict(seed=seed, records=records, regex=constrained, expected=expected,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                finish=choice["finish_reason"], input_tokens=response["usage"]["prompt_tokens"],
                output_tokens=response["usage"]["completion_tokens"], elapsed_s=elapsed,
                visible_sha256=hashlib.sha256(text.encode()).hexdigest(),
                visible_characters=len(text), standalone_values=values,
                strict=stopped and text.strip() == expected,
                single_value_correct=lookup_value(text, choice["finish_reason"]) == expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--count", type=int, default=32, help="fresh prompts for each record count")
    parser.add_argument("--records", type=int, nargs="+", default=[32, 128])
    args = parser.parse_args()
    if args.count < 1 or not all(n > 0 for n in args.records):
        parser.error("count and records must be positive")
    with args.out.open("x") as out:
        for index in range(args.count):
            sizes = args.records if index % 2 == 0 else reversed(args.records)
            for records in sizes:
                seed = args.seed + index + args.records.index(records) * args.count
                for constrained in ((False, True) if seed % 2 == 0 else (True, False)):
                    out.write(json.dumps(request(args.url, args.model, seed, records, constrained),
                                         separators=(",", ":")) + "\n")
                    out.flush()
