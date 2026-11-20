#!/usr/bin/env python3
"""Replay text workloads against one or more OpenAI-compatible Spark endpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def workload(kind, seed, records):
    rng = random.Random(seed)
    a, b = rng.sample(range(10, 99), 2)
    tag = f"Case {seed:08x}. "
    if kind == "prose":
        return tag + f"Write a field guide to caring for {a} orchard trees in a dry summer.", None
    if kind == "code":
        return tag + f"Write a Python function that merges batches of {a} sorted event logs. Include an example and explain its complexity.", None
    if kind == "lookup" or seed % 4 == 0:
        rows = [f"bin-{i:04d} = {rng.randrange(100000, 999999)}" for i in range(records)]
        key, value = rng.choice(rows).split(" = ")
        return tag + f"Return only the value of {key}.\n" + "\n".join(rows), value
    if seed % 4 == 1:
        return tag + f"A shop packs {a} boxes with {b} items each, then ships {a} items. Reply with the remaining item count only.", str(a * (b - 1))
    if seed % 4 == 2:
        return tag + f'Return one JSON object with keys "low" and "high" containing the smaller and larger of {a} and {b}.', {"low": min(a, b), "high": max(a, b)}
    return tag + f"What does this Python expression evaluate to? Reply with the integer only.\nsum(x % 7 for x in range({a}, {a + b}))", str(sum(x % 7 for x in range(a, a + b)))


def stream(endpoint, model, prompt, limit, timeout, thinking="default"):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": limit, "stream": True,
               "stream_options": {"include_usage": True}}
    if thinking != "default":
        payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    req = urllib.request.Request(endpoint + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_delta = first_text = finish = usage = None
    parts = []
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                elapsed = time.perf_counter() - start
                if first_delta is None and (content or delta.get("reasoning") or delta.get("reasoning_content")):
                    first_delta = elapsed
                if content:
                    if first_text is None:
                        first_text = elapsed
                    parts.append(content)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not done or finish not in ("length", "stop") or first_delta is None:
        raise ValueError(f"incomplete generation (finish={finish!r}, done={done})")
    if not usage or any(type(usage.get(k)) is not int or usage[k] <= 0
                        for k in ("prompt_tokens", "completion_tokens")):
        raise ValueError("the server must return positive prompt and completion token counts")
    return {"elapsed_s": time.perf_counter() - start, "first_delta_s": first_delta,
            "first_text_s": first_text, "finish": finish, "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"], "answer": "".join(parts)}


def matches(answer, expected):
    if isinstance(expected, dict):
        try:
            parsed = json.loads(answer)
            return (isinstance(parsed, dict) and parsed.keys() == expected.keys()
                    and all(type(parsed[k]) is int and parsed[k] == v for k, v in expected.items()))
        except ValueError:
            return False
    return answer.strip() == expected


def sample(args, index, seed):
    endpoint = index % len(args.endpoint)
    prompt, expected = workload(args.workload, seed, args.records)
    row = {"case": seed, "endpoint_index": endpoint, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    try:
        row.update(stream(args.endpoint[endpoint], args.model, prompt, args.max_tokens,
                          args.timeout, args.thinking))
        if expected is not None:
            row.update(expected=expected, correct=row["finish"] == "stop" and matches(row["answer"], expected))
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError) as error:
        # HTTP failures, truncated streams and bad usage must stay visible in the receipt.
        if isinstance(error, urllib.error.HTTPError):
            row["error"] = f"HTTP {error.code}: {error.read(800).decode(errors='replace')}"
        else:
            row["error"] = str(error)
    return row


def summary(rows, wall):
    ok = [r for r in rows if "error" not in r]
    out = {"completed": len(ok), "failed": len(rows) - len(ok), "wall_s": wall}
    graded = [r for r in ok if "correct" in r]
    if graded:
        out["correct"] = sum(r["correct"] for r in graded)
    if len(ok) != len(rows):
        return out
    ttft = sorted(r["first_delta_s"] * 1000 for r in ok)
    out.update(output_tokens_per_s=sum(r["output_tokens"] for r in ok) / wall,
               median_first_delta_ms=statistics.median(ttft),
               p95_first_delta_ms=ttft[math.ceil(0.95 * len(ttft)) - 1],
               length_limited=sum(r["finish"] == "length" for r in ok))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="served model name")
    p.add_argument("--endpoint", action="append", help="base URL; repeat for independent replicas")
    p.add_argument("--workload", choices=("prose", "code", "lookup", "quality"), default="prose")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                   help="set enable_thinking for chat templates which support it")
    p.add_argument("--records", type=int, default=16, help="records per lookup prompt")
    p.add_argument("--timeout", type=int, default=300, help="HTTP read timeout in seconds")
    p.add_argument("--out", type=Path, required=True, help="new output directory for the run")
    args = p.parse_args()
    args.endpoint = [e.rstrip("/") for e in (args.endpoint or ["http://127.0.0.1:8000"])]
    if min(args.requests, args.concurrency, args.repeats, args.max_tokens, args.records, args.timeout) < 1:
        p.error("counts, timeout and output budget must be positive")
    if args.requests < max(args.concurrency, len(args.endpoint)):
        p.error("supply at least one request per concurrent client and per endpoint")
    if args.out.exists():
        p.error("output directory already exists; choose a fresh path")
    args.out.mkdir(parents=True, exist_ok=False)
    settings = {k: v for k, v in vars(args).items() if k not in ("endpoint", "out")}
    settings.update(started_utc=datetime.now(timezone.utc).isoformat(), temperature=0,
                    endpoints=len(args.endpoint), warmups_per_endpoint=1,
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    recipe_file = ROOT / "recipes" / (args.model + ".json")
    if recipe_file.parent == ROOT / "recipes" and recipe_file.is_file():
        r = json.loads(recipe_file.read_text())
        settings["catalog_recipe"] = dict(r, image=json.loads((ROOT / "images.json").read_text())[r["image"]])
    def save(name, data):
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    save("settings.json", settings)
    warmup = [sample(args, i, args.seed - 1 - i) for i in range(len(args.endpoint))]
    save("warmup.json", warmup)
    if any("error" in row for row in warmup):
        sys.exit(f"warm-up failed, see {args.out / 'warmup.json'}")
    runs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for trial in range(args.repeats):
            start_unix_ns = time.time_ns()
            start = time.perf_counter()
            jobs = [pool.submit(sample, args, i, args.seed + trial * args.requests + i) for i in range(args.requests)]
            rows = [job.result() for job in jobs]
            wall = time.perf_counter() - start
            end_unix_ns = time.time_ns()
            result = dict(summary(rows, wall), start_unix_ns=start_unix_ns, end_unix_ns=end_unix_ns)
            save(f"trial-{trial + 1}.json", {"summary": result, "requests": rows})
            print(json.dumps(result), flush=True)
            if result["failed"]:
                sys.exit("incomplete trial; see the per-request errors")
            runs.append(result)
    save("summary.json", {"trials": runs, "median_output_tokens_per_s": statistics.median(
        r["output_tokens_per_s"] for r in runs)})
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
