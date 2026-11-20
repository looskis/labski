#!/usr/bin/env python3
"""Measure cold prompt processing through an OpenAI-compatible streaming API.

Example (run on rank 0 to avoid WAN latency):
  python3 prefill.py --base http://127.0.0.1:8020 --label baseline

Context sizes are targets, not asserted token counts. Every row records the actual
server-reported prompt count. Client TTFT includes scheduling and first-token
work; engine prefill timing, when supplied, is reported separately. A run is
eligible for cold summaries only when cached_tokens is explicitly zero.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import threading
import time
import urllib.request
import uuid


def prompt(words, seed):
    rng = random.Random(seed)
    filler = " ".join(f"w{rng.randrange(100000)}" for _ in range(words))
    return filler + "\nSummarize the preceding text in one word."


def headers():
    result = {"Content-Type": "application/json"}
    if os.environ.get("BENCH_API_KEY"):
        result["Authorization"] = "Bearer " + os.environ["BENCH_API_KEY"]
    return result


def sized_prompt(args, target, seed):
    words = max(1, round((target - 32) / 3.8))
    def count(content):
        payload = {"model": args.model,
                   "messages": [{"role": "user", "content": content}],
                   "chat_template_kwargs": {"enable_thinking": False},
                   "add_generation_prompt": True}
        req = urllib.request.Request(args.base.rstrip("/") + "/tokenize",
                                     json.dumps(payload).encode(), headers())
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            return json.load(response)["count"]
    for _ in range(12):
        content = prompt(words, seed)
        actual = count(content)
        if actual == target:
            return content, words
        difference = target - actual
        if 0 < difference <= 32:
            padded = content + " a" * difference
            if count(padded) == target:
                return padded, words
        step = round(difference / max(1, actual / words))
        if step == 0:
            step = 1 if difference > 0 else -1
        words = max(1, words + step)
    raise RuntimeError(f"Could not construct exactly {target} chat prompt tokens")


def stream_events(response):
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def request(args, target, seed, barrier=None, warmup=False):
    words = max(1, round((target - 32) / 3.8))
    content = prompt(words, seed)
    if args.exact_contexts:
        content, words = sized_prompt(args, target, seed)
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 1, "temperature": 0, "stream": True, "cache_prompt": False,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(args.base.rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(), headers())
    record = {"target_tokens": target, "filler_words": words, "seed": seed,
              "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
              "warmup": warmup}
    if barrier:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    record["start_monotonic_s"] = start
    usage, engine, timings, first, text = {}, {}, {}, None, []
    event_offsets = []
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            for event in stream_events(response):
                now = time.perf_counter()
                event_offsets.append(now - start)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                usage.update(event.get("usage") or {})
                engine.update(event.get("tensorfold") or {})
                timings.update(event.get("timings") or {})
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    chunk = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                    if chunk:
                        if first is None:
                            first = now
                        text.append(chunk)
        record["wall_s"] = time.perf_counter() - start
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        tokens = usage.get("prompt_tokens")
        seconds = engine.get("prefill_seconds")
        if seconds is None and timings.get("prompt_ms") is not None:
            seconds = timings["prompt_ms"] / 1000
        processed_tokens = timings.get("prompt_n", tokens)
        record.update(usage=usage, engine=engine, cached_tokens=cached,
                      timings=timings, engine_prefill_seconds=seconds,
                      cold_verified=(cached == 0), prompt_tokens=tokens,
                      ttft_s=(first - start if first is not None else None),
                      first_content_monotonic_s=first,
                      event_offsets_s=event_offsets, output="".join(text))
        # Do not invent a TTFT for an EOS-only response or count a cache hit.
        record["prompt_tokens_per_ttft_s"] = (
            tokens / (first - start) if cached == 0 and tokens and first is not None else None)
        record["engine_prefill_tps"] = (
            processed_tokens / seconds if cached == 0 and processed_tokens and isinstance(seconds, (int, float))
            and seconds > 0 else None)
        if not tokens:
            record["error"] = "Server omitted a positive prompt_tokens count"
        elif args.exact_contexts and not warmup and tokens != target:
            record["error"] = f"Tokenizer count {target} differs from inference count {tokens}"
    except Exception as error:
        record.update(error=str(error), wall_s=time.perf_counter() - start)
    return record


def summarize(records):
    groups = {}
    for row in records:
        if row.get("warmup"):
            continue
        key = f"target_{row['target_tokens']}_concurrency_{row['concurrency']}"
        groups.setdefault(key, []).append(row)
    result = {}
    for key, rows in groups.items():
        item = {"requests": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
                "cold_verified": sum(r.get("cold_verified", False) for r in rows),
                "actual_prompt_tokens": [r.get("prompt_tokens") for r in rows]}
        for metric in ("ttft_s", "prompt_tokens_per_ttft_s", "engine_prefill_tps"):
            values = [r[metric] for r in rows if r.get("cold_verified")
                      and not r.get("error") and r.get(metric) is not None]
            if values:
                item[metric] = {"median": statistics.median(values),
                                "min": min(values), "max": max(values), "n": len(values)}
        result[key] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--label", required=True)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=32768,
                        help="Excluded fresh prompt used to warm the long-prefill path")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--prompt-seed", default=None,
                        help="Reuse identical timed prompts across recipes; cache verification remains mandatory")
    parser.add_argument("--exact-contexts", action="store_true",
                        help="Calibrate chat prompts with /tokenize before starting each timer")
    args = parser.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    loads = [int(x) for x in args.concurrency.split(",")]
    if min(contexts + loads + [args.trials, args.warmup_tokens]) < 1:
        parser.error("contexts, concurrency, and trials must be positive")
    run_id = args.run_id or uuid.uuid4().hex
    prompt_seed = args.prompt_seed or run_id
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in run_id):
        parser.error("run-id must contain only letters, digits, hyphens, underscores, or dots")
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / run_id
    destination.mkdir()  # Refuse to overwrite an earlier run.
    metadata = {**vars(args), "out": str(args.out), "run_id": run_id,
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "max_tokens": 1, "thinking": False, "temperature": 0,
                "prompt_type": "synthetic random words; exact via /tokenize" if args.exact_contexts
                               else "synthetic random words; token lengths are approximate",
                "client_ttft_definition": "request start to first nonempty content/reasoning event"}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (destination / "requests.jsonl").open("x") as output:
        def save(row):
            records.append(row)
            output.write(json.dumps(row) + "\n")
            output.flush()
            print(json.dumps(row), flush=True)
        warm = request(args, args.warmup_tokens, run_id + "-warmup", warmup=True)
        save(warm)
        if warm.get("error"):
            raise SystemExit("Warmup failed; inspect requests.jsonl")
        for target in contexts:
            for trial in range(args.trials):
                for count in loads:
                    barrier = threading.Barrier(count)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                        futures = [pool.submit(request, args, target,
                                   f"{prompt_seed}-{target}-{trial}-{count}-{i}", barrier)
                                   for i in range(count)]
                        for i, future in enumerate(futures):
                            row = future.result()
                            row.update(trial=trial + 1, concurrency=count, stream=i)
                            save(row)
    summary = summarize(records)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"results": str(destination), "summary": summary}, indent=2))
    if any(r.get("error") or not r.get("cold_verified") for r in records if not r.get("warmup")):
        raise SystemExit("Some requests failed or lacked explicit cache-miss confirmation; see results")


if __name__ == "__main__":
    main()
