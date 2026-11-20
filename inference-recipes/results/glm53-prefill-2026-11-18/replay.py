"""Replay the archived cold-prefill token IDs and retain per-request TTFT."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def measure(url, model, case):
    payload = dict(model=model, prompt=case["tokens"], temperature=0, max_tokens=1,
                   stream=True, return_token_ids=True, add_special_tokens=False,
                   cache_prompt=False, stream_options={"include_usage": True})
    request = urllib.request.Request(url + "/v1/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    first = None
    usage = {}
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(event["error"])
            if first is None and any(c.get("text") or c.get("token_ids") for c in event.get("choices", [])):
                first = time.perf_counter() - start
            usage.update(event.get("usage") or {})
        else:
            raise ValueError("stream ended without a completion marker")
    if first is None or usage.get("prompt_tokens") != len(case["tokens"]) or usage.get("completion_tokens") != 1:
        raise ValueError("incomplete generation or token counts differ from the corpus")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        with urllib.request.urlopen(url + "/metrics", timeout=15) as resp:
            config = [line for line in resp if line.startswith(b"vllm:cache_config_info{")]
        if not config or any(b'enable_prefix_caching="False"' not in line for line in config):
            raise ValueError("cold prefix cache could not be verified")
    elif cached != 0:
        raise ValueError("server reused cached tokens")
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    corpus = gzip.decompress((ROOT / "corpus.jsonl.gz").read_bytes())
    digest = json.loads((ROOT / "summary.json").read_text())["corpus_sha256"]
    if hashlib.sha256(corpus).hexdigest() != digest:
        raise ValueError("archived corpus checksum mismatch")
    rates = {}
    with args.out.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_tokens", "trial", "warmup", "token_sha256", "ttft_s", "input_tokens_per_s"])
        for raw in corpus.splitlines():
            case = json.loads(raw)
            ttft = measure(args.url.rstrip("/"), args.model, case)
            n = case["target_tokens"]
            writer.writerow([n, case["trial"], int(case["warmup"]), case["token_sha256"], ttft, n / ttft])
            f.flush()
            if not case["warmup"]:
                rates.setdefault(n, []).append(n / ttft)
    for n, values in rates.items():
        print(f"{n:5} inputs: median {statistics.median(values):.1f} input tok/s")


if __name__ == "__main__":
    main()
