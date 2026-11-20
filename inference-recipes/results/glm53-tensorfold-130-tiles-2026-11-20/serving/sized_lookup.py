"""Test exact six-digit retrieval at prefill lengths used by the TensorFold dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

MODEL = "GLM-5.3-Flash"


def post(base, path, body):
    request = urllib.request.Request(base.rstrip("/") + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def completion(base, text):
    body = {**request_body(text), "temperature": 0, "max_tokens": 32,
            "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    parts, usage, finish = [], {}, None
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage.update(event.get("usage") or {})
            for choice in event.get("choices", []):
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    if finish not in ("length", "stop"):
        raise ValueError("reply did not finish")
    return usage, finish, "".join(parts)


def request_body(text):
    return {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False}}


def fixture(base, target, depth, seed):
    rng = random.Random(seed)
    labels = rng.sample(range(1000, 9999), target // 4)
    values = rng.sample(range(100000, 1000000), len(labels))

    def prompt(count, pad=0):
        index = (count - 1) * depth // 100
        listing = "\n".join(f"bin-{k} = {v}" for k, v in zip(labels[:count], values[:count]))
        text = f"Inventory {seed}\n{listing}\nEnd of inventory." + " ." * pad
        return text + f"\nReturn only the six-digit value of bin-{labels[index]}.", str(values[index])

    def length(text):
        return post(base, "/tokenize", {**request_body(text), "add_generation_prompt": True})["count"]

    lo, hi = 1, len(labels)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if length(prompt(mid)[0]) <= target:
            lo = mid
        else:
            hi = mid - 1
    plain, _ = prompt(lo)
    missing = target - length(plain)
    if not 0 <= missing < 40:
        raise ValueError("unexpected token count for inventory")
    for pad in [missing, *[i for i in range(40) if i != missing]]:
        text, expected = prompt(lo, pad)
        if length(text) == target:
            return text, expected, lo, pad
    raise ValueError("could not hit exact input length")


def main(args):
    with args.out.open("x") as f:
        for target in (2048, 4096, 8192):
            for depth in (12, 84):
                for case in range(4):
                    seed = 81000 + case + depth * 100 + target * 1000
                    text, expected, count, pad = fixture(args.base, target, depth, seed)
                    usage, finish, answer = completion(args.base, text)
                    assert usage["prompt_tokens"] == target
                    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
                    record = dict(input_tokens=target, depth_percent=depth, seed=seed,
                                  inventory_records=count, padding_repeats=pad, expected=expected,
                                  prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                                  answer=answer.strip(), finish=finish,
                                  output_tokens=usage["completion_tokens"],
                                  exact=finish == "stop" and answer.strip() == expected)
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
                    f.flush()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    cli.add_argument("--base", default="http://127.0.0.1:18220")
    main(cli.parse_args())
