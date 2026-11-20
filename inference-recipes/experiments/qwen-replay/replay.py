"""Replay a sequence of code edits or unrelated prose on an OpenAI-compatible vLLM server."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from urllib.request import Request, urlopen


PROSE = (
    "How does a sundial show local solar time?",
    "Describe a weaver's choices when repairing a torn basket.",
    "Explain the role of echoes when a bat tracks prey.",
    "How would a museum store a fragile watercolor?",
    "Write a guide to replacing a broken bicycle chain.",
    "Describe how a tide table helps a harbor pilot.",
    "Explain the work of yeast in a sourdough starter.",
    "How does a translator handle an idiom in a novel?",
    "Describe the preparation of a telescope before observing Saturn.",
    "How do gardeners protect tree roots during a dry spell?",
    "Explain how a film composer uses silence.",
    "Describe the steps of making paper from cotton rags.",
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def case(kind, turn, session):
    if kind == "prose":
        return PROSE[turn * 4 + session] + " Write a detailed explanation.", None
    fields = ("subtotal", "balance", "credit", "reserve")
    field = fields[session]
    name = ("orchard", "harbor", "library", "workshop")[session]
    code = (
        f"def {name}_total(entries):\n"
        f"    {field} = 0\n"
        "    for quantity, unit_price in entries:\n"
        "        if quantity > 0:\n"
        f"            {field} += quantity * unit_price\n"
        f"    return {field} + {10 + turn}\n"
    )
    value = 11 + turn
    prompt = (f"Edit this Python function. Change only the final additive constant to {value}. "
              "Return the whole function as plain Python with no Markdown or commentary.\n\n" + code)
    return prompt, code.replace(f" + {value - 1}\n", f" + {value}\n")


def matches_edit(answer, expected):
    if expected is None:
        return None
    try:
        return ast.dump(ast.parse(answer.strip())) == ast.dump(ast.parse(expected.strip()))
    except (SyntaxError, ValueError):
        return False


def draft_counts(url):
    pattern = re.compile(r'^vllm:spec_decode_(num_(?:drafts|draft_tokens|accepted_tokens))_total(?:\{[^}]*\})?\s+(\S+)')
    counts = {}
    with urlopen(url + "/metrics", timeout=15) as response:
        for line in response:
            match = pattern.match(line.decode())
            if match:
                name, value = match.groups()
                counts[name] = counts.get(name, 0) + int(float(value))
    return counts


def request(args, turn, session, warmup=False):
    prompt, expected = case(args.workload, turn, session)
    if warmup:
        prompt, expected = "Reply with the single word ready.", None
    body = dict(model=args.model, messages=[dict(role="user", content=prompt)], temperature=0,
                max_tokens=192 if args.workload == "edits" else 96, stream=True,
                stream_options={"include_usage": True}, return_token_ids=True,
                chat_template_kwargs={"enable_thinking": False})
    req = Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    ids, text, events = [], [], []
    usage = finish = None
    done = False
    with urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            data = json.loads(raw)
            if data.get("error"):
                raise ValueError(data["error"])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                tokens = choice.get("token_ids") or []
                if tokens:
                    ids.extend(tokens)
                    events.append((time.perf_counter() - started, len(tokens)))
                content = choice.get("delta", {}).get("content")
                if content:
                    text.append(content)
                finish = choice.get("finish_reason") or finish
    if not done or finish not in ("stop", "length") or not usage or not ids:
        raise ValueError(f"incomplete stream, finish={finish!r}, output IDs={len(ids)}")
    if len(ids) != usage["completion_tokens"] or not 0 < len(ids) <= body["max_tokens"]:
        raise ValueError(f"token IDs/usage mismatch: {len(ids)} vs {usage}")
    answer = "".join(text)
    token_interval = (events[-1][0] - events[0][0]) / (len(ids) - 1) if len(ids) > 1 else None
    return dict(turn=turn, session=session, prompt_sha256=digest(prompt),
                output_ids_sha256=digest(json.dumps(ids, separators=(",", ":"))),
                visible_sha256=digest(answer), edit_exact=matches_edit(answer, expected),
                input_tokens=usage["prompt_tokens"], output_tokens=len(ids), finish=finish,
                elapsed_s=time.perf_counter()-started, first_token_s=events[0][0],
                mean_token_interval_s=token_interval, stream_events=events)


def run(args):
    # Fail before sending traffic if this receipt already exists.
    with args.out.open("x") as output:
        request(args, 0, 0, warmup=True)
        with ThreadPoolExecutor(args.clients) as pool:
            for turn in range(3):
                before = draft_counts(args.url)
                start_ns = time.time_ns()
                start = time.perf_counter()
                rows = list(pool.map(lambda session: request(args, turn, session), range(4)))
                wall = time.perf_counter() - start
                end_ns = time.time_ns()
                after = draft_counts(args.url)
                record = dict(workload=args.workload, clients=args.clients, turn=turn,
                              start_unix_ns=start_ns, end_unix_ns=end_ns, wall_s=wall,
                              draft_counters={k: v - before.get(k, 0) for k, v in after.items()},
                              output_tokens_per_s=sum(r["output_tokens"] for r in rows)/wall,
                              median_first_token_s=statistics.median(r["first_token_s"] for r in rows),
                              requests=rows)
                output.write(json.dumps(record)+"\n")
                output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", choices=("edits", "prose"), required=True)
    parser.add_argument("--clients", type=int, choices=(1,4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
