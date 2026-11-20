"""Measure serial Kimi requests using the native server's token counts and Linux I/O counters."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


QUESTIONS = {
    "sum": ("What is seven plus five? Reply with just the number.", "12"),
    "city": ("Which city is the capital of Japan? Reply with just the city name.", "Tokyo"),
    "lookup": ("Inventory: oak = 4291, lime = 7163, pine = 2857. What is the value of lime? Reply with just the number.", "7163"),
}
TIMING = re.compile(r"ds4: K3 serve: (\d+) prompt tokens, (\d+) generated in [\d.]+ s")


def counters(pid, device):
    io = dict(line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
    fields = (Path("/sys/class/block") / device / "stat").read_text().split()
    return {"process_read_bytes": int(io["read_bytes"]), "device_read_bytes": 512 * int(fields[2])}


def native_counts(log_path, offset):
    deadline = time.monotonic() + 5
    timings = []
    with log_path.open("rb") as log:
        while not timings and time.monotonic() < deadline:
            log.seek(offset)
            timings = TIMING.findall(log.read().decode(errors="replace"))
            if not timings:
                time.sleep(0.05)
    if len(timings) != 1:
        raise RuntimeError("expected exactly one native timing line during this request")
    return tuple(map(int, timings[0]))


def run(args, model, case, ordinal):
    prompt, expected = QUESTIONS[case]
    payload = dict(model=model, messages=[dict(role="user", content=prompt)], thinking=False,
                   temperature=0, seed=42, max_tokens=args.max_tokens, stream=True)
    request = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start_offset = args.server_log.stat().st_size
    before = counters(args.pid, args.device)
    first_delta = first_text = finish = None
    visible = []
    done = False
    start_epoch = time.time_ns()
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = line[5:].strip()
            if event == b"[DONE]":
                done = True
                break
            choices = json.loads(event)["choices"]
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            elapsed = time.perf_counter() - start
            if first_delta is None and (content or delta.get("reasoning_content") or delta.get("reasoning")):
                first_delta = elapsed
            if content:
                if first_text is None:
                    first_text = elapsed
                visible.append(content)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    wall = time.perf_counter() - start
    end_epoch = time.time_ns()
    after = counters(args.pid, args.device)
    if not done or first_delta is None or finish not in ("stop", "length"):
        raise RuntimeError("incomplete server response")
    input_tokens, output_tokens = native_counts(args.server_log, start_offset)
    if input_tokens < 1 or not 1 <= output_tokens <= args.max_tokens:
        raise RuntimeError("invalid native token counts")
    answer = "".join(visible)
    return dict(ordinal=ordinal, case=case, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                prompt=prompt, expected=expected, answer=answer,
                exact=finish == "stop" and answer.strip().casefold() == expected.casefold(),
                finish=finish, max_tokens=args.max_tokens, input_tokens=input_tokens, output_tokens=output_tokens,
                before=before, after=after, first_delta_s=first_delta, first_text_s=first_text, wall_s=wall,
                start_unix_ns=start_epoch, end_unix_ns=end_epoch, output_tokens_per_s=output_tokens / wall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--pid", type=int, required=True, help="host PID of the native server")
    parser.add_argument("--device", required=True, help="block device containing the expert store")
    parser.add_argument("--server-log", type=Path, required=True, help="combined native stderr from this server")
    parser.add_argument("--order", nargs="+", choices=list(QUESTIONS), required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/v1/models", timeout=15) as response:
        model = json.load(response)["data"][0]["id"]
    with args.out.open("x") as stream:
        for ordinal, case in enumerate(args.order, 1):
            row = run(args, model, case, ordinal)
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
