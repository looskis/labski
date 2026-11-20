"""Replay two GLM questions with different chat-completions reasoning settings."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

def run(model, corpus, out, url):
    items = {row["id"]: row for row in map(json.loads, corpus.read_text().splitlines())}
    controls = [
        ("default", {}),
        ("top-level", {"enable_thinking": False}),
        ("template", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("effort-none", {"reasoning_effort": "none"}),
        ("both", {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}),
    ]
    with out.open("x") as stream:
        for question, modes in [(770, controls), (3682, reversed(controls))]:
            item = items[question]
            for mode, extra in modes:
                body = dict(model=model, messages=[dict(role="user", content=item["prompt"])],
                            temperature=0, max_tokens=128, **extra)
                req = Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=180) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                msg = choice["message"]
                answer = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                usage = result["usage"]
                assert usage["prompt_tokens"] > 0 and 0 < usage["completion_tokens"] <= 128
                row = dict(id=question, mode=mode, prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                           expected=item["expected"], finish=choice["finish_reason"],
                           correct=choice["finish_reason"]=="stop" and answer.strip()==item["expected"],
                           visible_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                           visible_chars=len(answer), reasoning_chars=len(reasoning),
                           input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"])
                stream.write(json.dumps(row)+"\n")
                stream.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    run(args.model, args.corpus, args.out, args.url)
