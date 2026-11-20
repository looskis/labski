"""Measure all ten label scores for the same first generated token in each checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mmlu_probe


def score(url, model, labels, item):
    body = dict(model=model, messages=[dict(role="user", content=item["prompt"])], temperature=0,
                max_tokens=1, allowed_token_ids=labels, logprobs=True, logprob_token_ids=labels,
                return_tokens_as_token_ids=True, return_token_ids=True,
                chat_template_kwargs={"enable_thinking": False})
    start = time.perf_counter()
    data = mmlu_probe.post(url+"/v1/chat/completions", body)
    elapsed = time.perf_counter()-start
    choice = data["choices"][0]
    generated = choice["token_ids"]
    if data["usage"]["completion_tokens"] != 1 or len(generated)!=1 or generated[0] not in labels:
        raise ValueError(f"expected exactly one allowed answer token, received {generated!r}")
    scores = {}
    for entry in choice["logprobs"]["content"][0]["top_logprobs"]:
        token = entry["token"]
        if token.startswith("token_id:"):
            numeric = int(token.split(":",1)[1])
            if numeric not in labels:
                continue
            letter = mmlu_probe.LETTERS[labels.index(numeric)]
        elif token in tuple(mmlu_probe.LETTERS):
            letter = token
        else:
            continue
        scores[letter] = entry["logprob"]
    if set(scores) != set(mmlu_probe.LETTERS) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f"expected ten finite label logprobs, received {scores!r}")
    answer = mmlu_probe.LETTERS[labels.index(generated[0])]
    ranks = sorted(scores.values(), reverse=True)
    if scores[answer] != ranks[0]:
        raise ValueError("generated label differs from the largest returned label score")
    ids = data.get("prompt_token_ids")
    if not isinstance(ids, list) or len(ids)!=data["usage"]["prompt_tokens"]:
        raise ValueError("missing tokenized prompt for the two-checkpoint comparison")
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(",",":")).encode()).hexdigest()
    return dict(id=item["id"], subject=item["subject"], prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                input_ids_sha256=ids_hash, input_tokens=len(ids), expected=item["expected"], answer=answer,
                correct=answer==item["expected"], answer_token_id=generated[0], label_ids=labels, logprobs=scores,
                margin=ranks[0]-ranks[1], elapsed_s=elapsed, model=model)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("corpus", type=Path)
    cli.add_argument("--url", default="http://127.0.0.1:8000")
    cli.add_argument("--out", type=Path)
    cli.add_argument("--worker", action="store_true", help="accept corpus IDs on stdin and emit live scored responses")
    args = cli.parse_args()
    items = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    labels = mmlu_probe.letter_ids(args.url,args.model)
    if args.worker:
        cases = {str(i["id"]):i for i in items}
        print("READY",flush=True)
        for line in sys.stdin:
            print(json.dumps(score(args.url,args.model,labels,cases[line.strip()]),separators=(",",":")),flush=True)
        return
    if args.out is None:
        cli.error("--out is required in batch mode")
    index = Counter()
    with args.out.open("x") as out:
        for item in items:
            row = score(args.url,args.model,labels,item)
            position = index[item["subject"]]
            row["split"] = "original" if position<8 else "held-out"
            index[item["subject"]] += 1
            out.write(json.dumps(row,separators=(",",":"))+"\n")
            out.flush()


if __name__ == "__main__":
    main()
