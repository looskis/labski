"""Measure cross-tokenizer likelihood over identical UTF-8 continuations."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wikitext_probe import ARTICLE_COUNT, WINDOW, post


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ids_hash(ids):
    return sha256(",".join(map(str, ids)).encode()).hexdigest()


def write(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args):
    from tokenizers import Tokenizer
    reference, candidate = (Tokenizer.from_file(str(p)) for p in (args.reference, args.candidate))
    corpus = records(args.articles)
    original = {a["article_row"]: a for a in records(args.scores)}
    if len(corpus) != ARTICLE_COUNT or {a["article_row"] for a in corpus} != original.keys():
        raise ValueError("expected the same 62 original articles and scores")

    selected, excluded = [], []
    for article in corpus:
        key, text = article["article_row"], article["text"]
        old = original[key]
        ids = reference.encode(text, add_special_tokens=False).ids[:WINDOW]
        if ids_hash(ids) != old["token_ids_sha256"] or sha256(text.encode()).hexdigest() != old["passage_sha256"]:
            raise ValueError(f"reference tokenizer or original text differs at {key}")
        context = reference.decode(ids[:1], skip_special_tokens=False)
        passage = reference.decode(ids, skip_special_tokens=False)
        if not context or not passage.startswith(context) or not text.startswith(passage):
            raise ValueError(f"reference failed to reconstruct article prefix {key}")
        if reference.encode(passage, add_special_tokens=False).ids != ids:
            raise ValueError(f"reference prefix failed to round-trip at {key}")
        c = candidate.encode(context, add_special_tokens=False).ids
        whole = candidate.encode(passage, add_special_tokens=False).ids
        if candidate.decode(whole, skip_special_tokens=False) != passage:
            raise ValueError(f"candidate tokenizer changed the passage at {key}")
        if not c or whole[:len(c)] != c or candidate.decode(c, skip_special_tokens=False) != context:
            excluded.append(key)
            continue
        selected.append(dict(article_row=key, context=context, passage=passage,
                             original_text_sha256=old["passage_sha256"], reference_ids_sha256=ids_hash(ids),
                             candidate_ids_sha256=ids_hash(whole), candidate_context_tokens=len(c)))
    write(args.out, selected)
    print(json.dumps(dict(included=len(selected), excluded_boundary_rows=excluded)))


def measure(args):
    rows = []
    corpus = records(args.corpus)
    if not corpus or len({r["article_row"] for r in corpus}) != len(corpus):
        raise ValueError("expected distinct article rows")
    original = {r["article_row"]: r for r in records(args.scores)} if args.action == "project" else {}
    for item in corpus:
        key, passage, context = item["article_row"], item["passage"], item["context"]
        if not passage.startswith(context) or not context or passage == context:
            raise ValueError(f"invalid byte boundary at {key}")
        if args.action == "project":
            old = original[key]
            if old["passage_sha256"] != item["original_text_sha256"] or old["token_ids_sha256"] != item["reference_ids_sha256"]:
                raise ValueError(f"archived scores used a different reference window at {key}")
            loss, tokens = old["nll_sum"], old["scored_tokens"]
        else:
            url = args.url.rstrip("/")
            def tokenize(text):
                return post(url + "/tokenize", dict(model=args.model, prompt=text, add_special_tokens=False))["tokens"]
            ids, prefix = tokenize(passage), tokenize(context)
            if ids_hash(ids) != item["candidate_ids_sha256"] or len(prefix) != item["candidate_context_tokens"] or ids[:len(prefix)] != prefix:
                raise ValueError(f"endpoint tokenizer differs at {key}")
            reply = post(url + "/v1/completions", dict(model=args.model, prompt=ids, max_tokens=1, temperature=0,
                                                      add_special_tokens=False, prompt_logprobs=0))
            scores = reply["choices"][0]["prompt_logprobs"]
            if len(scores) != len(ids) or len(ids) != reply["usage"]["prompt_tokens"] or scores[0] is not None:
                raise ValueError(f"incomplete token scores at {key}")
            logps = [scores[i][str(ids[i])]["logprob"] for i in range(len(prefix), len(ids))]
            if not all(math.isfinite(x) and x <= 0 for x in logps):
                raise ValueError(f"invalid token scores at {key}")
            loss, tokens = -math.fsum(logps), len(logps)
        if not math.isfinite(loss) or loss < 0 or tokens < 1:
            raise ValueError(f"invalid aggregated likelihood at {key}")
        rows.append(dict(article_row=key, context_sha256=sha256(context.encode()).hexdigest(),
                         passage_sha256=sha256(passage.encode()).hexdigest(), scored_tokens=tokens,
                         scored_bytes=len(passage.encode()) - len(context.encode()), nll_sum=loss))
    write(args.out, rows)
    count = sum(r["scored_bytes"] for r in rows)
    print(json.dumps(dict(articles=len(rows), scored_bytes=count,
                          bits_per_byte=math.fsum(r["nll_sum"] for r in rows) / (math.log(2) * count))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    prep = modes.add_parser("prepare")
    for name in ("articles", "scores", "reference", "candidate", "out"):
        prep.add_argument(name, type=Path)
    project = modes.add_parser("project")
    project.add_argument("scores", type=Path)
    live = modes.add_parser("score")
    live.add_argument("model")
    live.add_argument("--url", default="http://127.0.0.1:8000")
    for mode in (project, live):
        mode.add_argument("corpus", type=Path)
        mode.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        measure(args)
