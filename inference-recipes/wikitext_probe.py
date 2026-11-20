"""Compare next-token log likelihood on matched WikiText-2 article prefixes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PARQUET_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
ARTICLE_COUNT = 62
WINDOW = 512


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, dest):
    import pyarrow.parquet as parquet

    if digest(source.read_bytes()) != PARQUET_SHA256:
        raise ValueError("WikiText-2 test file differs from the pinned release")
    articles = []
    for row, raw in enumerate(parquet.read_table(source, columns=["text"])["text"].to_pylist()):
        line = raw.strip()
        if re.fullmatch(r"= [^=].*? =", line):
            articles.append({"article_row": row, "text": ""})
        elif line and not line.startswith("= =") and articles:
            article = articles[-1]
            if len(article["text"]) < 8192:
                article["text"] = (article["text"] + "\n" + line).lstrip()[:8192]
    if len(articles) != ARTICLE_COUNT or any(not item["text"] for item in articles):
        raise ValueError("unexpected WikiText-2 article boundaries")
    with dest.open("w") as out:
        for article in articles:
            out.write(json.dumps(article, ensure_ascii=False) + "\n")


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def score(model, url, corpus, dest):
    articles = [json.loads(line) for line in corpus.read_text().splitlines()]
    if len({row["article_row"] for row in articles}) != ARTICLE_COUNT or len(articles) != ARTICLE_COUNT:
        raise ValueError("expected one passage per test article")
    url = url.rstrip("/")
    totals = []
    pending = dest.with_suffix(dest.suffix + ".partial")
    with pending.open("w") as out:
        for article in articles:
            body = dict(model=model, prompt=article["text"], add_special_tokens=False)
            ids = post(url + "/tokenize", body)["tokens"][:WINDOW]
            if len(ids) < 2:
                raise ValueError(f"too few tokens for article row {article['article_row']}")
            reply = post(url + "/v1/completions", dict(model=model, prompt=ids, max_tokens=1,
                         temperature=0, add_special_tokens=False, prompt_logprobs=0))
            scores = reply["choices"][0]["prompt_logprobs"]
            if reply["usage"]["prompt_tokens"] != len(ids) or len(scores) != len(ids) or scores[0] is not None:
                raise ValueError("server did not score the requested token positions")
            logps = [position[str(token)]["logprob"] for token, position in zip(ids[1:], scores[1:])]
            if not all(math.isfinite(x) and x <= 0 for x in logps):
                raise ValueError("invalid target log probability")
            row = dict(article_row=article["article_row"], scored_tokens=len(logps),
                       token_ids_sha256=digest(",".join(map(str, ids)).encode()),
                       passage_sha256=digest(article["text"].encode()),
                       nll_sum=-math.fsum(logps))
            out.write(json.dumps(row) + "\n")
            totals.append(row)
    pending.replace(dest)
    count = sum(row["scored_tokens"] for row in totals)
    mean = math.fsum(row["nll_sum"] for row in totals) / count
    print(json.dumps(dict(articles=len(totals), scored_tokens=count, mean_nll=mean,
                          perplexity=math.exp(mean))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    prep = cmds.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("dest", type=Path)
    run = cmds.add_parser("score")
    run.add_argument("model")
    run.add_argument("corpus", type=Path)
    run.add_argument("dest", type=Path)
    run.add_argument("--url", default="http://127.0.0.1:18200")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.dest)
    else:
        score(args.model, args.url, args.corpus, args.dest)
