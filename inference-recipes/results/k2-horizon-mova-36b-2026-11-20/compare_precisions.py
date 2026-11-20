"""Compare two one-Spark K2 Horizon MoVA runs on matched prompts and continuations."""

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median


def rows(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream]


def matched(a, b, keys, count):
    if len(a) != count or len(b) != count:
        raise ValueError(f'expected {count} paired rows')
    identities = [tuple(row[k] for k in keys) for row in a]
    if len(set(identities)) != count or identities != [tuple(row[k] for k in keys) for row in b]:
        raise ValueError(f'inputs differ on {keys}')
    return list(zip(a, b))


def questions(pairs):
    return dict(cases=len(pairs), fp8_correct=sum(a['correct'] for a, _ in pairs), bf16_correct=sum(b['correct'] for _, b in pairs),
                fp8_only_correct=sum(a['correct'] and not b['correct'] for a, b in pairs),
                bf16_only_correct=sum(b['correct'] and not a['correct'] for a, b in pairs),
                matching_answers=sum(a['answer'] == b['answer'] for a, b in pairs))


def compare(fp8, bf16):
    quality = {}
    for name, count in [('grammar112', 112), ('effort-low-128', 14), ('effort-low-512', 14)]:
        a, b = (rows(p / (name + '.jsonl.gz')) for p in (fp8, bf16))
        quality[name] = questions(matched(a, b, ('subject', 'id', 'prompt_sha256', 'expected'), count))
    a, b = (json.loads((p / 'lookup.json').read_bytes())['requests'] for p in (fp8, bf16))
    quality['lookup'] = questions(matched(a, b, ('case', 'prompt_sha256', 'expected'), 8))
    timing = {}
    for p, mode in ((fp8, 'fp8'), (bf16, 'bf16')):
        ts = rows(p / 'trials.jsonl'); energy = rows(next(p.glob('*-energy.jsonl')))
        pairs = matched(ts, energy, ('concurrency', 'repeat', 'start_unix_ns', 'end_unix_ns'), 6)
        timing[mode] = {}
        for c in (1, 4):
            runs = [(t, e) for t, e in pairs if t['concurrency'] == c]
            if len(runs) != 3 or any(t['failed'] or t['completed'] != 4 or e['output_tokens'] <= 0 for t, e in runs):
                raise ValueError('incomplete timed trials')
            timing[mode][str(c)] = dict(output_tokens_s=median(t['output_tokens_per_s'] for t, _ in runs),
                p95_first_delta_ms=median(t['p95_first_delta_ms'] for t, _ in runs),
                gpu_joules_per_output_token=median(e['joules_per_output_token'] for _, e in runs))
    a, b = (rows(p / 'requests.jsonl.gz') for p in (fp8, bf16));matched(a, b, ('concurrency', 'repeat', 'case', 'prompt_sha256'), 24)
    a, b = (rows(p / 'bytes.jsonl') for p in (fp8, bf16));pairs = matched(a, b, ('article_row', 'passage_sha256', 'context_sha256', 'scored_bytes'), 60)
    size = sum(l['scored_bytes'] for l, _ in pairs)
    bpb = lambda items: math.fsum(r['nll_sum'] for r in items) / (size * math.log(2))
    likelihood = dict(passages=len(pairs), scored_bytes=size, fp8_bits_per_byte=bpb(a), bf16_bits_per_byte=bpb(b),
                      bf16_lower_nll_passages=sum(right['nll_sum'] < left['nll_sum'] for left, right in pairs),
                      tied_nll_passages=sum(right['nll_sum'] == left['nll_sum'] for left, right in pairs))
    a, b = (rows(p / 'wikitext.jsonl') for p in (fp8, bf16));pairs = matched(a, b, ('article_row', 'passage_sha256', 'token_ids_sha256', 'scored_tokens'), 62)
    count = sum(left['scored_tokens'] for left, _ in pairs)
    likelihood.update(scored_tokens=count, fp8_perplexity=math.exp(math.fsum(r['nll_sum'] for r in a) / count),
                      bf16_perplexity=math.exp(math.fsum(r['nll_sum'] for r in b) / count))
    return dict(timing=timing, quality=quality, likelihood=likelihood)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fp8', type=Path)
    parser.add_argument('bf16', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.fp8, args.bf16), indent=2))
