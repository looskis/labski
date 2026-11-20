"""Compare reported top tokens and direct greedy choices against pinned token windows."""

import argparse
import gzip
import json
from pathlib import Path

from next_token import PREFIXES, STARTS, WIDTH, ids_digest


def read_run(path, windows):
    with gzip.open(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    cases = [*windows, windows[0]]
    if len(rows) != len(cases):
        raise ValueError(f'expected eight windows and one repeat in {path}')
    for call, (row, case) in enumerate(zip(rows, cases)):
        ids, ranked = case['ids'], row['top_ids']
        if (row['call'] != call or row['start'] != case['start'] or row['repeated'] != (call == len(windows))
                or row['token_ids_sha256'] != case['token_ids_sha256'] or len(ranked) != WIDTH - 1
                or any(not top or top != sorted(set(top)) for top in ranked)
                or sum(token in top for token, top in zip(ids[1:], ranked)) != row['observed_in_top']):
            raise ValueError(f'invalid ranking for call {call} in {path}')
        if [p['prefix_tokens'] for p in row['prefixes']] != list(PREFIXES):
            raise ValueError(f'unexpected prefix lengths in {path}')
        for p in row['prefixes']:
            length = p['prefix_tokens']
            if p['prefix_ids_sha256'] != ids_digest(ids[:length]) or p['observed_next_id'] != ids[length]:
                raise ValueError(f'unexpected direct probe at {length} in {path}')
    return rows


def paired(fp32, bf16, cases):
    positions = [(a, b, token) for left, right, case in zip(fp32, bf16, cases)
                 for a, b, token in zip(left['top_ids'], right['top_ids'], case['ids'][1:])]
    correct = [(token in a, token in b) for a, b, token in positions]
    probes = [(a['generated_id'], b['generated_id'], a['observed_next_id'],
               left['top_ids'][a['prefix_tokens'] - 1], right['top_ids'][b['prefix_tokens'] - 1])
              for left, right in zip(fp32, bf16) for a, b in zip(left['prefixes'], right['prefixes'])]
    return dict(scored=len(positions), same_reported_top=sum(a == b for a, b, _ in positions),
                fp32_top_with_multiple_ids=sum(len(a) > 1 for a, _, _ in positions),
                bf16_top_with_multiple_ids=sum(len(b) > 1 for _, b, _ in positions),
                both_correct=sum(a and b for a, b in correct), neither_correct=sum(not a and not b for a, b in correct),
                fp32_only_correct=sum(a and not b for a, b in correct), bf16_only_correct=sum(b and not a for a, b in correct),
                direct_probes=len(probes), same_direct=sum(a == b for a, b, *_ in probes),
                fp32_direct_correct=sum(a == t for a, _, t, _, _ in probes),
                bf16_direct_correct=sum(b == t for _, b, t, _, _ in probes),
                fp32_only_direct_correct=sum(a == t and b != t for a, b, t, _, _ in probes),
                bf16_only_direct_correct=sum(b == t and a != t for a, b, t, _, _ in probes),
                fp32_direct_in_reported_top=sum(a in top for a, _, _, top, _ in probes),
                bf16_direct_in_reported_top=sum(b in top for _, b, _, _, top in probes))


def compare(windows, fp32, bf16):
    cases = [json.loads(line) for line in windows.read_text().splitlines()]
    if ([c['start'] for c in cases] != list(STARTS) or
            any(len(c['ids']) != WIDTH or ids_digest(c['ids']) != c['token_ids_sha256'] for c in cases)):
        raise ValueError('invalid input windows')
    a, b = read_run(fp32, cases), read_run(bf16, cases)
    repeats = {}
    for dtype, run in [('fp32', a), ('bf16', b)]:
        first, again = run[0], run[-1]
        repeats[dtype] = dict(scored=WIDTH - 1, same_reported_top=sum(x == y for x, y in zip(first['top_ids'], again['top_ids'])),
                              direct_probes=len(PREFIXES), same_direct=sum(x['generated_id'] == y['generated_id']
                              for x, y in zip(first['prefixes'], again['prefixes'])))
    chunks = []
    for first, last in ((1, 2047), (2048, 4095), (4096, 6143)):
        pairs = [(left, right) for x, y in zip(a[:-1], b[:-1])
                 for left, right in zip(x['top_ids'][first - 1:last], y['top_ids'][first - 1:last])]
        chunks.append(dict(first_position=first, last_position=last, scored=len(pairs),
                           same_reported_top=sum(left == right for left, right in pairs)))
    by_prefix = []
    for i, length in enumerate(PREFIXES):
        pairs = [(x['prefixes'][i]['generated_id'], y['prefixes'][i]['generated_id'], c['ids'][length])
                 for x, y, c in zip(a, b, cases)]
        by_prefix.append(dict(prefix_tokens=length, direct_probes=len(pairs), same_direct=sum(x == y for x, y, _ in pairs),
                              fp32_direct_correct=sum(x == t for x, _, t in pairs),
                              bf16_direct_correct=sum(y == t for _, y, t in pairs)))
    return dict(overall=paired(a[:-1], b[:-1], cases), first_window_repeat=repeats,
                nominal_chunks=chunks, direct_by_prefix_length=by_prefix,
                windows=[dict(start=c['start'], first_changed_target_position=next((i for i, (u, v) in
                              enumerate(zip(x['top_ids'], y['top_ids']), 1) if u != v), None),
                              **paired([x], [y], [c])) for x, y, c in zip(a, b, cases)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('windows', 'fp32', 'bf16'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.windows, args.fp32, args.bf16), indent=2))
