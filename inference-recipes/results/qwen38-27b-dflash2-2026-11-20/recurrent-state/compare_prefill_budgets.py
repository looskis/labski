"""Compare Qwen next-token choices across two prefill budgets and two state dtypes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from compare_next_token import read_cases, read_run


def compare(windows, fp32_low, bf16_low, fp32_high, bf16_high):
    cases = read_cases(windows)
    runs = [read_run(path, cases)[:-1] for path in (fp32_low, bf16_low, fp32_high, bf16_high)]
    labels = ['state_at_2048', 'state_at_4096', 'budget_in_fp32', 'budget_in_bf16']
    totals, direct, masks = Counter(), Counter(), Counter()
    by_window = []
    for case, rows in zip(cases, zip(*runs)):
        first = {label: None for label in labels}
        changed = Counter()
        for pos, (a, b, c, d) in enumerate(zip(*(row['top_ids'] for row in rows)), 1):
            differences = (a != b, c != d, a != c, b != d)
            for label, differs in zip(labels, differences):
                if differs:
                    changed[label] += 1
                    if first[label] is None:
                        first[label] = pos
            masks[(differences[0], differences[1])] += 1
        totals.update(changed)
        for prefix_rows in zip(*(row['prefixes'] for row in rows)):
            a, b, c, d = (r['generated_id'] for r in prefix_rows)
            for label, differs in zip(labels, (a != b, c != d, a != c, b != d)):
                direct[label] += differs
        by_window.append(dict(start=case['start'], first_changed_position=first,
                              changed={label: changed[label] for label in labels}))
    return dict(scored_positions=sum(masks.values()), direct_prefixes=sum(len(row['prefixes']) for row in runs[0]),
                changed_rankings={label: totals[label] for label in labels},
                changed_direct_choices={label: direct[label] for label in labels},
                precision_disagreement_positions=dict(neither_budget=masks[False, False], only_2048=masks[True, False],
                                                     only_4096=masks[False, True], both_budgets=masks[True, True]),
                by_window=by_window)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('windows', 'fp32_at_2048', 'bf16_at_2048', 'fp32_at_4096', 'bf16_at_4096'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.windows, args.fp32_at_2048, args.bf16_at_2048,
                             args.fp32_at_4096, args.bf16_at_4096), indent=2))
