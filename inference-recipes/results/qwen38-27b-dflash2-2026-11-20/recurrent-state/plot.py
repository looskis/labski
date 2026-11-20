"""Plot measured four-client throughput and paired long-context top-token changes."""

import gzip
import json
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

HERE = Path(__file__).resolve().parent


def trials(folder):
    rows = [json.loads(line) for line in (HERE / folder / 'trials.jsonl').read_text().splitlines()]
    values = [r['output_tokens_per_s'] for r in rows if r['concurrency'] == 4]
    if len(values) != 3:
        raise ValueError(f'expected three trials in {folder}')
    return values


def direct_rankings(directory, dtype):
    with gzip.open(directory / f'next-{dtype}.jsonl.gz', 'rt') as scores:
        return [r for r in map(json.loads, scores) if not r['repeated']]


def disagreements(directory, edges):
    fp32, bf16 = direct_rankings(directory, 'fp32'), direct_rankings(directory, 'bf16')
    if len(fp32) != 8 or [r['token_ids_sha256'] for r in fp32] != [r['token_ids_sha256'] for r in bf16]:
        raise ValueError('unmatched long-context runs')
    fractions = []
    for left, right in zip(edges, edges[1:]):
        # top_ids[0] is the prediction of target position 1. Position 0 is unscored.
        section = slice(max(0, left - 1), right - 1)
        choices = [x != y for a, b in zip(fp32, bf16) for x, y in zip(a['top_ids'][section], b['top_ids'][section])]
        fractions.append(sum(choices) / len(choices))

    return fractions


def main():
    edges = list(range(0, 6145, 256))
    low = disagreements(HERE / 'long-context', edges)
    high = disagreements(HERE / 'long-context' / 'prefill-4096', edges)
    fg, guide, control, candidate, changed = '#21333c', '#dce3e6', '#586c79', '#08745e', '#9c520d'
    fig, (speed, agreement) = plt.subplots(1, 2, figsize=(11, 4.65), gridspec_kw={'width_ratios': [1, 1.45]})
    fig.subplots_adjust(left=.123, right=.97, bottom=.235, top=.72, wspace=.31)
    fig.text(.045, .91, 'Qwen3.8 27B recurrent-state precision on one DGX Spark', fontsize=17, weight='bold', color=fg)
    fig.text(.045, .837, 'Mixed NVFP4/FP8 weights   ·   Seven DFlash2 proposals   ·   3 GiB cache', fontsize=10, color=control)
    for ax in (speed, agreement):
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color(guide)
        ax.tick_params(axis='both', labelsize=9, color=guide, labelcolor=fg)
        ax.set_axisbelow(True)
    speed.set_title('Four-client output throughput', fontsize=12, weight='bold', loc='left', pad=9, color=fg)
    for y, folder, color in [(2, 'fp32', control), (1, 'bf16', candidate), (0, 'bf16-repeat', candidate)]:
        values = trials(folder)
        speed.hlines(y, min(values), max(values), lw=3, color=color, alpha=.52)
        speed.scatter(median(values), y, s=50, color=color, zorder=3)
        speed.annotate(f'{median(values):.2f}', (max(values), y), xytext=(5, 0), textcoords='offset points', va='center', fontsize=9, color=fg)
    speed.set(xlim=(0, 126), ylim=(-.6, 2.65), xticks=[0, 25, 50, 75, 100], xlabel='Generated tokens/s')
    speed.set_yticks([2, 1, 0], ['FP32', 'BF16', 'BF16 repeat'])
    speed.tick_params(axis='y', length=0)
    speed.grid(axis='x', color=guide, lw=.8)
    agreement.set_title('Long-context top-token disagreement', fontsize=12, weight='bold', loc='left', pad=9, color=fg)
    agreement.grid(axis='y', color=guide, lw=.8)
    for boundary in (2048, 4096):
        agreement.axvline(boundary, color=control, lw=.85, ls='--')
    agreement.stairs(low, edges, baseline=None, color=changed, lw=2.2, label='2,048-token budget')
    agreement.stairs(high, edges, baseline=None, color='#22687d', lw=2, linestyle='--', label='4,096-token budget')
    ceiling = max(4, math.ceil(max(max(low), max(high)) / .03))
    agreement.set(xlim=(0, 6144), ylim=(-.005, ceiling * .03 + .007), xticks=[0, 2048, 4096, 6144],
                  yticks=[i * .03 for i in range(ceiling + 1)], xlabel='Target token position')
    agreement.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=8.3, labelcolor=fg)
    agreement.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    agreement.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.text(.045, .063, 'Left shows three-trial medians and ranges at the 2,048-token budget. Right pools eight windows in 256-position bins.', fontsize=9, color=control)
    fig.savefig(HERE / 'recurrent-state.png', dpi=165, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
