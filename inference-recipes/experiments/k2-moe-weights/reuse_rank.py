"""Build the opposite K2 expert rank using a verified compact shard and upstream ranges."""

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import fetch_shard
from assemble import verify_file
from slice_shard import read_header


def shared_spans(original, compact, original_header_size, compact_header_size):
    spans = []
    for key, field in sorted(compact.items(), key=lambda item: item[1]['data_offsets']):
        full = original[key]
        start, end = full['data_offsets']
        local, stop = field['data_offsets']
        if field['shape'] != full['shape'] or field['dtype'] != full['dtype'] or end - start != stop - local:
            raise ValueError(f'inconsistent cached tensor {key}')
        start += original_header_size
        end += original_header_size
        local += compact_header_size
        if spans and spans[-1][1] == start and local == spans[-1][2] + spans[-1][1] - spans[-1][0]:
            spans[-1][1] = end
        else:
            spans.append([start, end, local])
    return spans


def restore(sibling, out, rank):
    if rank not in (0, 1):
        raise ValueError('choose destination rank 0 or 1')
    name = sibling.name
    if (out / name).exists():
        return fetch_shard.fetch(name, rank, out)
    proof = json.loads((fetch_shard.HERE / 'two-rank-parity.json').read_bytes())
    record = next((r for r in proof['shards'] if r['rank'] == 1 - rank and r['file'] == name), None)
    if record is None:
        raise ValueError('choose a compact shard from the verified K2 revision')
    meta = next(r for r in json.loads((fetch_shard.HERE / '375b-nvfp4.json').read_bytes())['indexed_files'] if r['file'] == name)
    verify_file(sibling, record)
    with sibling.open('rb') as src:
        compact_header, fields = read_header(src)
        fields.pop('__metadata__', None)
        spans, ends = [], []
        counts = dict(upstream_response_bytes=0, successful_http_ranges=0, cached_compact_bytes_read=0)

        def remote(start, end):
            data = fetch_shard.get_range(name, start, end, meta['file_bytes'])
            counts['upstream_response_bytes'] += len(data)
            counts['successful_http_ranges'] += 1
            return data

        def read(requested, start, end, size):
            if requested != name or size != meta['file_bytes']:
                raise ValueError('unexpected source file or size')
            if start == 0:
                header = remote(start, end)
                if end != meta['header_bytes'] or hashlib.sha256(header).hexdigest() != meta['header_sha256']:
                    raise ValueError('unexpected source header')
                spans[:] = shared_spans(json.loads(header[8:]), fields, len(header), len(compact_header))
                ends[:] = [span[1] for span in spans]
                return header
            pos = start
            blocks = bytearray()
            for index in range(bisect_right(ends, start), len(spans)):
                shared_start, shared_end, local = spans[index]
                if shared_start >= end:
                    break
                lo, hi = max(start, shared_start), min(end, shared_end)
                # Keep tiny gaps in a larger HTTP range to avoid extra requests.
                if hi - lo <= fetch_shard.GAP and (lo != start or hi != end):
                    continue
                if pos < lo:
                    blocks.extend(remote(pos, lo))
                src.seek(local + lo - shared_start)
                part = src.read(hi - lo)
                if len(part) != hi - lo:
                    raise OSError('truncated cached shard')
                blocks.extend(part)
                counts['cached_compact_bytes_read'] += len(part)
                pos = hi
            if pos < end:
                blocks.extend(remote(pos, end))
            if len(blocks) != end - start:
                raise ValueError('reconstructed range length differs from request')
            return blocks

        result = fetch_shard.fetch(name, rank, out, read_range=read)
    return dict(**result, sibling_rank=1 - rank, **counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('opposite_rank_shard', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--rank', type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    print(json.dumps(restore(args.opposite_rank_shard, args.destination, args.rank), indent=2))
