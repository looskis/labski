"""Read the pinned target and draft indexes and SafeTensors headers for two four-Spark candidates."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile
import urllib.request

MODELS = {
    "ax": ("skt/A.X-K2-NVFP4", "9e2e804e80f8d1b3afba5d7938173cec1ed46b49"),
    "exaone": ("LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-NVFP4", "360716a7e9cb1161906061d23fb9c68fe3767d6c"),
    "ax-draft": ("skt/A.X-K2-DSpark", "0b278334dfce313dd54dca45b79c5df40f7bf212"),
    "exaone-draft": ("LGAI-EXAONE/K-EXAONE-2.0-750B-A37B-DSpark", "e42409dbd652ade0bc043fd85ef0c86044f97b46"),
}
ROUTED = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def read(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=90) as res:
        return res.read()


def metadata(root, name, files):
    raw = read(root + name)
    info = files[name]
    if info.get("lfs"):
        valid = hashlib.sha256(raw).hexdigest() == info["lfs"]["sha256"]
    else:
        valid = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == info["blobId"]
    if len(raw) != info["size"] or not valid:
        raise ValueError(f"invalid upstream metadata: {name}")
    return json.loads(raw)


def header(root, name, size, cache):
    path = cache / (name + ".header")
    if path.exists():
        return path.read_bytes()

    def fetch(a, b):
        url = root + name + f"?download=true&range={a}-{b}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b-1}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=90) as r:
            if r.status != 206 or r.headers.get("Content-Range") != f"bytes {a}-{b-1}/{size}":
                raise ValueError(f"invalid range for {name}")
            body = r.read(b - a + 1)
        if len(body) != b - a:
            raise ValueError(f"truncated range for {name}")
        return body

    prefix = fetch(0, 8)
    length, = struct.unpack("<Q", prefix)
    if not 2 <= length <= 100_000_000:
        raise ValueError(f"invalid header size in {name}")
    data = prefix + fetch(8, 8 + length)
    with tempfile.NamedTemporaryFile(dir=cache) as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"concurrent header mismatch: {name}")
    return data


def probe(tag, cache):
    repo, rev = MODELS[tag]
    info = json.loads(read(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"))
    if info["sha"] != rev:
        raise ValueError("snapshot revision mismatch")
    files = {r["rfilename"]: r for r in info["siblings"] if "/" not in r["rfilename"]}
    root = f"https://huggingface.co/{repo}/resolve/{rev}/"
    config = metadata(root, "config.json", files)
    indexes = defaultdict(set)
    if "model.safetensors.index.json" in files:
        mapping = metadata(root, "model.safetensors.index.json", files)["weight_map"]
        for k, v in mapping.items():
            indexes[v].add(k)
    names = {n for n in files if n.endswith(".safetensors")}
    if indexes and set(indexes) != names:
        raise ValueError("unexpected weight files outside the index")
    selected = ["model-dspark.safetensors"] if tag == "exaone-draft" else sorted(names)
    cache = cache / rev
    cache.mkdir(parents=True, exist_ok=True)
    def one(name):
        return name, header(root, name, files[name]["size"], cache)
    groups = defaultdict(Counter)
    layers = defaultdict(Counter)
    placements = Counter()
    shard_results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, raw in pool.map(one, selected):
            fields = json.loads(raw[8:])
            fields.pop("__metadata__", None)
            if indexes and fields.keys() != indexes[name]:
                raise ValueError(f"header and index differ for {name}")
            if len(raw) != 8 + struct.unpack("<Q", raw[:8])[0]:
                raise ValueError("invalid cached header")
            end = 0
            counts = Counter()
            for key, field in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
                a, b = field["data_offsets"]
                if a != end or b < a:
                    raise ValueError(f"invalid offsets: {key}")
                end = b
                n = b - a
                match = ROUTED.match(key)
                if tag.endswith("-draft") or key.startswith("mtp."):
                    kind = "draft"
                elif match:
                    kind = "target_routed_experts"
                    layer, expert = map(int, match.groups())
                    if not 0 <= expert < 256:
                        raise ValueError("unexpected expert ID")
                    placements[expert // 64] += n
                    layers[layer][field["dtype"]] += n
                elif ".mlp.experts." in key:
                    raise ValueError(f"unrecognized expert tensor: {key}")
                else:
                    kind = "target_common"
                groups[kind][field["dtype"]] += n
                counts[kind] += n
            if end + len(raw) != files[name]["size"]:
                raise ValueError(f"header does not describe full shard: {name}")
            shard_results.append(dict(file=name, file_bytes=files[name]["size"], header_bytes=len(raw),
                    header_sha256=hashlib.sha256(raw).hexdigest(), tensors=len(fields), payload_bytes=dict(counts)))
    return dict(model=repo, revision=rev, architectures=config["architectures"], indexed_weight_files=len(names),
                root_weight_bytes=sum(files[n]["size"] for n in names), inspected_files=shard_results,
                group_dtype_bytes=dict(groups), routed_payload_by_rank=dict(sorted(placements.items())),
                routed_dtype_bytes_by_layer=dict(sorted(layers.items())),
                root_index_tensors=sum(map(len, indexes.values())) if indexes else None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", choices=MODELS)
    cli.add_argument("header_cache", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("choose a new result path")
    result = probe(args.model, args.header_cache)
    with tempfile.NamedTemporaryFile(dir=args.output.parent) as f:
        f.write((json.dumps(result, indent=2) + "\n").encode())
        f.flush()
        os.link(f.name, args.output)
