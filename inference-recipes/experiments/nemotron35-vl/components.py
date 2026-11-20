"""Read the indexed tensor sizes of the pinned NVIDIA Nemotron 3.5 Super VL checkpoint."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct

REVISION = "ae1368329b1fc76b71534c5d04b87b35d9cc6e15"

def group(name):
    if name.startswith("language_model.mtp."):
        return "native_mtp"
    if name.startswith(("language_model.backbone.", "language_model.lm_head.")):
        return "text_backbone_and_head"
    if name.startswith("vision_model."):
        return "vision_encoder"
    if name.startswith(("vision_projector.", "mlp1.")):
        return "vision_projection"
    raise ValueError(f"unknown top-level component in {name}")

def account(snapshot):
    raw_index = (snapshot / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(raw_index).hexdigest() != "c368acdbcfe882236af1ec79df4a1fb6127e6ed9f742d59c01c21fe96ddc2741":
        raise ValueError("index differs from the pinned model revision")
    idx = json.loads(raw_index)
    files = defaultdict(set)
    for key, name in idx["weight_map"].items():
        if Path(name).name != name:
            raise ValueError("unexpected shard path")
        files[name].add(key)
    sizes, counts = Counter(), Counter()
    shards = []
    for name, keys in sorted(files.items()):
        p = snapshot / name
        with p.open("rb") as f:
            prefix = f.read(8)
            length, = struct.unpack("<Q", prefix)
            if not 2 <= length <= 100_000_000:
                raise ValueError(f"invalid SafeTensors header in {name}")
            raw = f.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated header in {name}")
        fields = json.loads(raw)
        fields.pop("__metadata__", None)
        if fields.keys() != keys:
            raise ValueError(f"index and header names differ in {name}")
        end = 0
        for key, field in sorted(fields.items(), key=lambda item: item[1]["data_offsets"]):
            start, stop = field["data_offsets"]
            if start != end or stop < start:
                raise ValueError(f"invalid offsets for {key}")
            section = group(key)
            sizes[section] += stop - start
            counts[section] += 1
            end = stop
        if end + length + 8 != p.stat().st_size:
            raise ValueError(f"wrong length for {name}")
        shards.append(dict(file=name, bytes=p.stat().st_size, header_bytes=length + 8,
                           header_sha256=hashlib.sha256(prefix + raw).hexdigest()))
    return dict(revision=REVISION, index_metadata_total_size=idx["metadata"]["total_size"],
                file_bytes=sum(r["bytes"] for r in shards), payload_bytes=sum(sizes.values()),
                files=shards, components=[dict(name=g, tensors=counts[g], payload_bytes=sizes[g])
                                         for g in sorted(sizes)])

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    print(json.dumps(account(cli.parse_args().snapshot), indent=2))
