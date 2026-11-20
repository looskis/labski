"""Build the Q8 spine and DSA indexer after all 94 text shards pass packing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from pack_shard import digest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("tools", type=Path, help="the pinned engine's tools/kimi-k3 directory")
    cli.add_argument("--expert-dir", default="/model/experts", help="expert path inside the serving container")
    args = cli.parse_args()
    work = args.work.resolve(strict=True)
    tools = args.tools.resolve(strict=True)
    spec = json.loads(Path(__file__).with_name("source.json").read_text())
    for name, expected in spec["tool_hashes"].items():
        if digest(tools / name) != expected:
            raise ValueError(f"unexpected conversion tool: {name}")
    total_experts = 0
    for shard, obj in spec["text_shards"].items():
        receipt = json.loads((work / "receipts" / (shard + ".json")).read_text())
        hashes = work / "receipts" / (shard + ".sha256")
        if (receipt["shard"] != shard or receipt["source_sha256"] != obj["sha256"]
                or digest(hashes) != receipt["packed_file_manifest_sha256"]):
            raise ValueError(f"invalid packing receipt: {shard}")
        subprocess.run(["sha256sum", "--quiet", "--strict", "-c", str(hashes)], cwd=work, check=True)
        total_experts += receipt["expert_files"]
    if total_experts != 82432 or len(list((work / "experts").glob("L*-E*.bin"))) != total_experts:
        raise ValueError("incomplete routed expert store")
    final = work / "model"
    if final.exists():
        raise FileExistsError(f"model output already exists at {final}")
    with tempfile.TemporaryDirectory(prefix="finalize-", dir=work) as tmp:
        stage = Path(tmp)
        archive = work / "indexer-input"
        metadata = stage / "metadata"
        subprocess.run([sys.executable, str(tools / "k31_extract.py"), "--src", str(archive),
                        "--out-meta", str(metadata)], check=True)
        out = stage / "model"
        out.mkdir()
        prefix = out / "indexer"
        subprocess.run([sys.executable, str(tools / "k31_indexer_extract.py"), str(archive), str(prefix)], check=True)
        indexer = json.loads(prefix.with_suffix(".json").read_text())["layers"]
        if len(indexer) != 24 or sum(map(len, indexer.values())) != 120:
            raise ValueError("incomplete DSA indexer")
        spine = out / "spine.gguf"
        subprocess.run([sys.executable, str(tools / "k3_spine_to_gguf.py"), "--meta", str(metadata),
                        "--src", str(work / "resident/tensors"), "--out", str(spine),
                        "--primary-experts", args.expert_dir, "--name", "Kimi K3.1 Pro", "--compact"], check=True)
        layout = json.loads((out / "spine.gguf.sparse.json").read_text())
        if spine.stat().st_size != layout["real_bytes"] or layout["sparse_bytes"] <= layout["real_bytes"]:
            raise ValueError("invalid compact spine layout")
        receipt = dict(compact_spine_sha256=digest(spine), **layout, expert_files=total_experts,
                       indexer_sha256=digest(prefix.with_suffix(".bin")), indexer_layers=len(indexer))
        (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        os.truncate(spine, layout["sparse_bytes"])
        out.rename(final)


if __name__ == "__main__":
    main()
