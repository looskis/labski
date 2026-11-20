"""Pack one verified Kimi K3.1 Pro text shard with Argodrive's extractor."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

INDEX_SHA256 = "6500b553254ae3955e19bbf93d8cf3eee17101df4430493b33ce44d8ea1e2925"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard", type=Path)
    ap.add_argument("index", type=Path, help="pinned model.safetensors.index.json")
    ap.add_argument("work", type=Path)
    ap.add_argument("extractor", type=Path, help="pinned upstream k31_extract.py")
    args = ap.parse_args()
    shard = args.shard.resolve(strict=True)
    index = args.index.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    work = args.work.resolve()
    work.mkdir(exist_ok=True, parents=True)
    manifest = json.loads(Path(__file__).with_name("source.json").read_text())
    spec = manifest["text_shards"][shard.name]
    locks = work / "locks"
    locks.mkdir(exist_ok=True)
    lock = (locks / shard.name).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if shard.stat().st_size != spec["bytes"] or digest(shard) != spec["sha256"]:
        raise ValueError(f"input failed pinned SHA-256 or size check: {shard.name}")
    if digest(index) != INDEX_SHA256 or digest(extractor) != manifest["tool_hashes"]["k31_extract.py"]:
        raise ValueError("unexpected model index or extractor revision")
    module = runpy.run_path(str(extractor))
    header_len, header = module["read_header"](str(shard))
    mapping = json.loads(index.read_bytes())["weight_map"]
    if set(header) != {name for name, file in mapping.items() if file == shard.name}:
        raise ValueError(f"index and header disagree for {shard.name}")
    experts = {}
    resident = []
    for name in sorted(header):
        match = module["EXPERT_RE"].fullmatch(name)
        if match:
            layer, expert, proj, field = match.groups()
            experts.setdefault((int(layer), int(expert)), {})[proj, field] = name
        elif name.startswith("language_model."):
            resident.append(name)
    order = module["ORDER"]
    if any(set(planes) != set(order) for planes in experts.values()):
        raise ValueError("an expert is split across source shards")
    layers = sorted({layer for layer, _ in experts})
    if layers and (len(layers) != 1 or len(experts) != 896):
        raise ValueError("expected one complete 896-expert layer in each MoE shard")
    with tempfile.TemporaryDirectory(prefix="input-", dir=work) as tmp:
        view = Path(tmp)
        (view / shard.name).symlink_to(shard)
        (view / "model.safetensors.index.json").symlink_to(index)
        cmd = [sys.executable, str(extractor), "--src", tmp, "--out-resident", str(work / "resident")]
        if layers:
            cmd += ["--out-experts", str(work / "experts"), "--layers", f"{layers[0]}-{layers[0]}"]
        subprocess.run(cmd, check=True)

    counts = {"resident_planes": len(resident), "expert_files": len(experts)}
    archive = work / "indexer-input"
    archive.mkdir(exist_ok=True)
    receipts = work / "receipts"
    receipts.mkdir(exist_ok=True)
    receipt = receipts / (shard.name + ".sha256")
    with shard.open("rb") as src, receipt.with_suffix(".pending").open("w") as checks:
        def plane(name):
            lo, hi = header[name]["data_offsets"]
            src.seek(8 + header_len + lo)
            b = src.read(hi - lo)
            if len(b) != hi - lo:
                raise ValueError(f"short plane {name}")
            return b

        def validate(relpath, parts):
            h = hashlib.sha256()
            for part in parts:
                h.update(part)
            expected = h.hexdigest()
            target = work / relpath
            if digest(target) != expected:
                module["write_atomic"](str(target), parts)
                if digest(target) != expected:
                    raise ValueError(f"packed bytes differ from the verified source: {relpath}")
            checks.write(f"{expected}  {relpath}\n")

        # The official indexer reads only the header and the five indexer planes
        # in each full-attention layer. Preserve their original file offsets.
        with (archive / shard.name).open("wb") as sparse:
            sparse.write(src.read(8 + header_len))
            for name in resident:
                part = plane(name)
                validate(Path("resident/tensors") / name, [part])
                if ".self_attn.indexer." in name:
                    sparse.seek(8 + header_len + header[name]["data_offsets"][0])
                    sparse.write(part)
            sparse.flush()
            os.fsync(sparse.fileno())
        for (layer, expert), planes in sorted(experts.items()):
            parts = [plane(planes[key]) for key in order]
            validate(Path("experts") / f"L{layer}-E{expert}.bin", parts)
        checks.flush()
        os.fsync(checks.fileno())
    receipt.with_suffix(".pending").replace(receipt)
    result = dict(shard=shard.name, source_sha256=spec["sha256"],
                  packed_file_manifest_sha256=digest(receipt), **counts)
    done = receipt.with_suffix(".json")
    done.with_suffix(".tmp").write_text(json.dumps(result, indent=2) + "\n")
    done.with_suffix(".tmp").replace(done)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
