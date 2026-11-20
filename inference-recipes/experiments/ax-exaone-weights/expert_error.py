"""Measure weight-only conversion error on fixed complete K-EXAONE BF16 expert matrices."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from probe import MODELS, metadata, read

LAYERS = (2, 3, 4, 75, 76, 77)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def selected():
    for layer in LAYERS:
        seed = f"labski-k-exaone-v1/{layer}/"
        expert = min(range(256), key=lambda i: hashlib.sha256((seed + str(i)).encode()).digest())
        for projection in PROJECTIONS:
            yield f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"


def fetch(cache, out):
    repo, rev = MODELS["exaone"]
    url = f"https://huggingface.co/{repo}/resolve/{rev}/"
    info = json.loads(read(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"))
    files = {r["rfilename"]: r for r in info["siblings"]}
    index = metadata(url, "model.safetensors.index.json", files)["weight_map"]
    report = json.loads(Path(__file__).with_name("exaone.json").read_bytes())
    shards = {r["file"]: r for r in report["inspected_files"]}
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for key in selected():
        shard = index[key]
        h = (cache / rev / (shard + ".header")).read_bytes()
        if hashlib.sha256(h).hexdigest() != shards[shard]["header_sha256"]:
            raise ValueError(f"unexpected header for {shard}")
        field = json.loads(h[8:])[key]
        shape = field["shape"]
        a, b = [len(h) + offset for offset in field["data_offsets"]]
        if field["dtype"] != "BF16" or sorted(shape) != [2048, 6144] or b - a != shape[0] * shape[1] * 2:
            raise ValueError(f"unexpected dtype or shape for {key}")
        path = out / (key + ".bin")
        expected = f"bytes {a}-{b-1}/{shards[shard]['file_bytes']}"
        if not path.exists():
            req = urllib.request.Request(url + shard + f"?download=true&range={a}-{b}",
                                         headers={"Range": f"bytes={a}-{b-1}", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=120) as reply:
                if reply.status != 206 or reply.headers.get("Content-Range") != expected:
                    raise ValueError("unexpected upstream range response")
                buf = reply.read(b - a + 1)
            if len(buf) != b - a:
                raise ValueError("truncated tensor")
            with tempfile.NamedTemporaryFile(dir=out) as tmp:
                tmp.write(buf)
                tmp.flush()
                try:
                    os.link(tmp.name, path)
                except FileExistsError:
                    if path.read_bytes() != buf:
                        raise ValueError("conflicting downloaded tensor")
        data = path.read_bytes()
        if len(data) != b - a:
            raise ValueError("existing tensor is truncated")
        records.append(dict(tensor=key, shape=shape, shard=shard, offset=a, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    return records


def block_fp8(torch, w):
    rows, width = w.shape
    tiles = w.reshape(rows // 128, 128, width // 128, 128)
    scale = tiles.abs().amax(dim=(1, 3), keepdim=True) / 448
    quant = (tiles / scale.clamp_min(1e-30)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quant.float() * scale).reshape_as(w)


def block_nvfp4(torch, w):
    tiles = w.reshape(-1, 16)
    global_scale = w.abs().max() / (448 * 6)
    if not global_scale:
        return w.clone()
    local = (tiles.abs().amax(dim=1, keepdim=True) / (6 * global_scale)).clamp(0, 448)
    scale = local.to(torch.float8_e4m3fn).float() * global_scale
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    magnitude = tiles.abs() / scale.clamp_min(1e-30)
    cuts = (levels[:-1] + levels[1:]) / 2
    nearest = torch.bucketize(magnitude, cuts)
    return (tiles.sign() * levels[nearest] * scale).reshape_as(w)


def errors(torch, candidate, reference):
    return dict(squared_error=(candidate - reference).double().square().sum().item(),
                reference_squared_norm=reference.double().square().sum().item())


def evaluate(samples):
    import torch
    torch.set_num_threads(2)
    records = json.loads((samples / "samples.json").read_bytes())
    if [r["tensor"] for r in records] != list(selected()):
        raise ValueError("unexpected matrix sample")
    result = []
    with torch.inference_mode():
        for row in records:
            data = bytearray((samples / (row["tensor"] + ".bin")).read_bytes())
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("stored tensor differs from the receipt")
            w = torch.frombuffer(data, dtype=torch.bfloat16).reshape(row["shape"]).float()
            g = torch.Generator().manual_seed(85331 + w.shape[1])
            x = torch.randn((32, w.shape[1]), generator=g).bfloat16().float()
            ref = x @ w.T
            scores = {}
            for name, fn in (("block_fp8", block_fp8), ("block_nvfp4", block_nvfp4)):
                candidate = fn(torch, w)
                if not candidate.isfinite().all():
                    raise ValueError("nonfinite reconstruction")
                scores[name] = dict(weight=errors(torch, candidate, w),
                                    projection=errors(torch, x @ candidate.T, ref))
            result.append(dict(tensor=row["tensor"], source_sha256=row["sha256"], scores=scores))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    download = sub.add_parser("fetch")
    download.add_argument("header_cache", type=Path)
    download.add_argument("samples", type=Path)
    measure = sub.add_parser("evaluate")
    measure.add_argument("samples", type=Path)
    measure.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.action == "fetch":
        output = args.samples / "samples.json"
        rows = fetch(args.header_cache, args.samples)
    else:
        output = args.output
        rows = evaluate(args.samples)
    encoded = (json.dumps(rows, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=output.parent) as tmp:
        tmp.write(encoded)
        tmp.flush()
        try:
            os.link(tmp.name, output)
        except FileExistsError:
            if output.read_bytes() != encoded:
                raise ValueError("existing report differs")
