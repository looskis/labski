"""Read matching BF16 and NVFP4 expert panels from pinned GLM-5.3-Flash checkpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen


BF16 = "https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/f9fed3bbc82291073373d67452e9f25317ef7577/"
FP4 = "https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/resolve/1d00738646cbb0c6d0a9eb403dac5f366f1473a7/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(url, offset=0, count=None):
    headers = {}
    if count is not None:
        end = offset + count - 1
        headers["Range"] = f"bytes={offset}-{end}"
        url += f"?panel={offset}-{end}"
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                if count is not None and (response.status != 206 or
                        response.headers.get("Content-Range", "").split("/")[0] != f"bytes {offset}-{end}"):
                    raise ValueError("checkpoint server returned a different byte range")
                data = response.read() if count is None else response.read(count + 1)
                if count is not None and len(data) != count:
                    raise ValueError("short or oversized checkpoint response")
                return data
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    cli.add_argument("--nvfp4", type=Path, help="use a local pinned NVFP4 snapshot in place of Hub ranges")
    cli.add_argument("--rows", type=int, default=64, help="contiguous output rows per projection")
    cli.add_argument("--dense-prefix", action="store_true", help="read the nine complete dense MLP matrices in layers 0–2")
    args = cli.parse_args()
    if args.rows < 1 or 2048 % args.rows:
        cli.error("--rows must be a positive divisor of 2048")
    args.out.mkdir(parents=True, exist_ok=False)

    def get(base, name, offset=0, count=None):
        if base == FP4 and args.nvfp4 is not None:
            with (args.nvfp4 / name).open("rb") as file:
                file.seek(offset)
                data = file.read() if count is None else file.read(count)
            if count is not None and len(data) != count:
                raise ValueError("short local checkpoint range")
            return data
        return read(base + name, offset, count)

    weights = {base: json.loads(get(base, "model.safetensors.index.json"))["weight_map"]
               for base in (BF16, FP4)}
    panels = []
    tensors = set()
    for layer in ((0, 1, 2) if args.dense_prefix else (3, 24, 44)):
        def order(i):
            return digest(f"labski-glm53-scales-v1/{layer}/{i}".encode())
        experts = [None] if args.dense_prefix else sorted(range(288), key=order)[:2]
        for expert in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mlp = f"model.language_model.layers.{layer}.mlp."
                prefix = mlp + ("" if expert is None else f"experts.{expert}.") + proj + "."
                panels.append((layer, expert, proj, prefix))
                tensors.add((BF16, weights[BF16][prefix + "weight"]))
                for name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    tensors.add((FP4, weights[FP4][prefix + name]))

    def header(pair):
        base, name = pair
        length, = struct.unpack("<Q", get(base, name, 0, 8))
        if not 2 <= length < 100_000_000:
            raise ValueError("invalid SafeTensors header size")
        return pair, (8 + length, json.loads(get(base, name, 8, length)))

    with ThreadPoolExecutor(max_workers=4) as pool:
        headers = dict(pool.map(header, sorted(tensors)))

    def sample(panel):
        layer, expert, proj, prefix = panel
        paths = {}
        intermediate = 12288 if expert is None else 2048
        rows, width = (4096, intermediate) if proj == "down_proj" else (intermediate, 4096)
        count = rows if args.dense_prefix else args.rows
        start = (int(digest(prefix.encode())[:8], 16) % (rows // count)) * count
        for base, name, label, dtype, shape, size in [
                (BF16, "weight", "reference", "BF16", [rows, width], 2),
                (FP4, "weight", "packed", "U8", [rows, width // 2], 1),
                (FP4, "weight_scale", "scale", "F8_E4M3", [rows, width // 16], 1),
                (FP4, "weight_scale_2", "global_scale", "F32", None, 4),
                (FP4, "input_scale", "input_scale", "F32", None, 4)]:
            key = prefix + name
            shard = weights[base][key]
            offset, meta = headers[(base, shard)]
            spec = meta[key]
            lo, hi = spec["data_offsets"]
            if spec["dtype"] != dtype or (shape is not None and spec["shape"] != shape):
                raise ValueError(f"{key} has unexpected dtype or dimensions: {spec}")
            if shape is None:
                if hi - lo != size:
                    raise ValueError(f"{key} needs one scale")
                length = size
            else:
                stride = shape[1] * size
                if hi - lo != shape[0] * stride:
                    raise ValueError(f"{key} has inconsistent dimensions")
                lo += start * stride
                length = count * stride
            data = get(base, shard, offset + lo, length)
            filename = f"layer-{layer}-expert-{expert}-{proj}-{label}.bin"
            (args.out / filename).write_bytes(data)
            paths[label] = dict(file=filename, sha256=digest(data), shard=shard, offset=offset+lo, bytes=length)
        return dict(layer=layer, expert=expert, projection=proj, start_row=start,
                    rows=count, width=width, tensors=paths)

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(sample, panels))
    (args.out / "manifest.json").write_text(json.dumps(dict(bf16=BF16, nvfp4=FP4, panels=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
