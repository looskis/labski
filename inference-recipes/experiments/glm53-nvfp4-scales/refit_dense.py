"""Refit all nine dense prefix MLP matrices, emitting only changed parameters."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from compare import fit


def tensor(folder, row, key, dtype):
    entry = row["tensors"][key]
    raw = (folder / entry["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"input digest changed for {entry['file']}")
    return torch.frombuffer(bytearray(raw), dtype=dtype)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("samples", type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    manifest = json.loads((args.samples / "manifest.json").read_bytes())
    expected = {(layer, name) for layer in range(3) for name in ("gate_proj", "up_proj", "down_proj")}
    panels = manifest["panels"]
    if len(panels) != 9 or {(r["layer"], r["projection"]) for r in panels} != expected:
        cli.error("supply the complete --dense-prefix collection")
    args.out.mkdir(parents=True, exist_ok=False)
    scale_patch, recode_patch, originals = {}, {}, {}
    with torch.inference_mode():
        for row in panels:
            n, k = row["rows"], row["width"]
            shape = (4096, 12288) if row["projection"] == "down_proj" else (12288, 4096)
            if (n, k) != shape or row["start_row"] != 0 or row["expert"] is not None:
                raise ValueError("expected a full dense projection")
            prefix = f"model.language_model.layers.{row['layer']}.mlp.{row['projection']}."
            reference = tensor(args.samples, row, "reference", torch.bfloat16).reshape(n, k)
            packed = tensor(args.samples, row, "packed", torch.uint8).reshape(n, k // 2)
            scales = tensor(args.samples, row, "scale", torch.float8_e4m3fn).reshape(n, k // 16)
            gs = tensor(args.samples, row, "global_scale", torch.float32).cuda()
            new_codes = torch.empty_like(packed)
            new_scales = torch.empty_like(scales)
            fixed_code_scales = torch.empty_like(scales)
            for i in range(0, n, 256):
                sl = slice(i, i + 256)
                arms = fit(reference[sl].cuda(), packed[sl].cuda(), scales[sl].cuda(), gs)
                fixed_code_scales[sl].copy_(arms[1][2].cpu())
                new_codes[sl].copy_(arms[2][1].cpu())
                new_scales[sl].copy_(arms[2][2].cpu())
            scale_patch[prefix + "weight_scale"] = fixed_code_scales
            recode_patch[prefix + "weight"] = new_codes
            recode_patch[prefix + "weight_scale"] = new_scales
            originals[prefix + "weight"] = row["tensors"]["packed"]["sha256"]
            originals[prefix + "weight_scale"] = row["tensors"]["scale"]["sha256"]
            print(prefix, n, k, flush=True)
    for mode, tensors in [("scale_only", scale_patch), ("scale_and_codes", recode_patch)]:
        target = args.out / (mode + ".safetensors")
        save_file(tensors, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        receipt = dict(sha256=digest, source_sha256={k: originals[k] for k in tensors})
        (args.out / (mode + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
