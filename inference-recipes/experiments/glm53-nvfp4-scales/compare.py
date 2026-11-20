"""Compare published and refitted NVFP4 groups with the same native GB10 GEMM."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.nvfp4.b12x import _apply_b12x_nvfp4_linear
from vllm.utils.b12x import get_b12x_intrinsics


LEVELS = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def unpack(packed):
    return torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, 16)


def pack(codes, rows):
    pairs = codes.reshape(rows, -1, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).contiguous()


def fit(reference, packed, scales, global_scale):
    target = reference.float().reshape(-1, 16)
    levels = torch.tensor(LEVELS, device=target.device)
    values = torch.cat([levels, -levels])
    codes = unpack(packed)
    original = values[codes.long()]
    all_scales = torch.arange(127, device=target.device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    actual_scales = all_scales * global_scale
    thresholds = (actual_scales[:-1] + actual_scales[1:]) / 2
    norm = original.square().sum(1)
    optimum = (target * original).sum(1) / norm.clamp_min(0.25)
    index = torch.bucketize(optimum, thresholds).to(torch.uint8)
    existing = scales.view(torch.uint8).reshape(-1)
    if existing.max().item() > 126:
        raise ValueError("expected finite nonnegative block scales")

    def errors(group_codes, scale_ids):
        weights = values[group_codes.long()] * actual_scales[scale_ids.long(), None]
        return (target - weights).square().sum(1)

    base_loss = errors(codes, existing)
    scale_loss = errors(codes, index)
    better = scale_loss < base_loss
    scale_only = torch.where(better, index, existing)
    best_loss = torch.minimum(scale_loss, base_loss)
    best_scale, best_codes = scale_only.clone(), codes.clone()
    unsigned = target.abs()
    sign = (target < 0).to(torch.uint8) << 3
    mid = (levels[:-1] + levels[1:]) / 2
    for i in range(1, 127):
        # Midpoints select the nearest magnitude at this scale. Both choices agree in squared error on a tie.
        mags = torch.bucketize(unsigned, mid * actual_scales[i])
        err = (unsigned - levels[mags] * actual_scales[i]).square().sum(1)
        better = err < best_loss
        best_scale = torch.where(better, i, best_scale)
        best_codes = torch.where(better[:, None], mags.to(torch.uint8) | sign, best_codes)
        best_loss = torch.minimum(best_loss, err)
    rows = packed.shape[0]
    return [
        ("published", packed, scales),
        ("scale_only", packed, scale_only.reshape_as(scales).view(torch.float8_e4m3fn)),
        ("scale_and_codes", pack(best_codes, rows), best_scale.reshape_as(scales).view(torch.float8_e4m3fn)),
    ]


def patched_arms(panel, packed, scales, patches):
    if panel["expert"] is not None or panel["start_row"] != 0:
        raise ValueError("dense patches require complete prefix projections")
    prefix = f"model.language_model.layers.{panel['layer']}.mlp.{panel['projection']}."
    arms = [("published", packed, scales)]
    for mode, (file, originals) in patches.items():
        weights, block_scales = packed, scales
        for suffix, source, previous in [("weight_scale", "scale", scales), ("weight", "packed", packed)]:
            if mode == "scale_only" and suffix == "weight":
                continue
            key = prefix + suffix
            if originals[key] != panel["tensors"][source]["sha256"]:
                raise ValueError(f"patch refers to a different source tensor for {key}")
            value = file.get_tensor(key)
            if value.shape != previous.shape or value.dtype != previous.dtype:
                raise ValueError(f"patch shape or dtype differs for {key}")
            if suffix == "weight":
                weights = value.cuda()
            else:
                block_scales = value.cuda()
        arms.append((mode, weights, block_scales))
    return arms


def measure(root, panel, patches=None):
    rows, width = panel["rows"], panel["width"]

    def load(label, dtype):
        record = panel["tensors"][label]
        data = (root / record["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"panel digest changed for {label}")
        return torch.frombuffer(bytearray(data), dtype=dtype).cuda()

    ref = load("reference", torch.bfloat16).reshape(rows, width)
    packed = load("packed", torch.uint8).reshape(rows, width // 2)
    scales = load("scale", torch.float8_e4m3fn).reshape(rows, width // 16)
    gs = load("global_scale", torch.float32)
    ins = load("input_scale", torch.float32)
    if not (torch.isfinite(gs).all() and gs.item() > 0 and torch.isfinite(ins).all() and ins.item() > 0):
        raise ValueError("positive finite global scales required")
    values = torch.tensor(LEVELS + [-x for x in LEVELS], device="cuda")
    arms = patched_arms(panel, packed, scales, patches) if patches else fit(ref, packed, scales, gs)
    ref32 = ref.float()
    ref_norm = ref32.double().square().sum().item()
    generator = torch.Generator(device="cpu").manual_seed(51863 + width)
    raw = torch.randn(64, width, generator=generator)
    sparse = raw.clone()
    sparse[:, ::64] *= 8
    # Use exactly the same BF16 inputs across all three arms. Keep maxima in the published input scale's range.
    inputs = {k: (v.cuda() * (0.8 * 2688 * ins / v.abs().max())).bfloat16()
              for k, v in [("gaussian", raw), ("sparse_outliers", sparse)]}
    refs = {k: v.float() @ ref32.T for k, v in inputs.items()}
    records = []
    for name, weights, block_scale in arms:
        unpacked = (values[unpack(weights).long()] * block_scale.float().reshape(-1, 1) * gs).reshape_as(ref)
        native = {}
        stored = get_b12x_intrinsics().swizzle_block_scale(block_scale)
        for distribution, x in inputs.items():
            y = _apply_b12x_nvfp4_linear(x, weights, stored, ins.reciprocal(), gs * ins, None)
            if y.shape != (64, rows) or not torch.isfinite(y).all():
                raise ValueError("native GEMM returned invalid output")
            exact = refs[distribution]
            native[distribution] = dict(squared_error=(y.float() - exact).double().square().sum().item(),
                                        reference_squared_norm=exact.double().square().sum().item())
        records.append(dict(arm=name, weight_squared_error=(unpacked - ref32).double().square().sum().item(),
                            weight_reference_squared_norm=ref_norm, native=native,
                            changed_scale_bytes=(block_scale.view(torch.uint8) != scales.view(torch.uint8)).sum().item(),
                            changed_nibbles=(unpack(weights) != unpack(packed)).sum().item()))
    return dict(layer=panel["layer"], expert=panel["expert"], projection=panel["projection"],
                start_row=panel["start_row"], rows=rows, input_width=width, arms=records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("samples", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--patches", type=Path, help="read already-fitted dense prefix tensors")
    args = cli.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads((args.samples / "manifest.json").read_bytes())
    with ExitStack() as files, torch.inference_mode():
        patches = {}
        if args.patches:
            from safetensors import safe_open
            from overlay import digest_file

            for mode in ("scale_only", "scale_and_codes"):
                path = args.patches / (mode + ".safetensors")
                receipt = json.loads((args.patches / (mode + ".json")).read_bytes())
                if digest_file(path) != receipt["sha256"]:
                    raise ValueError(f"patch digest changed for {mode}")
                handle = files.enter_context(safe_open(path, framework="pt", device="cpu"))
                if set(handle.keys()) != set(receipt["source_sha256"]):
                    raise ValueError(f"unexpected patch keys for {mode}")
                patches[mode] = handle, receipt["source_sha256"]
        out = files.enter_context(args.out.open("x"))
        for panel in source["panels"]:
            row = measure(args.samples, panel, patches)
            out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            out.flush()
            print(row["layer"], row["expert"], row["projection"], flush=True)


if __name__ == "__main__":
    main()
