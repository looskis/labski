"""Compare one actual Kimi routed-expert branch against an independent float64 matrix calculation."""
import argparse
import json
from pathlib import Path

import numpy as np

LEVELS = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)
LATENT, FF, USED = 3584, 3072, 16


def matrix(stream, rows, cols):
    packed = np.fromfile(stream, dtype=np.uint8, count=rows * cols // 2).reshape(rows, cols // 2)
    scale = np.fromfile(stream, dtype=np.uint8, count=rows * cols // 32).reshape(rows, cols // 32)
    weights = np.empty((rows, cols), dtype=np.float64)
    weights[:, 0::2] = LEVELS[packed & 15]
    weights[:, 1::2] = LEVELS[packed >> 4]
    weights.reshape(rows, cols // 32, 32)[:] *= np.exp2(scale.astype(np.float64) - 127)[:, :, None]
    return weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experts", type=Path)
    ap.add_argument("gpu_output", type=Path)
    args = ap.parse_args()
    x = ((np.arange(LATENT) * 16807 + 377) % 65521 - 32760).astype(np.float64) / 32768
    reference = np.zeros(LATENT, dtype=np.float64)
    for i in range(USED):
        with (args.experts / f"L1-E{(i * 53 + 7) % 896}.bin").open("rb") as src:
            gate = matrix(src, FF, LATENT) @ x
            down = matrix(src, LATENT, FF)
            up = matrix(src, FF, LATENT) @ x
            if src.read(1):
                raise ValueError("unexpected expert payload after its six tensor planes")
        activated = (4 * np.tanh(gate / 4) / (1 + np.exp(-gate))) * (25 * np.tanh(up / 25))
        reference += float(np.float32((i + 1) / 136)) * (down @ activated)
    gpu = np.fromfile(args.gpu_output, dtype="<f4").astype(np.float64)
    if gpu.shape != reference.shape or not np.all(np.isfinite(gpu)) or not np.any(reference):
        raise ValueError("invalid routed-expert output")
    error = gpu - reference
    result = dict(output_dimensions=len(gpu), max_absolute_error=float(np.abs(error).max()),
                  relative_l2_error=float(np.linalg.norm(error) / np.linalg.norm(reference)),
                  cosine=float(np.dot(gpu, reference) / (np.linalg.norm(gpu) * np.linalg.norm(reference))),
                  cpu_reference_l2=float(np.linalg.norm(reference)), gpu_output_l2=float(np.linalg.norm(gpu)))
    print(json.dumps(result))
    if not np.allclose(gpu, reference, rtol=1e-4, atol=1e-5):
        raise ValueError("GPU result exceeds the float64 comparison tolerance")


if __name__ == "__main__":
    main()
