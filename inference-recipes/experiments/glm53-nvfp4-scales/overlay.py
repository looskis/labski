"""Replace the nine dense GLM projections in the pinned vLLM loader."""

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for data in iter(lambda: src.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def install(loader_class):
    original = loader_class.get_all_weights

    def weights(self, model_config, model):
        mode = os.environ["LABSKI_DENSE_MODE"]
        if mode not in {"published", "scale_only", "scale_and_codes"}:
            raise ValueError(f"unknown GLM patch mode {mode}")
        directory = Path("/labski/glm-patches")
        patch = "scale_only" if mode == "scale_only" else "scale_and_codes"
        receipt = json.loads((directory / (patch + ".json")).read_bytes())
        path = directory / (patch + ".safetensors")
        if digest_file(path) != receipt["sha256"]:
            raise ValueError("dense prefix patch digest changed")
        suffixes = ["weight_scale"] if patch == "scale_only" else ["weight", "weight_scale"]
        expected = {f"model.language_model.layers.{i}.mlp.{proj}.{suffix}"
                    for i in range(3) for proj in ("gate_proj", "up_proj", "down_proj") for suffix in suffixes}
        if set(receipt["source_sha256"]) != expected:
            raise ValueError("dense prefix receipt has a different tensor set")
        with safe_open(path, framework="pt", device="cpu") as sf:
            if set(sf.keys()) != expected:
                raise ValueError("dense prefix patch has a different tensor set")
            seen = set()
            for name, value in original(self, model_config, model):
                if name in expected:
                    current = value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy()
                    if name in seen or hashlib.sha256(current).hexdigest() != receipt["source_sha256"][name]:
                        raise ValueError(f"unexpected source tensor for {name}")
                    replacement = sf.get_tensor(name)
                    if value.shape != replacement.shape or value.dtype != replacement.dtype:
                        raise ValueError(f"shape or dtype differs for {name}")
                    seen.add(name)
                    if mode != "published":
                        value = replacement
                yield name, value
            if seen != expected:
                raise ValueError(f"missing dense prefix tensors: {sorted(expected - seen)}")
        print("LABSKI_DENSE_VERIFIED", json.dumps(dict(mode=mode, source_keys=len(seen),
              replaced_keys=0 if mode == "published" else len(seen), patch_sha256=receipt["sha256"])), flush=True)

    loader_class.get_all_weights = weights
