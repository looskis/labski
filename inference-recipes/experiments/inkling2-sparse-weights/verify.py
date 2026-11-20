"""Compare all rank-visible tensors in one original and sparse Inkling shard."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open
import torch

from stage import EXPERT, RANKS


def installed_loader():
    source = Path("/sgl-workspace/sglang/python/sglang/srt/models/inkling_common/inkling2_moe.py")
    raw = source.read_bytes()
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Inkling2Experts")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_shard")
    # Execute the installed shape checks, rank slicing and copy on CPU, before scale swizzling.
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_load_shard"], hashlib.sha256(raw).hexdigest()


def digest(tensor):
    byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(byte_view)).hexdigest()


def compare(original, sparse, rank):
    torch.set_num_threads(1)
    loader, source_digest = installed_loader()
    placement = SimpleNamespace(ep_rank=rank, ep_size=RANKS, tp_rank=0, tp_size=1)
    tensors = []
    with safe_open(original, framework="pt", device="cpu") as src, safe_open(sparse, framework="pt", device="cpu") as dst:
        if src.keys() != dst.keys():
            raise ValueError("tensor names differ")
        for name in src.keys():
            weight, staged = src.get_tensor(name), dst.get_tensor(name)
            if weight.shape != staged.shape or weight.dtype != staged.dtype:
                raise ValueError(f"shape or dtype changed for {name}")
            is_expert = EXPERT.fullmatch(name) is not None
            if is_expert:
                width = weight.shape[0] // RANKS
                reference = weight.narrow(0, rank * width, width)
                loaded = torch.empty_like(reference)
                loader(placement, loaded, staged, dim=None, expert_axis=True)
            else:
                reference, loaded = weight, staged
            wanted = digest(reference)
            if digest(loaded) != wanted:
                raise ValueError(f"rank-visible bytes differ for {name}")
            tensors.append(dict(tensor=name, rank_sliced=is_expert, dtype=str(loaded.dtype),
                                bytes=loaded.numel() * loaded.element_size(), sha256=wanted))
            del reference, loaded, weight, staged
    return dict(rank=rank, shard=original.name, installed_loader_file_sha256=source_digest, tensors=tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("sparse", type=Path)
    parser.add_argument("--rank", type=int, choices=range(RANKS), required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.original, args.sparse, args.rank), indent=2))
