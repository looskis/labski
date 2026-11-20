"""Check Ultra's target/draft shard boundary and query vLLM's per-rank Mamba state shapes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import struct
from types import SimpleNamespace

import torch
import vllm
from safetensors import safe_open
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP


def state_sizes(config, proposals, state_dtype):
    cache = SimpleNamespace(mamba_cache_dtype="auto", mamba_ssm_cache_dtype=state_dtype, use_replayssm=False)
    settings = SimpleNamespace(model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
                               cache_config=cache, parallel_config=SimpleNamespace(tensor_parallel_size=4),
                               num_speculative_tokens=proposals)
    shapes = NemotronHForCausalLM.get_mamba_state_shape_from_config(settings)
    dtypes = NemotronHForCausalLM.get_mamba_state_dtype_from_config(settings)
    count = config.layers_block_type.count("mamba")
    return dict(proposals=proposals, state_dtype=state_dtype, mamba_layers=count,
                arrays=[dict(shape=list(shape), dtype=str(dtype),
                             bytes_across_layers=math.prod(shape) * dtype.itemsize * count)
                        for shape, dtype in zip(shapes, dtypes, strict=True)])


def inspect_snapshot(root):
    index = json.loads((root / "model.safetensors.index.json").read_bytes())
    config = SimpleNamespace(**json.loads((root / "config.json").read_bytes()))
    by_file = defaultdict(set)
    for key, filename in index["weight_map"].items():
        if Path(filename).name != filename:
            raise ValueError("unexpected shard path")
        by_file[filename].add(key)
    if len(by_file) != 113:
        raise ValueError("expected the 113-shard NVFP4 release")
    files, components = [], defaultdict(Counter)
    for filename, keys in sorted(by_file.items()):
        path = root / filename
        with path.open("rb") as src:
            prefix = src.read(8)
            length, = struct.unpack("<Q", prefix)
            if not 2 <= length <= 100_000_000:
                raise ValueError(f"bad header length in {filename}")
            raw = src.read(length)
        fields = json.loads(raw)
        fields.pop("__metadata__", None)
        roles, offset = set(), 0
        with safe_open(path, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != keys or fields.keys() != keys:
                raise ValueError(f"index mismatch in {filename}")
            for name, field in sorted(fields.items(), key=lambda pair: pair[1]["data_offsets"]):
                view = shard.get_slice(name)
                start, end = field["data_offsets"]
                if start != offset or end < start or view.get_shape() != field["shape"] or view.get_dtype() != field["dtype"]:
                    raise ValueError(f"unexpected layout for {name}")
                offset = end
                target = NemotronHForCausalLM.hf_to_vllm_mapper.map_name(name) is not None
                draft = not NemotronHMTP.is_unused_checkpoint_weight(None, name)
                if target == draft or draft != name.startswith("mtp."):
                    raise ValueError(f"target and draft selection disagree for {name}")
                role = "target" if target else "draft"
                roles.add(role)
                group = role + ("_routed" if ".mixer.experts." in name else "_other")
                components[group][field["dtype"]] += end - start
        if len(roles) != 1 or path.stat().st_size != offset + 8 + length:
            raise ValueError(f"unexpected file size or mixed roles in {filename}")
        files.append(dict(file=filename, role=roles.pop(), tensors=len(keys), bytes=path.stat().st_size,
                          header_sha256=hashlib.sha256(prefix + raw).hexdigest(), header_bytes=8 + length))
    if sum(row["bytes"] - row["header_bytes"] for row in files) != index["metadata"]["total_size"]:
        raise ValueError("payload size differs from the index")
    implementations = [NemotronHForCausalLM, NemotronHMTP]
    return dict(vllm=vllm.__version__, machine=platform.machine(), files=files,
                payload_bytes_by_group_and_dtype=components,
                runtime_modules={cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                                 for cls in implementations},
                tp4_state=[state_sizes(config, proposals, dtype) for proposals in (0, 4)
                           for dtype in ("float32", "bfloat16")])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    record = inspect_snapshot(args.snapshot)
    with args.out.open("x") as out:
        json.dump(record, out, indent=2)
        out.write("\n")
