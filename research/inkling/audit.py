"""Reproduce the architecture counts and paired decision results with Python 3."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from urllib.request import Request, urlopen


REVISIONS = {
    "Inkling": "828496eeae4c243ff1a22f7f28ff83694f2f7bc9",
    "Inkling-Small": "8cc5877b44d343f88b92086aa1fb72897950f06a",
    "Inkling-2-Small": "4d564489295e30e9341a9822a8046f0c9ded1a7b",
    "Inkling-2-Mini": "ed40e7da625f594fc9c4990f5f66be24873c22d7",
    "TML-d24-varloop-research": "657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2",
}
ARCHIVE = (Path(__file__).resolve().parents[2] / "inference-recipes/results"
           / "inkling-mistral-four-spark-2026-11-20/quality.jsonl.gz")


def model_url(model, filename):
    return f"https://huggingface.co/thinkingmachines/{model}/resolve/{REVISIONS[model]}/{filename}"


def read_json(model, filename):
    with urlopen(model_url(model, filename), timeout=45) as response:
        return json.load(response)


def read_range(url, start, end):
    # A distinct redirect per range avoids reusing a signed URL for the other range.
    request = Request(f"{url}?header_range={start}-{end}",
                      headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=45) as response:
        expected = f"bytes {start}-{end}"
        received = response.headers.get("Content-Range", "").split("/")[0]
        if response.status != 206 or received != expected:
            raise ValueError(f"Expected {expected}, received HTTP {response.status} {received}")
        length = end - start + 1
        data = response.read(length)
        if len(data) != length:
            raise ValueError(f"Truncated response for {expected}")
        return data


def inventory_mini():
    model = "Inkling-2-Mini"
    index = read_json(model, "model.safetensors.index.json")
    shard, = set(index["weight_map"].values())
    url = model_url(model, shard)
    size = int.from_bytes(read_range(url, 0, 7), "little")
    if not 0 < size <= 2 * 1024**2:
        raise ValueError(f"Unexpected SafeTensors header size {size}")
    header = json.loads(read_range(url, 8, 7 + size))
    header.pop("__metadata__", None)
    if header.keys() != index["weight_map"].keys():
        raise ValueError("Header disagrees with the checkpoint index")
    sizes = {key: tensor["data_offsets"][1] - tensor["data_offsets"][0]
             for key, tensor in header.items()}
    if sum(sizes.values()) != index["metadata"]["total_size"]:
        raise ValueError("Indexed bytes disagree with tensor offsets")
    head = "model.llm.unembed.weight"
    vocab, width = header[head]["shape"]
    return {
        "target_tensor_count": len(sizes),
        "target_tensor_bytes": sum(sizes.values()),
        "vision_tensor_bytes": sum(n for key, n in sizes.items() if key.startswith("model.visual.")),
        "input_embedding_bytes": sizes["model.llm.embed.weight"],
        "output_head_shape": [vocab, width],
        "output_head_dtype": header[head]["dtype"],
        "output_head_bytes": sizes[head],
        "ten_output_rows_bytes": 10 * (sizes[head] // vocab),
    }


def layer_counts(config):
    local = set(config["local_layer_ids"])
    layers = config["num_hidden_layers"]
    return {
        "unique_layers": layers,
        "unique_local": len(local),
        "unique_global": layers - len(local),
        "width": config["hidden_size"],
        "local_window": config["sliding_window_size"],
        "routed_experts": config["n_routed_experts"],
        "selected_experts": config["num_experts_per_tok"],
        "shared_experts": config["n_shared_experts"],
    }


def estimate_cache(config, passes, tokens=8192):
    loop = config["loop_config"]
    layers = config["num_hidden_layers"]
    core = range(loop["core_start"], loop["core_end"])
    visits = list(range(layers)) + list(core) * (passes - 1)
    local = sum(layer in config["local_layer_ids"] for layer in visits)
    global_layers = len(visits) - local
    local_slots = local * min(tokens, config["sliding_window_size"])
    local_width = config["swa_head_dim"] * config["swa_num_key_value_heads"]
    global_width = config["head_dim"] * config["num_key_value_heads"]
    # K and V, each in BF16, with independent state for each layer visit.
    kv_bytes = 2 * 2 * (local_slots * local_width + global_layers * tokens * global_width)
    return {"passes": passes, "layer_visits": len(visits), "global_visits": global_layers,
            "local_visits": local, "tokens": tokens, "ideal_bf16_kv_bytes": kv_bytes}


def paired_results():
    with gzip.open(ARCHIVE, "rt") as stream:
        rows = [row for row in map(json.loads, stream) if row["model"] == "inkling-original"]
    result = {}
    for variant in sorted({row["variant"] for row in rows}):
        ordinary, forced = {}, {}
        for row in rows:
            if row["variant"] != variant or row["panel"] not in {"quality", "grammar"}:
                continue
            panel = ordinary if row["panel"] == "quality" else forced
            item = row["id"]
            if item in panel:
                raise ValueError(f"Duplicate {variant}/{row['panel']}/{item}")
            panel[item] = row
        if not ordinary or ordinary.keys() != forced.keys():
            raise ValueError(f"Unpaired questions for {variant}")
        outcomes = Counter()
        for item, a in ordinary.items():
            b = forced[item]
            if any(a[k] != b[k] for k in ("expected", "subject", "prompt_sha256", "input_tokens")):
                raise ValueError(f"Question metadata differs for {variant}/{item}")
            outcomes[(a["correct"], b["correct"])] += 1
        result[variant] = {
            "paired_questions": len(ordinary),
            "both_correct": outcomes[True, True],
            "ordinary_only": outcomes[True, False],
            "forced_only": outcomes[False, True],
            "neither_correct": outcomes[False, False],
            "forced_answers": Counter(r["answer"] for r in forced.values()),
            "gold_answers": Counter(r["expected"] for r in ordinary.values()),
            "ordinary_completion_tokens": Counter(r["output_tokens"] for r in ordinary.values()),
            "forced_completion_tokens": Counter(r["output_tokens"] for r in forced.values()),
            "ordinary_finish": Counter(r["finish"] for r in ordinary.values()),
            "forced_finish": Counter(r["finish"] for r in forced.values()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-only", action="store_true",
                        help="Read only the local Spark archive, with no network access.")
    args = parser.parse_args()
    report = {"original_inkling_paired": paired_results()}
    if not args.archive_only:
        configs = {name: read_json(name, "config.json")["text_config"] for name in REVISIONS}
        report["checkpoints"] = {name: dict(revision=REVISIONS[name], **layer_counts(config))
                                 for name, config in configs.items()}
        report["production_mini_weights"] = inventory_mini()
        variable = configs["TML-d24-varloop-research"]
        report["variable_loop_estimates"] = [estimate_cache(variable, p) for p in (1, 2, 3, 5)]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
