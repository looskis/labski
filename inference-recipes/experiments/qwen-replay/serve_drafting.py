"""Launch the single-rank Qwen3.8 27B recipe with a target-verified draft policy."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spark


def launch(args):
    recipe = json.loads(spark.RECIPES["qwen38-27b-nvfp4"].read_text())
    if args.mode == "ngram":
        recipe["options"]["--speculative-config"] = dict(
            method="ngram", num_speculative_tokens=5, prompt_lookup_min=2, prompt_lookup_max=5)
    if args.mode in ("suffix0", "suffix32"):
        recipe["options"]["--speculative-config"] = dict(
            method="suffix", num_speculative_tokens=5, suffix_decoding_max_tree_depth=16,
            suffix_decoding_max_cached_requests=0 if args.mode == "suffix0" else 32)
    rank = SimpleNamespace(recipe="qwen27-draft-" + args.mode, nodes=1, rank=0, port=args.port,
                           head=None, node_ip=None, interface=None, hcas=None, transport=None)
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser().resolve()
    cmd = spark.serve_command(rank, recipe, cache)
    if args.mode.startswith("suffix"):
        if not args.extension or not (args.extension / "arctic_inference/suffix_decoding").is_dir():
            raise ValueError("--extension must point to the built Python package directory")
        cmd[2:2] = ["--volume", f"{args.extension.resolve()}:/labski-suffix:ro", "--env", "PYTHONPATH=/labski-suffix"]
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        spark.check_host(recipe, cache)
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("none", "ngram", "suffix0", "suffix32"))
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    try:
        launch(parser.parse_args())
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+"\n")
