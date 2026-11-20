"""Serve the published or refitted three-layer dense prefix on a direct Spark pair."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import spark

LOADER_DIR = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader"
LOADER = LOADER_DIR + "/default_loader.py"
SOURCE_SHA = "995afc9196c02ca600094d73aec7f07e3fbff0d873f1952c64e98e325b3b857d"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("published", "scale_only", "scale_and_codes"))
    cli.add_argument("--patches", required=True, type=Path)
    cli.add_argument("--cache", required=True, type=Path)
    cli.add_argument("--rank", type=int, choices=(0, 1), required=True)
    for flag in ("--node-ip", "--head", "--interface", "--hcas"):
        cli.add_argument(flag, required=True)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--master-port", type=int, default=29542)
    args = cli.parse_args()
    args.recipe, args.nodes, args.transport = "glm53-nvfp4-prefill", 2, "roce"
    recipe = json.loads(spark.RECIPES[args.recipe].read_bytes())
    cache, patches = args.cache.expanduser().resolve(), args.patches.resolve()
    patch_name = "scale_only" if args.mode == "scale_only" else "scale_and_codes"
    if not all((patches / (patch_name + ext)).is_file() for ext in (".json", ".safetensors")):
        cli.error("prepare the dense prefix patch and checksum receipt on this rank")
    spark.check_host(recipe, cache)
    cmd = spark.serve_command(args, recipe, cache)
    image = json.loads((spark.ROOT / "images.json").read_bytes())[recipe["image"]]
    original = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", image, LOADER])
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError("the vLLM default loader differs from the tested revision")
    with tempfile.TemporaryDirectory(prefix="labski-glm-loader-") as tmp:
        patched = Path(tmp) / "default_loader.py"
        patched.write_bytes(original + b"\nfrom .labski_dense_overlay import install as _labski_install\n_labski_install(DefaultModelLoader)\n")
        cmd[cmd.index("--name") + 1] = "labski-glm-dense-serving"
        index = cmd.index("--entrypoint")
        cmd[index:index] = ["--volume", f"{patched}:{LOADER}:ro", "--volume",
                            f"{HERE / 'overlay.py'}:{LOADER_DIR}/labski_dense_overlay.py:ro",
                            "--volume", f"{patches}:/labski/glm-patches:ro",
                            "--env", "LABSKI_DENSE_MODE=" + args.mode]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
