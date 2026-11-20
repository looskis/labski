"""Start the pinned Flash-Next reader with a switchable stock and exact-cache experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import spark

MODULE = "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/_shared/disk_table.py"
EXPECTED = "601348d52d488057911b5a0b22d36f0e922d80cb956c2658901bee56e63e7c9a"


def serve(args):
    recipe = json.loads(spark.RECIPES["qwen38-flash-next-fp4-local"].read_text())
    del recipe["options"]["--enable-prefix-caching"]
    recipe["options"]["--no-enable-prefix-caching"] = True
    rank = SimpleNamespace(recipe="qwen38-flash-next-ple-ab", nodes=1, rank=0, port=args.port,
                           interface=None, hcas=None, head=None, node_ip=None, transport=None)
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cache = Path(os.environ.get("HF_HUB_CACHE", home / "hub")).expanduser().resolve()
    spark.check_host(recipe, cache)
    cmd = spark.serve_command(rank, recipe, cache)
    image = spark.IMAGES[recipe["image"]]
    source = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat", image, MODULE])
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError("unexpected B12X reader source")
    output = args.directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hook = "trace_ple" if args.trace else "retained_rows"
    if not args.trace:
        pending = output / "control.next"
        pending.write_text("cache64 serving\n" if args.retain_rows else "idle idle\n")
        pending.replace(output / "control")
    source += ("\nimport sys\nsys.path.insert(0, '/labski-ple-code')\n"
               f"from {hook} import install as _install_ple_hook\n"
               "_install_ple_hook(DiskRowCache, '/labski-ple-run')\n").encode()
    module = output / "disk_table.py"
    module.write_bytes(source)
    cmd[2:2] = ["--volume", f"{module}:{MODULE}:ro", "--volume", f"{HERE}:/labski-ple-code:ro",
                "--volume", f"{output}:/labski-ple-run"]
    if args.kernel_cache:
        for key, dest in (("b12x","/root/.cache/b12x"), ("vllm","/root/.cache/vllm"),
                          ("flashinfer","/root/.cache/flashinfer"), ("triton","/root/.triton")):
            cmd[2:2] = ["--volume", f"{args.kernel_cache.resolve()/key}:{dest}"]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--kernel-cache", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace", action="store_true", help="record gathers while the replay marker is active")
    mode.add_argument("--retain-rows", action="store_true", help="start with the 64 MiB row cache active")
    parser.add_argument("--port", type=int, default=8000)
    serve(parser.parse_args())
