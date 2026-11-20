"""Use two Sparks as two replicas or one tensor-parallel Qwen endpoint."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shape", choices=("replicas", "tp2"))
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True, help="physical Spark index")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--api-host", type=ipaddress.IPv4Address, default=ipaddress.IPv4Address("127.0.0.1"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--head", help="rank zero address for tensor parallelism")
    parser.add_argument("--node-ip", help="local fabric address for tensor parallelism")
    parser.add_argument("--interface", help="local fabric interface for tensor parallelism")
    parser.add_argument("--hcas", help="comma-separated RoCE adapters for tensor parallelism")
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.api_host.is_unspecified:
        parser.error("specify the API's loopback or serving-interface address")
    if args.shape == "tp2" and args.rank == 1 and not args.api_host.is_loopback:
        parser.error("the second tensor-parallel rank has no API endpoint")
    args.nodes = 2 if args.shape == "tp2" else 1
    args.transport = "roce" if args.nodes == 2 else None
    if args.nodes == 1:
        args.rank = 0
    args.recipe = "qwen38-27b-nvfp4"
    recipe = json.loads(spark.RECIPES[args.recipe].read_bytes())
    recipe["sequences"] = 16 * args.nodes
    recipe["options"]["--kv-cache-memory-bytes"] = 3 * 1024**3
    args.cache = args.cache.expanduser().resolve()
    if not args.dry_run:
        spark.check_host(recipe, args.cache)
    cmd = spark.serve_command(args, recipe, args.cache)
    cmd[cmd.index("--name") + 1] = "labski-qwen-pair-serving"
    if "--host" in cmd:
        cmd[cmd.index("--host") + 1] = str(args.api_host)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
