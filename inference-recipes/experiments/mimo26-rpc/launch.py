#!/usr/bin/env python3
"""Run a llama.cpp RPC worker or a single- or multi-Spark text server."""
import argparse
import ipaddress
import os
import re
from pathlib import Path

IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3"
MODEL = "mimo26-pro-mopd-q2k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path, help="directory containing the extracted runtime")
    parser.add_argument("--tcp", action="store_true", help="disable RPC RDMA on this process")
    sub = parser.add_subparsers(dest="role", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("host", type=ipaddress.IPv4Address, help="this worker's trusted fabric address")
    worker.add_argument("--port", type=int, default=50052)
    head = sub.add_parser("serve")
    head.add_argument("first_shard", type=Path)
    head.add_argument("peers", nargs="*", metavar="HOST:PORT", help="one endpoint per worker in split order")
    head.add_argument("--alias", default=MODEL)
    head.add_argument("--slots", type=int, default=4)
    head.add_argument("--context", type=int, default=8192, help="tokens per slot")
    head.add_argument("--split", help="relative layer shares for head, then workers")
    head.add_argument("--mtp", type=Path, help="use a local MTP sidecar to propose two tokens")
    args = parser.parse_args()
    engine = args.engine.resolve(strict=True)
    cmd = ["docker", "run", "--rm", "--name", "labski-mimo-" + args.role, "--gpus", "all",
           "--network", "host", "--ipc", "host", "--ulimit", "memlock=-1", "--memory", "116g",
           "--memory-swap", "116g", "--device", "/dev/infiniband", "-v", f"{engine}:/engine:ro"]
    if args.tcp:
        cmd += ["-e", "GGML_RPC_NO_RDMA=1"]
    if args.role == "worker":
        if args.host.is_unspecified:
            parser.error("select the worker's fabric address")
        binary = ["/engine/ggml-rpc-server", "--device", "CUDA0", "--host", str(args.host),
                  "--port", str(args.port)]
    else:
        first = args.first_shard.absolute()
        shard_name = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", first.name)
        if shard_name:
            prefix, index, count = shard_name.groups()
            if index != "00001" or int(count) < 1:
                parser.error("pass the first GGUF shard")
            files = [first.with_name(f"{prefix}-{i:05d}-of-{count}.gguf") for i in range(1, int(count) + 1)]
        else:
            files = [first]
        if args.slots < 1 or args.context < 1:
            parser.error("slots and context must be positive")
        if len(args.peers) not in (0, 1, 3):
            parser.error("specify zero, one or three worker endpoints")
        # Snapshot symlinks point outside their own directory. Bind every blob
        # under its GGUF shard name so the loader can discover all siblings.
        for file in files:
            cmd += ["-v", f"{file.resolve(strict=True)}:/models/{file.name}:ro"]
        devices = ["CUDA0"] + [f"RPC{i}" for i in range(len(args.peers))]
        rpc = ["--rpc", ",".join(args.peers)] if args.peers else []
        split = args.split or ",".join("1" for _ in devices)
        binary = ["/engine/llama-server", "--model", "/models/" + first.name, "--alias", args.alias,
                  *rpc, "--device", ",".join(devices),
                  "--split-mode", "layer", "--tensor-split", split, "--gpu-layers", "all",
                  "--fit", "off", "--ctx-size", str(args.slots * args.context), "--parallel", str(args.slots),
                  "--batch-size", "512", "--ubatch-size", "256", "--jinja", "--reasoning", "off",
                  "--no-cache-prompt", "--cache-ram", "0", "--verbosity", "4",
                  "--host", "127.0.0.1", "--port", "8000"]
        if args.mtp:
            draft = args.mtp.resolve(strict=True)
            cmd += ["-v", f"{draft}:/draft.gguf:ro"]
            binary += ["--spec-type", "draft-mtp", "--spec-draft-model", "/draft.gguf",
                       "--spec-draft-n-max", "2"]
    cmd += ["--entrypoint", "sh", IMAGE, "-ec",
            'export LD_LIBRARY_PATH="/engine:${LD_LIBRARY_PATH:-}"; exec "$@"', "sh", *binary]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
