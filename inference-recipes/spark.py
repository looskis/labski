#!/usr/bin/env python3
"""Prepare or launch one local DGX Spark rank. Requires only Python's stdlib."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RECIPES = {p.stem: p for p in sorted((ROOT / "recipes").glob("*.json"))}
IMAGES = json.loads((ROOT / "images.json").read_text())
CONTAINER_CACHE = Path("/weights")


def snapshot(checkpoint):
    folder = "models--" + checkpoint["model"].replace("/", "--")
    return Path(folder) / "snapshots" / checkpoint["revision"]


def checkpoints(recipe):
    yield recipe
    if "draft" in recipe:
        yield recipe["draft"]


def prepare(recipe, cache, dry_run):
    image = IMAGES[recipe["image"]]
    commands = [["docker", "pull", "--platform", "linux/arm64", image]]
    if not dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    for ckpt in checkpoints(recipe):
        cmd = ["docker", "run", "--rm", "--platform", "linux/arm64", "--network", "host",
               "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HF_TOKEN",
               "--env", "HF_HOME=/tmp/huggingface", "--volume", f"{cache}:{CONTAINER_CACHE}",
               "--entrypoint", "hf", image, "download", ckpt["model"],
               "--revision", ckpt["revision"], "--cache-dir", str(CONTAINER_CACHE)]
        if ckpt.get("exclude"):
            cmd += ["--exclude", *ckpt["exclude"]]
        commands.append(cmd)
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not dry_run:
            subprocess.run(cmd, check=True)


def serve_command(args, recipe, cache):
    nodes = recipe["nodes"][0] if args.nodes is None else args.nodes
    if nodes not in recipe["nodes"] or not 0 <= args.rank < nodes:
        raise ValueError(f"choose nodes from {recipe['nodes']} and a rank from 0 to nodes - 1")
    if not 0 < args.port < 65536:
        raise ValueError("API port must be between 1 and 65535")

    cmd = ["docker", "run", "--rm", "--name", "labski-" + args.recipe,
           "--platform", "linux/arm64", "--gpus", "all", "--network", "host",
           "--ipc", "host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
           "--volume", f"{cache}:{CONTAINER_CACHE}:ro",
           "--label", f"labski.nodes={nodes}", "--label", f"labski.rank={args.rank}"]
    env = {"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(CONTAINER_CACHE), **recipe.get("env", {})}
    if "container_memory_gib" in recipe:
        limit = f"{recipe['container_memory_gib']}g"
        cmd += ["--memory", limit, "--memory-swap", limit]
    model_path = str(CONTAINER_CACHE / snapshot(recipe))
    serve = ["serve", model_path, "--served-model-name", args.recipe,
             "--tensor-parallel-size", str(nodes), "--max-model-len", str(recipe["context"]),
             "--max-num-seqs", str(recipe["sequences"]),
             "--gpu-memory-utilization", str(recipe["memory_fraction"])]
    if nodes > 1:
        if not all((args.head, args.node_ip, args.interface, args.hcas)):
            raise ValueError("distributed serving needs --head, --node-ip, --interface and --hcas")
        for addr in (args.head, args.node_ip):
            ipaddress.IPv4Address(addr)
        if (args.head == args.node_ip) != (args.rank == 0):
            raise ValueError("only rank 0 uses the head IP as its node IP")
        if not 0 < args.master_port < 65536 or args.master_port == args.port:
            raise ValueError("choose a valid rendezvous port distinct from the API port")
        cmd += ["--device", "/dev/infiniband", "--cap-add", "IPC_LOCK"]
        env.update(VLLM_HOST_IP=args.node_ip, NCCL_SOCKET_IFNAME="=" + args.interface,
                   GLOO_SOCKET_IFNAME=args.interface, NCCL_IB_HCA="=" + args.hcas,
                   NCCL_IB_DISABLE="0", TP_SOCKET_IFNAME=args.interface)
        serve += ["--distributed-executor-backend", "mp", "--nnodes", str(nodes),
                  "--node-rank", str(args.rank), "--master-addr", args.head,
                  "--master-port", str(args.master_port)]
    elif args.head or args.node_ip or args.interface or args.hcas:
        raise ValueError("fabric settings require a distributed recipe")
    if args.rank == 0:
        serve += ["--host", "127.0.0.1", "--port", str(args.port)]
    else:
        serve.append("--headless")
    for flag, value in recipe["options"].items():
        serve.append(flag)
        if flag == "--speculative-config" and "draft" in recipe:
            value = dict(value, model=str(CONTAINER_CACHE / snapshot(recipe["draft"])))
        if isinstance(value, dict):
            serve.append(json.dumps(value, separators=(",", ":")))
        elif value is not True:
            serve.append(str(value))
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    return cmd + ["--entrypoint", "vllm", IMAGES[recipe["image"]], *serve]


def check_host(recipe, cache):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("run serve on an ARM64 Linux Spark, or use --dry-run to inspect it")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if "GB10" not in gpus:
        raise ValueError("nvidia-smi did not report a GB10 on this host")
    mem = next(line for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemTotal:"))
    # Firmware and the kernel reserve part of the nominal 64 / 128 GB.
    if int(mem.split()[1]) * 1024 < 0.9 * recipe["per_node_gb"] * 10**9:
        raise ValueError(f"this recipe targets {recipe['per_node_gb']} GB per node")
    for ckpt in checkpoints(recipe):
        if not (cache / snapshot(ckpt) / "config.json").is_file():
            raise ValueError(f"missing {ckpt['model']} snapshot on this node; run prepare")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="show checkpoint precision and supported node counts")
    for verb in ("prepare", "serve"):
        cli = sub.add_parser(verb)
        cli.add_argument("recipe", choices=RECIPES)
        cli.add_argument("--dry-run", action="store_true", help="print commands without running them")
    cli.add_argument("--nodes", type=int, help="default to the smallest recipe topology")
    cli.add_argument("--rank", type=int, default=0)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--master-port", type=int, default=29501)
    cli.add_argument("--head", help="ConnectX IPv4 address of rank 0")
    cli.add_argument("--node-ip", help="this rank's ConnectX IPv4 address")
    cli.add_argument("--interface", help="local ConnectX Ethernet interface")
    cli.add_argument("--hcas", help="comma-separated local RoCE HCA names")
    args = parser.parse_args()
    if args.action == "list":
        for name, file in RECIPES.items():
            r = json.loads(file.read_text())
            sizes = "/".join(map(str, r["nodes"]))
            print(f"{name:24} {sizes:5} x {r['per_node_gb']:3} GB  {r['precision']:24} {r['model']}")
        return
    recipe = json.loads(RECIPES[args.recipe].read_text())
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub")).expanduser().resolve()
    if args.action == "prepare":
        prepare(recipe, cache, args.dry_run)
        return
    command = serve_command(args, recipe, cache)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        check_host(recipe, cache)
        os.execvp(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
