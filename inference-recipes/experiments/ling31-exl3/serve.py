"""Serve the 100.81 GB Ling EXL3 release on one DGX Spark."""
import argparse
import os
from pathlib import Path
import shlex

NAME = "ling31-exl3-low"
WEIGHT_BYTES = 100805781294
REQUIRED_FILES = ("config.json", "quantization_config.json", "chat_template.jinja", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json", "model.safetensors.index.json")


def command(work, snapshot, mtp=False):
    weights = sorted(snapshot.glob("*.safetensors"))
    if len(weights) != 13 or sum(p.stat().st_size for p in weights) != WEIGHT_BYTES:
        raise ValueError("expected 13 Ling 3.1 Flash 1019 mixed EXL3 shards totalling 100,805,781,294 bytes")
    wheels = list((work / "wheels").glob("exllamav3-1.6.6-*-linux_aarch64.whl"))
    if len(wheels) != 1:
        raise ValueError("build the ARM64 exllamav3 1.6.6 wheel first")
    # Expose the checkpoint under its API name. The upstream Tabby configuration
    # describes five GPUs, so stage only the tokenizer, model index and weights.
    view = work / "models" / NAME
    view.mkdir(parents=True, exist_ok=True)
    if (view / "tabby_config.yml").exists():
        raise ValueError(f"the model view at {view} contains a second Tabby configuration")
    mounts = set()
    for raw in [*weights, *(snapshot / n for n in REQUIRED_FILES)]:
        src = raw.resolve(strict=True)
        dst = view / raw.name
        if dst.exists() or dst.is_symlink():
            if not dst.samefile(src):
                raise ValueError(f"the existing model view refers to a different source at {dst}")
        else:
            dst.symlink_to(src)
        if dst.is_symlink():
            mounts.add(src.parent)
    config = Path(__file__).resolve().with_name("config.yml")
    cmd = ["docker", "run", "--rm", "--platform", "linux/arm64", "--name", "labski-" + NAME, "--gpus", "all",
           "--network", "host", "--ipc", "host", "--ulimit", "memlock=-1", "--memory", "116g", "--memory-swap", "116g",
           "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONUNBUFFERED=1", "-e", "HOME=/tmp",
           "-v", f"{view.parent}:/models:ro", "-v", f"{work / 'wheels'}:/wheels:ro", "-v", f"{config}:/app/config.yml:ro"]
    if mtp:
        cmd.extend(["-e", "TABBY_DRAFT_MODEL_DRAFT_MODE=mtp", "-e", "TABBY_DRAFT_MODEL_DRAFT_NUM_TOKENS=2"])
    for path in sorted(mounts):
        cmd.extend(["-v", f"{path}:{path}:ro"])
    wheel = "/wheels/" + wheels[0].name
    return cmd + ["--entrypoint", "sh", "labski-ling31-tabby:a32a82e", "-ec",
                  'python3 -m pip install --no-deps --disable-pip-version-check "$1"; exec python3 main.py', "sh", wheel]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--mtp", action="store_true", help="use the checkpoint’s native two-token draft")
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    cmd = command(args.work.resolve(strict=True), args.snapshot.resolve(strict=True), args.mtp)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        os.execvp(cmd[0], cmd)
