"""Serve the text-only MiniMax M3.1 Flash Maya-XS pack on one GB10."""
import argparse
import json
import os
from pathlib import Path
import shlex


def launch(args):
    work = args.directory.resolve(strict=True)
    for rel in ("build-vllm/maya-mm3", "pack/mm3.txt", "pack/tokenizer/chat_template.jinja"):
        if not (work / rel).is_file():
            raise ValueError(f"missing {rel} in the prepared directory")
    name = "minimax-m31-flash-maya-xs"
    config = dict(exe="/job/build-vllm/maya-mm3", cwd="/job/source", tokenizer="/job/pack/tokenizer",
                  model_name=name, gpu=[0], installer={"family":"minimax"}, port=args.port,
                  host="127.0.0.1", sampling={"temperature":0}, reasoning_effort="low",
                  args=["--pack","/job/pack","--max-context","8192","--parallel",str(args.slots),
                        "--kv-cache","bf16","--kv-streaming","off","--dspark","off",
                        "--vision","off","--suffix-draft","0"])
    if args.resident_cap is not None:
        config["args"].extend(["--resident-cap", str(args.resident_cap)])
    target = work / f"labski-serve-{args.port}.json"
    target.write_text(json.dumps(config, indent=2)+"\n")
    image = json.loads((Path(__file__).resolve().parents[2] / "images.json").read_text())["vllm-0.33.0"]
    cmd = ["docker","run","--rm","--platform","linux/arm64","--name","labski-"+name,"--gpus","all",
           "--network","host","--ipc","host","--memory","112g","--memory-swap","112g","--ulimit","memlock=-1",
           "-e","HF_HUB_OFFLINE=1","-e","HOME=/tmp","-v",f"{work}:/job","-w","/job/source","--entrypoint","python3",
           image,"-m","serve.server","--engine","maya","--config","/job/"+target.name,"--host","127.0.0.1",
           "--port",str(args.port)]
    print(shlex.join(cmd),flush=True)
    os.execvp(cmd[0],cmd)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--slots", type=int, choices=(1,4), default=4)
    cli.add_argument("--port", type=int, default=8000)
    cli.add_argument("--resident-cap", type=int, help="cap GPU-resident routed experts per MoE layer (0–128)")
    args = cli.parse_args()
    if not 0 < args.port < 65536:
        cli.error("--port must be in 1..65535")
    if args.resident_cap is not None and not 0 <= args.resident_cap <= 128:
        cli.error("--resident-cap must be in 0..128")
    launch(args)
