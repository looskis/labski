# MiMo-V2.6-Pro MOPD ARVQ on four Sparks

The hybrid-21 checkpoint contains 363.11 GB of SafeTensors. The four-rank vLLM candidate uses the published ARM64 runtime, two-token MTP, four 8K slots, and a 4 GiB KV pool per rank. Its comparison workload is the four-Spark Q2_K baseline. Drowzeys measured the upstream hybrid with a different context and sampling configuration. Local ARVQ serving and answer quality are pending. [Recipe](../../recipes/mimo26-pro-arvq-tp4.json) · [Local Q2_K baseline](../../results/mimo-v26-pro-mopd-rpc-2026-11-20/) · [Upstream runtime and benchmarks](https://github.com/drowzeys/keys-MiMo-V2.6-Pro-MOPD-C3-Jarrelscy-ARVQ-4-DGX-Sparks-1M-Context/tree/d8a10997f2c6b1b6140b204fc5dddddd247d894b) · [Pinned model](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21/tree/bb83094e0075076cb0b553c965412e20b061b9c2)

Use the ordinary prepare command for a separate snapshot on every rank. The alternative below keeps the model files on the head’s NVMe and mounts a read-only view on each worker through the ConnectX fabric. At this checkpoint size, one copy would save about 1.09 TB of weight storage across three workers. Startup traffic and any reads during inference must be measured.

## Shared snapshot

The lab tested the mount on two physical Sparks with rclone v1.76.0 and a 639,446,688-byte GGUF. The worker requested an HTTP range and mapped the remote file with the local disk cache disabled. Four 4 MiB windows at offsets 0, 1,048,576, 17,825,796 and 635,252,384 had the same concatenated SHA-256 on both nodes, `b0215c10a1ca0e56298c7ae6bc9b028b591c4ff7b2f71bff53432f05aec3fe59`. This check covers random-access reads of a fixture. [Fixture revision](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a) · [rclone release](https://github.com/rclone/rclone/releases/tag/v1.76.0) · [HTTP serving](https://rclone.org/commands/rclone_serve_http/) · [FUSE mount](https://rclone.org/commands/rclone_mount/)

On each Spark set `HF_HUB_CACHE` and set `HEAD_FABRIC` to the head’s trusted fabric address. The HTTP endpoint serves the selected public snapshot to peers that can reach that address. On the head, stage the pinned model and start the read-only file server. Keep it running while the workers use the mount.

```sh
repo=drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21
rev=bb83094e0075076cb0b553c965412e20b061b9c2
sub=models--drowzeys--keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21/snapshots/$rev
hf download "$repo" --revision "$rev" --cache-dir "$HF_HUB_CACHE"
rclone serve http "$HF_HUB_CACHE/$sub" --copy-links --read-only --addr "$HEAD_FABRIC:20986"
```

On each of the three workers, set the same `rev` and `sub` values, then mount into an empty directory. Linux needs FUSE and `fusermount3`. Run the mount under the account that manages the serving container. The container must be able to read the mounted files.

```sh
mkdir -p "$HF_HUB_CACHE/$sub"
rclone mount :http: "$HF_HUB_CACHE/$sub" --http-url "http://$HEAD_FABRIC:20986/" \
  --read-only --allow-other --vfs-cache-mode off --buffer-size 8M --daemon
```

Use the four-rank ring launch instructions with recipe ID `mimo26-pro-arvq-tp4`. Pull the pinned serving image on each rank when using the shared snapshot. The lab candidate selects the switchless NCCL library with `--ring-nccl` and `--ring-only-init`. Once all serving ranks have exited, unmount each worker with `fusermount3 -u "$HF_HUB_CACHE/$sub"`, then stop the file server. [Four-rank launch](../../ring-tcp.md#use-roce-on-the-same-four-cables) · [Switchless NCCL build](../../results/inkling2-gb10-runtime-2026-11-19/)
