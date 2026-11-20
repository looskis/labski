# Measure and publish a run

Run `bench.py` from a client with Python 3.9+ and an SSH tunnel to the head. It speaks the streaming chat-completions API. Use the same client and network path for both sides of a comparison.

## Compare precision

On one 128 GB Spark, run `qwen38-27b-bf16`, then `qwen38-27b-nvfp4` with one rank. Run these commands for each deployment, changing `MODEL` to its recipe ID. The two profiles use an 8,192-token window, four scheduler slots, the same KV dtype, no drafting and disabled prefix caching. The target checkpoint changes from roughly 55.6 GB to 21.9 GB of root weight shards. These are download sizes. Record observed memory pressure on the Spark to evaluate working memory. [BF16 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B) · [Mixed NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)

```sh
MODEL=qwen38-27b-bf16
python3 bench.py "$MODEL" --workload prose --concurrency 1 --out "runs/$MODEL-prose-c1"
python3 bench.py "$MODEL" --workload code --concurrency 4 --out "runs/$MODEL-code-c4"
python3 bench.py "$MODEL" --workload quality --requests 32 --repeats 1 --max-tokens 1024 --out "runs/$MODEL-quality"
python3 bench.py "$MODEL" --workload lookup --records 256 --out "runs/$MODEL-lookup"
```

Set `--thinking off` or `--thinking on` for a Qwen template when that mode is part of the experiment. The default leaves the template’s own behavior in effect.

The quality workload uses deterministic arithmetic, JSON extraction, retrieval and code tracing. It keeps expected and actual answers, exact-match results and finish reasons. Inspect mismatches and truncated responses. These small checks can catch a regression. For claims about coding or tool use, add a task-level evaluation for the advertised capability.

## Compare topologies

The lab ran the replica sweep with Qwen3 0.6B on all four Sparks. [Measured results and reproduction command](results/qwen3-four-replicas-2026-11-19/)

Use the same `qwen38-27b-nvfp4` checkpoint for each row. Keep total requests, client concurrency and seeds fixed. On four independent one-rank services, forward their ports to the same client and run:

```sh
python3 bench.py qwen38-27b-nvfp4 --workload code --concurrency 4 \
  --endpoint http://127.0.0.1:8101 --endpoint http://127.0.0.1:8102 \
  --endpoint http://127.0.0.1:8103 --endpoint http://127.0.0.1:8104 \
  --out runs/qwen38-four-replicas-c4
```

For the four-rank service, forward the head to port 8101 and run:

```sh
python3 bench.py qwen38-27b-nvfp4 --workload code --concurrency 4 \
  --endpoint http://127.0.0.1:8101 --out runs/qwen38-tp4-c4
```

Repeat with one and two ranks and at concurrency one. A ring, a direct pair and a switched fabric have different paths between ranks. Validate connectivity and NCCL collectives on the selected fabric before timing a distributed service. A Wi-Fi management connection suffices for independent replicas.

## Read the measurements

Each endpoint gets one excluded warm-up. A trial defaults to 16 requests and the run defaults to five trials. The generator varies prompts per request and repeat, then replays those same prompts on the next configuration when the seed and settings match. The output budget is 256 tokens, temperature zero. Actual input and output counts come from the server. Budget-limited reasoning may consume all the output tokens before any visible answer.

- **Delivered output tok/s** divides all reported completion tokens by the trial’s elapsed client wall time. At concurrency one this includes prompt processing and delivery. At higher concurrency it measures aggregate service throughput. Report the median of complete trial rates.
- **First-delta latency** measures time to the first nonempty reasoning or answer delta, ignoring role and usage events. The trial summary gives the median and nearest-rank p95. Each request also records its time to first visible answer, total elapsed time, finish reason and actual lengths.
- **Cold prefill** requires verified cache state, fixed input IDs and a one-token response. Use the [GLM cold-prefill replay](results/glm53-prefill-2026-11-18/) for that metric. Long-prompt chat reports include generation.
- **Failed or incomplete requests** are retained with the trial, and suppress the aggregate rate. Each run writes to a new directory. Preserve failures when reporting a tuning sweep.

## Evidence to include with a public result

Save the recipe, model and draft revisions, image digest, client placement, rank/replica count, actual memory per Spark, driver, fabric layout and link speed, API template/thinking mode and any overrides. The harness embeds the catalog configuration when its model alias matches. Verify that against the container which served the requests. While the service is running, these commands collect the local versions and topology labels:

```sh
nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv
grep -E '^(MemTotal|MemAvailable|SwapFree):' /proc/meminfo
uname -r
rdma link show
docker inspect "labski-$MODEL" --format '{{.Config.Image}} {{.Image}} {{json .Config.Labels}}'
docker exec "labski-$MODEL" vllm --version
```

Capture memory on every rank during warm-up and steady load. On GB10, the OS and GPU share memory. Compare minimum observed system MemAvailable and swap alongside the runtime’s KV allocation. Include the sample interval for any peak estimate. Publish the command, raw trial records, exact-match counts, failures and limitations under `results/`, then link the recorded result from the catalog. Local scratch output lives in the ignored `runs/` directory.
