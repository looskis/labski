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

## GLM reasoning effort and answer budgets

GLM‑5.3‑Flash's official template accepts `reasoning_effort` values `low`, `high` and `max`. It defaults to `max` and ends the rendered generation prompt with `<think>`. The template does not read Qwen's `enable_thinking` flag. On the lab's NVFP4 endpoint, a 32-token lookup with the Qwen flag returned reasoning and a literal `</think>` before the correct value. Across eight approximately 17K-token lookups this produced zero exact-format passes and four length-limited responses. Those eight outcomes describe that endpoint and budget. A raw exact-format failure alone cannot distinguish retrieval, formatting and reasoning-budget effects. [Pinned GLM template](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/1d00738646cbb0c6d0a9eb403dac5f366f1473a7/chat_template.jinja)

Use the [MMLU-Pro preparation command](results/qwen38-27b-precision-2026-11-19/#reproduce) to create the same source questions, then run the following on the GLM API host. The probe tokenizes with the checkpoint's native effort, submits those exact IDs to completions, and scores the final option after the reasoning delimiter. The comparison fixes the budget at 256 generated tokens, selects two items per subject, and records whether a final answer was reached. Retain truncations in the denominator. Scores at this budget do not estimate maximum-effort capability with an unrestricted budget. [Effort probe](glm_effort.py)

```sh
for effort in low max; do
  python3 glm_effort.py glm53-chunk4096 /tmp/mmlu-112.jsonl \
    --effort "$effort" --per-subject 2 --max-tokens 256 --out "glm-$effort.jsonl"
done
```

On a pair of Sparks low effort scored 23/28 under this fixed budget. Maximum effort scored 14/28, with twelve truncations before the final answer. [Measured effort comparison](results/glm53-effort-2026-11-19/)

## Measure energy on GB10

On each participating rank, start the sampler before warm-up and let it run past the final trial. Check `timedatectl show -p NTPSynchronized` on the client and each rank. Choose a duration long enough for the run. Copy each CSV to the client under a distinct name.

```sh
python3 gpu_energy.py sample --seconds 300 --out rank-0.csv
```

The benchmark writes UTC nanosecond start and end times for each measured trial alongside its monotonic duration. Supply every participating rank's trace in rank order. The command interpolates the cumulative counter at both boundaries, sums the energy increments and divides by the actual generated token count.

```sh
python3 gpu_energy.py join runs/pair/prose-c4/trial-1.json rank-0.csv rank-1.csv
```

The reported unit is NVML joules per delivered output token, including GPU activity during prefill, decoding and scheduling inside the trial. The default sample interval is 200 ms. Clock alignment and the counter's update cadence limit precision for short trials. Report the number of ranks, trial duration and whether any background workloads shared the GPU. Whole-system or wall-outlet energy requires a separate meter. [NVIDIA cumulative energy counter](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)


### Score native chat reasoning effort

For a chat template that accepts a named reasoning effort, the probe can send that setting directly and score only the final answer. This command selects the same first two items in each of the 14 pinned subjects across configurations. Use a fresh output path for each run.

```sh
python3 mmlu_probe.py run inkling2-small-native labski-mmlu-112.jsonl \
  --reasoning-effort low --max-tokens 256 --per-subject 2 --out inkling-low.jsonl
```

Each item records the prompt hash, requested effort, finish reason, reported input and generated tokens, elapsed time and number of reasoning characters exposed by the server. Only a stopped response whose final content exactly matches the expected letter counts as correct. Match the prompt hashes and token budgets before comparing scores. [Scoring implementation](mmlu_probe.py)
