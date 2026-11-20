# Qwen3 0.6B on one 128 GB Spark

The lab ran the [BF16 recipe](../../recipes/qwen3-0.6b.json) on a GB10 reporting 121 GiB system memory, driver 580.178.04, Ubuntu 24.04.5 LTS and kernel 7.0.0-1025-nvidia. The vLLM 0.33.0 ARM64 image served a chat completion. With thinking disabled, the model returned “17 + 25 = 42.” to an arithmetic prompt that requested the number alone. The arithmetic was correct and the extra text missed the formatting instruction. [Request and response](completion.json)

The local Spark client ran three trials of four consecutive prose requests with a 64-token output budget. Median delivered throughput was **118.93 output tokens/s** across trials. All 12 requests finished at the output cap. Median time to the first reasoning or answer delta was 25.4–25.9 ms across trials. This is a short bring-up workload. [Trial summary](summary.json) · [Settings and checkpoint](settings.json) · [Raw requests, trial 1](trial-1.json)

```sh
python3 bench.py qwen3-0.6b --workload prose --requests 4 --concurrency 1 \
  --repeats 3 --max-tokens 64 --out runs/qwen3-0.6b-bringup
```

On this driver the NGC 26.10.pre1 container loaded the weights but failed with an unsupported PTX toolchain at engine initialization, including an eager-mode retry. The successful launch used the vLLM 0.33.0 digest in the settings, eager execution, and a 1 GiB KV budget.
