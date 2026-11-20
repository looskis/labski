# Inkling 2 Small on GB10

**Thinking Machines’ native 393.48 GB Inkling 2 Small checkpoint served text on four 128 GB DGX Sparks** using SGLang, one tensor and expert rank per Spark, a 4,096-token request limit, and the switchless RoCE ring. This uses the published mixed NVFP4/MXFP8 weights and BF16 KV. At low and high requested reasoning effort it answered both the arithmetic and number-retrieval checks exactly. [Pinned recipe](../../recipes/inkling2-small-native.json) · [Six chat completions](native/native-effort.jsonl)

**For the tested 4K service, use two scheduler slots and a 0.5 sliding-window ratio.** It served two clients at 39.34 aggregate generated tokens/s and 2.98 GPU J/output token across the four Sparks. All six inventory lookups returned the exact answer, including both repeated-prefix inputs at 3,597 tokens. [Two-slot recipe](../../recipes/inkling2-small-native-c2.json)

A single-rank hardware probe found an architecture mismatch before loading the 393 GB checkpoint. The designated vLLM ARM64 image recognizes Inkling 2, but its routed-expert implementation loads the FlashInfer SM100 module and requests kernels for CUDA major architecture 10. The fleet's GB10 reports compute capability **12.1**. Executing that architecture-selection call on a Spark raised `RuntimeError: No supported CUDA architectures found for major versions [10].` PyTorch reported `2.14.0+cu130` and vLLM reported `0.1.dev25249+gb974b306f.cu130`. This is a kernel compatibility result. It does not measure model memory, quality or throughput. [vLLM deployment guide](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small) · [Probe source](probe.py)

```sh
docker run --rm -i --gpus all --entrypoint python3 \
  vllm/vllm-openai:inkling-2-small@sha256:bc9f13e3fca71bea6a8603cc880fc705b6ebe3c21b438371d744ace7de07f0e7 \
  - < probe.py
```

The four-node deployment uses Aiden Le's pinned GB10 SGLang image and its FlashInfer expert hooks. It retains the official checkpoint, a BF16 KV cache, and a short 4K text workload. Expert parallelism distributes whole experts. Plain decoding and an 8K total token pool keep the initial fit experiment small. For the lab's routed ConnectX ring the launcher offers a TCP collective baseline. The original author's four-node deployment uses switched RoCE and a different KV and drafting configuration. [Lab recipe](../../recipes/inkling2-small-native.json) · [Upstream GB10 implementation](https://github.com/aidendle94/sparkrun-recipes/tree/3d7e798cfaecf1298576b79c945455bb5a75ac46/recipes/inkling-2-small/sglang)

The SGLang image runs on all four lab hosts. Its OCI revision label is `7832fd9+image`, and the baked `/patches/sitecustomize.py` has SHA-256 `9545ffa8280ce5210df0b0134802adae323307b12978ffc54b146e78e3419ac8`. The ARM64 image includes the FlashInfer MoE and paged-decode hooks on its Python path. The installed launcher advertises every flag used by the lab recipe. The later upstream recipe reference describes configuration changes since that image build. These checks cover the image and command line. The native completions above exercise the full checkpoint. [Source revision recorded by the image](https://github.com/aidendle94/sparkrun-recipes/tree/7832fd9d9d123837973b05c16197cac98370804b/recipes/inkling-2-small/sglang)


## Four-rank SGLang transport probe

The original SGLang image loaded NCCL 2.30.7 and timed out while moving a RoCE queue pair into the ready-to-receive state. A second run replaced the library with the unmodified NCCL 2.31.2 extracted from the working Qwen vLLM image (SHA-256 `7e2c3af111ba76885ed2a63c404e719af7253d430b26775488e83eb0d47c129d`). All four ranks exited with errors in that run as well. [Per-rank stock library checks](stock-library-controls.jsonl) With a ring-specific build of NCCL 2.31.2, the same SGLang image completed eight GPU all-reduces at each of three sizes on all four ranks. Every rank checked every element against the expected sum. Per-rank median latency ranged from 0.438 to 0.443 ms at 1 KiB, 0.490 to 0.491 ms at 64 KiB, and 2.589 to 2.633 ms at 1 MiB. Both neighbor ports on each machine recorded RDMA bytes. These are collective microbenchmarks. [Build and launch settings](../../transport/nccl-switchless/) · [Rank-level timing](ring-patched-collectives.jsonl) · [Port counters](ring-patched-bytes.csv)

## Select the checkpoint’s native precision metadata

With checkpoint revision `4d56448`, SGLang selected its generic ModelOpt loader and all four ranks stopped at model construction with `Inkling2 requires its checkpoint precision metadata`. Two files in the official snapshot describe quantization. The top-level shortcut in `config.json` says ModelOpt NVFP4. The separate `hf_quant_config.json` identifies Inkling 2 and describes five parameter groups, including both NVFP4 and MXFP8. The pinned SGLang image checks the shortcut first. [Checkpoint configuration](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/4d564489295e30e9341a9822a8046f0c9ded1a7b/config.json) · [Native precision map](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/4d564489295e30e9341a9822a8046f0c9ded1a7b/hf_quant_config.json)

The lab recipe passes `--json-model-override-args '{"quantization_config":null}'`. SGLang then reads the checkpoint’s native precision map and all four ranks enter weight loading. The 25 SafeTensors files, totaling 393,480,084,520 bytes, match the older upstream-tested revision by file hash. The newer revision changed both the chat template and the text model’s `log_scaling_n_floor` from 128000 to 8192. The override preserves both changes. [Compared revisions and tensor-manifest hashes](checkpoint-metadata.json) · [Recipe](../../recipes/inkling2-small-native.json)

Keep the management node’s unified memory available during startup. A separate checkpoint download used 16.6 GiB of resident memory on rank zero. With that job active, SGLang observed 95.07 GiB on the smallest rank and rejected the imbalance against peers with about 113 GiB. After restarting the download with two workers, streamed HTTP and a 2 GiB container limit, the four ranks reported 111.53–113.25 GiB available at the start of weight loading. The runtime’s memory-balance check stayed enabled.

## Fit the short-context KV pools

After all four ranks loaded their weights, SGLang’s default sliding-window ratio of 0.1 assigned 819 window tokens under the 8,192-token cap. The allocator needs at least 1,024 tokens to admit one request (a 1,023-token window plus one page). All four ranks rejected the smaller pool. Setting `--swa-full-tokens-ratio 0.25` allocated **8,192 full-attention tokens and 2,048 sliding-window tokens per rank** in BF16 with the same cap. In that launch SGLang measured a 95.12–95.54 GiB decrease in available memory through weight loading on each rank. All four completed that phase in about 162 seconds. [Per-rank memory and allocation results](native-load.jsonl) · [Short-context recipe](../../recipes/inkling2-small-native.json)

## Reproduce the first completions

Build the ring-specific library and prepare the pinned checkpoint on each Spark. Use one rank per host and supply the local rank address, neighbor HCAs and link subnet from the ring instructions. The serving image already includes the GB10 kernels. [NCCL build](../../transport/nccl-switchless/) · [Example ring addresses](../../ring-tcp.md#use-roce-on-the-same-four-cables)

```sh
python3 spark.py prepare inkling2-small-native
python3 spark.py serve inkling2-small-native --nodes 4 --rank "$RANK" \
  --head "$HEAD_IP" --node-ip "$NODE_IP" --interface "$FABRIC_IF" \
  --transport ring-roce --hcas "$CCW_HCA,$CW_HCA" --fabric-range "$LINK_CIDR" --link-prefix 30 \
  --ring-nccl "$NCCL_LIBRARY" --ring-only-init
```

Send separate greedy chat requests with `max_tokens=256` and `chat_template_kwargs.reasoning_effort` set to `none`, `low` or `high`. The prompts were “Return only the sum of 19 and 23.” and “Bin N5 holds the number 321907. Return only the number held in bin N5.” Low and high each gave two exact answers. At `none` the sum was exact and the lookup response included an explanation alongside the correct number. The first arithmetic request took 154.42 seconds while prefill kernels compiled. The other five requests took 2.11–6.38 seconds, with different output lengths. Each rank spent about 128 seconds capturing its batch-one decode graph before API startup. Available memory after graph capture was 14.04–15.19 GiB per rank. [Reply hashes, token counts and timing](native/native-effort.jsonl) · [Per-rank graph capture](native/decode-graph.jsonl)

## One-slot throughput and GPU energy

With one scheduler slot, the four-Spark endpoint delivered **22.99 generated tokens/s at one client** and **23.04 aggregate tokens/s at two clients**. The second client queued. The median summed GPU energy was **5.18 and 5.21 J per output token**, respectively. All 24 measured prose requests reached their 96-token cap. The rates include prefill and scheduling, and the token count includes reasoning. Each concurrency had an excluded warm-up and three timed trials with four prompts each. A short CPU tokenizer check ran during the two-client block. GPU energy comes from synchronized per-rank NVML counters. [Per-trial timing](native/one-slot-trials.jsonl) · [Per-request measurements](native/one-slot-requests.jsonl.gz) · [Settings](native/one-slot-settings.jsonl) · [Integrated energy](native/inkling-base-energy.jsonl) · [Raw counters](native/inkling-base-energy.csv.gz)

At the template’s default effort, all four short inventory responses stopped with the exact six-digit number. These were separate from the timed prose cases. [Lookup trial](native/one-slot-lookup.json)

## Two-slot throughput and shared-prefix context

| Four Sparks, native Inkling | One client, output tok/s | Two clients, aggregate output tok/s | One client, GPU J/token | Two clients, GPU J/token |
| --- | ---: | ---: | ---: | ---: |
| One slot and 2,048 sliding-window entries | 22.99 | 23.04 | 5.18 | 5.21 |
| Two slots and 4,096 sliding-window entries | 22.66 | 39.34 | 5.37 | 2.98 |

At two clients the second configuration delivered **71% more aggregate throughput** and used **43% fewer GPU joules per generated token**. At one client throughput was 1.4% lower. The 24 timed prompts, input-token counts and 96-token outputs matched case for case. The three two-client trials with two slots delivered 38.24, 41.35 and 39.34 tok/s. The change set comprised two scheduler slots, a batch-two decode graph and a larger sliding-window allocation. Both runs used the same weights, BF16 KV, 8,192 full-attention entries, 4K request limit and four-rank RoCE transport. [Two-slot settings](native/two-slot-settings.jsonl) · [Per-rank pool and graph sizes](native/two-slot-pools.jsonl) · [Trials](native/two-slot-trials.jsonl) · [Request hashes and timings](native/two-slot-requests.jsonl.gz) · [GPU energy](native/inkling-two-energy.jsonl) · [Raw counters](native/inkling-two-energy.csv.gz)

With the two-slot configuration the six consecutive inventory requests all returned the exact number at 519, 2,058 and 3,597 input tokens. The two longest requests took 2.01 and 2.00 seconds without a cache flush. At low reasoning effort, this launch scored 8/28 under the fixed 256-token limit against 7/28 with one slot. One item changed from incorrect to correct. Twenty responses hit the budget versus 21 with one slot. These are single passes. To launch this version, substitute `inkling2-small-native-c2` for the recipe and model name in the four-rank commands above. [Retrieval trials](native/two-slot-context.jsonl) · [Matched low-effort responses](native/c2-low-quality.jsonl.gz)

## Reasoning effort at a 256-token limit

On the same 28 MMLU-Pro questions, with greedy decoding and one scheduler slot, requested effort `none` gave **14/28 exact answers** while `low` gave **7/28**. The run at low effort generated 6,531 tokens, compared with 3,404 at none. Low effort exhausted the budget on 21 items and none exhausted it on 11. Moving to low gained two correct items and lost nine. The native chat template added two prompt tokens at low on every question. The score requires a stopped response containing just the correct letter. At none, three stopped responses included additional text. This measures short-answer delivery under a fixed budget. [None-effort items](native/none-quality.jsonl.gz) · [Low-effort items](native/low-quality.jsonl.gz) · [Probe command](../../measurement.md#score-native-chat-reasoning-effort)

## Shared-prefix retrieval near the context limit

The one-slot endpoint returned the exact inventory number for both target depths at 519 input tokens and both at 2,058 tokens. After those four requests, the next lookup shared its inventory prefix with the earlier prompts. The client still had no response about 185 seconds after the preceding completion, and the lab cancelled it. After aborting the pending request and flushing SGLang’s prefix cache, the **same 3,597-token request** stopped with the exact number in **2.91 seconds**. This is one observed stall and one recovery. The one-slot run did not exercise the second target depth at 3,597 tokens. [Request hashes, lengths and repeat](native/one-slot-context.jsonl)

To reproduce the sequence, run `python3 context_corpus.py /path/to/tokenizer.json` using the pinned tokenizer, then submit the generated prompts in order to chat completions with temperature zero, a 256-token limit and native reasoning effort `none`. Depths of 0.12 and 0.84 refer to the target’s fractional position in the inventory. The generator measures plain-text tokens before the model adds chat-template tokens. [Prompt generator](context_corpus.py)
