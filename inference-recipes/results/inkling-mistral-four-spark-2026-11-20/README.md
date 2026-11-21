# Inkling and Mistral Large 4 on four DGX Sparks

Four 128 GB Sparks served the full text weights of the 947B-parameter Inkling Q3_K_M and Mistral Large 4 UD-IQ3_XXS. Each launch used four client slots, 8,192 tokens per slot and three RPC workers. With the original Inkling, the FP16 cache configuration led at four clients and returned one more correct short answer. With Mistral, changing layer shares moved median prose throughput by less than 2% at either client count.

| Model and launch | One client, output tok/s | Four clients, aggregate output tok/s | Short answers, exact / 28 | Inventory lookups, exact / 8 |
| --- | ---: | ---: | ---: | ---: |
| Inkling, unified FP16 cache | 6.15 | 9.91 | 18 | 8 |
| Inkling, separate slots with Q8_0 keys | 6.37 | 9.69 | 17 | 8 |
| Mistral, balanced layer shares | 4.49 | 9.43 | 0 | 7 |
| Mistral, equal layer shares | 4.42 | 9.51 | 0 | 6 |

Each cell for throughput is the median of three four-request prose trials, with an excluded warm-up and a 96-output-token cap. The score for short answers required a stopped, exact answer within 16 generated tokens. All 28 Mistral responses in each launch reached that cap. Its separate 112-question, one-token forced-letter check scored 10/112 with balanced shares and 11/112 with equal shares. The eight inventory requests allowed 128 output tokens. Each configuration had one server launch. [Trial settings, timings and individual requests](trials.jsonl.gz) · [Prompt hashes, answers, finish reasons and scores](quality.jsonl.gz)

## Inkling

Both launches used layer shares of `35,32,32,35` in head and worker order and set numeric zero thinking effort. The FP16 launch used a unified KV buffer and a physical microbatch of 256. The second launch used separate slots, Q8_0 keys, FP16 values and a physical microbatch of 64. The measurements capture the combined effect of these changes. The checkpoint, engine build, worker transport and both launch commands are in the [Inkling recipe](../../experiments/inkling-original/#stage-the-original-model-for-serving).

## Mistral Large 4

Build the CUDA SM121a engine from upstream llama.cpp revision `6aee50801d091f78b3baacd505f5e95d8e8f6838` with RPC and RDMA enabled. Use the same runtime build on all four Sparks. Stage all ten UD-IQ3_XXS shards on the head. Start the three workers, using RDMA on the two adjacent links and TCP to the opposite Spark. [Pinned model shards](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF/tree/1eb4373d37e76f6e7b4b8efa20e1b1ca40b3b964/UD-IQ3_XXS) · [Engine source](https://github.com/ggml-org/llama.cpp/tree/6aee50801d091f78b3baacd505f5e95d8e8f6838) · [Build flags](../../experiments/mimo26-rpc/compile.sh) · [Worker setup](../../experiments/mimo26-rpc/#launch)

From the repository root, set `ENGINE` to the extracted runtime directory, `FIRST_SHARD` to the first GGUF shard, and the three worker variables to their fabric addresses. The balanced launch used the command below. For the equal split, change `--split` to `1,1,1,1`. Both launches used unified FP16 KV and a physical microbatch of 256.

```sh
python3 inference-recipes/experiments/mimo26-rpc/launch.py "$ENGINE" serve "$FIRST_SHARD" \
  "$ADJACENT_1:50052" "$OPPOSITE:50052" "$ADJACENT_2:50052" \
  --split 35,36,36,39 --kv-unified --slots 4 --context 8192 --ubatch-size 256 \
  --alias mistral-large4-iq3xxs
```
