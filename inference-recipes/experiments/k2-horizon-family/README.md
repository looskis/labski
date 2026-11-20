# K2 Horizon size and precision on a shared text sample

The 7B and 3.7B tokenizers reproduced the 32B model’s exact token-ID hashes for all 62 WikiText-2 article prefixes in the lab archive. The 0.9B tokenizer matched none. A per-token perplexity comparison across those two tokenizers would change the unit of measurement. This experiment compares the cost, in bits, of predicting the same UTF-8 continuation from the same preceding text.

The 32B BF16 baseline assigned **0.5692 bits/byte** to 139,378 continuation bytes. The 32B FP8 and NVFP4 releases assigned **0.5723** and **0.5936 bits/byte** to the same bytes. All three values come from archived full-model log probabilities on one Spark. Lower values mean the model assigned more probability to the observed text. [BF16 article scores](reference/bf16.jsonl) · [NVFP4 article scores](reference/nvfp4.jsonl) · [One-Spark serving results](../../results/k2-horizon-32b-2026-11-20/) [FP8 article scores](reference/fp8.jsonl)

The 7B BF16 model reached 55.00 generated tok/s at four clients and 0.7140 bits/byte. Its stock low-effort parser returned no strictly correct open answers. The corrected parser returned 9/14 at 512 tokens, all eight streamed lookups and all 16 fresh record values in both API modes. [Live parser test](../../results/k2-horizon-family-2026-11-20/7b-parser/) [Size, precision and parser measurements](../../results/k2-horizon-family-2026-11-20/)

## How the comparison works

1. Reconstruct the original 32B window for each article from its recorded token-ID and source-text hashes. Use the text of its first token as the unscored context.
2. Tokenize the same full text and context separately with the 0.9B tokenizer. Keep an article only when the context ends at the same byte boundary in its normal tokenization of the full text. Two of 62 articles failed this check, dataset rows 789 and 1430. Some retained contexts comprise two or three 0.9B tokens.
3. For each retained article, sum the negative log probabilities of actual continuation tokens. The denominator is the total number of UTF-8 continuation bytes across the 60 articles. Divide the total negative log likelihood by that denominator and by the natural logarithm of two.

The live scorer verifies the token IDs returned by the serving endpoint. The projection command reuses an archived score only if both its underlying article and its token sequence match. The 7B, 3.7B and 0.9B BF16 launch configurations use the same driver-compatible vLLM image, 8,192-token limit, four scheduler slots, low reasoning effort and 3 GiB KV pool as the 32B experiment. [7B configuration](../../recipes/k2-horizon-7b-bf16.json) · [3.7B configuration](../../recipes/k2-horizon-3.7b-bf16.json) · [0.9B configuration](../../recipes/k2-horizon-0.9b-bf16.json)

## Reuse the official MoVA FP8 cache when staging BF16

The official FP8 release leaves all MoVA value experts, attention, shared experts, and the first three dense layers in BF16. Across the 48 pairs of official shard headers, we found **21.805 GB of BF16 tensors** with matching names, shapes and dtypes. Starting with the existing FP8 checkpoint, this leaves **53.086 GB** of BF16 ranges and headers to fetch for a 74.892 GB checkpoint, a projected **29.12%** reduction. A shard is installed only when its complete serialized bytes match the publisher’s BF16 LFS SHA-256. [Pinned sizes, header hashes, planned bytes and file digests](mova-shards.json) · [IFM quantization details](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B-FP8/blob/deaeefb1bb9e0c080be6d8844f3b222f90fd4bc7/README.md)

Two shards have passed that check on a physical Spark. Shard 1 needed just **1,368 bytes** of upstream BF16 header for a 167.78 MB output. For mixed shard 4, the Spark reused **416.35 MB** from its FP8 file and requested **1,179.69 MB** from a peer with a verified BF16 copy over a direct ConnectX-7 link. This produced the exact **1,596,042,512-byte** official file. The receiving Spark already had 612.37 MB of the FP8 sibling and fetched only its remaining 394.03 MB from that peer. Those sibling-transfer bytes are separate from the BF16 reconstruction cost. [Physical Spark receipts](mova-reuse-pilots.json)

In a separate recovery test on the Spark, we interrupted the public program while reconstructing mixed shard 4 from range reads against the verified local BF16 source. It kept **134,268,872 bytes** of the partial output, resumed without repeating a completed payload range, and produced the same official SHA-256. An additional invocation verified and skipped the finished file. [Interrupted-run receipt](mova-reuse-recovery.json)

On the Spark holding the FP8 snapshot, set `FP8_SNAPSHOT` and the destination `BF16_SNAPSHOT`. The second form reads a verified BF16 shard from a peer that supports HTTP ranges on the fabric. The utility checks existing final files and resumes its own partial outputs. It prepares weight shards. The target model also needs the configuration and tokenizer files from its pinned BF16 revision. [Reconstruction utility](reuse_mova.py)

```sh
shard=model-00004-of-00048.safetensors
python3 reuse_mova.py "$FP8_SNAPSHOT/$shard" "$BF16_SNAPSHOT"
# To read from a peer, set BF16_ORIGIN to the snapshot URL on its fabric interface.
python3 reuse_mova.py "$FP8_SNAPSHOT/$shard" "$BF16_SNAPSHOT" --bf16-shard-url "$BF16_ORIGIN/$shard"
```

## Reproduce

From this directory, set `ARTICLES` to the prepared 62-article file. Fetch the two pinned tokenizer JSON files and prepare a corpus in a fresh directory. The preparation step checks the 7B tokenizer against the original 32B token IDs on every article. [WikiText preparation](../glm53-nvfp4-scales/#score-the-same-text-under-all-three-sets-of-weights) · [7B checkpoint](https://huggingface.co/IFM/K2-Horizon-7B/tree/1ad5223c80f635ede5fb926f6304dc7228246a4b) · [0.9B checkpoint](https://huggingface.co/IFM/K2-Horizon-0.9B/tree/fc5c9dc7c13df2360acee789fca2cd55d439e644)

```sh
mkdir -p work
curl -fL https://huggingface.co/IFM/K2-Horizon-7B/resolve/1ad5223c80f635ede5fb926f6304dc7228246a4b/tokenizer.json -o work/7b.json
curl -fL https://huggingface.co/IFM/K2-Horizon-0.9B/resolve/fc5c9dc7c13df2360acee789fca2cd55d439e644/tokenizer.json -o work/0.9b.json
prior=../../results/k2-horizon-32b-2026-11-20
uv run --with tokenizers python byte_likelihood.py prepare "$ARTICLES" "$prior/bf16/wikitext.jsonl" \
  work/7b.json work/0.9b.json work/corpus.jsonl
python3 byte_likelihood.py project "$prior/nvfp4/wikitext.jsonl" work/corpus.jsonl work/nvfp4.jsonl
```

With the 0.9B model served under its recipe name, score the same bytes using its own tokenizer.

```sh
python3 byte_likelihood.py score k2-horizon-0.9b-bf16 work/corpus.jsonl work/0.9b-scores.jsonl
```
