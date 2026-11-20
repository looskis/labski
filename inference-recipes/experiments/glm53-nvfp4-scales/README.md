# Refit GLM-5.3-Flash in its deployed NVFP4 format

The panel experiment compares the NVIDIA checkpoint with the original Z.ai BF16 weights and uses the GB10’s native B12X matrix multiply. It holds both global weight and input scales fixed. The published results cover 18 routed-expert panels. [Weight and native output errors](../../results/glm53-flash-nvfp4-scales-2026-11-20/)

## Refit the dense prefix for a full-model comparison

Every text token passes through the dense MLPs in layers 0, 1 and 2. The full-model candidate changes their gate, up and down projections, covering 452,984,832 weights. The rest of the checkpoint uses the published parameters. The first candidate changes nine block-scale tensors. The second changes those scales and the nine associated packed tensors. The serving kernel and the checkpoint parameter shapes are identical in all three modes. Across all nine full matrices the joint fit lowered native GB10 output error from 13.32% to 12.38% on Gaussian inputs and from 13.19% to 12.23% on the sparse-outlier distribution. On two Sparks the joint refit scored 23/28 free answers against 22/28 for the original, and 7/8 strict lookups against 8/8. Throughput stayed near 13.6 tok/s. [Full-model comparison](../../results/glm53-flash-nvfp4-scales-2026-11-20/serving/) [Full matrix measurements](../../results/glm53-flash-nvfp4-scales-2026-11-20/#all-nine-dense-projections-in-the-first-three-layers)

From this directory, fetch the matching BF16 and published NVFP4 tensors. The lab read 906 MB of BF16 weights from the pinned release and the corresponding packed data from an existing local NVFP4 snapshot. Remove the `--nvfp4` argument to fetch both sources over HTTP. Every returned tensor is recorded with its source filename, byte range and SHA-256.

```sh
python3 sample.py dense-prefix --dense-prefix --nvfp4 "$NVFP4_SNAPSHOT"
image=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
docker run --rm --gpus all --ipc=host --network none --memory 12g \
  -v "$PWD:/work" -w /work --entrypoint python3 "$image" \
  refit_dense.py dense-prefix --out patches
docker run --rm --gpus all --ipc=host --network none --memory 12g \
  -v "$PWD:/work" -w /work --entrypoint python3 "$image" \
  compare.py dense-prefix --patches patches --out dense-projections.jsonl
```

Make the resulting `patches` directory available on both Sparks. Each rank needs its own complete local copy of the pinned NVIDIA checkpoint. Run the launcher on both with `RANK` set to 0 or 1, the same `HEAD_IP`, and each rank’s private fabric IP, interface and RoCE HCAs. The underlying configuration is the lab’s measured two-Spark 8K-chunk cold-prefill recipe. [Base recipe](../../recipes/glm53-nvfp4-prefill.json)

```sh
python3 serve_dense.py published --patches patches --cache "$HF_HUB_CACHE" \
  --rank "$RANK" --head "$HEAD_IP" --node-ip "$FABRIC_IP" \
  --interface "$INTERFACE" --hcas "$HCAS" --port 18200
```

Run the other two arms by replacing `published` with `scale_only` or `scale_and_codes`. Stop each server using `docker stop labski-glm-dense-serving` on both ranks between arms. The launcher hashes the vLLM loader extracted from the pinned image and mounts a temporary copy with a parameter substitution hook. At load time each rank verifies the patch digest, the original tensor bytes, the full expected key set, and each replacement’s shape and dtype. A completed iterator logs `LABSKI_DENSE_VERIFIED` with its verified and replaced tensor counts. The published control goes through the same loader and validates the original bytes.

### Score the same text under all three sets of weights

The likelihood probe uses the opening body text of all 62 WikiText-2 raw test articles. It excludes titles and section headings, keeps up to 512 tokens per article and scores each token after the first with its preceding context. It requests the actual target token's log probability from the running endpoint. The output records passage and token hashes, token counts and summed negative log likelihood per article. Divide the total negative log likelihood by the total scored tokens and exponentiate for perplexity. Compare only runs with matching hashes and counts. This is a short English article-prefix diagnostic. [Salesforce WikiText-2 test split](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1) · [Dataset authors and licensing](https://huggingface.co/datasets/Salesforce/wikitext)

```sh
curl -fL -o test.parquet https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet
uv run --with pyarrow python ../../wikitext_probe.py prepare test.parquet articles.jsonl
python3 ../../wikitext_probe.py score glm53-nvfp4-prefill articles.jsonl published-nll.jsonl
```

Run the scoring command against each refit after starting it, using a separate output file each time. With both files available, `python3 ../../wikitext_probe.py compare published-nll.jsonl refit-nll.jsonl` computes the candidate’s mean NLL change and perplexity ratio. Its 95% percentile interval resamples the 62 paired article prefixes 20,000 times, retaining each article’s token count.
