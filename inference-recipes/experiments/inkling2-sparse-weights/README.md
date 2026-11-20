# Rank-local storage for native Inkling 2 Small

The four-Spark serving image requires each packed expert tensor to have its original shape before it selects 96 of 384 experts. Sparse files let each node keep the original SafeTensors headers and offsets while storing just the expert rows that its loader consumes. All common tensors, outer scale denominators and checkpoint headers are copied in full. The published 4-bit and 8-bit codes and scales are copied as bytes. [Measured four-Spark baseline](../../results/inkling2-gb10-runtime-2026-11-19/) · [Pinned model](https://huggingface.co/thinkingmachines/Inkling-2-Small/tree/4d564489295e30e9341a9822a8046f0c9ded1a7b)

The 24 shards in the model index occupy **390.06 GB** and contain 1,770 tensors. Expert rows account for 376.09 GB. The byte-range plan retains **107.99 GB per node**, or **431.95 GB across four ranks**, versus 1,560.23 GB for four copies of the indexed shards. This is a 72.31% reduction in indexed weight bytes. Filesystem block rounding, tokenizer files and the separate draft checkpoint are additional. [Per-shard byte ranges and header hashes](plan.json)

We staged all four rank layouts on a Spark’s ext4 NVMe. The 96 sparse files occupied **431,957,938,176 allocated bytes** (431.96 GB) while retaining 1,560.23 GB of logical file length. Each retained byte range was hashed during the copy. The measurements exclude the ten serving assets, which add 3.45 GB per standalone snapshot. [96 file allocations and SHA-256 digests](staged-shards.jsonl) · [Pinned serving assets](assets.json)

The complete CPU comparison covered **7,080 rank-visible tensors in all 96 files**. On each rank it checked 272 sliced expert tensors and 1,498 common tensors, totaling 107,988,363,584 payload bytes. Every SHA-256 matched its corresponding original weight or expert slice. The native expert loader performed its shape checks, rank selection and copy before device-specific scale swizzling. [Per-rank totals and loader digest](full-parity.json) · [Every tensor digest](full-parity-tensors.jsonl.gz)

On a Spark running ext4, we staged rank 3 of shard 22, which contains both NVFP4 and MXFP8 experts. The sparse file retained its **19,893,058,608-byte logical size** and occupied **5,273,714,688 physical bytes**. We compared all 54 rank-visible tensors against the original, including ten expert tensors. All **5,273,672,456 tensor bytes** matched. The two-slot endpoint served the complete sparse checkpoint on four physical Sparks. At a 768-token limit it reproduced all 14 final answers, including the same 13 correct letters. All six lookups were exact. [Paired serving comparison](../../results/inkling2-gb10-runtime-2026-11-19/rank-local/) [Exact per-tensor checksums and allocation](rank3-shard22.json) · [Verification program](verify.py)

## Reproduce the file check

Use a separate directory for each rank on a filesystem supporting sparse files. The input directory must be the pinned snapshot. The stage command verifies the size and header digest against the lab’s plan. A sparse file represents only its assigned rank. The unowned expert ranges read back as zero. Use the original snapshot for any other expert placement.

```sh
shard=model-00022-of-00024.safetensors
python3 stage.py "$SNAPSHOT" --rank 3 --file "$shard" --out rank3
du -B1 "rank3/$shard"
stat -c %s "rank3/$shard"
```

Run the parity check inside the same pinned ARM64 image. Bind the Hugging Face cache root so that the snapshot's relative links to blobs resolve. Set `SNAP_REL` to the snapshot directory relative to that cache root.

```sh
image='aidendle94/sparkrun-sglang-inkling2-gb10:experimental-1.0@sha256:fdcbdcf0b4aa686610985daac7caa3a5e9e3d0f3a74436e49b0c2db036818fc2'
docker run --rm --network none --cpus 2 --memory 12g --memory-swap 12g \
  -v "$PWD:/work:ro" -v "$HF_HUB_CACHE:/hub:ro" -w /work --entrypoint python3 \
  "$image" verify.py "/hub/$SNAP_REL/$shard" "rank3/$shard" --rank 3
```

## Stage a complete rank for serving

Repeat on each node, with `RANK` set to its expert rank (0 through 3). Set `SNAPSHOT` to the original pinned snapshot. Use a separate cache directory for the rank layout. This command also copies the tokenizer, configuration and auxiliary draft, checking their published digests. All ten staged assets were independently rehashed on a Spark.

```sh
revision=4d564489295e30e9341a9822a8046f0c9ded1a7b
rel="models--thinkingmachines--Inkling-2-Small/snapshots/$revision"
export HF_HUB_CACHE="$PWD/inkling-rank-$RANK"
python3 stage.py "$SNAPSHOT" --rank "$RANK" --out "$HF_HUB_CACHE/$rel" --with-assets
```

Use this `HF_HUB_CACHE` when launching `inkling2-small-native-c2` from `inference-recipes/`. The launcher checks the marker against the model revision, rank and expert-parallel size, and checks that the indexed shards exist. When transferring sparse files, use a sparse-aware copy such as `rsync -S`. Omit `--out` to print the plan without writing weights. [Stager](stage.py) · [Four-rank serving configuration](../../recipes/inkling2-small-native-c2.json)
