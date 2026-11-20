# What each K2 Horizon 375B NVFP4 expert rank needs on disk

The official checkpoint has 12 indexed SafeTensors files totaling **229.60 GB**. We fetched the full header of every shard at the pinned revision and matched all **134,300 tensor names** to the index. Inside the 58 routed layers, the 192 experts account for 206.90 GB. The remaining tensors account for 22.69 GB, including attention, embeddings, the output head, shared experts and routers. Headers add 16.83 MB. [IFM checkpoint](https://huggingface.co/IFM/K2-Horizon-375B-A23B-NVFP4/tree/d3a6e26e4d59243ed6e569e39ca6facf4da84c05) · [Per-shard sizes and header digests](375b-nvfp4.json)

The checkpoint stores **183.91 GB of packed 4-bit expert codes**, **22.99 GB of FP8 block scales**, and 0.27 MB of FP32 global scales. All 668 common tensors are BF16. The 367.82 billion routed weights therefore cost 4.50001 bits per weight including their scales. Across the full model, the official NVFP4 weight files are 3.30× smaller than IFM's 758.34 GB BF16 files. [Quantized components and dtype counts](375b-nvfp4.json) · [Official BF16 snapshot](https://huggingface.co/IFM/K2-Horizon-375B-A23B/tree/bb3bfb4b532488c05e84a090301f8259c68b319c)

| Expert ranks | Experts per rank per routed layer | Weight data and headers per rank | Combined across ranks | Exact byte ranges per rank |
| ---: | ---: | ---: | ---: | --- |
| 2 | 96 | 126.15 GB (117.49 GiB) | 252.31 GB | 1,737 / 1,733 |
| 4 | 48 | 74.43 GB (69.32 GiB) | 297.72 GB | 2,568 / 1,035 / 979 / 956 |

The calculation gives each rank contiguous expert IDs and a complete copy of every common tensor and header. For two ranks, dividing the original checkpoint by two would underestimate the input each rank needs by 10.57 GiB under this policy. A loader can partition common tensors again for tensor parallelism, and it can require additional buffers and KV memory. These figures describe checkpoint storage and transfer. Serving fit and output quality still require a live run.

A range fetcher can reduce the number of HTTP requests by reading across small gaps. On two ranks, merging gaps of up to 4 KiB takes the request count to **1,157 per rank** and adds only **133,632 bytes per rank**. Those counts allow a request to span an entire contiguous run. With a 64 MiB maximum per request, the counts are 3,366 / 3,158 (ranks 0 / 1) before merging and 2,786 / 2,582 afterward. The counts include the original header of each file. The rank's packed weights, block scales and global scales are all included. [Two- and four-rank range counts and bytes](375b-nvfp4.json)

A physical ARM64 DGX Spark independently opened all 12 pinned headers with SafeTensors and checked all 134,300 names, shapes and dtypes against the headers. It reproduced the published rank plans. The sparse fixtures occupied 16.86 MB and the check read zero weight payload bytes. [Spark metadata check](spark-headers.json)

## Compact shard with real weights

For the first 20.00 GB source shard, we wrote a regular SafeTensors file containing only rank 1 of a two-rank expert placement. This source shard also contains dense weights, so the compact file was **14.58 GB**. An independent CPU read matched all **3,519 tensor names, shapes and dtypes**, then compared all **14,584,549,248 payload bytes** against the original. This included 3,456 routed tensors and 63 common tensors. The complete source file matched IFM’s LFS SHA-256. This pilot ran on the preparation machine. A multi-Spark server has yet to load the compact K2 checkpoint. [Source and payload hashes](rank1-shard01.json) · [Writer and byte comparison](slice_shard.py)

## Two complete rank layouts

The lab built both rank directories from all **229.60 GB** of official NVFP4 shards. Each compact directory contains **126.15 GB** of weights. Together they use **252.29 GB**, 45.06% less disk than two complete copies (459.21 GB). Rank 0 owns experts 0–95, rank 1 owns 96–191, and both retain the 668 common tensors.

Each of the 12 complete source files matched its publisher LFS SHA-256. An independent SafeTensors reader compared the name, shape, dtype and complete payload of every tensor in the 24 compact shards against the full sources. It compared **252,274,991,616 payload bytes** across **134,968 tensor instances**, counting common tensors on both ranks. Both ranks contain the same number of payload bytes. Their serialized headers differ by 69,224 bytes. [Per-shard source and output hashes, per-rank indexes and asset digests](two-rank-parity.json)

On a physical ARM64 Spark, vLLM 0.33.0’s native linear expert-placement function selected the same 66,816 routed tensor names and all 668 common names as each compact index. This checks rank ownership through the installed runtime. Two-Spark weight residency, generation and accuracy remain unmeasured. [Native placement check](check_placement.py) · [Spark result](spark-expert-placement.json)

From this directory, set `SNAPSHOT` to the pinned official source, and write each rank into its own destination. The assembler checks every compact shard’s SHA-256, the seven runtime assets and the final rank-specific index. It can resume from previously verified shards. The source and compact output require independent paths. [Assembler](assemble.py)

```sh
python3 assemble.py "$SNAPSHOT" "$RANK0_SNAPSHOT" --rank 0
python3 assemble.py "$SNAPSHOT" "$RANK1_SNAPSHOT" --rank 1
```

On a 128 GB Spark, the assembler verified all twelve rank 0 compact files in the local cache and reproduced the pinned rank index SHA-256. The weight files totaled **126,145,868,808 bytes** and the index mapped **67,484 tensors**. This assembly reused files already on the Spark and filled missing data directly from the publisher. In two interrupted compact files on different Sparks, 743,440,384 and 10,816,061,440 bytes of existing output were retained. The Sparks recovered the two remaining suffixes using 13,169,786,072 upstream bytes including headers. Both final file hashes matched the independent builds from the full source. [Spark assembly and two recovery receipts](spark-rank-assembly.json)

## Download one expert partition directly to a Spark

A physical Spark rebuilt rank 0 of shard 9 from **10.299 GB of response bodies** across 242 successful HTTP ranges. The original shard was 19.961 GB, so the requests saved **48.4%** of the source bytes. The resulting 10.298 GB file had the same SHA-256 as the compact file built independently from the complete source. On a second Spark, rank 1 of shard 2 took 10.143 GB across 220 ranges from a 20.002 GB source, a 49.3% reduction. Its 10.142 GB compact file also matched the independent full-source SHA-256. Response-body counts include the source header and the small gaps between nearby tensors. They exclude HTTP headers, redirects and retry traffic. [Spark receipt](direct-ranges.json) · [Downloader](fetch_shard.py)

On a third shard, a Spark ran the published standard-library downloader against the upstream server. It built rank 0 of shard 11 from **11.078 GB** of planned response bodies across 238 ranges. The result matched the independent SHA-256 for the 11,077,141,648-byte compact file, and a second invocation verified and skipped it. [Spark receipt](direct-ranges.json) · [Downloader at the measured revision](https://github.com/looskis/labski/blob/843ca31/inference-recipes/experiments/k2-moe-weights/fetch_shard.py)

Run the following on each Spark from this directory. Set `RANK` to 0 or 1 and choose separate directories for the small upstream metadata and this Spark’s compact weights. `fetch_shard.py` uses the Python standard library and a local filesystem that supports hard links and sparse files. During a shard build, allow temporary disk for both the fetched ranges and the compact file.

```sh
set -eu
: "${RANK:?set RANK to 0 or 1}" "${METADATA:?choose a metadata directory}" "${DEST:?choose a rank directory}"
hf download IFM/K2-Horizon-375B-A23B-NVFP4 \
  --revision d3a6e26e4d59243ed6e569e39ca6facf4da84c05 \
  --include '*.json' '*.jinja' '*.py' --local-dir "$METADATA"
for shard in $(python3 -c 'import json; print(*sorted({r["file"] for r in json.load(open("two-rank-parity.json"))["shards"]}))'); do
  python3 fetch_shard.py "$shard" "$DEST" --rank "$RANK"
done
python3 assemble.py "$METADATA" "$DEST" --rank "$RANK"
```

We also interrupted the public downloader after two completed payload ranges on rank 1 of shard 12, using reads from the verified upstream file to simulate the HTTP response bodies. It resumed without rereading either payload range, reproduced the independently verified **6,079,636,016-byte** compact file, and verified and skipped the file on a further invocation. The final assembler reproduced the complete rank 1 index using only the eight metadata files and the twelve compact shards. [Recovery and final assembly check](direct-ranges.json)

## Reproduce

Run the inventory against the pinned snapshot. It reads the 12 headers and index, checks tensor offsets and the header SHA-256 values, and prints the rank totals for gap limits from zero through one MiB. The calculation assumes whole-expert placement and no extra expert replicas. [Inventory program](inventory.py)

```sh
python3 inventory.py "$SNAPSHOT"
```

To repeat the first-shard check, install the Python `safetensors` and `numpy` packages for the independent reader. The output file must be new.

```sh
shard=model-00001-of-00012.safetensors
python3 slice_shard.py "$SNAPSHOT/$shard" "rank1/$shard" --ranks 2 --rank 1
python3 slice_shard.py "$SNAPSHOT/$shard" "rank1/$shard" --ranks 2 --rank 1 --verify-existing
```
