# What each K2 Horizon 375B NVFP4 expert rank needs on disk

The official checkpoint has 12 indexed SafeTensors files totaling **229.60 GB**. We fetched the full header of every shard at the pinned revision and matched all **134,300 tensor names** to the index. Inside the 58 routed layers, the 192 experts account for 206.90 GB. The remaining tensors account for 22.69 GB, including attention, embeddings, the output head, shared experts and routers. Headers add 16.83 MB. [IFM checkpoint](https://huggingface.co/IFM/K2-Horizon-375B-A23B-NVFP4/tree/d3a6e26e4d59243ed6e569e39ca6facf4da84c05) · [Per-shard sizes and header digests](375b-nvfp4.json)

The checkpoint stores **183.91 GB of packed 4-bit expert codes**, **22.99 GB of FP8 block scales**, and 0.27 MB of FP32 global scales. All 668 common tensors are BF16. The 367.82 billion routed weights therefore cost 4.50001 bits per weight including their scales. Across the full model, the official NVFP4 weight files are 3.30× smaller than IFM's 758.34 GB BF16 files. [Quantized components and dtype counts](375b-nvfp4.json) · [Official BF16 snapshot](https://huggingface.co/IFM/K2-Horizon-375B-A23B/tree/bb3bfb4b532488c05e84a090301f8259c68b319c)

| Expert ranks | Experts per rank per routed layer | Weight data and headers per rank | Combined across ranks | Exact byte ranges per rank |
| ---: | ---: | ---: | ---: | --- |
| 2 | 96 | 126.15 GB (117.49 GiB) | 252.31 GB | 1,737 / 1,733 |
| 4 | 48 | 74.43 GB (69.32 GiB) | 297.72 GB | 2,568 / 1,035 / 979 / 956 |

The calculation gives each rank contiguous expert IDs and a complete copy of every common tensor and header. For two ranks, dividing the original checkpoint by two would underestimate the input each rank needs by 10.57 GiB under this policy. A loader can partition common tensors again for tensor parallelism, and it can require additional buffers and KV memory. These figures describe checkpoint storage and transfer. Serving fit and output quality still require a live run.

A range fetcher can reduce the number of HTTP requests by reading across small gaps. On two ranks, merging gaps of up to 4 KiB takes the request count to **1,157 per rank** and adds only **133,632 bytes per rank**. Those counts allow a request to span an entire contiguous run. With a 64 MiB maximum per request, the counts are 3,366 / 3,158 (ranks 0 / 1) before merging and 2,786 / 2,582 afterward. The counts include the original header of each file. The rank's packed weights, block scales and global scales are all included. [Two- and four-rank range counts and bytes](375b-nvfp4.json)

## Compact shard with real weights

For the first 20.00 GB source shard, we wrote a regular SafeTensors file containing only rank 1 of a two-rank expert placement. This source shard also contains dense weights, so the compact file was **14.58 GB**. An independent CPU read matched all **3,519 tensor names, shapes and dtypes**, then compared all **14,584,549,248 payload bytes** against the original. This included 3,456 routed tensors and 63 common tensors. The complete source file matched IFM’s LFS SHA-256. This pilot ran on the preparation machine. A multi-Spark server has yet to load the compact K2 checkpoint. [Source and payload hashes](rank1-shard01.json) · [Writer and byte comparison](slice_shard.py)

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
