# Store only the Step-5 experts each Spark owns

The default four-rank Step-5-Preview-Spark loader assigns 88 of 352 routed experts in each MoE layer to each Spark. Replicating the upstream snapshot puts all 352 experts on every disk. Keeping the shared weights and each rank’s own experts brought the four-rank weight footprint from **1,365.52 GB to 485.27 GB**, a **64.46% reduction**. Each node needs 121.32 GB of weight files. These figures are logical on-disk bytes for safetensors files. Model configuration, tokenizer, runtime image and build caches are additional. [Pinned checkpoint](https://huggingface.co/0xSero/Step-5-Preview-Spark/tree/51a46a7f62eee1225d124170da903bba23d04a26) · [Upstream deployment](https://github.com/0xSero/Step-5-Preview-Four-Sparks/tree/662949a6c1244c1e8a5c44ed62cedb89a207cbcc)

| Placement | Expert IDs at a node | Routed-expert files per node | Common weights per node | Total safetensors per node |
| --- | --- | ---: | ---: | ---: |
| Four complete snapshots | 0–351 | 293.42 GB | 47.96 GB | 341.38 GB |
| Rank 0 | 0–87 | 73.35 GB | 47.96 GB | 121.32 GB |
| Rank 1 | 88–175 | 73.35 GB | 47.96 GB | 121.32 GB |
| Rank 2 | 176–263 | 73.35 GB | 47.96 GB | 121.32 GB |
| Rank 3 | 264–351 | 73.35 GB | 47.96 GB | 121.32 GB |

The partitioner rewrote only the 88 expert-file headers and copied the selected packed tensors in their original byte order. All 88 source files passed the pinned publisher SHA-256 manifest. Across the four derived layouts, an independent safetensors reader checked the keys, dtype, shape and complete payload of all **371,712 tensors** against the original. The comparison covered 293,373,324,288 payload bytes. The loader in the serving image has SHA-256 `1a5cecaa5ee896bc7bf09d3d3e4ff958bad5e22e5f341c0679a700b99fadebb4`. This placement requires its whole-expert mode, `STEP5_EXPERT_TP=0` (the default). Expert tensor parallelism uses a slice of every expert on every rank and requires the full expert set. [Exact per-file sizes and hashes](layout.json) · [Independent tensor comparison](tensor-parity.json)

The published four-Spark performance measurements used the full snapshot on each node. The smaller layout has a separate full-model replay in the lab queue. [Serving measurements](../../results/step5-preview-exl3-2026-11-20/)

## Download one rank directly from the Hub

A fresh Spark can fetch the shared weights and only the ranges needed for its own experts. In two live downloads, rank 0 built layer 3 from 842.60 MB of response bodies in 29 ranges, and rank 3 built layer 90 from 834.83 MB in 12 ranges. Each original file was 3.33 GB. Both outputs matched the locally partitioned files byte for byte. The downloader verifies each completed file against the published layout and skips verified files when restarted.

Across all 88 layers the range plan calls for 73.42–74.15 GB of expert payload downloads per rank. Including shared safetensors, the four-rank plan totals **486.35 GB of response bodies**, 64.38% below downloading four complete sets of weights. The total is calculated from all original tensor headers. The two layer measurements above exercised the public Hub. HTTP headers, retries and non-weight files are additional. [Per-rank plan and measured range requests](http-ranges.json)

Use a fresh local cache on each Spark. Set `RANK` to its rank and `HF_HUB_CACHE` to the cache directory used by the launcher. The staging directory below lives on the same volume. Run these commands from this experiment directory.

```sh
revision=51a46a7f62eee1225d124170da903bba23d04a26
staging="$HF_HUB_CACHE/step5-rank-staging"
snapshot="$HF_HUB_CACHE/models--0xSero--Step-5-Preview-Spark/snapshots/$revision"
python3 fetch_rank.py "$staging" --rank "$RANK"
hf download 0xSero/Step-5-Preview-Spark --revision "$revision" \
  --local-dir "$staging/rank$RANK" --exclude 'exl3/experts/*'
(
  cd "$staging/rank$RANK" &&
  echo 'a427f0ae33b2158f7550d21caa5e508a93b1e859283e0fcddf9ce7973b69f3c8  sha256-manifest.txt' | sha256sum -c - &&
  awk '$2 !~ /^exl3\/experts\//' sha256-manifest.txt | sha256sum -c -
) &&
mkdir -p "$(dirname "$snapshot")" &&
test ! -e "$snapshot" && mv "$staging/rank$RANK" "$snapshot"
```

Keep the original publisher manifest as a record of the source revision. The generated expert files use the SHA-256 values in `layout.json`. Start the served model directly from this cache using the four-rank recipe.

## Partition one source copy

Set `SOURCE` to the full checkpoint at revision `51a46a7f62eee1225d124170da903bba23d04a26`. Use a fresh output path on a volume with space for 293.42 GB of expert files. From this directory, run the following.

```sh
python3 partition.py "$SOURCE" "$LAYOUT"
cmp layout.json "$LAYOUT/layout.json"
uv run --with numpy --with safetensors python verify.py "$SOURCE" "$LAYOUT"
```

The result contains `rank0` through `rank3`. When an individual Spark already has the full pinned checkpoint, it can write just its own 73.35 GB of expert files and verify every file against the published layout.

```sh
python3 partition.py "$SOURCE" "$LOCAL_LAYOUT" --rank "$RANK" --layout layout.json
```

## Stage a complete rank directory

Use an isolated Hugging Face cache for these derived files. On the staging machine, select the destination for each rank and copy the common files plus its own expert files. Set `SNAPSHOT` to a fresh path under the destination user’s chosen cache. Use `LOCAL_LAYOUT` in place of `LAYOUT` for a locally built rank.

```sh
revision=51a46a7f62eee1225d124170da903bba23d04a26
# On a local destination, create the snapshot path first.
SNAPSHOT="$HF_HUB_CACHE/models--0xSero--Step-5-Preview-Spark/snapshots/$revision"
mkdir -p "$SNAPSHOT"
rsync -aL --exclude='/exl3/experts/' --exclude='/sha256-manifest.txt' "$SOURCE/" "$SNAPSHOT/"
rsync -a "$LAYOUT/rank$RANK/" "$SNAPSHOT/"
```

The publisher’s full-file manifest describes the original expert headers. Keep it with the source snapshot. Verify the rank-specific files on each destination before serving.

```sh
python3 - "$RANK" layout.json <<'PY' > rank-experts.sha256
import json, sys
for row in json.load(open(sys.argv[2]))["files"]:
    if row["rank"] == int(sys.argv[1]):
        print(row["sha256"], row["file"], sep="  ")
PY
(cd "$SNAPSHOT" && sha256sum -c -) < rank-experts.sha256
```

Use the four-rank serving command in the Step-5 recipe with that cache and the matching rank number. Running the ordinary `prepare` command in this cache would stage the complete upstream checkpoint again. [Four-Spark launch](../../results/step5-preview-exl3-2026-11-20/#reproduce)
