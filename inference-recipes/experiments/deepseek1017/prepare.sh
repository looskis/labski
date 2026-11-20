#!/bin/bash
# Build the upstream engine and materialize four rank files from the original 1017 text weights.
set -euo pipefail
snapshot=$(realpath "${1:?give the downloaded Hugging Face snapshot}")
mkdir -p "${2:?give a staging directory on the same filesystem as the weight blobs}"
stage=$(realpath "$2")
image=${IMAGE:-labski-deepseek1017:8ffa67d-arm64}
checkout() {
    local path=$1 url=$2 commit=$3
    if ! test -d "$path/.git"; then git init -q "$path"; git -C "$path" remote add origin "$url"; fi
    if ! git -C "$path" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$path" fetch --depth 1 origin "$commit"
    fi
    git -C "$path" checkout -q --detach "$commit"
    test "$(git -C "$path" rev-parse HEAD)" = "$commit"
}
checkout "$stage/engine" https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold.git 8ffa67d53ef15bab997399135f85e8b0910f30be
checkout "$stage/kit-source" https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10.git 6033ea48a27004b2715db82190e2a61f33411d92
(
    cd "$stage/engine"
    export BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
    source scripts/config.sh
    hash=$(image_hash)
    docker build --platform linux/arm64 -t "$image" -f zig/Dockerfile --build-arg BASE_IMAGE="$BASE_IMAGE" \
      --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" --build-arg ZIG_VERSION="$ZIG_VERSION" \
      --build-arg ZIG_SHA256="$ZIG_SHA256" --build-arg PATCHES_HASH="$hash" .
)
mkdir -p "$stage/head-model" "$stage/common/engram"
python3 - "$snapshot" "$stage" <<'PY'
from pathlib import Path
import json,os,shutil,sys
raw,stage=map(Path,sys.argv[1:])
common,view=stage/'common',stage/'head-model'
weight_map=json.loads((raw/'model.safetensors.index.json').read_text())['weight_map']
shards=set(weight_map.values())
if len(shards)!=48 or sum((raw/n).stat().st_size for n in shards)!=510296708312:
    raise ValueError('expected the complete 48-shard DeepSeek V4.1 Flash 1017 snapshot')
def link(src,dst):
    src=src.resolve(strict=True)
    if not dst.exists():os.link(src,dst)
    elif not dst.samefile(src):raise ValueError(f'{dst} already refers to another source')
for src in raw.iterdir():
    if src.is_file() and src.name!='chat_template.jinja':link(src,view/src.name)
for folder in (view,common):shutil.copyfile(stage/'engine/model/chat_template.jinja',folder/'chat_template.jinja')
for name in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'):
    link(raw/name,common/name)
engram={v for k,v in weight_map.items() if '.engram.' in k}
if len(engram)!=2 or sum((raw/n).stat().st_size for n in engram)!=203073077576:
    raise ValueError('unexpected Engram file inventory')
for name in engram:link(raw/name,common/'engram'/name)
PY
kit="$stage/kit-source/kit"
IMAGE="$image" "$kit/verify_kit.sh" "$stage/head-model" "$stage/head-model" > "$stage/kit-check.log" 2>&1 || true
python3 - "$stage/kit-check.log" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if not ('356 of 356 files match' in s or
        ('355 of 356 files match' in s and '1 differ, 0 missing' in s and 'DIFFERS (bias): vision/gate_bias_vl.safetensors' in s)):
    raise ValueError(s[-1500:])
PY
mkdir -p "$stage/common/kit"
cp -a "$kit/." "$stage/common/kit/"
run=(docker run --rm --network none --cpus 8 --memory 64g --memory-swap 64g -e HOME=/tmp
     -v "$stage/head-model:/model:ro" -v "$stage:/staging")
if ! test -s "$stage/common/dsv41_token_map.json"; then
    "${run[@]}" --entrypoint python3 "$image" /opt/tensorfold/tools/inputs/make_token_map.py /model /staging/common/dsv41_token_map.partial
    mv "$stage/common/dsv41_token_map.partial" "$stage/common/dsv41_token_map.json"
fi
for rank in 0 1 3 2; do
    mkdir -p "$stage/rank$rank"
    index=$("${run[@]}" --entrypoint tf-dsv41-rank-cache "$image" /model /tmp/unused "$rank" 4 --parity --index-only | sha256sum | cut -c1-20)
    file="$stage/rank$rank/rank${rank}of4-$index.bin"
    if ! test -s "$file"; then
        "${run[@]}" --entrypoint tf-dsv41-rank-cache "$image" /model "/staging/rank$rank" "$rank" 4 --parity
    fi
    test "$(stat -c %s "$file")" -gt 70000000000
done
