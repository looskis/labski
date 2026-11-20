#!/bin/bash
# Build exllamav3 for GB10 and its matching TabbyAPI runtime.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${1:?give a build directory on the Spark}"
work=$(realpath "$1")
base=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
exllama=bd1f519172406a9560bd5c90e61000691bb69a07
tabby=a32a82e255d9ffe1b6e805cb37527a1f53d9a62f
checkout() {
    local path=$1 url=$2 rev=$3
    if ! test -d "$path/.git"; then
        git init -q "$path"
        git -C "$path" remote add origin "$url"
    fi
    if ! git -C "$path" cat-file -e "$rev^{commit}" 2>/dev/null; then git -C "$path" fetch --depth 1 origin "$rev"; fi
    git -C "$path" checkout -q --detach "$rev"
    test "$(git -C "$path" rev-parse HEAD)" = "$rev"
}
checkout "$work/exllama" https://github.com/turboderp-org/exllamav3.git "$exllama"
checkout "$work/tabby" https://github.com/theroyallab/tabbyAPI.git "$tabby"
mkdir -p "$work/wheels" "$work/runtime/tabby"
docker run --rm --platform linux/arm64 --gpus all --cpus 7 --memory 32g --memory-swap 32g \
    -e MAX_JOBS=4 -e TORCH_CUDA_ARCH_LIST=12.1 -e CUDA_HOME=/usr/local/cuda -e HOME=/tmp \
    -v "$work:/job" -w /job/exllama --entrypoint python3 "$base" \
    -m pip wheel --no-deps --no-build-isolation --no-cache-dir . --wheel-dir /job/wheels
git -C "$work/tabby" archive "$tabby" | tar -xf - -C "$work/runtime/tabby"
cp "$work/exllama/requirements.txt" "$work/runtime/exllama-requirements.txt"
docker build --platform linux/arm64 -f "$here/Dockerfile" -t labski-ling31-tabby:a32a82e "$work/runtime"
