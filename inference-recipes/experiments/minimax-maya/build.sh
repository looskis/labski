#!/bin/sh
# Build the pinned MiniMax engine on an ARM64 Spark.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "${1:?pass a directory for the sources and build}"
work=$(CDPATH= cd -- "$1" && pwd)
cd "$work"
checkout() {
    dir=$1
    url=$2
    rev=$3
    if ! test -d "$dir/.git"; then git init -q "$dir" && git -C "$dir" remote add origin "$url"; fi
    if ! git -C "$dir" cat-file -e "$rev^{commit}" 2>/dev/null; then
        git -C "$dir" fetch --depth 1 origin "$rev"
    fi
    git -C "$dir" checkout --quiet --detach "$rev"
    test "$(git -C "$dir" rev-parse HEAD)" = "$rev"
}
checkout source https://github.com/mw00/project-maya.git e75b3ef7c5f725ef50fafe8ae74b16f77982f85e
checkout llama https://github.com/ggml-org/llama.cpp.git 3cf03257f219afbe7334045ff7c6a06ac68c627d
if ! git -C source apply --reverse --check "$here/arm64.patch" 2>/dev/null; then
    git -C source apply --check "$here/arm64.patch"
    git -C source apply "$here/arm64.patch"
fi
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["vllm-0.33.0"])' "$here/../../images.json")
docker run --rm --platform linux/arm64 --cpus 5 --memory 12g --memory-swap 12g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$work:/job" --entrypoint sh "$image" -ec '
    python3 -m pip install --quiet --target /tmp/labski-cmake --no-deps cmake==3.31.6
    url=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-profiler-api-13-0_13.0.85-1_arm64.deb
    curl -fsSL "$url" -o /tmp/cuda-profiler-api.deb
    echo "420b0974a875e2f2484f784ad1eda88cc6ea4ec1a67b6b40a04dab93c663aabb  /tmp/cuda-profiler-api.deb" | sha256sum -c -
    dpkg-deb -x /tmp/cuda-profiler-api.deb /tmp/labski-profiler
    export CPLUS_INCLUDE_PATH=/tmp/labski-profiler/usr/local/cuda-13.0/targets/sbsa-linux/include
    cmake=/tmp/labski-cmake/cmake/data/bin/cmake
    "$cmake" -S /job/source -B /job/build-vllm -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=121a -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=/job/llama
    "$cmake" --build /job/build-vllm --target maya-mm3 -j4
'
sha256sum "$work/build-vllm/maya-mm3"
