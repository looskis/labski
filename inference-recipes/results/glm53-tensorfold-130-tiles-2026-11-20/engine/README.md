# Build the shape-specific GLM Q8 dispatch in TensorFold 1.3

The patch adds two prompt projection kernels and selects between them for the seven measured row counts at N = 12,576 and K = 4,096. The original dispatch remains the default. We built this engine on a GB10 and verified that both Sparks ran the same executable and reported `tensorfold-native 1.3.0`. The measured speedups to date are from the isolated projection kernel. Full-model prefill and answer comparisons are in the lab queue. [Kernel measurements and reference checks](../) · [Engine revision](https://github.com/ashhart/TensorFold/tree/81de26f649677aa72ae0034d4af6e010cd947038) · [Mia’s Spark integration](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/d8a7439e8c5aac8645c423ccd24189690359ab69)

## Prepare the source

Run on an ARM64 Spark with Git, Docker and curl. Start in this directory. The working directory will hold source, the toolchain, compiler caches and the native installation.

```sh
set -eu
patch="$PWD/0014-labski-q8-prompt-tiles.patch"
work=$(mktemp -d)
engine_revision=81de26f649677aa72ae0034d4af6e010cd947038
recipe_revision=d8a7439e8c5aac8645c423ccd24189690359ab69
mkdir "$work/src" "$work/recipe" "$work/zig" "$work/out" "$work/cache" "$work/global"
curl -fL --retry 3 "https://codeload.github.com/ashhart/TensorFold/tar.gz/$engine_revision" -o "$work/engine.tar.gz"
curl -fL --retry 3 "https://codeload.github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tar.gz/$recipe_revision" -o "$work/recipe.tar.gz"
tar -xzf "$work/engine.tar.gz" -C "$work/src" --strip-components=1
tar -xzf "$work/recipe.tar.gz" -C "$work/recipe" --strip-components=1
(
  cd "$work/src"
  set -- "$work"/recipe/patches/engine/*.patch
  test "$#" = 13
  for file in "$@" "$patch"; do
    git apply --check "$file" && git apply "$file"
  done
  echo 'a5789718fc8cb986235de1829433c2df474826f557843c4185f27903824212e4  zig/kernels/cuda/glm_mm.cu' | sha256sum -c -
  echo 'e213595907209cd9ceb56eea8d822b641b336eb3ec367a80f9677b80c7ff0ac6  zig/src/platform/environment.zig' | sha256sum -c -
  echo '709f103e0e280e7d4202232b672757e434845f783a36e8ad8f12e6d28960f4cf  zig/src/families/glm/cuda_mm.zig' | sha256sum -c -
)
```

## Compile and select a dispatch

The lab used the pinned ARM64 vLLM image for its CUDA 13.0.88 compiler, Zig 0.17.0, SM121, and three compile jobs. The first full-model launch reached two-rank NCCL initialization and checkpoint verification, then stopped while looking up the unregistered dispatch flag. The revised patch registers both flags in the strict GLM environment table and checks registration during compilation. After rebuilding, the executable on both Sparks had SHA-256 `a2ace4aee215a2293feea522af519b08292fe56164e6b589d72f81742c33a31a`. Check the digest after copying your build to the second Spark.

```sh
curl -fL --retry 3 https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o "$work/zig.tar.xz"
(cd "$work" && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  zig.tar.xz' | sha256sum -c -)
tar -xJf "$work/zig.tar.xz" -C "$work/zig" --strip-components=1
image=vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
docker run --rm --network none --memory 14g --memory-swap 14g --cpus 3 \
  -v "$work:/work" -w /work/src \
  -e ZIG_GLOBAL_CACHE_DIR=/work/global -e ZIG_LOCAL_CACHE_DIR=/work/cache \
  --entrypoint /work/zig/zig "$image" build native -j3 --prefix /work/out \
  --summary failures -Doptimize=fast -Dcpu=native -Dstrip=false \
  -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
sha256sum "$work/out/native/bin/tensorfold-native"
```

Mount the executable at `/opt/tensorfold/native/bin/tensorfold-native` in the compatible TensorFold runtime container on both ranks. Set `TF_GLM_Q8_PROMPT_TRACE=1` on both and compare `TF_GLM_Q8_PROMPT_TILES=0` with `TF_GLM_Q8_PROMPT_TILES=1`. The trace records the first dispatch per measured row count per calling thread. The original path logs `tuned_tile=0`. Tuned rows log `128` or `256`. [Lab patch](0014-labski-q8-prompt-tiles.patch) · [TensorFold notice](upstream-NOTICE.txt) · [Apache 2.0 license](upstream-LICENSE.txt) · [Retained MIT notice](upstream-MIT.txt)
