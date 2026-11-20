#!/usr/bin/env bash
set -euo pipefail
cmake -S /work/src -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=ON -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_NUMBER=11459 -DLLAMA_BUILD_COMMIT=f498f864f
cmake --build /work/build --parallel 10 --target llama-server ggml-rpc-server
cd /work/build/bin
tar czf /work/runtime.tar.gz .
