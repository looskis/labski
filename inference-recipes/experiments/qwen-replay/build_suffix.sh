#!/bin/sh
# Build the CPU suffix extension against the Python in the pinned ARM64 vLLM image.
set -eu
cd /experiment
archive=arctic_inference-0.1.1.tar.gz
source_url=https://files.pythonhosted.org/packages/2a/d2/3bcebda042d08cfa0f35a2bbc6dd168d60fea60b6678824665c7a3665418/arctic_inference-0.1.1.tar.gz
if ! test -f "$archive"; then curl -fL "$source_url" -o "$archive.partial" && mv "$archive.partial" "$archive"; fi
echo "36857e2fbe3ead34821a04be98dfab95086050c1e38199c8fdfe308234fdcf63  $archive" | sha256sum -c -
mkdir -p /tmp/arctic-src /tmp/arctic-build python/arctic_inference/suffix_decoding
tar -xzf "$archive" -C /tmp/arctic-src --strip-components=1
python3 -m pip install --quiet --no-deps cmake==3.31.6 nanobind==2.9.2
cp /tmp/arctic-src/arctic_inference/__init__.py python/arctic_inference/
cp /tmp/arctic-src/arctic_inference/suffix_decoding/*.py python/arctic_inference/suffix_decoding/
cmake -S /tmp/arctic-src/csrc/suffix_decoding -B /tmp/arctic-build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/experiment/python/arctic_inference/suffix_decoding
cmake --build /tmp/arctic-build -j 4
PYTHONPATH=/experiment/python python3 -c 'from arctic_inference.suffix_decoding import SuffixDecodingCache; SuffixDecodingCache(max_tree_depth=16, max_cached_requests=4)'
sha256sum python/arctic_inference/suffix_decoding/_C*.so
