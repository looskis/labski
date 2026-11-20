#!/bin/sh
# Download and verify the three text-inference assets, then create Maya's tensor index.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=$(CDPATH= cd -- "${1:?pass the directory used by build.sh}" && pwd)
test -f "$work/source/tools/mm3/pack_gguf.py"
image=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["vllm-0.33.0"])' "$here/../../images.json")
docker run --rm --platform linux/arm64 --cpus 2 --memory 4g --memory-swap 4g \
    --volume "$here/weights.sha256:/labski-weights.sha256:ro" --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_XET=1 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=120 -v "$work:/job" --entrypoint sh "$image" -ec '
    hf download peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF --revision 0ac574071c3322f02b540f3547f9ef1bc48eb494 \
       --include "Maya-XS/*" --local-dir /job/checkpoint --max-workers 2
    cd /job/checkpoint/Maya-XS
    sha256sum -c /labski-weights.sha256
    python3 /job/source/tools/mm3/pack_gguf.py --model /job/checkpoint/Maya-XS --out /job/pack \
      --indexers /job/checkpoint/Maya-XS/MiniMax-M3.1-Flash-Maya-XS-maya-indexers.bin --text-only
'
