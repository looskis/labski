#!/usr/bin/env bash
set -eu

: "${RANK:?set the rank to 0 or 1}"
: "${HEAD_IP:?set the ConnectX IPv4 address of rank 0}"
: "${IFACE:?set the local ConnectX data interface}"
: "${HCAS:?set the local RDMA HCA list}"
: "${RUNTIME_IMAGE:?set the locally built TensorFold runtime image}"
: "${TF_NATIVE_BINARY:?set the absolute path of the patched executable}"
: "${MODEL_DIR:?set the absolute path of the pinned GLM checkpoint}"
: "${DRAFT_DIR:?set the absolute path of the pinned DFlash2 checkpoint}"
: "${STATE_DIR:?set a writable absolute path for TensorFold state}"
: "${PROMPT_TILES:?set 0 for upstream dispatch or 1 for the lab dispatch}"

case "$RANK/$PROMPT_TILES" in
    0/0|0/1|1/0|1/1) ;;
    *) echo 'RANK and PROMPT_TILES must each be 0 or 1' >&2; exit 2 ;;
esac
test -x "$TF_NATIVE_BINARY"
test -r "$MODEL_DIR/chat_template.jinja"
test -r "$DRAFT_DIR/model.safetensors"
mkdir -p "$STATE_DIR/home"
native=/opt/tensorfold/native/bin/tensorfold-native

exec docker run -d --name labski-tf130-glm --init --restart=no \
    --gpus all --ipc=host --network=host --shm-size=16g --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband \
    -v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/dflash2:ro" \
    -v "$STATE_DIR:/state" -v "$TF_NATIVE_BINARY:$native:ro" \
    -e HOME=/state/home -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 \
    -e TENSORFOLD_NO_LIVE=1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e TENSORFOLD_MEMORY_RESERVE_GIB=9.5 -e TENSORFOLD_PREFILL_SHARE=0.68 \
    -e TF_GLM_DENSE_BITS=8 -e TF_GLM_HEAD_BITS=8 -e TF_GLM_CLEAR_THINKING=1 \
    -e TF_KV_PAGES=on -e TF_GLM_CE_EXCHANGE=0 -e TF_GLM_ABSORB_MMA=1 \
    -e TF_GLM_W8A8=0 -e TF_CUDA_DEVICE=0 -e TF_GLM_WORLD=2 \
    -e "TF_GLM_RANK=$RANK" -e TF_GLM_HOST_CACHE_GIB=0 -e "TF_BIN=$native" \
    -e "NCCL_COMM_ID=$HEAD_IP:29683" -e "NCCL_SOCKET_IFNAME=$IFACE" \
    -e "NCCL_IB_HCA=$HCAS" -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \
    -e TF_ROCE=1 -e TF_ROCE_MAX_KB=4096 -e "TF_ROCE_HCA=$HCAS" \
    -e TF_GLM_PDL=1 -e TF_GLM_CHUNK_SOLO=4096 \
    -e "TF_GLM_Q8_PROMPT_TILES=$PROMPT_TILES" -e TF_GLM_Q8_PROMPT_TRACE=1 \
    --entrypoint "$native" "$RUNTIME_IMAGE" serve /model \
    --name GLM-5.3-Flash --host 127.0.0.1 --port 18220 --context 65536 \
    --parallel 1 --max-tokens 1024 --kv-cache fp8 --no-update-check --no-thinking \
    --chat-template /model/chat_template.jinja --drafter /dflash2 --drafter-bits 4 \
    --prompt-cache-gib 0
