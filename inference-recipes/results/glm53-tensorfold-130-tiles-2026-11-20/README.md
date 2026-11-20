# Choosing the GLM Q8 prompt tile in TensorFold 1.3.0 on GB10

On one DGX Spark, the fastest grid layout for the GLM-5.3-Flash 12,576 × 4,096 Q8 projection changed with the number of prompt rows. With 2,048 rows and FP32 output, a two-stage 128-row block traversing rows first took **3.00 ms**, compared with **4.18 ms** for the published three-stage kernel. At 4,096 rows the same candidate took **13.98 ms**. A two-stage 256-row block traversing columns first took **7.04 ms**, against **8.79 ms** for the published kernel. This measures one matrix multiply with constructed inputs. End-to-end throughput and model answers require a serving comparison. [All 756 timing records](measurements.jsonl.gz) · [Upstream implementation](https://github.com/ashhart/TensorFold/blob/81de26f649677aa72ae0034d4af6e010cd947038/zig/kernels/cuda/glm_mm.cu)

The proposed selection uses 128-row, row-first blocks at the four measured sizes through 2,560 and 256-row, column-first blocks at the three measured sizes from 3,072. Both have two pipeline stages. On BF16 output at 3,072 rows the two choices were effectively tied, 5.248 and 5.253 ms. The table applies the same choice for both output types.

| Prompt rows | FP32 published, ms | FP32 selected, ms | FP32 speedup | BF16 published, ms | BF16 selected, ms | BF16 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2.120 | 1.492 | 1.42× | 2.100 | 1.444 | 1.45× |
| 2048 | 4.180 | 2.997 | 1.39× | 3.943 | 2.933 | 1.34× |
| 2049 | 4.454 | 3.124 | 1.43× | 4.153 | 3.101 | 1.34× |
| 2560 | 5.305 | 3.883 | 1.37× | 4.963 | 3.703 | 1.34× |
| 3072 | 6.477 | 5.347 | 1.21× | 5.884 | 5.253 | 1.12× |
| 3584 | 7.500 | 6.229 | 1.20× | 7.022 | 6.104 | 1.15× |
| 4096 | 8.792 | 7.042 | 1.25× | 8.092 | 6.797 | 1.19× |

Each launch executed one K slice with the underlying TensorFold template. Column-first and row-first refer to which tile index varies along the grid’s x dimension. The control calls the published `glm_mm8_128` symbol. The other five configurations vary stage count, block height and grid order. Nine randomized rounds per shape and dtype each timed four launches per configuration on one CUDA stream. CUDA events cover the projection alone. The input and weights were reused across timed launches. Concurrent serving and dependent-launch overlap are outside this measurement. [Launcher](tiles.cu) · [Fixtures and timing](measure.py)

All six kernels produced identical bytes across every output of every tested shape and dtype, including the partial final row block at 2,049. Each test used seeded signed integer inputs stored as BF16, all 256 Q8 codes and per-group FP16 scales in exact multiples of 1/32. An independent FP64 matrix multiply checked sampled output positions, including boundary rows and the last output column. Exactness on this constructed input family leaves model-level parity open.

## Reproduce

Check out TensorFold at the pinned revision. Mia’s November 20 recipe selects the same engine commit. Its 13 patches leave the CUDA source and headers used here intact. The lab compiled the pinned sources for ARM64 and SM121a in the following PyTorch image. Set `TF` to the path for a fresh TensorFold checkout and run these commands from this results directory. [Mia recipe revision](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/d8a7439e8c5aac8645c423ccd24189690359ab69) · [TensorFold license](https://github.com/ashhart/TensorFold/blob/81de26f649677aa72ae0034d4af6e010cd947038/LICENSE)

```sh
git clone https://github.com/ashhart/TensorFold.git "$TF"
git -C "$TF" checkout --detach 81de26f649677aa72ae0034d4af6e010cd947038
image=nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
docker run --rm --network none -v "$PWD:/work" -v "$TF:/tf:ro" -w /work \
  --entrypoint nvcc "$image" -O3 -std=c++17 -arch=sm_121a -shared -Xcompiler=-fPIC \
  -I/tf/zig/kernels/cuda tiles.cu -o tiles.so
docker run --rm --gpus all --network none --memory 12g -v "$PWD:/work" -w /work \
  --entrypoint python3 "$image" measure.py --library tiles.so --out measurements.jsonl
```
