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

## Held-out prompt lengths

Before collecting another sweep, we fixed a simple prediction for lengths between the measured points. Use the two-stage 128-row, row-first kernel below 2,816 rows and the two-stage 256-row, column-first kernel from 2,816 rows upward. The boundary is the midpoint between the last observed 128-row choice at 2,560 and the first 256-row choice at 3,072. The serving patch continues to select only its seven measured lengths.

The test uses 2,047, 2,304, 2,688, 2,816, 2,944, 3,071, 3,073, 3,328 and 3,840 rows, both output types, input seed 80411 and order seed 73519. For each length and dtype, compare the predicted layout's median with the published kernel and with the faster of the two candidates. Report every case more than 5% slower than the better candidate. This tests shape generalization on GB10. [Measurement program](measure.py)

## Held-out results

The preregistered rule beat the upstream kernel in all 18 held-out shape and output-type combinations. The speedup ranged from 1.13× to 1.46×. It selected the faster of the two candidates on all nine FP32 shapes and five of nine BF16 shapes. At 2,816, 2,944 and 3,073 rows, switching BF16 to the larger block cost 13.4%, 10.4% and 11.8% against the faster candidate. At 3,071 BF16 rows the cost was 1.5%. These results support separate dispatch choices for FP32 and BF16 near the transition. [Prediction recorded before the run](https://github.com/looskis/labski/commit/ecf8974) · [All 972 held-out timing records](held-out-measurements.jsonl.gz)

| Prompt rows | FP32 upstream, ms | FP32 predicted, ms | Excess over faster candidate | BF16 upstream, ms | BF16 predicted, ms | Excess over faster candidate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2047 | 4.332 | 2.975 | 0.0% | 4.085 | 2.892 | 0.0% |
| 2304 | 4.842 | 3.515 | 0.0% | 4.562 | 3.374 | 0.0% |
| 2688 | 5.787 | 4.486 | 0.0% | 5.339 | 3.989 | 0.0% |
| 2816 | 5.925 | 4.914 | 0.0% | 5.570 | 4.816 | 13.4% |
| 2944 | 6.352 | 5.320 | 0.0% | 5.984 | 5.259 | 10.4% |
| 3071 | 6.642 | 5.381 | 0.0% | 6.232 | 5.294 | 1.5% |
| 3073 | 6.850 | 5.845 | 0.0% | 6.461 | 5.732 | 11.8% |
| 3328 | 7.104 | 5.718 | 0.0% | 6.771 | 5.586 | 0.0% |
| 3840 | 8.264 | 6.581 | 0.0% | 7.840 | 6.547 | 0.0% |

Every candidate matched every upstream output byte on the synthetic fixtures. The independent FP64 check covered 1,024 output positions per shape and output type. Times are medians over nine randomized rounds, with four kernel launches per timing. To reproduce this sweep, run the measurement command below with `--rows 2047 2304 2688 2816 2944 3071 3073 3328 3840 --seed 80411 --order-seed 73519` and a fresh output path.

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

The lab built the two winning tile layouts into TensorFold 1.3 and verified the same executable on both GB10 ranks. The switch is opt-in and records which shape was selected. [Integrated patch, exact source hashes and build](engine/)
