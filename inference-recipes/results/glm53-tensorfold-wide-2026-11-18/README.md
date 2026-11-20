# A wider prefill tile for GLM-5.3-Flash on GB10

On two Sparks, a 256-row CUDA tile reduced median client time to first token for a 32,768-token chat prompt from **18.08 to 17.33 seconds**. The change reuses each packed weight tile across twice as many prompt rows in one Q8 projection with output width 12,576 and inner dimension 4,096. The opt-in dispatch applies at 2,048 prompt rows and above on SM121. It preserves the accumulation order. Both sides of the comparison ran the same patched binary and the same EXL3 checkpoint. The lab collected these measurements November 17–18, 2026. [Kernel patch](wide-prompt.patch) · [Request records](requests.csv)

| Input tokens | Control, input tok/s | Wider tile, input tok/s | Gain | INT8 activation variant, input tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,596 | 1,645 | 3.1% | 1,752 |
| 32,768 | 1,813 | 1,891 | 4.3% | 2,009 |
| 65,536 | 1,805 | 1,880 | 4.2% | 2,001 |

Rates divide actual prompt tokens by client time to the first generated token. Each measurement used C1, one output token, a cold prefix cache and one excluded warm-up. Each control and wide-tile cell has three timed requests. The INT8 variant has two per cell. All timed prompts match by hash across the overlapping trials. W8A8 is a separate activation-precision experiment on the original build. The timing covers prefill and the first generated token.

**Quality.** The wider tile matched the control's per-passage negative log likelihood on four passages and all ten short answers byte for byte. The nine timed requests produced the same first-token text. The focused CUDA tests compared the operator to baseline bytes and FP64 references and detected a deliberately incorrect operator. W8A8 gained 10.8% over control at 32K. Its reported NLL rose from 0.75264 to 0.75318, and it misspelled a reversal the control answered correctly. This sample establishes a concrete quality trade-off for W8A8. Its size limits any broader accuracy claim. [Answers and likelihoods](quality.json) · [Operator test output](operator-checks.txt)

## Reproduce

On the lead Spark, point `WIDE` at this result directory and configure the upstream two-Spark recipe for your fabric and model paths. The model revision was `76c0b5173166d2795dd48860f45d8224817f894c`. The engine base is `f350cd3ea253a042e205746928271f9d8f71a82d` plus the patches in the pinned Mia recipe. The build patch pins the ARM64 CUDA base and uses the upstream checksum-verified Zig download. The original lab build supplied the same Zig archive locally. [Upstream two-Spark setup](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/80be4d6882837d8e0840827ae2aab6124fd48da3)

```sh
export WIDE="$(pwd)"   # from this result directory
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104.git glm-wide
cd glm-wide
git checkout --detach 80be4d6882837d8e0840827ae2aab6124fd48da3
git apply "$WIDE/build-on-spark.patch"
cp "$WIDE/wide-prompt.patch" patches/engine/0122-wide-prompt.patch
# Configure scripts/local.sh for your head and worker as in the upstream README.
export MODEL_REVISION=76c0b5173166d2795dd48860f45d8224817f894c
export W8A8=0 DENSE=q8 HEAD=q8 ABSORB_MMA=1 DRAFTER=dflash2
export PROMPT_CACHE_GIB=0 CACHE_FIRST=0 TF_GLM_PROMPT_WIDE=1
./start.sh
python3 "$WIDE/prefill.py" --label wide --contexts 4096,32768,65536 \
  --exact-contexts --concurrency 1 --trials 3 --prompt-seed glm53-moon-20261118
```

Repeat with `TF_GLM_PROMPT_WIDE=0 ./start.sh restart` and `--label control`. The operator checks can be run with `docker run --rm --gpus all --entrypoint /opt/tensorfold/bin/tf-glm-wide-contract "tensorfold-glm53-spark:$(scripts/prepare.sh --label)"`. Run the upstream `./stop.sh` after measurement. [TensorFold attribution](TensorFold-NOTICE.txt)
