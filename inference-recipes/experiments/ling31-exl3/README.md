# Ling 3.1 Flash 1019 mixed EXL3 on one DGX Spark

This recipe targets the 560B parameter MoE with 25B active parameters per token. The community checkpoint uses 1.0 or 1.5 bits per weight for the routed experts, with higher precision for other tensors. Its 13 SafeTensors files total 100.81 GB. The quantizer published this checkpoint for a five-GPU RTX 3090 system. The Spark recipe builds exllamav3 1.6.6 for ARM64 and SM121, runs a matching TabbyAPI revision on one GB10 and uses the checkpoint's chat template. [Quantizer's model card](https://huggingface.co/auryn-macmillan/Ling-3.1-flash-1019-EXL3-1.5-1.0bpw/blob/5da2462decd55ba876f2331b462105650f6b31ac/README.md) · [exllamav3 source](https://github.com/turboderp-org/exllamav3/tree/bd1f519172406a9560bd5c90e61000691bb69a07) · [TabbyAPI source](https://github.com/theroyallab/tabbyAPI/tree/a32a82e255d9ffe1b6e805cb37527a1f53d9a62f)

With native two-token drafting it reached 22.67 generated tokens/s at one client and 42.56 aggregate for four, with 15/28 constrained-letter answers and six exact table lookups. Ordinary decoding reached 15.77 and 36.74 tok/s and scored 14/28 on the same grammar probe. [Measured timing, energy and answer formats](../../results/ling31-flash-1019-exl3-2026-11-20/)

## Build and launch

Run from this directory on the Spark with Docker, the NVIDIA Container Toolkit, Git and the Hugging Face CLI. The build uses the pinned NVIDIA PyTorch 26.07 ARM64 image, its CUDA 13.3 toolkit and four compiler jobs. TabbyAPI's Linux entry point uses uvloop on ARM64, so its runtime image explicitly installs the ARM64 package. The wheel remains in the build directory for reuse. [Build script](build.sh) · [Runtime image](Dockerfile)

```sh
work="$PWD/ling31-build"
bash build.sh "$work"
snapshot=$(hf download auryn-macmillan/Ling-3.1-flash-1019-EXL3-1.5-1.0bpw \
  --revision 5da2462decd55ba876f2331b462105650f6b31ac)
python3 serve.py "$work" "$snapshot" --mtp
```

Omit `--mtp` for the ordinary decoding control. The API binds to loopback port 8000 and serves the model as `ling31-exl3-low`. Its configuration allows four requests at once, 8,192 tokens per sequence, a 32,768-token cache with eight-bit keys and values, and 512-token prefill chunks. Its default chat template requests `thinking_option: off`. The launcher builds a local model view containing the weights and tokenizer and mounts their source files read-only. [Runtime settings](config.yml) · [Launcher](serve.py)
