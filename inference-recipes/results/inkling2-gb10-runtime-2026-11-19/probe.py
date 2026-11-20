"""Exercise the architecture selection used by the stock Inkling 2 expert GEMM."""
import json
import torch
from flashinfer.jit.core import current_compilation_context

print(json.dumps({"gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
                  "torch": torch.__version__}), flush=True)
# vllm/models/inkling2/ops/trtllm_bmm.py calls this during _module().
print(current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10]))
