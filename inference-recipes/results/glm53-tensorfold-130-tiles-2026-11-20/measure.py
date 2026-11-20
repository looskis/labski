"""Time Q8 prompt projectors from the same TensorFold source on one GB10."""
import argparse
import ctypes
import json
from pathlib import Path
import random

import torch

VARIANTS = ["upstream128s3", "128s3_rows_first", "128s2", "128s2_rows_first", "256s2", "256s2_rows_first"]
N, K = 12576, 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 2049, 2560, 3072, 3584, 4096])
    parser.add_argument("--seed", type=int, default=20261120, help="seed for the tensor values")
    parser.add_argument("--order-seed", type=int, default=31270, help="seed for case and launch order")
    args = parser.parse_args()
    if len(set(args.rows)) != len(args.rows) or min(args.rows) < 1:
        parser.error("--rows must contain distinct positive counts")
    lib = ctypes.CDLL(str(args.library.resolve()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    lib.tile_weights.argtypes = [ptr, ptr, ptr, ptr, integer, integer, ptr]
    lib.project.argtypes = [integer, ptr, ptr, ptr, ptr, integer, integer, integer, integer, ptr]
    def check(code):
        if code != 0:
            raise RuntimeError(f"CUDA runtime error {code}")
    check(lib.configure())
    torch.manual_seed(args.seed)
    stream = torch.cuda.current_stream().cuda_stream
    rnd = random.Random(args.order_seed)
    codes = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8)
    scales = torch.randint(1, 9, (N, K // 64), device="cuda").half() / 32
    packed, tiled_scales = torch.empty_like(codes), torch.empty_like(scales)
    check(lib.tile_weights(codes.data_ptr(), scales.data_ptr(), packed.data_ptr(),
                           tiled_scales.data_ptr(), N, K, stream))
    with args.out.open("x") as out:
        cases = [(m, dtype) for m in args.rows
                 for dtype in (torch.float32, torch.bfloat16)]
        rnd.shuffle(cases)
        for m, dtype in cases:
            # Small exactly representable values give an independent exact FP64 check even after summation.
            x = torch.randint(-8, 9, (m, K), device="cuda").bfloat16()
            baseline = torch.empty((m, N), dtype=dtype, device="cuda")
            test = torch.empty_like(baseline)
            def launch(i, dest):
                check(lib.project(i, x.data_ptr(), packed.data_ptr(), tiled_scales.data_ptr(),
                                  dest.data_ptr(), int(dtype == torch.float32), m, N, K, stream))
            launch(0, baseline)
            rr = sorted({row for row in (0, 127, 128, 255, 256, m - 1, *rnd.sample(range(m), min(m, 26)))
                         if row < m})
            cc = sorted({0, 63, 64, N - 1, *rnd.sample(range(N), 28)})
            weights = (codes[cc].double() - 128) * scales[cc].double().repeat_interleave(64, dim=1)
            reference = x[rr].double() @ weights.T
            if not torch.equal(baseline[rr][:, cc], reference.to(dtype)):
                raise ValueError("published Q8 kernel disagrees with exact FP64 reference")
            for i in range(len(VARIANTS)):
                test.fill_(float("nan"))
                launch(i, test)
                if not torch.equal(baseline.view(torch.uint8), test.view(torch.uint8)):
                    raise ValueError(f"output bits differ for {m} rows and {VARIANTS[i]}")
            torch.cuda.synchronize()
            for repeat in range(9):
                order = list(range(len(VARIANTS)))
                rnd.shuffle(order)
                for i in order:
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(4):
                        launch(i, test)
                    stop.record()
                    stop.synchronize()
                    row = dict(rows=m, columns=N, width=K, variant=VARIANTS[i], repeat=repeat + 1,
                               ms_per_launch=start.elapsed_time(stop) / 4, reference_cells=len(rr)*len(cc),
                               identical_outputs=m*N, output_dtype=str(dtype).split(".")[-1])
                    out.write(json.dumps(row) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()
