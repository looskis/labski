"""Check a CUDA collective on every Spark rank before a tensor-parallel launch."""
from datetime import timedelta
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def main():
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    expected = world * (world + 1) // 2
    try:
        for size in (1024, 65536, 1048576):
            buf = torch.empty(size // 4, dtype=torch.float32, device="cuda")
            samples = []
            for trial in range(8):
                buf.fill_(rank + 1)
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_reduce(buf)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if not bool(torch.all(buf == expected)):
                    raise RuntimeError(f"all-reduce failed on rank {rank} at {size} bytes")
                if trial:
                    samples.append(elapsed * 1000)
            print(json.dumps(dict(rank=rank, world=world, bytes=size, checks=8,
                                  transport=os.getenv("NCCL_NET", "automatic"),
                                  median_ms=statistics.median(samples), times_ms=samples)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
