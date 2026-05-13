import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    recv = torch.empty((1024, 1024), device="cuda", dtype=torch.bfloat16)
    if rank == 0:
        chunks = [
            torch.full_like(recv, fill_value=float(i + 1), device="cuda")
            for i in range(world_size)
        ]
    else:
        chunks = None

    dist.scatter(recv, chunks, src=0)
    torch.cuda.synchronize()
    ok = bool(torch.all(recv == float(rank + 1)).item())
    print(
        f"rank={rank} mean={recv.float().mean().item():.3f} "
        f"min={recv.float().min().item():.3f} max={recv.float().max().item():.3f} ok={ok}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
