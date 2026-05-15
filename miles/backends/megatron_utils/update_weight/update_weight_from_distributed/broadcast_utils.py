import socket
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.utils.distributed_utils import init_process_group


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
    engine_tp_rank_filters: Sequence[Sequence[int]] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    if len(engine_gpu_counts) != len(rollout_engines):
        raise ValueError(
            f"engine_gpu_counts must match rollout_engines, got {len(engine_gpu_counts)} and "
            f"{len(rollout_engines)}."
        )
    if engine_tp_rank_filters is not None:
        if len(engine_tp_rank_filters) != len(rollout_engines):
            raise ValueError(
                "engine_tp_rank_filters must match rollout_engines, got "
                f"{len(engine_tp_rank_filters)} and {len(rollout_engines)}."
            )
        effective_engine_gpu_counts = []
        for engine_index, (tp_ranks, gpu_count) in enumerate(
            zip(engine_tp_rank_filters, engine_gpu_counts, strict=True)
        ):
            if not tp_ranks:
                raise ValueError(f"engine_tp_rank_filters[{engine_index}] cannot be empty.")
            if len(set(tp_ranks)) != len(tp_ranks):
                raise ValueError(
                    f"engine_tp_rank_filters[{engine_index}] contains duplicate ranks: {tp_ranks}."
                )
            invalid_ranks = [rank for rank in tp_ranks if rank < 0 or rank >= gpu_count]
            if invalid_ranks:
                raise ValueError(
                    f"engine_tp_rank_filters[{engine_index}] has ranks outside [0, {gpu_count}): "
                    f"{invalid_ranks}."
                )
            effective_engine_gpu_counts.append(len(tp_ranks))
    else:
        effective_engine_gpu_counts = list(engine_gpu_counts)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(effective_engine_gpu_counts) + 1

    refs = []
    rank_cursor = 1
    for i, engine in enumerate(rollout_engines):
        tp_ranks = None if engine_tp_rank_filters is None else list(engine_tp_rank_filters[i])
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                backend="nccl",
                tp_ranks=tp_ranks,
            )
        )
        rank_cursor += effective_engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int | None,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 -> engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version) if weight_version is not None else None,
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def update_weights_from_distributed_send_recv(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int | None,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), send tensors with NCCL send/recv (rank 0 -> engines).
    """
    names = [name for name, _ in converted_named_tensors]
    dtypes = [param.dtype for _, param in converted_named_tensors]
    shapes = [param.shape for _, param in converted_named_tensors]
    refs = [
        engine.update_weights_from_distributed.remote(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            weight_version=str(weight_version) if weight_version is not None else None,
            transfer_mode="send_recv_tp0",
        )
        for engine in rollout_engines
    ]

    group_world_size = dist.get_world_size(group)
    if group_world_size != 2:
        raise ValueError(
            "send/recv distributed weight update expects a trainer-to-relay-TP0 group "
            f"with world_size=2, got {group_world_size}."
        )
    ops = [
        dist.P2POp(
            dist.isend,
            param.data,
            group=group,
            group_peer=1,
        )
        for _, param in converted_named_tensors
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()

    return refs
