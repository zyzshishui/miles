import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

from .mixin import DistBucketedWeightUpdateMixin
from .broadcast_utils import (
    connect_rollout_relay_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed_send_recv,
)

def _coordinator_actor_name() -> str:
    job_id = ray.get_runtime_context().get_job_id()
    return f"miles-sendrecv-broadcast-coordinator-{job_id}"


@dataclass
class _RelayUpdateRecord:
    pp_rank: int
    bucket_id: int
    update_refs: list[ray.ObjectRef]


@ray.remote(num_cpus=0)
def _run_relay_fanout_and_resume(
    rollout_engines: Sequence[ActorHandle],
    rollout_engine_lock: ActorHandle,
    relay_engine: ActorHandle,
    peer_engines: Sequence[ActorHandle],
    relay_gpu_count: int,
    next_fanout_port: int,
    weight_version: int,
    relay_update_refs: Sequence[ray.ObjectRef],
) -> dict[str, int]:
    _ensure_success(ray.get(list(relay_update_refs)), "receive relay weights from NCCL")
    _post_process_weights(
        rollout_engines=[relay_engine],
        restore_weights_before_load=False,
        post_process_quantization=False,
        post_load_weights=False,
        action="flush relay pending weights",
    )
    next_fanout_port = _fanout_relay_weights_to_peer_instances(
        rollout_engine_lock=rollout_engine_lock,
        relay_engine=relay_engine,
        peer_engines=peer_engines,
        relay_gpu_count=relay_gpu_count,
        next_fanout_port=next_fanout_port,
        weight_version=weight_version,
    )
    _post_process_weights(
        rollout_engines=rollout_engines,
        restore_weights_before_load=False,
        post_process_quantization=True,
        post_load_weights=True,
        action="post-process rollout weights",
    )
    ray.get(
        [
            engine.update_weight_version.remote(weight_version=str(weight_version))
            for engine in rollout_engines
        ]
    )
    ray.get([engine.continue_generation.remote() for engine in rollout_engines])
    return {"next_fanout_port": next_fanout_port}


@ray.remote(num_cpus=0)
class _SendRecvBroadcastCoordinator:
    def __init__(self) -> None:
        self._records_by_version: dict[int, list[_RelayUpdateRecord]] = {}
        self._done_sources_by_version: dict[int, set[int]] = {}
        self._pending_fanout_ref: ray.ObjectRef | None = None
        self._terminal_error: str | None = None

    def configure(
        self,
        *,
        expected_sources: int,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        relay_engine: ActorHandle,
        peer_engines: Sequence[ActorHandle],
        relay_gpu_count: int,
        next_fanout_port: int,
    ) -> None:
        if self._terminal_error is not None:
            raise RuntimeError(
                "Cannot reconfigure sendrecv_broadcast after a failed background update: "
                f"{self._terminal_error}"
            )
        if self._pending_fanout_ref is not None:
            raise RuntimeError("Cannot reconfigure sendrecv_broadcast while fanout is pending.")
        if self._records_by_version or self._done_sources_by_version:
            raise RuntimeError("Cannot reconfigure sendrecv_broadcast while update records are pending.")

        self._expected_sources = expected_sources
        self._rollout_engines = rollout_engines
        self._rollout_engine_lock = rollout_engine_lock
        self._relay_engine = relay_engine
        self._peer_engines = peer_engines
        self._relay_gpu_count = relay_gpu_count
        self._next_fanout_port = next_fanout_port

    def add_relay_update_refs(
        self,
        *,
        weight_version: int,
        pp_rank: int,
        bucket_id: int,
        update_refs: Sequence[ray.ObjectRef],
    ) -> None:
        self._records_by_version.setdefault(weight_version, []).append(
            _RelayUpdateRecord(
                pp_rank=pp_rank,
                bucket_id=bucket_id,
                update_refs=list(update_refs),
            )
        )

    def mark_source_done(self, *, weight_version: int, pp_rank: int) -> dict[str, int | bool]:
        done_sources = self._done_sources_by_version.setdefault(weight_version, set())
        done_sources.add(pp_rank)
        if len(done_sources) != self._expected_sources:
            return {"scheduled": False, "transfers": 0}

        records = sorted(
            self._records_by_version.pop(weight_version, []),
            key=lambda record: (record.pp_rank, record.bucket_id),
        )
        self._done_sources_by_version.pop(weight_version, None)
        relay_update_refs = [
            update_ref
            for record in records
            for update_ref in record.update_refs
        ]
        self._pending_fanout_ref = _run_relay_fanout_and_resume.remote(
            rollout_engines=self._rollout_engines,
            rollout_engine_lock=self._rollout_engine_lock,
            relay_engine=self._relay_engine,
            peer_engines=self._peer_engines,
            relay_gpu_count=self._relay_gpu_count,
            next_fanout_port=self._next_fanout_port,
            weight_version=weight_version,
            relay_update_refs=relay_update_refs,
        )
        return {"scheduled": True, "transfers": len(relay_update_refs)}

    def wait_pending_fanout(self) -> dict[str, int]:
        if self._terminal_error is not None:
            raise RuntimeError(
                "sendrecv_broadcast background update failed: "
                f"{self._terminal_error}"
            )
        if self._pending_fanout_ref is None:
            return {"next_fanout_port": self._next_fanout_port}
        try:
            result = ray.get(self._pending_fanout_ref)
        except Exception as exc:
            self._pending_fanout_ref = None
            self._terminal_error = repr(exc)
            raise
        self._next_fanout_port = result["next_fanout_port"]
        self._pending_fanout_ref = None
        return result


def _fanout_relay_weights_to_peer_instances(
    *,
    rollout_engine_lock: ActorHandle,
    relay_engine: ActorHandle,
    peer_engines: Sequence[ActorHandle],
    relay_gpu_count: int,
    next_fanout_port: int,
    weight_version: int,
) -> int:
    if not peer_engines:
        return next_fanout_port

    _acquire_rollout_engine_lock(rollout_engine_lock)
    try:
        master_address, first_port = ray.get(
            relay_engine._get_current_node_ip_and_free_port.remote(
                start_port=next_fanout_port,
                consecutive=relay_gpu_count,
            )
        )
        next_fanout_port = first_port + relay_gpu_count + 1
        ports = ",".join(str(first_port + rank) for rank in range(relay_gpu_count))
        group_name = f"miles-sendrecv-broadcast-fanout-v{weight_version}"
        world_size = len(peer_engines) + 1

        init_refs = [
            relay_engine.init_weights_send_group_for_remote_instance.remote(
                master_address=master_address,
                ports=ports,
                group_rank=0,
                world_size=world_size,
                group_name=group_name,
                backend="nccl",
            )
        ]
        init_refs.extend(
            peer_engine.init_weights_send_group_for_remote_instance.remote(
                master_address=master_address,
                ports=ports,
                group_rank=peer_rank,
                world_size=world_size,
                group_name=group_name,
                backend="nccl",
            )
            for peer_rank, peer_engine in enumerate(peer_engines, start=1)
        )
        _ensure_success(ray.get(init_refs), f"initialize relay fanout group {group_name}")

        send_refs = [
            relay_engine.send_recv_weights_to_remote_instance.remote(
                master_address=master_address,
                ports=ports,
                group_name=group_name,
            )
        ]
        send_refs.extend(
            peer_engine.send_recv_weights_to_remote_instance.remote(
                master_address=master_address,
                ports=ports,
                group_name=group_name,
            )
            for peer_engine in peer_engines
        )
        _ensure_success(
            ray.get(send_refs),
            f"send/recv relay weights through {group_name}",
        )
    finally:
        ray.get(rollout_engine_lock.release.remote())

    return next_fanout_port


def _acquire_rollout_engine_lock(rollout_engine_lock: ActorHandle) -> None:
    while not ray.get(rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)


def _ensure_success(responses: Sequence[dict | None], action: str) -> None:
    for response in responses:
        if response is not None and not response.get("success", True):
            raise RuntimeError(f"Failed to {action}: {response.get('message', response)}")


def _post_process_weights(
    *,
    rollout_engines: Sequence[ActorHandle],
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    post_load_weights: bool,
    action: str,
) -> None:
    _ensure_success(
        ray.get(
            [
                engine.post_process_weights.remote(
                    restore_weights_before_load=restore_weights_before_load,
                    post_process_quantization=post_process_quantization,
                    post_load_weights=post_load_weights,
                )
                for engine in rollout_engines
            ]
        ),
        action,
    )


class UpdateWeightSendRecvBroadcast(DistBucketedWeightUpdateMixin):
    """
    Send each training PP stage's canonical buckets to one rollout relay, then
    let the relay synchronize the loaded model to peer rollout instances.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._next_fanout_port = 20000
        self._relay_update_group = None
        self._bucket_id = 0
        self._pending_coordinator_submit_refs: list[ray.ObjectRef] = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)

        self._relay_engine = rollout_engines[0]
        self._peer_engines = list(rollout_engines[1:])
        self._relay_gpu_count = engine_gpu_counts[0]
        self._pp_rank = get_parallel_state().pp.rank
        self._connect_coordinator()

        if self._is_source:
            self._group_name = f"miles-sendrecv-broadcast-train-pp_{self._pp_rank}"
            if self._relay_update_group is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self._relay_update_group,
                    [self._relay_engine],
                )
            self._relay_update_group = connect_rollout_relay_from_distributed(
                self._group_name,
                self._relay_engine,
            )

    def _connect_coordinator(self) -> None:
        coordinator_name = _coordinator_actor_name()
        if dist.get_rank() == 0:
            try:
                self._coordinator = ray.get_actor(coordinator_name)
            except ValueError:
                self._coordinator = _SendRecvBroadcastCoordinator.options(
                    name=coordinator_name
                ).remote()
            ray.get(
                self._coordinator.configure.remote(
                    expected_sources=get_parallel_state().pp.size,
                    rollout_engines=self.rollout_engines,
                    rollout_engine_lock=self.rollout_engine_lock,
                    relay_engine=self._relay_engine,
                    peer_engines=self._peer_engines,
                    relay_gpu_count=self._relay_gpu_count,
                    next_fanout_port=self._next_fanout_port,
                )
            )
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() != 0:
            self._coordinator = ray.get_actor(coordinator_name)


    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        if not self._is_source or not converted_named_tensors:
            return

        bucket_id = self._bucket_id
        self._bucket_id += 1
        self._acquire_rollout_engine_lock()
        try:
            update_ref = update_weights_from_distributed_send_recv(
                self._group_name,
                self._relay_update_group,
                None,
                self._relay_engine,
                converted_named_tensors,
            )
            _ensure_success(
                [ray.get(update_ref)],
                "receive relay weights from NCCL",
            )
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        self._pending_coordinator_submit_refs.append(
            self._coordinator.add_relay_update_refs.remote(
                weight_version=self.weight_version,
                pp_rank=self._pp_rank,
                bucket_id=bucket_id,
                update_refs=[update_ref],
            )
        )
        converted_named_tensors.clear()

        if pbar:
            pbar.update(1)

    def _finalize_and_resume_engines(self, post_load_weights: bool = False) -> None:
        if self._is_source:
            ray.get(self._pending_coordinator_submit_refs)
            self._pending_coordinator_submit_refs = []
            ray.get(
                self._coordinator.mark_source_done.remote(
                    weight_version=self.weight_version,
                    pp_rank=self._pp_rank,
                )
            )

    def get_pending_update_coordinator(self) -> ActorHandle | None:
        return getattr(self, "_coordinator", None)

    def wait_pending_fanout(self) -> None:
        gloo_group = get_gloo_group()
        error_message = None
        if dist.get_rank() == 0 and hasattr(self, "_coordinator"):
            try:
                result = ray.get(self._coordinator.wait_pending_fanout.remote())
                self._next_fanout_port = result["next_fanout_port"]
            except Exception as exc:
                error_message = repr(exc)
        errors = [None] * dist.get_world_size(group=gloo_group)
        dist.all_gather_object(errors, error_message, group=gloo_group)
        if any(errors):
            raise RuntimeError(
                "sendrecv_broadcast background update failed: "
                f"{next(error for error in errors if error)}"
            )

    def _acquire_rollout_engine_lock(self) -> None:
        _acquire_rollout_engine_lock(self.rollout_engine_lock)
