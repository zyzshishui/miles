from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    broadcast_utils,
    mixin,
    sendrecv_broadcast,
)
UpdateWeightSendRecvBroadcast = sendrecv_broadcast.UpdateWeightSendRecvBroadcast


class FakeRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class FakeEngine:
    def __init__(self, host="10.0.0.1", first_port=23456):
        self._get_current_node_ip_and_free_port = FakeRemoteMethod((host, first_port))
        self.init_weights_send_group_for_remote_instance = FakeRemoteMethod(
            {"success": True, "message": "ok"}
        )
        self.init_weights_update_group = FakeRemoteMethod({"success": True, "message": "ok"})
        self.destroy_weights_update_group = FakeRemoteMethod({"success": True, "message": "ok"})
        self.update_weights_from_distributed = FakeRemoteMethod(
            {"success": True, "message": "ok"}
        )
        self.send_weights_to_remote_instance = FakeRemoteMethod({"success": True, "message": "ok"})
        self.send_recv_weights_to_remote_instance = FakeRemoteMethod(
            {"success": True, "message": "ok"}
        )


class FakeLock:
    def __init__(self):
        self.acquire = FakeRemoteMethod(True)
        self.release = FakeRemoteMethod(True)


class FakeCoordinator:
    def __init__(self):
        self.configure = FakeRemoteMethod(None)
        self.add_relay_update_refs = FakeRemoteMethod("coordinator-submit-ref")
        self.mark_source_done = FakeRemoteMethod({"scheduled": True, "transfers": 0})
        self.wait_pending_fanout = FakeRemoteMethod({"next_fanout_port": 20003})


def _new_updater():
    updater = UpdateWeightSendRecvBroadcast.__new__(UpdateWeightSendRecvBroadcast)
    updater.args = SimpleNamespace(sglang_pp_size=1)
    updater._pp_rank = 0
    updater._pending_coordinator_submit_refs = []
    return updater


def _source_parallel_state():
    return SimpleNamespace(
        intra_dp_cp=SimpleNamespace(rank=0),
        tp=SimpleNamespace(rank=0),
    )


def test_validate_relay_fanout_requires_rollout_engine():
    updater = _new_updater()

    with pytest.raises(ValueError, match="at least one rollout engine"):
        updater._validate_relay_fanout_layout([], [])


def test_validate_relay_fanout_requires_homogeneous_engine_gpu_counts():
    updater = _new_updater()

    with pytest.raises(NotImplementedError, match="homogeneous rollout engine GPU counts"):
        updater._validate_relay_fanout_layout([object(), object()], [4, 2])


def test_fanout_initializes_one_ranked_send_group_and_uses_p2p_sendrecv(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    updater = _new_updater()
    relay = FakeEngine()
    peer0 = FakeEngine()
    peer1 = FakeEngine()
    updater._relay_engine = relay
    updater._peer_engines = [peer0, peer1]
    updater._relay_gpu_count = 2
    updater._next_fanout_port = 20000
    updater.weight_version = 7
    updater.rollout_engine_lock = FakeLock()

    updater._next_fanout_port = sendrecv_broadcast._fanout_relay_weights_to_peer_instances(
        rollout_engine_lock=updater.rollout_engine_lock,
        relay_engine=updater._relay_engine,
        peer_engines=updater._peer_engines,
        relay_gpu_count=updater._relay_gpu_count,
        next_fanout_port=updater._next_fanout_port,
        weight_version=updater.weight_version,
    )

    _, relay_init_kwargs = relay.init_weights_send_group_for_remote_instance.calls[0]
    _, peer0_init_kwargs = peer0.init_weights_send_group_for_remote_instance.calls[0]
    _, peer1_init_kwargs = peer1.init_weights_send_group_for_remote_instance.calls[0]
    assert relay_init_kwargs["group_rank"] == 0
    assert peer0_init_kwargs["group_rank"] == 1
    assert peer1_init_kwargs["group_rank"] == 2
    assert (
        relay_init_kwargs["world_size"]
        == peer0_init_kwargs["world_size"]
        == peer1_init_kwargs["world_size"]
        == 3
    )
    assert relay_init_kwargs["ports"] == peer0_init_kwargs["ports"] == peer1_init_kwargs["ports"] == "23456,23457"
    assert relay_init_kwargs["group_name"] == peer0_init_kwargs["group_name"] == peer1_init_kwargs["group_name"]
    assert len(relay._get_current_node_ip_and_free_port.calls) == 1
    assert len(relay.send_weights_to_remote_instance.calls) == 0
    assert len(peer0.send_weights_to_remote_instance.calls) == 0
    assert len(peer1.send_weights_to_remote_instance.calls) == 0
    assert len(relay.send_recv_weights_to_remote_instance.calls) == 1
    assert len(peer0.send_recv_weights_to_remote_instance.calls) == 1
    assert len(peer1.send_recv_weights_to_remote_instance.calls) == 1
    assert len(updater.rollout_engine_lock.release.calls) == 1


def test_update_bucket_uses_nccl_send_recv_to_send_to_relay(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(mixin, "get_parallel_state", _source_parallel_state)
    helper_calls = []

    def fake_update_weights_from_distributed_send_recv(
        group_name,
        group,
        weight_version,
        rollout_engines,
        converted_named_tensors,
    ):
        helper_calls.append(
            (
                group_name,
                group,
                weight_version,
                rollout_engines,
                list(converted_named_tensors),
            )
        )
        return ["relay-update-ref"]

    monkeypatch.setattr(
        sendrecv_broadcast,
        "update_weights_from_distributed_send_recv",
        fake_update_weights_from_distributed_send_recv,
    )

    updater = _new_updater()
    relay = FakeEngine()
    updater._relay_engine = relay
    updater._relay_update_group = "relay-nccl-group"
    updater.rollout_engine_lock = FakeLock()
    updater._group_name = "miles-sendrecv-broadcast-train-pp_0"
    updater._bucket_id = 0
    updater.weight_version = 3
    updater._coordinator = FakeCoordinator()

    tensor = torch.ones(2, dtype=torch.float32)
    tensors = [("model.layers.0.weight", tensor)]
    updater._update_weight_implementation(tensors)

    assert tensors == []
    assert len(helper_calls) == 1
    group_name, group, weight_version, rollout_engines, sent_tensors = helper_calls[0]
    assert group_name == "miles-sendrecv-broadcast-train-pp_0"
    assert group == "relay-nccl-group"
    assert weight_version is None
    assert rollout_engines == [relay]
    assert len(sent_tensors) == 1
    assert sent_tensors[0][0] == "model.layers.0.weight"
    assert sent_tensors[0][1] is tensor
    assert updater._pending_coordinator_submit_refs == ["coordinator-submit-ref"]
    _, submit_kwargs = updater._coordinator.add_relay_update_refs.calls[0]
    assert submit_kwargs == {
        "weight_version": 3,
        "pp_rank": 0,
        "bucket_id": 0,
        "update_refs": ["relay-update-ref"],
    }
    assert len(updater.rollout_engine_lock.acquire.calls) == 1
    assert len(updater.rollout_engine_lock.release.calls) == 1


def test_connect_rollout_engines_can_limit_update_group_to_relay_tp0(monkeypatch):
    created_groups = []

    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(
        broadcast_utils.ray._private.services,
        "get_node_ip_address",
        lambda: "127.0.0.1",
    )
    monkeypatch.setattr(
        broadcast_utils,
        "init_process_group",
        lambda **kwargs: created_groups.append(kwargs) or "trainer-update-group",
    )

    engine = FakeEngine()
    group = broadcast_utils.connect_rollout_engines_from_distributed(
        SimpleNamespace(rollout_num_gpus_per_engine=2),
        "group",
        [engine],
        engine_gpu_counts=[2],
        engine_tp_rank_filters=[[0]],
    )

    assert group == "trainer-update-group"
    assert len(engine.init_weights_update_group.calls) == 1
    args, kwargs = engine.init_weights_update_group.calls[0]
    assert args[2:] == (1, 2, "group")
    assert kwargs == {"backend": "nccl", "tp_ranks": [0]}
    assert created_groups[0]["world_size"] == 2
    assert created_groups[0]["rank"] == 0
    assert created_groups[0]["group_name"] == "group"


def test_update_weights_from_distributed_send_recv_uses_p2p_ops(monkeypatch):
    created_ops = []
    waited_works = []

    class FakeWork:
        def __init__(self, op):
            self.op = op

        def wait(self):
            waited_works.append(self.op)

    def fake_p2p_op(op, tensor, *, group, group_peer):
        created_ops.append((op, tensor, group, group_peer))
        return created_ops[-1]

    def fake_batch_isend_irecv(ops):
        return [FakeWork(op) for op in ops]

    monkeypatch.setattr(broadcast_utils.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(broadcast_utils.dist, "P2POp", fake_p2p_op)
    monkeypatch.setattr(broadcast_utils.dist, "batch_isend_irecv", fake_batch_isend_irecv)

    relay = FakeEngine()
    tensor0 = torch.ones(2, dtype=torch.float32)
    tensor1 = torch.ones(3, dtype=torch.float16)

    refs = broadcast_utils.update_weights_from_distributed_send_recv(
        "group",
        "process-group",
        None,
        [relay],
        [("weight0", tensor0), ("weight1", tensor1)],
    )

    assert refs == [{"success": True, "message": "ok"}]
    assert len(relay.update_weights_from_distributed.calls) == 1
    _, update_kwargs = relay.update_weights_from_distributed.calls[0]
    assert update_kwargs["names"] == ["weight0", "weight1"]
    assert update_kwargs["shapes"] == [tensor0.shape, tensor1.shape]
    assert update_kwargs["group_name"] == "group"
    assert update_kwargs["weight_version"] is None
    assert update_kwargs["transfer_mode"] == "send_recv_tp0"
    assert [(op, group, peer) for op, _, group, peer in created_ops] == [
        (broadcast_utils.dist.isend, "process-group", 1),
        (broadcast_utils.dist.isend, "process-group", 1),
    ]
    assert [tensor.data_ptr() for _, tensor, _, _ in created_ops] == [
        tensor0.data_ptr(),
        tensor1.data_ptr(),
    ]
    assert len(waited_works) == len(created_ops)
    assert all(waited is created for waited, created in zip(waited_works, created_ops, strict=True))


def test_update_bucket_releases_lock_when_nccl_send_fails(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(mixin, "get_parallel_state", _source_parallel_state)

    def fail_update_weights_from_distributed_send_recv(*args, **kwargs):
        raise RuntimeError("nccl failed")

    monkeypatch.setattr(
        sendrecv_broadcast,
        "update_weights_from_distributed_send_recv",
        fail_update_weights_from_distributed_send_recv,
    )

    updater = _new_updater()
    updater._relay_engine = FakeEngine()
    updater._relay_update_group = "relay-nccl-group"
    updater.rollout_engine_lock = FakeLock()
    updater._group_name = "miles-sendrecv-broadcast-train-pp_0"
    updater._bucket_id = 0

    with pytest.raises(RuntimeError, match="nccl failed"):
        updater._update_weight_implementation([("weight", torch.ones(1))])

    assert len(updater.rollout_engine_lock.release.calls) == 1


def test_connect_coordinator_rank_zero_keeps_created_actor_handle_and_configures_it(monkeypatch):
    class FakeCoordinatorActor:
        options_kwargs = None

        @classmethod
        def options(cls, **kwargs):
            cls.options_kwargs = kwargs
            return cls()

        def remote(self):
            return FakeCoordinator()

    barriers = []
    monkeypatch.setattr(sendrecv_broadcast, "_coordinator_actor_name", lambda: "coordinator")
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sendrecv_broadcast, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sendrecv_broadcast.dist, "barrier", lambda group: barriers.append(group))
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(
        sendrecv_broadcast.ray,
        "get_actor",
        lambda name: (_ for _ in ()).throw(ValueError(name)),
    )
    monkeypatch.setattr(sendrecv_broadcast, "_SendRecvBroadcastCoordinator", FakeCoordinatorActor)
    monkeypatch.setattr(
        sendrecv_broadcast,
        "get_parallel_state",
        lambda: SimpleNamespace(pp=SimpleNamespace(size=2)),
    )

    updater = _new_updater()
    updater.rollout_engines = [FakeEngine(), FakeEngine()]
    updater.rollout_engine_lock = FakeLock()
    updater._relay_engine = updater.rollout_engines[0]
    updater._peer_engines = [updater.rollout_engines[1]]
    updater._relay_gpu_count = 2
    updater._next_fanout_port = 20000

    updater._connect_coordinator()

    assert isinstance(updater._coordinator, FakeCoordinator)
    assert len(updater._coordinator.configure.calls) == 1
    _, configure_kwargs = updater._coordinator.configure.calls[0]
    assert configure_kwargs["expected_sources"] == 2
    assert configure_kwargs["rollout_engines"] == updater.rollout_engines
    assert configure_kwargs["rollout_engine_lock"] == updater.rollout_engine_lock
    assert configure_kwargs["relay_engine"] == updater._relay_engine
    assert configure_kwargs["peer_engines"] == updater._peer_engines
    assert configure_kwargs["relay_gpu_count"] == 2
    assert configure_kwargs["next_fanout_port"] == 20000
    assert FakeCoordinatorActor.options_kwargs == {"name": "coordinator"}
    assert barriers == ["gloo"]


def test_connect_coordinator_refreshes_existing_actor_handles(monkeypatch):
    existing_coordinator = FakeCoordinator()
    barriers = []
    monkeypatch.setattr(sendrecv_broadcast, "_coordinator_actor_name", lambda: "coordinator")
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sendrecv_broadcast, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sendrecv_broadcast.dist, "barrier", lambda group: barriers.append(group))
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(sendrecv_broadcast.ray, "get_actor", lambda name: existing_coordinator)
    monkeypatch.setattr(
        sendrecv_broadcast,
        "get_parallel_state",
        lambda: SimpleNamespace(pp=SimpleNamespace(size=2)),
    )

    updater = _new_updater()
    updater.rollout_engines = [FakeEngine(), FakeEngine(), FakeEngine()]
    updater.rollout_engine_lock = FakeLock()
    updater._relay_engine = updater.rollout_engines[0]
    updater._peer_engines = list(updater.rollout_engines[1:])
    updater._relay_gpu_count = 4
    updater._next_fanout_port = 20009

    updater._connect_coordinator()

    assert updater._coordinator is existing_coordinator
    assert len(existing_coordinator.configure.calls) == 1
    _, configure_kwargs = existing_coordinator.configure.calls[0]
    assert configure_kwargs["rollout_engines"] == updater.rollout_engines
    assert configure_kwargs["relay_engine"] == updater._relay_engine
    assert configure_kwargs["peer_engines"] == updater._peer_engines
    assert configure_kwargs["relay_gpu_count"] == 4
    assert configure_kwargs["next_fanout_port"] == 20009
    assert barriers == ["gloo"]


def test_finalize_notifies_coordinator_after_submitting_bucket_refs(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(mixin, "get_parallel_state", _source_parallel_state)

    updater = _new_updater()
    updater._coordinator = FakeCoordinator()
    updater.weight_version = 9
    updater._pending_coordinator_submit_refs = ["submit-0", "submit-1"]

    updater._finalize_and_resume_engines()

    assert updater._pending_coordinator_submit_refs == []
    _, done_kwargs = updater._coordinator.mark_source_done.calls[0]
    assert done_kwargs == {"weight_version": 9, "pp_rank": 0}


def test_wait_pending_fanout_collects_result_and_syncs_errors(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sendrecv_broadcast, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_world_size", lambda group: 1)
    gathered_errors = []

    def fake_all_gather_object(output, obj, group):
        gathered_errors.append((obj, group))
        output[0] = obj

    monkeypatch.setattr(sendrecv_broadcast.dist, "all_gather_object", fake_all_gather_object)

    updater = _new_updater()
    updater._coordinator = FakeCoordinator()
    updater._next_fanout_port = 20000

    updater.wait_pending_fanout()

    assert updater._next_fanout_port == 20003
    assert gathered_errors == [(None, "gloo")]


def test_wait_pending_fanout_propagates_rank_zero_failure(monkeypatch):
    def raise_failure(ref):
        raise RuntimeError("fanout exploded")

    monkeypatch.setattr(sendrecv_broadcast.ray, "get", raise_failure)
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sendrecv_broadcast, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        sendrecv_broadcast.dist,
        "all_gather_object",
        lambda output, obj, group: output.__setitem__(0, obj),
    )

    updater = _new_updater()
    updater._coordinator = FakeCoordinator()

    with pytest.raises(RuntimeError, match="fanout exploded"):
        updater.wait_pending_fanout()


def test_ensure_success_raises_on_failed_sglang_response():
    with pytest.raises(RuntimeError, match="boom"):
        sendrecv_broadcast._ensure_success(
            [{"success": False, "message": "boom"}], "fanout"
        )
