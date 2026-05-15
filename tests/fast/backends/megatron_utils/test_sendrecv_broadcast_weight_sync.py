from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
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
        self.update_weights_from_distributed = FakeRemoteMethod(
            {"success": True, "message": "ok"}
        )
        self.send_weights_to_remote_instance = FakeRemoteMethod({"success": True, "message": "ok"})


class FakeLock:
    def __init__(self):
        self.acquire = FakeRemoteMethod(True)
        self.release = FakeRemoteMethod(True)


class FakeRemoteTask:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _new_updater():
    updater = UpdateWeightSendRecvBroadcast.__new__(UpdateWeightSendRecvBroadcast)
    updater.args = SimpleNamespace(sglang_pp_size=1)
    updater._pp_rank = 0
    updater._pending_relay_update_records = []
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


def test_fanout_initializes_ranked_send_groups_and_broadcasts(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.ray, "get", lambda ref: ref)
    updater = _new_updater()
    relay = FakeEngine()
    peer = FakeEngine()
    updater._relay_engine = relay
    updater._peer_engines = [peer]
    updater._relay_gpu_count = 2
    updater._next_fanout_port = 20000
    updater.weight_version = 7
    updater.rollout_engine_lock = FakeLock()

    updater._fanout_relay_weights_to_peer_instances()

    _, relay_init_kwargs = relay.init_weights_send_group_for_remote_instance.calls[0]
    _, peer_init_kwargs = peer.init_weights_send_group_for_remote_instance.calls[0]
    assert relay_init_kwargs["group_rank"] == 0
    assert peer_init_kwargs["group_rank"] == 1
    assert relay_init_kwargs["world_size"] == peer_init_kwargs["world_size"] == 2
    assert relay_init_kwargs["ports"] == peer_init_kwargs["ports"] == "23456,23457"
    assert relay_init_kwargs["group_name"] == peer_init_kwargs["group_name"]
    assert len(relay.send_weights_to_remote_instance.calls) == 1
    assert len(peer.send_weights_to_remote_instance.calls) == 1


def test_update_bucket_uses_nccl_to_send_to_relay(monkeypatch):
    monkeypatch.setattr(mixin, "get_parallel_state", _source_parallel_state)
    helper_calls = []

    def fake_update_weights_from_distributed(
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
        "update_weights_from_distributed",
        fake_update_weights_from_distributed,
    )

    updater = _new_updater()
    relay = FakeEngine()
    updater._relay_engine = relay
    updater._relay_update_group = "relay-nccl-group"
    updater.rollout_engine_lock = FakeLock()
    updater._group_name = "miles-sendrecv-broadcast-train-pp_0"
    updater._bucket_id = 0
    updater.weight_version = 3

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
    assert updater._pending_relay_update_records == [
        (0, 0, ["relay-update-ref"])
    ]


def test_finalize_schedules_background_fanout(monkeypatch):
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(sendrecv_broadcast, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(sendrecv_broadcast.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        sendrecv_broadcast.dist,
        "all_gather_object",
        lambda output, obj, group: output.__setitem__(0, obj),
    )
    fanout_task = FakeRemoteTask({"next_fanout_port": 20003})
    monkeypatch.setattr(sendrecv_broadcast, "_run_relay_fanout_and_resume", fanout_task)

    updater = _new_updater()
    relay = FakeEngine()
    peer = FakeEngine()
    updater.rollout_engines = [relay, peer]
    updater.rollout_engine_lock = FakeLock()
    updater._relay_engine = relay
    updater._peer_engines = [peer]
    updater._relay_gpu_count = 2
    updater._next_fanout_port = 20000
    updater.weight_version = 9
    updater._pending_fanout_ref = None
    updater._pending_relay_update_records = [
        (0, 1, ["pp0-bucket1-ref"]),
        (0, 0, ["pp0-bucket0-ref"]),
    ]

    updater._finalize_and_resume_engines()

    assert updater._pending_fanout_ref == {"next_fanout_port": 20003}
    assert fanout_task.calls[0]["relay_engine"] is relay
    assert fanout_task.calls[0]["peer_engines"] == [peer]
    assert fanout_task.calls[0]["weight_version"] == 9
    assert fanout_task.calls[0]["relay_update_refs"] == [
        "pp0-bucket0-ref",
        "pp0-bucket1-ref",
    ]
    assert updater._pending_relay_update_records == []


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
    updater._pending_fanout_ref = {"next_fanout_port": 20003}
    updater._next_fanout_port = 20000

    updater.wait_pending_fanout()

    assert updater._pending_fanout_ref is None
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
    updater._pending_fanout_ref = object()

    with pytest.raises(RuntimeError, match="fanout exploded"):
        updater.wait_pending_fanout()


def test_ensure_success_raises_on_failed_sglang_response():
    with pytest.raises(RuntimeError, match="boom"):
        UpdateWeightSendRecvBroadcast._ensure_success(
            [{"success": False, "message": "boom"}], "fanout"
        )
