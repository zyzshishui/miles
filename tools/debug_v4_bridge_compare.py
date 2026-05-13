import gc
import os

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger
from miles.utils.rocm_distributed import patch_rocm_scatter_with_broadcast
from miles.utils.transformers_patch import with_transformers_patch
from tools.convert_hf_to_torch_dist import (
    get_args,
    patch_mbridge_safetensor_current_device,
    patch_weight_to_mcore_format_preserve_fp32,
)


TARGET_GLOBAL_NAMES = {
    "decoder.layers.0.self_attention.wq_a.weight",
    "decoder.layers.0.self_attention.wq_b.weight",
    "decoder.layers.0.self_attention.wkv.weight",
    "decoder.layers.0.self_attention.wo_a.weight",
    "decoder.layers.0.self_attention.wo_b.weight",
}


def _target_global_names():
    extra = os.getenv("MILES_DEBUG_TARGET_GLOBAL_NAMES")
    if not extra:
        return TARGET_GLOBAL_NAMES
    return {x.strip() for x in extra.split(",") if x.strip()}


def _expected_local(bridge, local_name, hf_names, param):
    hf_weights = [bridge.safetensor_io.load_one_hf_weight(x) for x in hf_names]
    mcore_weight = bridge._weight_to_mcore_format(local_name, hf_weights)
    if ".mlp.experts.linear_fc" in local_name:
        pieces = bridge._weight_split_across_tp(
            local_name, mcore_weight, param, bridge.mpu.etp_size
        )
        idx = bridge.mpu.etp_rank
    else:
        pieces = bridge._weight_split_across_tp(
            local_name, mcore_weight, param, bridge.mpu.tp_size
        )
        idx = bridge.mpu.tp_rank
    return list(pieces)[idx].to(device=param.device, dtype=param.dtype)


def main():
    configure_logger()
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    global_rank = int(os.getenv("RANK") or 0)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", world_size=world_size, rank=global_rank)

    args = get_args()
    with with_transformers_patch():
        init(args)
        model = get_model(
            get_model_provider_func(args),
            ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
        )
        bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        patch_mbridge_safetensor_current_device()
        patch_rocm_scatter_with_broadcast()
        bridge.safetensor_io = bridge._get_safetensor_io(args.hf_checkpoint)
        patch_weight_to_mcore_format_preserve_fp32()
        if os.getenv("MILES_DEBUG_LOAD_TORCH_DIST") == "1":
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            load_checkpoint(
                model,
                None,
                None,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )
        else:
            bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)

    module = model[0]
    state = module.state_dict()
    local_to_global = bridge._weight_name_mapping_mcore_local_to_global(module)
    global_to_local = {v: k for k, v in local_to_global.items()}

    if os.getenv("MILES_DEBUG_LIST_NAMES") == "1":
        pattern = os.getenv("MILES_DEBUG_LIST_PATTERN")
        for local_name, global_name in sorted(local_to_global.items(), key=lambda x: x[1]):
            if pattern and pattern not in global_name and pattern not in local_name:
                continue
            param = state[local_name]
            print(
                "NAME "
                f"rank={global_rank} tp={bridge.mpu.tp_rank} etp={bridge.mpu.etp_rank} "
                f"global={global_name} local={local_name} "
                f"shape={tuple(param.shape)} dtype={param.dtype} "
                f"nan={torch.isnan(param.float()).any().item()}",
                flush=True,
            )
        dist.barrier()
        dist.destroy_process_group()
        return

    for global_name in sorted(_target_global_names()):
        local_name = global_to_local.get(global_name)
        if local_name is None:
            if global_rank == 0:
                print(f"MISSING_GLOBAL {global_name}", flush=True)
            continue
        param = state[local_name]
        hf_names = bridge._weight_name_mapping_mcore_to_hf(global_name)
        expected = _expected_local(bridge, local_name, hf_names, param)
        diff = (param.float() - expected.float()).abs()
        print(
            "COMPARE "
            f"rank={global_rank} tp={bridge.mpu.tp_rank} name={global_name} "
            f"local_name={local_name} "
            f"shape={tuple(param.shape)} max={diff.max().item():.8f} "
            f"mean={diff.mean().item():.8f} equal={torch.equal(param, expected)}",
            flush=True,
        )
        del expected, diff
        gc.collect()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
