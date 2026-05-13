# DeepSeek V4 Flash ROCm 支持状态

本文记录当前分支对 `deepseek-ai/DeepSeek-V4-Flash` / DeepSeek V4 Flash 在 ROCm
MI355X 环境上的支持进度、已完成修改、验证结果和剩余阻塞。

当前验证环境：

- 机器：8 * MI355X
- 基础容器：`rocm/sgl-dev:rocm720-mi35x-0363e6c-20260510-DSv4`
- SGLang 工作区：`/sgl-workspace/sglang`
- 目标镜像参考：`radixark/miles:deepseek-v4`
- 训练框架：Miles + Megatron + SGLang collocate
- 目标模型：`deepseek-ai/DeepSeek-V4-Flash`

## 当前支持程度

已支持并验证：

- 8 卡 ROCm 上跑通 DeepSeek V4 Flash 1-layer / 256-expert 的 SGLang rollout。
- 8 卡 ROCm 上跑通 1-layer / 256-expert 的 Megatron logprob replay。
- 8 卡 ROCm 上跑通 1-layer / 8-expert partial model 的 SGLang rollout。
- 8 卡 ROCm 上跑通 1-layer / 8-expert partial model 的 Megatron logprob replay。
- HF checkpoint 可通过 MBridge 直接加载到 Megatron，用于避开当前 torch-dist 转换不稳定路径。
- DeepSeek V4 Flash 0415 checkpoint 的 KV-QAT 语义已和 SGLang 对齐。
- ROCm 上 SGLang ordinary MoE path 的 `routed_scaling_factor` 漏乘问题已修复。
- TileLang ROCm sparse MLA/indexer kernel 中容易修复的兼容问题已处理到可以进入 forward/logprob 验证。
- partial model 支持 `DeepSeek-V4-Flash-FP8-1layer-8experts`，可用于低显存 smoke / 精度排查。

尚未在当前环境完成：

- 真正的 backward / optimizer training step 当前被 ROCm/HSA 可分配显存异常阻塞。
- full 43-layer DeepSeek V4 Flash 尚未完成端到端训练验证。
- 多节点未验证。
- 当前容器内的完整训练还不能作为通过标准；forward、rollout、logprob parity 可以作为已验证范围。

## 验证结果

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| 1-layer / 256-expert rollout-only，8 GPU | 通过 | run id: `kvqat_rolloutonly_scaled_1layer_tp8ep8dp1_20260512T234913Z` |
| 1-layer / 256-expert train-only logprob replay，8 GPU | 通过 | run id: `kvqat_trainonly_scaled_1layer_hf_mbridge_0415_kvqat_trace_nor3_from_freshrollout_20260512T235157Z` |
| 1-layer / 256-expert rollout/logprob diff | 通过 | rollout `-1.5027389526367188`，Megatron `-1.476759672164917`，mean diff `0.0259793`，在 `0.03` 容忍内 |
| SGLang/Megatron hidden 对齐 | 通过 | `pre_mlp_ln mean 0.0011724`，`mlp_out mean 0.0021304`，`final_norm mean 0.0016235` |
| 1-layer / 8-expert partial rollout-only，8 GPU | 通过 | run id: `rolloutonly_8experts_1layer_tp8ep8dp1_20260513T004053Z` |
| 1-layer / 8-expert partial logprob-only replay，8 GPU | 通过 | run id: `logprobonly_8experts_numexperts8_skipinitfix_1layer_hf_mbridge_from_rolloutonly_8experts_1layer_tp8ep8dp1_20260513T004053Z_20260513T005136Z` |
| 1-layer / 8-expert partial rollout/logprob diff | 通过 | rollout `-1.324514627456665`，Megatron `-1.334312915802002`，diff `-0.009798288345337` |
| 1-layer / 8-expert partial backward | 阻塞 | 进入 `actor_train` 后 ROCr 报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` |
| synthetic 8-token backward | 阻塞 | 同样进入 `actor_train` 后 ROCr OOM |
| 最小 HIP allocator 测试 | 阻塞 | 无 KFD 进程时 `hipMemGetInfo` 只暴露约 `4.05 GiB` free，`hipMalloc(8GB)` 失败 |

关键日志位于：

- `/root/miles/logs/v4-flash/kvqat_rolloutonly_scaled_1layer_tp8ep8dp1_20260512T234913Z.log`
- `/root/miles/logs/v4-flash/kvqat_trainonly_scaled_1layer_hf_mbridge_0415_kvqat_trace_nor3_from_freshrollout_20260512T235157Z.log`
- `/root/miles/logs/v4-flash/rolloutonly_8experts_1layer_tp8ep8dp1_20260513T004053Z.log`
- `/root/miles/logs/v4-flash/logprobonly_8experts_numexperts8_skipinitfix_1layer_hf_mbridge_from_rolloutonly_8experts_1layer_tp8ep8dp1_20260513T004053Z_20260513T005136Z.log`

操作记录位于：

- `/tmp/ops-logs/4d20d2c8-miles/2026-05-12/2225-v4-flash-rocm-precision/ops.md`
- `/tmp/ops-logs/4d20d2c8-miles/2026-05-12/2225-v4-flash-rocm-precision/verify.md`
- `/tmp/ops-logs/4d20d2c8-miles/2026-05-12/2225-v4-flash-rocm-precision/summary.md`

## 已完成工作

### 1. 坚持 8 卡 collocate / partial 路线

排查过程中不再走单卡路线。优先目标是 8 卡 collocate 训练整模型；当前显存/allocator
异常下，降级为 8 卡 partial model 做 forward、rollout、logprob、精度闭环验证。

`scripts/run_deepseek_v4.py` 增加了：

- `DeepSeek-V4-Flash-FP8-1layer`
- `DeepSeek-V4-Flash-FP8-1layer-8experts`
- 单节点 full Flash 的 TP8/EP8 默认并行配置
- partial model 的 `model_type` patch
- 0415 checkpoint 默认环境：`MILES_DSV4_CKPT_VERSION=0415`、`MEGATRON_USE_KV_QAT=1`
- debug replay 时避免不必要的 collocate/router/weight-sync 路径

新增 `scripts/models/deepseek-v4-flash-1layer.sh`，用于 1-layer partial 配置。

### 2. 修复 ROCm MBridge scatter 精度损坏

ROCm 下 MBridge 用 `torch.distributed.scatter` 分发 TP/ETP shard 时，非 source rank 的
BF16 shard 会出现 corruption。新增：

- `miles/utils/rocm_distributed.py`

其中 `patch_rocm_scatter_with_broadcast()` 在 ROCm 上把 blocking scatter 替换为
broadcast-loop scatter。它已接入：

- `miles/backends/megatron_utils/checkpoint.py`
- `tools/convert_hf_to_torch_dist.py`

这解决了 checkpoint load 后 TP rank 1-7 权重不一致导致的 logprob 偏移。

### 3. 用 MBridge 直接从 HF 加载 Megatron

新增 `--load-hf-with-mbridge`，允许 Megatron raw provider 直接从 HF checkpoint 加载权重，
绕过当前不稳定的 torch-dist 中间格式。

涉及文件：

- `miles/utils/arguments.py`
- `miles/backends/megatron_utils/checkpoint.py`
- `tools/convert_hf_to_torch_dist.py`

同时修正 MBridge safetensor load 的 current-device 选择，避免 ROCm 多进程加载时落到错误设备。

### 4. 对齐 DeepSeek V4 Flash 0415 KV-QAT 语义

`miles_plugins/models/deepseek_v4/deepseek_v4.py` 现在默认把 0415 checkpoint 视为
KV-QAT 路径，并只对 noPE KV 部分做 quant-dequant：

- `kv_vanilla[..., :-rd] = fp8_simulate_qat(kv_vanilla[..., :-rd], 64)`

RoPE 部分不再被错误地一起 QAT。这是 rollout/logprob 对齐的关键修复。

`miles_plugins/models/deepseek_v4/ops/v4_indexer.py` 也修正了 `fp8_simulate_qat(q, 128)`
调用方式，避免 ROCm/TileLang 调试时的参数不兼容。

### 5. 修复 SGLang ROCm MoE routed scaling

SGLang 的 ordinary ROCm MoE path 在 AITER 可 import 时没有乘 `routed_scaling_factor`，
导致 SGLang rollout logits 和 Megatron 系统性漂移。当前环境中已直接修改：

- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py`

修复后，非 CUDA/MUSA/XPU 路径会始终应用 `routed_scaling_factor`。新的 Dockerfile
也把这个修复以内联 patch 的形式写入，避免重建环境后丢失。

### 6. 修复 ROCm Ray / visible devices 兼容

Ray 在 ROCm 下会同时处理 `CUDA_VISIBLE_DEVICES`、`HIP_VISIBLE_DEVICES`、
`ROCR_VISIBLE_DEVICES`。这次调整使 Ray driver、train actor、SGLang actor 使用同一套
ROCm visible-device 语义。

涉及文件：

- `miles/utils/external_utils/command_utils.py`
- `miles/ray/actor_group.py`
- `miles/ray/train_actor.py`
- `miles/backends/sglang_utils/sglang_engine.py`

同时增加了非 `ray submit` 的 direct execution 分支，方便在已有 Ray cluster / 当前容器内排查。

### 7. 增加 debug replay / logprob-only 路径

为了在显存紧张环境下先闭环精度，新增和修正了：

- `--skip-train`
- `--skip-initial-update-weights`
- `--disable-distributed-optimizer`
- `--load-hf-with-mbridge`
- debug train-only 时不启动 rollout router
- skip-train 时不包 DDP、不创建 optimizer/scheduler、不创建 weight backuper
- skip-train 时仍保存 debug rollout data、计算 logprob / mismatch 指标

涉及文件：

- `train.py`
- `miles/utils/arguments.py`
- `miles/backends/megatron_utils/model.py`
- `miles/backends/megatron_utils/actor.py`
- `miles/ray/rollout.py`

### 8. 修复 replay shape 和 indexer replay

DeepSeek V4 indexer replay 在 SP/non-SP 区域会出现 3D/4D shape 差异。已修复：

- `miles/backends/megatron_utils/actor.py`
- `miles/backends/megatron_utils/replay_utils.py`
- `miles/utils/replay_base.py`

现在 replay gather 统一按最后一维 top-k 处理，indexer replay 可以接受
`(batch, seq, layer, topk)` 和 `(tokens, layer, topk)` 两类输入。

### 9. 修复 DeepSeek V4 ROCm TileLang kernel 兼容问题

涉及文件：

- `miles_plugins/models/deepseek_v4/ops/attention_core.py`
- `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py`
- `miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py`

修复点：

- `attention_core.py` 保留原始输出 dtype。
- sparse MLA forward 去掉未使用 shared allocation。
- ROCm heads >= 64 时 forward 强制 `num_stages=1`。
- ROCm padded heads >= 64 时 backward `block_H` 上限降到 32。

这些修改让 TileLang ROCm sparse MLA/indexer 路径可以继续推进到 forward/logprob 验证。

### 10. 修复权重更新后的 SGLang post-process

DeepSeek V4 ROCm 下即使没有 HF quantization config，也可能需要 SGLang 的 post-load kernel
layout，例如 AITER MoE 会 shuffle unquantized expert weights。已调整：

- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`

现在 `model_name == "deepseekv4"` 时会在非 LoRA 权重更新后执行 post-process。

### 11. 修复 DeepSeek V4 Megatron -> HF APE hotfix mirror

`miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py` 中，APE mirror hotfix 现在只在
非 2604 模式且 `SGLANG_ENABLE_APE_HOTFIX` 开启时应用，避免默认 2604/0415 路径重复重排。

### 12. 兼容当前依赖版本

小修包括：

- `miles/router/router.py`：FastAPI startup event 注册兼容。
- `miles/utils/reloadable_process_group.py`：兼容没有 `Exception.add_note` 的 Python。
- `miles/utils/transformers_patch.py`：改用当前 SGLang 的 DeepSeek V4 HF loader。
- `miles/backends/sglang_utils/sglang_engine.py`：SGLang server args 构造和 launch 时包住 transformers patch。

### 13. 新增排查工具

新增临时 debug 工具：

- `tools/debug_rocm_scatter.py`
- `tools/debug_v4_bridge_compare.py`
- `tools/debug_v4_forward_trace_hook.py`
- `tools/debug_v4_nan_hook.py`

这些用于定位 scatter corruption、MBridge/Megatron 权重差异、forward hidden drift 和 NaN。

## 新增 Dockerfile

新增：

- `docker/Dockerfile.rocm_MI350-5_DeepSeek-V4`

它从 `docker/Dockerfile.rocm_MI350-5` 复制而来，并写入了 DeepSeek V4 相关环境修改：

- DeepSeek V4 Flash 默认 runtime env：
  - `MILES_DSV4_CKPT_VERSION=0415`
  - `MEGATRON_USE_KV_QAT=1`
  - `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`
  - `SGLANG_OPT_USE_TILELANG_INDEXER=true`
  - `SGLANG_OPT_USE_AITER_MHC_PRE=true`
  - `SGLANG_OPT_USE_AITER_MHC_POST=true`
  - `SGLANG_DSV4_FP4_EXPERTS=0`
  - 以及 ROCm/SGLang V4 相关开关
- SGLang DeepSeek V4 ROCm MoE `routed_scaling_factor` correctness patch。
- TileLang ROCm source build：
  - repo: `tile-ai/tilelang`
  - commit: `a55a82302bf7f3c5af635b5c9146f728185cc900`
  - build env: `USE_ROCM=1 USE_CUDA=0 ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950`

这个 Dockerfile 还没有在本文档编写时完整重建验证；当前验证是在已有容器中手动修环境后完成。

## 当前阻塞

训练 backward 当前不是模型逻辑或 logprob 精度问题，而是当前容器/驱动状态下 ROCm HSA
allocator 暴露的可分配显存异常：

- `amd-smi` / sysfs 可看到每张卡约 `309 GB` VRAM。
- 没有 KFD 进程时，最小 HIP 测试的 `hipMemGetInfo` 只返回约 `4.05 GiB` free。
- `hipMalloc(8GB)` 直接失败。
- 8-expert partial 的正常 replay 能进入 `actor_train`，随后 ROCr 报
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。
- synthetic 8-token replay 也在同一位置失败。
- `rocm-smi --gpureset` 无效。
- `amd-smi reset --reload-driver --gpu all` 返回 `AMDSMI_STATUS_AMDGPU_RESTART_ERR`。

因此，当前结论是：DeepSeek V4 Flash 的 ROCm forward / rollout / logprob 精度链路已经
打通，actual training step 被当前机器或容器的 HSA allocatable-memory 状态阻塞。

## 后续建议

1. 在一个 `hipMemGetInfo` 能正确暴露大部分 VRAM 的干净容器或重置后的机器上，先重跑
   8-expert partial backward。
2. 用 `docker/Dockerfile.rocm_MI350-5_DeepSeek-V4` 重建镜像，确认内联 SGLang patch 和
   TileLang ROCm build 可复现。
3. backward 通过后，再把验证从 1-layer / 8-expert partial 扩到 1-layer / 256-expert，
   最后扩到 full layer。
4. 如果 full model 训练仍受显存限制，继续保留 8 卡 partial model 作为 CI/smoke target。
