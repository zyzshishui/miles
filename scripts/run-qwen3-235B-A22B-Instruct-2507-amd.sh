#!/bin/bash
# Qwen3-235B-A22B-Instruct-2507: 下载、转 torch_dist、训练，模型目录 /workspace
#
# Usage:
#   1. 下载模型:   bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh download
#   2. 转 torch_dist: bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh convert
#   3. 多机训练 (3 节点: 2 train + 1 rollout):
#      Node 1 (head):  bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh head
#      Node 2 (train): MASTER_ADDR=<head_ip> bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh worker
#      Node 3 (rollout): MASTER_ADDR=<head_ip> bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh worker
#      Node 1 (submit): nohup bash scripts/run-qwen3-235B-A22B-Instruct-2507-amd.sh submit > nohup_235b.out

set -euo pipefail

# 模型与路径（与 run-qwen3-32B-amd.sh 一致风格，模型放 /workspace）
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
HF_REPO="Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
TORCH_DIST_PATH="${MODEL_DIR}/${MODEL_NAME}_torch_dist"

# ============== Common Setup ==============
setup_env() {
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

    MILES_IFNAME=$(python3 -c "
import socket, struct, fcntl, os
for iface in os.listdir('/sys/class/net/'):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, struct.pack('256s', bytes(iface[:15], 'utf-8')))[20:24])
        if ip.startswith('10.28.'):
            print(iface)
            break
    except: pass
")
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$MILES_IFNAME}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$MILES_IFNAME}
    export NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)

    # RCCL RDMA/RoCE configuration for cross-node communication
    export NCCL_IB_HCA=${NCCL_IB_HCA:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}
    # NOTE: On these nodes, only GID indices 0/1 are populated; 3 is all-zero and will break NET/IB connects.
    # Prefer an IPv4-mapped RoCEv2 GID when available (typically index 1).
    if [[ -z "${NCCL_IB_GID_INDEX+x}" ]]; then
        if [[ -r /sys/class/infiniband/rdma0/ports/1/gids/1 ]] && [[ "$(cat /sys/class/infiniband/rdma0/ports/1/gids/1)" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
            export NCCL_IB_GID_INDEX=1
        else
            export NCCL_IB_GID_INDEX=0
        fi
    fi
    export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-SYS}
    export NCCL_DEBUG=INFO
}

cleanup() {
    echo "Cleaning up processes..."
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2
    # Remove stale Ray session data to avoid session name conflict on restart
    rm -rf /tmp/ray/ 2>/dev/null || true
}

# ============== Download ==============
run_download() {
    echo "=============================================="
    echo "Downloading ${HF_REPO} to ${HF_CHECKPOINT}"
    echo "=============================================="
    mkdir -p "${MODEL_DIR}"
    if [ -d "${HF_CHECKPOINT}" ] && [ -f "${HF_CHECKPOINT}/config.json" ]; then
        echo "Skip download: ${HF_CHECKPOINT} already exists with config.json"
        exit 0
    fi
    huggingface-cli download "${HF_REPO}" --local-dir "${HF_CHECKPOINT}" --local-dir-use-symlinks False
    echo "Download done: ${HF_CHECKPOINT}"
}

# ============== Convert HF -> torch_dist ==============
run_convert() {
    setup_env
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
    cd "${REPO_ROOT}"

    if [ -d "${TORCH_DIST_PATH}" ]; then
        echo "Skip convert: ${TORCH_DIST_PATH} already exists"
        exit 0
    fi

    echo "=============================================="
    echo "Converting ${HF_CHECKPOINT} -> ${TORCH_DIST_PATH}"
    echo "Using ${NUM_GPUS} GPUs"
    echo "=============================================="
    if [ ! -d "${HF_CHECKPOINT}" ] || [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
        echo "Error: HF checkpoint not found at ${HF_CHECKPOINT}. Run: bash $0 download"
        exit 1
    fi

    MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")
    export PYTHONPATH="${MEGATRON_LM_PATH}:${REPO_ROOT}"

    source "${SCRIPT_DIR}/models/qwen3-235B-A22B.sh"
    torchrun --nproc-per-node "${NUM_GPUS}" \
        tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${HF_CHECKPOINT}" \
        --save "${TORCH_DIST_PATH}"

    echo "Convert done: ${TORCH_DIST_PATH}"
}

# ============== Head Node ==============
run_head() {
    setup_env
    cleanup

    export MASTER_ADDR=$(hostname -I | awk '{print $1}')

    echo "=============================================="
    echo "Starting Ray Head Node"
    echo "Head IP: ${MASTER_ADDR}"
    echo "GPUs: ${NUM_GPUS}"
    echo "Network: ${GLOO_SOCKET_IFNAME}"
    echo "=============================================="

    ray start --head --node-ip-address ${MASTER_ADDR} \
        --num-gpus ${NUM_GPUS} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    echo ""
    echo "Head started. Run on worker nodes:"
    echo "  MASTER_ADDR=${MASTER_ADDR} bash $0 worker"
    echo ""
    echo "Then submit job:"
    echo "  bash $0 submit"
}

# ============== Worker Node ==============
run_worker() {
    if [ -z "${MASTER_ADDR:-}" ]; then
        echo "Error: MASTER_ADDR not set"
        echo "Usage: MASTER_ADDR=<head_ip> bash $0 worker"
        exit 1
    fi

    setup_env
    cleanup

    WORKER_IP=$(hostname -I | awk '{print $1}')

    echo "=============================================="
    echo "Starting Ray Worker Node"
    echo "Head IP: ${MASTER_ADDR}"
    echo "Worker IP: ${WORKER_IP}"
    echo "GPUs: ${NUM_GPUS}"
    echo "=============================================="

    ray start --address=${MASTER_ADDR}:6379 \
        --num-gpus ${NUM_GPUS} \
        --node-ip-address ${WORKER_IP} \
        --disable-usage-stats

    echo ""
    echo "Worker started. Submit job on head node: bash $0 submit"
    while true; do
        sleep 60
        echo "[$(date)] Worker running..."
    done
}

# ============== Submit Job ==============
run_submit() {
    setup_env

    echo "Checking Ray cluster..."
    ray status

    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    cd "${SCRIPT_DIR}/.."

    export MASTER_ADDR=$(hostname -I | awk '{print $1}')
    export PYTHONBUFFERED=16

    source "${SCRIPT_DIR}/models/qwen3-235B-A22B.sh"

    CKPT_ARGS=(
        --hf-checkpoint "${HF_CHECKPOINT}"
        --ref-load "${TORCH_DIST_PATH}"
        # --load ${MODEL_DIR}/${MODEL_NAME}_miles/
        # --save ${MODEL_DIR}/${MODEL_NAME}_miles/
        # --save-interval 20000
    )

    ROLLOUT_ARGS=(
        --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type deepscaler
        --num-rollout 3000
        --rollout-batch-size 8
        --n-samples-per-prompt 8
        --rollout-max-response-len 8192
        --rollout-temperature 1
        --global-batch-size 64
        --balance-data
    )

    EVAL_ARGS=(
        # --eval-interval 20
        # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
        # --n-samples-per-eval-prompt 16
        # --eval-max-response-len 16384
        # --eval-top-p 1
    )

    # 235B-A22B: TP=4, PP=2, EP=8 (2 train nodes x 8 GPUs = 16, DP=2)
    # PP=2: 每节点1个stage, bubble降低; EP=8: MoE all-to-all全节点内
    # 通信: TP/DP/EP全在节点内, 仅PP p2p跨节点
    PERF_ARGS=(
        --tensor-model-parallel-size 4
        --sequence-parallel
        --pipeline-model-parallel-size 2
        --context-parallel-size 1
        --expert-model-parallel-size 8
        --expert-tensor-parallel-size 1
        --decoder-last-pipeline-num-layers 47
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 16384
    )

    GRPO_ARGS=(
        --advantage-estimator grpo
        --use-kl-loss
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --entropy-coef 0.00
        --eps-clip 4e-4
    )

    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr 1e-6
        --lr-decay-style constant
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
    )

    WANDB_ARGS=(
        --use-wandb
        --wandb-project qwen3-235B-A22B
        --wandb-group 2t-1r
        --wandb-key ${WANDB_KEY}
    )

    PROFILE_ARGS=(
        --use-pytorch-profiler
        --profile-target train_overall
        --profile-step-start 1
        --profile-step-end 2
        --tensorboard-dir /workspace/miles/profile_output
        # Default: cpu + cuda activities, no extra data collection (fast dump).
        # Uncomment below to collect more data (increases dump time):
        # --profile-activities cuda          # skip CPU ops to reduce dump size
        # --profile-record-shapes            # record tensor shapes
        # --profile-with-stack               # record source file/line (very slow dump)
        # --profile-memory                   # track memory alloc/dealloc
        # --profile-with-flops               # estimate FLOPs for matmul/conv
    )

    # 1 rollout node 8 GPUs
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --sglang-mem-fraction-static 0.7
        --sglang-cuda-graph-bs 1 2 4 8
        --use-miles-router
    )

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --no-gradient-accumulation-fusion
        --no-check-for-nan-in-loss-and-grad
    )

    MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")

    echo "=============================================="
    echo "Submitting Qwen3-235B-A22B-Instruct-2507 Training"
    echo "Mode: 2 nodes train + 1 node rollout"
    echo "Train: 2 x 8 GPUs = 16 (TP=4, PP=2, EP=8, DP=2)"
    echo "Rollout: 1 x 8 GPUs"
    echo "Model: ${HF_CHECKPOINT}, ref: ${TORCH_DIST_PATH}"
    echo "=============================================="

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="{
          \"env_vars\": {
             \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
             \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
             \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
             \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
             \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
             \"NCCL_IB_GID_INDEX\": \"${NCCL_IB_GID_INDEX}\",
             \"NCCL_NET_GDR_LEVEL\": \"${NCCL_NET_GDR_LEVEL}\",
             \"NCCL_DEBUG\": \"${NCCL_DEBUG}\",
             \"MILES_HOST_IP\": \"${MASTER_ADDR}\",
             \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
          }
        }" \
        -- python3 train.py \
        --actor-num-nodes 2 \
        --actor-num-gpus-per-node 8 \
        --rollout-num-gpus 8 \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]} \
        ${PROFILE_ARGS[@]}
}

show_usage() {
    echo "Usage: bash $0 <command>"
    echo ""
    echo "Commands:"
    echo "  download  - Download Qwen/Qwen3-235B-A22B-Instruct-2507 to \${MODEL_DIR:-/workspace}"
    echo "  convert   - Convert HF checkpoint to torch_dist (need run on single node with 8 GPUs)"
    echo "  head      - Start Ray head node"
    echo "  worker    - Start Ray worker node"
    echo "  submit    - Submit training job"
    echo ""
    echo "Example:"
    echo "  bash $0 download && bash $0 convert"
    echo "  bash $0 head   # on node1"
    echo "  MASTER_ADDR=<head_ip> bash $0 worker   # on node2, node3"
    echo "  bash $0 submit # on node1"
}

# ============== Main ==============
if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

case "$1" in
    download)
        run_download
        ;;
    convert)
        run_convert
        ;;
    head)
        run_head
        ;;
    worker)
        run_worker
        ;;
    submit)
        run_submit
        ;;
    *)
        echo "Error: Unknown command '$1'"
        show_usage
        exit 1
        ;;
esac
