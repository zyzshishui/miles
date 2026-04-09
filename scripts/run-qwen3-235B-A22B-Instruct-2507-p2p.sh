#!/bin/bash
# Qwen3-235B-A22B-Instruct-2507: P2P RDMA weight transfer, 2 train + 2 rollout
#
# Usage (4 nodes):
#   Node 1 (head/train):  bash scripts/run-qwen3-235B-A22B-Instruct-2507-p2p.sh head
#   Node 2-4 (workers):   MASTER_ADDR=<head_ip> bash scripts/run-qwen3-235B-A22B-Instruct-2507-p2p.sh worker
#   Node 1 (submit):      nohup bash scripts/run-qwen3-235B-A22B-Instruct-2507-p2p.sh submit > nohup_p2p_rdma.out

set -euo pipefail

# ============== Paths ==============
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
TORCH_DIST_PATH="${MODEL_DIR}/${MODEL_NAME}_torch_dist"
MILES_DIR="${MILES_DIR:-/workspace/p2prdma/miles}"
SGLANG_DIR="${SGLANG_DIR:-/workspace/p2prdma/sglang}"
export MODEL_ARGS_ROTARY_BASE=5000000
export MILES_SOCKET_IFNAME="${MILES_SOCKET_IFNAME:-eno1}"
export MILES_MOONCAKE_IB_DEVICE="${MILES_MOONCAKE_IB_DEVICE:-}"

# ============== Network Helpers ==============
resolve_ip_from_ifname() {
    local ifname="${1:?interface name is required}"
    python3 - "$ifname" <<'PY'
import fcntl
import socket
import struct
import sys

ifname = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    result = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))
except OSError as exc:
    raise SystemExit(f"Failed to resolve IPv4 for interface {ifname}: {exc}")
print(socket.inet_ntoa(result[20:24]))
PY
}

resolve_local_ip() {
    resolve_ip_from_ifname "${MILES_SOCKET_IFNAME}"
}

# ============== Environment ==============
setup_env() {
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-"expandable_segments:True"}
    export NUM_GPUS=$(echo ${HIP_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
    export MILES_HOST_IP="$(resolve_local_ip)"
    export SGLANG_LOCAL_IP_NIC="${SGLANG_LOCAL_IP_NIC:-${MILES_SOCKET_IFNAME}}"
    export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-${MILES_HOST_IP}}"
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${MILES_SOCKET_IFNAME}}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${MILES_SOCKET_IFNAME}}
    export SGLANG_MOONCAKE_IB_DEVICE="${SGLANG_MOONCAKE_IB_DEVICE:-${MILES_MOONCAKE_IB_DEVICE}}"
    export NCCL_DEBUG=${NCCL_DEBUG:-VERSION}
}

cleanup() {
    pkill -9 sglang 2>/dev/null || true; sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true; sleep 2
}

# ============== Ray Head/Worker ==============
run_head() {
    setup_env; cleanup
    export MASTER_ADDR="${MILES_HOST_IP}"
    echo "=== Ray Head (235B P2P) IFACE=${MILES_SOCKET_IFNAME} IP=${MASTER_ADDR} GPUs=${NUM_GPUS} MOONCAKE_IB=${SGLANG_MOONCAKE_IB_DEVICE:-auto} ==="
    ray start --head \
        --node-ip-address ${MASTER_ADDR} \
        --num-gpus ${NUM_GPUS} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265
    echo "Worker cmd: MASTER_ADDR=${MASTER_ADDR} bash $0 worker"
}

run_worker() {
    [ -z "${MASTER_ADDR:-}" ] && echo "Error: MASTER_ADDR not set" && exit 1
    setup_env; cleanup
    local worker_ip="${MILES_HOST_IP}"
    echo "=== Ray Worker (235B P2P) IFACE=${MILES_SOCKET_IFNAME} Head=${MASTER_ADDR} Worker=${worker_ip} MOONCAKE_IB=${SGLANG_MOONCAKE_IB_DEVICE:-auto} ==="
    ray start \
        --address=${MASTER_ADDR}:6379 \
        --num-gpus ${NUM_GPUS} \
        --node-ip-address ${worker_ip} \
        --disable-usage-stats
    while true; do sleep 60; echo "[$(date)] Worker alive"; done
}

# ============== Submit ==============
run_submit() {
    setup_env
    ray status

    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
    cd "${SCRIPT_DIR}/.."

    export MASTER_ADDR="${MILES_HOST_IP}"
    export PYTHONUNBUFFERED=1
    export DEPRECATED_MEGATRON_COMPATIBLE=1

    source "${SCRIPT_DIR}/models/qwen3-235B-A22B.sh"

    MEGATRON_LM_PATH=$(python3 -c \
        "import megatron, os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" \
        2>/dev/null || echo "/app/Megatron-LM")

    CKPT_ARGS=(
        --hf-checkpoint "${HF_CHECKPOINT}"
        --load "${TORCH_DIST_PATH}"
    )

    ROLLOUT_ARGS=(
        --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type deepscaler
        --num-rollout 3000
        --rollout-batch-size 32
        --n-samples-per-prompt 8
        --rollout-max-response-len 8192
        --rollout-temperature 1
        --global-batch-size 256
        --balance-data
    )

    PERF_ARGS=(
        --tensor-model-parallel-size 4
        --sequence-parallel
        --pipeline-model-parallel-size 2
        --context-parallel-size 1
        --expert-model-parallel-size 8
        --expert-tensor-parallel-size 1
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 9216
    )

    GRPO_ARGS=(
        --advantage-estimator grpo
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --entropy-coef 0.00
        --eps-clip 0.2
        --eps-clip-high 0.28
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
        # --disable-weights-backuper
        # --exp-avg-dtype bf16
        # --exp-avg-sq-dtype bf16
    )

    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --sglang-ep-size 8
        --sglang-mem-fraction-static 0.7
        --sglang-max-total-tokens 163840
        --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
        --sglang-remote-instance-weight-loader-start-seed-via-transfer-engine
    )

    if [ -n "${SGLANG_MOONCAKE_IB_DEVICE}" ]; then
        SGLANG_ARGS+=(--sglang-mooncake-ib-device "${SGLANG_MOONCAKE_IB_DEVICE}")
    fi

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
    )

    P2P_ARGS=(
        --update-weight-transfer-mode p2p
        --update-weight-buffer-size "$((2 * 1024 * 1024 * 1024))"
        --check-weight-update-equal
    )

    WANDB_ARGS=(
        --use-wandb
        --wandb-project miles-p2p-rdma
        --wandb-group Qwen3-235B-A22B-Instruct-2507
        --wandb-key ${WANDB_KEY}
    )

    TRAIN_ARGS=(
        train_async.py
        --update-weights-interval 2
        --actor-num-nodes 2
        --actor-num-gpus-per-node 8
        --rollout-num-gpus 16
        "${MODEL_ARGS[@]}"
        "${CKPT_ARGS[@]}"
        "${ROLLOUT_ARGS[@]}"
        "${PERF_ARGS[@]}"
        "${GRPO_ARGS[@]}"
        "${OPTIMIZER_ARGS[@]}"
        "${SGLANG_ARGS[@]}"
        "${MISC_ARGS[@]}"
        "${P2P_ARGS[@]}"
        "${WANDB_ARGS[@]}"
    )

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="{
            \"env_vars\": {
                \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/:${SGLANG_DIR}/python:${MILES_DIR}\",
                \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
                \"DEPRECATED_MEGATRON_COMPATIBLE\": \"1\",
                \"RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES\": \"1\",
                \"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES\": \"1\",
                \"HIP_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
                \"PYTORCH_HIP_ALLOC_CONF\": \"expandable_segments:True\",
                \"MILES_SOCKET_IFNAME\": \"${MILES_SOCKET_IFNAME}\",
                \"MILES_MOONCAKE_IB_DEVICE\": \"${MILES_MOONCAKE_IB_DEVICE}\",
                \"SGLANG_LOCAL_IP_NIC\": \"${SGLANG_LOCAL_IP_NIC}\",
                \"SGLANG_MOONCAKE_IB_DEVICE\": \"${SGLANG_MOONCAKE_IB_DEVICE}\",
                \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
                \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
                \"no_proxy\": \"${MASTER_ADDR},127.0.0.1\",
                \"MASTER_ADDR\": \"${MASTER_ADDR}\"
            }
        }" \
        -- python3 "${TRAIN_ARGS[@]}"
}

print_ip() {
    setup_env
    echo "${MILES_HOST_IP}"
}

# ============== Main ==============
[ $# -lt 1 ] && echo "Usage: bash $0 {head|worker|submit|print-ip}" && exit 1
case "$1" in
    head)   run_head ;;
    worker) run_worker ;;
    submit) run_submit ;;
    print-ip) print_ip ;;
    *)      echo "Unknown command: $1"; exit 1 ;;
esac
