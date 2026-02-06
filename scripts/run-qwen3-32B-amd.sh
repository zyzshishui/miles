#!/bin/bash
# Usage (3 nodes: 1 train + 2 rollout):
#   Node 1 (head/train):   bash scripts/run-qwen3-32B-amd-3node-1train-2rollout.sh head
#   Node 2-3 (rollout):    MASTER_ADDR=<head_ip> bash scripts/run-qwen3-32B-amd-3node-1train-2rollout.sh worker
#   Node 1 (submit):       nohup bash scripts/run-qwen3-32B-amd-3node-1train-2rollout.sh submit > nohup_disaggregated.out

set -euo pipefail

# ============== Common Setup ==============
setup_env() {
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    
    # Auto-detect network interface (for 10.28.x.x subnet)
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
}

cleanup() {
    echo "Cleaning up processes..."
    pkill -9 sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 2
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
    echo "Head started. Run on worker nodes (2 rollout nodes):"
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
    echo "Network: ${GLOO_SOCKET_IFNAME}"
    echo "=============================================="
    
    ray start --address=${MASTER_ADDR}:6379 \
        --num-gpus ${NUM_GPUS} \
        --node-ip-address ${WORKER_IP} \
        --disable-usage-stats
    
    echo ""
    echo "Worker started. Now submit job on head node:"
    echo "  bash $0 submit"
    echo ""
    echo "Press Ctrl+C to stop worker"
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
    
    MODEL_DIR="${MODEL_DIR:-/workspace}"
    DATA_DIR="${DATA_DIR:-/workspace}"
    
    source "${SCRIPT_DIR}/models/qwen3-32B.sh"
    
    # Model paths
    MODEL_NAME="Qwen3-32B"
    HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
    
    CKPT_ARGS=(
        --hf-checkpoint ${HF_CHECKPOINT}
        --ref-load ${MODEL_DIR}/${MODEL_NAME}_torch_dist
        --load ${MODEL_DIR}/${MODEL_NAME}_miles/
        --save ${MODEL_DIR}/${MODEL_NAME}_miles/
        --save-interval 20
    )
    
    ROLLOUT_ARGS=(
        --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
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
    
    EVAL_ARGS=(
        # --eval-interval 20
        # --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
        # --n-samples-per-eval-prompt 16
        # --eval-max-response-len 16384
        # --eval-top-p 1
    )
    
    # 1 train node (8 GPUs), TP=8, PP=1
    PERF_ARGS=(
        --tensor-model-parallel-size 8
        --sequence-parallel
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
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
    )
    
    WANDB_ARGS=(
        --use-wandb
        --wandb-project qwen3-32B
        --wandb-group qwen3-32B-3node-1train-2rollout
        --wandb-key ${WANDB_KEY}
    )
    
    # 2 rollout nodes (16 GPUs total), TP=8 per engine
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --sglang-mem-fraction-static 0.7
        --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
        --use-miles-router
    )
    
    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --no-gradient-accumulation-fusion  # Required for AMD
        --no-check-for-nan-in-loss-and-grad
    )
    
    MEGATRON_LM_PATH=$(python3 -c "import megatron; import os; print(os.path.dirname(os.path.dirname(megatron.__file__)))" 2>/dev/null || echo "/app/Megatron-LM")
    
    echo "=============================================="
    echo "Submitting Disaggregated Training Job"
    echo "Mode: 1 node train + 2 nodes rollout"
    echo "Train: 1 x 8 GPUs = 8 GPUs (TP=8)"
    echo "Rollout: 2 x 8 GPUs = 16 GPUs"
    echo "=============================================="
    
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="{
          \"env_vars\": {
             \"PYTHONPATH\": \"${MEGATRON_LM_PATH}/\",
             \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
             \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
             \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
             \"MILES_HOST_IP\": \"${MASTER_ADDR}\",
             \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
          }
        }" \
        -- python3 train.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node 8 \
        --rollout-num-gpus 16 \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${WANDB_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]}
}

show_usage() {
    echo "Usage: bash $0 <command>"
    echo ""
    echo "Commands:"
    echo "  head      - Start Ray head node (on train node)"
    echo "  worker    - Start Ray worker node (on rollout nodes)"
    echo "  submit    - Submit training job"
    echo ""
    echo "Example workflow:"
    echo "  1. Start head:      bash $0 head"
    echo "  2. Start workers:   MASTER_ADDR=<head_ip> bash $0 worker  (on 2 nodes)"
    echo "  3. Submit job:      bash $0 submit"
}

# ============== Main ==============

if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

case "$1" in
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
