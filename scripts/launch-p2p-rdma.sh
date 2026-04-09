#!/bin/bash
# Multi-node P2P RDMA launcher for miles
#
# Usage:
#   TRAIN_SCRIPT=scripts/run-qwen3-30B-A3B-p2p.sh bash scripts/launch-p2p-rdma.sh start
#   TRAIN_SCRIPT=scripts/run-qwen3-235B-A22B-Instruct-2507-p2p.sh bash scripts/launch-p2p-rdma.sh start
#
# Commands:
#   start      Full: containers + install + ray + submit
#   stop       Stop Ray cluster
#   status     Show cluster status
#   submit     Submit training job
#   logs       Tail training logs
#   cleanup    Stop + remove containers
#   containers / install / ray   Individual steps

set -euo pipefail

# ============== Configuration ==============
NODES=("mia1-p02-g23" "mia1-p02-g46" "mia1-p02-g05" "mia1-p02-g45")
HEAD_NODE="${NODES[0]}"

DOCKER_IMAGE="rlsys/miles:MI350-355-latest"
CONTAINER_NAME="yuzhen_miles_0330"
HOST_DATA_DIR="/it-share-2/data/yuzhzhou"
AITER_CACHE_ROOT="/tmp/yuzhzhou_aiter_cache"
WORKSPACE="/workspace"
MILES_DIR="${WORKSPACE}/p2prdma/miles"
SGLANG_DIR="${WORKSPACE}/p2prdma/sglang"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/run-qwen3-235B-A22B-Instruct-2507-p2p.sh}"
NETWORK_IFNAME="${NETWORK_IFNAME:-${MILES_SOCKET_IFNAME:-eno1}}"
MILES_MOONCAKE_IB_DEVICE="${MILES_MOONCAKE_IB_DEVICE:-}"
TRAIN_ENV="MILES_SOCKET_IFNAME=${NETWORK_IFNAME} MILES_MOONCAKE_IB_DEVICE=${MILES_MOONCAKE_IB_DEVICE}"
SSH_OPTS="-F /dev/null -i ~/.ssh/cluster_id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=30"

# ============== Helpers ==============
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ssh_cmd() {
    local host=$1; shift
    ssh ${SSH_OPTS} "$host" "$@"
}

docker_exec() {
    local host=$1; shift
    ssh_cmd "$host" "docker exec ${CONTAINER_NAME} bash -c '$*'"
}

check_container() {
    ssh_cmd "$1" "docker ps -q -f name=^${CONTAINER_NAME}\$" 2>/dev/null | grep -q .
}

get_container_ip() {
    docker_exec "$1" "cd ${MILES_DIR} && ${TRAIN_ENV} bash ${TRAIN_SCRIPT} print-ip"
}

# ============== Actions ==============
start_containers() {
    log "Starting containers on ${#NODES[@]} nodes..."
    for host in "${NODES[@]}"; do
        if check_container "$host"; then
            log "  $host: already running"
            continue
        fi
        log "  $host: creating..."
        ssh_cmd "$host" "mkdir -p ${AITER_CACHE_ROOT}/${CONTAINER_NAME} && docker run -itd --network=host --privileged \
            --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
            --ipc=host --shm-size 64G --group-add video \
            --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
            -v ${HOST_DATA_DIR}:${WORKSPACE} \
            -w ${MILES_DIR} \
            -v ${AITER_CACHE_ROOT}/${CONTAINER_NAME}:/root/.aiter \
            -v /data/yuzhzhou/cache/huggingface:/root/.cache/huggingface \
            -v /data/yuzhzhou/cache/torch:/root/.cache/torch \
            -v /data/yuzhzhou/cache/pip:/root/.cache/pip \
            -v /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71:/usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71:ro \
            -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
            -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
            -v ${HOST_DATA_DIR}/rccl-net-plugin:/opt/rocm/lib/rccl-net-plugin:ro \
            -e LD_LIBRARY_PATH=/opt/rocm/lib/rccl-net-plugin:/opt/rocm/lib \
            -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
            -e HSA_NO_SCRATCH_RECLAIM=1 \
            -e HF_HOME=/root/.cache/huggingface \
            -e WANDB_KEY=cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e \
            --name ${CONTAINER_NAME} \
            ${DOCKER_IMAGE}" || true
    done
    log "All containers ready."
}

install_packages() {
    log "Installing sglang + miles + mooncake + ray patch on all nodes..."
    for host in "${NODES[@]}"; do
        docker_exec "$host" '
            pip uninstall sglang -y -q 2>/dev/null
            cd '"${SGLANG_DIR}"'/python && pip install -e . --no-build-isolation --no-deps -q 2>&1 | tail -2
            cd '"${MILES_DIR}"' && pip install -e . -q 2>&1 | tail -2
            rm -rf /tmp/torch_memory_saver
            git clone --depth 1 https://github.com/fzyzcjy/torch_memory_saver.git /tmp/torch_memory_saver >/tmp/torch_memory_saver.clone.log 2>&1
            cd /tmp/torch_memory_saver && pip install -e . -q 2>&1 | tail -2
            pip install mooncake-transfer-engine-non-cuda -q 2>&1 | tail -1

            python3 -c "
path = \"/opt/venv/lib/python3.10/site-packages/ray/_private/accelerators/amd_gpu.py\"
with open(path) as f:
    lines = f.readlines()
out = []
i = 0
while i < len(lines):
    if \"ROCR_VISIBLE_DEVICES\" in lines[i] and \"if\" in lines[i] and \"os.environ\" in lines[i]:
        out.append(lines[i]); i += 1
        while i < len(lines) and (\"raise\" in lines[i] or \"Please use\" in lines[i] or lines[i].strip() == \")\"):
            i += 1
        out.append(\"            os.environ.pop(\\\"ROCR_VISIBLE_DEVICES\\\", None)\n\")
        out.append(\"            return AMDGPUAcceleratorManager.get_current_process_visible_accelerator_ids()\n\")
    else:
        out.append(lines[i]); i += 1
with open(path, \"w\") as f:
    f.writelines(out)
print(\"ray patched\")
"
            echo "=== done on $(hostname) ==="
        ' &
    done
    wait
    log "All packages installed."
}

start_ray() {
    log "Starting Ray cluster..."
    local head_ip
    head_ip=$(get_container_ip "${HEAD_NODE}")
    log "Using interface ${NETWORK_IFNAME}; head IP=${head_ip}; Mooncake IB=${MILES_MOONCAKE_IB_DEVICE:-auto}"

    # Cleanup all nodes
    for host in "${NODES[@]}"; do
        docker_exec "$host" \
            "pkill -9 -f '${TRAIN_SCRIPT}' 2>/dev/null || true; pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 python; rm -rf /tmp/ray/" \
            2>/dev/null || true
    done
    sleep 3

    # Start head
    docker_exec "${HEAD_NODE}" "cd ${MILES_DIR} && ${TRAIN_ENV} bash ${TRAIN_SCRIPT} head"
    local retries=0
    while ! docker_exec "${HEAD_NODE}" "ray status" &>/dev/null; do
        retries=$((retries + 1))
        [ $retries -ge 6 ] && log "ERROR: head timeout" && exit 1
        sleep 5
    done

    # Start workers
    for ((i = 1; i < ${#NODES[@]}; i++)); do
        docker_exec "${NODES[$i]}" \
            "cd ${MILES_DIR} && ${TRAIN_ENV} MASTER_ADDR=${head_ip} bash ${TRAIN_SCRIPT} worker" &
    done
    sleep 15

    docker_exec "${HEAD_NODE}" "ray status" || true
    echo ""
    log "Ray cluster ready: ${NODES[*]}"
}

submit_job() {
    log "Submitting: ${TRAIN_SCRIPT}"
    docker_exec "${HEAD_NODE}" \
        "cd ${MILES_DIR} && nohup env ${TRAIN_ENV} bash ${TRAIN_SCRIPT} submit > nohup_p2p_rdma.out 2>&1 &"
    log "Logs: ssh ${HEAD_NODE} 'docker exec ${CONTAINER_NAME} tail -f ${MILES_DIR}/nohup_p2p_rdma.out'"
}

stop_cluster() {
    log "Stopping cluster..."
    for host in "${NODES[@]}"; do
        docker_exec "$host" \
            "pkill -9 -f '${TRAIN_SCRIPT}' 2>/dev/null || true; pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 python; rm -rf /tmp/ray/" \
            2>/dev/null || true
    done
    log "Stopped."
}

cleanup_all() {
    stop_cluster
    for host in "${NODES[@]}"; do
        ssh_cmd "$host" "docker rm -f ${CONTAINER_NAME}" 2>/dev/null || true
    done
    log "Cleaned up."
}

show_status() {
    for host in "${NODES[@]}"; do
        echo "=== $host ==="
        if check_container "$host"; then
            docker_exec "$host" "ray status 2>/dev/null | head -8" || echo "  Ray not running"
        else
            echo "  Container not running"
        fi
    done
}

tail_logs() {
    ssh_cmd "${HEAD_NODE}" \
        "docker exec -it ${CONTAINER_NAME} tail -f ${MILES_DIR}/nohup_p2p_rdma.out"
}

full_start() {
    start_containers
    sleep 3
    install_packages
    sleep 2
    start_ray
    sleep 2
    submit_job
}

# ============== Main ==============
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
if [ $# -lt 1 ]; then
    echo "Usage: TRAIN_SCRIPT=scripts/run-xxx-p2p.sh $0 {start|stop|status|submit|logs|cleanup|containers|install|ray}"
    exit 1
fi

case "$1" in
    start)      full_start ;;
    stop)       stop_cluster ;;
    status)     show_status ;;
    submit)     submit_job ;;
    logs)       tail_logs ;;
    cleanup)    cleanup_all ;;
    containers) start_containers ;;
    install)    install_packages ;;
    ray)        start_ray ;;
    *)          echo "Unknown command: $1"; exit 1 ;;
esac
