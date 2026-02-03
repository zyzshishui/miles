#!/bin/bash
# Multi-node launcher for Qwen3-Coder-480B training
# Usage:
#   bash scripts/launch-multi-node.sh start    # Start all nodes
#   bash scripts/launch-multi-node.sh stop     # Stop all nodes
#   bash scripts/launch-multi-node.sh status   # Check status
#   bash scripts/launch-multi-node.sh submit   # Submit training job

set -euo pipefail

# ============== Configuration ==============
# Node list (first one is head node)
NODES=(
    "mia1-p02-g46"    # Head node (train)
    "mia1-p02-g23"
    "mia1-p02-g05"
    "mia1-p01-g43"
    "mia1-p02-g10"    
    "mia1-p02-g49"
)

# Docker configuration
DOCKER_IMAGE="rlsys/miles:MI350-355-latest"
CONTAINER_NAME="yuzhen_miles"
WORKSPACE_DIR="/workspace"
MILES_DIR="/workspace/miles"

# Training script
TRAIN_SCRIPT="scripts/run-qwen3-coder-480B-A35B-amd.sh"

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ssh_cmd() {
    local host=$1
    shift
    ssh ${SSH_OPTS} "$host" "$@"
}

# Execute command in docker container on remote host
docker_exec() {
    local host=$1
    shift
    ssh_cmd "$host" "docker exec ${CONTAINER_NAME} bash -c '$*'"
}

check_container() {
    local host=$1
    ssh_cmd "$host" "docker ps -q -f name=^${CONTAINER_NAME}\$" 2>/dev/null | grep -q .
}

# Get container IP
get_container_ip() {
    local host=$1
    docker_exec "$host" "hostname -I | cut -d\" \" -f1"
}

# ============== Start Docker Containers ==============
start_containers() {
    log "Starting Docker containers on all nodes..."
    
    for host in "${NODES[@]}"; do
        log "Checking $host..."
        
        if check_container "$host"; then
            log "  Container already running on $host"
        else
            log "  Starting container on $host..."
            ssh_cmd "$host" "docker run -itd --network=host --privileged --device=/dev/kfd --device=/dev/dri \
                --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
                --security-opt seccomp=unconfined \
                -v /it-share-2/data/yuzhzhou:/workspace \
                -w /workspace \
                -v /data/yuzhzhou/cache/huggingface:/root/.cache/huggingface \
                -v /data/yuzhzhou/cache/torch:/root/.cache/torch \
                -v /data/yuzhzhou/cache/pip:/root/.cache/pip \
                -e HF_HOME=/root/.cache/huggingface \
                -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
                -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
                -e WANDB_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
                -e WANDB_API_KEY='cd411df8b73eb3f5c1ae1220cc1ec4e3c9d1f86e' \
                --name ${CONTAINER_NAME} \
                ${DOCKER_IMAGE}" || log "  Warning: Container might already exist on $host"
        fi
    done
    
    log "All containers started."
}

# ============== Install Miles ==============
install_miles() {
    log "Installing miles on all nodes..."
    
    for host in "${NODES[@]}"; do
        log "Installing on $host..."
        docker_exec "$host" "cd ${MILES_DIR} && pip install -e . -q" &
    done
    
    wait
    log "Miles installed on all nodes."
}

# ============== Start Ray Cluster ==============
start_ray_cluster() {
    log "Starting Ray cluster..."
    
    local head_node="${NODES[0]}"
    local head_ip
    
    # Get head node IP
    head_ip=$(get_container_ip "$head_node")
    log "Head node: $head_node (IP: $head_ip)"
    
    # Cleanup and start head
    log "Starting Ray head on $head_node..."
    docker_exec "$head_node" "cd ${MILES_DIR} && bash ${TRAIN_SCRIPT} head" &
    local head_pid=$!
    
    # Wait for head to start
    sleep 10
    
    # Start workers
    for ((i=1; i<${#NODES[@]}; i++)); do
        local worker_node="${NODES[$i]}"
        log "Starting Ray worker on $worker_node..."
        docker_exec "$worker_node" "cd ${MILES_DIR} && MASTER_ADDR=${head_ip} bash ${TRAIN_SCRIPT} worker" &
    done
    
    log "Waiting for cluster to initialize..."
    sleep 15
    
    # Check cluster status
    log "Checking Ray cluster status..."
    docker_exec "$head_node" "ray status" || true
    
    log "Ray cluster started. Head IP: $head_ip"
    echo ""
    echo "=============================================="
    echo "Ray cluster is ready!"
    echo "Head node: $head_node ($head_ip)"
    echo "Worker nodes: ${NODES[@]:1}"
    echo ""
    echo "To submit training job, run:"
    echo "  bash scripts/launch-multi-node.sh submit"
    echo "=============================================="
}

# ============== Submit Job ==============
submit_job() {
    local head_node="${NODES[0]}"
    
    log "Submitting training job from $head_node..."
    docker_exec "$head_node" "cd ${MILES_DIR} && nohup bash ${TRAIN_SCRIPT} submit > nohup_disaggregated.out 2>&1 &"
    
    log "Job submitted. Check logs with:"
    echo "  ssh $head_node 'docker exec ${CONTAINER_NAME} tail -f ${MILES_DIR}/nohup_disaggregated.out'"
}

# ============== Stop Cluster ==============
stop_cluster() {
    log "Stopping Ray cluster on all nodes..."
    
    for host in "${NODES[@]}"; do
        log "Stopping Ray on $host..."
        docker_exec "$host" "pkill -9 sglang; ray stop --force; pkill -9 ray; pkill -9 python" 2>/dev/null || true
    done
    
    log "Ray cluster stopped."
}

# ============== Stop Containers ==============
stop_containers() {
    log "Stopping Docker containers on all nodes..."
    
    for host in "${NODES[@]}"; do
        log "Stopping container on $host..."
        ssh_cmd "$host" "docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}" 2>/dev/null || true
    done
    
    log "All containers stopped."
}

# ============== Status ==============
show_status() {
    log "Checking status of all nodes..."
    echo ""
    
    for host in "${NODES[@]}"; do
        echo "=== $host ==="
        if check_container "$host"; then
            echo "  Container: Running"
            local ip=$(get_container_ip "$host" 2>/dev/null || echo "unknown")
            echo "  IP: $ip"
            echo "  Ray status:"
            docker_exec "$host" "ray status 2>/dev/null | head -10" || echo "    Ray not running"
        else
            echo "  Container: Not running"
        fi
        echo ""
    done
}

# ============== Full Start ==============
full_start() {
    log "Starting full multi-node setup..."
    
    start_containers
    sleep 5
    
    install_miles
    sleep 3
    
    start_ray_cluster
}

# ============== Main ==============
show_usage() {
    echo "Usage: $0 {start|stop|status|submit|containers|install|ray|stop-ray}"
    echo ""
    echo "Commands:"
    echo "  start       - Full start: containers + install + ray cluster"
    echo "  stop        - Stop everything: ray + containers"
    echo "  status      - Show status of all nodes"
    echo "  submit      - Submit training job"
    echo ""
    echo "Advanced commands:"
    echo "  containers  - Start Docker containers only"
    echo "  install     - Install miles only"
    echo "  ray         - Start Ray cluster only"
    echo "  stop-ray    - Stop Ray cluster only (keep containers)"
}

if [ $# -lt 1 ]; then
    show_usage
    exit 1
fi

case "$1" in
    start)
        full_start
        ;;
    stop)
        stop_cluster
        stop_containers
        ;;
    status)
        show_status
        ;;
    submit)
        submit_job
        ;;
    containers)
        start_containers
        ;;
    install)
        install_miles
        ;;
    ray)
        start_ray_cluster
        ;;
    stop-ray)
        stop_cluster
        ;;
    *)
        echo "Error: Unknown command '$1'"
        show_usage
        exit 1
        ;;
esac
