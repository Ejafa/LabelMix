#!/bin/bash
# GPU Monitor - View GPUs across all cluster nodes
# Usage:
#   ./gpu_monitor.sh           # One-shot nvidia-smi summary for all nodes
#   ./gpu_monitor.sh nvitop    # Run nvitop interactively on a specific node
#   ./gpu_monitor.sh nvitop 0  # Run nvitop on node 0 (launcher, local)
#   ./gpu_monitor.sh nvitop 1  # Run nvitop on worker-0
#   ./gpu_monitor.sh nvitop 2  # Run nvitop on worker-1

SSH_PORT=${JIZHI_SSH_PORT:-36000}
NVITOP_BIN="/apdcephfs_fsgm/share_303853033/ethangeng/konstantin-garbers/ggez/miniconda3/envs/labelmix/bin/nvitop"

# Parse NODE_IP_LIST: "28.12.129.140:8,28.12.25.40:8,28.12.130.213:8"
IFS=',' read -ra NODE_ENTRIES <<< "${NODE_IP_LIST}"
NODE_IPS=()
for entry in "${NODE_ENTRIES[@]}"; do
    ip="${entry%%:*}"
    NODE_IPS+=("$ip")
done

echo "========================================"
echo "  Cluster GPU Monitor"
echo "  Nodes: ${#NODE_IPS[@]}  |  GPUs/node: 8  |  GPU: H20"
echo "  SSH Port: ${SSH_PORT}"
echo "========================================"
echo ""

if [[ "$1" == "nvitop" ]]; then
    node_idx="${2:-0}"
    if [[ "$node_idx" -ge "${#NODE_IPS[@]}" ]]; then
        echo "Error: Node index $node_idx out of range (0-$((${#NODE_IPS[@]}-1)))"
        exit 1
    fi
    target_ip="${NODE_IPS[$node_idx]}"
    echo "Launching nvitop on node $node_idx ($target_ip)..."
    echo ""
    if [[ "$target_ip" == "$LOCAL_IP" ]]; then
        exec "$NVITOP_BIN" -m
    else
        exec ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" -t "$target_ip" "$NVITOP_BIN -m"
    fi
else
    # One-shot summary of all nodes
    for i in "${!NODE_IPS[@]}"; do
        ip="${NODE_IPS[$i]}"
        if [[ "$i" -eq 0 ]]; then
            role="launcher"
        else
            role="worker-$((i-1))"
        fi
        echo -e "\033[1;36m=== Node $i ($role) - $ip ===\033[0m"
        if [[ "$ip" == "$LOCAL_IP" ]]; then
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
        else
            ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" "$ip" \
                "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>&1 || echo "  [unreachable]"
        fi
        echo ""
    done
fi
