#!/bin/bash

# 参数配置
SWEEP_ID="jn1473733091/xminigrid/1fq9xwfw"  # 替换为你的实际 sweep_id
GPUS_TO_USE=(  2 5 6 7)  # 指定要使用的 GPU 编号
AGENTS_PER_GPU=1  # 每个 GPU 上运行的 wandb agent 数量
VENV_PATH="/opt/anaconda3/envs/jax_xland"  # 虚拟环境路径

# 激活虚拟环境
source ${VENV_PATH}/bin/activate

# 循环启动 wandb agent，并在每个 screen 会话中运行
for GPU in "${GPUS_TO_USE[@]}"; do
  for (( i=0; i<AGENTS_PER_GPU; i++ )); do
    # 定义 screen 会话名称
    SCREEN_SESSION="wandb_agent_gpu${GPU}_agent${i}"

    echo "Starting wandb agent in screen session '$SCREEN_SESSION' on GPU $GPU (agent $i)"

    # 启动新的 screen 会话并运行 wandb agent
    screen -dmS "$SCREEN_SESSION" bash -c "CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false wandb agent $SWEEP_ID"

    # 等待一小段时间以避免同时启动多个进程
    sleep 1
  done
done

echo "All wandb agents have been started in separate screen sessions."

# 提示用户查看 screen 状态
echo "Use 'screen -ls' to list all screen sessions."
echo "To attach to a session, use 'screen -r SESSION_NAME'."
echo "To monitor GPU utilization, use 'nvidia-smi'."
echo "Use the wandb dashboard to check experiment progress."
