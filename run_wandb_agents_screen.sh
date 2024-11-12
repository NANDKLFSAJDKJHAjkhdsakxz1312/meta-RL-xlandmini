#!/bin/bash

# configs
SWEEP_ID="jn1473733091/xminigrid/xd1puzxf" 
GPUS_TO_USE=( 4 5 6 7)  
AGENTS_PER_GPU=1  
VENV_PATH="/opt/anaconda3/envs/jax_xland" 


source ${VENV_PATH}/bin/activate

# start agent in screen
for GPU in "${GPUS_TO_USE[@]}"; do
  for (( i=0; i<AGENTS_PER_GPU; i++ )); do
   
    SCREEN_SESSION="wandb_agent_gpu${GPU}_agent${i}"

    echo "Starting wandb agent in screen session '$SCREEN_SESSION' on GPU $GPU (agent $i)"

   
    screen -dmS "$SCREEN_SESSION" bash -c "CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false wandb agent $SWEEP_ID"

    
    sleep 1
  done
done

echo "All wandb agents have been started in separate screen sessions."


echo "Use 'screen -ls' to list all screen sessions."
echo "To attach to a session, use 'screen -r SESSION_NAME'."
echo "To monitor GPU utilization, use 'nvidia-smi'."
echo "Use the wandb dashboard to check experiment progress."
