#!/bin/bash

#SBATCH --job-name=open_clip_exp
#SBATCH --output=./output.log
#SBATCH --error=./output.err
#SBATCH --partition=normal
#SBATCH --environment=open_clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=05:59:00

export MASTER_ADDR=$(hostname)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

export MIXTERA_SERVER_ADDR=$MASTER_ADDR
export MIXTERA_SERVER_DIR=/iopsstor/scratch/cscs/tkerimog/open_clip/mixtera_server
export MIXTERA_JOB_ID="mixtera_openclip_$(date +'%Y%m%d_%H%M%S')" 
export MIXTERA_SERVER_PORT=12345

export PYTHON_EXEC=/iopsstor/scratch/cscs/tkerimog/open_clip/open_clip_env/bin/python

$PYTHON_EXEC -u -m mixtera.network.server.entrypoint \
    $MIXTERA_SERVER_DIR \
    --host $MIXTERA_SERVER_ADDR\
    --port $MIXTERA_SERVER_PORT &

sleep 5

srun --accel-bind=gn -ul --container-writable --environment=open_clip bash -c "
\$PYTHON_EXEC -u /iopsstor/scratch/cscs/tkerimog/open_clip/open_clip-mixtera/src/open_clip_train/main.py \
--save-frequency 1 \
--model ViT-B-32 \
--epochs 5 \
--seed 0 \
--local-loss \
--gather-with-grad \
--train-data '/iopsstor/scratch/cscs/tkerimog/datasets/cc12m-wds/cc12m-train-{0000..2175}.tar' \
--train-num-samples 100000 \
--dataset-type mixtera_webdataset \
--batch-size 256 \
--workers 4 \
--report-to wandb
"


