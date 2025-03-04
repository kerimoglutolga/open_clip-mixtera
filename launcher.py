#!/usr/bin/env python
import os
import json
import shutil
import subprocess
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # -------------------------------------------------------------------------
    # Step 1: Construct Job Name and Create Experiment Directory
    # -------------------------------------------------------------------------
    # Build a job name using run_name, model, mixture name, epochs, batch_size, and workers.
    job_name = f"{cfg.open_clip.run_name}_{cfg.open_clip.model}_{cfg.mixture.name}_ep{cfg.open_clip.epochs}_bs{cfg.open_clip.batch_size}_w{cfg.open_clip.workers}"
    print(f"Job name: {job_name}")

    # Create the experiment directory under experiments_base.
    experiments_dir = os.path.join(cfg.experiments_base, job_name)
    os.makedirs(experiments_dir, exist_ok=True)
    print(f"Created experiment directory: {experiments_dir}")

    # -------------------------------------------------------------------------
    # Step 2: Copy the Mixtera Server Directory into the Experiment Directory
    # -------------------------------------------------------------------------
    mixtera_server_src = cfg.mixtera.server_dir
    # Use the basename of the source directory for the destination folder name.
    mixtera_server_dest = os.path.join(experiments_dir, os.path.basename(mixtera_server_src.rstrip("/")))

    # Copy the mixtera server directory if it does not already exist.
    if not os.path.exists(mixtera_server_dest):
        print(f"Copying mixtera server directory from {mixtera_server_src} to {mixtera_server_dest} ...")
        shutil.copytree(mixtera_server_src, mixtera_server_dest)
        print("Mixtera server directory copied successfully.")

    # -------------------------------------------------------------------------
    # Step 3: Build the SLURM SBATCH Script
    # -------------------------------------------------------------------------
    # Dump mixture components as a JSON-loadable string.
    from omegaconf import OmegaConf
    mixture_dict = OmegaConf.to_container(cfg.mixture, resolve=True)
    mixture_json = json.dumps(mixture_dict)


    sbatch_script = f"""#!/bin/bash
#SBATCH --account=a-a09
#SBATCH --job-name=open_clip_exp
#SBATCH --output={experiments_dir}/output.log
#SBATCH --error={experiments_dir}/output.err
#SBATCH --partition=normal
#SBATCH --environment=open_clip
#SBATCH --nodes={cfg.slurm.nodes}
#SBATCH --ntasks-per-node={cfg.slurm.ntasks_per_node}
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time={cfg.slurm.time}

export MASTER_ADDR=$(hostname)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT={cfg.master_port}

export MIXTERA_SERVER_ADDR=$(hostname)
export MIXTERA_SERVER_DIR={cfg.mixtera.server_dir}
export MIXTERA_JOB_ID="{cfg.mixtera.job_id}"
export MIXTERA_SERVER_PORT={cfg.mixtera.server_port}

export PYTHON_EXEC={cfg.python_exec}

# Dump mixture components as JSON loadable string
export MIXTERA_MIXTURE='{mixture_json}'

$PYTHON_EXEC -u -m mixtera.network.server.entrypoint \\
    $MIXTERA_SERVER_DIR \\
    --host $MIXTERA_SERVER_ADDR\\
    --port $MIXTERA_SERVER_PORT &

sleep 5

srun --accel-bind=gn -ul --container-writable --environment=open_clip bash -c "
$PYTHON_EXEC -u /iopsstor/scratch/cscs/tkerimog/open_clip/open_clip-mixtera/src/open_clip_train/main.py \\
--save-frequency {cfg.open_clip.save_frequency} \\
--model {cfg.open_clip.model} \\
--epochs {cfg.open_clip.epochs} \\
--seed 0 \\
--local-loss \\
--gather-with-grad \\
--train-data '/iopsstor/scratch/cscs/tkerimog/datasets/cc12m-wds/cc12m-train-{{0000..2175}}.tar' \\
--train-num-samples {cfg.open_clip.num_samples} \\
--dataset-type mixtera_webdataset \\
--batch-size {cfg.open_clip.batch_size} \\
--workers {cfg.open_clip.workers} \\
--report-to {cfg.open_clip.report_to} \\
--logs {experiments_dir} \\
--save-most-recent \\
--force-patch-dropout {cfg.open_clip.force_patch_dropout} \\
--imagenet-val {cfg.open_clip["imagenet-val"]} \\
--name "{job_name}" \\
--wandb-project-name "{cfg.open_clip.wandb_project_name}" \\
"
"""
    # Write the sbatch script to a file inside the experiment directory.
    sbatch_file = os.path.join(experiments_dir, "job.sbatch")
    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)

    print("Generated sbatch script:")
    print(sbatch_script)

    # -------------------------------------------------------------------------
    # Step 4: Submit the Job with SLURM
    # -------------------------------------------------------------------------
    result = subprocess.run(["sbatch", sbatch_file], capture_output=True, text=True)
    print("SLURM submission output:")
    print(result.stdout)
    if result.stderr:
        print("SLURM submission errors:")
        print(result.stderr)


if __name__ == "__main__":
    main()
