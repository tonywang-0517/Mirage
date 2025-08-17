#!/bin/bash
#SBATCH --job-name=Mirage-awan750
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/awan750/workspace/Mirage/sbatch_logs/%x-%j.out
#SBATCH --error=/data/awan750/workspace/Mirage/sbatch_logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=awan750@aucklanduni.ac.nz


# Conda
source /data/awan750/miniconda3/etc/profile.d/conda.sh
conda activate mirage

# Isaac Sim
export ISAACSIM_PATH="/data/awan750/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Avoid $HOME writes
export XDG_DATA_HOME=/data/awan750/.local/share
export XDG_CACHE_HOME=/data/awan750/.cache
export XDG_CONFIG_HOME=/data/awan750/.config

# IsaacLab
source /data/awan750/workspace/IsaacLab/_isaac_sim/setup_conda_env.sh

# Run training
cd /data/awan750/workspace/Mirage/
python mirage/train_agent.py +exp=full_body_tracker/transformer +robot=g1  +simulator=isaaclab  motion_file=data/yaml_files/train_g1.pt +experiment_name=full_body_tracker_g1_noDR ++headless=True
