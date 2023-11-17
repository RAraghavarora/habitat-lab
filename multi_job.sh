#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=logs.ddppo.out
#SBATCH --error=logs.ddppo.err
#SBATCH --gpus 2
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks-per-node 1
#SBATCH --mem=40GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=rrc
#SBATCH --reservation=rrc

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
source /home2/raghav.arora/miniconda3/etc/profile.d/conda.sh
conda activate habitat
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

set -x
# srun python -u -m habitat_baselines.run \
#     --config-name=rearrange/rl_rearrange_easy.yaml habitat_baselines.evaluate=True

srun python -u -m habitat_baselines.run --config-name=rearrange/rl_hierarchical.yaml