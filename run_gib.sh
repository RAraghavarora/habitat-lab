#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH -G 4
#SBATCH -n 4
#SBATCH --cpus-per-task=1
#SBATCH -J igib
#SBATCH --output=bob.out
#SBATCH --error=bob.err
#SBATCH --account=rrc
#SBATCH --mail-type=all
#SBATCH --mail-user=reepicheep_logs@protonmail.com
#SBATCH --mem-per-cpu=8GB

module load u18/cuda/11.6 u18/cudnn/8.4.0-cuda-11.6

date
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
python try_gib.py
echo $?
