#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH -G 1
#SBATCH -n 2
#SBATCH --cpus-per-task=1
#SBATCH -J igib
#SBATCH --output=bob.out
#SBATCH --error=bob.err
#SBATCH --reservation=rrc
#SBATCH --mail-type=all
#SBATCH --mail-user=reepicheep_logs@protonmail.com
#SBATCH --mem-per-cpu=8GB

date
source ~/miniconda3/etc/profile.d/conda.sh
conda activate habitat
python try_gib.py
echo $?
