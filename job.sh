#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -n 10
#SBATCH -J rl_skill
#SBATCH --output=plot.out
#SBATCH --error=plot.err
#SBATCH --mail-type=all
#SBATCH --mail-user=reepicheep_logs@protonmail.com
#SBATCH --mem-per-cpu=2048
#SBATCH --reservation=rrc


source /home2/raghav.arora/miniconda3/etc/profile.d/conda.sh
conda activate habitat
python -u -m habitat_baselines.run --config-name=rearrange/rl_skill.yaml 
echo $?
