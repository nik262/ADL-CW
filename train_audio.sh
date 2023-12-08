#!/usr/bin/env bash

#SBATCH --account=COMS030144
#SBATCH --job-name=ext-3
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:180:00
#SBATCH --mem=16GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python train_audio.py --learning-rate 0.05 --momentum 0.94 --epochs 20 --extension False
