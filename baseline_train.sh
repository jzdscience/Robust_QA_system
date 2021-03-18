#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=baseline_train.out
source ~/.bashrc
conda activate robustqa

python train.py --do-train --run-name baseline --visualize-predictions --pretrained-model save/mlm_train-05/checkpoint