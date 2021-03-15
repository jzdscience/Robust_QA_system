#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=adv_train.out
source ~/.bashrc
conda activate robustqa

python train.py --do-train --run-name adv_train --visualize-predictions --adv-training --dis-lambda 0.4