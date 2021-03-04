#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=adv_eval.out
source ~/.bashrc
conda activate robustqa

python train.py --do-eval --run-name adv_eval --visualize-predictions --adv-training --eval-dir datasets/oodomain_val --save-dir save/adv_train-03