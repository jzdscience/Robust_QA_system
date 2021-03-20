#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=baseline_finetuning.out
source ~/.bashrc
conda activate robustqa

python train.py --do-finetuning --run-name baseline_finetuning --visualize-predictions --save-dir save/baseline-02-finetuning-unfreeze --train-datasets race,relation_extraction,duorc --train-dir datasets/oodomain_train --val-dir datasets/oodomain_val --num-epochs 20 --eval-every 20 --lr 3e-6