#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=adv_finetuning.out
source ~/.bashrc
conda activate robustqa

python train.py --do-finetuning --run-name adv_finetuning --visualize-predictions --adv-training --save-dir save/adv_train-01-finetuning-01 --train-datasets race,relation_extraction,duorc --train-dir datasets/oodomain_train --val-dir datasets/oodomain_val --num-epochs 15 --eval-every 100 --freeze --lr 3e-6