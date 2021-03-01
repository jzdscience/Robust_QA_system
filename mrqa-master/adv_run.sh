#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=adversarial_run.out
source ~/.bashrc
conda activate robustqa
echo "this is good"
python  main.py \
         --epochs 2 \
         --batch_size 16 \
         --lr 3e-5 \
         --do_lower_case \
         --use_cuda \
         --do_valid \
         --adv \
         --dis_lambda 0.01 
