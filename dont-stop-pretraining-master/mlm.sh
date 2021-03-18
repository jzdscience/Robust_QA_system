#!/bin/sh
#SBATCH -p G-1GPU-8Cpu-58GB
#SBATCH -N 1
#SBATCH --gres=gpu:1 
#SBATCH --mail-user=ju.zhang.jz1@roche.com 
#SBATCH --mail-type=ALL
#SBATCH --job-name="nlp_qa"
#SBATCH --output=mlm.out
source ~/.bashrc
conda activate robustqa

python -m scripts.train \
        --config training_config/classifier.jsonnet \
        --serialization_dir model_logs/citation_intent_base \
        --hyperparameters ROBERTA_CLASSIFIER_SMALL \
        --dataset citation_intent \
        --model roberta-base \
        --device 0 \
        --perf +f1 \
        --evaluate_on_test