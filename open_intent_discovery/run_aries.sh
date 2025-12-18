#!/bin/bash
#SBATCH --job-name=aries_usnid     
#SBATCH --time=24:00:00            
#SBATCH --gpus=2                   
#SBATCH --cpus-per-task=8          
#SBATCH --mem=32G                  
#SBATCH --output=logs/aries_%j.out 
#SBATCH --error=logs/aries_%j.err  

source ~/miniconda3/etc/profile.d/conda.sh
conda activate textoir

cd ~/TEXTOIR/open_intent_discovery

mkdir -p logs

python -u run.py \
    --dataset aries \
    --method UnsupUSNID \
    --setting unsupervised \
    --known_cls_ratio 0 \
    --seed 42 \
    --train \
    --config_file_name UnsupUSNID \
    --cluster_num_factor 1.0 \
    --gpu_id 0 \
    --save_results \
    --output_dir './outputs/'