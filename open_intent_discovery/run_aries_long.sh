#!/bin/bash
#SBATCH --job-name=usnid     
#SBATCH --time=4-0         
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  
#SBATCH --output=TEXTOIR_USNID_%j.out 
#SBATCH --error=TEXTOIR_USNID_%j.err  
#SBATCH --qos=high

source ~/miniconda3/etc/profile.d/conda.sh
conda activate textoir

cd ~/nlp/TEXTOIR/open_intent_discovery || exit

mkdir -p logs

if [[ "$1" == "scibert" ]]; then
    echo "Iniciando treinamento com SCIBERT..."
    for cluster_num_factor in 2.0 3.0
    do
        for seed in 0 1 2
        do
            python -u run.py \
            --dataset aries \
            --method UnsupUSNID \
            --setting unsupervised \
            --known_cls_ratio 0 \
            --seed $seed \
            --train \
            --config_file_name UnsupUSNID \
            --cluster_num_factor $cluster_num_factor \
            --gpu_id 0 \
            --save_model \
            --save_results \
            --output_dir "./outputs/scibert_F${cluster_num_factor}_S${seed}/"
        done
    done

elif [[ "$1" == "specter2" ]]; then
        # echo "Iniciando treinamento com SPECTER..."
        # python -u run.py \
        # --dataset aries \
        # --method UnsupUSNID \
        # --setting unsupervised \
        # --known_cls_ratio 0 \
        # --seed 42 \
        # --train \
        # --config_file_name UnsupUSNID \
        # --cluster_num_factor 1.0 \
        # --gpu_id 0 \
        # --save_model \
        # --save_results \
        # --output_dir './outputs/'
        echo "NADA implementado ainda"
else
    echo "ERRO: Modelo inválido ou nenhum parâmetro passado."
    echo "Uso correto: ./run_aries.sh scibert"
fi
