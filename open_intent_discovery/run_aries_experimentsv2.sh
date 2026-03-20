#!/bin/bash
#SBATCH --job-name=usnid     
#SBATCH --time=7-0         
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
    for predict_k in 'density_usnid' 'silhouette_based'
    do
        for dataset in 'aries_high_masked'
        do
            for cluster_num_factor in 2.0 3.0 4.0
            do
                for seed in 0 1 2 3
                do
                    python -u run.py \
                    --predict_k $predict_k \
                    --dataset $dataset \
                    --method UnsupUSNID \
                    --setting unsupervised \
                    --known_cls_ratio 0 \
                    --seed $seed \
                    --train \
                    --config_file_name UnsupUSNID \
                    --cluster_num_factor $cluster_num_factor \
                    --backbone 'bert_USNID_Unsup' \
                    --gpu_id '1' \
                    --save_model \
                    --save_results \
                    --model_path "/home/caiosouza/nlp/models/scibert_scivocab_uncased" \
                    --output_dir "./outputs/scibert_F${cluster_num_factor}_S${seed}_dt${dataset}_kpred${predict_k}/"
                done
            done
        done
    done

elif [[ "$1" == "specter2" ]]; then
        echo "Iniciando treinamento com SPECTER2..."
        for predict_k in 'density_usnid' 'silhouette_based'
        do
            for dataset in 'aries_high_masked'
            do
                for cluster_num_factor in 2.0 3.0 4.0
                do
                    for seed in 0 1 2 3
                    do
                        python -u run.py \
                        --predict_k $predict_k \
                        --dataset $dataset \
                        --method UnsupUSNID \
                        --setting unsupervised \
                        --known_cls_ratio 0 \
                        --seed $seed \
                        --train \
                        --config_file_name UnsupUSNID \
                        --cluster_num_factor $cluster_num_factor \
                        --backbone 'bert_USNID_Unsup' \
                        --gpu_id '1' \
                        --save_model \
                        --save_results \
                        --model_path "/home/caiosouza/nlp/models/specter2_base" \
                        --output_dir "./outputs/specter2_F${cluster_num_factor}_S${seed}_dt${dataset}_kpred${predict_k}/"
                    done
                done
            done
        done
# "BAAI/bge-base-en-v1.5" experimental. Talvez quebre por ser um sbert. 
elif [[ "$1" == "bge" ]]; then
        echo "Iniciando treinamento com BGE-(SBERT)..."
        for predict_k in 'density_usnid' 'silhouette_based'
        do
            for dataset in 'aries_high_masked'
            do
                for cluster_num_factor in 2.0 3.0 4.0
                do
                    for seed in 0 1 2 3
                    do
                        python -u run.py \
                        --predict_k $predict_k \
                        --dataset $dataset \
                        --method UnsupUSNID \
                        --setting unsupervised \
                        --known_cls_ratio 0 \
                        --seed $seed \
                        --train \
                        --config_file_name UnsupUSNID \
                        --cluster_num_factor $cluster_num_factor \
                        --backbone 'sbert' \
                        --gpu_id '1' \
                        --save_model \
                        --save_results \
                        --model_path "/home/caiosouza/nlp/models/bge-base-en-v1.5" \
                        --output_dir "./outputs/bge_F${cluster_num_factor}_S${seed}_dt${dataset}_kpred${predict_k}/"
                    done
                done
            done
        done
else
    echo "ERRO: Modelo inválido ou nenhum parâmetro passado."
    echo "Uso correto: ./run_aries.sh scibert ou bge ou specter2"
fi


