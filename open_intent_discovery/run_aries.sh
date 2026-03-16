#!/bin/bash
#SBATCH --job-name=aries_usnid     
#SBATCH --time=1-0          
#SBATCH --gpus=1                   
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  
#SBATCH --output=TEXTOIRaries_%j.out 
#SBATCH --error=TEXTOIRaries_%j.err  
#SBATCH --qos=high

source ~/miniconda3/etc/profile.d/conda.sh
conda activate textoir

cd ~/nlp/TEXTOIR/open_intent_discovery || exit

mkdir -p logs

if [[ "$1" == "scibert" ]]; then
    echo "Iniciando treinamento com SCIBERT..."
    python -u run.py \
    --dataset aries \
    --method UnsupUSNID \
    --setting unsupervised \
    --known_cls_ratio 0 \
    --seed 42 \
    --train \
    --config_file_name UnsupUSNID \
    --cluster_num_factor 4.0 \
    --gpu_id 0 \
    --save_model \
    --save_results \
    --output_dir './outputs/'

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
