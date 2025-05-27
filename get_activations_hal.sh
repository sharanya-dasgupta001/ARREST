#!/bin/bash
model="${model:-llama2_7B}"
dataset="${dataset:-truthfulqa}"

echo
echo "=============================="
echo "Getting Activations"
echo "Model  : $model"
echo "Dataset: $dataset"
echo "=============================="
echo

python hallucination_activations.py --model_name "$model" --dataset_name "$dataset"
python hallucination_activations.py --model_name "$model" --dataset_name "$dataset" --with_answer