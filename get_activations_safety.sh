#!/bin/bash
model="${model:-"llama2_7B"}"
dataset="${dataset:-"malicious-instruct"}"

echo
echo "=============================="
echo "Getting Activations"
echo "Model  : $model"
echo "Dataset: $dataset"
echo "=============================="
echo

if [ "$model" = "llama2_7B" ]; then
    alignedmodel="llama2_7B_chat"
elif [ "$model" = "llama3_8B" ]; then
    alignedmodel="llama3_8B_instruct"
elif [ "$model" = "Qwen2.5_7B" ]; then
    alignedmodel="Qwen2.5_7B_instruct"
elif [ "$model" = "Yi1.5_9B" ]; then
    alignedmodel="Yi1.5_9B_chat"
else
    echo "Model not recognized. Please set the model variable to a valid model name."
    exit 1
fi

run_safety_activations() {
    python safety_activations.py --model_name "$1" --dataset_name "$dataset" $2
}
run_safety_activations "$model"
run_safety_activations "$model" "--contrastive positive"
run_safety_activations "$model" "--contrastive negative"
run_safety_activations "$alignedmodel"

