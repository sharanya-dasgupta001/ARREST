
# python intervene_classifier.py --model_name llama2_7B --dataset_name truthfulqa --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama2_7B --dataset_name truthfulqa --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_iti.py --model_name llama2_7B --dataset_name truthfulqa --alpha1 15 --alpha2 0.5 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name truthfulqa --alpha 0.75 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama2_7B --dataset_name truthfulqa --alpha 0.75 --num_layers 1  --lambda_contrastive 0.5


# python get_activations.py --model_name llama2_7B --dataset_name truthfulqa
# python get_activations.py --model_name llama2_7B --dataset_name triviaqa
# python get_activations.py --model_name llama2_7B --dataset_name tydiaqa
# python get_activations.py --model_name llama2_7B --dataset_name haluevalqa
# python get_activations.py --model_name llama2_7B --dataset_name haluevaldia
# python get_activations.py --model_name llama2_7B --dataset_name haluevalsum
# python get_activations.py --model_name llama2_7B --dataset_name coqa

python BLEURT.py /home/iplab/LLM/mitigation_results/responses_hal