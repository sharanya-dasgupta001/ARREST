
# ################################## GET ACTIVATIONS ################################################
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name malicious-instruct --reject yes
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name advbench --reject yes
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --reject yes
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name trustllm --reject yes

# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name malicious-instruct --reject no
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name advbench --reject no
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --reject no
# python get_activationsV2.py --model_name Yi1.5_9B --dataset_name trustllm --reject no
# 
# python get_activations.py --model_name llama2_7B --dataset_name triviaqa

# python get_activations.py --model_name llama2_7B --dataset_name tydiaqa
# python get_activations.py --model_name llama2_7B --dataset_name haluevalqa
# python get_activations.py --model_name llama2_7B --dataset_name haluevaldia
# python get_activations.py --model_name llama2_7B --dataset_name haluevalsum
# python get_activations.py --model_name llama2_7B --dataset_name coqa
# python get_activations.py --model_name llama2_7B --dataset_name sorry-Bench

# python get_activations.py --model_name Yi1.5_9B --dataset_name malicious-instruct
# python get_activations.py --model_name Yi1.5_9B --dataset_name advbench
# python get_activations.py --model_name Yi1.5_9B --dataset_name jailbreak-bench
# python get_activations.py --model_name Yi1.5_9B --dataset_name trustllm

# python get_activations.py --model_name Yi1.5_9B_chat --dataset_name malicious-instruct
# python get_activations.py --model_name Yi1.5_9B_chat --dataset_name advbench
# python get_activations.py --model_name Yi1.5_9B_chat --dataset_name jailbreak-bench
# python get_activations.py --model_name Yi1.5_9B_chat --dataset_name trustllm



################################## GET ANSWERS ################################################
# python get_answers.py --model_name llama3_8B --dataset_name malicious-instruct 
# python get_answers.py --model_name llama3_8B --dataset_name jailbreak-bench
# python get_answers.py --model_name llama3_8B --dataset_name advbench
# python get_answers.py --model_name llama3_8B --dataset_name trustllm

# python get_answers.py --model_name llama3_8B_instruct --dataset_name malicious-instruct 
# python get_answers.py --model_name llama3_8B_instruct --dataset_name jailbreak-bench
# python get_answers.py --model_name llama3_8B_instruct --dataset_name advbench
# python get_answers.py --model_name llama3_8B_instruct --dataset_name trustllm

# python get_answers.py --model_name vicuna_7B --dataset_name malicious-instruct 
# python get_answers.py --model_name vicuna_7B --dataset_name jailbreak-bench
# python get_answers.py --model_name vicuna_7B --dataset_name advbench
# python get_answers.py --model_name vicuna_7B --dataset_name trustllm

# python get_answers.py --model_name Ministral_8B --dataset_name malicious-instruct 
# python get_answers.py --model_name Ministral_8B --dataset_name jailbreak-bench
# python get_answers.py --model_name Ministral_8B --dataset_name advbench
# python get_answers.py --model_name Ministral_8B --dataset_name trustllm

# python get_answers.py --model_name Qwen2.5_7B --dataset_name malicious-instruct 
# python get_answers.py --model_name Qwen2.5_7B --dataset_name jailbreak-bench
# python get_answers.py --model_name Qwen2.5_7B --dataset_name advbench
# python get_answers.py --model_name Qwen2.5_7B --dataset_name trustllm

# python get_answers.py --model_name Yi1.5_9B --dataset_name malicious-instruct 
# python get_answers.py --model_name Yi1.5_9B --dataset_name jailbreak-bench
# python get_answers.py --model_name Yi1.5_9B --dataset_name advbench
# python get_answers.py --model_name Yi1.5_9B --dataset_name trustllm

# ################################## GET ACTIVATIONS ################################################
# python get_activationsV2.py --model_name llama3_8B --dataset_name malicious-instruct --reject yes
# python get_activationsV2.py --model_name llama3_8B --dataset_name advbench --reject yes
# python get_activationsV2.py --model_name llama3_8B --dataset_name jailbreak-bench --reject yes
# python get_activationsV2.py --model_name llama3_8B --dataset_name trustllm --reject yes

# python get_activationsV2.py --model_name llama3_8B --dataset_name malicious-instruct --reject no
# python get_activationsV2.py --model_name llama3_8B --dataset_name advbench --reject no
# python get_activationsV2.py --model_name llama3_8B --dataset_name jailbreak-bench --reject no
# python get_activationsV2.py --model_name llama3_8B --dataset_name trustllm --reject no

# python get_activations.py --model_name llama2_7B --dataset_name over-refusal
# python get_activations.py --model_name llama2_7B --dataset_name sorry-Bench

# python get_activations.py --model_name llama3_8B --dataset_name malicious-instruct
# python get_activations.py --model_name llama3_8B --dataset_name advbench
# python get_activations.py --model_name llama3_8B --dataset_name jailbreak-bench
# python get_activations.py --model_name llama3_8B --dataset_name trustllm
# python get_activations.py --model_name llama3_8B_instruct --dataset_name over-refusal
# python get_activations.py --model_name llama3_8B_instruct --dataset_name sorry-Bench
################################## ORIGINAL ITI ################################################

# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 5 


# python intervene_new_iti.py --model_name llama3_8B --dataset_name malicious-instruct --alpha 5 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name jailbreak-bench --alpha 5 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name malicious-instruct --alpha 10 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name jailbreak-bench --alpha 10 --num_fold 1


# python intervene_new_iti.py --model_name llama3_8B --dataset_name malicious-instruct --alpha 15 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name jailbreak-bench --alpha 15 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name malicious-instruct --alpha 20 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name jailbreak-bench --alpha 20 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name malicious-instruct --alpha 50 --num_fold 1
# python intervene_new_iti.py --model_name llama3_8B --dataset_name jailbreak-bench --alpha 50 --num_fold 1


# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 100
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 15 --num_heads 100
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 200
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 15 --num_heads 200
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 24 --num_fold 1


# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 20 --num_heads 10 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 20 --num_heads 10 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 20 --num_heads 24 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 20 --num_heads 24 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 20 --num_heads 96 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 20 --num_heads 96 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 20 --num_heads 192 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 20 --num_heads 192 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 20 --num_heads 384 --num_fold 1
# python intervene_new_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 20 --num_heads 384 --num_fold 1
# python discriminator.py /home/iplab/LLM/mitigation_results/responses
### Best params : alpha = 15 , num_heads = 10-24


################################## ADVERSARIAL GEN ################################################

# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 0
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 10
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-10
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 1 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.75 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 0.75 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.5 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 0.5 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 5 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 5 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# ython intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 15 --num_heads 1 --batch_size 16 --num_epochs 50 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_layer.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 1 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_layer.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_heads 2 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
#### Best params : alpha = 1 , num_heads = 1 , lambda_recon = 1e-5

################################## ADVERSARIAL GEN&DISC ################################################

# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.5 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.75 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.25 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.5 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.75 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.25 --num_heads 1 --batch_size 16 --num_epochs 100 --lr 1e-5 --lambda_recon 1e-5 

#### Best params :

############################ ADVERSARIAL + ITI ###########################

# python intervene_generator_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 24
# python intervene_generator_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 15 --num_heads 24 
# python intervene_classifier.py --model_name llama3_8B --dataset_name advbench --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama3_8B --dataset_name trustllm --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama3_8B --dataset_name trustllm --alpha 0.25 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama3_8B --dataset_name trustllm --alpha 0.75 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama3_8B --dataset_name trustllm --alpha 1.0 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial.py --model_name llama3_8B --dataset_name trustllm --alpha 1.25 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_iti.py --model_name llama3_8B --dataset_name advbench --alpha1 15 --alpha2 0.5 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama3_8B --dataset_name advbench --alpha 0.75 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama3_8B --dataset_name advbench --alpha 0.75 --num_layers 1  --lambda_contrastive 0.5

# python discriminator.py /home/iplab/LLM/mitigation_results/responses

# python intervene_classifier.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 1 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_iti.py --model_name llama2_7B --dataset_name malicious-instruct --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama2_7B --dataset_name malicious-instruct --alpha 0.5 --num_layers 1  --lambda_contrastive 0.5

# python intervene_classifier.py --model_name llama2_7B --dataset_name advbench --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama2_7B --dataset_name advbench --alpha 1 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_iti.py --model_name llama2_7B --dataset_name advbench --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name advbench --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama2_7B --dataset_name advbench --alpha 0.5 --num_layers 1  --lambda_contrastive 0.5

# python intervene_classifier.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 1 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_iti.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama2_7B --dataset_name jailbreak-bench --alpha 0.5 --num_layers 1  --lambda_contrastive 0.5


# python intervene_classifier.py --model_name llama2_7B --dataset_name trustllm --alpha 15 --num_heads 24
# python intervene_adversarial.py --model_name llama2_7B --dataset_name trustllm --alpha 1 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_iti.py --model_name llama2_7B --dataset_name trustllm --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_gen_disc.py --model_name llama2_7B --dataset_name trustllm --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
# python intervene_adversarial_contrastive.py --model_name llama2_7B --dataset_name trustllm --alpha 0.5 --num_layers 1  --lambda_contrastive 0.5



# python intervene_no.py --model_name llama3_8B --dataset_name malicious-instruct
# python intervene_no.py --model_name llama3_8B --dataset_name advbench
# python intervene_no.py --model_name llama3_8B --dataset_name jailbreak-bench
# python intervene_no.py --model_name llama3_8B --dataset_name trustllm







python intervene_classifier.py --model_name Yi1.5_9B --dataset_name malicious-instruct --alpha 15 --num_heads 24
python intervene_adversarial.py --model_name Yi1.5_9B --dataset_name malicious-instruct --alpha 1 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_iti.py --model_name Yi1.5_9B --dataset_name malicious-instruct --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_gen_disc.py --model_name Yi1.5_9B --dataset_name malicious-instruct --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_contrastive.py --model_name Yi1.5_9B --dataset_name malicious-instruct --alpha 1 --num_layers 1  --lambda_contrastive 0.5

python intervene_classifier.py --model_name Yi1.5_9B --dataset_name advbench --alpha 15 --num_heads 24
python intervene_adversarial.py --model_name Yi1.5_9B --dataset_name advbench --alpha 1 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_iti.py --model_name Yi1.5_9B --dataset_name advbench --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_gen_disc.py --model_name Yi1.5_9B --dataset_name advbench --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_contrastive.py --model_name Yi1.5_9B --dataset_name advbench --alpha 1 --num_layers 1  --lambda_contrastive 0.5

python intervene_classifier.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --alpha 15 --num_heads 24
python intervene_adversarial.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --alpha 1 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_iti.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_gen_disc.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_contrastive.py --model_name Yi1.5_9B --dataset_name jailbreak-bench --alpha 1 --num_layers 1  --lambda_contrastive 0.5


python intervene_classifier.py --model_name Yi1.5_9B --dataset_name trustllm --alpha 15 --num_heads 24
python intervene_adversarial.py --model_name Yi1.5_9B --dataset_name trustllm --alpha 1 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_iti.py --model_name Yi1.5_9B --dataset_name trustllm --alpha1 15 --alpha2 1 --num_heads 24 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_gen_disc.py --model_name Yi1.5_9B --dataset_name trustllm --alpha 0.5 --num_layers 1 --lambda_recon 1e-5 
python intervene_adversarial_contrastive.py --model_name Yi1.5_9B --dataset_name trustllm --alpha 1 --num_layers 1  --lambda_contrastive 0.5