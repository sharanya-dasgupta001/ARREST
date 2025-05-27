import os,json
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import pyvene as pv
import re
from utils import Collector

hf_token = "<INPUT_YOUR_HF_ACCESS_TOKEN>"

# Example refusal and non-refusal responses for contrastive suffix generation
refusal_responses = ["I'm sorry, but I can't help with that request",
                        "That’s not something I’m able to do",
                        "I can’t assist with that",
                        "I’m unable to help with that",
                        "For safety and ethical reasons, I’m not able to provide that information",
                        "I can't help with that" ,
                        "I’m afraid I can’t assist with that",
                        "That’s not something I can help with",
                        "Unfortunately, I can't fulfill that request",
                        "I have to decline to provide that request",
                        "I refuse to provide that kind of content",
                        "I refuse to answer that question",
                        "I must politely decline to fulfill that request",
                        "I must decline your request",
                        "I'm afraid I must decline your request"]
non_refusal_responses = ["Sure, here's what I found",
                            "Here's the information you requested",
                            "Let me explain:",
                            "Here's what I found",
                            "Let me help with that",
                            "That's a good question. Here's the answer",
                            "Let's take a look at that",
                            "To put it simply",
                            "What you’re asking about",
                            "Let’s explore that together"]

# Setup device for torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dictionary of model aliases to HuggingFace model IDs
MODELS_NAMES = {
    'llama2_7B': "meta-llama/Llama-2-7b-hf", 
    'llama2_7B_chat' : "meta-llama/Llama-2-7b-chat-hf", 
    
    'llama3_8B': "meta-llama/Llama-3.1-8B", 
    "llama3_8B_instruct": "meta-llama/Llama-3.1-8B-Instruct", 
    
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B", 
    'Qwen2.5_7B_instruct' : "Qwen/Qwen2.5-7B-Instruct", 
    
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B", 
    "Yi1.5_9B_chat" : "01-ai/Yi-1.5-9B-Chat", 
    }

def load_trust_dataset():
    """
    Load and parse the TrustLLM safety dataset from Hugging Face Hub.
    """
    from huggingface_hub import hf_hub_download
    import shutil
    save_path = './llm_trust_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"llm_trust_dataset/misuse.json"):
        try:
            file_path = hf_hub_download(repo_id="TrustLLM/TrustLLM-dataset", filename="safety/misuse.json", repo_type="dataset")
            shutil.copy(file_path, os.path.join(os.getcwd(), "llm_trust_dataset/misuse.json"))
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    with open("llm_trust_dataset/misuse.json", "r") as file:
        data = json.load(file)  
        return Dataset.from_list(data)
    
def wrapper(intervener):
    """
    A wrapper for Pyvene intervention functions.
    """
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped
    
    
def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    """
    Runs a forward pass and extracts hidden states from specified layers.
    """
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": False})[1]
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return head_wise_hidden_states

def formatter(model_name, dataset_name, dataset, tokenizer, suffix=None): 
    """
    Formats dataset prompts for model input, with optional suffixes for contrastive learning.
    """
    all_prompts = []
    if suffix is not None:
        base_prompts = {'jailbreak-bench' : "{prompt}. {additional}",
        'advbench' : "{prompt}. {additional}"}
        base_prompt = base_prompts.get(dataset_name, "{prompt} {additional}")
        for index, row in enumerate(dataset):
            if dataset_name in ['malicious-instruct','advbench','trustllm']:
                prompt = tokenizer(base_prompt.format(prompt=row['prompt'],
                                                                    additional = random.choice(suffix)), return_tensors='pt').input_ids
            elif dataset_name == 'jailbreak-bench':
                prompt = tokenizer(base_prompt.format(prompt=row['Goal'],
                                                                    additional = random.choice(suffix)), return_tensors='pt').input_ids
                
            all_prompts.append(prompt)
    
    else :
        base_prompts = {
        'jailbreak-bench' : "{prompt}.",
        'advbench' : "{prompt}."}
        base_prompt = base_prompts.get(dataset_name, "{prompt}")
    
        if model_name in  ['llama2_7B_chat']:
            def format_prompt(user_input):
                return f"[INST] {user_input} [/INST]"
        elif model_name in ['Yi1.5_9B_chat']:
            def format_prompt(user_input):
                return ( f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        f"<|im_start|>user\n{user_input}<|im_end|>\n"
                        f"<|im_start|>assistant\n")
        else :
            def format_prompt(user_input):
                return user_input
            
        for index, row in enumerate(dataset):
            if dataset_name in ['malicious-instruct','advbench','trustllm']:
                prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['prompt'])), return_tensors='pt').input_ids
            elif dataset_name == 'jailbreak-bench':
                prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['Goal'])),return_tensors='pt').input_ids
                
            all_prompts.append(prompt)
        
    return all_prompts

def main(): 
    """
    Main entry point: loads model/dataset, collects and saves layer-wise activations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B', choices=list(MODELS_NAMES.keys()), help='Name of the model to use')
    parser.add_argument('--dataset_name', type=str, default='malicious-instruct', choices=['malicious-instruct', 'advbench', 'jailbreak-bench', 'trustllm'], help='Name of the dataset to use')
    parser.add_argument('--contrastive', type=str, default='neutral', choices=['neutral', 'positive', 'negative'], help='Type of dataset to use')
    args = parser.parse_args()

    MODEL = MODELS_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        use_auth_token=hf_token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="./models",
        attn_implementation="eager").to(DEVICE)

    
    # Load datasets
    if args.dataset_name == 'malicious-instruct':
        dataset = load_dataset("walledai/MaliciousInstruct")['train']
    elif args.dataset_name == 'advbench':
        dataset  =  load_dataset("walledai/AdvBench")['train']
    elif args.dataset_name == 'jailbreak-bench':
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors") 
        dataset  =  ds['harmful'] 
    elif args.dataset_name == 'trustllm':
        dataset = load_trust_dataset()
    else: 
        raise ValueError("Invalid dataset name")

    print("\nTokenizing prompts\n")

    # Prepare prompts
    if args.contrastive == 'neutral':
        prompts = formatter(args.model_name, args.dataset_name, dataset, tokenizer)
    else:
        if args.contrastive == 'positive':
            suffix = refusal_responses
        elif args.contrastive == 'negative':
            suffix = non_refusal_responses
        else:
            raise ValueError("Invalid contrastive type")
        prompts = formatter(args.model_name, args.dataset_name, dataset, tokenizer, suffix)
        
    # Create collectors and pyvene config
    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1)
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)
    all_head_wise_activations = []
    tqdm.write('')
    for _, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Getting activations for {args.dataset_name} dataset ..."):
        head_wise_activations = get_llama_activations_pyvene(collected_model, collectors, prompt, DEVICE)
        all_head_wise_activations.append(head_wise_activations.copy())

    # Save collected activations
    os.makedirs(f'./safety/hidden/', exist_ok=True)
    print("\nSaving head wise activations")
    if args.contrastive == 'neutral':
        if 'chat' in args.model_name or 'instruct' in args.model_name:
            np.save(f'./safety/hidden/{re.sub(r"(_chat|_instruct)", "", args.model_name)}_aligned_{args.dataset_name}_head_wise.npy', all_head_wise_activations)
        else:
            np.save(f'./safety/hidden/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)
    else:
        np.save(f'./safety/hidden/{args.model_name}_{args.dataset_name}_head_wise_{args.contrastive}.npy', all_head_wise_activations)
    print("\n__________________________________________________________________________________________\n")

if __name__ == '__main__':
    main()
    