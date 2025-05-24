import torch
from einops import rearrange
import numpy as np
import pickle
import os, json
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import pyvene as pv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_NAMES = {
    # 'opt6.7B': "facebook/opt-6.7b", # base model
    'llama2_7B': "meta-llama/Llama-2-7b-hf", # base model
    'llama2_7B_chat' : "meta-llama/Llama-2-7b-chat-hf", # fine tuned
    
    'llama3_8B': "meta-llama/Llama-3.1-8B", # base model
    "llama3_8B_instruct": "meta-llama/Llama-3.1-8B-Instruct", # fine tuned
    
    'Qwen2.5_7B' : "Qwen/Qwen2.5-7B", # base model
    'Qwen2.5_7B_instruct' : "Qwen/Qwen2.5-7B-Instruct", # fine tuned
    
    'Yi1.5_9B' : "01-ai/Yi-1.5-9B", # base model
    "Yi1.5_9B_chat" : "01-ai/Yi-1.5-9B-Chat", # fine tuned
    
    'Ministral_8B' : "mistralai/Ministral-8B-Instruct-2410", # fine tuned
    'vicuna_7B' : "lmsys/vicuna-7b-v1.5" # fine tuned
}

def formatter(model_name, dataset_name, dataset, tokenizer): 
    
    
    if 'jailbreak-bench' in dataset_name or 'advbench' in dataset_name:
        base_prompt = "{prompt}. "
    else:                                                           
        base_prompt = "{prompt} "
        
    all_prompts = []
    

    # def format_prompt(text):
    #     if text.endswith("?"):
    #         return f"Q: {text} A:"
    #     elif text.endswith("."):
    #         return text
    #     elif text.endswith(" "):
    #         return f"{text.strip()}."
    #     else:
    #         return f"{text}."
    
    for index, row in enumerate(dataset):
        if dataset_name == 'sorry-Bench' :
            prompt = tokenizer(base_prompt.format(prompt=row['turns'][0]), return_tensors='pt')
        elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(base_prompt.format(prompt=row['prompt']), return_tensors='pt')
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(base_prompt.format(prompt=row['Goal']),return_tensors='pt')
            
        all_prompts.append(prompt)

        
    # for index, row in enumerate(dataset):
    #     if dataset_name == 'sorry-Bench' :
    #         prompt = tokenizer(format_prompt(row['turns'][0]), return_tensors='pt')
    #     elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
    #         prompt = tokenizer(format_prompt(row['prompt']), return_tensors='pt')
    #     elif dataset_name == 'jailbreak-bench':
    #         prompt = tokenizer(format_prompt(row['Goal']),return_tensors='pt')
            
    #     all_prompts.append(prompt)
        
    return all_prompts

def run_answers(model_name, dataset_name, dataset,  model=None,
                tokenizer=None, verbose=False, device=DEVICE, instruction_prompt="default"):
    frame = dataset.to_pandas()
    tokens = formatter(model_name, dataset_name, dataset, tokenizer)

    sequences = []
    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(tokens, desc="run_answers")):
            max_len = inputs.input_ids.shape[-1] + 64

            # --- intervention code --- #
            inputs = inputs.to(device)
            # _, output = model.generate(inputs, top_k=1, max_length=max_len, num_return_sequences=1,)
            output = model.generate(**inputs, top_k=1, max_length=max_len, num_return_sequences=1,)
            # output = model.generate(input_ids, top_k=1, max_length=max_len, num_return_sequences=1,)

            model_gen_tokens = output[:, inputs.input_ids.shape[-1]:]
            model_gen_str = tokenizer.decode(model_gen_tokens[0], skip_special_tokens=True)
            model_gen_str = model_gen_str.strip()

            # try: 
            #     # remove everything after 'Q:'
            #     model_gen_str = model_gen_str.split("Q:")[0].strip()
            #     # keep everything after A: 
            #     model_gen_str = model_gen_str.split("A:")[1].strip()
            # except: 
                # pass

            if verbose: 
                print("MODEL_OUTPUT: ", model_gen_str)
            
            frame.loc[idx, 'intervened_answers'] = model_gen_str
            sequences.append(model_gen_str)

            # --- intervention code --- #

    if device:
        torch.cuda.empty_cache()
        
    if dataset_name == 'sorry-Bench' :
        frame['turns'] = frame['turns'].apply(lambda x: x[0])
        frame = frame[['turns', 'intervened_answers']].rename(columns={'turns': 'prompt'})
    elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
        frame = frame[['prompt', 'intervened_answers']]
    elif dataset_name == 'jailbreak-bench':
        frame = frame[['Goal', 'intervened_answers']].rename(columns={'Goal': 'prompt'})
        
    return frame

def alt_evaluate(model_name,dataset_name, models, output_path, device=DEVICE,
                    verbose=False, instruction_prompt="default"): 
    
    if dataset_name == 'sorry-Bench' :
        questions  = load_dataset("sorry-bench/sorry-bench-202503")['train']
    elif dataset_name == 'over-refusal':
        questions = load_dataset("bench-llm/or-bench", "or-bench-80k")['train']
    elif dataset_name == 'malicious-instruct':
        questions = load_dataset("walledai/MaliciousInstruct")['train']
    elif dataset_name == 'advbench':
        questions  =  load_dataset("walledai/AdvBench")['train']
    elif dataset_name == 'jailbreak-bench':
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors") 
        # Please pick one among the available configs: ['behaviors', 'judge_comparison'] 
        questions  =  ds['harmful'] # dict_keys(['harmful', 'benign'])
    elif dataset_name == 'trustllm':
        questions = load_trust_dataset()
    else: 
        raise ValueError("Invalid dataset name")

    
    for mdl in models.keys(): 

        llama_model = models[mdl]
        llama_tokenizer = AutoTokenizer.from_pretrained(HF_NAMES[mdl])
        questions = run_answers(model_name, dataset_name, questions, model=llama_model, tokenizer=llama_tokenizer,
                        device=device, verbose=verbose,
                        instruction_prompt=instruction_prompt)

        questions.to_csv(output_path)



def load_trust_dataset():
    """
    Downloads and processes the TrustLLM dataset.
    Returns:
        Dataset: The processed TrustLLM dataset.
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
    
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default=None, help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--instruction_prompt', default='default', help='instruction prompt for truthfulqa benchmarking, "default" or "informative"', type=str, required=False)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset_name == 'sorry-Bench' :
        dataset  = load_dataset("sorry-bench/sorry-bench-202503")['train']
    elif args.dataset_name == 'over-refusal':
        dataset = load_dataset("bench-llm/or-bench", "or-bench-80k")['train']
    elif args.dataset_name == 'malicious-instruct':
        dataset = load_dataset("walledai/MaliciousInstruct")['train']
    elif args.dataset_name == 'advbench':
        dataset  =  load_dataset("walledai/AdvBench")['train']
    elif args.dataset_name == 'jailbreak-bench':
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors") 
        # Please pick one among the available configs: ['behaviors', 'judge_comparison'] 
        dataset  =  ds['harmful'] # dict_keys(['harmful', 'benign'])
    elif args.dataset_name == 'trustllm':
        dataset = load_trust_dataset()
    else: 
        raise ValueError("Invalid dataset name")
    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(dataset)), args.num_fold)

    # create model
    MODEL = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="/home/iplab/LLM/models/",
        attn_implementation="eager").to(DEVICE)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id


    
        
    filename = f'{args.dataset_name}_{args.model_name}'#_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}'

    if args.use_center_of_mass:
        filename += '_com'
    if args.use_random_dir:
        filename += '_random'
        
                            
    alt_evaluate(
        args.model_name,
        args.dataset_name,
        models={args.model_name: model},#intervened_model},
        output_path=f'/home/iplab/LLM/mitigation_results/responses/{filename}.csv',
        device=DEVICE, 
        instruction_prompt=args.instruction_prompt
    )



if __name__ == "__main__":
    main()