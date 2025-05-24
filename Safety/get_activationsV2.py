# Pyvene method of getting activations
import os,json, random
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
import pyvene as pv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_NAMES = {
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

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].to(torch.float32).detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].to(torch.float32).reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return b
    
    
def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
    hidden_states = hidden_states.to(torch.float32).detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


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

def formatter(reject, dataset_name, dataset, tokenizer, suffix): 
    
    # base_prompt = "{prompt} {additional}"
    base_prompts = {
        'truthfulqa': "Answer the question concisely. Q: {question} A:",
        'triviaqa': "Answer the question concisely. Q: {question} A:",
        'tydiqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'coqa': "Answer the question concisely based on the context: \n {story} \n Q: {question} A:",
        'haluevaldia': "You are an assistant that answers questions concisely and accurately. Use the knowledge and conversation to respond naturally to the most recent message.\nKnowledge: {knowledge}.\nConversations: {dialogue_history}.\nYour Response:",
        'haluevalqa': "Answer the question concisely based on the context: \n {context} \n Q: {question} A:",
        'haluevalsum': "{document} \n Please summarize the above article concisely. A:",
        'jailbreak-bench' : "{prompt}.",
        'advbench' : "{prompt}.",
    }

    base_prompt = base_prompts.get(dataset_name, "{prompt}")
    all_prompts = []
    
    # def format_prompt(text, additional):
    #     if text.endswith("?"):
    #         return f"Q: {text} A: {additional}"
    #     elif text.endswith("."):
    #         return f"{text} {additional}"
    #     elif text.endswith(" "):
    #         return f"{text.strip()}. {additional}"
    #     else:
    #         return f"{text}. {additional}"
        
    for index, row in enumerate(dataset):
        if dataset_name == 'sorry-Bench' :
            prompt = tokenizer(base_prompt.format(prompt=row['turns'][0],
                                                                additional = random.choice(suffix)), return_tensors='pt').input_ids
        elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(base_prompt.format(prompt=row['prompt'],
                                                                additional = random.choice(suffix)), return_tensors='pt').input_ids
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(base_prompt.format(prompt=row['Goal'],
                                                                additional = random.choice(suffix)), return_tensors='pt').input_ids
            
        all_prompts.append(prompt)
    
    # for index, row in enumerate(dataset):
    #     if dataset_name == 'sorry-Bench' :
    #         prompt = tokenizer(format_prompt(row['turns'][0],additional = random.choice(suffix)), return_tensors='pt').input_ids
    #     elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
    #         prompt = tokenizer(format_prompt(row['prompt'],additional = random.choice(suffix)), return_tensors='pt').input_ids
    #     elif dataset_name == 'jailbreak-bench':
    #         prompt = tokenizer(format_prompt(row['Goal'],additional = random.choice(suffix)), return_tensors='pt').input_ids
            
    #     all_prompts.append(prompt)
    
    if reject == 'yes':
        all_labels = torch.ones(len(all_prompts))
    else :
        all_labels = torch.zeros(len(all_prompts))
        
    return all_prompts, all_labels

def main(): 


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--reject', type=str, default='yes')
    args = parser.parse_args()

    MODEL = MODELS_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        cache_dir="/home/iplab/LLM/models/",
        attn_implementation="eager").to(DEVICE)

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
                        "I'm afraid I must decline your request",
                        
                        ]
    non_refusal_responses = ["Sure, here's what I found",
                            "Here's the information you requested",
                            "Let me explain:",
                            "Here's what I found",
                            "Let me help with that",
                            "That's a good question. Here's the answer",
                            "Let's take a look at that",
                            "To put it simply",
                            "What you’re asking about",
                            "Let’s explore that together",
                            
    ]

    if args.reject == 'yes':
        suffix = refusal_responses
    else:
        suffix = non_refusal_responses

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

    print("Tokenizing prompts")

    prompts, labels = formatter(args.reject, args.dataset_name, dataset, tokenizer, suffix)


    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc=f"Getting activations for {args.dataset_name} dataset ..."):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, DEVICE)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations.copy())

            
    os.makedirs(f'/home/iplab/LLM/mitigation_results/{args.model_name}/', exist_ok=True)
    # print("Saving labels")
    # np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_reject_{args.reject}_labels.npy', labels)

    # print("Saving layer wise activations")
    # np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_reject_{args.reject}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_reject_{args.reject}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()