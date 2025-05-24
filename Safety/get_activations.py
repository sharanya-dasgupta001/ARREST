# Pyvene method of getting activations
import os,json
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

def load_coqa_dataset():
    """
    Downloads and processes the CoQA dataset.
    Returns:
        Dataset: The processed CoQA dataset.
    """
    import urllib.request
    save_path = './coqa_dataset'
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(f"{save_path}/coqa-dev-v1.0.json"):
        # Download the CoQA dataset if not already present
        url = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"
        try:
            urllib.request.urlretrieve(url, f"{save_path}/coqa-dev-v1.0.json")
        except Exception as e:
            print(f"Failed to download coqa dataset file: {e}")
    
    # Load and process the dataset
    with open('./coqa_dataset/coqa-dev-v1.0.json', 'r') as infile:
        data = json.load(infile)['data']
        dataset = {
            'story': [],
            'question': [],
            'answer': [],
            'additional_answers': [],
            'id': []
        }
        for sample in data:
            story = sample['story']
            questions = sample['questions']
            answers = sample['answers']
            additional_answers = sample['additional_answers']
            for question_index, question in enumerate(questions):
                dataset['story'].append(story)
                dataset['question'].append(question['input_text'])
                dataset['answer'].append({
                    'text': answers[question_index]['input_text'],
                    'answer_start': answers[question_index]['span_start']
                })
                dataset['id'].append(sample['id'] + '_' + str(question_index))
                additional_answers_list = [
                    additional_answers[str(i)][question_index]['input_text'] for i in range(3)
                ]
                dataset['additional_answers'].append(additional_answers_list)
                story += f' Q: {question["input_text"]} A: {answers[question_index]["input_text"]}'
                if story[-1] != '.':
                    story += '.'
        return Dataset.from_dict(dataset)
    
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

def formatter(model_name, dataset_name, dataset, tokenizer): 
    
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
        if dataset_name in ['truthfulqa', 'triviaqa']:
            prompt = tokenizer(format_prompt(base_prompt.format(question=row['question'])), return_tensors='pt').input_ids
        elif dataset_name == 'tydiqa':
            prompt = tokenizer(format_prompt(base_prompt.format(context=row['context'], question=row['question'])), return_tensors='pt').input_ids
        elif dataset_name == 'coqa':
            prompt = tokenizer(format_prompt(base_prompt.format(story=row['story'], question=row['question'])), return_tensors='pt').input_ids
        elif dataset_name == 'haluevaldia':
            prompt = tokenizer(format_prompt(base_prompt.format(knowledge=row['knowledge'], dialogue_history=row['dialogue_history'])), return_tensors='pt').input_ids
        elif dataset_name == 'haluevalqa':
            prompt = tokenizer(format_prompt(base_prompt.format(context=row['knowledge'], question=row['question'])), return_tensors='pt').input_ids
        elif dataset_name == 'haluevalsum':
            prompt = tokenizer(format_prompt(base_prompt.format(document=row['document'])), return_tensors='pt').input_ids
        ##########################################################################################################################################################
        elif dataset_name == 'sorry-Bench' :
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['turns'][0])), return_tensors='pt').input_ids
        elif dataset_name in ['over-refusal','malicious-instruct','advbench','trustllm']:
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['prompt'])), return_tensors='pt').input_ids
        elif dataset_name == 'jailbreak-bench':
            prompt = tokenizer(format_prompt(base_prompt.format(prompt=row['Goal'])),return_tensors='pt').input_ids
            
        all_prompts.append(prompt)
    
    if model_name in  ['llama2_7B_chat']:
        all_labels = torch.ones(len(all_prompts))
    else :
        all_labels = torch.zeros(len(all_prompts))
        
    return all_prompts, all_labels

def main(): 


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
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

    
    # Load the TruthfulQA dataset's validation split
    if args.dataset_name == "truthfulqa":
        dataset = load_dataset("truthful_qa", 'generation')['validation']
    elif args.dataset_name == 'triviaqa':
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch
        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == 'tydiqa':
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        dataset = dataset.filter(lambda row: "english" in row["id"])
    elif args.dataset_name == 'coqa':
        dataset =  load_coqa_dataset()
    elif args.dataset_name == 'haluevaldia':
        dataset = load_dataset("pminervini/HaluEval", "dialogue")['data']
    elif args.dataset_name == 'haluevalqa':
        dataset = load_dataset("pminervini/HaluEval", "qa")['data']
    elif args.dataset_name == 'haluevalsum':
        dataset = load_dataset("pminervini/HaluEval", "summarization")['data']
    ############################################################################
    elif args.dataset_name == 'sorry-Bench' :
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

    prompts, labels = formatter(args.model_name, args.dataset_name, dataset, tokenizer)


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
        # if i == len(prompts) // 2 :
        #     print("Saving layer wise activations part 1")
        #     np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}_{args.dataset_name}_layer_wise_1.npy', all_layer_wise_activations)
        #     print("Saving head wise activations part 1")
        #     np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}_{args.dataset_name}_head_wise_1.npy', all_head_wise_activations)
        #     all_layer_wise_activations.clear()
        #     all_head_wise_activations.clear()

            
    os.makedirs(f'/home/iplab/LLM/mitigation_results/{args.model_name}/', exist_ok=True)
    # print("Saving labels")
    # np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    # print("Saving layer wise activations")
    # np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'/home/iplab/LLM/mitigation_results/{args.model_name}/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()